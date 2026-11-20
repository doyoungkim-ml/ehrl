"""Real local-trainer checkpoint hooks with a CPU-only worker adapter.

Only the distributed parent is replaced. The hook bodies and ACE serializer are
production code; no GPU training or remote filesystem support is claimed.
"""

import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_local_ace_checkpoint_resume import local_class, trainer


def config(root, mode="auto", resume_from=None):
    return OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(root),
                "default_hdfs_dir": None,
                "resume_mode": mode,
                "resume_from_path": str(resume_from) if resume_from else None,
            }
        }
    )


def instance(parent, root, **kwargs):
    result = local_class(parent)()
    result.__dict__.update(trainer().__dict__)
    result.config = config(root, **kwargs)
    return result


def test_save_writes_ace_sidecar_before_parent_publishes_latest(tmp_path):
    class Parent:
        def _save_checkpoint(self):
            folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            assert (folder / "ace_state.json").is_file(), "parent published a checkpoint without ACE state"
            (folder / "actor").mkdir()
            (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(self.global_steps))
            return "parent-result"

    original = instance(Parent, tmp_path)
    assert original._save_checkpoint() == "parent-result"
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "5"
    document = json.loads((tmp_path / "global_step_5/ace_state.json").read_text())
    assert document["local_train_task_seen_counts"] == original.local_train_task_seen_counts


def test_load_restores_from_explicit_source_not_new_output_directory(tmp_path):
    from rllm.algorithms.ace.checkpoint import save_ace_state

    source = tmp_path / "previous" / "global_step_5"
    source.mkdir(parents=True)
    original = trainer()
    save_ace_state(original, source)

    class Parent:
        def _load_checkpoint(self):
            self.global_steps = 5
            self.model_loaded = True
            self._dataloader_resume_iterator_finished = False

    resumed = instance(Parent, tmp_path / "new", mode="resume_path", resume_from=source)
    resumed.local_train_playbooks = {}
    resumed.local_train_task_seen_counts = {}
    resumed._load_checkpoint()
    assert resumed.model_loaded
    assert resumed.local_train_task_seen_counts == original.local_train_task_seen_counts
    assert resumed.local_train_playbooks["task/a"].to_string() == original.local_train_playbooks["task/a"].to_string()


def test_auto_uses_loaded_step_not_a_reread_of_latest_marker(tmp_path):
    from rllm.algorithms.ace.checkpoint import save_ace_state

    original = trainer()
    (tmp_path / "global_step_5").mkdir()
    save_ace_state(original, tmp_path / "global_step_5")

    class Parent:
        def _load_checkpoint(self):
            self.global_steps = 5
            self._dataloader_resume_iterator_finished = False
            # Another writer moving latest must not select a different ACE state.
            (tmp_path / "latest_checkpointed_iteration.txt").write_text("8")

    resumed = instance(Parent, tmp_path)
    resumed.local_train_task_seen_counts = {}
    resumed._load_checkpoint()
    assert resumed.local_train_task_seen_counts == original.local_train_task_seen_counts


@pytest.mark.parametrize("mode", ["disable", "auto"])
def test_fresh_training_does_not_load_stale_sidecar(tmp_path, mode):
    class Parent:
        def _load_checkpoint(self):
            return 0

    resumed = instance(Parent, tmp_path, mode=mode)
    resumed.global_steps = 0
    (tmp_path / "global_step_0").mkdir()
    (tmp_path / "global_step_0/ace_state.json").write_text("not a checkpoint")
    assert resumed._load_checkpoint() == 0


def test_sidecar_write_failure_does_not_advance_latest(tmp_path, monkeypatch):
    from rllm.algorithms.ace import checkpoint as checkpoint_state

    parent_called = []

    class Parent:
        def _save_checkpoint(self):
            parent_called.append(True)

    def fail(*args):
        raise OSError("simulated full disk")

    monkeypatch.setattr(checkpoint_state, "save_ace_state", fail)
    with pytest.raises(OSError, match="full disk"):
        instance(Parent, tmp_path)._save_checkpoint()
    assert not parent_called


def test_remote_sidecars_are_explicitly_unsupported(tmp_path):
    class Parent:
        def _save_checkpoint(self):
            raise AssertionError("must reject before writing an incomplete remote checkpoint")

    original = instance(Parent, tmp_path)
    original.config.trainer.default_hdfs_dir = "hdfs://test/checkpoints"
    with pytest.raises(NotImplementedError, match="local"):
        original._save_checkpoint()


def test_unpatched_verl_cannot_silently_drop_dataloader_state(tmp_path):
    class Parent:
        def _load_checkpoint(self):
            self.global_steps = 5

    resumed = instance(Parent, tmp_path)
    with pytest.raises(RuntimeError, match="VeRL.*dataloader"):
        resumed._load_checkpoint()


@pytest.mark.parametrize("fail", [False, True])
def test_validation_does_not_change_next_training_rng_or_playbook(fail):
    from test_local_ace_checkpoint_resume import STANDARD, Playbook, isolated_class

    cls = isolated_class(
        STANDARD, "LocalACERLAgentPPOTrainer", {"_validate_agent", "_validate_agent_impl", "_capture_component_schedule_state", "_restore_component_schedule_state"}, {"ACERLAgentPPOTrainer": object}
    )
    subject = cls()
    subject.__dict__.update(trainer().__dict__)
    subject.local_eval_epochs = 1
    subject._clear_validation_policy_trajectory_cache = lambda: None
    subject._phase_scope = lambda *args, **kwargs: nullcontext()
    subject._log_validation_turn_table = lambda **kwargs: None
    book_before = subject.playbook.to_string()
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()

    def validate_epoch():
        random.random()
        np.random.random()
        torch.rand(1)
        subject.playbook = Playbook("evaluation-only learning")
        if fail:
            raise RuntimeError("validation failure")
        return {}, {}, [], None

    subject._run_local_validation_epoch = validate_epoch
    try:
        if fail:
            with pytest.raises(RuntimeError, match="validation failure"):
                subject._validate_agent()
        else:
            subject._validate_agent()
        assert subject.playbook.to_string() == book_before
        assert random.getstate() == python_state
        assert np.array_equal(np.random.get_state()[1], numpy_state[1])
        assert np.random.get_state()[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
