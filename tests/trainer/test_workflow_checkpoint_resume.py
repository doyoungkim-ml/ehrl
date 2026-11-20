"""Actual workflow hooks/loop with pinned VeRL save/load and CPU torchdata.

The CPU worker exercises model/optimizer persistence; GPU/Ray work is excluded.
Historical workflow checkpoints do not save driver RNG, so these tests establish
step/epoch/loader continuation, not exact stochastic training continuation.
"""

import ast
import copy
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_local_ace_checkpoint_resume import isolated_class
from test_local_ace_resume_integration import CPUWorker
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "rllm/trainer/verl/agent_workflow_trainer.py"
METHODS = {"_load_checkpoint", "_save_checkpoint", "_get_training_start_epoch", "_iter_training_epoch"}


def real_parent(monkeypatch):
    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("set VERL_RESUME_SOURCE to the prepared pinned VeRL checkout")
    source = Path(source)
    ns = {"os": os, "torch": torch, "Role": SimpleNamespace(Critic="critic"), "deprecated": lambda message: lambda cls: cls}
    tree = ast.parse((source / "verl/utils/checkpoint/checkpoint_manager.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"find_latest_ckpt_path", "get_checkpoint_tracker_filename"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "checkpoint_manager.py", "exec"), ns)
    parent = isolated_class(source / "verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", {"_save_checkpoint", "_load_checkpoint"}, ns, bases=[])
    fs = ModuleType("verl.utils.fs")
    fs.local_mkdir_safe = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(sys.modules, "verl.utils.fs", fs)
    return parent


def workflow_class(parent, *, loop=False):
    cls = copy.deepcopy(next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "AgentWorkflowPPOTrainer"))
    methods = set(METHODS)
    if loop:
        methods.add("fit_agent")
        fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit_agent")
        epoch = next(n for n in fit.body if isinstance(n, ast.For))
        batch = next(n for n in epoch.body if isinstance(n, ast.For))
        increment = next(n for n in batch.body if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "self.global_steps")
        terminal = next(n for n in batch.body if isinstance(n, ast.If) and ast.unparse(n.test).startswith("self.global_steps >= self.total_training_steps"))
        batch.body = ast.parse("self.records.append((epoch, self.global_steps, batch_dict.tolist()))").body + [increment, terminal]
        # Keep actual initialization, load, completion checks and epoch control.
        # Only tracing/validation dependencies and PPO batch work are replaced.
        fit.body = [n for n in fit.body if not (isinstance(n, ast.ImportFrom) and n.module in {"rllm.utils.run_state", "rllm.utils.tracking"})]
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    ns = {
        "RayPPOTrainer": parent,
        "json": json,
        "Path": Path,
        "OmegaConf": OmegaConf,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "pprint": lambda *a: None,
        "prepare_training_run": lambda config: None,
        "finish_data_exhausted": lambda trainer: None,
        "Tracking": lambda **kw: SimpleNamespace(finish=lambda: None),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), ns)
    return ns["AgentWorkflowPPOTrainer"]


def fresh(cls, root, *, workers=0):
    t = cls()
    t.global_steps = 0
    t.actor_rollout_wg = CPUWorker()
    t.use_critic = False
    t.val_reward_fn = None
    t.records = []
    t.synced_versions = []
    t.checkpoint_manager = SimpleNamespace(update_weights=lambda version: t.synced_versions.append(version))
    t.total_training_steps = 100
    dataset = list(range(16))
    t.train_dataloader = StatefulDataLoader(dataset, batch_size=4, num_workers=workers, sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(42)))
    t.config = OmegaConf.create({
        "trainer": {
            "default_local_dir": str(root), "default_hdfs_dir": None,
            "resume_mode": "auto", "resume_from_path": None,
            "del_local_ckpt_after_load": False, "total_epochs": 12,
            "val_before_train": False, "test_freq": -1,
            "project_name": "test", "experiment_name": "workflow", "logger": [],
        },
        "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
    })
    return t


def save(t, *, step=55, epoch=9):
    t.global_steps = step
    t._training_epoch = epoch
    t.actor_rollout_wg.update(torch.arange(4))
    t._save_checkpoint()
    return Path(t.config.trainer.default_local_dir) / f"global_step_{step}"


def resume(t, checkpoint):
    t.config.trainer.resume_mode = "resume_path"
    t.config.trainer.resume_from_path = str(checkpoint)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("consumed,finished", [(0, False), (2, False), (4, False), (4, True)])
def test_resume_55_preserves_real_loader_tail_and_actual_epoch(tmp_path, monkeypatch, workers, consumed, finished):
    cls = workflow_class(real_parent(monkeypatch), loop=True)
    original = fresh(cls, tmp_path / "original", workers=workers)
    for _ in range(2):
        list(original.train_dataloader)
    iterator = iter(original.train_dataloader)
    for _ in range(consumed):
        next(iterator)
    if finished:
        assert next(iterator, None) is None
    checkpoint = save(original)
    expected_batches = [(9, batch.tolist()) for batch in iterator]
    for epoch in range(10, 12):
        expected_batches.extend((epoch, batch.tolist()) for batch in original.train_dataloader)
    resumed = fresh(cls, tmp_path / "resumed", workers=workers)
    resume(resumed, checkpoint)
    resumed.fit_agent()
    assert resumed.records == [(epoch, step, values) for step, (epoch, values) in enumerate(expected_batches, 56)]
    assert resumed.synced_versions == [55]
    assert resumed.global_steps == 56 + len(expected_batches)
    assert not (tmp_path / "resumed/global_step_55").exists()


@pytest.mark.parametrize("checkpoint_step,first_step", [(None, 0), (0, 1), (55, 56)])
def test_actual_loop_preserves_fresh_zero_and_resumes_next_update(tmp_path, monkeypatch, checkpoint_step, first_step):
    cls = workflow_class(real_parent(monkeypatch), loop=True)
    t = fresh(cls, tmp_path / "output")
    if checkpoint_step is not None:
        next(iter(t.train_dataloader))
        save(t, step=checkpoint_step, epoch=0)
        t = fresh(cls, tmp_path / "output")
    t.total_training_steps = first_step + 2
    t.fit_agent()
    assert [step for _, step, _ in t.records] == [first_step, first_step + 1]
    assert t.global_steps == t.total_training_steps
    assert t.synced_versions == [checkpoint_step or 0]


def test_terminal_checkpoint_does_not_repeat_last_update(tmp_path, monkeypatch):
    cls = workflow_class(real_parent(monkeypatch), loop=True)
    original = fresh(cls, tmp_path / "output")
    next(iter(original.train_dataloader))
    save(original, step=1, epoch=0)
    t = fresh(cls, tmp_path / "output")
    t.total_training_steps = 2
    t.fit_agent()
    assert t.records == []
    assert t.synced_versions == [1]


def test_explicit_source_sidecar_and_cpu_optimizer_are_restored(tmp_path, monkeypatch):
    cls = workflow_class(real_parent(monkeypatch))
    original = fresh(cls, tmp_path / "source")
    next(iter(original.train_dataloader))
    checkpoint = save(original)
    t = fresh(cls, tmp_path / "destination")
    wrong = tmp_path / "destination/global_step_55"
    wrong.mkdir(parents=True)
    (wrong / "epoch_state.json").write_text('{"epoch": 2, "global_step": 55}')
    resume(t, checkpoint)
    t._load_checkpoint()
    assert t._training_epoch == 9
    assert t._workflow_checkpoint_loaded
    for key, value in original.actor_rollout_wg.model.state_dict().items():
        torch.testing.assert_close(t.actor_rollout_wg.model.state_dict()[key], value, rtol=0, atol=0)
    expected = original.actor_rollout_wg.optimizer.state_dict()
    actual = t.actor_rollout_wg.optimizer.state_dict()
    assert actual["param_groups"] == expected["param_groups"]
    for key, state in expected["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(actual["state"][key][field], value, rtol=0, atol=0)


def test_auto_uses_actually_loaded_step_when_marker_moves(tmp_path, monkeypatch):
    cls = workflow_class(real_parent(monkeypatch))
    original = fresh(cls, tmp_path / "source")
    next(iter(original.train_dataloader))
    save(original)
    t = fresh(cls, tmp_path / "source")
    load = t.actor_rollout_wg.load_checkpoint

    def load_then_move(path, del_local_after_load):
        load(path, del_local_after_load)
        (tmp_path / "source/latest_checkpointed_iteration.txt").write_text("999")

    t.actor_rollout_wg.load_checkpoint = load_then_move
    t._load_checkpoint()
    assert (t.global_steps, t._training_epoch) == (55, 9)


@pytest.mark.parametrize("document", [
    None, "{", "[]",
    '{"epoch": true, "global_step": 55}',
    '{"epoch": -1, "global_step": 55}',
    '{"epoch": "9", "global_step": 55}',
    '{"epoch": 9, "global_step": 54}',
    '{"epoch": 9, "global_step": 55.0}',
])
def test_missing_corrupt_or_mismatched_epoch_fails_before_epoch_mutation(tmp_path, monkeypatch, document):
    cls = workflow_class(real_parent(monkeypatch))
    original = fresh(cls, tmp_path / "source")
    next(iter(original.train_dataloader))
    checkpoint = save(original)
    path = checkpoint / "epoch_state.json"
    if document is None:
        path.unlink()
    else:
        path.write_text(document)
    t = fresh(cls, tmp_path / "output")
    t._training_epoch = 77
    resume(t, checkpoint)
    with pytest.raises(ValueError, match="Invalid workflow epoch checkpoint"):
        t._load_checkpoint()
    assert t._training_epoch == 77
    assert not t._workflow_checkpoint_loaded


@pytest.mark.parametrize("metadata", [None, 0, "false"])
def test_resume_requires_pinned_loader_metadata(metadata, tmp_path):
    class Parent:
        def _load_checkpoint(self):
            self.global_steps = 55
            self._dataloader_resume_iterator_finished = metadata

    t = fresh(workflow_class(Parent), tmp_path)
    with pytest.raises(RuntimeError, match="pinned VeRL"):
        t._load_checkpoint()


def test_invalid_raw_loader_checkpoint_is_rejected(tmp_path, monkeypatch):
    cls = workflow_class(real_parent(monkeypatch))
    original = fresh(cls, tmp_path / "source")
    checkpoint = save(original)
    torch.save({}, checkpoint / "data.pt")
    t = fresh(cls, tmp_path / "output")
    resume(t, checkpoint)
    with pytest.raises(ValueError, match="Invalid dataloader checkpoint"):
        t._load_checkpoint()
    assert not t._workflow_checkpoint_loaded


def test_sidecar_precedes_parent_and_latest_marker(tmp_path, monkeypatch):
    cls = workflow_class(real_parent(monkeypatch))
    t = fresh(cls, tmp_path)
    t.global_steps = 55
    t._training_epoch = 9
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("54")
    parent_save = t.actor_rollout_wg.save_checkpoint

    def inspect_then_save(*args, **kwargs):
        assert json.loads((tmp_path / "global_step_55/epoch_state.json").read_text()) == {"epoch": 9, "global_step": 55}
        assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "54"
        return parent_save(*args, **kwargs)

    t.actor_rollout_wg.save_checkpoint = inspect_then_save
    t._save_checkpoint()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "55"


def test_sidecar_failure_leaves_parent_and_marker_untouched(tmp_path, monkeypatch):
    import rllm.utils.run_state as run_state

    cls = workflow_class(real_parent(monkeypatch))
    t = fresh(cls, tmp_path)
    t.global_steps = 55
    t._training_epoch = 9
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("54")
    calls = []
    t.actor_rollout_wg.save_checkpoint = lambda *args, **kwargs: calls.append(args)

    def fail(*args):
        raise OSError("sidecar failure")

    monkeypatch.setattr(run_state, "atomic_json", fail)
    with pytest.raises(OSError, match="sidecar failure"):
        t._save_checkpoint()
    assert calls == []
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "54"
