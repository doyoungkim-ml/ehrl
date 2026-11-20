"""Training reward evidence, persistence and true data-epoch boundaries on CPU."""

import ast
import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_local_ace_checkpoint_resume import isolated_class
from test_workflow_checkpoint_resume import fresh, real_parent

from rllm.utils.training_reward import (
    REWARD_KEYS,
    TrainingRewardTracker,
    initialize_training_reward,
    raw_reward_metrics,
    read_checkpoint,
    validate_state,
    workflow_reward_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "rllm/trainer/verl/agent_ppo_trainer.py"


def observe(tracker, rewards, invalid=0):
    tracker.observe(dict(zip(REWARD_KEYS, (sum(rewards), len(rewards), invalid), strict=True)))


def finish(tracker, epoch, rewards, invalid=0):
    tracker.begin_epoch(epoch)
    observe(tracker, rewards, invalid)
    return tracker.complete_epoch()


def test_weighted_native_rewards_and_five_completed_below_best_epochs():
    t = TrainingRewardTracker()
    observe(t, [1])
    observe(t, [0, 0, 0])
    assert t.complete_epoch()["mean"] == 0.25
    for epoch in range(1, 6):
        finish(t, epoch, [0])
        assert t.state["below_best_streak"] == epoch
    assert finish(t, 6, [0.25])["below_best_streak"] == 0
    assert finish(t, 7, [0.5])["best"] == 0.5
    assert finish(t, 8, [0])["below_best_streak"] == 1
    assert finish(t, 9, [0.5 - 1e-14])["below_best_streak"] == 2
    assert t.state["best"] == 0.5


def test_invalid_or_incomplete_epochs_break_patience_without_changing_best():
    t = TrainingRewardTracker(coverage_complete=False)
    assert not finish(t, 0, [0.9])["eligible"]
    assert t.state["best"] is None
    finish(t, 1, [0.7])
    finish(t, 2, [0.1])
    entry = finish(t, 3, [0.2], invalid=1)
    assert entry["mean"] is None
    assert entry["below_best_streak"] == 0
    assert entry["best"] == 0.7
    t.begin_epoch(4)
    t.observe({})
    assert not t.complete_epoch()["eligible"]


def test_raw_zero_is_valid_but_missing_nonfinite_and_infra_are_not():
    rows = [
        {"trajectory_reward": 0}, {"trajectory_reward": np.float32(0.5)},
        {"trajectory_reward": torch.tensor(0.5)}, {"trajectory_reward": 1, "termination_reason": "ENV_ERROR"},
        {"trajectory_reward": 0, "termination_reason": "ENV_EXECUTION_UNCERTAIN"},
        {"trajectory_reward": float("nan")}, {"trajectory_reward": float("inf")}, {},
    ]
    assert raw_reward_metrics(rows, expected_count=10) == dict(zip(REWARD_KEYS, (1.0, 3, 7), strict=True))


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(version=True), lambda s: s.update(metric_id="other"),
    lambda s: s.update(best=99), lambda s: s.update(below_best_streak=5),
    lambda s: s["completed_epochs"][0].update(mean=0.5),
    lambda s: s["completed_epochs"][0].update(eligible=False),
    lambda s: s["completed_epochs"][1].update(epoch=5),
    lambda s: s["completed_epochs"][0].update(reward_sum=float("nan")),
    lambda s: s["completed_epochs"][0].update(coverage_complete=False),
    lambda s: s["completed_epochs"][0].update(best=True),
    lambda s: s.update(epoch=7),
    lambda s: s.pop("active"),
    lambda s: s["completed_epochs"][0].pop("mean"),
])
def test_corrupt_present_state_fails(mutation):
    t = TrainingRewardTracker()
    finish(t, 0, [1])
    finish(t, 1, [0])
    state = copy.deepcopy(t.state)
    mutation(state)
    with pytest.raises(ValueError):
        validate_state(state)


def test_workflow_counts_once_per_episode_and_refuses_ambiguous_actor_population():
    batch = SimpleNamespace(
        batch={"traj_rewards": torch.tensor([[0.5], [0.5], [0.0], [0.4], [0.8]])},
        non_tensor_batch={"episode_ids": ["a", "a", "b", "c", "c"], "trajectory_ids": ["a1", "a1", "b1", "c1", "c2"], "termination_reasons": ["env_done"] * 5},
    )
    assert workflow_reward_metrics(batch, 4) == dict(zip(REWARD_KEYS, (0.5, 2, 2), strict=True))


@pytest.mark.parametrize("stepwise", [False, True])
def test_base_generation_measures_native_reward_before_transform(stepwise):
    method = "generate_agent_steps" if stepwise else "generate_agent_trajectory"
    cls = isolated_class(SOURCE, "AgentPPOTrainer", {method}, {"RayPPOTrainer": object, "marked_timer": lambda *a: nullcontext()})
    subject = cls()
    subject.async_rollout_mode = True
    rows = [{"idx": 1, "trajectory_reward": 0.0}, {"idx": 0, "trajectory_reward": 1.0}, {"idx": 2, "trajectory_reward": 0.0, "termination_reason": "ENV_ERROR"}]
    subject.generate_agent_trajectories_async = lambda **kwargs: iter(rows)

    def transform(trajectories, **kwargs):
        for row in trajectories:
            row["trajectory_reward"] = 99
        output = SimpleNamespace(meta_info={})
        return output if stepwise else (output, {})

    subject._transform_agent_steps = subject._transform_agent_trajectories = transform
    if stepwise:
        metrics = subject.generate_agent_steps(uids=["a", "b", "c"]).meta_info["training_reward_metrics"]
    else:
        _, metrics = subject.generate_agent_trajectory()
    assert {key: metrics[key] for key in REWARD_KEYS} == dict(zip(REWARD_KEYS, (1.0, 2, 1), strict=True))


def actual_agent_class(parent, *, loop=False, workflow=False):
    source = ROOT / "rllm/trainer/verl/agent_workflow_trainer.py" if workflow else SOURCE
    name = "AgentWorkflowPPOTrainer" if workflow else "AgentPPOTrainer"
    methods = {"_load_checkpoint", "_save_checkpoint", "_get_training_start_epoch", "_iter_training_epoch"}
    if not loop:
        return isolated_class(source, name, methods, {"RayPPOTrainer": parent, "Path": Path, "json": json})
    cls = copy.deepcopy(next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name))
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit_agent")
    epoch = next(n for n in fit.body if isinstance(n, ast.For))
    batch = next(n for n in epoch.body if isinstance(n, ast.For))
    # Retain the actual observation call, epoch controls, save/load hooks and
    # StopIteration finalizer. Only distributed PPO work is replaced.
    observation = next(n for n in ast.walk(batch) if isinstance(n, ast.Expr) and ast.unparse(n).startswith("self._training_reward_tracker.observe("))
    reward_setup = """
metrics = raw_reward_metrics([{'trajectory_reward': 1 / (epoch + 1)} for item in batch_dict])
rollout_uids = list(batch_dict)
reward_metrics = metrics
expected_reward_count = len(batch_dict)
"""
    batch.body = ast.parse(reward_setup).body + [observation] + ast.parse("""
self.records.append((epoch, batch_dict.tolist()))
if self.interrupt_at is not None and len(self.records) == self.interrupt_at:
    self._save_checkpoint()
    return
if not self.reject_all:
    self.global_steps += 1
if self.global_steps >= self.total_training_steps:
    return
""").body
    fit.body = [n for n in fit.body if not (isinstance(n, ast.ImportFrom) and n.module in {"rllm.utils.run_state", "rllm.utils.tracking"})]
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods | {"fit_agent"}]
    from collections import Counter, defaultdict
    ns = {
        "RayPPOTrainer": parent, "Counter": Counter, "defaultdict": defaultdict, "Path": Path, "json": json, "OmegaConf": OmegaConf,
        "pprint": lambda *a: None, "prepare_training_run": lambda c: None, "Tracking": lambda **kw: SimpleNamespace(), "raw_reward_metrics": raw_reward_metrics,
        "finish_data_exhausted": lambda trainer: None,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), ns)
    return ns[name]


def agent(cls, folder):
    t = fresh(cls, folder)
    t.config.trainer.total_epochs = 6
    t.interrupt_at = None
    t.reject_all = False
    return t


@pytest.mark.parametrize("boundary", ["mid", "last_before_stop", "finished"])
@pytest.mark.parametrize("workflow", [False, True])
def test_actual_agent_resume_preserves_sums_best_patience_and_batch_order(tmp_path, monkeypatch, boundary, workflow):
    cls = actual_agent_class(real_parent(monkeypatch), loop=True, workflow=workflow)
    uninterrupted = agent(cls, tmp_path / "whole" / "checkpoints")
    uninterrupted.fit_agent()
    original = agent(cls, tmp_path / "split" / "checkpoints")
    if boundary == "finished":
        original.config.trainer.total_epochs = 1
        original.fit_agent()
        original._save_checkpoint()
    else:
        original.interrupt_at = 2 if boundary == "mid" else 4
        original.fit_agent()
    before_resume = copy.deepcopy(original.records)
    resumed = agent(cls, tmp_path / "split" / "checkpoints")
    resumed.fit_agent()
    assert before_resume + resumed.records == uninterrupted.records
    assert resumed._training_reward_tracker.state == uninterrupted._training_reward_tracker.state
    assert resumed._training_reward_tracker.state["below_best_streak"] == 5


def test_all_rejected_epochs_still_publish_live_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    cls = actual_agent_class(real_parent(monkeypatch), loop=True)
    t = agent(cls, tmp_path / "run" / "checkpoints")
    t.reject_all = True
    t.fit_agent()
    receipt = json.loads((tmp_path / "run/training_reward_state.json").read_text())
    assert receipt["slurm_job_id"] == "123"
    assert isinstance(receipt["written_at"], float) and receipt["written_at"] > 0
    assert receipt["state"]["below_best_streak"] == 5
    assert len(receipt["state"]["completed_epochs"]) == 6
    assert not (tmp_path / "run/checkpoints/latest_checkpointed_iteration.txt").exists()


def test_max_step_return_does_not_complete_partial_epoch(tmp_path, monkeypatch):
    cls = actual_agent_class(real_parent(monkeypatch), loop=True)
    t = agent(cls, tmp_path / "checkpoints")
    t.total_training_steps = 3
    t.fit_agent()
    assert t._training_reward_tracker.state["completed_epochs"] == []
    assert t._training_reward_tracker.state["active"]["reward_count"] == 8


def test_checkpoint_step_validation_and_legacy_coverage(tmp_path, monkeypatch):
    cls = actual_agent_class(real_parent(monkeypatch), loop=True)
    original = agent(cls, tmp_path / "checkpoints")
    original.interrupt_at = 2
    original.fit_agent()
    checkpoint = tmp_path / "checkpoints" / f"global_step_{original.global_steps}"
    with pytest.raises(ValueError, match="checkpoint"):
        read_checkpoint(checkpoint / "reward_epoch_state.json", global_step=99)
    (checkpoint / "reward_epoch_state.json").unlink()
    resumed = agent(cls, tmp_path / "checkpoints")
    with pytest.warns(RuntimeWarning, match="historical epoch rewards"):
        resumed.fit_agent()
    history = resumed._training_reward_tracker.state["completed_epochs"]
    assert history[0]["eligible"] is False
    assert history[0]["mean"] is None
    assert history[1]["eligible"] is True


def test_save_sidecar_precedes_parent_marker_and_failure_cannot_publish(tmp_path, monkeypatch):
    cls = actual_agent_class(real_parent(monkeypatch))
    t = agent(cls, tmp_path / "checkpoints")
    t._training_epoch = 7
    t.global_steps = 12
    initialize_training_reward(t, 7)
    calls = []
    old = t.actor_rollout_wg.save_checkpoint
    def check_then_save(*args, **kwargs):
        calls.append(read_checkpoint(tmp_path / "checkpoints/global_step_12/reward_epoch_state.json", 12))
        return old(*args, **kwargs)
    t.actor_rollout_wg.save_checkpoint = check_then_save
    t._save_checkpoint()
    assert calls[0]["epoch"] == 7
    import rllm.utils.run_state as run_state
    monkeypatch.setattr(run_state, "atomic_json", lambda *a: (_ for _ in ()).throw(OSError("sidecar failed")))
    t.global_steps = 13
    with pytest.raises(OSError, match="sidecar failed"):
        t._save_checkpoint()
    assert (tmp_path / "checkpoints/latest_checkpointed_iteration.txt").read_text() == "12"


def test_startup_receipt_replaces_stale_unsaved_history_without_controller_rng_draws(tmp_path, monkeypatch):
    import random

    from rllm.utils.training_reward import publish_training_reward, restore_training_reward, save_training_reward

    monkeypatch.setenv("SLURM_JOB_ID", "old")
    t = SimpleNamespace(config=OmegaConf.create({"trainer": {"default_local_dir": str(tmp_path / "checkpoints"), "resume_mode": "auto"}}), global_steps=3, _training_epoch=0)
    initialize_training_reward(t, 0)
    observe(t._training_reward_tracker, [1])
    t._training_reward_tracker.complete_epoch()
    save_training_reward(t)
    for epoch in range(1, 6):
        finish(t._training_reward_tracker, epoch, [0])
    publish_training_reward(t)
    assert json.loads((tmp_path / "training_reward_state.json").read_text())["state"]["below_best_streak"] == 5
    t._dataloader_resume_iterator_finished = True
    monkeypatch.setenv("SLURM_JOB_ID", "new")
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    restore_training_reward(t, True)
    initialize_training_reward(t, 1)
    assert random.getstate() == python_state
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_state, strict=True))
    assert torch.equal(torch.get_rng_state(), torch_state)
    receipt = json.loads((tmp_path / "training_reward_state.json").read_text())
    assert receipt["slurm_job_id"] == "new"
    assert receipt["state"]["below_best_streak"] == 0
    assert len(receipt["state"]["completed_epochs"]) == 1
