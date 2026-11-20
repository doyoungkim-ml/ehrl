"""Exercise the maintained workflow rejection loop with pinned DataProto on CPU.

Only GPU compute and distributed replicas are fixtures. Generation, real group
selection, accumulation, early continue, sync and counter reset execute original
trainer statements. The replica fixture models HYBRID level-2 sleep.
"""

import ast
import copy
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_workflow_actor_batching import runtime as _pinned_runtime_fixture

ROOT = Path(__file__).resolve().parents[2]
# Register the existing fixture without shadowing an imported name in tests.
runtime = _pinned_runtime_fixture


@contextmanager
def timer(name, values, **kwargs):
    yield
    values[name] = values.get(name, 0) + 1


def controller_class(protocol):
    source = ROOT / "rllm/trainer/verl/agent_workflow_trainer.py"
    name = "AgentWorkflowPPOTrainer"
    cls = copy.deepcopy(next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name))
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit_agent")
    epoch = next(n for n in fit.body if isinstance(n, ast.For))
    loop = next(n for n in epoch.body if isinstance(n, ast.For))
    step = next(n for n in loop.body if isinstance(n, ast.With) and "'step'" in ast.unparse(n.items[0].context_expr))
    cutoff = next(i for i, n in enumerate(step.body) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("self.config.rllm.stepwise_advantage.enable"))
    sync = next(n for n in ast.walk(step) if isinstance(n, ast.With) and "'update_weights'" in ast.unparse(n.items[0].context_expr))
    count = "kept_groups"
    step.body = step.body[:cutoff] + ast.parse(f"self.record_update(batch, num_tasks, {count}, dict(timing_raw))").body + [sync]
    reset_start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "batch" and isinstance(n.value, ast.Constant) and n.value.value is None)
    increment = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "self.global_steps")
    loop.body = loop.body[: loop.body.index(step) + 1] + loop.body[reset_start : increment + 1]
    init_start = next(i for i, n in enumerate(fit.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "batch")
    fit.body = ast.parse("from rllm.utils.training_reward import initialize_training_reward, complete_training_reward_epoch, workflow_reward_metrics\nstart_epoch = 0\ninitialize_training_reward(self, start_epoch)").body + [n for n in fit.body[init_start : fit.body.index(epoch)] if isinstance(n, ast.Assign)] + [epoch]
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"fit_agent", "_iter_training_epoch", "_maybe_restore_rollout_after_skipped_batch"}]
    gate_ns = {"np": np}
    gate = ROOT / "rllm/trainer/verl/rejection_gate.py"
    exec(compile(gate.read_text(), str(gate), "exec"), gate_ns)
    ns = {
        "np": np, "torch": torch, "uuid": uuid,
        "Counter": Counter, "defaultdict": defaultdict,
        "DataProto": protocol.DataProto, "marked_timer": timer,
        "pprint": lambda *a: None,
        "group_rejection_verdict": gate_ns["group_rejection_verdict"],
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), ns)
    return ns[name]


class Replicas:
    def __init__(self, version, free_cache):
        self.awake = True
        self.version = version
        self.free_cache = free_cache
        self.events = []

    def sleep_replicas(self):
        self.events.append(("sleep", self.version))
        if self.free_cache:
            self.awake = False

    def update_weights(self, global_steps=None):
        assert not self.free_cache or not self.awake
        if global_steps is not None:
            self.version = global_steps
        self.events.append(("restore" if global_steps is None else "sync", self.version))
        self.awake = True


class Epochs:
    def __init__(self, epochs):
        self.epochs = iter(epochs)

    def __iter__(self):
        return iter(next(self.epochs))


def trainer(runtime, waves, *, required=2, next_step=0, version=0, free_cache=True, manager=True):
    subject = controller_class(runtime)()
    subject.global_steps = next_step
    subject.config = OmegaConf.create({
        "trainer": {"total_epochs": len(waves)},
        "data": {"train_batch_size": required},
        "actor_rollout_ref": {"rollout": {"n": 2, "free_cache_engine": free_cache}},
        "rllm": {"rejection_sample": {"enable": True, "mode": "variance"}},
    })
    subject.train_dataloader = Epochs([[{"input_ids": torch.zeros((len(pairs), 1), dtype=torch.long), "reward_pairs": np.array(pairs)} for pairs in epoch] for epoch in waves])
    subject.state = Replicas(version, free_cache and manager)
    if manager:
        subject.checkpoint_manager = subject.state
    subject._start_profiling = lambda enabled: None
    subject.agent_execution_engine = SimpleNamespace(set_training_step=lambda *a, **k: None)
    subject.generated_versions = []
    subject.updates = []

    def generate(batch, timing_raw):
        assert subject.state.awake, "generation attempted with sleeping replicas"
        subject.generated_versions.append(subject.state.version)
        timing_raw["generate_trajectories"] = timing_raw.get("generate_trajectories", 0) + 1
        uids = batch.non_tensor_batch["task_ids"]
        counts = Counter()
        rewards = []
        for uid, pair in zip(uids, batch.non_tensor_batch["reward_pairs"], strict=True):
            rewards.append(pair[counts[uid]])
            counts[uid] += 1
        n = len(rewards)
        return runtime.DataProto.from_dict(
            tensors={"traj_rewards": torch.tensor(rewards, dtype=torch.float32)[:, None]},
            non_tensors={
                "is_correct": np.array(rewards) > 0,
                "episode_ids": np.array([str(uuid.uuid4()) for _ in rewards], dtype=object),
                "trajectory_ids": np.array([str(uuid.uuid4()) for _ in rewards], dtype=object),
                "metrics": np.array([{} for _ in rewards], dtype=object),
                "termination_reasons": np.array(["env_done"] * n, dtype=object),
            },
            meta_info={"repeat_counts": [1] * n, "dropped_episodes": []},
        )

    def record(batch, tasks, kept, times):
        assert not subject.state.free_cache or not subject.state.awake
        subject.updates.append({
            "step": subject.global_steps, "tasks": tasks, "kept": kept,
            "selected": len(np.unique(batch.non_tensor_batch["task_ids"])),
            "timing": times, "epoch": getattr(subject, "_training_epoch", None),
        })

    subject.generate_trajectories = generate
    subject.record_update = record
    return subject


def test_seven_groups_refill_to_eight_without_advancing_weights(runtime):
    t = trainer(runtime, [[[[0, 1]] * 7 + [[0, 0]], [[0, 1]]]], required=8, next_step=56, version=55)
    t.fit_agent()
    assert t.generated_versions == [55, 55]
    assert len(t.updates) == 1
    assert t.updates[0] == {
        "step": 56, "tasks": 9, "kept": 8, "selected": 8,
        "timing": {"generate_trajectories": 2, "restore_rollout_after_skip": 1, "step": 1, "start_profile": 2},
        "epoch": 0,
    }
    assert t.global_steps == 57
    assert t.state.events == [("sleep", 55), ("restore", 55), ("sleep", 55), ("sync", 56)]


def test_repeated_rejections_and_epoch_boundary_preserve_partial_batch(runtime):
    t = trainer(runtime, [[[[0, 1]], [[0, 0]]], [[[0, 1]]]])
    t.fit_agent()
    assert t.generated_versions == [0, 0, 0]
    assert [(r["tasks"], r["kept"], r["step"]) for r in t.updates] == [(3, 2, 0)]
    assert t.global_steps == 1
    history = t._training_reward_tracker.state["completed_epochs"]
    assert [(entry["reward_sum"], entry["reward_count"], entry["mean"], entry["eligible"]) for entry in history] == [(1.0, 4, 0.25, True), (1.0, 2, 0.5, True)]


@pytest.mark.parametrize("free_cache,manager", [(False, True), (True, False), (False, False)])
def test_refill_without_free_cache_or_checkpoint_manager(runtime, free_cache, manager):
    t = trainer(runtime, [[[[0, 0]], [[0, 1], [0, 1]]]], free_cache=free_cache, manager=manager)
    t.fit_agent()
    assert len(t.updates) == 1
    assert not any(event == "restore" for event, _ in t.state.events)


@pytest.mark.parametrize("rejection", [True, False])
def test_accepted_batches_keep_sleep_update_sequence(runtime, rejection):
    t = trainer(runtime, [[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]])
    t.config.rllm.rejection_sample.enable = rejection
    t.fit_agent()
    assert [r["step"] for r in t.updates] == [0, 1]
    assert t.state.events == [("sleep", 0), ("sync", 0), ("sleep", 0), ("sync", 1)]


def test_restore_failure_stops_before_next_generation(runtime):
    t = trainer(runtime, [[[[0, 0]], [[0, 1], [0, 1]]]])

    def fail(global_steps=None):
        raise RuntimeError("transfer failed")

    t.checkpoint_manager.update_weights = fail
    with pytest.raises(RuntimeError, match="transfer failed"):
        t.fit_agent()
    assert t.generated_versions == [0]
    assert t.updates == []


def test_workflow_variance_gate_keeps_mixed_positive_rewards(runtime):
    t = trainer(runtime, [[[[0.2, 0.8]], [[0.4, 0.9]]]])
    t.fit_agent()
    assert [r["kept"] for r in t.updates] == [2]
