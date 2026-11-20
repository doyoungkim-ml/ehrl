"""CPU regressions for the real trainer's rejected-batch control flow.

Execute the original generation, sleep, reward filtering, early continues,
weight-sync call and step increment from fit_agent's AST. GPU model operations
outside this path are replaced by a stateful checkpoint-manager/actor fixture;
reward tensors and group selection use real PyTorch/NumPy. No Ray/GPU imports.
"""

import ast
import copy
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "rllm/trainer/verl/agent_ppo_trainer.py"


def controller_class():
    tree = ast.parse(SOURCE.read_text())
    cls = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AgentPPOTrainer"))
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit_agent")
    epoch_loop = next(n for n in fit.body if isinstance(n, ast.For))
    batch_loop = next(n for n in epoch_loop.body if isinstance(n, ast.For))
    step = next(n for n in batch_loop.body if isinstance(n, ast.With))
    adv = next(n for n in step.body if isinstance(n, ast.With))
    # Keep the complete production rejection block, including all continue paths.
    cut = next(i for i, n in enumerate(adv.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "batch.meta_info['global_token_num']")
    sync = next(n for n in ast.walk(fit) if isinstance(n, ast.With) and any("'update_weights'" in ast.unparse(x.context_expr) for x in n.items))
    adv.body = adv.body[:cut]
    step.body = step.body[: step.body.index(adv) + 1]
    increment = next(n for n in batch_loop.body if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "self.global_steps")
    batch_loop.body = batch_loop.body[: batch_loop.body.index(step) + 1] + ast.parse("self.record_update(batch)").body + [sync, increment]
    fit.body = ast.parse("from rllm.utils.training_reward import initialize_training_reward, complete_training_reward_epoch\nstart_epoch = self._get_training_start_epoch()\ninitialize_training_reward(self, start_epoch)").body + [epoch_loop]
    # The fixture initializes global_steps to the next optimizer step, matching
    # production immediately after checkpoint loading and the initial increment.
    cls.bases = []
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and (n.name in {"fit_agent", "_iter_training_epoch", "_get_training_start_epoch", "_augment_accepted_training_batch"} or "rollout" in n.name and ("sleep" in n.name or "skip" in n.name))
    ]
    ns = {"np": np, "torch": torch, "uuid": uuid, "DataProto": Batch, "logger": None, "marked_timer": lambda *a, **k: nullcontext(), "pprint": lambda *a: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), ns)
    return ns["AgentPPOTrainer"]


class TensorBatch(dict):
    def __len__(self):
        return next(iter(self.values())).shape[0]


class Batch:
    """Only DataProto transport operations; selection itself is production code."""

    def __init__(self, plan):
        self.plan = plan
        n = len(plan["rewards"]) // 2
        self.batch = TensorBatch(input_ids=torch.zeros((n, 1), dtype=torch.long))
        self.non_tensor_batch = {}
        self.meta_info = {}

    @classmethod
    def from_single_dict(cls, data):
        return cls(data)

    def repeat(self, repeat_times, interleave):
        assert repeat_times == 2 and interleave
        self.batch = TensorBatch({k: v.repeat_interleave(repeat_times, dim=0) for k, v in self.batch.items()})
        self.non_tensor_batch = {k: np.repeat(v, repeat_times) for k, v in self.non_tensor_batch.items()}
        return self

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        result = copy.copy(self)
        result.batch = TensorBatch({k: v[indices] for k, v in self.batch.items()})
        result.non_tensor_batch = {k: v[indices] for k, v in self.non_tensor_batch.items()}
        result.meta_info = dict(self.meta_info)
        return result

    def select_idxs(self, indices):
        return self[indices]

    def sample_level_repeat(self, counts):
        assert all(c == 1 for c in counts)
        return self

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        self.meta_info.update(other.meta_info)
        return self

    @classmethod
    def concat(cls, batches):
        result = copy.copy(batches[0])
        result.batch = TensorBatch({k: torch.cat([b.batch[k] for b in batches]) for k in result.batch})
        result.non_tensor_batch = {k: np.concatenate([b.non_tensor_batch[k] for b in batches]) for k in result.non_tensor_batch}
        return result


class ReplicaState:
    def __init__(self, free_cache, version):
        self.free_cache = free_cache
        self.awake = True
        self.version = version
        self.events = []

    def sleep_replicas(self):
        self.events.append(("sleep", self.version))
        self.awake = False

    def update_weights(self, global_steps=None):
        assert not self.free_cache or not self.awake, "weight sync must retain VeRL's sleep precondition"
        if global_steps is not None:
            self.version = global_steps
        self.awake = True
        self.events.append(("sync", self.version))


def trainer(plans, *, free_cache=True, manager=True, stepwise=False, world_size=2, next_step=1):
    self = controller_class()()
    self.global_steps = next_step
    self._training_epoch = 0
    self.config = SimpleNamespace(
        trainer=SimpleNamespace(total_epochs=1),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2, free_cache_engine=free_cache)),
        rllm=SimpleNamespace(stepwise_advantage=SimpleNamespace(enable=stepwise, mode="broadcast"), rejection_sample=SimpleNamespace(enable=True)),
    )
    self.train_dataloader = plans
    self.state = ReplicaState(free_cache and manager, next_step - 1)
    if manager:
        self.checkpoint_manager = self.state
    self.use_critic = False
    self.use_rm = False
    self.actor_rollout_wg = SimpleNamespace(world_size=world_size)
    self.generated_versions = []
    self.updates = []
    self.init_envs_and_agents = lambda batch: setattr(self, "current", batch)
    self._consume_current_step_reward_metrics = lambda **kw: {}
    self._pad_dataproto_to_world_size = lambda batch: batch

    def generate(**kwargs):
        from rllm.utils.training_reward import raw_reward_metrics

        assert self.state.awake, "next batch attempted generation while rollout replicas were asleep"
        self.generated_versions.append(self.state.version)
        b = self.current
        b.batch["token_level_scores"] = torch.tensor(b.plan["rewards"], dtype=torch.float32)[:, None]
        n = len(b.plan["rewards"])
        b.non_tensor_batch.update(is_pad_step=np.zeros(n, dtype=bool), is_last_step=np.array(b.plan.get("last_steps", [True] * n)), idxs=np.arange(n))
        b.meta_info["repeat_counts"] = [1] * n
        b.meta_info["training_reward_metrics"] = raw_reward_metrics([{"trajectory_reward": reward} for reward in b.plan["rewards"]])
        return b

    self.generate_agent_trajectory = lambda **kw: (generate(**kw), self.current.meta_info.pop("training_reward_metrics"))
    self.generate_agent_steps = generate
    self.record_update = lambda batch: self.updates.append(self.global_steps)
    return self


def test_rejected_batch_restores_same_weights_before_next_generation():
    t = trainer([{"rewards": [0, 0, 1, 1]}, {"rewards": [0, 1]}])
    t.fit_agent()
    assert t.generated_versions == [0, 0]
    assert t.updates == [1]
    epoch = t._training_reward_tracker.state["completed_epochs"][0]
    assert (epoch["reward_sum"], epoch["reward_count"], epoch["mean"], epoch["eligible"]) == (3.0, 6, 0.5, True)
    assert t.global_steps == 2
    assert t.state.events == [("sleep", 0), ("sync", 0), ("sleep", 0), ("sync", 1)]


@pytest.mark.parametrize("stepwise", [False, True])
def test_batch_rounded_to_zero_restores_before_next_generation(stepwise):
    if stepwise:
        plans = [{"rewards": [0, 1], "last_steps": [True, False]}, {"rewards": [0, 1]}]
        world_size = 2
    else:
        plans = [{"rewards": [0, 1]}, {"rewards": [0, 1, 0, 1]}]
        world_size = 4
    t = trainer(plans, stepwise=stepwise, world_size=world_size)
    t.fit_agent()
    assert t.generated_versions == [0, 0]
    assert t.updates == [1]
    assert t.global_steps == 2


@pytest.mark.parametrize("free_cache,manager", [(False, True), (True, False), (False, False)])
def test_skip_preserves_free_cache_off_and_managerless_compatibility(free_cache, manager):
    t = trainer([{"rewards": [0, 0]}, {"rewards": [0, 1]}], free_cache=free_cache, manager=manager)
    t.fit_agent()
    assert t.updates == [1]
    assert t.state.events == ([("sync", 1)] if manager else [])


def test_accepted_batches_keep_original_sleep_sync_sequence():
    t = trainer([{"rewards": [0, 1]}, {"rewards": [0.5, 0.5]}])
    t.fit_agent()
    assert t.updates == [1, 2]
    assert t.generated_versions == [0, 1]
    assert t.state.events == [("sleep", 0), ("sync", 1), ("sleep", 1), ("sync", 2)]


def test_resume_and_repeated_rejections_do_not_advance_weight_version():
    t = trainer([{"rewards": [0, 0]}, {"rewards": [1, 1]}, {"rewards": [0, 1]}], next_step=31)
    t.fit_agent()
    assert t.generated_versions == [30, 30, 30]
    assert t.updates == [31]
    assert t.global_steps == 32
    assert t._training_epoch == 0


def test_rejection_disabled_does_not_add_restoration():
    t = trainer([{"rewards": [0, 0]}, {"rewards": [1, 1]}])
    t.config.rllm.rejection_sample.enable = False
    t.fit_agent()
    assert t.updates == [1, 2]
    assert t.state.events == [("sleep", 0), ("sync", 1), ("sleep", 1), ("sync", 2)]


def test_critic_and_reward_model_still_run_after_memory_release():
    t = trainer([{"rewards": [0, 0]}, {"rewards": [0, 1]}])
    calls = []

    def gpu_work(batch):
        assert not t.state.awake, "GPU work must retain the original memory headroom"
        calls.append(t.global_steps)
        return batch

    t.use_critic = t.use_rm = True
    t.critic_wg = SimpleNamespace(compute_values=gpu_work)
    t.rm_wg = SimpleNamespace(compute_rm_score=gpu_work)
    t.fit_agent()
    assert calls == [1, 1, 1, 1]
    assert t.updates == [1]


def test_restoration_failure_stops_instead_of_generating_with_bad_weights():
    t = trainer([{"rewards": [0, 0]}, {"rewards": [0, 1]}])

    def fail(global_steps=None):
        raise RuntimeError("weight transfer failed")

    t.checkpoint_manager.update_weights = fail
    with pytest.raises(RuntimeError, match="weight transfer failed"):
        t.fit_agent()
    assert t.generated_versions == [0]
    assert not t.updates


def test_local_ace_fit_wrappers_inherit_rejected_batch_recovery():
    t = trainer([{"rewards": [0, 0]}, {"rewards": [0, 1]}])
    ns = {"AgentPPOTrainer": t.__class__}
    for filename, name in [("trainer_base.py", "ACERLAgentPPOTrainer"), ("trainer.py", "LocalACERLAgentPPOTrainer")]:
        tree = ast.parse((ROOT / "rllm/algorithms/ace" / filename).read_text())
        cls = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name))
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit_agent"]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), filename, "exec"), ns)
    t.__class__ = ns["LocalACERLAgentPPOTrainer"]
    t._epoch_task_data = {}
    t._get_current_playbook_text = lambda: "unchanged playbook"
    t.fit_agent()
    assert t.generated_versions == [0, 0]
    assert t.updates == [1]
