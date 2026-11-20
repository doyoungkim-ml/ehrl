"""CPU regressions for workflow PPO updates with dropped rollout episodes.

Execute actual workflow helpers and the prepared VeRL padding/iterator sources.
Only DataProto/TensorDict transport is substituted; tensor masks, autograd, and
minibatch iteration are real. This does not claim a Ray/GPU optimizer update.
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import math
import os
import subprocess
from contextlib import nullcontext
from functools import reduce
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "rllm/trainer/verl/agent_workflow_trainer.py"


class TensorDict(dict):
    """TensorDict row transport sufficient for the real pinned iterator."""

    def __init__(self, source=None, batch_size=None):
        super().__init__(source or {})
        if isinstance(batch_size, int):
            batch_size = [batch_size]
        self.batch_size = torch.Size(batch_size or [len(next(iter(self.values())))])
        self.shape = self.batch_size


class Batch:
    """DataProto transport only; padding/selection decisions are production code."""

    def __init__(self, tensors, non_tensors, meta_info=None):
        self.batch = tensors
        self.non_tensor_batch = non_tensors
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(self.batch["response_mask"])

    def __getitem__(self, indices):
        return Batch(
            {key: value[indices] for key, value in self.batch.items()},
            {key: value[indices] for key, value in self.non_tensor_batch.items()},
            copy.deepcopy(self.meta_info),
        )

    def select_idxs(self, indices):
        return self[indices]

    def to_tensordict(self):
        return TensorDict(self.batch, batch_size=[len(self)])

    @classmethod
    def concat(cls, batches):
        return cls(
            {key: torch.cat([batch.batch[key] for batch in batches]) for key in batches[0].batch},
            {key: np.concatenate([batch.non_tensor_batch[key] for batch in batches]) for key in batches[0].non_tensor_batch},
            copy.deepcopy(batches[0].meta_info),
        )


def real_functions(path, names, namespace):
    functions = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    assert len(functions) == len(names)
    exec_nodes(functions, namespace, filename=str(path))


def exec_nodes(nodes, namespace, filename="<actual-workflow-block>"):
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *copy.deepcopy(nodes)],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)


@pytest.fixture(scope="module")
def runtime():
    source_path = os.environ.get("VERL_RESUME_SOURCE")
    assert source_path, "VERL_RESUME_SOURCE must point to the prepared pinned source; this coverage cannot be skipped"
    source = Path(source_path)
    manifest = json.loads((ROOT / "compat/verl/manifest.json").read_text())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert revision == manifest["base_revision"], "Workflow padding must be checked against the maintained VeRL pin"
    ns = {"torch": torch, "np": np, "copy": copy, "math": math, "reduce": reduce, "logging": logging, "DataProto": Batch, "TensorDict": TensorDict}
    real_functions(source / "verl/protocol.py", {"pad_dataproto_to_divisor"}, ns)
    real_functions(source / "verl/utils/tensordict_utils.py", {"make_iterator", "index_select_tensor_dict"}, ns)
    real_functions(source / "verl/utils/torch_functional.py", {"masked_sum", "masked_mean", "clip_by_value"}, ns)
    ns["verl_F"] = SimpleNamespace(**ns)
    real_functions(source / "verl/trainer/ppo/core_algos.py", {"agg_loss", "compute_value_loss"}, ns)
    cls = next(node for node in ast.parse(WORKFLOW.read_text()).body if isinstance(node, ast.ClassDef))
    cls.bases = []
    cls.body = [
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_pad_workflow_ppo_update_batch", "_pad_dataproto_to_world_size", "_remove_padding"}
    ]
    exec_nodes([cls], ns)
    return ns


def rows(n):
    return Batch(
        {
            "prompts": torch.arange(n).unsqueeze(1),
            "input_ids": torch.zeros(n, 4, dtype=torch.long),
            "response_mask": torch.tensor([[1, 0, 1]] * n, dtype=torch.long).reshape(n, 3),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
            "values": torch.zeros(n, 3),
            "returns": torch.ones(n, 3),
        },
        {
            "is_valid": np.ones(n, dtype=bool),
            "is_pad_step": np.zeros(n, dtype=bool),
            "is_last_step": np.ones(n, dtype=bool),
            "task_ids": np.array([f"task-{i}" for i in range(n)], dtype=object),
        },
        {"global_token_num": [4] * n},
    )


def trainer(runtime, use_critic=True):
    obj = runtime["AgentWorkflowPPOTrainer"]()
    obj.use_critic = use_critic
    obj.use_reference_policy = obj.use_rm = False
    obj.hybrid_engine = True
    obj.critic_wg = SimpleNamespace(world_size=2)
    obj.actor_rollout_wg = SimpleNamespace(world_size=2)
    return obj


def test_dropped_episodes_reproduce_pinned_iterator_failure_then_fit(runtime):
    obj = trainer(runtime)
    batch = rows(62)
    world_only = obj._pad_dataproto_to_world_size(batch)
    assert len(world_only) == 62
    # This is the exact source assertion that failed on two 31-row ranks.
    with pytest.raises(AssertionError, match="31 % 32 != 0"):
        runtime["make_iterator"](world_only.select_idxs(list(range(31))).to_tensordict(), mini_batch_size=32, epochs=1)
    metrics = {}
    padded = obj._pad_workflow_ppo_update_batch(batch, worker_group=obj.critic_wg, global_mini_batch_size=64, metrics=metrics, metric_prefix="critic")
    assert len(padded) == 64
    assert len(batch) == 62 and batch.batch["response_mask"].sum() == 124
    assert batch.meta_info["global_token_num"] == [4] * 62
    assert padded.batch["response_mask"][62:].sum() == 0
    assert padded.non_tensor_batch["is_pad_step"].sum() == 2
    assert not padded.non_tensor_batch["is_valid"][62:].any()
    assert not padded.non_tensor_batch["is_last_step"][62:].any()
    assert metrics == {"batch/critic_prepad_size": 62, "batch/critic_pad_size": 2, "batch/critic_pad_divisor": 64}
    observed_tokens = 0
    for rank in range(2):
        shard = padded.select_idxs(list(range(rank * 32, (rank + 1) * 32))).to_tensordict()
        minibatches = list(runtime["make_iterator"](shard, mini_batch_size=32, epochs=1, seed=42, dataloader_kwargs={"shuffle": True}))
        assert len(minibatches) == 1
        observed_tokens += int(minibatches[0]["response_mask"].sum())
    assert observed_tokens == 124


def test_existing_world_padding_cannot_become_real_targets(runtime):
    obj = trainer(runtime)
    batch = rows(61)
    prior = obj._pad_dataproto_to_world_size(batch)
    assert len(prior) == 62 and prior.batch["response_mask"][-1].sum() == 2  # existing inference behavior
    padded = obj._pad_workflow_ppo_update_batch(prior, worker_group=obj.critic_wg, global_mini_batch_size=64)
    assert len(padded) == 64
    assert padded.non_tensor_batch["is_pad_step"].sum() == 3
    assert padded.batch["response_mask"].sum() == batch.batch["response_mask"].sum()
    assert prior.batch["response_mask"][-1].sum() == 2  # input is untouched
    assert padded.non_tensor_batch["task_ids"][:61].tolist() == batch.non_tensor_batch["task_ids"].tolist()


def test_worker_minibatch_sizes_are_independent_and_lcm_dispatch_safe(runtime):
    obj = trainer(runtime)
    batch = rows(94)
    actor = obj._pad_workflow_ppo_update_batch(batch, worker_group=obj.actor_rollout_wg, global_mini_batch_size=32)
    critic = obj._pad_workflow_ppo_update_batch(batch, worker_group=obj.critic_wg, global_mini_batch_size=64)
    assert (len(actor), len(critic), len(batch)) == (96, 128, 94)
    assert actor.batch["response_mask"].sum() == critic.batch["response_mask"].sum() == 188
    uneven = obj._pad_workflow_ppo_update_batch(rows(17), worker_group=SimpleNamespace(world_size=6), global_mini_batch_size=8)
    assert len(uneven) == math.lcm(6, 8) == 24


def test_synthetic_value_targets_have_zero_autograd_contribution(runtime):
    obj = trainer(runtime)
    batch = obj._pad_workflow_ppo_update_batch(rows(62), worker_group=obj.critic_wg, global_mini_batch_size=64)
    predictions = torch.zeros_like(batch.batch["values"], requires_grad=True)
    loss, _ = runtime["compute_value_loss"](
        vpreds=predictions, values=batch.batch["values"], returns=batch.batch["returns"],
        response_mask=batch.batch["response_mask"], cliprange_value=0.2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert predictions.grad[:62, [0, 2]].abs().sum() > 0
    assert predictions.grad[62:].abs().sum() == 0
    assert predictions.grad[:, 1].abs().sum() == 0


@pytest.mark.parametrize("count", [0, 3])
def test_no_rows_or_no_action_tokens_fail_before_worker_dispatch(runtime, count):
    batch = rows(count)
    batch.batch["response_mask"].zero_()
    obj = trainer(runtime)
    with pytest.raises(ValueError, match="no valid rows|no action tokens"):
        obj._pad_workflow_ppo_update_batch(batch, worker_group=obj.critic_wg, global_mini_batch_size=64)


def test_actual_critic_update_passes_masked_complete_minibatch(runtime):
    obj = trainer(runtime)
    obj.config = SimpleNamespace(
        critic=SimpleNamespace(ppo_mini_batch_size=8),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=8)),
    )
    received = []
    obj._update_critic = lambda batch: (received.append(batch) or SimpleNamespace(meta_info={"metrics": {}}))
    fit = next(node for node in ast.walk(ast.parse(WORKFLOW.read_text())) if isinstance(node, ast.FunctionDef) and node.name == "fit_agent")
    critic = next(
        node for node in ast.walk(fit)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.use_critic" and "critic_output = self._update_critic" in ast.unparse(node)
    )
    ns = {
        "self": obj, "batch": rows(62), "metrics": {}, "timing_raw": {},
        "marked_timer": lambda *args, **kwargs: nullcontext(), "reduce_metrics": lambda metrics: metrics,
    }
    original = ns["batch"]
    exec_nodes([critic], ns)
    assert ns["batch"] is original and len(original) == 62
    assert len(received) == 1 and len(received[0]) == 64
    assert received[0].batch["response_mask"].sum() == 124
    assert not received[0].batch["response_mask"][62:].any()
