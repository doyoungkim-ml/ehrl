"""Real-tensor estimator/loss checks; VeRL functions execute from verified source.

Only DataProto's row transport is substituted. GPU/Ray worker imports are outside
this CPU gate, while upstream GRPO, GAE, masking and clipped loss run unchanged.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("baseline_advantages", ROOT / "rllm/trainer/verl/advantages.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compute_raft_advantages = MODULE.compute_raft_advantages
compute_advantage_with_overrides = MODULE.compute_advantage_with_overrides


class Batch:
    def __init__(self, rewards, masks, estimators=None, uids=None):
        self.batch = {"token_level_rewards": rewards, "response_mask": masks}
        self.non_tensor_batch = {"uid": np.array(uids if uids is not None else ["task"] * len(rewards), dtype=object)}
        if estimators is not None:
            self.non_tensor_batch["adv_estimator"] = np.array(estimators, dtype=object)

    def __len__(self):
        return len(self.batch["response_mask"])

    def select_idxs(self, indices):
        selected = copy.copy(self)
        selected.batch = {k: v[indices] for k, v in self.batch.items()}
        selected.non_tensor_batch = {k: v[indices] for k, v in self.non_tensor_batch.items()}
        return selected


def extract(path, names, namespace):
    functions = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    assert len(functions) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def verl_math():
    source = os.environ.get("VERL_RESUME_SOURCE")
    assert source, "VERL_RESUME_SOURCE must name the prepared pinned source; this is required coverage"
    source = Path(source)
    namespace = {"torch": torch, "np": np, "defaultdict": defaultdict, "AlgoConfig": type("AlgoConfig", (), {})}
    extract(source / "verl/utils/torch_functional.py", {"masked_sum", "masked_mean", "masked_var", "masked_whiten", "clip_by_value"}, namespace)
    namespace["verl_F"] = SimpleNamespace(**namespace)
    extract(source / "verl/trainer/ppo/core_algos.py", {"compute_grpo_outcome_advantage", "compute_gae_advantage_return", "compute_policy_loss_vanilla", "compute_value_loss", "agg_loss"}, namespace)
    return namespace


def test_raft_clamps_outcomes_and_never_learns_observation_tokens():
    rewards = torch.tensor([[0., 0., 2.], [0., 0., -1.], [0., 0., 0.]], requires_grad=True)
    masks = torch.tensor([[1., 0., 1.], [1., 1., 0.], [1., 0., 0.]])
    advantages, returns = compute_raft_advantages(rewards, masks)
    assert torch.equal(advantages, torch.tensor([[2., 0., 2.], [0., 0., 0.], [0., 0., 0.]]))
    assert torch.equal(advantages, returns)
    assert not advantages.requires_grad


@pytest.mark.parametrize("rewards,masks", [
    (torch.ones(2), torch.ones(2)),
    (torch.ones(2, 3), torch.ones(2, 2)),
    (torch.tensor([[float("nan")]]), torch.ones(1, 1)),
    (torch.ones(1, 1), torch.tensor([[.5]])),
])
def test_invalid_raft_records_fail_closed(rewards, masks):
    with pytest.raises(ValueError):
        compute_raft_advantages(rewards, masks)


@pytest.mark.parametrize("tags", [None, ["grpo", None]])
def test_ordinary_batches_delegate_unchanged(tags):
    batch = Batch(torch.zeros(2, 3), torch.ones(2, 3), tags)
    calls = []

    def backend(selected, **kwargs):
        calls.append((selected, kwargs))
        return selected

    result = compute_advantage_with_overrides(batch, compute_fn=backend, adv_estimator="grpo", gamma=1., lam=.95)
    assert result is batch
    assert calls == [(batch, {"adv_estimator": "grpo", "gamma": 1., "lam": .95})]


def test_mixed_estimators_preserve_order_and_isolate_grpo_groups(verl_math):
    # Same task UID on every row is deliberate: RAFT copies must not alter the
    # GRPO population's mean/std, even when identifiers are shared.
    rewards = torch.tensor([[0., 0.], [0., 10.], [0., 1.], [0., 0.]])
    masks = torch.tensor([[1., 0.], [1., 1.], [1., 1.], [1., 0.]])
    batch = Batch(rewards, masks, ["grpo", "raft", "grpo", "raft"])
    calls = []

    def backend(selected, *, adv_estimator, **kwargs):
        calls.append((adv_estimator, selected.non_tensor_batch["uid"].tolist(), len(selected)))
        advantages, returns = verl_math["compute_grpo_outcome_advantage"](
            selected.batch["token_level_rewards"], selected.batch["response_mask"], selected.non_tensor_batch["uid"], norm_adv_by_std_in_grpo=False
        )
        selected.batch.update(advantages=advantages, returns=returns)
        return selected

    result = compute_advantage_with_overrides(batch, compute_fn=backend, adv_estimator="grpo")
    assert result is batch
    assert calls == [("grpo", ["task", "task"], 2)]
    assert torch.equal(batch.batch["advantages"], torch.tensor([[-.5, 0.], [10., 10.], [.5, .5], [0., 0.]]))
    assert torch.equal(batch.batch["returns"], batch.batch["advantages"])


def test_misaligned_tags_fail_before_scoring():
    batch = Batch(torch.zeros(2, 2), torch.ones(2, 2), ["raft"])
    with pytest.raises(ValueError, match="align"):
        compute_advantage_with_overrides(batch, compute_fn=lambda *a, **k: pytest.fail("called backend"), adv_estimator="grpo")


def test_raft_initial_gradient_matches_weighted_imitation_and_ppo_clips(verl_math):
    mask = torch.tensor([[1., 0., 1.], [1., 1., 0.]])
    advantages, _ = compute_raft_advantages(torch.tensor([[0., 0., 2.], [0., 0., 0.]]), mask)
    config = SimpleNamespace(clip_ratio=.2, clip_ratio_low=None, clip_ratio_high=None, global_batch_info={})
    config.get = lambda key, default: default
    log_probs = torch.tensor([[-.9, -.4, -.7], [-.8, -.6, -.5]], requires_grad=True)
    loss, _ = verl_math["compute_policy_loss_vanilla"](log_probs.detach(), log_probs, advantages, mask, config=config)
    actual = torch.autograd.grad(loss, log_probs)[0]
    expected = torch.autograd.grad(-(log_probs * advantages * mask).sum() / mask.sum(), log_probs)[0]
    assert torch.equal(actual, expected)
    assert actual[0, 1] == 0
    assert torch.equal(actual[1], torch.zeros(3))
    # A later policy beyond the positive clip boundary has zero PG gradient.
    moved = (log_probs.detach() + .4).requires_grad_()
    clipped, _ = verl_math["compute_policy_loss_vanilla"](log_probs.detach(), moved, advantages, mask, config=config)
    assert torch.equal(torch.autograd.grad(clipped, moved)[0], torch.zeros_like(moved))


def test_pinned_gae_bootstraps_critic_and_skips_observation_values(verl_math):
    rewards = torch.tensor([[0., 0., 1.], [0., 0., 0.]])
    mask = torch.tensor([[1., 0., 1.], [1., 0., 1.]])
    values = torch.tensor([[.2, 999., .4], [.3, -999., .5]])
    advantages, returns = verl_math["compute_gae_advantage_return"](rewards, values, mask, gamma=1., lam=.95)
    # At action 0: delta=.4-.2, then .95*(1-.4); observations
    # contribute neither a new discount step nor their critic values.
    torch.testing.assert_close(returns[:, [0, 2]], torch.tensor([[.97, 1.], [.025, 0.]]))
    assert torch.isfinite(advantages).all()
    shifted_values = values.clone()
    shifted_values[:, 1] = 0
    shifted, _ = verl_math["compute_gae_advantage_return"](rewards, shifted_values, mask, gamma=1., lam=.95)
    torch.testing.assert_close((shifted * mask), (advantages * mask))
