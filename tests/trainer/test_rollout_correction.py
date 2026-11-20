"""FP8 rollout correction: settings, weights, transport and the actor loss.

The gate supplies VERL_RESUME_SOURCE. VeRL's rollout-correction helper, its
DataProto, no-padding conversion, microbatch iterator and the vanilla PPO
policy loss execute from that source (AST-loaded without Ray/GPU imports).
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_workflow_actor_batching import load_nodes, runtime  # noqa: F401

from rllm.algorithms.erl.replay import make_internalization_copy, pack_replay_tensors
from rllm.trainer.verl.rollout_correction import (
    apply_rollout_correction,
    pad_rollout_rows,
    rollout_correction_settings,
    sampling_matches_training,
    trajectory_rollout_tensors,
)

ROOT = Path(__file__).resolve().parents[2]


def config(**overrides):
    base = {
        "algorithm": {"rollout_correction": {"rollout_is": "token", "rollout_is_threshold": 2.0, "rollout_rs": None, "bypass_mode": False, "rollout_is_batch_normalize": False}},
        "actor_rollout_ref": {"rollout": {"temperature": 0.7, "top_p": 1.0, "top_k": -1, "logprobs_mode": "processed_logprobs", "quantization": "fp8"}},
        "rllm": {"stepwise_advantage": {"enable": False}},
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


def test_settings_are_disabled_by_default_and_validated():
    assert rollout_correction_settings(OmegaConf.create({"algorithm": {"rollout_correction": {"rollout_is": None}}})) is None
    assert rollout_correction_settings(config()) == {"rollout_is": "token", "rollout_is_threshold": 2.0, "quantization": "fp8"}
    for overrides in (
        {"algorithm": {"rollout_correction": {"rollout_is": "sequence"}}},
        {"algorithm": {"rollout_correction": {"rollout_rs": "token_k1"}}},
        {"algorithm": {"rollout_correction": {"bypass_mode": True}}},
        {"algorithm": {"rollout_correction": {"rollout_is_threshold": "0.5_2.0"}}},
        {"actor_rollout_ref": {"rollout": {"top_p": 0.9}}},
        {"actor_rollout_ref": {"rollout": {"top_k": 20}}},
        {"actor_rollout_ref": {"rollout": {"logprobs_mode": "raw_logprobs"}}},
        {"rllm": {"stepwise_advantage": {"enable": True}}},
    ):
        with pytest.raises(ValueError):
            rollout_correction_settings(config(**overrides))


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"temperature": 0.7, "top_p": 1.0, "top_k": -1, "logprobs": 1, "max_tokens": 9}, True),
        ({"temperature": 0.7}, True),
        ({"temperature": 0.6, "top_p": 1.0}, False),
        ({"temperature": 0.7, "top_p": 0.9}, False),  # ERL updater sampling
        ({"temperature": 0.7, "top_k": 20}, False),
        ({"temperature": 0.0}, False),
        ({"temperature": 0.7, "repetition_penalty": 1.1}, False),
        ({"temperature": 0.7, "logit_bias": {1: 2.0}}, False),
    ],
)
def test_only_the_training_distribution_is_comparable(params, expected):
    assert sampling_matches_training(params, 0.7) is expected


@pytest.fixture
def verl_correction(runtime, monkeypatch):  # noqa: F811
    """Expose VeRL's real rollout-correction helper under its import path."""
    source = Path(os.environ["VERL_RESUME_SOURCE"]) / "verl"
    namespace = {"torch": torch, "math": math, "Optional": __import__("typing").Optional, "Any": __import__("typing").Any, "DataProto": runtime.DataProto}
    functional = {"torch": torch}
    load_nodes(source / "utils/torch_functional.py", ["masked_sum", "masked_mean"], functional)
    namespace["verl_F"] = SimpleNamespace(masked_sum=functional["masked_sum"], masked_mean=functional["masked_mean"])
    tree = ast.parse((source / "trainer/ppo/rollout_corr_helper.py").read_text())
    constant = next(node for node in tree.body if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SAFETY_BOUND")
    exec(compile(ast.Module(body=[constant], type_ignores=[]), "rollout_corr_helper", "exec"), namespace)
    load_nodes(
        source / "trainer/ppo/rollout_corr_helper.py",
        ["_parse_rollout_is_threshold", "compute_rollout_correction_weights", "compute_is_metrics", "compute_offpolicy_metrics", "compute_rollout_correction_and_rejection_mask"],
        namespace,
    )
    helper = ModuleType("verl.trainer.ppo.rollout_corr_helper")
    helper.__dict__.update(namespace)
    for name in ("verl", "verl.trainer", "verl.trainer.ppo"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    return runtime


def correction_batch(runtime):  # noqa: F811
    # Row 0: two turns, env/template tokens at positions 2-3 masked.
    # Row 1: an ERL updater row (no comparable sampler log-probs).
    # Row 2: a transport copy that must not count in metrics.
    response_mask = torch.tensor([[1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    old = torch.tensor([[-0.10, -1.00, 0.0, 0.0, -0.50, -2.00], [-0.3, -0.3, -0.3, 0.0, 0.0, 0.0], [-0.1, -0.1, 0.0, 0.0, 0.0, 0.0]])
    rollout = torch.tensor([[-0.12, -2.00, -9.0, -9.0, -0.40, -2.00], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, 0.0, 0.0, 0.0, 0.0]])
    available = torch.tensor([[1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    rows = len(response_mask)
    attention = torch.cat([torch.ones(rows, 2, dtype=torch.long), torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]])], dim=1)
    tensors = {
        "input_ids": torch.arange(8).repeat(rows, 1) + 10,
        "position_ids": torch.arange(8).repeat(rows, 1),
        "attention_mask": attention,
        "response_mask": response_mask,
        "old_log_probs": old,
        "rollout_log_probs": rollout,
        "rollout_log_prob_mask": available,
        "advantages": torch.tensor([[1.0] * 6, [-1.0] * 6, [0.5] * 6]),
    }
    flags = {"is_pad_step": np.array([False, False, True])}
    return runtime.DataProto.from_dict(tensors=tensors, non_tensors=flags, meta_info={"temperature": 0.7})


def test_weights_cover_only_comparable_action_tokens_and_metrics_exclude_the_rest(verl_correction):
    batch = correction_batch(verl_correction)
    batch, metrics = apply_rollout_correction(batch, {"rollout_is_threshold": 2.0}, excluded_rows=batch.non_tensor_batch["is_pad_step"])
    weights = batch.batch["rollout_is_weights"]
    expected_row0 = [math.exp(0.02), 2.0, 0.0, 0.0, math.exp(-0.1), 1.0]  # exp(1.0) clipped to 2
    torch.testing.assert_close(weights[0], torch.tensor(expected_row0), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weights[1], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(weights[2], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    available = torch.tensor([True, True, False, False, True, True])
    diff = (batch.batch["old_log_probs"][0] - batch.batch["rollout_log_probs"][0])[available]
    assert metrics["rollout_corr/rows_with_rollout_log_probs"] == 1
    assert metrics["rollout_corr/available_token_fraction"] == pytest.approx(4 / 9)
    assert metrics["rollout_corr/rollout_is_mean"] == pytest.approx(float(torch.tensor(expected_row0)[available].mean()), rel=1e-5)
    assert metrics["rollout_corr/rollout_is_ratio_fraction_high"] == pytest.approx(0.25)
    assert metrics["rollout_corr/rollout_is_clipped_fraction"] == pytest.approx(0.25)
    assert metrics["rollout_corr/kl"] == pytest.approx(float((-diff).mean()), rel=1e-5)
    assert metrics["rollout_corr/log_prob_abs_diff_max"] == pytest.approx(1.0)
    assert metrics["rollout_corr/log_prob_abs_diff_mean"] == pytest.approx(float(diff.abs().mean()), rel=1e-6)


def test_batch_without_comparable_tokens_is_uncorrected(verl_correction):
    batch = correction_batch(verl_correction)
    batch.batch["rollout_log_prob_mask"].zero_()
    batch, metrics = apply_rollout_correction(batch, {"rollout_is_threshold": 2.0})
    torch.testing.assert_close(batch.batch["rollout_is_weights"], batch.batch["response_mask"].float())
    assert metrics["rollout_corr/available_token_fraction"] == 0
    assert "rollout_corr/kl" not in metrics


def test_missing_rollout_log_probs_fail_loudly(verl_correction):
    batch = correction_batch(verl_correction)
    batch.batch.pop("rollout_log_probs")
    with pytest.raises(ValueError, match="rollout_log_probs"):
        apply_rollout_correction(batch, {"rollout_is_threshold": 2.0})


def test_weights_survive_no_padding_transport_and_scale_the_vanilla_policy_loss(verl_correction):
    """The fields losses.ppo_loss selects reach compute_policy_loss_vanilla."""
    runtime = verl_correction  # noqa: F811
    batch, _ = apply_rollout_correction(correction_batch(runtime), {"rollout_is_threshold": 2.0})
    td = runtime.left_right_2_no_padding(batch.to_tensordict())
    runtime.tu.assign_non_tensor(td, global_batch_size=3, mini_batch_size=3, epochs=1, seed=1, dataloader_kwargs={"shuffle": False})
    (microbatch,) = list(runtime.tu.make_iterator(td, mini_batch_size=3, epochs=1, seed=1, dataloader_kwargs={"shuffle": False}))
    # verl/workers/utils/losses.py: data.select(..., "rollout_is_weights").to_padded_tensor()
    selected = microbatch.select("response_mask", "old_log_probs", "advantages", "rollout_is_weights")
    torch.testing.assert_close(selected["rollout_is_weights"], batch.batch["rollout_is_weights"])

    source = Path(os.environ["VERL_RESUME_SOURCE"]) / "verl/trainer/ppo/core_algos.py"
    namespace = {
        "torch": torch,
        "Optional": __import__("typing").Optional,
        "Any": __import__("typing").Any,
        "ActorConfig": object,
        "AlgoConfig": type("AlgoConfig", (), {}),
        "register_policy_loss": lambda name: (lambda fn: fn),
    }
    functional = {"torch": torch}
    load_nodes(Path(os.environ["VERL_RESUME_SOURCE"]) / "verl/utils/torch_functional.py", ["masked_sum", "masked_mean"], functional)
    namespace["verl_F"] = SimpleNamespace(masked_mean=functional["masked_mean"], masked_sum=functional["masked_sum"])
    load_nodes(source, ["agg_loss", "compute_policy_loss_vanilla"], namespace)

    class ActorConfig(dict):
        clip_ratio, clip_ratio_low, clip_ratio_high, global_batch_info = 0.2, 0.2, 0.28, {}

    log_prob = selected["old_log_probs"] + 0.05
    arguments = dict(
        old_log_prob=selected["old_log_probs"], log_prob=log_prob, advantages=selected["advantages"], response_mask=selected["response_mask"].bool(), loss_agg_mode="token-mean", config=ActorConfig()
    )
    weighted, _ = namespace["compute_policy_loss_vanilla"](**arguments, rollout_is_weights=selected["rollout_is_weights"])
    plain, _ = namespace["compute_policy_loss_vanilla"](**arguments)
    ratio = torch.exp(torch.tensor(0.05))
    clipped = torch.clamp(ratio, 0.8, 1.28)
    advantages, mask = selected["advantages"], selected["response_mask"].bool()
    per_token = torch.maximum(-advantages * ratio, -advantages * clipped)
    expected = (per_token * selected["rollout_is_weights"])[mask].mean()
    torch.testing.assert_close(weighted, expected)
    assert not torch.isclose(weighted, plain)


def test_transport_helpers_align_and_truncate_like_tokens():
    trajectory = {"rollout_log_probs": torch.tensor([-0.1, 0.0, -0.3]), "rollout_log_prob_mask": torch.tensor([1, 0, 1])}
    log_probs, mask = trajectory_rollout_tensors(trajectory, 3)
    assert mask.tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        trajectory_rollout_tensors(trajectory, 4)
    assert trajectory_rollout_tensors({}, 2)[1].tolist() == [0, 0]
    padded = pad_rollout_rows([log_probs, torch.tensor([-1.0])], 2, torch.float32)
    torch.testing.assert_close(padded, torch.tensor([[-0.1, 0.0], [-1.0, 0.0]]))


def test_erl_packing_keeps_contextual_rows_and_drops_replaced_prompts():
    first = {
        "prompt_tokens": torch.tensor([1, 2]),
        "response_tokens": torch.tensor([3, 4, 5]),
        "response_masks": torch.tensor([1, 0, 1]),
        "rollout_log_probs": torch.tensor([-0.1, 0.0, -0.2]),
        "rollout_log_prob_mask": torch.tensor([1, 0, 1]),
        "trajectory_reward": 1.0,
    }
    distilled = make_internalization_copy(first, torch.tensor([7]))
    assert "rollout_log_probs" not in distilled and "rollout_log_prob_mask" not in distilled
    updater = {**first, "rollout_log_prob_mask": torch.zeros(3, dtype=torch.long)}
    packed = pack_replay_tensors([first, distilled, updater], max_prompt_length=3, max_response_length=4, pad_token_id=0, include_rollout_log_probs=True)
    assert packed["rollout_log_prob_mask"].tolist() == [[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    torch.testing.assert_close(packed["rollout_log_probs"][0], torch.tensor([-0.1, 0.0, -0.2, 0.0]))
    assert "rollout_log_probs" not in pack_replay_tensors([first], max_prompt_length=3, max_response_length=4, pad_token_id=0)


def load_workflow_transform():
    dataclass_spec = importlib.util.spec_from_file_location("workflow_dataclass", ROOT / "rllm/experimental/verl/dataclass.py")
    dataclasses = importlib.util.module_from_spec(dataclass_spec)
    dataclass_spec.loader.exec_module(dataclasses)
    namespace = {
        "torch": torch,
        "logger": __import__("logging").getLogger(__name__),
        "ProcessedStepData": dataclasses.ProcessedStepData,
        "Trajectory": object,
        "AccumulatedData": dataclasses.AccumulatedData,
    }
    load_nodes(ROOT / "rllm/experimental/verl/transform.py", ["_is_prefix", "_step_rollout_log_probs", "_process_trajectory", "_pad_sequence_batch"], namespace)
    return namespace, dataclasses


def workflow_step(prompt, completion, matched=True):
    output = SimpleNamespace(prompt_ids=prompt, completion_ids=completion, logprobs=[-0.01 * (i + 1) for i in range(len(completion))], logprobs_match_training=matched, multi_modal_inputs=None)
    return SimpleNamespace(model_output=output, reward=0.0, advantage=None)


def test_workflow_prefix_merge_carries_exact_log_probs():
    namespace, dataclasses = load_workflow_transform()
    steps = [workflow_step([1, 2], [3, 4]), workflow_step([1, 2, 3, 4, 5, 6], [7], matched=False), workflow_step([1, 2, 3, 4, 5, 6, 7, 8], [9, 10]), workflow_step([42], [11])]
    trajectory = SimpleNamespace(name="solver", steps=steps, reward=1.0)
    accumulated = dataclasses.AccumulatedData()
    rows = namespace["_process_trajectory"](trajectory, "task", accumulated, rollout_correction=True)
    assert rows == 2  # the last step does not extend the prefix
    assert accumulated.responses[0].tolist() == [3, 4, 5, 6, 7, 8, 9, 10]
    assert accumulated.traj_mask[0].tolist() == [1, 1, 0, 0, 1, 0, 1, 1]
    assert accumulated.rollout_logprob_masks[0].tolist() == [1, 1, 0, 0, 0, 0, 1, 1]
    torch.testing.assert_close(accumulated.rollout_logprobs[0][[0, 1, 6, 7]], torch.tensor([-0.01, -0.02, -0.01, -0.02]))
    assert accumulated.rollout_logprob_masks[1].tolist() == [1]
    # Without correction the legacy transform keeps merged rows log-prob free.
    legacy = dataclasses.AccumulatedData()
    namespace["_process_trajectory"](trajectory, "task", legacy)
    assert legacy.rollout_logprob_masks == [] and len(legacy.rollout_logprobs) == 1
