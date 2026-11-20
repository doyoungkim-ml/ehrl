"""CPU regressions for the production ACE trajectory-to-tensor transform.

The full production method runs with real PyTorch/NumPy. DataProto transports
are small stand-ins, and the VeRL padding adapter delegates to torch padding;
no Ray/GPU imports or model-sized tensors are needed.
"""

import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "rllm/trainer/verl/agent_ppo_trainer.py"
MINISWE_LIMITS = {
    "policy": {"max_prompt_length": 8192, "max_response_length": 32768},
    "reflector": {"max_prompt_length": 30720, "max_response_length": 2048},
    "curator": {"max_prompt_length": 8192, "max_response_length": 4096},
}


class TensorProto:
    """Transport only: every production tensor is retained without conversion."""

    def __init__(self, tensors, non_tensors=None):
        self.batch = tensors
        self.non_tensor_batch = non_tensors or {}

    @classmethod
    def from_dict(cls, tensors, non_tensors=None):
        return cls(tensors, non_tensors)

    @classmethod
    def concat(cls, batches):
        return cls({key: torch.cat([batch.batch[key] for batch in batches], dim=0) for key in batches[0].batch})


def pad_sequence_to_length(tensor, length, pad_token_id, left_pad=False):
    padding = max(0, length - tensor.shape[-1])
    return torch.nn.functional.pad(tensor, (padding, 0) if left_pad else (0, padding), value=pad_token_id)


def transform_trainer(tmp_path, *, prompt_limit=8192, response_limit=32768, component_limits=None):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentPPOTrainer")
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_transform_agent_trajectories"))
    # Replace only the dependency import; grouping, padding, slicing, masks,
    # rewards, logging and return construction are the original production code.
    method.body = [node for node in method.body if not (isinstance(node, ast.ImportFrom) and node.module == "verl.utils.torch_functional")]
    namespace = {"np": np, "torch": torch, "os": os, "json": json, "DataProto": TensorProto, "pad_sequence_to_length": pad_sequence_to_length}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), "exec"), namespace)
    cls = type("TransformTrainer", (), {"_transform_agent_trajectories": namespace[method.name]})
    instance = cls()
    instance.config = OmegaConf.create({"data": {"max_prompt_length": prompt_limit, "max_response_length": response_limit}, "trainer": {"default_local_dir": str(tmp_path)}})
    instance.tokenizer = SimpleNamespace(pad_token_id=0)
    instance.global_steps = 1
    instance.visualize_trajectory = lambda batch: None
    # Generation has restored the smaller policy limits before transformation.
    instance.agent_execution_engine = SimpleNamespace(max_prompt_length=prompt_limit, max_response_length=response_limit)
    if component_limits is not None:
        instance.component_token_limits = copy.deepcopy(component_limits)
    return instance


def trajectory(prompt_length, response_length, *, index=0):
    prompt = torch.arange(11, prompt_length + 11, dtype=torch.long)
    response = torch.arange(100001, response_length + 100001, dtype=torch.long)
    # A real generated pad-token ID must remain attended; lengths determine padding.
    prompt[0] = 0
    response[0] = 0
    mask = torch.ones(response_length, dtype=torch.long)
    mask[1::3] = 0  # e.g. environment/tool tokens outside the optimizer loss mask
    return {
        "idx": index,
        "prompt_tokens": prompt,
        "response_tokens": response,
        "response_masks": mask,
        "trajectory_reward": 0.25,
        "chat_completions": [],
        "metrics": {},
    }


def assert_preserved(batch, row, source, prompt_width, response_width):
    tensors = batch.batch
    prompt = source["prompt_tokens"]
    response = source["response_tokens"]
    prompt_length, response_length = len(prompt), len(response)
    assert tensors["prompts"].shape[1] == prompt_width
    assert tensors["responses"].shape[1] == response_width
    assert torch.equal(tensors["prompts"][row, -prompt_length:], prompt)
    assert not tensors["prompts"][row, : prompt_width - prompt_length].any()
    assert torch.equal(tensors["responses"][row, :response_length], response)
    assert not tensors["responses"][row, response_length:].any()
    assert torch.equal(tensors["response_mask"][row, :response_length], source["response_masks"])
    assert not tensors["response_mask"][row, response_length:].any()
    attention = tensors["attention_mask"][row]
    expected_attention = torch.zeros(prompt_width + response_width, dtype=torch.long)
    expected_attention[prompt_width - prompt_length : prompt_width + response_length] = 1
    assert torch.equal(attention, expected_attention)
    assert attention.sum().item() == prompt_length + response_length
    assert torch.equal(tensors["input_ids"][row][attention.bool()], torch.cat([prompt, response]))
    assert torch.equal(tensors["position_ids"][row][attention.bool()], torch.arange(prompt_length + response_length))
    assert tensors["token_level_scores"][row].sum().item() == pytest.approx(source["trajectory_reward"])
    assert tensors["token_level_scores"][row, response_length - 1].item() == pytest.approx(source["trajectory_reward"])


def test_mixed_miniswe_roles_preserve_full_generated_context_and_loss_masks(tmp_path):
    instance = transform_trainer(tmp_path, component_limits=MINISWE_LIMITS)
    sources = [trajectory(limits["max_prompt_length"], limits["max_response_length"], index=index) for index, limits in enumerate(MINISWE_LIMITS.values())]
    original_limits = copy.deepcopy(instance.component_token_limits)
    batch, _ = instance._transform_agent_trajectories(sources)
    for row, (source, limits) in enumerate(zip(sources, MINISWE_LIMITS.values(), strict=True)):
        assert_preserved(batch, row, source, 30720, 32768)
        assert batch.batch["attention_mask"][row].sum().item() == limits["max_prompt_length"] + limits["max_response_length"]
    assert batch.batch["input_ids"].shape == (3, 63488)
    assert instance.config.data.max_prompt_length == 8192
    assert instance.config.data.max_response_length == 32768
    assert instance.component_token_limits == original_limits
    assert instance.agent_execution_engine.max_prompt_length == 8192
    assert instance.agent_execution_engine.max_response_length == 32768


def test_separate_role_and_partial_batches_have_stable_concat_shapes(tmp_path):
    instance = transform_trainer(tmp_path, component_limits=MINISWE_LIMITS)
    sources = [trajectory(37, 19), trajectory(20001, 17), trajectory(81, 23)]
    batches = [instance._transform_agent_trajectories([source])[0] for source in sources]
    combined = TensorProto.concat(batches)
    together, _ = instance._transform_agent_trajectories(sources)
    for name, tensor in together.batch.items():
        assert torch.equal(combined.batch[name], tensor), name
    for row, source in enumerate(sources):
        assert_preserved(combined, row, source, 30720, 32768)


def test_component_response_limit_can_exceed_global_response_padding(tmp_path):
    limits = {
        "policy": {"max_prompt_length": 8, "max_response_length": 4},
        "reflector": {"max_prompt_length": 12, "max_response_length": 9},
        "curator": {"max_prompt_length": 6, "max_response_length": 5},
    }
    instance = transform_trainer(tmp_path, prompt_limit=8, response_limit=4, component_limits=limits)
    sources = [trajectory(12, 9), trajectory(6, 5)]
    batches = [instance._transform_agent_trajectories([source])[0] for source in sources]
    combined = TensorProto.concat(batches)
    for row, source in enumerate(sources):
        assert_preserved(combined, row, source, 12, 9)
    assert instance.config.data.max_response_length == 4


def test_global_data_limits_remain_minimum_padding_widths(tmp_path):
    limits = {"policy": {"max_prompt_length": 4, "max_response_length": 2}}
    instance = transform_trainer(tmp_path, prompt_limit=8, response_limit=4, component_limits=limits)
    source = trajectory(4, 2)
    batch, _ = instance._transform_agent_trajectories([source])
    assert_preserved(batch, 0, source, 8, 4)


def test_plain_agent_ppo_preserves_existing_crop_and_padding_behavior(tmp_path):
    instance = transform_trainer(tmp_path, prompt_limit=8, response_limit=4)
    source = trajectory(10, 6)
    batch, _ = instance._transform_agent_trajectories([source])
    assert torch.equal(batch.batch["prompts"][0], source["prompt_tokens"][-8:])
    assert torch.equal(batch.batch["responses"][0], source["response_tokens"][:4])
    assert torch.equal(batch.batch["response_mask"][0], source["response_masks"][:4])
    assert batch.batch["attention_mask"].sum().item() == 12
    assert batch.batch["token_level_scores"][0, 3].item() == pytest.approx(0.25)


def test_opt_in_role_labels_follow_the_transformed_action_mask(tmp_path):
    from rllm.trainer.verl.role_credit import role_credit_metrics

    instance = transform_trainer(tmp_path, prompt_limit=8, response_limit=4)
    instance.ace_config = {"log_role_credit": True}
    sources = [trajectory(4, 3, index=i) for i in range(3)]
    for source, role in zip(sources, ["policy", "reflector", "curator"], strict=True):
        source["_component_type"] = role
    batch, _ = instance._transform_agent_trajectories(sources)
    assert batch.non_tensor_batch["ace_component"].tolist() == ["policy", "reflector", "curator"]
    # Model the final rejection selection: reflector was dropped, policy and
    # curator survive. Environment tokens have large advantages but no credit.
    rows = [0, 2]
    final = SimpleNamespace(
        non_tensor_batch={"ace_component": batch.non_tensor_batch["ace_component"][rows], "uid": np.array(["p", "c"])},
        batch={"response_mask": batch.batch["response_mask"][rows], "advantages": torch.tensor([[1., 99., -1., 99.], [0., 99., 0., 99.]])},
    )
    metrics = role_credit_metrics(final)
    assert metrics["ace/role_credit/policy_action_tokens"] == 2
    assert metrics["ace/role_credit/policy_positive_advantage_tokens"] == 1
    assert metrics["ace/role_credit/policy_negative_advantage_tokens"] == 1
    assert metrics["ace/role_credit/reflector_rows"] == 0
    assert metrics["ace/role_credit/curator_nonzero_advantage_rows"] == 0


@pytest.mark.parametrize("prompt_length,response_length,kind", [(13, 4, "prompt"), (8, 10, "response")])
def test_ace_rejects_over_budget_tokens_instead_of_silently_cropping(tmp_path, prompt_length, response_length, kind):
    limits = {"policy": {"max_prompt_length": 12, "max_response_length": 9}}
    instance = transform_trainer(tmp_path, prompt_limit=8, response_limit=4, component_limits=limits)
    with pytest.raises(ValueError, match=f"ACE trajectory {kind} length .* refusing to truncate"):
        instance._transform_agent_trajectories([trajectory(prompt_length, response_length)])
