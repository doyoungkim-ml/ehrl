"""Real CPU transport/loss regressions for variable workflow actor row counts.

The gate supplies VERL_RESUME_SOURCE and verifies its compatibility manifest first.
AST loading excludes Ray/GPU imports only: DataProto, TensorDict dispatch/iteration,
no-padding conversion, and loss aggregation execute the prepared VeRL code. The
converter uses VeRL's vendored flash-attention padding functions on CPU. This is
not a distributed model/GPU integration test.
"""

import ast
import copy
import importlib.util
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import tensordict
import torch
from packaging.version import parse as parse_version
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "rllm/trainer/verl/agent_workflow_trainer.py"


def load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"), namespace)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("set VERL_RESUME_SOURCE to the prepared pinned VeRL checkout")
    source = Path(source) / "verl"
    tu = load_module(source / "utils/tensordict_utils.py", "workflow_test_tensordict")
    # DataProto.to_tensordict has a local import; wire it to the real module.
    utils = ModuleType("verl.utils")
    utils.tensordict_utils = tu
    monkeypatch.setitem(sys.modules, "verl.utils", utils)
    protocol = ModuleType("workflow_test_protocol")
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    namespace = protocol.__dict__
    namespace.update(np=np, torch=torch, tensordict=tensordict, TensorDict=TensorDict, copy=copy, os=os, math=math, logging=logging, dataclass=dataclass, field=field, parse_version=parse_version)
    load_nodes(source / "utils/py_functional.py", ["list_of_dict_to_dict_of_list", "union_two_dict"], namespace)
    load_nodes(
        source / "protocol.py",
        ["_DataProtoConfigMeta", "DataProtoConfig", "DataProtoItem", "DataProto", "pad_dataproto_to_divisor", "_array_equal", "_deep_equal", "union_numpy_dict", "union_tensor_dict"],
        namespace,
    )
    padding = load_module(source / "utils/npu_flash_attn_utils.py", "workflow_test_attention_padding")
    namespace.update(tu=tu, unpad_input=padding.unpad_input, index_first_axis=padding.index_first_axis)
    load_nodes(source / "workers/utils/padding.py", ["left_right_2_no_padding"], namespace)
    load_nodes(source / "utils/torch_functional.py", ["masked_sum"], namespace)
    namespace["verl_F"] = SimpleNamespace(masked_sum=namespace["masked_sum"])
    load_nodes(source / "trainer/ppo/core_algos.py", ["agg_loss"], namespace)
    trainer_tree = ast.parse(TRAINER.read_text())
    cls = next(node for node in trainer_tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentWorkflowPPOTrainer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_actor_update_batch")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(TRAINER), "exec"), namespace)
    return SimpleNamespace(**namespace)


def trainer(runtime, *, mapping=(0, 1), shuffle=False, loss_agg_mode="token-mean", query=False):
    cls = type("WorkflowActorBatch", (), {"_prepare_actor_update_batch": runtime._prepare_actor_update_batch})
    instance = cls()
    instance.actor_rollout_wg = SimpleNamespace(world_size=len(mapping), _dispatch_info={} if query else {"actor": list(mapping)}, _query_dispatch_info=lambda role: list(mapping))
    instance.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(shuffle=shuffle, loss_agg_mode=loss_agg_mode)))
    return instance


def batch(runtime, rows=70):
    ids = torch.arange(rows)
    mask = (torch.arange(4)[None, :] < (ids % 4 + 1)[:, None]).long()
    attention = torch.cat([torch.ones(rows, 3, dtype=torch.long), mask], dim=1)
    tensors = {
        "row_id": ids,
        "input_ids": torch.arange(7).repeat(rows, 1) + 10,
        "position_ids": torch.arange(7).repeat(rows, 1),
        "attention_mask": attention,
        "response_mask": mask,
        "advantages": (ids.double()[:, None] + 2).repeat(1, 4) / 71,
        "returns": (ids.double()[:, None] + 3).repeat(1, 4) / 70,
        "feature": (ids.double()[:, None] + 1) / 70 + torch.arange(4).double()[None, :] / 7,
    }
    flags = {"is_pad_step": np.zeros(rows, dtype=bool), "is_valid": np.ones(rows, dtype=bool), "is_last_step": np.ones(rows, dtype=bool)}
    return runtime.DataProto.from_dict(tensors=tensors, non_tensors=flags, meta_info={"temperature": 0.7, "multi_turn": True})


def distributed_minibatches(runtime, payload, *, epochs=1):
    td = runtime.left_right_2_no_padding(payload.to_tensordict())
    assert torch.equal(td["loss_mask"], payload.batch["response_mask"])
    assert all(row.numel() for row in td["input_ids"].unbind())
    runtime.tu.assign_non_tensor(td, global_batch_size=64, mini_batch_size=64, epochs=epochs, seed=42, dataloader_kwargs={"shuffle": False})
    shards = runtime.tu.chunk_tensordict(td, chunks=2)
    iterators = [runtime.tu.make_iterator(shard, mini_batch_size=32, epochs=epochs, seed=42, dataloader_kwargs={"shuffle": False}) for shard in shards]
    return list(zip(*iterators, strict=True))


def real_ids(shards):
    return torch.cat([shard["row_id"][shard["loss_mask"].any(dim=1)] for shard in shards]).tolist()


@pytest.mark.parametrize("old_padding", [False, True])
def test_failed_70_row_update_keeps_original_rank_minibatches_and_masks_copies(runtime, old_padding):
    original = batch(runtime)
    if old_padding:
        original.non_tensor_batch["is_pad_step"][-1] = True
    tensors_before = {key: val.clone() for key, val in original.batch.items()}
    flags_before = copy.deepcopy(original.non_tensor_batch)
    # Reproduce the exact failure before applying the adapter.
    with pytest.raises(AssertionError, match="35 % 32 != 0"):
        runtime.tu.make_iterator(runtime.tu.chunk_tensordict(original.to_tensordict(), chunks=2)[0], 32, 1)
    metrics = {}
    payload = trainer(runtime)._prepare_actor_update_batch(original, mini_batch_size=64, metrics=metrics)
    minibatches = distributed_minibatches(runtime, payload, epochs=2)
    expected = [list(range(32)) + list(range(35, 67)), list(range(32, 35)) + list(range(67, 69 if old_padding else 70))]
    assert [real_ids(shards) for shards in minibatches] == expected * 2
    assert len(payload) == 128
    assert payload.meta_info == original.meta_info
    assert metrics["batch/update_actor_pad_size"] == 58
    assert metrics["batch/update_actor_masked_pad_size"] == 58 + int(old_padding)
    for key, val in original.batch.items():
        assert torch.equal(val, tensors_before[key]), key
    for key, val in original.non_tensor_batch.items():
        assert np.array_equal(val, flags_before[key]), key
    padded = torch.from_numpy(payload.non_tensor_batch["is_pad_step"])
    for key in ("response_mask", "advantages", "returns"):
        assert not payload.batch[key][padded].any()
    assert payload.batch["attention_mask"][padded].any(dim=1).all()
    assert not payload.non_tensor_batch["is_valid"][padded.numpy()].any()
    assert not payload.non_tensor_batch["is_last_step"][padded.numpy()].any()


@pytest.mark.parametrize("masked_rank", [False, True])
def test_real_pinned_token_mean_gradients_and_sgd_steps_match_unpadded_reference(runtime, masked_rank):
    original = batch(runtime)
    original.non_tensor_batch["is_pad_step"][-1] = True
    if masked_rank:
        original.non_tensor_batch["is_pad_step"][35:] = True
    payload = trainer(runtime)._prepare_actor_update_batch(original, mini_batch_size=64)
    padded_weight = torch.nn.Parameter(torch.tensor(0.37, dtype=torch.float64))
    reference_weight = torch.nn.Parameter(padded_weight.detach().clone())
    padded_optimizer = torch.optim.SGD([padded_weight], lr=0.02)
    reference_optimizer = torch.optim.SGD([reference_weight], lr=0.02)
    for shards in distributed_minibatches(runtime, payload, epochs=2):
        ids = real_ids(shards)
        reference = original.select_idxs(np.asarray(ids)).batch
        global_tokens = sum(shard["loss_mask"].sum() for shard in shards)
        # FSDP all-reduces this token count before agg_loss, then averages gradients.
        losses = [
            runtime.agg_loss(
                (padded_weight * shard["feature"] - shard["returns"]).square() * shard["advantages"], shard["loss_mask"], "token-mean", dp_size=2, batch_num_tokens=global_tokens, global_batch_size=64
            )
            for shard in shards
        ]
        padded_loss = sum(losses) / 2
        reference_loss = runtime.agg_loss((reference_weight * reference["feature"] - reference["returns"]).square() * reference["advantages"], reference["response_mask"], "token-mean")
        assert all(torch.isfinite(loss) for loss in losses)
        torch.testing.assert_close(padded_loss, reference_loss, rtol=1e-12, atol=1e-12)
        padded_optimizer.zero_grad()
        reference_optimizer.zero_grad()
        padded_loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(padded_weight.grad, reference_weight.grad, rtol=1e-12, atol=1e-12)
        padded_optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(padded_weight, reference_weight, rtol=1e-12, atol=1e-12)


def test_uses_actual_actor_dp_mapping_instead_of_world_size(runtime):
    original = batch(runtime)
    instance = trainer(runtime, mapping=(0, 0, 1, 1), query=True)
    payload = instance._prepare_actor_update_batch(original, mini_batch_size=64)
    assert len(payload) == 128
    assert instance.actor_rollout_wg._dispatch_info == {"actor": [0, 0, 1, 1]}
    assert [len(real_ids(shards)) for shards in distributed_minibatches(runtime, payload)] == [64, 6]


def test_aligned_unpadded_batch_is_unchanged_even_with_shuffle(runtime):
    original = batch(runtime, rows=128)
    assert trainer(runtime, shuffle=True)._prepare_actor_update_batch(original, mini_batch_size=64) is original


def test_aligned_batch_masks_existing_padding_without_adding_rows(runtime):
    original = batch(runtime, rows=64)
    original.non_tensor_batch["is_pad_step"][-1] = True
    payload = trainer(runtime, shuffle=True)._prepare_actor_update_batch(original, mini_batch_size=64)
    assert len(payload) == 64
    assert payload.batch["response_mask"].sum() == original.batch["response_mask"].sum() - original.batch["response_mask"][-1].sum()


@pytest.mark.parametrize("settings,message", [({"shuffle": True}, "shuffle=True"), ({"loss_agg_mode": "seq-mean-token-mean"}, "token-mean")])
def test_rejects_padding_that_would_change_unsupported_training_semantics(runtime, settings, message):
    with pytest.raises(ValueError, match=message):
        trainer(runtime, **settings)._prepare_actor_update_batch(batch(runtime), mini_batch_size=64)


def test_rejects_fully_masked_global_minibatch(runtime):
    original = batch(runtime)
    original.non_tensor_batch["is_pad_step"][[32, 33, 34, 67, 68, 69]] = True
    with pytest.raises(ValueError, match="no loss tokens"):
        trainer(runtime)._prepare_actor_update_batch(original, mini_batch_size=64)
