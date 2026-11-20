"""Ragged mixed-record updates keep exact real-row and loss normalization semantics."""

import ast
import heapq
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_workflow_actor_batching import batch, distributed_minibatches, load_nodes, real_ids, runtime  # noqa: F401

from rllm.trainer.verl.losses import sequence_normalized_actor_loss
from rllm.trainer.verl.padding import SEQUENCE_COUNT_KEY, prepare_masked_actor_batch


def worker(mapping=(0, 1)):
    return SimpleNamespace(world_size=len(mapping), _dispatch_info={"actor": list(mapping)})


@pytest.mark.parametrize("row_count", [1, 63, 64, 65, 70, 129])
@pytest.mark.parametrize("loss_mode", ["token-mean", "seq-mean-token-sum"])
def test_ragged_actor_losses_gradients_and_updates_match_real_rows(runtime, row_count, loss_mode):  # noqa: F811
    original = batch(runtime, rows=row_count)
    before = {key: value.clone() for key, value in original.batch.items()}
    original.meta_info["mask_training_padding"] = True
    original.non_tensor_batch["adv_estimator"] = np.array(["grpo" if i % 3 else "raft" for i in range(row_count)])
    payload = prepare_masked_actor_batch(original, worker_group=worker(), mini_batch_size=64, shuffle=False)
    weight = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
    reference = torch.nn.Parameter(weight.detach().clone())
    optimizer = torch.optim.SGD([weight], lr=0.02)
    reference_optimizer = torch.optim.SGD([reference], lr=0.02)
    visits = []
    for shards in distributed_minibatches(runtime, payload):
        ids = real_ids(shards)
        visits.extend(ids)
        global_tokens = sum(shard["loss_mask"].sum() for shard in shards)
        losses = []
        for shard in shards:
            assert shard[SEQUENCE_COUNT_KEY].unique().tolist() == [len(ids)]
            runtime.tu.assign_non_tensor(shard, dp_size=2, batch_num_tokens=global_tokens)

            def real_loss(*, model_output, data, dp_group):
                return runtime.agg_loss(
                    model_output["losses"], data["response_mask"], loss_mode,
                    dp_size=data["dp_size"], batch_num_tokens=data["batch_num_tokens"], global_batch_size=data["global_batch_size"],
                ), {}

            loss, _ = sequence_normalized_actor_loss(
                loss_fn=real_loss,
                model_output={"losses": (weight * shard["feature"] - shard["returns"]).square() * shard["advantages"]},
                data=shard,
            )
            losses.append(loss)
        actual_loss = sum(losses) / 2
        selected = original.select_idxs(np.asarray(ids)).batch
        expected_loss = runtime.agg_loss((reference * selected["feature"] - selected["returns"]).square() * selected["advantages"], selected["response_mask"], loss_mode)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-12, atol=1e-12)
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        actual_loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(weight.grad, reference.grad, rtol=1e-12, atol=1e-12)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(weight, reference, rtol=1e-12, atol=1e-12)
    assert sorted(visits) == list(range(row_count))
    for key, value in original.batch.items():
        assert torch.equal(value, before[key])


def test_masked_padding_preserves_actual_dp_mapping_and_fails_closed_on_shuffle(runtime):  # noqa: F811
    original = batch(runtime, rows=65)
    payload = prepare_masked_actor_batch(original, worker_group=worker((0, 0, 1, 1)), mini_batch_size=64, shuffle=False)
    assert len(payload) == 128
    with pytest.raises(ValueError, match="shuffle=False"):
        prepare_masked_actor_batch(original, worker_group=worker(), mini_batch_size=64, shuffle=True)


@pytest.mark.parametrize("masked,rows", [(True, 1), (True, 65), (True, 128), (False, 64)])
def test_fit_defers_ragged_balancing_to_masked_actor_transport(runtime, masked, rows):  # noqa: F811
    """Execute the real fit branch and pinned VeRL rebalance, not a mock guard."""
    source = Path(os.environ["VERL_RESUME_SOURCE"]) / "verl"
    namespace = {"torch": torch, "heapq": heapq, "DataProto": runtime.DataProto}
    load_nodes(source / "utils/seqlen_balancing.py", ["calculate_workload", "karmarkar_karp", "get_seqlen_balanced_partitions", "log_seqlen_unbalance"], namespace)
    parent_path = source / "trainer/ppo/ray_trainer.py"
    parent_tree = ast.parse(parent_path.read_text())
    parent_cls = next(node for node in parent_tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    parent_balance = next(node for node in parent_cls.body if isinstance(node, ast.FunctionDef) and node.name == "_balance_batch")
    exec(compile(ast.Module(body=[parent_balance], type_ignores=[]), str(parent_path), "exec"), namespace)
    subject = SimpleNamespace(actor_rollout_wg=worker(), _get_dp_size=lambda *args: 2)
    subject._balance_batch = lambda data, **kwargs: namespace["_balance_batch"](subject, data, **kwargs)
    # The old unconditional call failed before reaching masked actor transport.
    with pytest.raises(AssertionError, match="65 % 2 != 0"):
        subject._balance_batch(batch(runtime, rows=65), metrics={})

    path = Path(__file__).resolve().parents[2] / "rllm/trainer/verl/agent_ppo_trainer.py"
    tree = ast.parse(path.read_text())
    fit = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "fit_agent")
    branch = next(node for node in ast.walk(fit) if isinstance(node, ast.If) and "mask_training_padding" in ast.unparse(node.test))
    original = batch(runtime, rows=rows)
    if masked:
        original.meta_info["mask_training_padding"] = True
    metrics = {}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec"), {"self": subject, "batch": original, "metrics": metrics})
    assert len(original) == rows
    if masked:
        assert original.batch["row_id"].tolist() == list(range(rows))
        assert metrics == {}
        payload = prepare_masked_actor_batch(original, worker_group=worker(), mini_batch_size=64, shuffle=False)
        real_rows = payload.batch["row_id"][payload.batch["response_mask"].any(dim=1)]
        assert sorted(real_rows.tolist()) == list(range(rows))
    else:
        assert metrics


def test_unmarked_actor_loss_delegates_identically():
    data = {"response_mask": torch.ones(1, 2)}
    calls = []
    sentinel = object()
    result = sequence_normalized_actor_loss(loss_fn=lambda **kw: calls.append(kw) or sentinel, model_output={}, data=data, dp_group="group")
    assert result is sentinel
    assert calls == [{"model_output": {}, "data": data, "dp_group": "group"}]


def test_actor_worker_installs_the_sequence_normalization_wrapper():
    path = Path(__file__).resolve().parents[2] / "rllm/trainer/verl/train_agent_ppo.py"
    worker_class = next(node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.ClassDef) and node.name == "MaskedPaddingActorRolloutRefWorker")
    method = next(node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name == "init_model")
    assert ast.unparse(method.decorator_list[0]) == "register(dispatch_mode=Dispatch.ONE_TO_ALL)"
    method.decorator_list = []
    worker_class.bases = [ast.Name(id="Base", ctx=ast.Load())]
    calls = []

    class Base:
        role = "actor_rollout"

        def init_model(self):
            self.loss_fn = lambda **kwargs: "original"
            self.actor = SimpleNamespace(set_loss_fn=lambda loss: calls.append(loss))

    ns = {"Base": Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[worker_class], type_ignores=[])), str(path), "exec"), ns)
    obj = ns[worker_class.name]()
    obj.init_model()
    assert len(calls) == 1
    assert calls[0](model_output={}, data={}) == "original"
