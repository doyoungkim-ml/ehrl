"""Loss-neutral transport for algorithms with variable-size record populations."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch

SEQUENCE_COUNT_KEY = "rllm_effective_global_batch_size"


def prepare_masked_actor_batch(batch: Any, *, worker_group: Any, mini_batch_size: int, shuffle: bool, metrics: dict | None = None) -> Any:
    """Retain each real row once while completing every data-parallel minibatch.

    Uneven populations are split into near-equal contiguous DP shards. Each shard
    is padded separately, keeping optimizer minibatch membership stable. The
    per-row sequence count survives dispatch and dynamic microbatching and lets
    the actor loss exclude transport-only rows from sequence-mean denominators.
    Token means already use the engine's global action-token count.
    """
    if shuffle:
        raise ValueError("Masked actor padding requires shuffle=False to preserve minibatch sequence counts")
    if "actor" not in worker_group._dispatch_info:
        worker_group._dispatch_info["actor"] = worker_group._query_dispatch_info("actor")
    mapping = worker_group._dispatch_info["actor"]
    dp_size = max(mapping) + 1
    if len(mapping) != worker_group.world_size or set(mapping) != set(range(dp_size)):
        raise ValueError("Invalid actor data-parallel dispatch mapping")
    if mini_batch_size <= 0 or mini_batch_size % dp_size:
        raise ValueError("Actor minibatch size must be positive and divisible by the data-parallel size")
    if not len(batch):
        raise ValueError("Masked actor padding requires at least one real record")

    local_mini_batch = mini_batch_size // dp_size
    indices = np.array_split(np.arange(len(batch)), dp_size)
    local_size = math.ceil(max(map(len, indices)) / local_mini_batch) * local_mini_batch
    shards = []
    for selected in indices:
        real_rows = len(selected)
        # Duplicate executable inputs only; all objective-bearing tensors on
        # synthetic rows are masked before conversion to the worker format.
        transport = np.concatenate([selected, np.full(local_size - real_rows, selected[0] if real_rows else 0, dtype=int)])
        shard = batch.select_idxs(transport)
        shard.meta_info = copy.deepcopy(batch.meta_info)
        pad_mask = np.arange(local_size) >= real_rows
        prior_padding = shard.non_tensor_batch.get("is_pad_step")
        if prior_padding is not None:
            pad_mask |= prior_padding
        shard.non_tensor_batch["is_pad_step"] = pad_mask
        for key in ("is_valid", "is_last_step"):
            if key in shard.non_tensor_batch:
                flags = shard.non_tensor_batch[key].copy()
                flags[pad_mask] = False
                shard.non_tensor_batch[key] = flags
        for key in ("response_mask", "advantages", "returns", "token_level_scores", "token_level_rewards", "token_level_kl_coefs", "rollout_is_weights", "rollout_log_prob_mask"):
            if key in shard.batch:
                values = shard.batch[key].clone()
                values[torch.as_tensor(pad_mask, device=values.device)] = 0
                shard.batch[key] = values
        shards.append(shard)

    counts = torch.empty(local_size, dtype=torch.int64, device=batch.batch["response_mask"].device)
    for start in range(0, local_size, local_mini_batch):
        stop = start + local_mini_batch
        count = sum(int(shard.batch["response_mask"][start:stop].any(dim=-1).sum()) for shard in shards)
        if count == 0:
            raise ValueError("Masked actor minibatch has no action tokens across data-parallel ranks")
        counts[start:stop] = count
    for shard in shards:
        shard.batch[SEQUENCE_COUNT_KEY] = counts.clone()
    payload = type(batch).concat(shards)
    payload.meta_info["global_token_num"] = payload.batch["attention_mask"].sum(dim=-1).tolist()
    if metrics is not None:
        metrics.update({
            "batch/update_actor_prepad_size": len(batch),
            "batch/update_actor_pad_size": len(payload) - len(batch),
            "batch/update_actor_pad_divisor": mini_batch_size,
            "batch/update_actor_dp_size": dp_size,
        })
    return payload
