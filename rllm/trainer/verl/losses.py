"""Narrow loss guards for synthetic, fully masked worker padding."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def mask_safe_value_loss(*, loss_fn: Callable, model_output: dict, data: Any, dp_group=None):
    """Keep empty critic microbatches finite without skipping model collectives.

    The pinned critic normalizes its token-mean loss by the local action count.
    Synthetic padding can produce a microbatch with no actions. Its forward has
    already run; a graph-connected zero still executes backward on every rank.
    Nonempty microbatches delegate to the original loss without reweighting.
    """
    mask = data["response_mask"]
    mask_values = mask.values() if mask.is_nested else mask
    if mask_values.any():
        return loss_fn(model_output=model_output, data=data, dp_group=dp_group)
    values = model_output["values"]
    values = values.values() if values.is_nested else values
    loss = values.sum() * 0.0
    return loss, {"critic/vf_loss": 0.0, "critic/vf_clipfrac": 0.0, "critic/vpred_mean": 0.0}


def sequence_normalized_actor_loss(*, loss_fn: Callable, model_output: dict, data: Any, dp_group=None):
    """Use real-sequence denominators on opt-in, masked actor transport rows."""
    from rllm.trainer.verl.padding import SEQUENCE_COUNT_KEY

    if SEQUENCE_COUNT_KEY not in data:
        return loss_fn(model_output=model_output, data=data, dp_group=dp_group)
    from verl.utils import tensordict_utils as tu

    counts = data[SEQUENCE_COUNT_KEY]
    count = int(counts.reshape(-1)[0])
    if count <= 0 or not (counts == count).all():
        raise ValueError("Actor microbatch carries inconsistent effective sequence counts")
    normalized_data = data.clone(recurse=False)
    tu.assign_non_tensor(normalized_data, global_batch_size=count)
    return loss_fn(model_output=model_output, data=normalized_data, dp_group=dp_group)
