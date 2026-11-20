"""Optional per-record advantage routing without changing the backend loss.

The default path delegates unchanged to VeRL. ERL's RAFT rows use nonnegative
outcome weights; they never enter a GRPO group's centering or standardization.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def compute_raft_advantages(rewards: torch.Tensor, response_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match official ERL's outcome-only RAFT estimator on action tokens."""
    if rewards.ndim != 2 or rewards.shape != response_mask.shape:
        raise ValueError("RAFT rewards and response mask must have matching [rows, tokens] shapes")
    if not torch.isfinite(rewards).all() or not torch.isfinite(response_mask).all():
        raise ValueError("RAFT rewards and response mask must be finite")
    if not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("RAFT response mask must be binary")
    with torch.no_grad():
        advantages = rewards.sum(dim=-1).clamp_min(0).unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_advantage_with_overrides(
    batch: Any,
    *,
    compute_fn: Callable,
    adv_estimator: Any,
    **kwargs: Any,
) -> Any:
    """Dispatch estimator subsets, retaining row order and original group UIDs.

    ``adv_estimator`` remains VeRL's configured default. The optional aligned
    non-tensor field with the same name contains row overrides. RAFT is evaluated
    locally so the protected VeRL dependency need not be modified or reinstalled.
    """
    overrides = batch.non_tensor_batch.get("adv_estimator")
    if overrides is None:
        return compute_fn(batch, adv_estimator=adv_estimator, **kwargs)
    if len(overrides) != len(batch):
        raise ValueError("Per-record advantage estimators must align with batch rows")

    def name(value: Any) -> str:
        value = getattr(value, "value", value)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid advantage estimator: {value!r}")
        return value

    default = name(adv_estimator)
    estimators = [default if item is None else name(item) for item in overrides]
    # Preserve the exact default call for ordinary batches with explicit tags.
    if all(item == default for item in estimators) and default != "raft":
        return compute_fn(batch, adv_estimator=adv_estimator, **kwargs)

    advantages = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
    returns = torch.zeros_like(advantages)
    for estimator in dict.fromkeys(estimators):
        indices = [index for index, choice in enumerate(estimators) if choice == estimator]
        selected = batch.select_idxs(indices)
        if estimator == "raft":
            selected_advantages, selected_returns = compute_raft_advantages(selected.batch["token_level_rewards"], selected.batch["response_mask"])
        else:
            selected = compute_fn(selected, adv_estimator=estimator, **kwargs)
            selected_advantages = selected.batch["advantages"]
            selected_returns = selected.batch["returns"]
        expected = (len(indices), advantages.shape[1])
        if selected_advantages.shape != expected or selected_returns.shape != expected:
            raise ValueError(f"Estimator {estimator!r} returned misaligned advantages or returns")
        positions = torch.as_tensor(indices, device=advantages.device)
        advantages[positions] = selected_advantages.to(advantages)
        returns[positions] = selected_returns.to(returns)
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = returns
    return batch
