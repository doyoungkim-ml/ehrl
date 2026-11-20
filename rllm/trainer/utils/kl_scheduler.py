"""Turn-level KL coefficient helpers for step-indexed rollout batches."""

from __future__ import annotations

import numpy as np
import torch


def compute_turn_level_kl_coef(
    step_indices: np.ndarray | torch.Tensor | list[int],
    schedule: str = "linear",
    kl_min: float = 0.0,
    kl_step: float = 0.1,
    kl_max: float = 1.0,
    max_turns: int = 5,
    exp_base: float = 2.0,
) -> np.ndarray | torch.Tensor:
    """Compute a KL coefficient for each turn index."""
    is_torch = isinstance(step_indices, torch.Tensor)

    if is_torch:
        step_indices_np = step_indices.cpu().numpy()
    else:
        step_indices_np = np.array(step_indices)

    if schedule == "linear":
        kl_coefs = kl_min + step_indices_np * kl_step
    elif schedule == "exponential":
        max_exp = exp_base**max_turns - 1
        if max_exp > 0:
            kl_coefs = kl_min + kl_step * max_turns * (np.power(exp_base, step_indices_np) - 1) / max_exp
        else:
            kl_coefs = np.full_like(step_indices_np, kl_min, dtype=np.float32)
    elif schedule == "constant":
        kl_coefs = np.full_like(step_indices_np, kl_min, dtype=np.float32)
    else:
        raise ValueError(f"Unknown KL schedule: {schedule}. Supported: linear, exponential, constant")

    kl_coefs = np.clip(kl_coefs, kl_min, kl_max).astype(np.float32)

    if is_torch:
        return torch.from_numpy(kl_coefs).to(step_indices.device)
    return kl_coefs


def compute_token_level_kl_coefs(
    step_indices: np.ndarray | list[int],
    response_lengths: np.ndarray | torch.Tensor | list[int],
    max_response_length: int,
    schedule: str = "linear",
    kl_min: float = 0.0,
    kl_step: float = 0.1,
    kl_max: float = 1.0,
    max_turns: int = 5,
) -> torch.Tensor:
    """Broadcast per-turn KL coefficients over response tokens."""
    batch_size = len(step_indices)

    step_kl_coefs = compute_turn_level_kl_coef(
        step_indices=step_indices,
        schedule=schedule,
        kl_min=kl_min,
        kl_step=kl_step,
        kl_max=kl_max,
        max_turns=max_turns,
    )

    token_kl_coefs = torch.zeros(batch_size, max_response_length, dtype=torch.float32)

    if isinstance(response_lengths, torch.Tensor):
        response_lengths = response_lengths.cpu().numpy()
    else:
        response_lengths = np.array(response_lengths)

    for i, (kl_coef, resp_len) in enumerate(zip(step_kl_coefs, response_lengths, strict=False)):
        resp_len = min(int(resp_len), max_response_length)
        if resp_len > 0:
            token_kl_coefs[i, :resp_len] = float(kl_coef)

    return token_kl_coefs


def get_kl_schedule_summary(
    schedule: str = "linear",
    kl_min: float = 0.0,
    kl_step: float = 0.1,
    kl_max: float = 1.0,
    max_turns: int = 5,
) -> str:
    """Return a short human-readable summary of the configured KL schedule."""
    step_indices = np.arange(max_turns)
    kl_coefs = compute_turn_level_kl_coef(
        step_indices=step_indices,
        schedule=schedule,
        kl_min=kl_min,
        kl_step=kl_step,
        kl_max=kl_max,
        max_turns=max_turns,
    )

    summary_lines = [f"Turn-Level KL Schedule ({schedule}):"]
    for turn, kl_coef in enumerate(kl_coefs):
        summary_lines.append(f"  Turn {turn}: kl_coef = {kl_coef:.4f}")

    return "\n".join(summary_lines)
