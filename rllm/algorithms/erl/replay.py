"""Lossless, prompt-only reconstruction of ERL internalization records.

Generation accounting belongs to ERL collection. These functions never generate
text, count replay as generation, or retokenize previously generated actions.
"""

from __future__ import annotations

import copy
import math

import torch


def validate_replay_trajectory(trajectory: dict, *, allow_empty: bool = False) -> bool:
    """Validate before dtype conversion; return False only for allowed empties."""
    prompt = torch.as_tensor(trajectory["prompt_tokens"])
    response = torch.as_tensor(trajectory["response_tokens"])
    mask = torch.as_tensor(trajectory["response_masks"])
    if prompt.ndim != 1 or response.ndim != 1 or mask.shape != response.shape:
        raise ValueError("ERL replay requires aligned one-dimensional tokens and masks")
    for tokens in (prompt, response):
        if not torch.isfinite(tokens).all() or (tokens < 0).any():
            raise ValueError("ERL token IDs must be finite nonnegative integers")
        if tokens.is_floating_point() and not (tokens == tokens.floor()).all():
            raise ValueError("ERL token IDs must be finite nonnegative integers")
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("ERL response masks must be finite and binary")
    if not math.isfinite(float(trajectory["trajectory_reward"])):
        raise ValueError("ERL requires finite verifier rewards")
    if not len(prompt) or not len(response):
        if allow_empty:
            return False
        raise ValueError("ERL replay cannot contain empty prompts or responses")
    return True


def verified_replay_reward(trajectory: dict) -> float:
    """Released RAFT uses positive raw reward, including partial credit."""
    reward = float(trajectory["trajectory_reward"])
    if not math.isfinite(reward):
        raise ValueError("ERL requires finite verifier rewards")
    return max(reward, 0.0)


def make_internalization_copy(trajectory: dict, plain_prompt_tokens) -> dict:
    """Replace only the initial conditioning; preserve action/observation IDs.

    The response suffix already contains the exact generated actions and masked
    tool observations, including raw Qwen thinking tokens. Re-rendering it from
    assistant text would change training targets and is deliberately avoided.
    """
    validate_replay_trajectory(trajectory)
    result = copy.deepcopy(trajectory)
    result["prompt_tokens"] = torch.as_tensor(plain_prompt_tokens).clone()
    validate_replay_trajectory(result)
    result["prompt_tokens"] = result["prompt_tokens"].long()
    # Preserve raw reward; the RAFT estimator clamps it, matching upstream.
    # This is intentionally unrelated to the controller's retry-success gate.
    result["trajectory_reward"] = float(trajectory["trajectory_reward"])
    # A contextual rollout denominator cannot be used after changing its input.
    for name in ("logprobs", "rollout_log_probs", "rollout_log_prob_mask", "old_log_probs", "ref_log_prob"):
        result.pop(name, None)
    return result


def pack_replay_tensors(trajectories: list[dict], *, max_prompt_length: int, max_response_length: int, pad_token_id: int, include_rollout_log_probs: bool = False) -> dict[str, torch.Tensor]:
    """Match AgentPPOTrainer's padding, caps, masks, and terminal-score placement.

    Kept independent of the logging transform: replay must neither overwrite the
    collected chat log nor change policy-only turn metrics or generated totals.
    The shared padded widths cover every role. Reject oversized records instead
    of changing a prompt that conditioned sampled actions or their token IDs.
    With rollout correction, rows without exact sampler log-probabilities
    (updater rows, internalized copies) carry an all-zero availability mask.
    """
    if not trajectories or min(max_prompt_length, max_response_length) <= 0:
        raise ValueError("ERL replay packing requires trajectories and positive token limits")
    n = len(trajectories)
    prompts = torch.full((n, max_prompt_length), pad_token_id, dtype=torch.long)
    responses = torch.full((n, max_response_length), pad_token_id, dtype=torch.long)
    attention = torch.zeros((n, max_prompt_length + max_response_length), dtype=torch.long)
    masks = torch.zeros_like(responses)
    scores = torch.zeros_like(responses, dtype=torch.float32)
    rollout_log_probs = torch.zeros((n, max_response_length), dtype=torch.float32)
    rollout_log_prob_mask = torch.zeros((n, max_response_length), dtype=torch.long)
    for i, trajectory in enumerate(trajectories):
        validate_replay_trajectory(trajectory)
        prompt = torch.as_tensor(trajectory["prompt_tokens"], dtype=torch.long)
        response = torch.as_tensor(trajectory["response_tokens"], dtype=torch.long)
        mask = torch.as_tensor(trajectory["response_masks"], dtype=torch.long)
        p, r = len(prompt), len(response)
        if p > max_prompt_length or r > max_response_length:
            raise ValueError(f"ERL record exceeds padded token widths: prompt={p}/{max_prompt_length}, response={r}/{max_response_length}")
        prompts[i, -p:] = prompt[-p:]
        responses[i, :r] = response[:r]
        masks[i, :r] = mask[:r]
        attention[i, max_prompt_length - p : max_prompt_length + r] = 1
        scores[i, r - 1] = float(trajectory["trajectory_reward"])
        if include_rollout_log_probs:
            from rllm.trainer.verl.rollout_correction import trajectory_rollout_tensors

            log_probs, available = trajectory_rollout_tensors(trajectory, r)
            rollout_log_probs[i, :r] = log_probs
            rollout_log_prob_mask[i, :r] = available * mask
    packed = {
        "input_ids": torch.cat([prompts, responses], dim=1),
        "attention_mask": attention,
        "position_ids": (attention.cumsum(dim=1) - 1) * attention,
        "responses": responses,
        "prompts": prompts,
        "token_level_scores": scores,
        "response_mask": masks,
    }
    if include_rollout_log_probs:
        packed["rollout_log_probs"] = rollout_log_probs
        packed["rollout_log_prob_mask"] = rollout_log_prob_mask
    return packed
