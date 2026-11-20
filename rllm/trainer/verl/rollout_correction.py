"""Token-level truncated importance sampling against the sampling engine.

The launcher knob ``ROLLOUT_QUANTIZATION=fp8`` samples with an FP8 vLLM
replica of the BF16 actor and enables ``algorithm.rollout_correction``. Each
trained row then carries the sampler's per-token log-probabilities
(``rollout_log_probs``) and an availability mask (``rollout_log_prob_mask``).
A token is available only when its log-probability was produced by the
training sampling distribution *in exactly the token context used by the
trainer*. Unavailable action tokens (rows whose prompt was replaced, rows
sampled with other sampling parameters, turns whose context was retokenized)
receive weight 1, which is the uncorrected behaviour of the BF16 baseline.

VeRL's ``compute_rollout_correction_and_rejection_mask`` computes the weights
and the mismatch metrics over available tokens only. The actor consumes
``rollout_is_weights`` in ``verl/workers/utils/losses.py`` (``ppo_loss``),
which multiplies the per-token policy-gradient loss in
``verl/trainer/ppo/core_algos.py`` (``compute_policy_loss_vanilla``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

ROLLOUT_LOG_PROBS_KEY = "rollout_log_probs"
ROLLOUT_LOG_PROB_MASK_KEY = "rollout_log_prob_mask"
ROLLOUT_IS_WEIGHTS_KEY = "rollout_is_weights"
# Sampling parameters that change the sampled distribution relative to the
# trainer's log_softmax(logits / temperature), with their neutral values.
_NEUTRAL_SAMPLING = {
    "top_p": (1.0,),
    "top_k": (-1, 0),
    "min_p": (0.0,),
    "repetition_penalty": (1.0,),
    "presence_penalty": (0.0,),
    "frequency_penalty": (0.0,),
}
_UNSUPPORTED_SAMPLING = ("logit_bias", "allowed_token_ids", "bad_words", "logits_processors")


def _get(config: Any, path: str, default=None):
    node = config
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, Mapping):
            node = node.get(part)
        else:
            node = getattr(node, part, None)
    return default if node is None else node


def rollout_correction_settings(config: Any) -> dict | None:
    """Return the enabled token-level correction, or None when it is disabled.

    Fails for configurations whose sampler log-probabilities cannot be compared
    with the trainer's recomputed log-probabilities.
    """
    correction = _get(config, "algorithm.rollout_correction")
    rollout_is = _get(correction, "rollout_is") if correction is not None else None
    if rollout_is is None:
        return None
    if rollout_is != "token":
        raise ValueError(f"Only token-level rollout correction is supported by the rLLM trainers, got rollout_is={rollout_is!r}")
    if _get(correction, "rollout_rs") is not None:
        raise ValueError("Rollout rejection sampling is not supported with partially available rollout log-probabilities")
    if bool(_get(correction, "bypass_mode", False)):
        raise ValueError("Rollout correction bypass mode would replace recomputed actor log-probabilities; it is not supported")
    if bool(_get(correction, "rollout_is_batch_normalize", False)):
        raise ValueError("Batch-normalized rollout correction weights are not supported")
    threshold = _get(correction, "rollout_is_threshold", 2.0)
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Truncated importance sampling needs one numeric upper threshold, got {threshold!r}") from exc
    if not math.isfinite(threshold_value) or threshold_value <= 0:
        raise ValueError(f"rollout_is_threshold must be finite and positive, got {threshold!r}")
    rollout = _get(config, "actor_rollout_ref.rollout")
    mode = _get(rollout, "logprobs_mode", "processed_logprobs")
    if mode != "processed_logprobs":
        raise ValueError(f"Rollout correction needs vLLM processed_logprobs (temperature-scaled), got logprobs_mode={mode!r}")
    sampling = {"temperature": _get(rollout, "temperature", 1.0), "top_p": _get(rollout, "top_p", 1.0), "top_k": _get(rollout, "top_k", -1)}
    if not sampling_matches_training(sampling, float(sampling["temperature"])):
        raise ValueError(f"Training rollout sampling {sampling} renormalizes vLLM log-probabilities; rollout correction requires top_p=1 and top_k<=0")
    if bool(_get(config, "rllm.stepwise_advantage.enable", False)):
        raise ValueError("Rollout correction is implemented for cumulative trajectories; disable rllm.stepwise_advantage")
    if bool(_get(config, "rllm.distill.enable", False)):
        raise ValueError("Rollout correction is not combined with workflow distillation")
    return {"rollout_is": "token", "rollout_is_threshold": threshold_value, "quantization": _get(rollout, "quantization")}


def sampling_matches_training(sampling_params: Mapping[str, Any], temperature: float) -> bool:
    """Whether vLLM processed log-probabilities equal the trainer's distribution.

    vLLM's processed log-probabilities follow temperature scaling and top-k /
    top-p renormalization. The trainer recomputes log_softmax(logits / T) with
    the configured rollout temperature, so only full-vocabulary sampling at that
    temperature is comparable.
    """
    try:
        sampled_temperature = float(sampling_params.get("temperature", 1.0))
    except (TypeError, ValueError):
        return False
    if sampled_temperature <= 0 or not math.isclose(sampled_temperature, float(temperature), rel_tol=0.0, abs_tol=1e-9):
        return False
    for key, neutral in _NEUTRAL_SAMPLING.items():
        value = sampling_params.get(key)
        if value is None:
            continue
        try:
            if float(value) not in neutral:
                return False
        except (TypeError, ValueError):
            return False
    return not any(sampling_params.get(key) for key in _UNSUPPORTED_SAMPLING)


def usable_step_log_probs(step: Mapping[str, Any]) -> list[float] | None:
    """Exact per-completion-token sampler log-probabilities for one model call."""
    logprobs = step.get("logprobs")
    completion = step.get("completion_ids")
    if not step.get("logprobs_match_training", False) or logprobs is None or completion is None:
        return None
    logprobs = list(logprobs)
    if len(logprobs) != len(completion) or not all(isinstance(value, int | float) and math.isfinite(value) for value in logprobs):
        return None
    return [float(value) for value in logprobs]


def trajectory_rollout_tensors(trajectory: Mapping[str, Any], response_length: int):
    """Return (log_probs, mask) tensors of the trajectory's response length."""
    import torch

    log_probs = trajectory.get(ROLLOUT_LOG_PROBS_KEY)
    mask = trajectory.get(ROLLOUT_LOG_PROB_MASK_KEY)
    if log_probs is None or mask is None:
        return torch.zeros(response_length, dtype=torch.float32), torch.zeros(response_length, dtype=torch.long)
    log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.long)
    if log_probs.shape != (response_length,) or mask.shape != (response_length,):
        raise ValueError(f"Rollout log-probabilities must align with the {response_length} response tokens; got {tuple(log_probs.shape)} and {tuple(mask.shape)}")
    return log_probs, mask


def pad_rollout_rows(rows: Sequence, width: int, dtype):
    """Right-pad 1-D rows to ``width`` (truncating like the token batch)."""
    import torch

    output = torch.zeros((len(rows), width), dtype=dtype)
    for index, row in enumerate(rows):
        row = torch.as_tensor(row, dtype=dtype)[:width]
        output[index, : len(row)] = row
    return output


def apply_rollout_correction(batch, settings: dict, *, excluded_rows=None) -> tuple[Any, dict]:
    """Attach ``rollout_is_weights`` computed against recomputed actor log-probs.

    ``excluded_rows`` (boolean per row) marks transport-only rows; they get
    weight 1 and are excluded from the metrics. Available tokens are those with
    ``response_mask * rollout_log_prob_mask``. Rows without any available token
    keep weight 1 on every action token.
    """
    import torch
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    tensors = batch.batch
    for key in ("old_log_probs", "response_mask", ROLLOUT_LOG_PROBS_KEY, ROLLOUT_LOG_PROB_MASK_KEY):
        if key not in tensors.keys():
            raise ValueError(f"Rollout correction is enabled but the training batch has no {key!r}")
    response_mask = tensors["response_mask"]
    old_log_probs = tensors["old_log_probs"].float()
    rollout_log_probs = tensors[ROLLOUT_LOG_PROBS_KEY].float().to(old_log_probs.device)
    available = (tensors[ROLLOUT_LOG_PROB_MASK_KEY].to(response_mask.device) > 0) & (response_mask > 0)
    if excluded_rows is not None:
        import numpy as np

        excluded = torch.as_tensor(np.asarray(excluded_rows, dtype=bool), device=available.device)
        available &= ~excluded.unsqueeze(-1)
    rows = available.any(dim=-1)
    action_tokens = int((response_mask > 0).sum())
    weights = (response_mask > 0).to(old_log_probs.dtype)
    metrics = {
        "rollout_corr/available_token_fraction": float(available.sum()) / max(action_tokens, 1),
        "rollout_corr/rows_with_rollout_log_probs": int(rows.sum()),
        "rollout_corr/rows_total": len(rows),
    }
    if rows.any():
        index = torch.nonzero(rows, as_tuple=False).squeeze(-1)
        selected_mask = available[index].to(response_mask.dtype)
        old_selected, rollout_selected = old_log_probs[index], rollout_log_probs[index]
        weights_proto, _, verl_metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob=old_selected,
            rollout_log_prob=rollout_selected,
            response_mask=selected_mask,
            rollout_is="token",
            rollout_is_threshold=settings["rollout_is_threshold"],
            rollout_is_batch_normalize=False,
            rollout_rs=None,
        )
        selected_weights = weights_proto.batch[ROLLOUT_IS_WEIGHTS_KEY].to(weights.dtype)
        selected_available = available[index]
        row_weights = weights[index]
        row_weights[selected_available] = selected_weights[selected_available]
        weights[index] = row_weights
        diff = (old_selected - rollout_selected).abs()[selected_available]
        metrics.update(verl_metrics)
        metrics["rollout_corr/log_prob_abs_diff_mean"] = float(diff.mean())
        metrics["rollout_corr/log_prob_abs_diff_max"] = float(diff.max())
        raw = torch.exp((old_selected - rollout_selected).clamp(-20.0, 20.0))[selected_available]
        metrics["rollout_corr/rollout_is_clipped_fraction"] = float((raw > settings["rollout_is_threshold"]).float().mean())
    tensors[ROLLOUT_IS_WEIGHTS_KEY] = weights.detach()
    return batch, metrics
