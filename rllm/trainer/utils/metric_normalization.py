from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Older verl actor workers may emit actor-update metrics at the root level.
# Keep W&B panels stable by normalizing them to the namespaced keys emitted by
# newer/current verl runs.
_ACTOR_METRIC_KEY_ALIASES = {
    "loss": "actor/pg_loss",
    "grad_norm": "actor/grad_norm",
    "lr": "actor/lr",
    "mfu": "perf/mfu/actor",
}


def normalize_actor_metric_keys(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return actor update metrics using canonical W&B namespaced keys.

    Existing canonical keys win over legacy root-level aliases. This prevents a
    mixed environment from overwriting newer/current verl values while still
    removing legacy keys such as ``loss``, ``grad_norm``, ``lr``, and ``mfu``
    from the logged actor metrics.
    """

    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        canonical_key = _ACTOR_METRIC_KEY_ALIASES.get(key, key)
        if canonical_key in normalized:
            continue
        if canonical_key in metrics and canonical_key != key:
            continue
        normalized[canonical_key] = value
    return normalized


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _infer_promised_flops_per_gpu() -> float | None:
    """Best-effort theoretical FLOP/s for the current GPU.

    The canonical source is VERL's FLOPs table. Return ``None`` when running in
    an environment where VERL/CUDA is unavailable so callers can skip the
    derived metric rather than logging a misleading value.
    """

    try:
        from verl.utils.flops_counter import get_device_flops
    except Exception:
        return None

    promised_tflops = _finite_float(get_device_flops(unit="T"))
    if promised_tflops is None or promised_tflops <= 0:
        return None
    return promised_tflops * 1e12


def add_actor_update_flop_metrics(
    metrics: Mapping[str, Any],
    *,
    n_gpus: int | float | None = None,
    promised_flops_per_gpu: float | None = None,
) -> dict[str, Any]:
    """Add actor-update FLOP totals derived from actor MFU and update time.

    ``perf/mfu/actor`` is measured over the actor PPO update batch. For local
    ACE that batch can include generator/policy, reflector, and curator
    trajectories, so the derived ``perf/flops/actor_update`` is the mixed
    actor-update total across GPUs, not rollout/inference FLOPs.

    Existing worker-emitted ``perf/flops/actor_update`` values win; this helper
    only derives the metric when it is absent.
    """

    enriched = normalize_actor_metric_keys(metrics)

    existing_flops = _finite_float(enriched.get("perf/flops/actor_update"))
    if existing_flops is not None:
        enriched["perf/mflops/actor_update"] = existing_flops / 1e6
        return enriched

    mfu = _finite_float(enriched.get("perf/mfu/actor"))
    update_seconds = _finite_float(enriched.get("timing_s/update_actor"))
    world_size = _finite_float(n_gpus)
    promised = _finite_float(promised_flops_per_gpu) if promised_flops_per_gpu is not None else _infer_promised_flops_per_gpu()

    if mfu is None or update_seconds is None or world_size is None or promised is None or mfu < 0 or update_seconds <= 0 or world_size <= 0 or promised <= 0:
        return enriched

    actor_update_flops = mfu * promised * world_size * update_seconds
    enriched["perf/flops/actor_update"] = actor_update_flops
    enriched["perf/mflops/actor_update"] = actor_update_flops / 1e6
    return enriched
