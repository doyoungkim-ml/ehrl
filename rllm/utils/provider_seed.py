"""Opt-in stable provider seeds for matched direct evaluation requests.

No host RNG is changed. Retries reuse the logical call identity and seed.
Provider seeds do not promise deterministic GPU execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar

SCHEME = "sha256_v1"
TASK_SEED_KEY = "_evaluation_provider_seed"
_identity: ContextVar[dict | None] = ContextVar("evaluation_provider_seed", default=None)
_requests: ContextVar[list | None] = ContextVar("evaluation_provider_requests", default=None)


def add_provider_seed_arguments(parser) -> None:
    parser.add_argument("--provider_seed_scheme", choices=("legacy", SCHEME), default="legacy", help="Stable per-task/attempt/role/turn request seeds; legacy preserves historical requests.")
    parser.add_argument("--provider_seed_namespace", default="", help="Registered namespace shared by compared arms; required with sha256_v1.")


def task_provider_seed(task) -> dict | None:
    return task.get(TASK_SEED_KEY) if isinstance(task, Mapping) else None


def provider_seed_config(args) -> dict | None:
    scheme = getattr(args, "provider_seed_scheme", "legacy")
    namespace = getattr(args, "provider_seed_namespace", "")
    if scheme == "legacy":
        if namespace:
            raise ValueError("provider_seed_namespace requires provider_seed_scheme=sha256_v1")
        return None
    if scheme != SCHEME or not isinstance(namespace, str) or not namespace.strip() or namespace != namespace.strip():
        raise ValueError("sha256_v1 requires a nonempty, trimmed provider_seed_namespace")
    seed = args.seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Provider evaluation seed must be a nonnegative integer")
    return {"scheme": SCHEME, "namespace": namespace, "evaluation_seed": seed}


def provider_seed_identity(config: dict | None, task_id: str, attempt: int, role: str = "actor", sample: int = 0, call: int = 0) -> dict | None:
    if config is None:
        return None
    if role not in ("actor", "reflector", "curator") or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (attempt, sample, call)):
        raise ValueError("Invalid provider request identity")
    return {**config, "task_id": str(task_id), "attempt": attempt, "role": role, "sample": sample, "call": call}


def derive_provider_seed(identity: dict) -> int:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def validate_seed_schedule(identities) -> int:
    seen = {}
    for identity in identities:
        seed = derive_provider_seed(identity)
        if seed in seen and seen[seed] != identity:
            raise ValueError("Provider seed collision; register a different namespace")
        seen[seed] = identity
    return len(seen)


def seed_schedule(config: dict, task_ids, attempts: int, max_turns: int, *, auxiliary: bool = False, group_size: int = 1):
    for task_id in task_ids:
        for attempt in range(attempts):
            for call in range(max_turns):
                yield provider_seed_identity(config, task_id, attempt, call=call)
        if auxiliary:
            for visit in range(attempts // group_size - 1):
                for sample in range(group_size):
                    yield provider_seed_identity(config, task_id, visit, role="reflector", sample=sample)
                yield provider_seed_identity(config, task_id, visit, role="curator")


@contextmanager
def provider_seed_scope(identity: dict | None, *, call: int | None = None):
    value = {**identity, "call": call} if identity is not None and call is not None else identity
    token = _identity.set(value)
    try:
        yield
    finally:
        _identity.reset(token)


async def with_provider_seed(awaitable, identity: dict | None):
    with provider_seed_scope(identity):
        return await awaitable


@contextmanager
def record_provider_seeds():
    records = []
    token = _requests.set(records)
    try:
        yield records
    finally:
        _requests.reset(token)


def provider_seed_receipt(config: dict | None) -> dict:
    records = list(_requests.get() or [])
    return {"provider_seed": config, "provider_requests": records, "provider_usage": provider_usage_totals(records)} if config is not None else {}


def apply_provider_seed(sampling_params: dict) -> None:
    identity = _identity.get()
    if identity is not None:
        seed = derive_provider_seed(identity)
        sampling_params["seed"] = seed
        if "seed" in (sampling_params.get("extra_body") or {}):
            sampling_params["extra_body"] = {**sampling_params["extra_body"], "seed": seed}


def record_provider_request(*, endpoint: str, model: str, sampling_params: dict, payload: dict) -> dict | None:
    identity, records = _identity.get(), _requests.get()
    if identity is not None and records is not None:
        body = {key: value for key, value in payload.items() if key not in {"timeout", "extra_headers", "extra_query", "extra_body"}}
        body.update(payload.get("extra_body") or {})
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        record = {"identity": dict(identity), "seed": sampling_params["seed"], "endpoint": endpoint, "model": model, "request_sha256": digest, "response_received": False, "usage": None}
        records.append(record)
        return record
    return None


def record_provider_response(record: dict | None, response) -> None:
    """Capture provider usage immediately, before response parsing can fail/retry."""
    if record is None:
        return
    record["response_received"] = True
    usage = getattr(response, "usage", None)
    values = {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values.values()):
        record["usage"] = values


def provider_usage_totals(records: list[dict]) -> dict:
    """Sum observed usage; total cost stays unknown if any dispatch lacks usage.

    SDK-internal HTTP retries have no separately returned usage. A dispatch with
    no returned usage is explicitly unknown, never a zero-token generation.
    """
    def summarize(items):
        known = [item["usage"] for item in items if isinstance(item.get("usage"), dict)]
        observed = {key: sum(usage[key] for usage in known) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        complete = len(known) == len(items)
        return {
            "request_dispatches": len(items),
            "responses_received": sum(item.get("response_received") is True for item in items),
            "usage_known_requests": len(known),
            "usage_unknown_requests": len(items) - len(known),
            **{f"observed_{key}": value for key, value in observed.items()},
            **{key: value if complete else None for key, value in observed.items()},
        }

    return {**{role: summarize([item for item in records if item["identity"]["role"] == role]) for role in ("actor", "reflector", "curator")}, "total": summarize(records)}
