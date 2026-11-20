"""Bounded grader facts for ACE feedback, without test names or patch content."""

from __future__ import annotations

import re
from typing import Any


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000_000:
        return None
    return value


def sanitize_terminal_feedback(info: Any) -> dict:
    """Allow only aggregate MiniSWE outcome facts; never copy arbitrary strings.

    Exception messages, test identifiers, tests_status and git_diff stay out of
    the transport. Unknown reasons become a fixed label. Calling this on its
    own output is safe, so native and reconstructed trajectories share a path.
    """
    if not isinstance(info, dict):
        return {}
    result = {}
    counts = info.get("test_counts")
    if isinstance(counts, dict):
        bounded_counts = {key: count for key in ("ftp_passed", "ftp_total", "ptp_passed", "ptp_total") if (count := _count(counts.get(key))) is not None}
        if bounded_counts:
            result["test_counts"] = bounded_counts
    for key in ("submitted", "all_passed"):
        if isinstance(info.get(key), bool):
            result[key] = info[key]
    for key in ("max_steps", "step"):
        count = _count(info.get(key))
        if count is not None:
            result[key] = count
    if isinstance(info.get("git_diff"), str):
        result["patch_present"] = bool(info["git_diff"].strip())
        result["patch_bytes"] = min(len(info["git_diff"].encode("utf-8")), 1_000_000_000)
    elif isinstance(info.get("patch_present"), bool):
        result["patch_present"] = info["patch_present"]
        size = info.get("patch_bytes")
        if isinstance(size, int) and not isinstance(size, bool) and 0 <= size <= 1_000_000_000:
            result["patch_bytes"] = size

    # MiniSweACEEnv's reward mixin normalizes reward_reason to shaped tiers;
    # env_reward_reason retains the original grader verdict and error flag.
    raw_reason = info.get("env_reward_reason") or info.get("reward_reason")
    if isinstance(raw_reason, str) and raw_reason:
        # Inspect a bounded prefix and emit only known labels/counts. The raw
        # string can include grader exceptions containing private test data.
        reason = raw_reason[:512].strip()
        grading_error = reason.startswith(("make_test_spec_failed", "get_eval_report_failed", "grading_harness_unavailable"))
        result["grading_error"] = grading_error
        if grading_error:
            result["reward_reason"] = next(label for label in ("make_test_spec_failed", "get_eval_report_failed", "grading_harness_unavailable") if reason.startswith(label))
        elif "patch_not_applied" in reason:
            result["reward_reason"] = "tests_failed (patch_not_applied)"
        elif reason.startswith("tests_failed"):
            failures = []
            for prefix in ("ftp", "ptp"):
                passed = result.get("test_counts", {}).get(f"{prefix}_passed")
                total = result.get("test_counts", {}).get(f"{prefix}_total")
                match = re.search(rf"\b{prefix}_fail=(\d{{1,7}})\b", reason)
                failed = max(0, total - passed) if passed is not None and total is not None else _count(int(match.group(1))) if match else None
                if failed is not None:
                    failures.append(f"{prefix}_fail={failed}")
            result["reward_reason"] = "tests_failed" + (f" ({', '.join(failures)})" if failures else "")
        elif reason in {"tests_passed", "resolved", "no_patch", "no_valid_patch"}:
            result["reward_reason"] = {"resolved": "tests_passed", "no_valid_patch": "no_patch"}.get(reason, reason)
        else:
            result["reward_reason"] = "unknown_grading_result"
    elif isinstance(info.get("grading_error"), bool):
        result["grading_error"] = info["grading_error"]
    return result
