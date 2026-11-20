#!/usr/bin/env python
"""Verify mini-swe reward-tier assignment against raw episode logs.

For every trajectory in an experiment's episodes/ tree, re-derive the reward
the workflow SHOULD have assigned from the logged evidence (git_diff and the
env's reward reason, per MiniSweWorkflow.compute_trajectory_reward) and compare
with the reward actually stored. Also sub-classify the env reason so the
0.2 tier splits into graded-failure vs patch_not_applied (the apply-failure
case, which the substring routing sends to 0.2, NOT 0.05 — 0.05/patch_no_eval
only catches harness errors or missing grading).

Emits one compact JSONL row per trajectory and a summary JSON with tier
counts, a reward x reason-class crosstab, rule violations, and per-class
examples. Analysis happens on the summary; this script only extracts.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

DIR_RE = re.compile(r"(train|val)_step_(\d+)_epoch_(\d+)$")

REWARD_NO_PATCH = 0.0
REWARD_PATCH_APPLY_FAILED = 0.05
REWARD_TESTS_FAILED = 0.2
REWARD_RESOLVED = 1.0


def expected_reward(git_diff: str, env_reason: str) -> tuple[float, str]:
    """Replica of MiniSweWorkflow.compute_trajectory_reward's routing."""
    if not git_diff.strip():
        return REWARD_NO_PATCH, "no_valid_patch"
    if "tests_passed" in env_reason:
        return REWARD_RESOLVED, "resolved"
    if "tests_failed" in env_reason:
        return REWARD_TESTS_FAILED, "tests_failed"
    return REWARD_PATCH_APPLY_FAILED, "patch_no_eval"


def classify_env_reason(env_reason: str) -> str:
    """Semantic sub-class of the env's reason string (finer than the tiers)."""
    r = env_reason or ""
    if "tests_passed" in r:
        return "tests_passed"
    if "patch_not_applied" in r:
        return "patch_not_applied"
    if "tests_failed" in r:
        return "tests_failed_graded"
    for tag in (
        "grading_harness_unavailable",
        "make_test_spec_failed",
        "get_eval_report_failed",
    ):
        if tag in r:
            return f"harness_error:{tag}"
    if not r.strip():
        return "empty_no_grading"
    return "other"


def scan_episode(path: str) -> list[dict]:
    try:
        ep = json.load(open(path))
    except Exception as e:
        return [{"_error": f"{os.path.basename(path)}: {type(e).__name__}"}]
    task = ep.get("task") or {}
    rows = []
    for ti, tr in enumerate(ep.get("trajectories") or []):
        info = tr.get("info") or {}
        steps = tr.get("steps") or []
        last_info = (steps[-1].get("info") or {}) if steps else {}
        git_diff = (last_info.get("git_diff") or "").strip()
        env_reason = info.get("env_reward_reason") or ""
        reward = tr.get("reward")
        exp_r, exp_label = expected_reward(git_diff, env_reason)
        tests_status = last_info.get("tests_status") or {}
        ftp = tests_status.get("FAIL_TO_PASS") or {}
        ptp = tests_status.get("PASS_TO_PASS") or {}
        rows.append(
            {
                "instance_id": task.get("instance_id") or last_info.get("instance_id"),
                "traj_idx": ti,
                "reward": reward,
                "reward_reason": info.get("reward_reason"),
                "env_reason": env_reason[:200],
                "reason_class": classify_env_reason(env_reason),
                "expected_reward": exp_r,
                "expected_reason": exp_label,
                "match": (reward == exp_r),
                "diff_len": len(git_diff),
                "submitted": last_info.get("submitted"),
                "all_passed": last_info.get("all_passed"),
                "ftp_pass": len(ftp.get("success") or []),
                "ftp_fail": len(ftp.get("failure") or []),
                "ptp_pass": len(ptp.get("success") or []),
                "ptp_fail": len(ptp.get("failure") or []),
                "n_steps": len(steps),
                "is_correct": ep.get("is_correct"),
                "termination_reason": ep.get("termination_reason"),
            }
        )
    return rows


def scan_dir(d: str) -> list[dict]:
    m = DIR_RE.search(os.path.basename(d))
    if not m:
        return []
    mode, step, epoch = m.group(1), int(m.group(2)), int(m.group(3))
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "episode_*.json"))):
        for r in scan_episode(f):
            if "_error" in r:
                print(f"[warn] unreadable episode {d}/{r['_error']}", flush=True)
                continue
            rows.append({"mode": mode, "step": step, "epoch": epoch, **r})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", help="experiment dir containing episodes/")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.exp_dir, "episodes", "*")) if os.path.isdir(d) and DIR_RE.search(os.path.basename(d)))
    print(f"[scan] {len(dirs)} step dirs under {args.exp_dir}", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(scan_dir, dirs):
            rows.extend(chunk)
    rows.sort(key=lambda r: (r["mode"], r["step"], str(r["instance_id"]), r["traj_idx"]))
    print(f"[scan] {len(rows)} trajectories", flush=True)

    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_reward = Counter(r["reward"] for r in rows)
    by_class = Counter(r["reason_class"] for r in rows)
    crosstab: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        crosstab[str(r["reward"])][r["reason_class"]] += 1
    violations = [r for r in rows if not r["match"]]
    # Sanity cross-checks beyond the routing replica:
    #   resolved rows must have zero recorded test failures (tests_status is only
    #   populated on the graded-failure path, so empty is also fine);
    #   zero-reward rows must not carry a non-empty diff with a graded pass.
    suspicious = [r for r in rows if (r["reward"] == 1.0 and (r["ftp_fail"] or r["ptp_fail"])) or (r["reward"] == 1.0 and r["all_passed"] is False) or (r["reward"] == 0.0 and r["diff_len"] > 0)]
    examples: dict[str, list[dict]] = {}
    for r in rows:
        examples.setdefault(r["reason_class"], [])
        if len(examples[r["reason_class"]]) < 3:
            examples[r["reason_class"]].append(
                {
                    k: r[k]
                    for k in (
                        "mode",
                        "step",
                        "instance_id",
                        "reward",
                        "reward_reason",
                        "env_reason",
                        "diff_len",
                        "submitted",
                        "ftp_pass",
                        "ftp_fail",
                        "ptp_pass",
                        "ptp_fail",
                        "n_steps",
                    )
                }
            )

    summary = {
        "exp_dir": args.exp_dir,
        "n_step_dirs": len(dirs),
        "n_trajectories": len(rows),
        "reward_counts": {str(k): v for k, v in sorted(by_reward.items(), key=lambda x: str(x[0]))},
        "reason_class_counts": dict(by_class.most_common()),
        "reward_x_reason_class": {k: dict(v) for k, v in sorted(crosstab.items())},
        "n_routing_violations": len(violations),
        "routing_violations_sample": violations[:50],
        "n_suspicious": len(suspicious),
        "suspicious_sample": suspicious[:50],
        "examples_per_class": examples,
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[scan] summary -> {args.out_summary}", flush=True)
    print(json.dumps({k: summary[k] for k in ("reward_counts", "reason_class_counts", "n_routing_violations", "n_suspicious")}, indent=2))


if __name__ == "__main__":
    main()
