#!/usr/bin/env python3
"""Per-episode outcome + test-fraction scanner for frac/frac2 runs (Fig data).

Walks an experiment's episode logs (logs/<exp>/episodes/train_step_*_epoch_*/
episode_*.json) and emits one CSV row per episode:

    run, step, epoch, outcome, reward, ftp_frac, ptp_frac

outcome is one of the mutually exclusive classes (per-step shares sum to 1.0):
    resolved          reward >= 1.0 ("tests_passed")
    applied           graded, unresolved (ftp_frac/ptp_frac defined)
    grader_crash      0.05 tier / counts-missing marker
    patch_not_applied patch produced but failed to apply
    no_patch          everything else (no patch produced)

Population = ALL rollouts in the episode logs (PRE-filter; refetch epochs kept
as separate rows — the plot pools them per step). Prints distinct
(reward_reason, outcome) combos for classification sanity.

Usage (on a cluster node):
  .venv-l40s/bin/python examples/mini_swe_bench/eval/helpers/scan_frac_fractions.py \
      --exp mini_swe_bench_grpo_Qwen3-8B_max_tasksall_n_gpus2_gpu_h200_frac2_tb4_2gpu_4h \
      --out /path/to/frac_fractions.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

DIR_RE = re.compile(r"train_step_(\d+)_epoch_(\d+)$")


def classify(wf_reason: str, env_reason: str) -> str:
    """Mirror workflow.compute_trajectory_reward's taxonomy exactly.

    Workflow reward_reason is authoritative: no_valid_patch | resolved |
    tests_failed | patch_no_eval. The env-level 'patch_not_applied' case is
    folded into workflow 'tests_failed' (env reason 'tests_failed
    (patch_not_applied)'), so split it back out via env_reward_reason — its
    ftp/ptp fracs are 0/0 artifacts (tests never ran), not measurements.
    """
    if wf_reason == "resolved":
        return "resolved"
    if wf_reason == "no_valid_patch":
        return "no_patch"
    if wf_reason == "patch_no_eval":
        return "grader_crash"
    if wf_reason == "tests_failed":
        return "patch_not_applied" if "patch_not_applied" in env_reason else "applied"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--logs_base", default="examples/mini_swe_bench/logs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, combos = [], Counter()
    step_dirs = sorted(glob.glob(os.path.join(args.logs_base, args.exp, "episodes", "train_step_*_epoch_*")))
    for d in step_dirs:
        m = DIR_RE.search(d)
        if not m:
            continue
        step, epoch = int(m.group(1)), int(m.group(2))
        for f in sorted(glob.glob(os.path.join(d, "episode_*.json"))):
            try:
                ep = json.load(open(f))
            except Exception as e:  # truncated write of a live run etc.
                print(f"[skip-unreadable] {f}: {e}")
                continue
            task = ep.get("task") or {}
            inst = task.get("instance") if isinstance(task, dict) else None
            iid = (inst or task or {}).get("instance_id", "") if isinstance(task, dict) else ""
            trajs = ep.get("trajectories") or []
            if not trajs:
                # Dead rollout (no trajectory produced) — real population member,
                # counted as its own class; excluded from reward/frac means.
                combos[("<no trajectory>", "dead")] += 1
                rows.append({"run": args.exp, "step": step, "epoch": epoch, "outcome": "dead", "reward": "", "ftp_frac": "", "ptp_frac": "", "instance_id": iid})
                continue
            info = trajs[0].get("info") or {}
            reward = trajs[0].get("reward")
            ftp, ptp = info.get("ftp_frac"), info.get("ptp_frac")
            wf_reason = str(info.get("reward_reason") or "")
            env_reason = str(info.get("env_reward_reason") or "")
            outcome = classify(wf_reason, env_reason)
            combos[(f"{wf_reason}/{env_reason.split('(')[-1].rstrip(')') if 'patch_not_applied' in env_reason else env_reason.split(' ')[0]}", outcome)] += 1
            rows.append(
                {
                    "run": args.exp,
                    "step": step,
                    "epoch": epoch,
                    "outcome": outcome,
                    "reward": reward,
                    "ftp_frac": "" if ftp is None else ftp,
                    "ptp_frac": "" if ptp is None else ptp,
                    "instance_id": iid,
                }
            )

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "step", "epoch", "outcome", "reward", "ftp_frac", "ptp_frac", "instance_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows from {len(step_dirs)} step dirs -> {args.out}")
    for (reason, outcome), n in sorted(combos.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  reason={reason!r:45s} -> {outcome}")


if __name__ == "__main__":
    main()
