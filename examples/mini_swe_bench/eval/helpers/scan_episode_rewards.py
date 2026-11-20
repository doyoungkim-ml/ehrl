#!/usr/bin/env python
"""Flatten mini-swe episode logs into one CSV row per trajectory, for the
reward-plateau case study on the 12h_full run.

Reads every ``episodes/*_step_*_epoch_*`` dir under an experiment and emits:
dirname, mode (train/val), step, epoch, segment (UTC date of the dir's first
episode, to separate restart segments), task_hash, instance_id, repo,
is_correct, termination_reason, reward (trajectories[].reward — the graded
terminal reward; see recover_metrics_from_episodes.py for why
steps[].reward alone is wrong), n_steps, llm_time, total_time.

Analysis happens locally on the CSV; this script only extracts.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

DIR_RE = re.compile(r"(train|val)_step_(\d+)_epoch_(\d+)$")


def scan_episode(path: str) -> list[dict]:
    try:
        ep = json.load(open(path))
    except Exception:
        return [{"_error": os.path.basename(path)}]
    base = {
        "task_hash": ep.get("task_hash"),
        "instance_id": (ep.get("task") or {}).get("instance_id"),
        "repo": (ep.get("task") or {}).get("repo"),
        "is_correct": ep.get("is_correct"),
        "termination_reason": ep.get("termination_reason"),
        "start_ts": (ep.get("timing") or {}).get("start_timestamp"),
        "llm_time": (ep.get("timing") or {}).get("llm_time"),
        "total_time": (ep.get("timing") or {}).get("total_time"),
    }
    rows = []
    for tr in ep.get("trajectories", []) or []:
        rows.append({**base, "reward": tr.get("reward"), "n_steps": len(tr.get("steps") or [])})
    if not rows:  # episode with no trajectory list — keep it visible
        rows.append({**base, "reward": None, "n_steps": 0})
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
                print(f"[warn] unreadable episode {d}/{r['_error']}")
                continue
            rows.append({"dirname": os.path.basename(d), "mode": mode, "step": step, "epoch": epoch, **r})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", help="experiment dir containing episodes/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.exp_dir, "episodes", "*_step_*_epoch_*")))
    print(f"[scan] {len(dirs)} step dirs under {args.exp_dir}")
    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rows in ex.map(scan_dir, dirs):
            all_rows.extend(rows)

    # segment = UTC date of the dir's earliest episode (separates restart segments)
    first_ts = {}
    for r in all_rows:
        ts = r.get("start_ts") or ""
        d = r["dirname"]
        if ts and (d not in first_ts or ts < first_ts[d]):
            first_ts[d] = ts
    for r in all_rows:
        r["segment"] = (first_ts.get(r["dirname"]) or "")[:10]

    cols = ["dirname", "mode", "step", "epoch", "segment", "task_hash", "instance_id", "repo", "is_correct", "termination_reason", "reward", "n_steps", "start_ts", "llm_time", "total_time"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[scan] wrote {len(all_rows)} trajectory rows -> {args.out}")


if __name__ == "__main__":
    main()
