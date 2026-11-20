#!/usr/bin/env python
"""Backfill the appworld-aligned turn/* + pass@k metrics for mini-swe training
steps that predate the metric-alignment code, from saved episode logs.

For each ``episodes/train_step_<N>_epoch_<E>`` dir under an experiment, aggregate
ALL ``episode_*.json`` (every gen-iteration of that step), reconstruct
per-trajectory step-reward sequences (``trajectories[].steps[].reward``) and the
pass@k task grouping (``task_hash``), then recompute with the SAME functions the
live trainer uses (``compute_turn_level_metrics`` /
``compute_pass_at_k_max_reward_from_step_rewards``) so the result is consistent
with new runs.

Live params (from wandb config of the aligned code): max_steps=30,
cutoff_mode="max", gamma=1.0. These define "the new table", so we apply them to
every recovered step regardless of the old run's config.

Recovers: turn/final_avg_reward, turn/final_max_reward,
turn/trajectory_max_reward_mean, turn/trajectory_max_reward_max,
turn/final_cumulative_reward, turn/best_avg_reward, train/pass_at_k_max_reward.
Validation column ``val_mean_is_correct`` (mean of the independently-logged
``is_correct``) cross-checks turn/trajectory_max_reward_mean for binary reward.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

REPO = os.environ.get("RLLM_DIR", str(Path(__file__).resolve().parents[4]))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from rllm.utils.compute_pass_at_k import (  # noqa: E402
    compute_pass_at_k_max_reward_from_step_rewards,
    compute_turn_level_metrics,
)

# Live-code turn-metric params (rllm.workflow.workflow_args.max_steps /
# rllm.turn_level_cutoff_mode / algorithm.gamma) — verified from wandb config.
MAX_STEPS = 30
CUTOFF_MODE = "max"
GAMMA = 1.0


def load_step_dir(d: str):
    """Aggregate every episode in one train_step_*_epoch_* dir into
    per-trajectory step-reward sequences + task ids + is_correct flags."""
    seqs, tasks, correct = [], [], []
    ep_files = glob.glob(os.path.join(d, "episode_*.json"))
    for f in ep_files:
        try:
            ep = json.load(open(f))
        except Exception:
            continue
        task_hash = ep.get("task_hash")
        is_correct = ep.get("is_correct")
        for tr in ep.get("trajectories", []) or []:
            steps = tr.get("steps", []) or []
            sr = [float(s.get("reward") or 0.0) for s in steps]
            # Replicate workflow.py:255-262 EXACTLY: the terminal graded reward
            # lives on trajectory.reward, NOT on steps[-1].reward, so _step_rewards
            # overwrites the last element with the trajectory reward. Reading
            # steps[].reward alone misses the graded reward for solved episodes.
            if sr and tr.get("reward") is not None:
                sr[-1] = float(tr["reward"])
            elif not sr and tr.get("reward") is not None:
                sr = [float(tr["reward"])]
            if sr:
                seqs.append(sr)
                tasks.append(task_hash)
                if is_correct is not None:
                    correct.append(bool(is_correct))
    return seqs, tasks, correct, len(ep_files)


def recover(exp_dir: str) -> list[dict]:
    step_dirs = sorted(glob.glob(os.path.join(exp_dir, "episodes", "train_step_*")))
    rows = []
    for d in step_dirs:
        bs = os.path.join(d, "batch_summary.json")
        meta = json.load(open(bs)) if os.path.exists(bs) else {}
        seqs, tasks, correct, n_files = load_step_dir(d)
        if not seqs:
            continue
        turn = compute_turn_level_metrics(seqs, gamma=GAMMA, cutoff_mode=CUTOFF_MODE, max_steps=MAX_STEPS, prefix="turn")
        pak = compute_pass_at_k_max_reward_from_step_rewards(seqs, tasks=tasks if any(t is not None for t in tasks) else None)
        rows.append(
            {
                "training_step": meta.get("training_step"),
                "epoch": meta.get("epoch"),
                "n_traj": len(seqs),
                "n_tasks": len(set(tasks)),
                "n_files": n_files,
                "turn/final_avg_reward": turn.get("turn/final_avg_reward"),
                "turn/final_max_reward": turn.get("turn/final_max_reward"),
                "turn/trajectory_max_reward_mean": turn.get("turn/trajectory_max_reward_mean"),
                "turn/trajectory_max_reward_max": turn.get("turn/trajectory_max_reward_max"),
                "turn/final_cumulative_reward": turn.get("turn/final_cumulative_reward"),
                "turn/best_avg_reward": turn.get("turn/best_avg_reward"),
                "train/pass_at_k_max_reward": pak,
                "val_mean_is_correct": (sum(correct) / len(correct)) if correct else None,
            }
        )
    return rows


def validate(rows: list[dict]) -> None:
    if not rows:
        print("[validate] no rows", file=sys.stderr)
        return
    r = rows[0]
    print(
        f"[validate] step={r['training_step']} epoch={r['epoch']} n_traj={r['n_traj']} "
        f"n_tasks={r['n_tasks']} final_avg={r['turn/final_avg_reward']:.4f} "
        f"final_max={r['turn/final_max_reward']:.4f} traj_max_mean={r['turn/trajectory_max_reward_mean']:.4f} "
        f"pass@k={r['train/pass_at_k_max_reward']:.4f} mean_is_correct={r['val_mean_is_correct']}"
    )
    # Identity proven earlier: final_max_reward == trajectory_max_reward_mean.
    assert abs(r["turn/final_max_reward"] - r["turn/trajectory_max_reward_mean"]) < 1e-9, "final_max != trajectory_max_reward_mean"
    print("[validate] identity final_max == trajectory_max_reward_mean  OK")
    # Independent cross-check: for (near-)binary reward, trajectory_max_reward_mean
    # == mean(is_correct). Warn (don't fail) if partial rewards make them differ.
    if r["val_mean_is_correct"] is not None:
        gap = abs(r["turn/trajectory_max_reward_mean"] - r["val_mean_is_correct"])
        tag = "OK" if gap < 1e-6 else f"gap={gap:.4f} (partial rewards?)"
        print(f"[validate] traj_max_mean vs mean(is_correct): {tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="…/examples/mini_swe_bench/logs/<experiment>")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    rows = recover(args.exp_dir)
    validate(rows)

    fields = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
