#!/usr/bin/env python3
"""Score evaluation metrics files the way the paper reports them.

pass@K is the expected best reward of K attempts drawn without replacement from
a task's eight independent attempts (vanilla pass8, or an ACE visits8 run with
ADAPTATION_MODE=discard, whose eight visits all read the empty playbook).
RSI@K is the best reward among the first K sequential visits of an ACE visits8
run with edits applied. Each value is the mean over held-out tasks with one
standard error of the mean (SEM) over tasks.

--training-log reports training compute: cumulative actor (and critic) update
FLOPs from a run's metrics.jsonl, in EFLOPs, as on the paper's x-axes.

    python scripts/paper_metrics.py RUN/eval/ace/visits8-apply-feedback_v1/step_410/metrics_step410.json
    python scripts/paper_metrics.py --independent RUN/eval/ace/visits8-discard-feedback_v1/step_410/metrics_step410.json
    python scripts/paper_metrics.py --training-log RUN/metrics.jsonl --steps 200,410
"""

from __future__ import annotations

import argparse
import json
import math
from math import comb
from pathlib import Path

# BF16 peak of one H100/H200 times the two GPUs of every paper run. Logs that lack
# perf/flops/actor_update are rebuilt from MFU and update time with this peak.
PEAK_FLOPS = 989e12 * 2


def task_rewards(path: Path) -> tuple[str, dict[str, list[float]]]:
    """Return ("sequential" or "independent", task id -> eight rewards in order)."""
    payload = json.loads(path.read_text())
    if "evaluations" not in payload:
        # Mastermind vanilla writes one flat record of eight samples per task.
        return "independent", {row["task_id"]: [float(s["reward"]) for s in row["samples"]] for row in payload["tasks"]}
    (entry,) = payload["evaluations"]
    epochs = entry["local_epochs"]
    if len(epochs) == 1:
        (epoch,) = epochs
        return "independent", {row["task_id"]: [float(s["reward"]) for s in row["samples"]] for row in epoch["tasks"]}
    visits: dict[str, list[float]] = {}
    for epoch in epochs:
        for row in epoch["tasks"]:
            visits.setdefault(row["task_id"], []).append(float(row["samples"][0]["reward"]))
    return "sequential", visits


def expected_max(rewards: list[float], k: int) -> float:
    """Expected maximum of k rewards drawn without replacement."""
    ordered = sorted(rewards)
    return sum(value * comb(index, k - 1) for index, value in enumerate(ordered)) / comb(len(ordered), k)


def training_eflops(path: Path) -> dict[int, float]:
    """Cumulative update EFLOPs after each logged training step."""
    records: dict[int, dict] = {}
    for line in path.open():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        step = record.get("training/global_step", record.get("step"))
        if step is not None:
            records[int(step)] = record  # a resumed step keeps its last record
    total, cumulative = 0.0, {}
    for step in sorted(records):
        record = records[step]
        if record.get("perf/flops/actor_update") is not None:
            actor = record["perf/flops/actor_update"]
        elif record.get("perf/mfu/actor") is not None and record.get("timing_s/update_actor") is not None:
            actor = record["perf/mfu/actor"] * record["timing_s/update_actor"] * PEAK_FLOPS
        else:
            continue
        critic = record.get("perf/mfu/critic", 0) * record.get("timing_s/update_critic", 0) * PEAK_FLOPS
        total += actor + critic
        cumulative[step] = total / 1e18
    return cumulative


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def score(path: Path, ks: list[int], independent: bool) -> dict:
    kind, rewards = task_rewards(path)
    if independent:
        kind = "independent"
    attempts = {len(values) for values in rewards.values()}
    if attempts != {8}:
        raise ValueError(f"{path}: expected eight attempts per task, found {sorted(attempts)}")
    metric = "pass" if kind == "independent" else "RSI"
    out = {"file": str(path), "tasks": len(rewards), "metric": metric}
    for k in ks:
        per_task = [expected_max(values, k) if kind == "independent" else max(values[:k]) for values in rewards.values()]
        out[f"{metric}@{k}"] = mean_sem(per_task)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics", nargs="*", type=Path, help="metrics_step<N>.json files written by eval.slurm")
    parser.add_argument("--k", default="1,2,3,4,5,6,7,8", help="comma-separated attempt counts to report (default: 1-8)")
    parser.add_argument("--independent", action="store_true", help="score an ACE visits8 file as independent attempts (edits-discarded control)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--training-log", type=Path, help="a training run's metrics.jsonl; print cumulative update EFLOPs")
    parser.add_argument("--steps", help="comma-separated training steps for --training-log (default: every 10th)")
    args = parser.parse_args(argv)
    if args.training_log:
        cumulative = training_eflops(args.training_log)
        wanted = [int(step) for step in args.steps.split(",")] if args.steps else [step for step in cumulative if step % 10 == 0]
        rows = {step: round(cumulative[step], 4) for step in wanted if step in cumulative}
        print(json.dumps(rows, indent=2) if args.json else "\n".join(f"step {step}: {value:.2f} EFLOPs" for step, value in rows.items()))
        return 0
    if not args.metrics:
        parser.error("give metrics files or --training-log")
    ks = [int(k) for k in args.k.split(",")]
    if any(not 1 <= k <= 8 for k in ks):
        parser.error("--k values must be between 1 and 8")
    results = [score(path, ks, args.independent) for path in args.metrics]
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    for result in results:
        values = "  ".join(f"{key}={value[0]:.3f}±{value[1]:.3f}" for key, value in result.items() if "@" in key)
        print(f"{result['file']}  tasks={result['tasks']}  {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
