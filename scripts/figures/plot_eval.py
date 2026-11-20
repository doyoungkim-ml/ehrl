#!/usr/bin/env python3
"""Plot local evaluation artifacts without W&B or benchmark runtime imports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

try:
    from .remote import source_directory_name
except ImportError:
    from remote import source_directory_name

RUN_PATTERN = re.compile(r"^(?P<tag>.+)_step(?P<step>\d+)(?P<variant>(?:_.*?)?)_tasks(?P<tasks>[^/]+)$")


def evaluation_entries(payload: dict, step: int, source: Path) -> list[dict]:
    """Return the ACE-style evaluation entries recorded for one checkpoint step.

    The Mastermind vanilla evaluator writes one flat single-visit record whose
    per-task rewards are already maximum-aggregated; it is presented as one
    measured local epoch so the same scalar and turn views apply.
    """
    if "evaluations" in payload:
        return [entry for entry in payload["evaluations"] if entry.get("step") == step]
    tasks = payload.get("tasks")
    if payload.get("step") != step or not isinstance(tasks, list) or not tasks:
        return []
    if payload.get("reward_aggregation") != "max" or "avg_reward" not in payload:
        raise ValueError(f"{source}: flat evaluation records need maximum-aggregated avg_reward")
    metrics = {key: value for key, value in payload.items() if isinstance(value, int | float) and not isinstance(value, bool)}
    metrics["max_reward_mean"] = payload["avg_reward"]
    task_rows = [{"task_id": row.get("task_id"), "reward": row.get("reward")} for row in tasks]
    epoch = {"local_eval_epoch": 1, "tasks": task_rows, "metrics": metrics}
    return [{"step": step, "local_eval_epochs": 1, "local_epochs": [epoch], "metrics": metrics, "evaluation_task_ids": [row["task_id"] for row in task_rows]}]


def load_points(runs_dir: Path, metric: str, task_label: str = "all", run_tags: list[str] | None = None, local_epoch: int | None = None, eval_preset: str | None = None, *, recursive: bool = True) -> list[dict]:
    """Read final evaluation metrics, keeping task slices and exact tags separate.

    The file name and evaluation entry must agree on checkpoint step. Duplicate
    observations fail rather than selecting whichever reward is largest.
    """
    points = []
    seen = set()
    pattern = "**/metrics_step*.json" if recursive else "metrics_step*.json"
    for path in sorted(runs_dir.glob(pattern)):
        if any(part in ("archive", "historical") for part in path.parts):
            continue
        config_path = path.parent / "evaluation_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        if eval_preset and config.get("preset") != eval_preset:
            continue
        match = RUN_PATTERN.match(source_directory_name(path.parent))
        if match:
            tag, variant, tasks, step = match["tag"], match["variant"].lstrip("_"), match["tasks"], int(match["step"])
        else:
            config_path = path.parent / "evaluation_config.json"
            if not config_path.is_file():
                raise ValueError(f"{path}: canonical evaluation needs evaluation_config.json with step, task_label, and preset")
            config = json.loads(config_path.read_text())
            step, tasks = config.get("step"), config.get("task_label")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0 or not isinstance(tasks, str) or not tasks:
                raise ValueError(f"{config_path}: expected a nonnegative integer step and explicit task_label")
            preset = config.get("preset")
            if not isinstance(preset, str) or not preset:
                raise ValueError(f"{config_path}: expected an explicit evaluation preset")
            tag = str(config.get("training_run_dir") or config.get("run_dir") or config.get("model") or path.parent.name)
            recorded_run = Path(tag)
            # RUN_DIR/eval/<method>/<variant>/step_N names its training run.
            if recorded_run.name == f"step_{step}" and "eval" in recorded_run.parts[:-1]:
                tag = str(next(parent.parent for parent in recorded_run.parents if parent.name == "eval"))
            variant = "/".join(str(value) for value in (config.get("method"), preset) if value)
        if task_label != "auto" and tasks != task_label:
            continue
        if run_tags and tag not in run_tags:
            continue
        if path.name != f"metrics_step{step}.json":
            raise ValueError(f"{path}: metric filename disagrees with configured step {step}")
        payload = json.loads(path.read_text())
        evaluations = evaluation_entries(payload, step, path)
        if not evaluations:
            raise ValueError(f"{path}: no evaluation entry for step {step}")
        # An artifact may contain successive snapshots; its last matching entry
        # contains the completed final local epoch, as written by ACE evaluation.
        entry = evaluations[-1]
        selected = entry
        if local_epoch is not None:
            matches = [epoch for epoch in entry.get("local_epochs", []) if epoch.get("local_eval_epoch") == local_epoch]
            if len(matches) != 1:
                raise ValueError(f"{path}: requested measured local epoch {local_epoch} is unavailable; evaluate it first")
            selected = matches[0]
        value = selected.get("metrics", {}).get(metric)
        if value is None:
            raise ValueError(f"{path}: final evaluation has no metric {metric!r}")
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{path}: metric {metric!r} must be a finite number")
        key = (tag, variant, step, tasks)
        if key in seen:
            raise ValueError(f"Duplicate evaluation for {key}: {path}. Select a single runs directory.")
        seen.add(key)
        points.append(
            {
                "run_tag": tag,
                "variant": variant,
                "step": step,
                "task_label": tasks,
                "metric": metric,
                "value": float(value),
                "local_eval_epochs": entry.get("local_eval_epochs", ""),
                "local_epoch": local_epoch or entry.get("local_eval_epochs", ""),
                "source": str(path.resolve()),
            }
        )
    if not points:
        raise ValueError(f"No matching evaluation metrics in {runs_dir} (tasks={task_label}). Run evaluation first, or set --runs-dir and --task-label to match its output.")
    if task_label == "auto" and len({point["task_label"] for point in points}) != 1:
        raise ValueError("Multiple task slices found; select one explicitly with --task-label")
    missing = set(run_tags or []) - {point["run_tag"] for point in points}
    if missing:
        raise ValueError(f"No evaluation metrics for requested run tags: {', '.join(sorted(missing))}")
    return sorted(points, key=lambda point: (point["run_tag"], point["variant"], point["step"]))


def render(points: list[dict], out: Path, benchmark: str, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, variant in dict.fromkeys((point["run_tag"], point["variant"]) for point in points):
        rows = [point for point in points if (point["run_tag"], point["variant"]) == (tag, variant)]
        label = f"{tag} [{variant}]" if variant else tag
        ax.plot([row["step"] for row in rows], [row["value"] for row in rows], marker="o", markersize=4, label=label)
    ax.set(xlabel="Checkpoint step", ylabel=points[0]["metric"], title=title or f"{benchmark}: evaluation (tasks={points[0]['task_label']})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", dpi=180)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--metric", default="success_rate", help="Exact key in the final evaluation metrics")
    parser.add_argument("--task-label", default="all", help="Exact task slice label; slices are never pooled")
    parser.add_argument("--run-tag", action="append", help="Exact run tag to include; repeat for comparisons")
    parser.add_argument("--local-epoch", type=int, help="Select a measured local epoch, e.g. 3 or 8; no extrapolation")
    parser.add_argument("--eval-preset", help="Select exact evaluation preset: visits8 (ACE) or pass8 (vanilla)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--check", action="store_true", help="Validate inputs and report point count without importing plotting dependencies")
    args = parser.parse_args(argv)
    try:
        points = load_points(args.runs_dir, args.metric, args.task_label, args.run_tag, args.local_epoch, args.eval_preset)
        if args.check:
            print(f"Validated {len(points)} points in {args.runs_dir}")
        else:
            render(points, args.out, args.benchmark, args.title)
            print(f"Wrote {args.out} and {args.out.with_suffix('.csv')}")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
