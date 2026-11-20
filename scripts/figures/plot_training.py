"""Plot exact scalar metrics from one experiment's metrics.jsonl.

The horizontal axis is training/global_step, not the W&B logging index.
Replayed steps remain in the CSV and are drawn as separate continuation segments.
CSV/JSON exports can be selected explicitly with --metrics-file and --step-key.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = rows.get("history", rows.get("records"))
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a list of scalar records")
        return rows
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON record") from error
    return rows


def number(value, *, field: str, source: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{source}: {field} must be a finite number")
    return result


def load_points(path: Path, metrics: list[str], step_key: str = "training/global_step") -> list[dict]:
    metrics = list(dict.fromkeys(metrics))
    points = []
    previous = {}
    segments = dict.fromkeys(metrics, 0)
    for record_index, record in enumerate(load_records(path)):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {record_index} is not an object")
        for metric in metrics:
            if record.get(metric) in (None, ""):
                continue
            if record.get(step_key) in (None, ""):
                raise ValueError(f"{path}: record {record_index} has {metric!r} but no {step_key!r}; select an explicit --step-key for older exports")
            step = number(record[step_key], field=step_key, source=path)
            if step < 0 or not step.is_integer():
                raise ValueError(f"{path}: {step_key} must be a nonnegative integer")
            value = number(record[metric], field=metric, source=path)
            if metric in previous and step <= previous[metric]:
                segments[metric] += 1
            previous[metric] = step
            points.append(
                {
                    "record": record_index,
                    "training_step": int(step),
                    "logging_index": record.get("logging/index", ""),
                    "resume_from_step": record.get("training/resume_from_step", ""),
                    "segment": segments[metric],
                    "metric": metric,
                    "value": value,
                    "source": str(path.resolve()),
                }
            )
    missing = set(metrics) - {point["metric"] for point in points}
    if missing:
        raise ValueError(f"{path}: no observations for exact metrics {', '.join(sorted(missing))}")
    return points


def render(points: list[dict], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    metrics = list(dict.fromkeys(point["metric"] for point in points))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.5 * len(metrics)), squeeze=False)
    for ax, metric in zip(axes[:, 0], metrics, strict=True):
        rows = [point for point in points if point["metric"] == metric]
        for segment in dict.fromkeys(point["segment"] for point in rows):
            values = [point for point in rows if point["segment"] == segment]
            ax.plot([point["training_step"] for point in values], [point["value"] for point in values], marker=".", linewidth=1.4, label=f"segment {segment + 1}")
        ax.set(xlabel="Training step", ylabel=metric)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        if max(point["segment"] for point in rows):
            ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, help="Explicit JSONL/JSON/CSV export; default: RUN/metrics.jsonl")
    parser.add_argument("--metric", action="append", help="Exact scalar key; repeat for panels; default: critic/score/mean")
    parser.add_argument("--step-key", default="training/global_step", help="Exact training-step field in an older explicit export")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--check", action="store_true", help="Validate input without writing outputs or importing Matplotlib")
    args = parser.parse_args(argv)
    source = args.metrics_file or args.run / "metrics.jsonl"
    try:
        points = load_points(source, args.metric or ["critic/score/mean"], args.step_key)
        if args.check:
            print(f"Validated {len(points)} observations in {source}")
        else:
            render(points, args.out, args.title or args.run.name)
            print(f"Wrote {args.out.with_suffix('.pdf')}, PNG and CSV")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
