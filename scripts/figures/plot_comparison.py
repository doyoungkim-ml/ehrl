#!/usr/bin/env python3
"""Plot an explicit benchmark comparison from local metrics and training caches.

Use --config campaign.json --out comparison.pdf. All configured points are
required. No W&B access, historical run discovery, metric fallback, or implicit
base-model point sharing is performed. See docs/visualization.md for the schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    from .plot_eval import evaluation_entries
    from .remote import source_directory_name
except ImportError:
    from plot_eval import evaluation_entries
    from remote import source_directory_name

BENCHMARKS = ("appworld", "mbpp", "mini_swe_bench", "mastermind")

STYLES = {
    "grpo_vanilla": ("#20b9ca", "-", "GRPO + vanilla"),
    "grpo_ace": ("#ed971e", "--", "GRPO + ACE"),
    "ace_rl": ("#d738ac", "-", "ACE-RL + ACE"),
}
DIRECTORY = re.compile(r"^.+_step(?P<step>\d+)_tasks(?P<tasks>.+?)(?:_final_policy)?$")
FIELDS = [
    "kind",
    "series",
    "label",
    "run",
    "step",
    "metric",
    "value",
    "split",
    "split_evidence",
    "task_label",
    "task_count",
    "task_ids_sha256",
    "local_eval_epochs",
    "model",
    "source",
    "source_sha256",
    "provenance",
]


def required_text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a nonempty string")
    return value


def step_number(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}: step must be a nonnegative integer")
    return value


def finite_number(value, context):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{context}: expected a finite number")
    return float(value)


def read_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def input_path(base, value):
    path = Path(required_text(value, "input file"))
    return (base / path).resolve()


def option(command, name):
    values = [command[i + 1] for i, token in enumerate(command[:-1]) if token == name]
    values += [token.split("=", 1)[1] for token in command if token.startswith(name + "=")]
    if len(values) > 1:
        raise ValueError(f"Completion record repeats {name}")
    return values[0] if values else None


def task_ids(entry, path):
    epochs = entry.get("local_epochs", [])
    if not isinstance(epochs, list) or any(not isinstance(epoch, dict) for epoch in epochs):
        raise ValueError(f"{path}: local_epochs must be a list of objects")
    identifiers = set()
    # Solved tasks disappear from later epochs; include the initial epoch's IDs.
    for record in [entry, *epochs]:
        tasks = record.get("tasks", [])
        if not isinstance(tasks, list):
            raise ValueError(f"{path}: tasks must be a list")
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("task_id"), str) or not task["task_id"]:
                raise ValueError(f"{path}: task identity is missing")
            identifiers.add(task["task_id"])
    if "evaluation_task_ids" in entry:
        declared = entry["evaluation_task_ids"]
        if not isinstance(declared, list) or not declared or any(not isinstance(identifier, str) or not identifier.strip() for identifier in declared) or len(declared) != len(set(declared)):
            raise ValueError(f"{path}: evaluation_task_ids must be nonempty valid unique IDs")
        expected = set(declared)
        if not identifiers <= expected:
            raise ValueError(f"{path}: observed adaptation task IDs are outside evaluation_task_ids")
        identifiers = expected
    if not identifiers:
        raise ValueError(f"{path}: no task identities; compatibility cannot be checked")
    return tuple(sorted(identifiers))


def load_evaluation(point, config, base, metric, local_epoch=None):
    if not isinstance(point, dict):
        raise ValueError("Every evaluation point must be an object")
    step = step_number(point.get("step"), "evaluation point")
    path = input_path(base, point.get("file"))
    provenance = required_text(point.get("provenance"), f"{path}: provenance")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected an object")
    recorded_path = path.parent / "evaluation_config.json"
    recorded = read_object(recorded_path) if recorded_path.is_file() else {}
    for record in (payload, recorded):
        benchmark = record.get("benchmark")
        if record is payload and "evaluations" not in payload:
            benchmark = benchmark or payload.get("dataset")
        if benchmark and config.get("benchmark") and benchmark != config["benchmark"]:
            raise ValueError(f"{path}: benchmark disagrees with configuration")
    if "step" in recorded and recorded["step"] != step:
        raise ValueError(f"{recorded_path}: step disagrees with configuration")
    if "task_label" in recorded and recorded["task_label"] != config["task_label"]:
        raise ValueError(f"{recorded_path}: task label disagrees with configuration")
    if path.name != f"metrics_step{step}.json":
        raise ValueError(f"{path}: filename disagrees with declared step {step}")
    match = DIRECTORY.match(source_directory_name(path.parent))
    if match and (int(match["step"]) != step or match["tasks"] != config["task_label"]):
        raise ValueError(f"{path}: directory step or task label disagrees with configuration")
    flat = "evaluations" not in payload
    # Mastermind vanilla records exactly one visit in a flat schema. The shared
    # adapter exposes its recorded maximum-aggregated reward as max_reward_mean.
    entries = evaluation_entries(payload, step, path) if flat else payload.get("evaluations")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError(f"{path}: expected exactly one evaluation, not an ambiguous snapshot collection")
    entry = entries[0]
    if step_number(entry.get("step"), str(path)) != step:
        raise ValueError(f"{path}: evaluation step disagrees with declared step {step}")
    selected = entry
    if local_epoch is not None:
        matches = [epoch for epoch in entry.get("local_epochs", []) if epoch.get("local_eval_epoch") == local_epoch]
        if len(matches) != 1:
            raise ValueError(f"{path}: measured local epoch {local_epoch} unavailable; evaluate it first")
        if metric.startswith("final_policy/"):
            if entry.get("local_eval_epochs") != local_epoch:
                raise ValueError(f"{path}: final policy pass is not after requested local epoch {local_epoch}")
        else:
            selected = matches[0]
    metrics = selected.get("metrics", {})
    if not isinstance(metrics, dict) or metric not in metrics:
        raise ValueError(f"{path}: missing exact metric {metric!r}")
    value = finite_number(metrics[metric], f"{path}: {metric}")
    model_key = "model" if flat else "model_name"
    model = required_text(payload.get(model_key), f"{path}: {model_key}")
    if "model" in point and point["model"] != model:
        raise ValueError(f"{path}: model disagrees with configured model")
    ids = task_ids(entry, path)
    if metric.startswith("final_policy/"):
        final = entry.get("final_policy_evaluation")
        sequential = entry.get("local_eval_protocol") == "sequential_adaptation_v1"
        final_kind = "last_measured_policy_visit" if sequential else "final_policy_only"
        if not isinstance(final, dict) or final.get("kind") != final_kind or final.get("adaptation_applied") is not False:
            raise ValueError(f"{path}: missing completed final policy-only pass")
        count = entry.get("local_eval_epochs")
        if not isinstance(count, int) or isinstance(count, bool) or final.get("after_local_eval_epoch") != count - int(sequential) or final.get("policy_visit") != count + int(not sequential):
            raise ValueError(f"{path}: inconsistent final policy visit")
        final_rows = final.get("tasks")
        if not isinstance(final_rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or not row["task_id"] for row in final_rows):
            raise ValueError(f"{path}: missing final policy task identities")
        final_ids = tuple(sorted(row["task_id"] for row in final_rows))
        if final_ids != ids:
            raise ValueError(f"{path}: final policy task coverage differs from the evaluation task set")
        if final.get("metrics", {}).get(metric.removeprefix("final_policy/")) != value:
            raise ValueError(f"{path}: final policy metric disagrees with its recorded pass")
        work = final.get("work", {})
        if not work.get("policy_rollouts") or any(work.get(key) != 0 for key in ("reflector_rollouts", "curator_rollouts", "candidate_rollouts")):
            raise ValueError(f"{path}: final policy pass contains adaptation work or no policy rollouts")
    evidence = "declared in config"
    for record in (recorded, payload, entry):
        for key in ("test_split", "split"):
            if key in record:
                if record[key] != config["split"]:
                    raise ValueError(f"{path}: split disagrees with configuration")
                evidence = "evaluation config" if record is recorded else "metrics"
    recorded_command = recorded.get("evaluation", [])
    if not isinstance(recorded_command, list) or not all(isinstance(token, str) for token in recorded_command):
        raise ValueError(f"{recorded_path}: invalid evaluation command")
    for flag in ("--test_split", "--split"):
        split = option(recorded_command, flag)
        if split is not None:
            if split != config["split"]:
                raise ValueError(f"{recorded_path}: split disagrees with configuration")
            if evidence == "declared in config":
                evidence = "evaluation config"
    marker = path.parent / "evaluation_complete.json"
    if config.get("require_completion", False) and not marker.is_file():
        raise ValueError(f"{path}: missing evaluation_complete.json")
    if marker.is_file():
        saved = read_object(marker)
        if saved.get("metrics_sha256") != digest:
            raise ValueError(f"{path}: completion hash does not match metrics")
        command = saved.get("evaluation")
        if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
            raise ValueError(f"{marker}: invalid evaluation command")
        if metric.startswith("final_policy/"):
            if sequential:
                if option(command, "--local_eval_protocol") != "sequential_adaptation_v1":
                    raise ValueError(f"{path}: completion command does not identify sequential adaptation")
                if final.get("provenance", {}).get("extra_policy_rollouts") != 0:
                    raise ValueError(f"{path}: sequential final result contains extra policy rollouts")
            elif "--final_policy_eval" not in command:
                raise ValueError(f"{path}: completion command did not request a final policy pass")
            if final["work"].get("policy_rollouts") != len(ids) * int(option(command, "--g_act") or 1):
                raise ValueError(f"{path}: incomplete final policy rollout count")
        splits = [split_value for flag in ("--test_split", "--split") if (split_value := option(command, flag)) is not None]
        if flat and not splits:
            splits = [payload.get("split")]
        if not splits or any(value != config["split"] for value in splits):
            raise ValueError(f"{path}: completion split disagrees with configuration")
        evidence = "completion record"
        if option(command, "--step" if flat else "--wandb_step") != str(step) or option(command, "--model" if flat else "--model_name") != model:
            raise ValueError(f"{path}: completion step or model disagrees with metrics")
        limit = option(command, "--limit" if flat else "--max_tasks")
        offset = int(option(command, "--offset" if flat else "--task_offset") or 0)
        single = option(command, "--task_id")
        label = f"{offset}-{offset + int(limit) - 1}" if limit and offset else (limit or (f"from{offset}" if offset else "all"))
        if single:
            label = "id_" + hashlib.sha256(single.encode()).hexdigest()[:12]
        if label != config["task_label"]:
            raise ValueError(f"{path}: completion task label disagrees with configuration")
        count = int(option(command, "--local_eval_epochs") or 1)
        start = int(option(command, "--start_epoch") or 0)
        epochs = entry.get("local_epochs", [])
        if entry.get("local_eval_epochs") != count + start or [epoch.get("local_eval_epoch") for epoch in epochs] != list(range(start + 1, start + count + 1)):
            raise ValueError(f"{path}: incomplete or inconsistent local evaluation epochs")
    return {
        "step": step,
        "metric": metric,
        "value": value,
        "split": config["split"],
        "split_evidence": evidence,
        "task_label": config["task_label"],
        "task_count": len(ids),
        "task_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "local_eval_epochs": local_epoch or entry.get("local_eval_epochs", ""),
        "model": model,
        "source": str(path),
        "source_sha256": digest,
        "provenance": provenance,
    }, ids


def load_comparison(config_path: Path, metric: str = "max_reward_mean", local_epoch: int | None = None, *, benchmark: str | None = None):
    config_path = config_path.resolve()
    config = read_object(config_path)
    declared = config.get("benchmark")
    if declared is not None and declared not in BENCHMARKS:
        raise ValueError(f"Unknown comparison benchmark: {declared!r}")
    if benchmark is not None:
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown comparison benchmark: {benchmark!r}")
        if declared is not None and declared != benchmark:
            raise ValueError("Comparison benchmark disagrees with selected benchmark")
        config["benchmark"] = benchmark
    for key in ("title", "split", "task_label"):
        required_text(config.get(key), key)
    if "require_completion" in config and not isinstance(config["require_completion"], bool):
        raise ValueError("require_completion must be true or false")
    if "evaluation_ylabel" in config:
        required_text(config["evaluation_ylabel"], "evaluation_ylabel")
    if "data_timestamp" in config:
        stamp = datetime.fromisoformat(required_text(config["data_timestamp"], "data_timestamp").replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("data_timestamp must include its timezone")
    series = config.get("series")
    if not isinstance(series, list) or not series or any(not isinstance(item, dict) for item in series):
        raise ValueError("series must be a nonempty list of objects")
    rows, by_id, sources = [], {}, {}
    reference_ids = None
    for item in series:
        identifier = item.get("id")
        if identifier not in STYLES or identifier in by_id:
            raise ValueError(f"Unknown or duplicate series ID: {identifier!r}")
        by_id[identifier] = item
        run = required_text(item.get("run"), f"{identifier}: run")
        label = required_text(item.get("label", STYLES[identifier][2]), f"{identifier}: label")
        points = item.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"{identifier}: points must be a nonempty list")
        series_metric = required_text(item.get("metric", metric), f"{identifier}: metric")
        seen_steps = set()
        for point in points:
            row, ids = load_evaluation(point, config, config_path.parent, series_metric, None if identifier == "grpo_vanilla" else local_epoch)
            if row["step"] in seen_steps:
                raise ValueError(f"{identifier}: duplicate checkpoint step {row['step']}")
            seen_steps.add(row["step"])
            if reference_ids is not None and ids != reference_ids:
                raise ValueError(f"{row['source']}: task identities differ between comparison points")
            reference_ids = ids
            prior = sources.get(row["source"])
            if prior and not (row["step"] == 0 and {prior, identifier} == {"grpo_ace", "ace_rl"}):
                raise ValueError("An evaluation file may be shared only by GRPO-ACE and ACE-RL at explicitly configured step zero")
            sources[row["source"]] = identifier
            rows.append({"kind": "evaluation", "series": identifier, "label": label, "run": run, **row})
    if len({row["metric"] for row in rows}) > 1 and "evaluation_ylabel" not in config:
        raise ValueError("Mixed evaluation metrics require an explicit evaluation_ylabel")
    training = config.get("training", [])
    if not isinstance(training, list):
        raise ValueError("training must be a list")
    seen_training = set()
    for item in training:
        if not isinstance(item, dict) or item.get("series") not in by_id:
            raise ValueError("Training must reference a configured evaluation series")
        identifier = item["series"]
        run = required_text(item.get("run"), "training run")
        train_metric = required_text(item.get("metric"), "training metric")
        if run != by_id[identifier]["run"] or run in seen_training:
            raise ValueError("Training run must match its evaluation series and occur only once")
        seen_training.add(run)
        if ("file" in item) == ("data" in item):
            raise ValueError("Training requires exactly one of file or data")
        source = input_path(config_path.parent, item["file"]) if "file" in item else config_path
        data = read_object(source).get(run) if "file" in item else item["data"]
        if not isinstance(data, list) or not data:
            raise ValueError(f"{source}: no training points for exact run {run!r}")
        provenance = required_text(item.get("provenance"), "training provenance")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        steps = set()
        for point in data:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Training points must be [step, value] pairs")
            step = step_number(point[0], "training point")
            value = finite_number(point[1], "training value")
            if step in steps:
                raise ValueError(f"{run}: duplicate training step {step}")
            steps.add(step)
            rows.append(
                {
                    "kind": "training",
                    "series": identifier,
                    "label": item.get("label", "GRPO" if identifier.startswith("grpo") else "ACE-RL"),
                    "run": run,
                    "step": step,
                    "metric": train_metric,
                    "value": value,
                    "source": str(source),
                    "source_sha256": digest,
                    "provenance": provenance,
                }
            )
    return config, sorted(rows, key=lambda row: (row["kind"], row["series"], row["step"]))


def render(config, rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if out.suffix.lower() != ".pdf":
        raise ValueError("--out must name a .pdf file; matching PNG and CSV are also written")
    evaluation = [row for row in rows if row["kind"] == "evaluation"]
    metrics = {row["metric"] for row in evaluation}
    metric = evaluation[0]["metric"]
    ylabel = {"max_reward_mean": "Padded evaluation reward", "success_rate": "Evaluation success rate", "avg_reward": "Evaluation average reward"}.get(metric, f"Evaluation: {metric}")
    has_training = any(row["kind"] == "training" for row in rows)
    # Distinct axis labels prevent a training reward from being called padded evaluation.
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    train_ax = ax.twinx() if has_training else None
    handles = []
    for item in config.get("training", []):
        points = [row for row in rows if row["kind"] == "training" and row["series"] == item["series"]]
        color = STYLES[item["series"]][0]
        (line,) = train_ax.plot(
            [row["step"] for row in points], [row["value"] for row in points], color=color, linewidth=0.8, alpha=0.32, label=f"{points[0]['label']} training ({points[0]['metric']})"
        )
        handles.append(line)
    for item in config["series"]:
        points = [row for row in evaluation if row["series"] == item["id"]]
        color, style, _ = STYLES[item["id"]]
        (line,) = ax.plot(
            [row["step"] for row in points],
            [row["value"] for row in points],
            color=color,
            linestyle=style,
            linewidth=1.35,
            marker="o",
            markersize=3.8,
            label=f"{points[0]['label']} evaluation" + (f" ({points[0]['metric']})" if len(metrics) > 1 else ""),
        )
        handles.append(line)
    title = config["title"]
    if config.get("data_timestamp"):
        title += f"\nInput data timestamp: {config['data_timestamp']}"
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(config.get("evaluation_ylabel", f"{ylabel} ({metric})"), fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(color="lightgray", alpha=0.55, linestyle="--", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if train_ax is not None:
        train_ax.set_ylabel("Training reward (metric shown in legend)", fontsize=9)
        train_ax.spines["top"].set_visible(False)
        # Matching limits keep train and evaluation rewards directly comparable.
        values = [row["value"] for row in rows]
        lo, hi = min(0.0, min(values)), max(0.1, max(values))
        ax.set_ylim(lo, hi + (hi - lo) * 0.07)
        train_ax.set_ylim(ax.get_ylim())
    else:
        ax.set_ylim(bottom=min(0.0, min(row["value"] for row in rows)))
    ax.margins(x=0.025)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), fontsize=8, frameon=False, ncol=1)
    ax.text(0.0, -0.15, f"Split: {config['split']}   Tasks: {config['task_label']} ({evaluation[0]['task_count']} task IDs)", transform=ax.transAxes, fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.12, right=0.88 if has_training else 0.97, top=0.87, bottom=0.27 + 0.025 * len(config.get("training", [])))
    out.parent.mkdir(parents=True, exist_ok=True)
    for destination in (out, out.with_suffix(".png")):
        fig.savefig(destination, bbox_inches="tight", dpi=200)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, help="Benchmark identity; checks configuration and recorded evaluation metadata")
    parser.add_argument("--config", required=True, type=Path, help="Explicit comparison JSON; input paths are relative to this file")
    parser.add_argument("--out", required=True, type=Path, help="Output PDF path (also writes PNG and CSV)")
    parser.add_argument("--metric", default="max_reward_mean", help="Exact final evaluation metric; no fallback")
    parser.add_argument("--local-epoch", type=int, help="Select a measured ACE epoch; vanilla keeps its single visit")
    parser.add_argument("--check", action="store_true", help="Validate all inputs without importing Matplotlib or writing output")
    args = parser.parse_args(argv)
    try:
        config, rows = load_comparison(args.config, args.metric, args.local_epoch, benchmark=args.benchmark)
        if args.check:
            print(f"Validated {sum(row['kind'] == 'evaluation' for row in rows)} evaluation points and {sum(row['kind'] == 'training' for row in rows)} training points")
        else:
            render(config, rows, args.out)
            print(f"Wrote {args.out}, {args.out.with_suffix('.png')}, and {args.out.with_suffix('.csv')}")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
