"""Extract model-separated gallery data from maintained experiment artifacts.

Only canonical run paths are discovered. The existing figure readers validate
recorded training and evaluation points; this module selects them and writes
the plotted values as CSV tables. ``gallery_render`` draws those tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from gallery_schema import (
    BENCHMARKS, METHODS, PLOT_METHODS, PLOTS, ROW_KEYS, STEP_PLOTS, TABLES, TOKEN_AXES, TURN_KEYS,
    read_tables,
)
from plot_eval import load_points as load_evaluations
from plot_training import load_records as load_training_records
from plot_training import load_points as load_training
from plot_training import number as training_number
from plot_turns import curve_rows
from remote_gallery import fetch_gallery_rows
from remote_turns import TURN_SPECS

ROOT = Path(__file__).resolve().parents[2]
BASE_NAMES = {
    "qwen3-4b-thinking": "Qwen3-4B",
    "qwen3-4b-nonthinking": "Qwen3-4B-nonthinking",
    "qwen3-4b-instruct-2507": "Qwen3-4B-Instruct-2507",
    "qwen3-4b-thinking-2507": "Qwen3-4B-Thinking-2507",
    "qwen3-0.6b": "Qwen3-0.6B",
}


def method_for(training_method: str, preset: str) -> str | None:
    if training_method == "base" and preset == "base":
        return "base"
    if training_method in {"grpo", "ppo"} and preset == training_method:
        return training_method
    if training_method == "erl" and re.fullmatch(r"rollout\d+", preset):
        return "erl"
    if training_method == "delayed_joint_ppo" and re.fullmatch(r"batch\d+(?:_rollout\d+)?", preset):
        return "joint_ppo"
    if training_method != "ace_rl":
        return None
    if re.fullmatch(r"oneeval_rollout\d+", preset):
        return "ace_rl"
    if re.fullmatch(r"delayed_oneeval_rollout\d+", preset):
        return "delayed_joint"
    return {"iterative_singlemode": "alternating"}.get(preset)


def source_root(host: str, override: str | None) -> str:
    if override:
        return str(Path(override).expanduser())
    if host == "local":
        return str(ROOT)
    value = os.environ.get(f"RLLM_PLOT_{host.upper()}_ROOT")
    if not value:
        raise ValueError(f"Unknown source {host!r}: pass --source-root or set RLLM_PLOT_{host.upper()}_ROOT to its checkout")
    return value


# Each method keeps one evaluation preset: ACE sequential visits, vanilla pass@8.
EVALUATION_PRESETS = {"ace": "visits8", "vanilla": "pass8"}


def base_result_dir(root: Path, benchmark: str, method: str, model: str) -> Path | None:
    parent = root / "examples" / benchmark / "eval" / method / "results" / EVALUATION_PRESETS[method]
    for name in dict.fromkeys((model, BASE_NAMES.get(model, model))):
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return None


def inventory(host: str, root: str, benchmark: str) -> list[dict]:
    """Return shallow run identities; never walk checkpoint or task artifacts."""
    if host == "local":
        parent = Path(root) / "examples" / benchmark / "train"
        rows = [
            {"path": str(path.resolve()), "training_method": path.parents[2].name,
             "preset": path.parent.name, "model": path.name,
             "training": (path / "metrics.jsonl").is_file(), "evaluation": (path / "eval").is_dir()}
            for path in parent.glob("*/runs/*/*") if path.is_dir()
        ]
        for model in {row["model"] for row in rows}:
            base = base_result_dir(Path(root), benchmark, "ace", model)
            if base is not None:
                vanilla = base_result_dir(Path(root), benchmark, "vanilla", model)
                rows.append({"path": str(base.resolve()), "training_method": "base", "preset": "base",
                             "model": model, "training": False, "evaluation": True,
                             "vanilla_path": str(vanilla.resolve()) if vanilla is not None else None})
        return rows
    code = """import json, pathlib, sys
parent = pathlib.Path(sys.argv[1]) / 'examples' / sys.argv[2] / 'train'
rows = []
for path in parent.glob('*/runs/*/*'):
    if path.is_dir():
        rows.append({'path': str(path.resolve()), 'training_method': path.parents[2].name,
                     'preset': path.parent.name, 'model': path.name,
                     'training': (path / 'metrics.jsonl').is_file(), 'evaluation': (path / 'eval').is_dir()})
base_names = {'qwen3-4b-thinking': 'Qwen3-4B', 'qwen3-4b-nonthinking': 'Qwen3-4B-nonthinking',
              'qwen3-4b-instruct-2507': 'Qwen3-4B-Instruct-2507',
              'qwen3-4b-thinking-2507': 'Qwen3-4B-Thinking-2507', 'qwen3-0.6b': 'Qwen3-0.6B'}
for model in {row['model'] for row in rows}:
    parent = pathlib.Path(sys.argv[1]) / 'examples' / sys.argv[2] / 'eval'
    base = next((p for name in dict.fromkeys((model, base_names.get(model, model)))
                 for p in [parent / 'ace/results/visits8' / name] if p.is_dir()), None)
    if base is not None:
        vanilla = next((p for name in dict.fromkeys((model, base_names.get(model, model)))
                        for p in [parent / 'vanilla/results/pass8' / name] if p.is_dir()), None)
        rows.append({'path': str(base.resolve()), 'training_method': 'base', 'preset': 'base',
                     'model': model, 'training': False, 'evaluation': True,
                     'vanilla_path': str(vanilla.resolve()) if vanilla is not None else None})
print(json.dumps(rows))
"""
    result = subprocess.run(
        ["ssh", host, shlex.join(["python3", "-c", code, root, benchmark])],
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"Cannot inspect {host}:{root}: {result.stderr.strip() or result.stdout.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Unexpected inventory from {host}: {result.stdout[:200]}") from error


def cached_inventory(host: str, root: str, benchmark: str, cache: Path, offline: bool) -> list[dict]:
    path = cache / host / f"{benchmark}-inventory.json"
    if offline:
        if not path.is_file():
            raise ValueError(f"No cached run inventory at {path}; refresh once without --offline")
        payload = json.loads(path.read_text())
        if payload.get("root") != root:
            raise ValueError(f"Cached inventory has a different source root: {path}")
        return payload["runs"]
    rows = inventory(host, root, benchmark)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"root": root, "runs": rows}, indent=2) + "\n")
    return rows


def selected_runs(rows: list[dict], model: str, methods: set[str]) -> dict[str, dict]:
    chosen = {}
    for row in rows:
        if row["model"] != model:
            continue
        method = method_for(row["training_method"], row["preset"])
        if method not in methods:
            continue
        if method in chosen:
            raise ValueError(f"Ambiguous {model}/{method}: {chosen[method]['path']} and {row['path']}")
        chosen[method] = row
    return chosen


def step_configs(parent: Path) -> list[Path]:
    """Configs of canonical step_N results; relayed or partial copies (step_N.*) are not results yet."""
    return sorted(path for path in parent.glob("step_*/evaluation_config.json")
                  if path.parent.name.removeprefix("step_").isdigit())


def evaluation_dir(run: Path, method: str, variant: str) -> Path:
    """A training run keeps each protocol in RUN_DIR/eval/<method>/<variant>/step_N."""
    return run / "eval" / method / variant


def recorded_method_matches(config: dict, method: str) -> bool:
    return config.get("method") in (None, method)


def eight_visit_evaluations(run: Path, method: str, kind: str, *, baseline: bool = False) -> tuple[list[dict], list[str]]:
    """Read measured pass@K or sequential visit-K curves, keeping their protocol explicit.

    A training run keeps each checkpoint in RUN_DIR/eval/<method>/<preset>/step_N;
    a base-model result writes directly into its result root.
    """
    directory = EVALUATION_PRESETS[method]
    parent = run if baseline else evaluation_dir(run, method, directory)
    points, errors = [], []
    # A recursive walk would descend into every task's trajectory tree.
    configs = step_configs(parent)
    if (parent / "evaluation_config.json").is_file():
        configs.insert(0, parent / "evaluation_config.json")
    for config_path in configs:
        try:
            config = json.loads(config_path.read_text())
            if config.get("preset") != directory or not config.get("task_label"):
                continue
            if not recorded_method_matches(config, method):
                continue
            if not list(config_path.parent.glob("metrics_step*.json")):
                continue
            rows = load_evaluations(config_path.parent, "max_reward_mean", "auto",
                                    eval_preset=config["preset"], recursive=False)
            if len(rows) != 1:
                raise ValueError("Expected one checkpoint evaluation")
            point = rows[0]
            payload = json.loads(Path(point["source"]).read_text())
            entry = payload.get("evaluations", [payload])[-1]
            protocol = entry.get("local_eval_protocol", payload.get("evaluation_protocol"))
            curve = []
            if kind == "pass":
                if protocol not in {"sequential_adaptation_v1", "independent_samples_v1"}:
                    raise ValueError(f"Unexpected pass@K protocol: {protocol}")
                metrics = entry.get("metrics", entry)
                for k in range(1, 9):
                    value = metrics.get(f"pass_at_{k}_success_rate")
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        curve.append({"k": k, "value": float(value)})
            else:
                if protocol != "sequential_adaptation_v1" or entry.get("g_act") != 1:
                    raise ValueError(f"Unexpected eight-visit protocol: {protocol}")
                for k in range(1, 9):
                    value = rsi_success(point, k)
                    if value is not None:
                        curve.append({"k": k, "value": value})
            if len(curve) != 8 or [row["k"] for row in curve] != list(range(1, 9)):
                raise ValueError("Expected measured K=1..8 values")
            point.update(curve=curve, protocol=protocol)
            points.append(point)
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{config_path.parent}: {error}")
    if len({point["step"] for point in points}) != len(points):
        raise ValueError(f"Duplicate {directory} evaluation steps in {run}")
    if len({point["task_label"] for point in points}) > 1:
        raise ValueError(f"Multiple {directory} task slices in {run}")
    return sorted(points, key=lambda point: point["step"]), errors


@lru_cache(maxsize=4096)
def evaluation_entry(source: str, step: int) -> dict:
    payload = json.loads(Path(source).read_text())
    entries = [entry for entry in payload.get("evaluations", [payload]) if entry.get("step") == step]
    if not entries:
        raise ValueError(f"{source}: no evaluation at step {step}")
    return entries[-1]


def rsi_success(point: dict, k: int) -> float | None:
    """Share of tasks reaching reward 1 by measured visit k."""
    rewards = padded_visit_rewards(point, k)
    return None if rewards is None else sum(value >= 1 for value in rewards.values()) / len(rewards)


def padded_visit_rewards(point: dict, k: int) -> dict[str, float] | None:
    """Fresh reward on visit k, with earlier maximum-reward tasks carried as 1."""
    entry = evaluation_entry(point["source"], point["step"])
    epochs = entry.get("local_epochs", [])
    selected = [epoch for epoch in epochs if isinstance(epoch.get("local_eval_epoch"), int)
                and not isinstance(epoch["local_eval_epoch"], bool) and epoch["local_eval_epoch"] <= k]
    if [epoch["local_eval_epoch"] for epoch in selected] != list(range(1, k + 1)):
        return None
    ids = entry.get("evaluation_task_ids") or [row.get("task_id") for row in selected[0].get("tasks", [])]
    if not ids or any(not isinstance(task_id, str) or not task_id for task_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{point['source']}: invalid evaluation task IDs")
    task_ids = set(ids)
    solved: set[str] = set()
    result = {}
    for epoch in selected:
        records = epoch.get("tasks", [])
        fresh = {}
        for row in records:
            task_id = row.get("task_id")
            reward = row.get("reward")
            if not isinstance(task_id, str) or task_id not in task_ids or task_id in fresh:
                raise ValueError(f"{point['source']}: duplicate or unexpected task in visit {epoch['local_eval_epoch']}")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                raise ValueError(f"{point['source']}: invalid reward in visit {epoch['local_eval_epoch']}")
            fresh[task_id] = float(reward)
        if task_ids - fresh.keys() - solved:
            raise ValueError(f"{point['source']}: missing unsolved tasks in visit {epoch['local_eval_epoch']}")
        solved.update(task_id for task_id, reward in fresh.items() if reward >= 1)
        result = {task_id: 1.0 if task_id in solved else fresh[task_id] for task_id in ids}
    return result


def padded_reward(point: dict, k: int) -> float | None:
    if "computed" in point:
        return point["computed"]["reward"].get(str(k))
    rewards = padded_visit_rewards(point, k)
    return None if rewards is None else sum(rewards.values()) / len(rewards)


def training_token_history(path: Path) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]], list[str]]:
    """Accumulate the latest training lineage, including its replayed steps."""
    by_record = {axis: {} for axis in TOKEN_AXES}
    by_step = {axis: {} for axis in TOKEN_AXES}
    totals = dict.fromkeys(TOKEN_AXES, 0.0)
    incomplete = dict.fromkeys(TOKEN_AXES, False)
    records = load_training_records(path)
    previous_step = None
    rollout_keys = sorted({key for row in records if isinstance(row, dict)
                           for key, value in row.items() if key.startswith("rollout/") and value not in (None, "")})
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: record {index} is not an object")
        step_value = row.get("training/global_step")
        if step_value in (None, ""):
            if any(row.get(f"rollout/{axis}") not in (None, "") for axis in TOKEN_AXES):
                raise ValueError(f"{path}: token record {index} has no training/global_step")
            continue
        step = training_number(step_value, field="training/global_step", source=path)
        if step < 0 or not step.is_integer():
            raise ValueError(f"{path}: training/global_step must be a nonnegative integer")
        step = int(step)
        # A new unresumed step 1 starts a different checkpoint lineage. Older
        # attempts can remain in the same journal, but cannot be charged to
        # checkpoints trained from this fresh base model.
        if step == 1 and previous_step is not None and previous_step > 1 and row.get("training/resume_from_step") in (None, ""):
            by_record = {axis: {} for axis in TOKEN_AXES}
            by_step = {axis: {} for axis in TOKEN_AXES}
            totals = dict.fromkeys(TOKEN_AXES, 0.0)
            incomplete = dict.fromkeys(TOKEN_AXES, False)
        previous_step = step
        has_training_metric = row.get("critic/score/mean") not in (None, "") or any(
            row.get(key) not in (None, "") for key in rollout_keys
        )
        for axis in TOKEN_AXES:
            value = row.get(f"rollout/{axis}")
            if value in (None, ""):
                if step > 0 and has_training_metric:
                    incomplete[axis] = True
                continue
            amount = training_number(value, field=f"rollout/{axis}", source=path)
            if amount < 0:
                raise ValueError(f"{path}: rollout/{axis} must be nonnegative")
            if incomplete[axis]:
                continue
            totals[axis] += amount
            by_record[axis][index] = totals[axis]
            by_step[axis][step] = totals[axis]
    return by_record, by_step, rollout_keys


def load_run_data(run: Path, plot_names: set[str], *, baseline: bool = False) -> dict:
    result = {"training": [], "pass8": [], "visits8": [],
              "rollout": {}, "token_by_record": {}, "token_by_step": {}, "rollout_keys": [],
              "turns": {}, "turns_nopad": {}, "turns_extended": {}, "turns_extended_nopad": {},
              "_errors": [], "_turn_missing": []}
    metrics_path = run / "metrics.jsonl"
    if metrics_path.is_file() and plot_names & STEP_PLOTS:
        result["token_by_record"], result["token_by_step"], result["rollout_keys"] = training_token_history(metrics_path)
        if "learning" in plot_names:
            result["training"] = load_training(metrics_path, ["critic/score/mean"])
        if "rollout" in plot_names and result["rollout_keys"]:
            points = load_training(metrics_path, result["rollout_keys"])
            result["rollout"] = {metric: [point for point in points if point["metric"] == metric]
                                 for metric in result["rollout_keys"]}
    # A base result root holds one method's preset; the other selection finds nothing.
    if baseline or (run / "eval").is_dir():
        if plot_names & {"comparison", "comparison_extended"}:
            result["pass8"], errors = eight_visit_evaluations(run, "vanilla", "pass", baseline=baseline)
            result["_errors"].extend(errors)
        if plot_names & {"learning", "variants", "comparison", "comparison_extended", "turns"}:
            result["visits8"], errors = eight_visit_evaluations(run, "ace", "visits", baseline=baseline)
            result["_errors"].extend(errors)
    return result


def add_turns(result: dict, *, progress=None) -> None:
    """Aggregate every available checkpoint's recorded turn trajectories."""
    visits8 = {point["step"]: point for point in result["visits8"]}
    steps = sorted(visits8)
    for index, step in enumerate(steps, 1):
        if progress is not None:
            progress(index, len(steps), step)
        # Each checkpoint's padded and unpadded curves share trajectory files.
        # Work on them together so the remote reader's cache can reuse rewards.
        point = visits8[step]
        for spec_source, output_key, visits, pad_solved in TURN_SPECS:
            if (spec_source != "visits8" or point["local_eval_epochs"] < visits
                    or step in result[output_key]):
                continue
            try:
                result[output_key][step] = curve_rows(
                    point, axis="turn", local_epoch=visits, pad_solved=pad_solved,
                )
            except FileNotFoundError:
                result["_turn_missing"].append(f"visits8 step {step} {output_key}: trajectories unavailable")
            except (OSError, ValueError, KeyError) as error:
                result["_errors"].append(f"visits8 step {step} {output_key}: {error}")


def _curve_rows(record: dict, key: str, k: int) -> list[dict]:
    """pass@k from pass8 or RSI@k from visits8, each measured by its own protocol."""
    rows = [{"step": point["step"], "value": next(row["value"] for row in point["curve"] if row["k"] == k)}
            for point in record.get(key, [])]
    return sorted((row for row in rows if row["value"] is not None), key=lambda row: row["step"])


def _base_pass_rows(record: dict, k: int) -> list[dict]:
    rows = _curve_rows(record, "pass8", k)
    if k == 1 and not rows:
        # A base model's first ACE visit is one unadapted attempt per task.
        rows = _curve_rows(record, "visits8", 1)
    return [row for row in rows if row["step"] == 0]


def export_tables(data: dict[str, dict]) -> dict[str, list[dict]]:
    """Flatten loaded runs into the plotted values of every gallery figure."""
    tables = {name: [] for name in TABLES}
    for method in METHODS:
        record = data.get(method)
        if record is None:
            continue
        by_record = record.get("token_by_record", {})
        by_step = record.get("token_by_step", {})
        series = [("critic/score/mean", record.get("training", [])),
                  *sorted(record.get("rollout", {}).items())]
        for metric, rows in series:
            for row in rows:
                tables["training"].append({
                    "method": method, "metric": metric, "segment": row["segment"],
                    "record": row["record"], "step": row["training_step"],
                    **{f"cumulative_{axis}": by_record.get(axis, {}).get(row["record"]) for axis in TOKEN_AXES},
                    "value": row["value"],
                })

        def evaluation(measure: str, k: int, rows: list[dict]) -> None:
            for row in rows:
                if row["value"] is None:
                    continue
                tables["evaluation"].append({
                    "method": method, "measure": measure, "k": k, "step": row["step"],
                    **{f"cumulative_{axis}": 0.0 if row["step"] == 0 else by_step.get(axis, {}).get(row["step"])
                       for axis in TOKEN_AXES},
                    "value": row["value"],
                })

        for k in range(1, 9):
            evaluation("pass", k, _base_pass_rows(record, k) if method == "base"
                       else _curve_rows(record, "pass8", k))
            evaluation("rsi", k, _curve_rows(record, "visits8", k))
        # Learning curves show the reward through the first three of eight visits.
        for point in sorted(record.get("visits8", []), key=lambda point: point["step"]):
            k = min(3, point.get("local_eval_epochs", 3))
            evaluation("reward_carried", k, [{"step": point["step"], "value": padded_reward(point, k)}])
        for curve in TURN_KEYS:
            for step, rows in sorted(record.get(curve, {}).items()):
                tables["turns"].extend({
                    "method": method, "curve": curve, "step": step,
                    "local_epoch": row.get("local_epoch", 1), "turn": row.get("turn"),
                    "turn_global": row["turn_global"], "value": row["value"],
                } for row in rows)
    return tables


def write_tables(directory: Path, tables: dict[str, list[dict]], manifest: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, fields in TABLES.items():
        with (directory / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: "" if row.get(key) is None else row[key] for key in fields}
                             for row in tables[name])
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def merge_tables(old: dict[str, list[dict]], new: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Keep earlier rows, replace rows the new extraction measured, and never blank a written cell."""
    merged = {}
    for name, key_fields in ROW_KEYS.items():
        rows = {tuple(row[field] for field in key_fields): row for row in old[name]}
        for row in new[name]:
            key = tuple(row[field] for field in key_fields)
            previous = rows.get(key)
            rows[key] = row if previous is None else {
                field: previous.get(field) if row.get(field) is None else row[field] for field in TABLES[name]
            }
        merged[name] = list(rows.values())
    return merged


def merge_manifest(old: dict, new: dict) -> dict:
    if old.get("source") != new["source"]:
        raise ValueError(f"Cannot append {new['source']} data to data extracted from {old.get('source')}")

    def union(key: str, order: tuple[str, ...]) -> list[str]:
        values = {*old.get(key, []), *new[key]}
        return [value for value in order if value in values]

    inputs = {method: dict(groups) for method, groups in old.get("loaded_inputs", {}).items()}
    for method, groups in new["loaded_inputs"].items():
        for key, sources in groups.items():
            inputs.setdefault(method, {})[key] = sorted({*inputs.get(method, {}).get(key, []), *sources})
    return {
        **new,
        "plots": union("plots", PLOTS),
        "methods": union("methods", METHODS),
        "selected_runs": {**old.get("selected_runs", {}), **new["selected_runs"]},
        "loaded_methods": union("loaded_methods", METHODS),
        "loaded_inputs": inputs,
        "rollout_metrics": {method: sorted({*old.get("rollout_metrics", {}).get(method, []),
                                            *new["rollout_metrics"].get(method, [])})
                            for method in {*old.get("rollout_metrics", {}), *new["rollout_metrics"]}},
        "missing_turn_inputs": {**old.get("missing_turn_inputs", {}), **new["missing_turn_inputs"]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="HOST:BENCHMARK, e.g. torch:appworld or local:mbpp")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Extract data for every plot")
    selection.add_argument("--plots", help="Comma-separated plot names")
    parser.add_argument("--methods", help="Comma-separated training methods; default: all")
    parser.add_argument("--model", help="Exact model directory; default: every model with runs")
    parser.add_argument("--source-root", help="Override the source checkout root")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/figures/remote")
    parser.add_argument("--output-dir", type=Path,
                        help="Parent of the per-model directories; data goes to <dir>/<model>/data")
    parser.add_argument("--offline", action="store_true", help="Reuse compact remote summaries and cached inventory")
    parser.add_argument("--full-turn-check", action="store_true",
                        help="Check every saved trajectory file before reusing cached turn curves; "
                             "by default a checkpoint is reused while its evaluation and completion records are unchanged")
    parser.add_argument("--append", action="store_true",
                        help="Merge into existing data: keep earlier rows, overwrite re-extracted rows, never blank a cell")
    parser.add_argument("--list", action="store_true", help="List figure and training method names")
    args = parser.parse_args(argv)
    try:
        if args.list:
            print("Plots: " + ", ".join(PLOTS))
            print("Methods: " + ", ".join(METHODS))
            return 0
        if not args.source or not (args.all or args.plots):
            raise ValueError("--source and either --all or --plots are required")
        host, benchmark = args.source.split(":", 1)
        if not host or benchmark not in BENCHMARKS:
            raise ValueError(f"--source must be HOST:BENCHMARK; benchmarks: {', '.join(sorted(BENCHMARKS))}")
        names = set(PLOTS if args.all else args.plots.split(","))
        methods = set(METHODS if not args.methods else args.methods.split(","))
        if not names or names - set(PLOTS):
            raise ValueError(f"Unknown plots: {', '.join(sorted(names - set(PLOTS)))}; choices: {', '.join(PLOTS)}")
        if not methods or methods - set(METHODS):
            raise ValueError(f"Unknown methods: {', '.join(sorted(methods - set(METHODS)))}; choices: {', '.join(METHODS)}")
        root = source_root(host, args.source_root)
        rows = cached_inventory(host, root, benchmark, args.cache_dir, args.offline)
        models = (sorted({row["model"] for row in rows if method_for(row["training_method"], row["preset"]) in methods})
                  if args.model in (None, "all") else [args.model])
        if not models:
            raise ValueError(f"No {benchmark} runs found under {host}:{root}/examples/{benchmark}/train")
        output_base = args.output_dir or ROOT / "examples" / benchmark / "figures" / "output"
        failures = []
        for model in models:
            model_failures = []
            selected = methods | ({"base"} if names & {"turns", "comparison", "comparison_extended"} else set())
            chosen = selected_runs(rows, model, selected)
            loaded = {}
            needed_methods = set().union(*(set(PLOT_METHODS.get(name, ())) for name in names))
            for method, row in chosen.items():
                if method not in needed_methods:
                    continue
                if not (row["training"] or row["evaluation"]):
                    continue
                try:
                    baseline = method == "base"
                    if host == "local":
                        run = Path(row["path"])
                        loaded[method] = load_run_data(run, names, baseline=baseline)
                    else:
                        loaded[method] = fetch_gallery_rows(
                            host, row["path"], args.cache_dir, names,
                            baseline=baseline, offline=args.offline, full_turn_check=args.full_turn_check,
                        )
                    model_failures.extend(f"{model}/{method}: {error}" for error in loaded[method]["_errors"])
                    if baseline and row.get("vanilla_path") and names & {"comparison", "comparison_extended"}:
                        try:
                            if host == "local":
                                loaded[method]["pass8"], errors = eight_visit_evaluations(
                                    Path(row["vanilla_path"]), "vanilla", "pass", baseline=True,
                                )
                            else:
                                vanilla = fetch_gallery_rows(
                                    host, row["vanilla_path"], args.cache_dir,
                                    names & {"comparison", "comparison_extended"},
                                    baseline=True, offline=args.offline,
                                )
                                loaded[method]["pass8"], errors = vanilla["pass8"], vanilla["_errors"]
                            model_failures.extend(f"{model}/{method} vanilla: {error}" for error in errors)
                        except (OSError, ValueError, KeyError) as error:
                            model_failures.append(f"{model}/{method} vanilla: {error}")
                    if host == "local" and "turns" in names and loaded[method]["visits8"]:
                        try:
                            prior_errors = len(loaded[method]["_errors"])
                            add_turns(loaded[method])
                            model_failures.extend(f"{model}/{method}: {error}"
                                                  for error in loaded[method]["_errors"][prior_errors:])
                        except ValueError as error:
                            model_failures.append(f"{model}/{method} turns: {error}")
                except (OSError, ValueError, KeyError, IndexError) as error:
                    model_failures.append(f"{model}/{method}: {error}")
            tables = {name: [] for name in TABLES}
            for method, record in loaded.items():
                try:
                    exported = export_tables({method: record})
                except (OSError, ValueError, KeyError, TypeError) as error:
                    model_failures.append(f"{model}/{method} export: {error}")
                    continue
                for name, table in exported.items():
                    tables[name].extend(table)
            destination = output_base / model / "data"
            manifest = {
                "source": args.source, "source_root": root, "benchmark": benchmark, "model": model,
                "plots": [name for name in PLOTS if name in names],
                "methods": [method for method in METHODS if method in methods],
                "selected_runs": {method: row["path"] for method, row in chosen.items()},
                "loaded_methods": [method for method in METHODS if method in loaded],
                "loaded_inputs": {
                    method: {key: sorted({point["source"] for point in values})
                             for key, values in record.items() if key in {"training", "pass8", "visits8"}}
                    for method, record in loaded.items()
                },
                "rollout_metrics": {method: record["rollout_keys"] for method, record in loaded.items()},
                "missing_turn_inputs": {method: record["_turn_missing"] for method, record in loaded.items()},
                "failures": model_failures,
            }
            if args.append and (destination / "manifest.json").is_file():
                previous, previous_manifest = read_tables(destination)
                tables = merge_tables(previous, tables)
                manifest = merge_manifest(previous_manifest, manifest)
            write_tables(destination, tables, manifest)
            counts = ", ".join(f"{len(tables[name])} {name}" for name in TABLES)
            print(f"{model}: {counts} rows → {destination}")
            failures.extend(model_failures)
        for failure in failures:
            print(f"FAILED {failure}", file=sys.stderr)
        return 1 if failures else 0
    except (OSError, ValueError, KeyError, IndexError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
