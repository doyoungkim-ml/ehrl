#!/usr/bin/env python3
"""Merge disjoint evaluation chunks into a tasksall result using explicit metric weights.

Counts and rollout/total_* are summed. Fresh means use active tasks or generated
trajectories; max_reward_mean includes frozen tasks. Turn summaries cannot be
reconstructed from chunk summaries and are omitted, with keys listed in provenance.
Source runs are preserved. Invalid or overlapping results are rejected before writing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CHUNK_RE = re.compile(r"^(?P<tag>.+)_step(?P<step>\d+)_tasks(?P<a>\d+)-(?P<b>\d+)$")
CHUNK_K_RE = re.compile(r"^(?P<tag>.+)_step(?P<step>\d+)_tasks(?P<k>\d+)$")
COUNT_KEYS = {
    "num_trajectories",
    "success_count",
    "n_active_tasks",
    "n_solved",
    "n_tasks",
    "ace/policy_rollouts",
    "ace/reflector_rollouts",
    "ace/curator_rollouts",
    "ace/total_rollouts",
    "ace/candidates",
    "ace/guidebooks",
}
TRAJECTORY_MEANS = {"success_rate", "avg_reward", "avg_steps", "ace/current_reward"}
SEQUENTIAL_PROTOCOL = "sequential_adaptation_v1"
LEGACY_PROTOCOL = "legacy_search_v1"


def parse_chunk(name: str, total: int):
    match = CHUNK_RE.fullmatch(name)
    if match:
        return match["tag"], int(match["step"]), int(match["a"]), int(match["b"])
    match = CHUNK_K_RE.fullmatch(name)
    if match and 0 < int(match["k"]) < total:
        return match["tag"], int(match["step"]), 0, int(match["k"]) - 1
    return None


def merge_metric_dicts(parts: list[tuple[dict, int]]) -> dict:
    """Sum counts and weight means; reject missing metrics from nonempty chunks."""
    if any(weight < 0 for _, weight in parts):
        raise ValueError("Metric weights must be nonnegative")
    merged = {}
    for key in sorted({key for metrics, _ in parts for key in metrics}):
        present = [(metrics[key], weight) for metrics, weight in parts if key in metrics and metrics[key] is not None]
        if any(weight > 0 and (key not in metrics or metrics[key] is None) for metrics, weight in parts):
            raise ValueError(f"Metric {key!r} is missing from a nonempty chunk")
        if not present:
            continue
        if all(isinstance(value, int | float) and not isinstance(value, bool) for value, _ in present):
            if not all(math.isfinite(value) for value, _ in present):
                raise ValueError(f"Metric {key!r} is not finite")
            if key in COUNT_KEYS or key.startswith("rollout/total_"):
                merged[key] = sum(value for value, _ in present)
            elif key == "local_eval_epochs":
                if len({value for value, _ in present}) != 1:
                    raise ValueError("local_eval_epochs differs across chunks")
                merged[key] = present[0][0]
            else:
                denominator = sum(weight for _, weight in present)
                merged[key] = sum(value * weight for value, weight in present) / denominator if denominator else 0.0
        else:
            if any(value != present[0][0] for value, _ in present):
                raise ValueError(f"Metric label {key!r} differs across chunks")
            merged[key] = present[0][0]
    return merged


def merge_stage(stages: list[dict], sizes: list[int], omitted: set[str]) -> dict:
    """Merge one aligned evaluation or local epoch, preserving its population."""
    merged = {key: value for key, value in stages[0].items() if key not in {"metrics", "tasks", "local_epochs", "artifacts"}}
    for key in ("step", "epoch", "label", "local_eval_epoch", "local_eval_epochs"):
        if any(stage.get(key) != stages[0].get(key) for stage in stages):
            raise ValueError(f"Evaluation {key} differs across chunks")
    tasks = [task for stage in stages for task in stage.get("tasks", [])]
    ids = [str(task["task_id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id across chunk results")
    if any("tasks" in stage for stage in stages):
        merged["tasks"] = tasks
    metrics = [stage.get("metrics", {}) for stage in stages]
    active = [metric.get("n_active_tasks", len(stage["tasks"]) if "tasks" in stage else size) for stage, metric, size in zip(stages, metrics, sizes, strict=True)]
    if any(not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= size for n, size in zip(active, sizes, strict=True)):
        raise ValueError("Invalid active task count")
    for stage, count in zip(stages, active, strict=True):
        if "tasks" in stage and len(stage["tasks"]) != count:
            raise ValueError("Task rows disagree with active task count")
    means, trajectory_means, padded = [], [], []
    for metric, active_count, size in zip(metrics, active, sizes, strict=True):
        omitted.update(key for key in metric if key.startswith("turn/"))
        means.append(({key: value for key, value in metric.items() if key not in TRAJECTORY_MEANS and key != "max_reward_mean" and not key.startswith("turn/")}, active_count))
        trajectory_means.append(({key: value for key, value in metric.items() if key in TRAJECTORY_MEANS}, metric.get("num_trajectories", active_count)))
        padded.append(({key: value for key, value in metric.items() if key == "max_reward_mean"}, size))
    merged["metrics"] = {**merge_metric_dicts(means), **merge_metric_dicts(trajectory_means), **merge_metric_dicts(padded)}
    return merged


def merge_sequential_stages(stages: list[dict], sizes: list[int], omitted: set[str]) -> dict:
    """Merge complete measured chains, rebuilding aliases from merged visits."""
    for key in ("step", "epoch", "local_eval_protocol", "g_act", "policy_visits", "local_eval_epochs", "adaptation_updates", "adaptation_update_slots_per_task"):
        if any(stage.get(key) != stages[0].get(key) for stage in stages):
            raise ValueError(f"Sequential evaluation {key} differs across chunks")
    count, samples_per_task = stages[0].get("policy_visits"), stages[0].get("g_act")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (count, samples_per_task)):
        raise ValueError("Sequential chunks require positive policy_visits and g_act")
    if stages[0].get("local_eval_epochs") != count:
        raise ValueError("Sequential chunks have inconsistent policy visit counts")
    slots = stages[0].get("adaptation_update_slots_per_task")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots not in (0, count - 1) or stages[0].get("adaptation_updates") != slots:
        raise ValueError("Sequential chunks have inconsistent adaptation update slots")
    all_ids, seen_attempts = [], set()
    totals = {role: 0 for role in ("policy", "reflector", "curator", "candidate", "total")}
    applied_total = 0
    for stage, size in zip(stages, sizes, strict=True):
        ids = stage.get("evaluation_task_ids")
        if not isinstance(ids, list) or len(ids) != size or any(not isinstance(task_id, str) or not task_id for task_id in ids) or len(set(ids)) != size:
            raise ValueError("Sequential chunk task declaration differs from its slice")
        all_ids.extend(ids)
        epochs = stage.get("local_epochs", [])
        if len(epochs) != count or [row.get("local_eval_epoch") for row in epochs] != list(range(1, count + 1)):
            raise ValueError("Sequential chunks require every ordered policy visit")
        applied = 0
        for visit, epoch in enumerate(epochs, start=1):
            rows = epoch.get("tasks", [])
            if len(rows) != size or {row.get("task_id") for row in rows} != set(ids):
                raise ValueError("Sequential chunks must retain all tasks on every visit")
            for row in rows:
                samples = row.get("samples", [])
                if len(samples) != samples_per_task or [sample.get("sample_index") for sample in samples] != list(range(samples_per_task)):
                    raise ValueError("Sequential chunks must retain every policy sample")
                for sample in samples:
                    attempt, reward = sample.get("attempt_id"), sample.get("reward")
                    if not isinstance(attempt, str) or not attempt or attempt in seen_attempts:
                        raise ValueError("Sequential chunks contain missing or duplicate attempt IDs")
                    if isinstance(reward, bool) or not isinstance(reward, int | float) or not math.isfinite(reward) or not isinstance(sample.get("success"), bool):
                        raise ValueError("Sequential chunks contain invalid sample outcomes")
                    seen_attempts.add(attempt)
                if not isinstance(row.get("adaptation_applied"), bool) or (visit == count and row["adaptation_applied"]):
                    raise ValueError("Sequential chunks contain invalid adaptation records")
                applied += row["adaptation_applied"]
        final = stage.get("final_policy_evaluation", {})
        if final.get("kind") != "last_measured_policy_visit" or final.get("policy_visit") != count or final.get("after_local_eval_epoch") != count - 1 or final.get("adaptation_applied") is not False:
            raise ValueError("Sequential chunks have inconsistent final policy visits")
        if final.get("tasks") != epochs[-1]["tasks"] or final.get("metrics") != epochs[-1]["metrics"] or stage.get("tasks") != epochs[-1]["tasks"]:
            raise ValueError("Sequential final policy must alias the last measured visit")
        expected_work = {"policy_rollouts": size * samples_per_task, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0}
        if final.get("work") != expected_work or final.get("provenance", {}).get("extra_policy_rollouts") != 0:
            raise ValueError("Sequential final policy has unexpected work")
        adaptation = stage.get("adaptation_evaluation", {})
        if adaptation.get("kind") != "measured_adaptation_visits" or adaptation.get("policy_visits") != count - 1 or adaptation.get("local_epochs") != epochs[:-1]:
            raise ValueError("Sequential adaptation must retain its measured visits")
        counts = stage.get("rollout_counts", {})
        if set(counts) != set(totals) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
            raise ValueError("Sequential chunks have invalid rollout counts")
        if counts["policy"] != size * samples_per_task * count or counts["candidate"] != 0 or counts["total"] != sum(counts[role] for role in totals if role != "total"):
            raise ValueError("Sequential chunks have inconsistent rollout counts")
        for role in ("policy", "reflector", "curator"):
            measured = [epoch.get("metrics", {}).get(f"ace/{role}_rollouts") for epoch in epochs]
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in measured) or sum(measured) != counts[role]:
                raise ValueError("Sequential rollout counts differ from measured visits")
        if any(epoch["metrics"].get("ace/candidate_policy_rollouts") != 0 or epoch["metrics"].get("ace/policy_rollouts") != size * samples_per_task for epoch in epochs):
            raise ValueError("Sequential visits have an inconsistent policy budget")
        if stage.get("actually_applied_context_updates") != applied:
            raise ValueError("Sequential applied update count differs from measured rows")
        for role in totals:
            totals[role] += counts[role]
        applied_total += applied
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate task_id across sequential chunks")
    epochs = [merge_stage(list(parts), sizes, omitted) for parts in zip(*(stage["local_epochs"] for stage in stages), strict=True)]
    last = epochs[-1]
    omitted.update(f"final_policy/{key}" for key in tuple(omitted) if key.startswith("turn/"))
    merged = dict(stages[0])
    merged.update(
        evaluation_task_ids=all_ids, local_epochs=epochs, tasks=last["tasks"],
        rollout_counts=totals, actually_applied_context_updates=applied_total,
        metrics={**last["metrics"], "local_eval_epochs": count, **{f"final_policy/{key}": value for key, value in last["metrics"].items()}},
        adaptation_evaluation={"kind": "measured_adaptation_visits", "policy_visits": count - 1, "local_epochs": epochs[:-1]},
        final_policy_evaluation={
            **stages[0]["final_policy_evaluation"], "tasks": last["tasks"], "metrics": last["metrics"],
            "work": {"policy_rollouts": sum(sizes) * samples_per_task, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0},
            "provenance": {"source": "merged_measured_visits", "extra_policy_rollouts": 0},
        },
    )
    return merged


def _extension_provenance(paths: list[Path]) -> dict | None:
    files = [path / "extended_from.json" for path in paths]
    if not any(path.exists() for path in files):
        return None
    if not all(path.exists() for path in files):
        raise ValueError("Only some chunks have extension provenance")
    records = [json.loads(path.read_text()) for path in files]
    for key in ("extended_from", "step", "start_epoch", "mode"):
        if any(record.get(key) != records[0].get(key) for record in records):
            raise ValueError(f"Extension provenance {key} differs across chunks")
    result = dict(records[0])
    for key in ("n_tasks", "n_solved_at_start"):
        if all(key in record for record in records):
            result[key] = sum(record[key] for record in records)
    snapshots = {}
    for record in records:
        for task_id, timestamp in record.get("snapshot_mtimes", {}).items():
            if task_id in snapshots:
                raise ValueError("Duplicate task_id in extension provenance")
            snapshots[task_id] = timestamp
    if snapshots:
        result["snapshot_mtimes"] = snapshots
    return result


def merge_step(chunks: list[str], total: int, dry_run: bool = False) -> str:
    if total < 1 or not chunks:
        raise ValueError("A positive total and at least one chunk are required")
    parsed = [(parse_chunk(Path(chunk).name, total), Path(chunk).resolve()) for chunk in chunks]
    if any(item is None for item, _ in parsed):
        raise ValueError("Unrecognized chunk directory name")
    tag, step, _, _ = parsed[0][0]
    if any(item[:2] != (tag, step) for item, _ in parsed):
        raise ValueError("Chunks have different run tags or checkpoint steps")
    if len({path.parent for _, path in parsed}) != 1:
        raise ValueError("Chunks must share a parent directory")
    ranges = sorted((item[2], item[3], path) for item, path in parsed)
    covered = [index for first, last, _ in ranges for index in range(first, last + 1)]
    if sorted(covered) != list(range(total)):
        return f"SKIP {tag} step{step}: incomplete or overlapping coverage"
    paths = [path for _, _, path in ranges]
    sizes = [last - first + 1 for first, last, _ in ranges]
    output = paths[0].parent / f"{tag}_step{step}_tasksall"
    if output.is_dir() and list(output.glob("metrics_step*.json")):
        return f"OK {tag} step{step}: merged result already exists"
    metric_paths = [path / f"metrics_step{step}.json" for path in paths]
    if not all(path.is_file() for path in metric_paths):
        return f"SKIP {tag} step{step}: a chunk has no metrics yet"
    data = [json.loads(path.read_text()) for path in metric_paths]
    protocols = set()
    for record in data:
        protocol = record.get("local_eval_protocol", LEGACY_PROTOCOL)
        if protocol not in (LEGACY_PROTOCOL, SEQUENTIAL_PROTOCOL):
            raise ValueError("Unknown local evaluation protocol in chunks")
        if any(stage.get("local_eval_protocol", LEGACY_PROTOCOL) != protocol for stage in record.get("evaluations", [])):
            raise ValueError("Chunk metrics disagree on local evaluation protocol")
        protocols.add(protocol)
    if len(protocols) != 1:
        raise ValueError("Chunks have different local evaluation protocols")
    sequential = protocols == {SEQUENTIAL_PROTOCOL}
    for key in ("experiment_name", "model_name", "local_eval_epochs"):
        if any(record.get(key) != data[0].get(key) for record in data):
            raise ValueError(f"Run {key} differs across chunks")
    evaluations = [record.get("evaluations", []) for record in data]
    if not evaluations[0] or any(len(items) != len(evaluations[0]) for items in evaluations):
        raise ValueError("Evaluation counts differ or are empty")
    omitted: set[str] = set()
    merged = {key: value for key, value in data[0].items() if key != "evaluations"}
    merged["evaluations"] = []
    for stages in zip(*evaluations, strict=True):
        if sequential:
            merged["evaluations"].append(merge_sequential_stages(list(stages), sizes, omitted))
            continue
        stage_result = merge_stage(list(stages), sizes, omitted)
        epoch_lists = [stage.get("local_epochs", []) for stage in stages]
        if any(len(epochs) != len(epoch_lists[0]) for epochs in epoch_lists):
            raise ValueError("Local epoch counts differ across chunks")
        if epoch_lists[0]:
            stage_result["local_epochs"] = [merge_stage(list(epochs), sizes, omitted) for epochs in zip(*epoch_lists, strict=True)]
            epoch_ids = [epoch["local_eval_epoch"] for epoch in stage_result["local_epochs"]]
            if len(epoch_ids) != len(set(epoch_ids)) or epoch_ids != sorted(epoch_ids):
                raise ValueError("Local epochs are duplicated or out of order")
        merged["evaluations"].append(stage_result)
    task_dirs = [task for path in paths for task in sorted(path.glob("task_*")) if task.is_dir()]
    if sequential:
        # Mastermind places the same per-task artifact schema under playbooks/.
        task_dirs.extend(task for path in paths for task in sorted(path.glob("playbooks/task_*")) if task.is_dir())
    if len({task.name for task in task_dirs}) != len(task_dirs):
        raise ValueError("Task directory collision across chunks")
    extension = _extension_provenance(paths)
    provenance = {
        "merged_from": [str(path) for path in paths],
        "total_tasks": total,
        "linked_tasks": len(task_dirs),
        "omitted_metric_keys": sorted(omitted),
        "job_ids": [(path / "slurm_job_id.txt").read_text().strip() for path in paths if (path / "slurm_job_id.txt").is_file()],
    }
    if dry_run:
        return f"WOULD-MERGE {tag} step{step} from {len(paths)} chunks; omitted metrics: {sorted(omitted)}"
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        for task in task_dirs:
            (staging / task.name).symlink_to(os.path.relpath(task, output))
        (staging / f"metrics_step{step}.json").write_text(json.dumps(merged, indent=2) + "\n")
        (staging / "merged_from.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if extension is not None:
            (staging / "extended_from.json").write_text(json.dumps(extension, indent=2) + "\n")
        if output.exists():
            output.rename(output.with_name(f"{output.name}_stale_{datetime.now():%Y%m%d%H%M%S%f}"))
        staging.rename(output)
    return f"MERGED {tag} step{step}: {len(paths)} chunks, {len(task_dirs)} task links; omitted metrics: {sorted(omitted)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_base", required=True, type=Path)
    parser.add_argument("--total", required=True, type=int)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.total < 1 or not args.runs_base.is_dir():
        parser.error("--total must be positive and --runs_base must be an existing directory")
    groups = defaultdict(list)
    for path in sorted(args.runs_base.glob("*_tasks*")):
        parsed = parse_chunk(path.name, args.total)
        if parsed and path.is_dir():
            groups[parsed[:2]].append(str(path))
    failed = False
    for chunks in groups.values():
        try:
            print(merge_step(chunks, args.total, args.dry_run))
        except (ValueError, KeyError, OSError) as error:
            print(f"ERROR {Path(chunks[0]).name}: {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
