"""Read gallery inputs on the source host and return compact plotting rows."""

from __future__ import annotations

import json
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

from remote import cache_path


# Send the local readers for this one invocation. The remote checkout may have
# a different source revision; temporary readers are removed after aggregation.
READER_FILES = (
    "quickplot.py", "gallery_schema.py", "remote_gallery.py", "remote_turns.py", "remote.py",
    "plot_eval.py", "plot_training.py", "plot_turns.py", "plot_comparison.py",
)
REMOTE_CODE = r"""
import json
import pathlib
import sys
import tempfile

request = json.load(sys.stdin)
with tempfile.TemporaryDirectory(prefix="quickplot-readers-") as directory:
    for name, source in request["readers"].items():
        (pathlib.Path(directory) / name).write_text(source)
    sys.path.insert(0, directory)
    import quickplot
    import plot_turns
    from concurrent.futures import ThreadPoolExecutor
    from functools import lru_cache

    run = pathlib.Path(sys.argv[1]).resolve(strict=True)
    names = set(request["plots"])
    record = quickplot.load_run_data(run, names, baseline=request["baseline"])
    turn_keys = ("turns", "turns_nopad", "turns_extended", "turns_extended_nopad")
    if request["turns"] and record["visits8"]:
        def selected_points(data):
            return {str(point["step"]): point for point in data["visits8"]}

        def turn_files(point, curves, step):
            key = "turns_extended_nopad" if step in curves["turns_extended_nopad"] else "turns_nopad"
            rows = curves[key].get(step, [])
            source = pathlib.Path(point["source"])
            files = {str(source), str(source.parent / "evaluation_config.json"),
                     str(source.parent / "evaluation_complete.json")}
            for encoded in {row.get("trajectory_sources", "") for row in rows}:
                if encoded:
                    files.update(json.loads(encoded))
            return sorted(files)

        def file_stats(paths):
            def one(path):
                try:
                    info = pathlib.Path(path).stat()
                    return path, [info.st_size, info.st_mtime_ns]
                except FileNotFoundError:
                    return path, None
            with ThreadPoolExecutor(max_workers=8) as pool:
                return dict(pool.map(one, paths))

        def completion_stamp(point):
            # The evaluation result, its configuration and its completion record
            # change whenever a checkpoint is evaluated again. Checking only
            # these avoids statting every saved trajectory on each refresh.
            source = pathlib.Path(point["source"])
            stamps = file_stats([str(source), str(source.parent / "evaluation_config.json"),
                                 str(source.parent / "evaluation_complete.json")])
            return None if any(value is None for value in stamps.values()) else stamps

        record["_turn_reused"] = []
        record["_turn_stamps"] = {}
        validated_stats = {}
        prior = request.get("prior")
        if prior:
            old = prior["record"]
            old_points = old["points"]
            current = selected_points(record)
            for step, point in current.items():
                keys = turn_keys
                if old_points.get(step) != point["source"] or any(step not in old["curves"][key] for key in keys):
                    continue
                previous_stats = old.get("stats", {}).get(step)
                stamp = completion_stamp(point)
                if (not request.get("full_turn_check") and stamp is not None and previous_stats is not None
                        and old.get("stamps", {}).get(step) == stamp):
                    current_stats = previous_stats
                    unchanged = True
                elif previous_stats is not None:
                    current_stats = file_stats(turn_files(point, old["curves"], step))
                    unchanged = previous_stats == current_stats
                else:
                    current_stats = file_stats(turn_files(point, old["curves"], step))
                    unchanged = (all(value is not None for path, value in current_stats.items()
                                     if path == point["source"] or "/chat_completions/eval/" in path)
                                 and all(value is None or value[1] <= prior["cached_at_ns"]
                                         for value in current_stats.values()))
                if unchanged:
                    for key in keys:
                        record[key][int(step)] = old["curves"][key][step]
                    record["_turn_reused"].append(int(step))
                    validated_stats[step] = current_stats
        plot_turns.task_trajectories = lru_cache(maxsize=8192)(plot_turns.task_trajectories)
        quickplot.add_turns(record, progress=lambda index, total, step: print(
            f"turn checkpoint {index}/{total} (step {step})"
            + (" cached" if step in record["_turn_reused"] else ""), file=sys.stderr, flush=True))
        record["_turn_stats"] = {}
        for step, point in selected_points(record).items():
            keys = turn_keys
            if all(int(step) in record[key] for key in keys):
                curves = {key: {step: record[key][int(step)]} for key in keys}
                curves.setdefault("turns_extended_nopad", {})
                record["_turn_stats"][step] = validated_stats.get(step) or file_stats(turn_files(point, curves, step))
                record["_turn_stamps"][step] = completion_stamp(point)
    # pass@K and RSI@K curves are already measured; only carried rewards need the source.
    for point in record["visits8"]:
        computed = {"reward": {}}
        try:
            if names & {"learning", "variants"}:
                k = min(3, point["local_eval_epochs"])
                computed["reward"][str(k)] = quickplot.padded_reward(point, k)
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["_errors"].append(f"{point['source']}: {error}")
        point["computed"] = computed
    print(json.dumps(record, separators=(",", ":"), allow_nan=False))
"""


def _restore_int_keys(record: dict) -> dict:
    for field in ("token_by_record", "token_by_step"):
        record[field] = {axis: {int(step): value for step, value in rows.items()}
                         for axis, rows in record[field].items()}
    for field in ("turns", "turns_nopad", "turns_extended", "turns_extended_nopad"):
        record[field] = {int(step): rows for step, rows in record[field].items()}
    return record


def fetch_gallery_rows(host: str, run: str, cache_dir: Path, plot_names: set[str],
                       *, baseline: bool = False, offline: bool = False, full_turn_check: bool = False) -> dict:
    """Refresh a compact summary, or reuse its last successful offline snapshot."""
    directory = cache_path(f"{host}:{run}", cache_dir)
    selection = hashlib.sha256(",".join(sorted(plot_names)).encode()).hexdigest()[:12]
    cache = directory / f".gallery-rows-{selection}.json"
    legacy = directory / ".gallery-rows.json"
    if offline:
        candidates = (cache, *sorted(directory.glob(".gallery-rows-*.json")), legacy)
        for candidate in dict.fromkeys(candidates):
            if candidate.is_file():
                saved = json.loads(candidate.read_text())
                if (saved.get("run") == run and saved.get("baseline") == baseline
                        and plot_names <= set(saved["plots"])):
                    return _restore_int_keys(saved["record"])
        raise ValueError(f"No cached gallery rows cover this request at {directory}; refresh without --offline")
    readers = Path(__file__).parent
    request = {"readers": {name: (readers / name).read_text() for name in READER_FILES},
               "plots": sorted(plot_names), "baseline": baseline, "turns": "turns" in plot_names,
               "full_turn_check": full_turn_check}
    if "turns" in plot_names:
        candidates = (cache, *sorted(directory.glob(".gallery-rows-*.json")), legacy)
        available = []
        for candidate in dict.fromkeys(candidates):
            if not candidate.is_file():
                continue
            saved = json.loads(candidate.read_text())
            if (saved.get("run") != run or saved.get("baseline") != baseline
                    or "turns" not in saved.get("plots", [])):
                continue
            count = (len(saved["record"].get("turns_nopad", {}))
                     + len(saved["record"].get("turns_extended_nopad", {})))
            available.append((count, candidate.stat().st_mtime_ns, saved))
        if available:
            _, cached_at_ns, saved = max(available, key=lambda item: (item[0], item[1]))
            old = saved["record"]
            points = {str(point["step"]): point["source"] for point in old.get("visits8", [])}
            request["prior"] = {"cached_at_ns": cached_at_ns,
                                "record": {"points": points,
                                           "curves": {key: old[key] for key in (
                                               "turns", "turns_nopad", "turns_extended", "turns_extended_nopad")},
                                           "stats": old.get("_turn_stats", {}),
                                           "stamps": old.get("_turn_stamps", {})}}
    command = ["ssh", host, shlex.join(["python3", "-c", REMOTE_CODE, run])]
    print(f"Reading {host}:{run} for {', '.join(sorted(plot_names))}", file=sys.stderr, flush=True)
    result = subprocess.run(command, input=json.dumps(request), text=True,
                            stdout=subprocess.PIPE, check=False)
    if result.returncode:
        raise ValueError(f"Remote gallery aggregation failed (exit {result.returncode})")
    try:
        record = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Remote gallery aggregation returned invalid JSON: {result.stdout[:200]}") from error
    directory.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"run": run, "plots": sorted(plot_names), "baseline": baseline,
                                 "record": record}) + "\n")
    return _restore_int_keys(record)
