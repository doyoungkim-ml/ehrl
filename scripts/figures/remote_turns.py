"""Compute turn curves beside remote evaluation artifacts and cache compact rows."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from remote import scalar_fingerprint


# The first three of eight measured visits give the short turn curves.
TURN_SPECS = (
    ("visits8", "turns", 3, True),
    ("visits8", "turns_nopad", 3, False),
    ("visits8", "turns_extended", 8, True),
    ("visits8", "turns_extended_nopad", 8, False),
)
TURN_KEYS = ("turns", "turns_nopad", "turns_extended", "turns_extended_nopad")
READER_FILES = ("plot_turns.py", "plot_eval.py", "plot_comparison.py", "remote.py")

REMOTE_CODE = r"""
import hashlib
import functools
import json
import pathlib
import sys

checkout = pathlib.Path(sys.argv[1]).resolve()
run = pathlib.Path(sys.argv[2]).resolve()
request = json.load(sys.stdin)
scripts = checkout / "scripts/figures"
for name, expected in request["reader_hashes"].items():
    actual = hashlib.sha256((scripts / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Remote turn reader differs from local source: {name}")
sys.path.insert(0, str(scripts))
import plot_turns
plot_turns.task_trajectories = functools.lru_cache(maxsize=8192)(plot_turns.task_trajectories)
curve_rows = plot_turns.curve_rows

curves = {key: {} for key in request["turn_keys"]}
missing = []
errors = []
for item in request["points"]:
    relative = pathlib.PurePosixPath(item["relative_source"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid evaluation source: {relative}")
    source = (run / relative).resolve()
    if not source.is_relative_to(run):
        raise ValueError(f"Evaluation source escapes run: {relative}")
    point = {**item["point"], "source": str(source)}
    for source_key, output_key, visits, pad_solved in request["turn_specs"]:
        if source_key != item["source_key"] or point["local_eval_epochs"] < visits:
            continue
        try:
            rows = curve_rows(point, axis="turn", local_epoch=visits, pad_solved=pad_solved)
        except FileNotFoundError:
            missing.append(f"{source_key} step {point['step']} {output_key}: trajectories unavailable")
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{source_key} step {point['step']} {output_key}: {error}")
        else:
            curves[output_key][str(point["step"])] = [
                {key: row[key] for key in ("turn_global", "turn", "local_epoch", "value")}
                for row in rows
            ]
print(json.dumps({"curves": curves, "missing": missing, "errors": errors}, separators=(",", ":")))
"""


def fetch_turn_rows(host: str, checkout: str, remote_run: str, local_run: Path,
                    record: dict, *, offline: bool) -> bool:
    """Populate a run with compact remote turn rows; return False for an offline cache miss."""
    readers = Path(__file__).parent
    reader_hashes = {name: hashlib.sha256((readers / name).read_bytes()).hexdigest()
                     for name in READER_FILES}
    aggregator_hash = hashlib.sha256(REMOTE_CODE.encode()).hexdigest()
    fingerprint = scalar_fingerprint(local_run)
    points = []
    for source_key in ("visits8",):
        for point in record[source_key]:
            relative = Path(point["source"]).resolve().relative_to(local_run.resolve())
            points.append({
                "source_key": source_key,
                "relative_source": relative.as_posix(),
                "point": {key: value for key, value in point.items() if key != "source"},
            })
    point_ids = [(item["source_key"], item["relative_source"], item["point"]["step"])
                 for item in points]
    cache = local_run / ".remote-turn-curves.json"
    if cache.is_file():
        cached = json.loads(cache.read_text())
        if (cached.get("source") == remote_run and cached.get("fingerprint") == fingerprint
                and cached.get("point_ids") == [list(item) for item in point_ids]
                and cached.get("aggregator_hash") == aggregator_hash
                and cached.get("reader_hashes") == reader_hashes):
            _populate(record, cached["result"])
            return True
    if offline:
        return False
    request = {"reader_hashes": reader_hashes, "turn_keys": TURN_KEYS,
               "turn_specs": TURN_SPECS, "points": points}
    command = ["ssh", host, shlex.join(["python3", "-c", REMOTE_CODE, checkout, remote_run])]
    result = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Remote turn aggregation failed: {result.stderr.strip() or result.stdout.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Remote turn aggregation returned invalid JSON: {result.stdout[:200]}") from error
    _populate(record, payload)
    cache.write_text(json.dumps({"source": remote_run, "fingerprint": fingerprint,
                                 "reader_hashes": reader_hashes,
                                 "aggregator_hash": aggregator_hash,
                                 "point_ids": point_ids, "result": payload}) + "\n")
    return True


def _populate(record: dict, payload: dict) -> None:
    for key in TURN_KEYS:
        record[key] = {int(step): rows for step, rows in payload["curves"][key].items()}
    record["_turn_missing"].extend(payload["missing"])
    record["_errors"].extend(payload["errors"])
