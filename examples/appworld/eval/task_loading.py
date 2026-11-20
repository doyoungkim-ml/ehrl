"""Load exact AppWorld evaluation splits without using the training registry."""

from __future__ import annotations

import json
import os
from pathlib import Path

SPLITS = {"train", "dev", "test_normal", "test_challenge"}
DEFAULT_TASK_IDS_DIR = Path(__file__).resolve().parents[1] / "data"


def _validate_task_ids(task_ids, source: str) -> list[str]:
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError(f"{source}: expected a nonempty JSON list of task IDs")
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids):
        raise ValueError(f"{source}: task IDs must be nonempty strings")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"{source}: duplicate task IDs would merge independent evaluation tasks")
    return task_ids


def _validate_slice(task_offset: int, num_tasks: int | None) -> None:
    if not isinstance(task_offset, int) or isinstance(task_offset, bool) or task_offset < 0:
        raise ValueError("task_offset must be a nonnegative integer")
    if num_tasks is not None and (not isinstance(num_tasks, int) or isinstance(num_tasks, bool) or num_tasks <= 0):
        raise ValueError("num_tasks must be a positive integer or None")


def load_appworld_tasks(
    *,
    split: str,
    num_tasks: int | None = None,
    task_offset: int = 0,
    task_ids_dir: str | Path | None = None,
) -> list[dict]:
    """Read the requested split's exported IDs, preserving their order and split.

    APPWORLD_TASK_IDS_DIR selects the export directory; standalone evaluations
    default to examples/appworld/data. No datasets are registered or prepared.
    An explicit task limit must fit the split so a truncated slice is never
    reported under a larger task-count label.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown AppWorld split {split!r}; expected one of {sorted(SPLITS)}")
    _validate_slice(task_offset, num_tasks)
    directory = Path(task_ids_dir or os.environ.get("APPWORLD_TASK_IDS_DIR") or DEFAULT_TASK_IDS_DIR).expanduser()
    candidates = [directory / f"appworld_{split}_task_ids.json", directory / f"{split}_task_ids.json", directory / f"{split}.json"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"No AppWorld task ID file for split {split!r} under {directory}. "
            f"Expected one of: {', '.join(candidate.name for candidate in candidates)}. "
            "Set APPWORLD_TASK_IDS_DIR to an existing export or run examples/appworld/export_task_ids.py."
        )
    try:
        task_ids = _validate_task_ids(json.loads(path.read_text()), str(path))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid task ID JSON in {path}: {error.msg}") from error
    if task_offset >= len(task_ids):
        raise ValueError(f"task_offset={task_offset} selects no tasks from AppWorld {split!r} ({len(task_ids)} tasks)")
    stop = len(task_ids) if num_tasks is None else task_offset + num_tasks
    if stop > len(task_ids):
        raise ValueError(f"Requested AppWorld {split!r} slice [{task_offset}:{stop}] exceeds its {len(task_ids)} tasks")
    return [{"task_id": task_ids[index], "split": split, "index": index} for index in range(task_offset, stop)]


def load_specific_tasks(task_ids: list[str]) -> list[dict]:
    """Build a validated explicit task selection; instructions arrive at reset."""
    ids = _validate_task_ids(task_ids, "Explicit AppWorld selection")
    return [{"task_id": task_id, "instruction": f"(Instruction loaded at environment reset for task {task_id})", "split": "custom"} for task_id in ids]
