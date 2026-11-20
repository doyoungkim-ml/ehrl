#!/usr/bin/env python3
"""Export AppWorld task IDs to JSON files for training data preparation.

APPWORLD_TASK_IDS_DIR selects a job-local output directory for training;
standalone exports default to the data directory next to this example.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    from appworld import load_task_ids

    output_dir = Path(os.environ.get("APPWORLD_TASK_IDS_DIR", Path(__file__).resolve().parent / "data"))
    output_dir.mkdir(parents=True, exist_ok=True)

    appworld_root = os.environ.get("APPWORLD_ROOT")
    if appworld_root:
        try:
            from appworld import update_root
        except ImportError:
            update_root = None
        if update_root is not None:
            update_root(appworld_root)

    for split in ["train", "dev", "test_normal", "test_challenge"]:
        task_ids = load_task_ids(split)
        path = output_dir / f"appworld_{split}_task_ids.json"
        path.write_text(json.dumps(task_ids, indent=2) + "\n")
        print(f"{split:20s}: {len(task_ids):4d} tasks -> {path}")


if __name__ == "__main__":
    main()
