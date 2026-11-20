"""Prepare AppWorld task IDs into DatasetRegistry/verl-ready parquet datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rllm.data import DatasetRegistry

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"


def _load_task_ids_from_json(split: str, data_dir: Path) -> list[str]:
    candidates = [
        data_dir / f"appworld_{split}_task_ids.json",
        data_dir / f"{split}_task_ids.json",
        data_dir / f"{split}.json",
    ]
    for path in candidates:
        if path.exists():
            task_ids = json.loads(path.read_text())
            print(f"Loaded {len(task_ids)} task IDs from {path}")
            return task_ids

    tried = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"No task ID file found for split {split!r} under {data_dir}. Tried: {tried}. Run `python examples/appworld/export_task_ids.py` first.")


def _task_ids_to_verl_data(
    task_ids: list[str],
    *,
    split: str,
    max_turns: int,
) -> list[dict]:
    records: list[dict] = []
    for index, task_id in enumerate(task_ids):
        records.append(
            {
                "data_source": "appworld",
                "prompt": [{"role": "user", "content": "Solve the following AppWorld task."}],
                "ability": "tool_use",
                "reward_model": {"style": "rule", "ground_truth": {"task_id": task_id}},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "task_id": task_id,
                    "max_turns": max_turns,
                },
                "uid": task_id,
            }
        )
    return records


def prepare_appworld_data(
    *,
    train_size: int | None = None,
    test_size: int | None = None,
    max_turns: int = 60,
    data_dir: str | None = None,
    train_split: str = "train",
    test_split: str = "dev",
    local_dir: str | None = None,
    load_from_local: bool = False,
):
    if DatasetRegistry.dataset_exists("appworld", "train") and DatasetRegistry.dataset_exists("appworld", "test"):
        print("AppWorld datasets already registered in memory")
        return (
            DatasetRegistry.load_dataset("appworld", "train"),
            DatasetRegistry.load_dataset("appworld", "test"),
        )

    data_dir_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    if load_from_local and local_dir:
        local_dir_path = Path(local_dir)
        train_path = local_dir_path / "train.parquet"
        test_path = local_dir_path / "test.parquet"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Expected parquet files at {train_path} and {test_path}. Run without --load-from-local first.")

        train_data = pd.read_parquet(train_path).to_dict("records")
        test_data = pd.read_parquet(test_path).to_dict("records")
        print(f"Loaded {len(train_data)} train and {len(test_data)} test rows from {local_dir_path}")
    else:
        print(f"Loading AppWorld task IDs from {data_dir_path}")
        train_ids = _load_task_ids_from_json(train_split, data_dir_path)
        test_ids = _load_task_ids_from_json(test_split, data_dir_path)

        train_data = _task_ids_to_verl_data(train_ids, split="train", max_turns=max_turns)
        test_data = _task_ids_to_verl_data(test_ids, split="test", max_turns=max_turns)

        if local_dir:
            local_dir_path = Path(local_dir)
            local_dir_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(train_data).to_parquet(local_dir_path / "train.parquet", index=False)
            pd.DataFrame(test_data).to_parquet(local_dir_path / "test.parquet", index=False)

    if train_size is not None:
        train_data = train_data[:train_size]
    if test_size is not None:
        test_data = test_data[:test_size]

    train_dataset = DatasetRegistry.register_dataset(name="appworld", split="train", data=train_data)
    test_dataset = DatasetRegistry.register_dataset(name="appworld", split="test", data=test_data)

    print(f"Registered AppWorld datasets: train={len(train_dataset.get_data())} ({train_split}), test={len(test_dataset.get_data())} ({test_split})")
    return train_dataset, test_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AppWorld dataset for GRPO training")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "dev", "test_normal", "test_challenge"],
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="dev",
        choices=["train", "dev", "test_normal", "test_challenge"],
    )
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument("--load-from-local", action="store_true")
    args = parser.parse_args()

    train_dataset, test_dataset = prepare_appworld_data(
        train_size=args.train_size,
        test_size=args.test_size,
        max_turns=args.max_turns,
        data_dir=args.data_dir,
        train_split=args.train_split,
        test_split=args.test_split,
        local_dir=args.local_dir,
        load_from_local=args.load_from_local,
    )

    print("Dataset preparation complete.")
    print(f"Train: {len(train_dataset.get_data())}")
    print(f"Test: {len(test_dataset.get_data())}")


if __name__ == "__main__":
    main()
