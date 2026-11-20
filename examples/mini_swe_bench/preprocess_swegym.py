#!/usr/bin/env python3
"""Preprocess SWE-Gym data into a DatasetRegistry-compatible parquet layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def get_default_data_dir() -> Path:
    if os.environ.get("MINISW_DATA_DIR"):
        return Path(os.environ["MINISW_DATA_DIR"])
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "rllm" / "data" / "datasets" / "swegym"


def preprocess_swegym(
    output_dir: Path,
    data_source: str = "SumanthRH/SWE-Gym-Subset",
    eval_data_source: str = "SumanthRH/SWE-bench_Verified",
    train_split: str = "train",
    eval_split: str = "test",
) -> tuple[Path, Path]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the optional `datasets` dependency to preprocess SWE-Gym.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_map_fn(split_name: str):
        def process_fn(example, idx):
            return {
                "data_source": split_name,
                "prompt": [{"role": "user", "content": example.get("problem_statement", "")}],
                "env_class": "null",
                "instance": dict(example),
            }

        return process_fn

    try:
        ds = load_dataset(data_source, "default")
        train_ds = ds[train_split] if train_split in ds else ds["train"] if "train" in ds else list(ds.values())[0]
    except Exception:
        train_ds = load_dataset(data_source, split="train")

    try:
        ds = load_dataset(eval_data_source, "default")
        val_ds = ds.get(eval_split) or ds.get("test") or ds.get("validation") or list(ds.values())[0]
    except Exception:
        val_ds = load_dataset(eval_data_source, split="test")

    train_data = train_ds.map(function=make_map_fn(data_source), with_indices=True)
    val_data = val_ds.map(function=make_map_fn(eval_data_source), with_indices=True)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validation.parquet"
    train_data.to_parquet(str(train_path))
    val_data.to_parquet(str(val_path))

    with open(output_dir / "train.jsonl", "w") as file:
        for row in train_data:
            file.write(json.dumps(row) + "\n")
    with open(output_dir / "validation.jsonl", "w") as file:
        for row in val_data:
            file.write(json.dumps(row) + "\n")

    with open(output_dir / "meta.json", "w") as file:
        json.dump(
            {
                "data_source": data_source,
                "eval_data_source": eval_data_source,
                "train_size": len(train_data),
                "val_size": len(val_data),
            },
            file,
            indent=2,
        )

    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-source", type=str, default="SumanthRH/SWE-Gym-Subset")
    parser.add_argument("--eval-data-source", type=str, default="SumanthRH/SWE-bench_Verified")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or str(get_default_data_dir()))
    train_path, val_path = preprocess_swegym(
        output_dir=output_dir,
        data_source=args.data_source,
        eval_data_source=args.eval_data_source,
    )
    print(f"Preprocessed -> {train_path}, {val_path}")


if __name__ == "__main__":
    main()
