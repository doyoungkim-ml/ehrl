from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from datasets import load_dataset

from rllm.data import DatasetRegistry


def prepare_mbpp_data(
    train_size: int | None = None,
    test_size: int | None = None,
    max_turns: int = 5,
    timeout: int = 10,
    local_dir: str | None = None,
    load_from_local: bool = False,
    cache_dir: str | None = None,
):
    """Prepare and register MBPP datasets for training and evaluation."""
    if DatasetRegistry.dataset_exists("mbpp", "train") and DatasetRegistry.dataset_exists("mbpp", "test"):
        print("MBPP datasets already registered in memory")
        return (
            DatasetRegistry.load_dataset("mbpp", "train"),
            DatasetRegistry.load_dataset("mbpp", "test"),
        )

    if load_from_local and local_dir:
        train_path = os.path.join(local_dir, "train.parquet")
        test_path = os.path.join(local_dir, "test.parquet")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            raise FileNotFoundError(f"Local parquet files not found in {local_dir}. Run without --load_from_local first to download and save.")

        print(f"Loading from local parquet files in {local_dir}...")
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        train_df["test_list"] = train_df["test_list"].apply(json.loads)
        test_df["test_list"] = test_df["test_list"].apply(json.loads)
        train_data = train_df.to_dict("records")
        test_data = test_df.to_dict("records")
        print(f"  Loaded {len(train_data)} train examples")
        print(f"  Loaded {len(test_data)} test examples")
    else:
        print("Downloading MBPP dataset from HuggingFace...")
        if cache_dir:
            os.environ["HF_DATASETS_CACHE"] = cache_dir
            os.environ["HF_HOME"] = os.path.dirname(cache_dir)
            print(f"  Using cache directory: {cache_dir}")

        dataset = load_dataset("google-research-datasets/mbpp", "full")
        train_raw = list(dataset["train"])
        test_raw = list(dataset["test"])

        print(f"  Downloaded {len(train_raw)} train examples")
        print(f"  Downloaded {len(test_raw)} test examples")

        def process_example(example: dict, index: int, split: str) -> dict:
            return {
                "task_id": example["task_id"],
                "text": example["text"],
                "code": example["code"],
                "test_list": example["test_list"],
                "test_setup_code": example.get("test_setup_code", ""),
                "data_source": "mbpp",
                "max_turns": max_turns,
                "timeout": timeout,
                "index": index,
                "uid": f"mbpp_{split}_{index}",
            }

        train_data = [process_example(example, index, "train") for index, example in enumerate(train_raw)]
        test_data = [process_example(example, index, "test") for index, example in enumerate(test_raw)]

        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            train_df["test_list"] = train_df["test_list"].apply(json.dumps)
            test_df["test_list"] = test_df["test_list"].apply(json.dumps)
            train_path = os.path.join(local_dir, "train.parquet")
            test_path = os.path.join(local_dir, "test.parquet")
            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path, index=False)
            print(f"  Saved train.parquet ({os.path.getsize(train_path) / 1024:.1f} KB)")
            print(f"  Saved test.parquet ({os.path.getsize(test_path) / 1024:.1f} KB)")

            train_df["test_list"] = train_df["test_list"].apply(json.loads)
            test_df["test_list"] = test_df["test_list"].apply(json.loads)
            train_data = train_df.to_dict("records")
            test_data = test_df.to_dict("records")

    if train_size is not None:
        train_data = train_data[:train_size]
        print(f"  Limited train to {len(train_data)} examples")
    if test_size is not None:
        test_data = test_data[:test_size]
        print(f"  Limited test to {len(test_data)} examples")

    train_dataset = DatasetRegistry.register_dataset(name="mbpp", split="train", data=train_data)
    test_dataset = DatasetRegistry.register_dataset(name="mbpp", split="test", data=test_data)

    print("\nRegistered MBPP datasets:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test: {len(test_data)} examples")

    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MBPP dataset for rLLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/mbpp/prepare_mbpp_data.py
  python examples/mbpp/prepare_mbpp_data.py --local_dir /scratch/$USER/data/mbpp
  python examples/mbpp/prepare_mbpp_data.py --local_dir /scratch/$USER/data/mbpp --load_from_local
  python examples/mbpp/prepare_mbpp_data.py --cache_dir /scratch/$USER/hf_cache
  python examples/mbpp/prepare_mbpp_data.py --train_size 10 --test_size 5
        """,
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Limit number of training examples (default: use all)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="Limit number of test examples (default: use all)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Maximum turns per MBPP task (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout for code execution in seconds (default: 10)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Directory to save/load parquet files. If provided, saves downloaded data here.",
    )
    parser.add_argument(
        "--load_from_local",
        action="store_true",
        help="Load from local parquet files instead of downloading from HuggingFace",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom HuggingFace cache directory",
    )
    args = parser.parse_args()

    train_dataset, test_dataset = prepare_mbpp_data(
        train_size=args.train_size,
        test_size=args.test_size,
        max_turns=args.max_turns,
        timeout=args.timeout,
        local_dir=args.local_dir,
        load_from_local=args.load_from_local,
        cache_dir=args.cache_dir,
    )

    print("\nDataset preparation complete!")
    print(f"Train examples: {len(train_dataset.get_data())}")
    print(f"Test examples: {len(test_dataset.get_data())}")


if __name__ == "__main__":
    main()
