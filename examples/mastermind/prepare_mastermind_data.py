from __future__ import annotations

import argparse

import numpy as np

from rllm.data import DatasetRegistry


def prepare_mastermind_data(train_size: int = 200, test_size: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_seeds = rng.integers(0, 100000, size=train_size)
    test_seeds = rng.integers(0, 100000, size=test_size)
    train_code_lengths = rng.choice([3, 4, 5], size=train_size, p=[0.3, 0.5, 0.2])
    test_code_lengths = rng.choice([3, 4, 5], size=test_size, p=[0.3, 0.5, 0.2])
    train_num_digits = rng.choice([4, 5, 6], size=train_size, p=[0.2, 0.3, 0.5])
    test_num_digits = rng.choice([4, 5, 6], size=test_size, p=[0.2, 0.3, 0.5])

    def build_row(task_seed, code_length, num_digits, index):
        return {
            "seed": int(task_seed),
            "code_length": int(code_length),
            "num_digits": int(num_digits),
            "max_turns": 20,
            "reward_exact_match": 10.0,
            "reward_value_match": 5.0,
            "index": index,
            "uid": f"mastermind_{task_seed}_{code_length}_{num_digits}",
        }

    train_data = [build_row(task_seed, train_code_lengths[index], train_num_digits[index], index) for index, task_seed in enumerate(train_seeds)]
    test_data = [build_row(task_seed, test_code_lengths[index], test_num_digits[index], index) for index, task_seed in enumerate(test_seeds)]

    train_dataset = DatasetRegistry.register_dataset("mastermind", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("mastermind", test_data, "test")
    return train_dataset, test_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mastermind datasets")
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dataset, test_dataset = prepare_mastermind_data(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    print("Sample train example:", train_dataset.get_data()[0])


if __name__ == "__main__":
    main()
