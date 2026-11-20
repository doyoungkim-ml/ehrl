"""Resolve MBPP built-in splits only when no explicit training files are supplied."""

from rllm.data import DatasetRegistry


def load_training_datasets(config):
    needed = [split for split, key in (("train", "train_files"), ("test", "val_files")) if not config.data.get(key)]
    if any(not DatasetRegistry.dataset_exists("mbpp", split) for split in needed):
        from examples.mbpp.prepare_mbpp_data import prepare_mbpp_data

        prepare_mbpp_data(max_turns=int(config.rllm.agent.max_steps))
    datasets = []
    for split in ("train", "test"):
        dataset = DatasetRegistry.load_dataset("mbpp", split) if split in needed else None
        if split in needed and dataset is None:
            raise ValueError(f"Missing MBPP {split} dataset; run python -m examples.mbpp.prepare_mbpp_data")
        datasets.append(dataset)
    return tuple(datasets)
