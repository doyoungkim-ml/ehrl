"""Original ERL using the benchmark's ordinary agent and dataset protocol."""

from __future__ import annotations

import hydra

from rllm.agents.mastermind_agent import MastermindAgent
from rllm.algorithms.erl.entrypoints import train_original_erl
from rllm.data import DatasetRegistry
from rllm.data.selection import configure_training_dataset_paths
from rllm.environments.mastermind.mastermind import MastermindEnv


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    needed = [split for split, key in (("train", "train_files"), ("test", "val_files")) if not config.data.get(key)]
    if any(not DatasetRegistry.dataset_exists("mastermind", split) for split in needed):
        from examples.mastermind.prepare_mastermind_data import prepare_mastermind_data

        prepare_mastermind_data()
    train_dataset = DatasetRegistry.load_dataset("mastermind", "train") if "train" in needed else None
    val_dataset = DatasetRegistry.load_dataset("mastermind", "test") if "test" in needed else None
    paths = configure_training_dataset_paths(config, train_dataset=train_dataset, val_dataset=val_dataset, subset_tag="mastermind_erl")
    print(f"Mastermind dataset paths: {paths}")
    train_original_erl(
        agent_class=MastermindAgent,
        env_class=MastermindEnv,
        config=config,
        adapter_module="examples.mastermind.train.erl.adapter",
    )


if __name__ == "__main__":
    main()
