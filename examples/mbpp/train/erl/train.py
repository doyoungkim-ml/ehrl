"""Original ERL using the benchmark's ordinary agent and dataset protocol."""

from __future__ import annotations

import hydra

from examples.mbpp.mbpp_dataset_utils import load_training_datasets
from rllm.agents.mbpp_agent import MBPPAgent
from rllm.algorithms.erl.entrypoints import train_original_erl
from rllm.data.selection import configure_training_dataset_paths
from rllm.environments.mbpp.mbpp import MBPPEnv


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset, val_dataset = load_training_datasets(config)

    dataset_paths = configure_training_dataset_paths(
        config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        subset_tag="mbpp_erl",
    )
    print(f"Train verl path: {dataset_paths['train_verl_path']}")
    print(f"Val verl path: {dataset_paths['val_verl_path']}")
    print(f"Train examples: {dataset_paths['effective_train_examples']}")
    print(f"Val examples: {dataset_paths['effective_val_examples']} (source total: {dataset_paths['total_val_examples']})")

    train_original_erl(
        agent_class=MBPPAgent,
        env_class=MBPPEnv,
        config=config,
        adapter_module="examples.mbpp.train.erl.adapter",
    )


if __name__ == "__main__":
    main()
