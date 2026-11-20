"""Train MBPP ACE-RL with a configured controller schedule."""

import hydra

from examples.mbpp.mbpp_dataset_utils import load_training_datasets
from examples.mbpp.train.ace_rl.training_entrypoints import LocalACERLTaskRunner, get_trainer_class_path
from rllm.agents.mbpp_agent import MBPPAgent
from rllm.algorithms.ace.entrypoints import LocalACERLAgentTrainer
from rllm.data.selection import configure_training_dataset_paths
from rllm.environments.mbpp.mbpp import MBPPEnv


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    get_trainer_class_path(config)
    train_dataset, val_dataset = load_training_datasets(config)
    paths = configure_training_dataset_paths(config, train_dataset=train_dataset, val_dataset=val_dataset, subset_tag="mbpp_ace_rl")
    print(f"MBPP dataset paths: {paths}")
    LocalACERLAgentTrainer(
        agent_class=MBPPAgent,
        env_class=MBPPEnv,
        config=config,
        task_runner_cls=LocalACERLTaskRunner,
    ).train()


if __name__ == "__main__":
    main()
