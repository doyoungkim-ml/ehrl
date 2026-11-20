"""Train a Mastermind agent with the current rLLM verl trainer stack."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from rllm.agents.mastermind_agent import MastermindAgent
from rllm.algorithms.ace.agents import ace_empty_playbook_prompt
from rllm.data import DatasetRegistry
from rllm.data.selection import configure_training_dataset_paths
from rllm.environments.mastermind.mastermind import MastermindEnv
from rllm.trainer.agent_trainer import AgentTrainer

ACE_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "ace_rl" / "prompts"


def configure_actor_prompt(config) -> None:
    """Apply ACE-RL's empty-playbook actor prompt when rllm.actor_prompt selects it."""
    prompt = ace_empty_playbook_prompt(config, ACE_PROMPTS_DIR)
    if prompt is not None:
        OmegaConf.update(config, "rllm.agent.agent_args.system_prompt", prompt[1], force_add=True)


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    configure_actor_prompt(config)
    needed = [split for split, key in (("train", "train_files"), ("test", "val_files")) if not config.data.get(key)]
    if any(not DatasetRegistry.dataset_exists("mastermind", split) for split in needed):
        from examples.mastermind.prepare_mastermind_data import prepare_mastermind_data

        prepare_mastermind_data()
    train_dataset = DatasetRegistry.load_dataset("mastermind", "train") if "train" in needed else None
    val_dataset = DatasetRegistry.load_dataset("mastermind", "test") if "test" in needed else None
    paths = configure_training_dataset_paths(config, train_dataset=train_dataset, val_dataset=val_dataset, subset_tag="mastermind_grpo")
    print(f"Mastermind dataset paths: {paths}")
    trainer = AgentTrainer(
        agent_class=MastermindAgent,
        env_class=MastermindEnv,
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
