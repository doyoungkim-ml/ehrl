"""Train an MBPP agent with the current rLLM verl trainer stack."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from examples.mbpp.mbpp_dataset_utils import load_training_datasets
from rllm.agents.mbpp_agent import MBPPAgent
from rllm.algorithms.ace.agents import ace_empty_playbook_prompt
from rllm.data.selection import configure_training_dataset_paths
from rllm.environments.mbpp.mbpp import MBPPEnv
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
    train_dataset, val_dataset = load_training_datasets(config)

    dataset_paths = configure_training_dataset_paths(
        config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        subset_tag="mbpp_grpo",
    )
    print(f"Train verl path: {dataset_paths['train_verl_path']}")
    print(f"Val verl path: {dataset_paths['val_verl_path']}")
    print(f"Train examples: {dataset_paths['effective_train_examples']}")
    print(f"Val examples: {dataset_paths['effective_val_examples']} (source total: {dataset_paths['total_val_examples']})")

    trainer = AgentTrainer(
        agent_class=MBPPAgent,
        env_class=MBPPEnv,
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
