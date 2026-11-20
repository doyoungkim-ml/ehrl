"""MiniSWEBench GRPO with the existing cumulative SWE workflow."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from examples.mini_swe_bench.train.common import load_datasets, preflight_checks

ACE_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "ace_rl" / "prompts"


def select_workflow_args(config) -> dict:
    """Use ACE-RL's agent and empty-playbook template when rllm.actor_prompt selects it."""
    from examples.mini_swe_bench.train.ace_rl.agent import MiniSweACEAgent
    from rllm.algorithms.ace.agents import ace_empty_playbook_prompt

    prompt = ace_empty_playbook_prompt(config, ACE_PROMPTS_DIR)
    if prompt is None:
        return {}
    OmegaConf.update(config, "rllm.workflow.workflow_args.agent_args.ace_prompt_template", prompt[0], force_add=True)
    return {"agent_cls": MiniSweACEAgent}


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    from examples.mini_swe_bench.workflow import MiniSweWorkflow
    from rllm.trainer.agent_trainer import AgentTrainer

    preflight_checks(config)
    workflow_args = select_workflow_args(config)
    train_dataset, val_dataset = load_datasets()
    config.rllm.workflow.use_workflow = True
    config.rllm.workflow.name = "mini_swe_bench"
    trainer = AgentTrainer(
        workflow_class=MiniSweWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
