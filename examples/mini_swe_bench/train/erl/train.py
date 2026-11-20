"""Original ERL using MiniSweWorkflow's plain SWE interaction protocol."""

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    from examples.mini_swe_bench.env import MiniSWEnv
    from examples.mini_swe_bench.train.common import load_datasets, preflight_checks
    from rllm.agents.swe_agent import SWEAgent
    from rllm.algorithms.erl.entrypoints import train_original_erl

    preflight_checks(config)
    train_dataset, val_dataset = load_datasets()
    workflow_args = OmegaConf.to_container(config.rllm.workflow.get("workflow_args", {})) or {}
    agent_args = {"scaffold": "sweagent", **workflow_args.get("agent_args", {})}
    max_steps = int(workflow_args.get("max_steps", config.rllm.agent.max_steps))
    config.rllm.agent.max_steps = max_steps
    if OmegaConf.select(config, "rllm.agent.engine_args.n_parallel_agents") is None:
        OmegaConf.update(config, "rllm.agent.engine_args.n_parallel_agents", int(config.rllm.workflow.n_parallel_tasks), force_add=True)
    if "timeout" in workflow_args:
        config.rllm.agent.trajectory_timeout = workflow_args["timeout"]
    env_args = {"max_steps": max_steps, "reward_timeout": int(workflow_args.get("reward_timeout", 600)), **workflow_args.get("env_args", {})}
    train_original_erl(
        config,
        agent_class=SWEAgent,
        env_class=MiniSWEnv,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
        adapter_module="examples.mini_swe_bench.train.erl.adapter",
    )


if __name__ == "__main__":
    main()
