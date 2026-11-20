"""MiniSWEBench bindings to standard and delayed-joint Local ACE-RL trainers."""

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from examples.mini_swe_bench.train.common import load_datasets, preflight_checks

HARNESS_DIR = Path(__file__).resolve().parent
ACE_DEFAULTS = {
    "prompts_dir": str(HARNESS_DIR / "prompts"),
    "initial_playbook": str(HARNESS_DIR / "playbooks" / "initial_playbook.txt"),
    "reflector_adapter_class": "examples.mini_swe_bench.train.ace_rl.adapter.MiniSweReflectorAdapter",
    "reflector_adapter_uses_tokenizer": True,
    "reflector_system_prompt": "Analyze the repository repair attempt using observed evidence. Respond with JSON only.",
    "curator_system_prompt": "Edit a concise task-specific repository repair playbook. Respond with JSON only.",
}
STANDARD_TRAINER = "rllm.algorithms.ace.trainer.LocalACERLAgentPPOTrainer"
DELAYED_JOINT_TRAINER = "rllm.algorithms.ace.trainer_delayed_joint.LocalACERLDelayedJointAgentPPOTrainer"
JOINT_PPO_TRAINER = "rllm.algorithms.ppo.trainer_joint.LocalACEJointPPOTrainer"
AGENT_CLASS = "examples.mini_swe_bench.train.ace_rl.agent.MiniSweACEAgent"
ENV_CLASS = "examples.mini_swe_bench.train.ace_rl.env.MiniSweACEEnv"


def _resolve_trainer_class_path(config):
    mode = OmegaConf.select(config, "rllm.ace.training_mode", default="standard")
    if mode == "standard":
        return STANDARD_TRAINER
    if mode == "delayed_joint":
        return DELAYED_JOINT_TRAINER
    if mode == "joint_ppo":
        return JOINT_PPO_TRAINER
    raise ValueError(f"Unsupported MiniSWEBench ACE training mode: {mode}")


def make_task_runner():
    import ray

    from rllm.algorithms.ace.entrypoints import run_local_ace_rl_task
    from rllm.trainer.verl.train_agent_ppo import TaskRunner

    @ray.remote(num_cpus=1)
    class MiniSweLocalACERLTaskRunner(TaskRunner):
        def run(self, config, agent_class=None, env_class=None, agent_args=None, env_args=None):
            return run_local_ace_rl_task(
                self,
                config=config,
                ace_defaults=ACE_DEFAULTS,
                trainer_class_path=_resolve_trainer_class_path(config),
                default_agent_class_path=AGENT_CLASS,
                default_env_class_path=ENV_CLASS,
                agent_class=agent_class,
                env_class=env_class,
                agent_args=agent_args,
                env_args=env_args,
                log_prefix="MiniSweLocalACERLTaskRunner",
            )

    return MiniSweLocalACERLTaskRunner


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Reject unsupported modes before dataset preparation or GPU initialization.
    _resolve_trainer_class_path(config)

    from examples.mini_swe_bench.train.ace_rl.agent import MiniSweACEAgent
    from examples.mini_swe_bench.train.ace_rl.env import MiniSweACEEnv
    from rllm.algorithms.ace.entrypoints import LocalACERLAgentTrainer

    preflight_checks(config)
    train_dataset, val_dataset = load_datasets()
    config.rllm.workflow.use_workflow = False
    trainer = LocalACERLAgentTrainer(
        agent_class=MiniSweACEAgent,
        env_class=MiniSweACEEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_runner_cls=make_task_runner(),
    )
    trainer.train()


if __name__ == "__main__":
    main()
