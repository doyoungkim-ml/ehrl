"""AppWorld local ACE+RL Hydra/Ray entrypoint helpers."""

from __future__ import annotations

from pathlib import Path

import ray

from examples.appworld.prepare_appworld_data import prepare_appworld_data
from rllm.algorithms.ace.entrypoints import (
    LocalACERLAgentTrainer,
    run_local_ace_rl_task,
)
from rllm.data import DatasetRegistry
from rllm.trainer.verl.train_agent_ppo import TaskRunner

_HARNESS_DIR = Path(__file__).resolve().parent
ACE_DEFAULTS = {
    "reflector_system_prompt": "You are an expert Python coding analyst. Respond with JSON only.",
    "curator_system_prompt": "You are a master curator of knowledge in Python programming. Respond with JSON only.",
    "prompts_dir": str(_HARNESS_DIR / "prompts"),
    "initial_playbook": str(_HARNESS_DIR / "playbooks" / "initial_playbook.txt"),
    "reflector_adapter_class": "examples.appworld.train.ace_rl.appworld_adapter.AppWorldReflectorAdapter",
    "reflector_adapter_uses_tokenizer": True,
}

APPWORLD_AGENT = "rllm.agents.appworld_react_agents_ace.AppWorldReactAgentACE"
APPWORLD_ENV = "rllm.environments.appworld.appworld_env.AppWorldEnv"
STANDARD_TRAINER = "rllm.algorithms.ace.trainer.LocalACERLAgentPPOTrainer"
ITERATIVE_TRAINER = "rllm.algorithms.ace.trainer_iterative.LocalACERLIterativeAgentPPOTrainer"
DELAYED_JOINT_TRAINER = "rllm.algorithms.ace.trainer_delayed_joint.LocalACERLDelayedJointAgentPPOTrainer"
JOINT_PPO_TRAINER = "rllm.algorithms.ppo.trainer_joint.LocalACEJointPPOTrainer"


@ray.remote(num_cpus=1)
class AppWorldLocalACERLTaskRunner(TaskRunner):
    def run(self, config, agent_class=None, env_class=None, agent_args=None, env_args=None):
        return run_local_ace_rl_task(
            self,
            config=config,
            ace_defaults=ACE_DEFAULTS,
            trainer_class_path=STANDARD_TRAINER,
            default_agent_class_path=APPWORLD_AGENT,
            default_env_class_path=APPWORLD_ENV,
            agent_class=agent_class,
            env_class=env_class,
            agent_args=agent_args,
            env_args=env_args,
            log_prefix="AppWorldLocalACERLTaskRunner",
        )


@ray.remote(num_cpus=1)
class AppWorldLocalACERLIterativeTaskRunner(TaskRunner):
    def run(self, config, agent_class=None, env_class=None, agent_args=None, env_args=None):
        return run_local_ace_rl_task(
            self,
            config=config,
            ace_defaults=ACE_DEFAULTS,
            trainer_class_path=ITERATIVE_TRAINER,
            default_agent_class_path=APPWORLD_AGENT,
            default_env_class_path=APPWORLD_ENV,
            agent_class=agent_class,
            env_class=env_class,
            agent_args=agent_args,
            env_args=env_args,
            log_prefix="AppWorldLocalACERLIterativeTaskRunner",
        )


@ray.remote(num_cpus=1)
class AppWorldLocalACERLDelayedJointTaskRunner(TaskRunner):
    def run(self, config, agent_class=None, env_class=None, agent_args=None, env_args=None):
        return run_local_ace_rl_task(
            self,
            config=config,
            ace_defaults=ACE_DEFAULTS,
            trainer_class_path=DELAYED_JOINT_TRAINER,
            default_agent_class_path=APPWORLD_AGENT,
            default_env_class_path=APPWORLD_ENV,
            agent_class=agent_class,
            env_class=env_class,
            agent_args=agent_args,
            env_args=env_args,
            log_prefix="AppWorldLocalACERLDelayedJointTaskRunner",
        )


@ray.remote(num_cpus=1)
class AppWorldLocalACEJointPPOTaskRunner(TaskRunner):
    def run(self, config, agent_class=None, env_class=None, agent_args=None, env_args=None):
        return run_local_ace_rl_task(
            self,
            config=config,
            ace_defaults=ACE_DEFAULTS,
            trainer_class_path=JOINT_PPO_TRAINER,
            default_agent_class_path=APPWORLD_AGENT,
            default_env_class_path=APPWORLD_ENV,
            agent_class=agent_class,
            env_class=env_class,
            agent_args=agent_args,
            env_args=env_args,
            log_prefix="AppWorldLocalACEJointPPOTaskRunner",
        )


def train_appworld_local_ace_rl(config, *, task_runner_cls):
    from rllm.agents.appworld_react_agents_ace import AppWorldReactAgentACE
    from rllm.environments.appworld.appworld_env import AppWorldEnv

    if not DatasetRegistry.dataset_exists("appworld", "train"):
        print("AppWorld datasets not found. Preparing datasets...")
        prepare_appworld_data()

    train_dataset = DatasetRegistry.load_dataset("appworld", "train")
    val_dataset = DatasetRegistry.load_dataset("appworld", "test")
    if train_dataset is None or val_dataset is None:
        raise ValueError("Failed to load AppWorld datasets. Run: python examples/appworld/prepare_appworld_data.py")

    print(f"Train verl path: {train_dataset.get_verl_data_path()}")
    print(f"Val verl path: {val_dataset.get_verl_data_path()}")
    print(f"Train examples: {len(train_dataset.get_data())}")
    print(f"Val examples: {len(val_dataset.get_data())}")

    trainer = LocalACERLAgentTrainer(
        agent_class=AppWorldReactAgentACE,
        env_class=AppWorldEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_runner_cls=task_runner_cls,
    )
    trainer.train()
