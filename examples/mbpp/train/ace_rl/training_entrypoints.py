"""MBPP assets and controller binding for ACE-RL."""

from pathlib import Path

import ray
from omegaconf import OmegaConf

from rllm.algorithms.ace.entrypoints import run_local_ace_rl_task
from rllm.trainer.verl.train_agent_ppo import TaskRunner

_BENCHMARK_DIR = Path(__file__).resolve().parent
ACE_DEFAULTS = {
    "reflector_system_prompt": "You are an expert Python coding analyst. Respond with JSON only.",
    "curator_system_prompt": "You are a master curator of knowledge in Python programming. Respond with JSON only.",
    "prompts_dir": str(_BENCHMARK_DIR / "prompts"),
    "initial_playbook": str(_BENCHMARK_DIR / "playbooks" / "initial_playbook.txt"),
    "reflector_adapter_class": "examples.mbpp.train.ace_rl.mbpp_adapter.MBPPReflectorAdapter",
}
TRAINERS = {
    "standard": "rllm.algorithms.ace.trainer.LocalACERLAgentPPOTrainer",
    "iterative": "rllm.algorithms.ace.trainer_iterative.LocalACERLIterativeAgentPPOTrainer",
    "delayed_joint": "rllm.algorithms.ace.trainer_delayed_joint.LocalACERLDelayedJointAgentPPOTrainer",
    "joint_ppo": "rllm.algorithms.ppo.trainer_joint.LocalACEJointPPOTrainer",
}


def get_trainer_class_path(config):
    mode = OmegaConf.select(config, "rllm.ace.training_mode", default="standard")
    if mode not in TRAINERS:
        raise ValueError(f"Unsupported ACE training_mode={mode!r}; choose {', '.join(TRAINERS)}")
    return TRAINERS[mode]


@ray.remote(num_cpus=1)
class LocalACERLTaskRunner(TaskRunner):
    def run(self, config, agent_class=None, env_class=None, agent_args=None, env_args=None):
        default_env_path = "rllm.environments.mbpp.mbpp.MBPPEnv"
        return run_local_ace_rl_task(
            self,
            config=config,
            trainer_class_path=get_trainer_class_path(config),
            default_agent_class_path="rllm.agents.mbpp_agent.MBPPAgent",
            default_env_class_path=default_env_path,
            agent_class=agent_class,
            env_class=env_class,
            agent_args=agent_args,
            env_args=env_args,
            ace_defaults=ACE_DEFAULTS,
            log_prefix="MBPPLocalACERLTaskRunner",
        )
