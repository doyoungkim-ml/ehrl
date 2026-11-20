"""Bind Mastermind datasets and agents to the shared ACE-RL trainer."""

from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

from rllm.agents.mastermind_agent import MastermindAgent
from rllm.algorithms.ace.entrypoints import LocalACERLAgentTrainer, run_local_ace_rl_task
from rllm.data import DatasetRegistry
from rllm.data.selection import configure_training_dataset_paths
from rllm.environments.mastermind.mastermind import MastermindEnv
from rllm.trainer.verl.train_agent_ppo import TaskRunner

_ASSETS = Path(__file__).resolve().parent
TRAINERS = {
    "standard": "rllm.algorithms.ace.trainer.LocalACERLAgentPPOTrainer",
    "iterative": "rllm.algorithms.ace.trainer_iterative.LocalACERLIterativeAgentPPOTrainer",
    "delayed_joint": "rllm.algorithms.ace.trainer_delayed_joint.LocalACERLDelayedJointAgentPPOTrainer",
    "joint_ppo": "rllm.algorithms.ppo.trainer_joint.LocalACEJointPPOTrainer",
}
ACE_DEFAULTS = {
    "prompts_dir": str(_ASSETS / "prompts"),
    "initial_playbook": str(_ASSETS / "playbooks" / "initial_playbook.txt"),
    "reflector_adapter_class": "examples.mastermind.train.ace_rl.mastermind_adapter.MastermindReflectorAdapter",
    "reflector_system_prompt": "You analyze Mastermind guesses and feedback. Respond with JSON only.",
    "curator_system_prompt": "You curate concise Mastermind reasoning hints. Respond with JSON only.",
}


def get_trainer_class_path(config):
    mode = OmegaConf.select(config, "rllm.ace.training_mode", default="standard")
    if mode not in TRAINERS:
        raise ValueError(f"Unsupported ACE training_mode={mode!r}; choose {', '.join(TRAINERS)}")
    return TRAINERS[mode]


@ray.remote(num_cpus=1)
class MastermindACETaskRunner(TaskRunner):
    def run(self, config, agent_class=None, env_class=None, agent_args=None, env_args=None):
        return run_local_ace_rl_task(
            self,
            config=config,
            trainer_class_path=get_trainer_class_path(config),
            default_agent_class_path="rllm.agents.mastermind_agent.MastermindAgent",
            default_env_class_path="rllm.environments.mastermind.mastermind.MastermindEnv",
            agent_class=agent_class,
            env_class=env_class,
            agent_args=agent_args,
            env_args=env_args,
            ace_defaults=ACE_DEFAULTS,
            log_prefix="MastermindACE",
        )


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Reject unsupported modes before dataset preparation or GPU initialization.
    get_trainer_class_path(config)
    needed = [split for split, key in (("train", "train_files"), ("test", "val_files")) if not config.data.get(key)]
    if any(not DatasetRegistry.dataset_exists("mastermind", split) for split in needed):
        from examples.mastermind.prepare_mastermind_data import prepare_mastermind_data

        prepare_mastermind_data()
    train_dataset = DatasetRegistry.load_dataset("mastermind", "train") if "train" in needed else None
    val_dataset = DatasetRegistry.load_dataset("mastermind", "test") if "test" in needed else None
    paths = configure_training_dataset_paths(config, train_dataset=train_dataset, val_dataset=val_dataset, subset_tag="mastermind_harness")
    print(f"Mastermind dataset paths: {paths}")
    LocalACERLAgentTrainer(
        agent_class=MastermindAgent,
        env_class=MastermindEnv,
        config=config,
        task_runner_cls=MastermindACETaskRunner,
    ).train()


if __name__ == "__main__":
    main()
