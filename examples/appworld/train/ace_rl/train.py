"""Train AppWorld ACE-RL with standard, iterative, or delayed-joint scheduling."""

from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from examples.appworld.train.ace_rl.training_entrypoints import (
    AppWorldLocalACEJointPPOTaskRunner,
    AppWorldLocalACERLDelayedJointTaskRunner,
    AppWorldLocalACERLIterativeTaskRunner,
    AppWorldLocalACERLTaskRunner,
    train_appworld_local_ace_rl,
)


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    mode = OmegaConf.select(config, "rllm.ace.training_mode", default="standard")
    runners = {
        "standard": AppWorldLocalACERLTaskRunner,
        "iterative": AppWorldLocalACERLIterativeTaskRunner,
        "delayed_joint": AppWorldLocalACERLDelayedJointTaskRunner,
        "joint_ppo": AppWorldLocalACEJointPPOTaskRunner,
    }
    if mode not in runners:
        raise ValueError(f"Unsupported AppWorld ACE training mode: {mode}")
    train_appworld_local_ace_rl(config, task_runner_cls=runners[mode])


if __name__ == "__main__":
    main()
