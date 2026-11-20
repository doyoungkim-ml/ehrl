"""mastermind delayed_joint_ppo: learned critic, GAE and clipped PPO updates."""

import hydra

from examples.mastermind.train.ace_rl.train import main as parent_main
from rllm.algorithms.ppo.configuration import validate_ppo_config


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    validate_ppo_config(config, delayed_joint=True)
    parent_main.__wrapped__(config)


if __name__ == "__main__":
    main()
