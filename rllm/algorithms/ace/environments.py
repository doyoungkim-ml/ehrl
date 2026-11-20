"""Single-turn ACE roles, preserving their existing trajectory metadata."""

from rllm.environments.base.prompt_response_env import SingleTurnEnv


class _ACERoleEnv(SingleTurnEnv):
    def reset(self, task=None):
        observation, info = super().reset(task)
        observation["num_tests"] = 0
        return observation, info

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["all_passed"] = False
        return observation, reward, done, info


class SingleTurnReflectorEnv(_ACERoleEnv):
    """Provide a reflection prompt; the controller assigns its evaluation reward."""


class SingleTurnCuratorEnv(_ACERoleEnv):
    """Provide a curation prompt; the controller assigns its evaluation reward."""
