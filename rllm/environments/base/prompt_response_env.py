"""Prompt-response environment for generations with externally assigned rewards."""

from typing import Any

from rllm.environments.base.base_env import BaseEnv


class SingleTurnEnv(BaseEnv):
    """Provide one prompt, then finish with a zero reward for the caller to score."""

    def __init__(self, prompt_text: str = "", task_id: str = "", **kwargs):
        super().__init__()
        self.prompt_text = prompt_text
        self.task_id = task_id
        self.done = False

    def reset(self, task=None):
        self.done = False
        observation = {
            "task_id": self.task_id,
            "task_description": self.prompt_text,
            "done": False,
        }
        return observation, {"task_id": self.task_id}

    def step(self, action: Any):
        self.done = True
        return {}, 0.0, True, {"task_id": self.task_id}

    @classmethod
    def from_dict(cls, env_args: dict) -> "SingleTurnEnv":
        return cls(
            prompt_text=env_args.get("prompt_text", ""),
            task_id=env_args.get("task_id", ""),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
