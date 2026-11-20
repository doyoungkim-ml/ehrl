"""Reusable prompt-response agent for auxiliary RL generations."""

import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class SingleTurnAgent(BaseAgent):
    """Generate one response while preserving the engine's raw token history."""

    def __init__(
        self,
        system_prompt: str = "",
        max_steps: int = 1,
        accumulate_thinking: bool = True,
        accumulate_history: bool = True,
        **kwargs,
    ):
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self._step: int = 0
        self.system_prompt = system_prompt
        self.max_steps = 1  # Always single turn
        self.accumulate_thinking = accumulate_thinking
        self.accumulate_history = accumulate_history
        self.current_observation: Any = None
        self.reset()

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        self._step = 0
        self.current_observation = None

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if self._trajectory.steps:
            cur_step = self._trajectory.steps[-1]
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info = info

        # Format observation as user message
        if isinstance(observation, dict):
            obs_str = observation.get("task_description", str(observation))
        else:
            obs_str = str(observation)

        self.messages.append({"role": "user", "content": obs_str})
        self.current_observation = obs_str

    def update_from_model(self, response: str, **kwargs) -> Action:
        content = response

        if not self.accumulate_thinking:
            _, sep, after = content.partition("</think>")
            if sep:
                content = after

        model_output = kwargs.get("model_output")
        if model_output is not None and self.accumulate_thinking:
            assistant_message = {
                "role": "assistant",
                "content": model_output.content or "",
                "reasoning": model_output.reasoning or "",
                "raw_completion": model_output.text or "",
            }
        else:
            assistant_message = {"role": "assistant", "content": content}
        self.messages.append(assistant_message)

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought="",
            action=content,
            model_response=content,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)
        self._step += 1

        return Action(action=content)

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
