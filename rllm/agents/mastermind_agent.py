from __future__ import annotations

import copy
import logging
import re
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory, make_assistant_message, resolve_model_response_content

logger = logging.getLogger(__name__)


class MastermindAgent(BaseAgent):
    """Agent for the multi-turn Mastermind code-breaking game."""

    SYSTEM_PROMPT: str = """You are a helpful assistant playing Mastermind, a code-breaking game.

Mastermind Quick Guide:
Goal: Guess the secret code within the allowed number of turns.

Rules:
1. The secret code consists of a sequence of digits.
2. Each digit is between 0 and num_digits-1.
3. After each guess, you receive feedback:
   - exact_matches: Number of digits with correct value AND correct position
   - value_matches: Number of digits with correct value but WRONG position

Rewards:
- Solve the complete code: reward 1
- Each exact match earns 1/code_length
- Each correct value in a wrong position earns 0.5/code_length

Strategy Tips:
1. Start with a guess that has diverse digits to gather information.
2. Use the feedback to eliminate impossible combinations.
3. Track which digits appear in the code based on value_matches.
4. Narrow down positions based on exact_matches.

You will be provided the current game state including your guess history.
Show your reasoning process, then provide your guess in ``` ``` format.
Your guess should be space-separated digits, e.g., ```0 1 2 3```.
"""

    def __init__(
        self,
        max_steps: int | None = None,
        use_accumulate_thinking: bool = True,
        use_accumulate_history: bool = True,
        system_prompt: str | None = None,
    ):
        if system_prompt is not None:
            self.SYSTEM_PROMPT = system_prompt
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self.step = 0
        self.accumulate_thinking = use_accumulate_thinking
        self.max_steps = max_steps
        self.accumulate_history = use_accumulate_history
        self.current_observation: Any = None
        self.reset()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if self._trajectory.steps:
            current_step = self._trajectory.steps[-1]
            current_step.reward = reward
            current_step.done = done
            current_step.info = info

        observation_text = self._format_observation(observation)
        user_content = f"Turn {self.step + 1}:\n{observation_text}"

        if isinstance(observation, dict) and "error" in observation:
            user_content += f"\n\nError: {observation['error']}\nPlease provide a valid guess."

        if self.max_steps is not None and self.max_steps - self.step > 0:
            user_content += f"\n\nRemaining turns: {self.max_steps - self.step}"

        self.messages.append({"role": "user", "content": user_content})
        self.current_observation = observation_text

    def _format_observation(self, observation: Any) -> str:
        if not isinstance(observation, dict):
            return str(observation)

        lines: list[str] = []
        if self.step == 0 and "instructions" in observation:
            lines.append(f"Instructions: {observation['instructions']}")
            lines.append("")

        if "code_length" in observation:
            lines.append(f"Code length: {observation['code_length']}")
        if "num_digits" in observation:
            lines.append(f"Valid digits: 0-{observation['num_digits'] - 1}")

        if "history" in observation and observation["history"]:
            lines.append("\nGuess History:")
            for entry in observation["history"]:
                lines.append(f"  Turn {entry['turn']}: {entry['guess']} -> exact={entry['exact_matches']}, value={entry['value_matches']}")

        if "last_guess" in observation:
            last_guess = observation["last_guess"]
            lines.append(f"\nLast guess: {last_guess['guess']}")
            lines.append(f"  Exact matches: {last_guess['exact_matches']}")
            lines.append(f"  Value matches: {last_guess['value_matches']}")
            lines.append(f"  Reward: {last_guess['reward']}")

        if "result" in observation:
            lines.append(f"\n{observation['result']}")
        if "secret_code" in observation:
            lines.append(f"Secret code was: {observation['secret_code']}")

        return "\n".join(lines)

    def update_from_model(self, response: str, **kwargs) -> Action:
        model_output = kwargs.get("model_output")
        content = resolve_model_response_content(
            response,
            model_output,
            strip_thinking=not self.accumulate_thinking,
        )

        thought, action_str = self._parse_model_response(content)
        self.messages.append(
            make_assistant_message(
                content,
                model_output,
                preserve_raw_completion=True,
            )
        )

        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought=thought,
            action=action_str,
            model_response=content,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)
        self.step += 1
        return Action(action=action_str)

    # A fenced guess, optionally after a language tag. Small models often write
    # "```python\n```0 1 2```", where the first fence pair holds only the tag.
    _GUESS_BLOCK = re.compile(r"```[A-Za-z]*[ \t]*\n?[ \t]*([0-9][0-9 ,]*?)[ \t]*\n?[ \t]*```")

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        thought = response
        action_str = ""
        guesses = list(self._GUESS_BLOCK.finditer(response))
        if guesses:
            last = guesses[-1]
            return response[: last.start()].strip(), last.group(1).strip()
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        if matches:
            last_match_content = matches[-1].strip()
            last_match_index = response.rfind(f"```{last_match_content}```")
            if last_match_index != -1:
                thought = response[:last_match_index].strip()
            action_str = last_match_content.strip()
            if "\n" in action_str:
                first_line = action_str.split("\n")[0].strip().lower()
                if first_line in ["python", "text", "guess", ""]:
                    action_str = "\n".join(action_str.split("\n")[1:]).strip()
        return thought, action_str

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        if self.accumulate_history:
            return self.messages
        if len(self.messages) <= 1:
            return self.messages
        return [self.messages[0], self.messages[-1]]

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self.step = 0
