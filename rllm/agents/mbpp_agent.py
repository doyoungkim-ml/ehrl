from __future__ import annotations

import copy
import logging
import re
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory, make_assistant_message, resolve_model_response_content

logger = logging.getLogger(__name__)


def truncate_string(value: str, max_length: int = 300) -> str:
    """Truncate a string for prompt feedback without dropping the ending context."""
    if len(value) <= max_length:
        return value
    half = max_length // 2
    return value[:half] + " ... (truncated) ... " + value[-half:]


class MBPPAgent(BaseAgent):
    """Multi-turn agent for MBPP code-generation tasks."""

    SYSTEM_PROMPT = """You are an expert Python programmer. Your task is to write Python code that solves the given problem.

Instructions:
1. Read the problem description carefully
2. Write clean, correct Python code that solves the problem
3. Your code will be tested against test cases
4. If tests fail, analyze the error messages and revise your code

Output Format:
- Show your reasoning process first
- Then provide your Python code in a markdown code block:
```python
# Your code here
```

Important:
- Make sure your function names match what the tests expect
- Handle edge cases properly
- Keep your code simple and readable
"""

    def __init__(
        self,
        max_steps: int | None = None,
        accumulate_thinking: bool = True,
        accumulate_history: bool = True,
        max_test_display: int = 3,
        system_prompt: str | None = None,
    ):
        if system_prompt is not None:
            self.SYSTEM_PROMPT = system_prompt
        self._trajectory = Trajectory()
        self.messages: list[dict[str, str]] = []
        self.step = 0
        self.accumulate_thinking = accumulate_thinking
        self.max_steps = max_steps
        self.accumulate_history = accumulate_history
        self.max_test_display = max_test_display
        self.current_observation: Any = None
        self.reset()

    def _format_test_results(self, test_results: list[dict[str, Any]]) -> str:
        if not test_results:
            return "No test results available."

        passed = [result for result in test_results if result.get("passed", False)]
        failed = [result for result in test_results if not result.get("passed", False)]

        lines: list[str] = []

        if passed:
            lines.append(f"Passed: {len(passed)}/{len(test_results)} tests")

        if failed:
            lines.append(f"\nFailed tests ({len(failed)}):")
            for index, test in enumerate(failed[: self.max_test_display], start=1):
                lines.append(f"\n  Test {index}: {truncate_string(test.get('test', 'Unknown test'), 200)}")
                if test.get("error"):
                    lines.append(f"  Error: {truncate_string(test['error'], 300)}")

            if len(failed) > self.max_test_display:
                remaining = len(failed) - self.max_test_display
                lines.append(f"\n  ... and {remaining} more failed test(s)")

        return "\n".join(lines)

    def _format_observation(self, observation: Any) -> str:
        if not isinstance(observation, dict):
            return str(observation)

        lines: list[str] = []

        if self.step == 0:
            if "task_description" in observation:
                lines.append(f"Problem:\n{observation['task_description']}")
            if "expected_function_name" in observation:
                lines.append(f"\nYour function must be named: `{observation['expected_function_name']}`")
            if "instructions" in observation:
                lines.append(f"\n{observation['instructions']}")
            if "num_tests" in observation:
                lines.append(f"\nNumber of test cases: {observation['num_tests']}")
        else:
            if "test_results" in observation:
                lines.append("Test Results:")
                lines.append(self._format_test_results(observation["test_results"]))
            if "result" in observation:
                lines.append(f"\n{observation['result']}")

        if "error" in observation:
            lines.append(f"\nError: {observation['error']}")

        return "\n".join(lines)

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs,
    ):
        if self._trajectory.steps:
            current_step = self._trajectory.steps[-1]
            current_step.reward = reward
            current_step.done = done
            current_step.info = info

        observation_text = self._format_observation(observation)
        if self.step == 0:
            user_content = observation_text
        else:
            user_content = f"Turn {self.step}:\n{observation_text}"

        if self.max_steps is not None:
            remaining = self.max_steps - self.step
            if remaining > 0:
                user_content += f"\n\nRemaining attempts: {remaining}"

        # The exact rendered reply, including the terminal one, for feedback_v1 ACE history.
        if self._trajectory.steps:
            self._trajectory.steps[-1].next_observation = user_content
        self.messages.append({"role": "user", "content": user_content})
        self.current_observation = observation_text

    def update_from_model(self, response: str, **kwargs) -> Action:
        model_output = kwargs.get("model_output")

        # Prefer parser-filtered content for visible/action text so terminal
        # chat tokens do not leak into checkpointed content or executable code.
        # If the parser content is empty during an incomplete Qwen thinking turn,
        # fall back to legacy raw text while still stripping terminal tokens.
        content = resolve_model_response_content(
            response,
            model_output,
            strip_thinking=not self.accumulate_thinking,
            prefer_empty_model_content=False,
        )

        thought, code = self._parse_model_response(content)

        # Keep sampled text even when reasoning accumulation is disabled. The
        # chat parser controls history visibility; actions use filtered content.
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
            action=code,
            model_response=content,
            observation=self.current_observation,
            model_output=model_output,
        )
        self._trajectory.steps.append(new_step)
        self.step += 1
        return Action(action=content)

    def _parse_model_response(self, response: str) -> tuple[str, str]:
        thought = response
        code = ""

        matches = re.findall(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
        if matches:
            code = matches[-1].strip()
            last_block_pattern = r"```(?:python)?\n?" + re.escape(code) + r"```"
            match = re.search(last_block_pattern, response, re.DOTALL)
            if match:
                thought = response[: match.start()].strip()

        return thought, code

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
        self.current_observation = None
