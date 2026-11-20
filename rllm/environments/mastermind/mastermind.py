from __future__ import annotations

import random
from typing import Any

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment


class MastermindEnv(MultiTurnEnvironment):
    """Mastermind code-breaking environment."""

    def __init__(
        self,
        task: dict | None = None,
        code_length: int = 4,
        num_digits: int = 6,
        max_turns: int = 20,
        reward_exact_match: float = 10.0,
        reward_value_match: float = 5.0,
        seed: int | None = None,
        **kwargs,
    ):
        if task is not None:
            code_length = task.get("code_length", code_length)
            num_digits = task.get("num_digits", num_digits)
            max_turns = task.get("max_turns", max_turns)
            reward_exact_match = task.get("reward_exact_match", reward_exact_match)
            reward_value_match = task.get("reward_value_match", reward_value_match)
            seed = task.get("seed", seed)

        super().__init__(task=task, max_turns=max_turns, **kwargs)
        self.code_length = code_length
        self.num_digits = num_digits
        self.reward_exact_match = reward_exact_match
        self.reward_value_match = reward_value_match
        self.seed = seed
        self.rng = random.Random(seed)
        self.secret_code: list[int] = []
        self.guess_history: list[tuple[list[int], int, int]] = []
        self.won = False

    def _generate_secret_code(self) -> list[int]:
        return [self.rng.randint(0, self.num_digits - 1) for _ in range(self.code_length)]

    def _evaluate_guess(self, guess: list[int]) -> tuple[int, int]:
        exact_matches = 0
        secret_remaining = []
        guess_remaining = []

        for i in range(self.code_length):
            if guess[i] == self.secret_code[i]:
                exact_matches += 1
            else:
                secret_remaining.append(self.secret_code[i])
                guess_remaining.append(guess[i])

        value_matches = 0
        secret_counts: dict[int, int] = {}
        for digit in secret_remaining:
            secret_counts[digit] = secret_counts.get(digit, 0) + 1
        for digit in guess_remaining:
            if digit in secret_counts and secret_counts[digit] > 0:
                value_matches += 1
                secret_counts[digit] -= 1
        return exact_matches, value_matches

    def _parse_action(self, action: Any) -> list[int] | None:
        if isinstance(action, list):
            try:
                return [int(x) for x in action]
            except (ValueError, TypeError):
                return None

        if isinstance(action, str):
            action = action.strip()
            if "," in action:
                try:
                    return [int(part.strip()) for part in action.split(",")]
                except ValueError:
                    return None
            if " " in action:
                try:
                    return [int(part) for part in action.split()]
                except ValueError:
                    return None
            try:
                return [int(char) for char in action]
            except ValueError:
                return None

        return None

    def _render_code(self, code: list[int]) -> str:
        return " ".join(str(digit) for digit in code)

    def _render_observation(self) -> dict:
        observation = {
            "turn": self.current_turn,
            "max_turns": self.max_turns,
            "remaining_turns": self.max_turns - self.current_turn,
            "code_length": self.code_length,
            "num_digits": self.num_digits,
            "valid_digits": list(range(self.num_digits)),
            "history": [],
            "won": self.won,
            "done": self.done,
            "instructions": (
                f"Guess the secret code of {self.code_length} digits. "
                f"Each digit is between 0 and {self.num_digits - 1}. "
                "Provide your guess as space-separated digits (e.g., '0 1 2 3'). "
                "Feedback: exact_matches = correct digit AND position, "
                "value_matches = correct digit but wrong position."
            ),
        }

        for turn, (guess, exact_matches, value_matches) in enumerate(self.guess_history, start=1):
            observation["history"].append(
                {
                    "turn": turn,
                    "guess": self._render_code(guess),
                    "exact_matches": exact_matches,
                    "value_matches": value_matches,
                }
            )
        return observation

    def reset(self, task: dict | None = None):
        if task is not None:
            self.task = task
            self.code_length = task.get("code_length", self.code_length)
            self.num_digits = task.get("num_digits", self.num_digits)
            self.max_turns = task.get("max_turns", self.max_turns)
            self.reward_exact_match = task.get("reward_exact_match", self.reward_exact_match)
            self.reward_value_match = task.get("reward_value_match", self.reward_value_match)
            if "seed" in task:
                self.seed = task["seed"]
                self.rng = random.Random(self.seed)

        self.current_turn = 0
        self.done = False
        self.won = False
        self.guess_history = []
        self.history = []
        self.secret_code = self._generate_secret_code()
        return self._render_observation(), {}

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        guess = self._parse_action(action)
        if guess is None:
            return 0.0, {**self._render_observation(), "error": f"Invalid guess format. Please provide {self.code_length} digits."}
        if len(guess) != self.code_length:
            return 0.0, {**self._render_observation(), "error": f"Guess must have exactly {self.code_length} digits. Got {len(guess)}."}
        for index, digit in enumerate(guess):
            if digit < 0 or digit >= self.num_digits:
                return 0.0, {
                    **self._render_observation(),
                    "error": f"Invalid digit {digit} at position {index}. Must be between 0 and {self.num_digits - 1}.",
                }

        exact_matches, value_matches = self._evaluate_guess(guess)
        self.guess_history.append((guess, exact_matches, value_matches))
        # Preserve relative credit while making a solved game worth one,
        # independent of the task's code length.
        if self.reward_exact_match <= 0 or not 0 <= self.reward_value_match <= self.reward_exact_match:
            raise ValueError("Mastermind requires reward_exact_match > 0 and 0 <= reward_value_match <= reward_exact_match")
        reward = ((exact_matches * self.reward_exact_match) + (value_matches * self.reward_value_match)) / (self.code_length * self.reward_exact_match)

        if exact_matches == self.code_length:
            self.won = True
            self.done = True

        observation = self._render_observation()
        observation["last_guess"] = {
            "guess": self._render_code(guess),
            "exact_matches": exact_matches,
            "value_matches": value_matches,
            "reward": reward,
        }

        if self.done or self.current_turn >= self.max_turns - 1:
            observation["secret_code"] = self._render_code(self.secret_code)
            if self.won:
                observation["result"] = f"Congratulations! You guessed the code in {self.current_turn + 1} turns!"
            else:
                observation["result"] = f"Game over. The secret code was: {self._render_code(self.secret_code)}"

        return reward, observation

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        self.history.append(action)
        reward, next_observation = self.get_reward_and_next_obs(self.task or {}, action)
        self.current_turn += 1
        if self.current_turn >= self.max_turns:
            self.done = True
        info = {
            "won": self.won,
            "turns_used": self.current_turn,
            "secret_code": self.secret_code if self.done else None,
        }
        # This guess's own feedback, so trajectory analysis sees the final guess too.
        if "error" in next_observation:
            info["guess_error"] = next_observation["error"]
        elif "last_guess" in next_observation:
            info["exact_matches"] = next_observation["last_guess"]["exact_matches"]
            info["value_matches"] = next_observation["last_guess"]["value_matches"]
        if self.done:
            next_observation["secret_code"] = self._render_code(self.secret_code)
        return next_observation, reward, self.done, info

    @staticmethod
    def from_dict(env_args: dict) -> MastermindEnv:
        return MastermindEnv(
            task=env_args,
            code_length=env_args.get("code_length", 4),
            num_digits=env_args.get("num_digits", 6),
            max_turns=env_args.get("max_turns", 20),
            reward_exact_match=env_args.get("reward_exact_match", 10.0),
            reward_value_match=env_args.get("reward_value_match", 5.0),
            seed=env_args.get("seed"),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
