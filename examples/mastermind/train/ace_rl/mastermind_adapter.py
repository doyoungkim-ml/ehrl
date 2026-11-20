"""Mastermind reflector evidence: one row per guess, and partial credit is not a solved game.

Training and evaluation give the reflector the same table. Evaluation steps carry the
parsed guess and the environment's feedback in step.info. Training rebuilds steps from
the chat, so the guess is read from the reply and its feedback from the environment
message that follows it.

The opt-in feedback_v1 protocol keeps this table. It already pairs every guess,
including the last one, with its environment feedback. The raw terminal reply
also states the secret code, which neither role may pass to the next attempt.
The table is bounded by the turn limit and the reply cap, so it has no token budget.
"""

import re

from rllm.agents.mastermind_agent import MastermindAgent
from rllm.algorithms.ace.context.prompt_adapter.base import DefaultReflectorAdapter

_LAST_GUESS = re.compile(r"Last guess: ([0-9 ]+)\n\s*Exact matches: (\d+)\n\s*Value matches: (\d+)")
_ERROR = re.compile(r"^Error: (.+)$", re.MULTILINE)


def _feedback(secret: list[int], guess: list[int]) -> tuple[int, int]:
    exact = sum(s == g for s, g in zip(secret, guess, strict=True))
    common = sum(min(secret.count(d), guess.count(d)) for d in set(guess))
    return exact, common - exact


def _parse_guess(text: str) -> list[int] | None:
    text = text.replace(",", " ").strip()
    parts = text.split() if " " in text else list(text)
    if not parts or not all(part.isdigit() for part in parts):
        return None
    return [int(part) for part in parts]


def _won(steps) -> bool:
    return any((getattr(step, "info", None) or {}).get("won") or (getattr(step, "info", None) or {}).get("all_passed") for step in steps)


class MastermindReflectorAdapter(DefaultReflectorAdapter):
    def __init__(self, max_turns=20, reply_max_chars=400, handoff_protocol="legacy", tokenizer=None, max_history_tokens=None):
        if handoff_protocol not in {"legacy", "feedback_v1"}:
            raise ValueError(f"Unsupported handoff_protocol: {handoff_protocol}")
        if max_history_tokens is not None:
            # Head/tail truncation could drop the last guess rows, which carry the terminal feedback.
            raise ValueError("Mastermind keeps its complete guess table; max_history_tokens is not supported")
        self.max_turns = max_turns
        self.reply_max_chars = reply_max_chars
        self.handoff_protocol = handoff_protocol
        self.tokenizer = tokenizer
        self.max_history_tokens = max_history_tokens

    def get_template_variables(self, trajectory, reward, playbook, **kwargs):
        steps = getattr(trajectory, "steps", []) or []
        won = _won(steps)
        outcome = "SUCCESS" if won else "FAILED"
        summary = f"Result: {outcome}\nTurns: {len(steps)}\nBest guess reward (0 to 1): {reward:g}\n"
        summary += "Partial digit matches earn reward even when the secret code was not solved."
        return {
            "outcome": outcome,
            "trajectory_summary": summary,
            "conversation_history": self.format_conversation_history(trajectory),
            "playbook": playbook,
            "failure_info": "" if won else "The agent did not guess the complete secret code.",
        }

    def finalize_reflection(self, result: dict, traj_data: dict) -> dict:
        """The environment knows whether the code was solved; a small reflector often does not."""
        steps = getattr(traj_data.get("trajectory"), "steps", []) or []
        return {**result, "outcome": "SUCCESS" if _won(steps) else "FAILED"}

    @staticmethod
    def _step_feedback(step, next_step, code_length, won_here):
        """Return ("error", reason), ("scored", (exact, value)), or ("unknown", None)."""
        info = getattr(step, "info", None) or {}
        if "guess_error" in info:
            return "error", info["guess_error"]
        if "exact_matches" in info:
            return "scored", (int(info["exact_matches"]), int(info["value_matches"]))
        following = getattr(step, "next_observation", None)
        if not following and next_step is not None:
            following = getattr(next_step, "observation", None)
        following = str(following or "")
        error = _ERROR.search(following)
        if error:
            return "error", error.group(1).strip()
        match = _LAST_GUESS.search(following)
        if match:
            return "scored", (int(match.group(2)), int(match.group(3)))
        if won_here and code_length:
            return "scored", (int(code_length), 0)
        return "unknown", None

    def format_conversation_history(self, trajectory) -> str:
        steps = getattr(trajectory, "steps", []) or []
        if not steps:
            return "(No steps recorded in trajectory)"
        task = getattr(trajectory, "task", None) or {}
        code_length = task.get("code_length") if isinstance(task, dict) else None
        num_digits = task.get("num_digits") if isinstance(task, dict) else None

        rows, replies = [], []
        scored: list[tuple[list[int], int, int]] = []
        first_turn: dict[tuple[int, ...], int] = {}
        counts = {"valid": 0, "rejected": 0, "repeated": 0, "inconsistent": 0}
        for turn, step in enumerate(steps, start=1):
            raw = str(getattr(getattr(step, "action", None), "action", getattr(step, "action", "")) or "")
            if "```" in raw:
                raw = MastermindAgent._parse_model_response(MastermindAgent, raw)[1]
            guess = _parse_guess(raw)
            shown = raw if raw else "(no guess line)"
            if len(shown) > 60:
                shown = shown[:60] + "..."
            next_step = steps[turn] if turn < len(steps) else None
            info = getattr(step, "info", None) or {}
            kind, detail = self._step_feedback(step, next_step, code_length, bool(info.get("won") or info.get("all_passed")))
            if kind == "error" or guess is None or (code_length and len(guess) != code_length):
                counts["rejected"] += 1
                reason = detail if kind == "error" else "not a guess the environment accepts"
                rows.append(f"Turn {turn}: {shown} -> REJECTED ({reason}) reward=0")
            else:
                notes = []
                key = tuple(guess)
                if key in first_turn:
                    counts["repeated"] += 1
                    notes.append(f"REPEATS turn {first_turn[key]}")
                else:
                    clashes = [str(first_turn[tuple(g)]) for g, e, v in scored if _feedback(guess, g) != (e, v)]
                    if clashes:
                        counts["inconsistent"] += 1
                        notes.append("contradicts feedback of turn " + ", ".join(clashes[:5]))
                    first_turn[key] = turn
                    if kind == "scored":
                        scored.append((guess, *detail))
                counts["valid"] += 1
                reward = getattr(step, "reward", None)
                reward_text = f" reward={reward:.3g}" if isinstance(reward, int | float) else ""
                suffix = f" [{'; '.join(notes)}]" if notes else ""
                result = f"exact={detail[0]}, value={detail[1]}" if kind == "scored" else "feedback not shown"
                rows.append(f"Turn {turn}: {' '.join(map(str, guess))} -> {result}{reward_text}{suffix}")

            reply = str(getattr(step, "model_response", None) or getattr(step, "thought", None) or "").strip()
            if len(reply) > self.reply_max_chars:
                reply = reply[: self.reply_max_chars] + " ...[cut]"
            replies.append(f"--- Turn {turn} reply ---\n{reply or '(empty reply)'}")

        header = []
        if code_length:
            header.append(f"Code length: {code_length}")
        if num_digits:
            header.append(f"Valid digits: 0-{num_digits - 1}")
        header.append(
            f"Accepted guesses: {counts['valid']} ({len(first_turn)} distinct), rejected: {counts['rejected']}, "
            f"repeats of an earlier guess: {counts['repeated']}, guesses contradicting earlier feedback: {counts['inconsistent']}"
        )
        return "\n".join(header + ["", "GUESS TABLE (environment feedback, always correct):", *rows, "", "GENERATOR REPLIES:", *replies])
