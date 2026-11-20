"""Prompt updater contract from the pinned Microsoft ERL release (MIT).

Reference: examples/erl/erl_flow.py and examples/erl_hotpot/erl_hotpot_flow.py
at 20cd6460753973b9a5834769e2feed02736f44c6. The generic updater instruction
and extraction are verbatim; state rendering substitutes benchmark-neutral
task/trajectory fields for the Hotpot-specific question/answer/tool fields.
"""

from __future__ import annotations

import re

DEFAULT_UPDATER_SYSTEM_PROMPT = (
    "You are an expert prompt updater. Analyze recent trajectories, rewards, and feedback to improve the solver system prompt. Return ONLY the revised prompt wrapped in <prompt>...</prompt> tags."
)

_PROMPT_PATTERN = re.compile(r"<prompt>(.*?)</prompt>", flags=re.IGNORECASE | re.DOTALL)


def extract_prompt_from_response(text: str) -> str | None:
    """Extract a revised prompt from updater output."""
    match = _PROMPT_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def build_updater_state(current_prompt: str, trajectory: dict, *, correct: bool) -> str:
    """One-example state, matching the release's default episode batch_size=1.

    The complete observed conversation is retained. Input length is checked at
    the generation boundary, never silently summarized or truncated here.
    """
    reward = float(trajectory["trajectory_reward"])
    lines = [
        "## Current System Prompt",
        current_prompt.strip(),
        "",
        "## Batch Metrics",
        f"- avg_first_reward: {reward:.4f}",
        f"- first_success_rate: {float(correct):.4f}",
        "- batch_size: 1.0000",
        "",
        "## Recent Attempts",
        "### Attempt 1",
        "Trajectory:",
    ]
    for index, message in enumerate(trajectory["chat_completions"], start=1):
        content = message.get("raw_completion") or message.get("content") or "<no content>"
        lines.append(f"- Turn {index} [{message.get('role', 'unknown')}]: {content}")
        for call in message.get("tool_calls") or []:
            lines.append(f"  tool_call: {call}")
    status = "CORRECT" if correct else "INCORRECT"
    lines.extend(
        [
            f"Reward: {reward:.4f} | Correct: {correct}",
            f"Feedback: {status} | reward={reward:.2f} | termination={trajectory.get('termination_reason', 'unknown')}",
            "",
            "Think step by step and provide an improved system prompt enclosed in <prompt>...</prompt> tags.",
        ]
    )
    return "\n".join(lines)
