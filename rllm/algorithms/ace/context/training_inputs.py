"""Pure input helpers for optional training-parity evaluation.

These mirror the trainer's formatting methods without changing training.
"""

import json
import re
from types import SimpleNamespace
from typing import Any

from .utils import truncate_conversation_history


def steps_from_training_chat(chat_completions: list[dict], step_rewards: list, *, all_passed: bool = False) -> list[SimpleNamespace]:
    """Reconstruct the Step-like objects used by ACE training reflectors.

    Only message content participates. Separate reasoning and raw_completion
    metadata remains excluded, just as in the trainer. Assistant content is
    repeated as ACTION without parsing code. Each user reply becomes the
    previous step's next_observation; a terminal user reply adds no step.
    """
    if not isinstance(chat_completions, list):
        return []
    steps = []
    pending_observation = None
    for message in chat_completions:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        content = message.get("content", "")
        if role == "user":
            # The environment's reply to the previous action, including a terminal reply.
            if steps and steps[-1].next_observation is None:
                steps[-1].next_observation = content
            pending_observation = content
            continue
        if role != "assistant":
            continue
        step_idx = len(steps)
        info = {}
        if all_passed and step_idx == max(0, len(step_rewards) - 1):
            info["all_passed"] = True
        steps.append(
            SimpleNamespace(
                observation=pending_observation,
                next_observation=None,
                thought="",
                action=content,
                model_response=content,
                reward=step_rewards[step_idx] if step_idx < len(step_rewards) else None,
                done=False,
                info=info,
            )
        )
        pending_observation = None
    return steps


def render_training_template(template: str, variables: dict) -> str:
    """Render both spaced and unspaced placeholders as the trainer does."""
    for key, value in variables.items():
        template = template.replace("{{ " + key + " }}", str(value))
        template = template.replace("{{" + key + "}}", str(value))
    return template


def fit_training_reflector_prompt(
    template: str,
    variables: dict,
    *,
    tokenizer: Any,
    chat_parser: Any,
    system_prompt: str,
    max_prompt_tokens: int,
) -> str:
    """Apply the trainer's final chat-template-aware history fitting guard.

    The adapter's history cap is separate. This second guard counts the role
    system message, template, task fields, playbook, and generation prefix.
    Only conversation history may shrink; oversized fixed fields fail.
    """
    variables = dict(variables)

    def render() -> str:
        return render_training_template(template, variables)

    def token_count(user_prompt: str) -> int:
        if chat_parser is not None:
            user_prompt = chat_parser.parse(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                add_generation_prompt=True,
                is_first_msg=True,
                accumulate_reasoning=False,
            )
        return len(tokenizer.encode(user_prompt, add_special_tokens=False))

    prompt = render()
    prompt_tokens = token_count(prompt)
    if prompt_tokens <= max_prompt_tokens:
        return prompt
    history = variables.get("conversation_history", "")
    if not history:
        raise ValueError(f"Reflector prompt exceeds its token budget without conversation history to truncate: {prompt_tokens} > {max_prompt_tokens}")
    variables["conversation_history"] = ""
    fixed_prompt_tokens = token_count(render())
    if fixed_prompt_tokens > max_prompt_tokens:
        raise ValueError(f"Reflector prompt's non-history fields exceed its token budget: {fixed_prompt_tokens} > {max_prompt_tokens}")
    history_budget = max_prompt_tokens - fixed_prompt_tokens
    while True:
        variables["conversation_history"] = truncate_conversation_history(history, max_tokens=history_budget, tokenizer=tokenizer)
        prompt = render()
        prompt_tokens = token_count(prompt)
        if prompt_tokens <= max_prompt_tokens:
            return prompt
        history_budget -= max(1, prompt_tokens - max_prompt_tokens)


def concatenate_training_insights(insights: list[dict]) -> str:
    """Match the trainer's curator input, including reasoning and failures."""
    merged = []
    for index, insight in enumerate(insights):
        reflection = insight.get("reflection", {})
        if not isinstance(reflection, dict) or not reflection:
            continue
        merged.append(f"--- Insight {index + 1} (Task {insight.get('task_idx', index)}) ---")
        merged.append(f"Outcome: {reflection.get('outcome', 'UNKNOWN')}")
        for field, label in (
            ("reasoning", "Reasoning"),
            ("key_insight", "Key Insight"),
            ("failure_reason", "Failure Reason"),
            ("correct_approach", "Correct Approach"),
        ):
            if reflection.get(field):
                merged.append(f"{label}: {reflection[field]}")
        merged.append("")
    return "\n".join(merged)


def parse_training_json_response(text: str):
    """Match the trainer's JSON extraction, including code-block fallbacks."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    patterns = [
        r"```json\s*\n?(.*?)```",
        r"```\s*\n?(.*?)```",
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, re.DOTALL):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    return {}


def coerce_training_reflection_payload(payload) -> dict:
    """Retain the trainer's fields without legacy eval normalization.

    Training unwraps a nested reflection object and accepts the first object in
    a list. Invalid or empty output contributes no curator insight.
    """
    if isinstance(payload, dict):
        nested = payload.get("reflection")
        return nested if isinstance(nested, dict) else payload
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                nested = item.get("reflection")
                return nested if isinstance(nested, dict) else item
    return {}
