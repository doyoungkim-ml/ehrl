"""
==============================================================================
utils.py
==============================================================================

Shared utilities for ACE (Agentic Context Engineering) implementation.

This module provides:
- JSON extraction from LLM responses
- Section slug generation for playbook bullet IDs
- Trajectory formatting for prompts
- Outcome derivation from trajectories

Compatible with FrozenLake prompts and ACE-AppWorld patterns.
"""

import asyncio
import json
import re
from typing import Any

# =============================================================================
# Section Slug Mapping
# =============================================================================

# Maps normalized section names to their slug prefixes (3-5 chars)
# Used for generating bullet IDs like [shr-00001], [cm-00002], etc.
SECTION_SLUG_MAP = {
    # FrozenLake sections
    "strategies_and_hard_rules": "shr",
    "common_mistakes": "cm",
    "grid_patterns": "gp",
    # AppWorld sections (for compatibility)
    "hard_rules": "hr",
    "strategies_and_insights": "si",
    "apis_to_use_for_specific_information": "api",
    "useful_code_snippets_and_templates": "code",
    "code_snippets_and_templates": "code",
    "common_mistakes_and_correct_strategies": "cms",
    "common_mistakes_to_avoid": "err",
    "problem_solving_heuristics_and_workflows": "psw",
    "problem_solving_heuristics": "prob",
    "troubleshooting_and_pitfalls": "ts",
    "others": "misc",
    "meta_strategies": "meta",
    "general": "gen",
}


def get_section_slug(section_name: str) -> str:
    """
    Convert section name to slug format (3-5 chars).

    Used for generating bullet IDs like [shr-00001].

    Args:
        section_name: Raw section name (e.g., "Strategies and Hard Rules")

    Returns:
        Slug string (e.g., "shr")

    Examples:
        >>> get_section_slug("strategies_and_hard_rules")
        'shr'
        >>> get_section_slug("Common Mistakes")
        'cm'
        >>> get_section_slug("VERIFICATION CHECKLIST")
        'vc'
    """
    # Normalize: lowercase, spaces->_, &->and, strip trailing ':'
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")

    # Check known mappings
    if clean_name in SECTION_SLUG_MAP:
        return SECTION_SLUG_MAP[clean_name]

    # Generate slug from first letters of words
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5] if w)


def normalize_section_name(section_name: str) -> str:
    """
    Normalize section name to standard format.

    Args:
        section_name: Raw section name

    Returns:
        Normalized section name (lowercase, underscores)
    """
    return section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")


# =============================================================================
# JSON Extraction
# =============================================================================


def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON object from LLM response.

    Handles multiple formats:
    - Pure JSON response
    - JSON wrapped in ```json ... ``` code blocks
    - JSON embedded in text
    - JSON after <think>...</think> blocks (Qwen3, DeepSeek, etc.)

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON as dict, or empty dict if parsing fails

    Examples:
        >>> extract_json_from_response('{"key": "value"}')
        {'key': 'value'}
        >>> extract_json_from_response('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}
        >>> extract_json_from_response('<think>planning...</think>\\n{"key": "value"}')
        {'key': 'value'}
    """
    if not response or not isinstance(response, str):
        return {}

    response = response.strip()

    # Try 1: Direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try 2: Strip <think>...</think> blocks (handles Qwen3, DeepSeek, etc.), or <thought>...<thought> blocks
    # This MUST come before brace-matching to avoid capturing JSON inside <think>
    think_pattern = r"<(?:think|thought)>[\s\S]*?</(?:think|thought)>"
    response_no_think = re.sub(think_pattern, "", response, flags=re.IGNORECASE).strip()

    # If we removed think blocks, try parsing the remainder
    if response_no_think != response and response_no_think:
        # Try direct parse of remainder
        try:
            return json.loads(response_no_think)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from remainder (handles whitespace, extra text)
        first_brace = response_no_think.find("{")
        last_brace = response_no_think.rfind("}")

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = response_no_think[first_brace : last_brace + 1]
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

    # Try 3: Extract from ```json ... ``` code block
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    code_block_match = re.search(code_block_pattern, response)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 4: Find JSON object in text (greedy match for outermost braces)
    # Only use this if we haven't already tried on response_no_think
    if response_no_think == response:
        first_brace = response.find("{")
        last_brace = response.rfind("}")

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = response[first_brace : last_brace + 1]
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

    # Try 5: Find JSON array in text
    first_bracket = response_no_think.find("[") if response_no_think else response.find("[")
    last_bracket = response_no_think.rfind("]") if response_no_think else response.rfind("]")
    search_text = response_no_think if response_no_think else response

    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        json_candidate = search_text[first_bracket : last_bracket + 1]
        try:
            result = json.loads(json_candidate)
            # Wrap array in dict for consistency
            return {"items": result} if isinstance(result, list) else result
        except json.JSONDecodeError:
            pass

    return {}


def extract_json_field(response: str, field: str, default: Any = None) -> Any:
    """
    Extract a specific field from JSON in LLM response.

    Args:
        response: Raw LLM response string
        field: Field name to extract
        default: Default value if field not found

    Returns:
        Field value or default
    """
    data = extract_json_from_response(response)
    return data.get(field, default)


# =============================================================================
# Trajectory Formatting
# =============================================================================


def format_trajectory_as_conversation_back(trajectory: Any) -> str:
    """
    Format a trajectory into conversation history string for prompts.

    Compatible with FrozenLake trajectory structure from rLLM.

    Args:
        trajectory: Trajectory object with steps containing messages

    Returns:
        Formatted conversation history string
    """
    history_parts = ["\n=== FULL EPISODE TRAJECTORY ===\n"]

    # Handle different trajectory formats
    steps = getattr(trajectory, "steps", None) or []

    for step_idx, step in enumerate(steps):
        history_parts.append(f"\n--- Step {step_idx + 1} ---\n")

        # Extract observation (environment state)
        observation = getattr(step, "observation", None)
        if observation:
            history_parts.append(f"OBSERVATION:\n{observation}\n")

        # Extract agent's reasoning/response
        response = getattr(step, "response", None)
        if response:
            # Handle different response formats
            if hasattr(response, "text"):
                history_parts.append(f"AGENT REASONING:\n{response.text}\n")
            elif isinstance(response, str):
                history_parts.append(f"AGENT REASONING:\n{response}\n")
            elif isinstance(response, dict):
                content = response.get("content", response.get("text", str(response)))
                history_parts.append(f"AGENT REASONING:\n{content}\n")

        # Extract action taken
        action = getattr(step, "action", None)
        if action:
            history_parts.append(f"ACTION: {action}\n")

        # Extract result/feedback
        result = getattr(step, "result", None)
        if result:
            history_parts.append(f"RESULT: {result}\n")

        # Handle chat completions format (ACE-AppWorld style)
        # chat_completions = getattr(step, 'chat_completions', None)
        # if chat_completions:
        #     for msg in chat_completions:
        #         if isinstance(msg, dict):
        #             role = msg.get('role', 'unknown').upper()
        #             content = msg.get('content', '')
        #             history_parts.append(f"[{step_idx}] {role}: {content}\n")

    return "\n".join(history_parts)


def _validate_observation_cap(obs_max_chars: int) -> None:
    if isinstance(obs_max_chars, bool) or not isinstance(obs_max_chars, int) or obs_max_chars < 0:
        raise ValueError("obs_max_chars must be a nonnegative integer (0 disables truncation)")


def format_trajectory_as_conversation(trajectory: Any, obs_max_chars: int = 300) -> str:
    """
    Format a trajectory into COMPACT conversation history string for prompts.

    Compatible with rLLM Step structure:
    - step.observation: Environment state (FrozenLake grid)
    - step.thought: <think>...</think> reasoning block
    - step.model_response: Full LLM response
    - step.action: Action taken (Action object or raw value)
    - step.reward: Step reward
    - step.done: Episode ended flag
    - step.info: Extra info (e.g., action_is_effective)

    Keeps model reasoning verbatim and caps observations.

    Args:
        obs_max_chars: Nonnegative per-step observation char cap. 0 disables
            the cap. Default 300 preserves the prior hardcoded behavior.
    """
    _validate_observation_cap(obs_max_chars)
    history_parts = ["\n=== EPISODE TRAJECTORY ===\n"]

    steps = getattr(trajectory, "steps", None) or []

    if not steps:
        return "(No steps recorded in trajectory)"

    for step_idx, step in enumerate(steps):
        history_parts.append(f"\n--- Step {step_idx + 1} ---")

        # 1. OBSERVATION (grid state) - truncate to essentials
        observation = getattr(step, "observation", None)
        if observation:
            obs_str = str(observation)
            # Cap each observation; obs_max_chars=0 disables the cap.
            if obs_max_chars and len(obs_str) > obs_max_chars:
                obs_str = obs_str[:obs_max_chars] + "..."
            history_parts.append(f"OBSERVATION:\n{obs_str}")

        # 2. AGENT REASONING - extract from thought or model_response
        thought = getattr(step, "thought", None)
        model_response = getattr(step, "model_response", None)

        raw_reasoning = thought or model_response or ""

        if raw_reasoning:
            history_parts.append(f"AGENT REASONING: {raw_reasoning}")

        # 3. ACTION TAKEN
        action = getattr(step, "action", None)
        if action is not None:
            # Handle Action object vs raw value
            if hasattr(action, "action"):
                action_val = action.action
            else:
                action_val = action
            history_parts.append(f"ACTION: {action_val}")

    return "\n".join(history_parts)


def format_trajectory_actions_only(
    trajectory: Any,
    obs_max_chars: int = 1500,
    include_initial_user_message: bool = True,
) -> str:
    """Format actions and their environment replies without model reasoning.

    SWEAgent stores replies in step.next_observation. The info field is a
    fallback for adapters that store next_observation there instead. Each
    reply follows its corresponding action, including the final action.

    Args:
        trajectory: rLLM trajectory object with .steps.
        obs_max_chars: Nonnegative per-observation char cap; 0 disables it.
        include_initial_user_message: Prepend step[0].observation. Disable
            when a prompt already includes the task in a separate slot.
    """
    _validate_observation_cap(obs_max_chars)
    steps = getattr(trajectory, "steps", None) or []
    if not steps:
        return "(No steps recorded in trajectory)"

    def _truncate(text: str) -> str:
        if not obs_max_chars or len(text) <= obs_max_chars:
            return text
        return text[:obs_max_chars] + f"\n... (truncated, {len(text)} total chars)"

    def _step_next_observation(step: Any) -> str:
        text = getattr(step, "next_observation", None)
        if text is None:
            info = getattr(step, "info", None) or {}
            if isinstance(info, dict):
                text = info.get("next_observation")
        return str(text) if text is not None else ""

    parts: list[str] = []

    if include_initial_user_message:
        initial_obs = getattr(steps[0], "observation", None)
        if initial_obs is not None and str(initial_obs):
            parts.append("=== Initial User Message ===")
            parts.append(_truncate(str(initial_obs)))

    for step_idx, step in enumerate(steps):
        turn = step_idx + 1

        action = getattr(step, "action", None)
        action = action.action if hasattr(action, "action") else action
        action_text = "(no action)" if action is None or (isinstance(action, str) and not action.strip()) else str(action)

        parts.append(f"=== Turn {turn} (Generator Action) ===")
        parts.append(action_text)

        obs_text = _step_next_observation(step)
        if not obs_text:
            obs_text = "(no observation)"
        else:
            obs_text = _truncate(obs_text)

        parts.append(f"=== Turn {turn} (Observation) ===")
        parts.append(obs_text)

    return "\n\n".join(parts)


def format_trajectory_with_feedback(
    trajectory: Any,
    obs_max_chars: int = 1500,
    *,
    tokenizer: Any = None,
    max_tokens: int | None = None,
) -> str:
    """Format the opt-in ``feedback_v1`` action/reply history.

    The legacy formatter above remains unchanged. Each action is followed by
    its observed reply, including the last action. Long replies keep both
    their beginning and end so terminal errors are not always lost to a prefix
    cap. Replies fall back to ``info.next_observation`` and then the next
    step's observation; no missing terminal reply is inferred from reward.
    Model reasoning is omitted rather than repeated alongside the action.

    ``obs_max_chars=0`` disables the per-observation cap. When ``max_tokens``
    is supplied, the actual tokenizer is required and the shared first/last
    step truncator enforces the complete history budget. Budget values and
    formatter selection must be part of the experiment identity.
    """
    _validate_observation_cap(obs_max_chars)
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if tokenizer is None:
            raise ValueError("tokenizer is required when max_tokens is set")
    steps = getattr(trajectory, "steps", None) or []
    if not steps:
        empty = "(No steps recorded in trajectory)"
        if max_tokens is not None:
            return truncate_conversation_history(empty, max_tokens=max_tokens, tokenizer=tokenizer, preserve_single_step_tail=True)
        return empty

    def excerpt(value: Any) -> str:
        text = str(value)
        if not obs_max_chars or len(text) <= obs_max_chars:
            return text
        marker = f"\n[... observation truncated; {len(text)} total chars ...]\n"
        # The marker itself is bounded even for intentionally tiny test caps.
        if obs_max_chars <= len(marker):
            head = (obs_max_chars + 1) // 2
            tail = obs_max_chars - head
            return text[:head] + (text[-tail:] if tail else "")
        remaining = obs_max_chars - len(marker)
        head = (remaining + 1) // 2
        tail = remaining - head
        return text[:head] + marker + (text[-tail:] if tail else "")

    parts = ["=== EPISODE TRAJECTORY (action/reply feedback_v1) ==="]
    initial_obs = getattr(steps[0], "observation", None)
    if initial_obs is not None:
        parts.append(f"INITIAL OBSERVATION:\n{excerpt(initial_obs)}")
    for index, step in enumerate(steps):
        parts.append(f"--- Step {index + 1} ---")
        action = getattr(step, "action", None)
        action = action.action if hasattr(action, "action") else action
        action_text = "(no action)" if action is None or (isinstance(action, str) and not action.strip()) else str(action)
        parts.append(f"ACTION:\n{action_text}")

        reply = getattr(step, "next_observation", None)
        if reply is None:
            info = getattr(step, "info", None)
            if isinstance(info, dict):
                reply = info.get("next_observation")
        if reply is None and index + 1 < len(steps):
            reply = getattr(steps[index + 1], "observation", None)
        reply_text = "(no environment reply recorded)" if reply is None else excerpt(reply)
        parts.append(f"ENVIRONMENT REPLY:\n{reply_text}")

    history = "\n\n".join(parts)
    if max_tokens is not None:
        history = truncate_conversation_history(history, max_tokens=max_tokens, tokenizer=tokenizer, preserve_single_step_tail=True)
    return history


def truncate_conversation_history(
    history: str,
    max_tokens: int,
    tokenizer: Any,
    *,
    preserve_single_step_tail: bool = False,
) -> str:
    """
    Truncate conversation history to fit within a token limit.

    Strategy:
    1. If under limit, return as-is
    2. Keep the largest equal number of first and last steps that fits
    3. Replace middle steps with "[... X steps truncated ...]"
    4. If even one complete step from each end cannot fit, retain token
       prefixes/suffixes of those two steps as a final safety measure

    Args:
        history: Formatted conversation history string (from format_trajectory_as_conversation)
        max_tokens: Maximum tokenizer tokens allowed.
        tokenizer: The tokenizer used for the eventual model prompt. It must
            provide ``encode`` and ``decode`` (e.g. a Hugging Face tokenizer).
        preserve_single_step_tail: Opt-in feedback contract: preserve the end
            of a one-step trajectory too. The legacy default retains only its
            beginning. Extremely small budgets prioritize terminal feedback.

    Returns:
        Truncated history string that fits within max_tokens

    Examples:
        >>> history = format_trajectory_as_conversation(trajectory)
        >>> truncated = truncate_conversation_history(history, max_tokens=8192, tokenizer=tokenizer)
    """
    if max_tokens < 1:
        return ""
    if not history:
        return history

    def token_ids(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def token_len(text: str) -> int:
        return len(token_ids(text))

    def token_prefix(text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        ids = token_ids(text)
        if len(ids) <= budget:
            return text
        return tokenizer.decode(ids[:budget], skip_special_tokens=False)

    def token_suffix(text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        ids = token_ids(text)
        if len(ids) <= budget:
            return text
        return tokenizer.decode(ids[-budget:], skip_special_tokens=False)

    # Early return if already under limit.
    if token_len(history) <= max_tokens:
        return history

    # Parse into steps using "--- Step " delimiter
    step_delimiter = "--- Step "
    parts = history.split(step_delimiter)

    # No step structure found: retain the beginning and end equally.
    if len(parts) <= 1:
        marker = "\n[... conversation truncated ...]\n"
        marker_tokens = token_len(marker)
        remaining = max(0, max_tokens - marker_tokens)
        result = token_prefix(history, remaining // 2) + marker + token_suffix(history, remaining - (remaining // 2))
        return result if token_len(result) <= max_tokens else token_prefix(result, max_tokens)

    # Separate header from steps
    header = parts[0]  # e.g., "=== EPISODE TRAJECTORY ===\n"
    steps = parts[1:]  # Step contents (without delimiter prefix)
    num_steps = len(steps)

    def render(steps_per_end: int) -> str:
        if steps_per_end == 0:
            return header
        if 2 * steps_per_end >= num_steps:
            return header + "".join(step_delimiter + step for step in steps)
        middle_removed = num_steps - (2 * steps_per_end)
        marker = f"\n[... {middle_removed} steps truncated for token budget ...]\n\n"
        return header + "".join(step_delimiter + step for step in steps[:steps_per_end]) + marker + "".join(step_delimiter + step for step in steps[-steps_per_end:])

    # N is always even: keep N/2 complete steps at each end.
    max_steps_per_end = num_steps // 2
    for steps_per_end in range(max_steps_per_end, 0, -1):
        candidate = render(steps_per_end)
        if token_len(candidate) <= max_tokens:
            return candidate

    # A single step at either end is too large. Retain both ends while still
    # respecting the hard token limit.
    marker = "\n[... middle of conversation truncated for token budget ...]\n\n"
    prefix = header + step_delimiter + steps[0]
    suffix = step_delimiter + steps[-1] if num_steps > 1 or preserve_single_step_tail else ""
    if preserve_single_step_tail and max_tokens <= token_len(marker):
        return token_suffix(history, max_tokens)
    available = max(0, max_tokens - token_len(marker))
    prefix_budget = available // 2
    suffix_budget = available - prefix_budget
    result = token_prefix(prefix, prefix_budget) + marker + token_suffix(suffix, suffix_budget)
    return result if token_len(result) <= max_tokens else token_prefix(result, max_tokens)


def _truncate_step_content(step_content: str, max_obs_chars: int = 200) -> str:
    """
    Truncate a single step's OBSERVATION and AGENT REASONING content.

    Args:
        step_content: Content of a single step (without "--- Step " prefix)
        max_obs_chars: Max chars for observation content (default 200)

    Returns:
        Step content with truncated observation and reasoning
    """
    lines = step_content.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Truncate OBSERVATION blocks
        if line.startswith("OBSERVATION:"):
            result_lines.append(line)
            i += 1
            # Collect and truncate observation content
            obs_lines = []
            while i < len(lines) and not lines[i].startswith(("AGENT REASONING:", "ACTION:", "--- Step")):
                obs_lines.append(lines[i])
                i += 1
            obs_text = "\n".join(obs_lines)
            if len(obs_text) > max_obs_chars:
                result_lines.append(obs_text[:max_obs_chars] + "... [TRUNCATED]")
            else:
                result_lines.extend(obs_lines)

        # Truncate AGENT REASONING blocks entirely
        elif line.startswith("AGENT REASONING:"):
            result_lines.append("AGENT REASONING: [TRUNCATED]")
            i += 1
            # Skip reasoning content
            while i < len(lines) and not lines[i].startswith(("ACTION:", "--- Step", "OBSERVATION:")):
                i += 1

        # Keep other lines (ACTION, etc.) intact
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def _extract_compact_decision(text: str, max_length: int = 150) -> str:
    """
    Extract just the key decision from verbose <think> reasoning.

    Strategies:
    1. If </think> exists, get content AFTER it (the actual action)
    2. Look for action patterns (```Right```, "I will move Right")
    3. Get last part of reasoning (likely conclusion)
    """
    import re

    if not text:
        return "(no reasoning)"

    text = str(text)

    # Strategy 1: Get content AFTER </think>
    if "</think>" in text:
        after_think = text.split("</think>")[-1].strip()
        if after_think:
            return after_think[:max_length]

    # Strategy 2: Find action declaration patterns
    action_patterns = [
        r"```(\w+)```",  # ```Right```
        r"I (?:will|should|need to) (?:move |go )?(\w+)",
        r"(?:Move|Go|Moving) (\w+)",
        r"(?:best|optimal) (?:move|action)[:\s]+(\w+)",
        r"chosen action[:\s]+(\w+)",
    ]

    for pattern in action_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"Chose: {match.group(1)}"

    # Strategy 3: Last part of <think> block (likely conclusion)
    if "<think>" in text:
        think_content = text.split("<think>")[-1].split("</think>")[0]
        if len(think_content) > max_length:
            return "..." + think_content[-max_length:].strip()
        return think_content.strip()[:max_length]

    # Strategy 4: Fallback - truncate
    if len(text) > max_length:
        return text[:max_length] + "..."

    return text.strip()


def format_messages_as_conversation(messages: list[dict]) -> str:
    """
    Format a list of messages into conversation history string.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Formatted conversation history string
    """
    history_parts = ["\n=== FULL CONVERSATION HISTORY ===\n"]

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        history_parts.append(f"[{i}] {role}: {content}\n")

    return "\n".join(history_parts)


# =============================================================================
# Outcome Derivation
# =============================================================================


def derive_outcome(trajectory: Any, reward: float = 0.0) -> str:
    """
    Derive outcome string from trajectory.

    Args:
        trajectory: Trajectory object
        reward: Final reward value

    Returns:
        "SUCCESS" or "FAILED"
    """
    # Check reward
    if reward > 0:
        return "SUCCESS"

    # Check trajectory attributes
    if hasattr(trajectory, "success") and trajectory.success:
        return "SUCCESS"

    if hasattr(trajectory, "done") and hasattr(trajectory, "reward"):
        if trajectory.done and trajectory.reward > 0:
            return "SUCCESS"

    return "FAILED"


def _detect_truncation(step: Any) -> tuple:
    """
    Detect if a step was likely truncated based on available signals.

    Truncation occurs when the model hits the token limit before outputting
    a valid action. This is detected by:
    - Action parsing failed (action is None or fallback value)
    - Response is very long (near token limit)
    - Incomplete <think> block (opened but not closed)
    - Response ends abruptly (mid-sentence)

    Args:
        step: Step object from trajectory

    Returns:
        Tuple of (is_truncated: bool, reason: str)
    """
    # Get action - check if parsing failed
    action = getattr(step, "action", None)
    action_str = str(action).lower() if action else ""

    # Common fallback values when action parsing fails
    action_failed = action is None or action_str in ("none", "look", "0", "") or action_str.strip() == ""

    if not action_failed:
        return (False, "")

    # Get response text - try multiple attributes
    response = getattr(step, "model_response", None) or getattr(step, "response", None) or getattr(step, "thought", None)

    if hasattr(response, "text"):
        response = response.text
    response_str = str(response) if response else ""

    if not response_str:
        return (False, "")

    response_len = len(response_str)

    # Check 1: Incomplete think block (strong signal)
    # Model started <think> but never closed it - clear truncation
    response_lower = response_str.lower()
    has_think_open = "<think>" in response_lower
    has_think_close = "</think>" in response_lower
    incomplete_think = has_think_open and not has_think_close

    if incomplete_think:
        return (True, "incomplete <think> block - reasoning was cut off")

    # Check 2: Response is very long (near typical token limits)
    # Rough estimate: ~4 chars per token
    # 4096 tokens ≈ 16000 chars, be conservative with 12000
    response_is_long = response_len > 12000

    # Check 3: Response ends abruptly (mid-sentence/mid-word)
    stripped = response_str.rstrip()
    proper_endings = (".", "!", "?", '"', "'", "```", "}", "]", ">", "*", ")")
    ends_abruptly = stripped and not any(stripped.endswith(e) for e in proper_endings)

    # Check 4: Response ends mid-word (very strong signal)
    # If last "word" is cut off (no space before end, not a complete word)
    last_chars = stripped[-20:] if len(stripped) > 20 else stripped
    ends_mid_word = last_chars and not last_chars[-1].isspace() and last_chars[-1].isalpha()

    # Determine truncation based on combinations
    if response_is_long and ends_mid_word:
        return (True, f"response cut off mid-word after {response_len} chars")

    if response_is_long and ends_abruptly:
        return (True, f"long response ({response_len} chars) ending abruptly")

    if response_is_long and action_failed:
        return (True, f"long response ({response_len} chars) with no valid action output")

    # Moderate-length response but clear truncation signs
    if response_len > 8000 and (ends_abruptly or ends_mid_word):
        return (True, f"response likely truncated after {response_len} chars")

    # Action failed but response doesn't show clear truncation signs
    # Could be invalid output format rather than truncation
    return (False, "")


def generate_failure_info(trajectory: Any, outcome: str) -> str:
    """
    Generate failure information from trajectory.

    Analyzes trajectory to determine failure cause for FrozenLake:
    - truncation: Model hit token limit before outputting action
    - fell_in_hole: Agent stepped onto a hole
    - timeout: Exceeded max steps
    - loop: Agent repeated positions
    - ineffective_moves: Moves didn't change position

    Args:
        trajectory: Trajectory object
        outcome: "SUCCESS" or "FAILED"

    Returns:
        Failure description string
    """
    if outcome == "SUCCESS":
        return "N/A - Episode was successful"

    # Analyze trajectory for failure cause
    steps = getattr(trajectory, "steps", [])

    if not steps:
        return "Unknown failure - no steps recorded"

    # Check last step for clues
    last_step = steps[-1] if steps else None

    if last_step:
        # NEW: Check for truncation FIRST (highest priority)
        # Truncation means the model ran out of tokens before outputting an action
        is_truncated, truncation_reason = _detect_truncation(last_step)
        if is_truncated:
            return f"Agent response was TRUNCATED ({truncation_reason}). The model spent too many tokens reasoning and hit the token limit before outputting a valid action."

        # Check for hole fall
        result = getattr(last_step, "result", "")
        info = getattr(last_step, "info", {}) or {}

        result_str = str(result).lower() if result else ""

        # Check result string and info dict for hole indicators
        if "hole" in result_str or "fell" in result_str:
            return "Agent stepped into a hole (O tile)"

        if isinstance(info, dict) and info.get("fell_in_hole"):
            return "Agent stepped into a hole (O tile)"

        if "timeout" in result_str or "max" in result_str:
            return f"Agent exceeded maximum steps ({len(steps)} steps taken)"

    # Check for loops (repeated positions)
    positions = []
    for step in steps:
        obs = getattr(step, "observation", "")
        if obs:
            # Try to extract position from observation
            pos = extract_player_position(str(obs))
            if pos:
                positions.append(pos)

    if positions:
        from collections import Counter

        pos_counts = Counter(positions)
        repeated = [pos for pos, count in pos_counts.items() if count > 2]
        if repeated:
            return f"Agent got stuck in a loop, repeating positions: {repeated}"

    # Check for ineffective moves
    if len(steps) > 1:
        last_actions = [getattr(s, "action", "") for s in steps[-3:]]
        if len(set(last_actions)) == 1 and last_actions[0]:
            return f"Agent repeated ineffective action '{last_actions[0]}' multiple times"

    return f"Agent failed to reach goal after {len(steps)} steps"


def extract_player_position(observation: str) -> tuple | None:
    """
    Extract player position (row, col) from FrozenLake grid observation.

    Args:
        observation: Grid string with 'P' marking player position

    Returns:
        (row, col) tuple or None if not found
    """
    if not observation:
        return None

    lines = observation.strip().split("\n")
    for row_idx, line in enumerate(lines):
        # Look for 'P' in the line
        col_idx = line.find("P")
        if col_idx != -1:
            # Adjust for spacing in formatted grids
            # If grid uses spaces between cells, adjust column
            # Simple heuristic: count non-space position
            clean_line = line.replace(" ", "")
            clean_col = clean_line.find("P")
            if clean_col != -1:
                return (row_idx, clean_col)
            return (row_idx, col_idx)

    return None


# =============================================================================
# Text Similarity (for Deduplication)
# =============================================================================


def compute_keyword_overlap(text1: str, text2: str) -> float:
    """
    Compute keyword overlap ratio between two texts.

    Used for deduplication in Pathway i (parallel processing).

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity (overlap ratio) between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Tokenize and normalize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Remove common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
    }

    words1 = words1 - stop_words
    words2 = words2 - stop_words

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def texts_are_similar(text1: str, text2: str, threshold: float = 0.7) -> bool:
    """
    Check if two texts are similar based on keyword overlap.

    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (default 0.7)

    Returns:
        True if texts are similar (overlap >= threshold)
    """
    return compute_keyword_overlap(text1, text2) >= threshold


# =============================================================================
# Async Utilities
# =============================================================================


async def gather_with_exceptions(*tasks, return_exceptions: bool = True):
    """
    Wrapper around asyncio.gather that handles exceptions gracefully.

    Args:
        *tasks: Async tasks to run
        return_exceptions: If True, return exceptions instead of raising

    Returns:
        List of results (may include Exception objects if return_exceptions=True)
    """
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


def filter_successful_results(results: list) -> list:
    """
    Filter out exception results from asyncio.gather output.

    Args:
        results: List of results from asyncio.gather

    Returns:
        List with only successful (non-exception) results
    """
    return [r for r in results if not isinstance(r, Exception)]


# =============================================================================
# Logging Utilities
# =============================================================================


def log_operation(operation_type: str, details: str, success: bool = True):
    """
    Log an operation with consistent formatting.

    Args:
        operation_type: Type of operation (e.g., "ADD", "REFLECTOR")
        details: Details about the operation
        success: Whether operation succeeded
    """
    status = "✅" if success else "❌"
    print(f"{status} [{operation_type}] {details}")
