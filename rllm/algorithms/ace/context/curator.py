"""
==============================================================================
curator.py
==============================================================================

Curator component for ACE (Agentic Context Engineering) implementation.

The Curator generates playbook operations based on Reflector analysis:
- Reviews existing playbook content
- Determines if new insights should be added
- Avoids redundancy with existing bullets
- Outputs ADD/UPDATE/REMOVE operations for playbook modifications

Note: Deduplication happens implicitly via Curator's prompt.

Template Engine: .replace() with {{variable}} syntax

Supported Template Variables (compatible with multiple prompt styles):

  Core Variables (standardized names):
    - {{current_playbook}}      : Current playbook content snapshot
    - {{guidebook}}             : Key insights from reflection (key_insight field)
    - {{task_instr}}            : Task instruction string
    - {{conversation_history}}  : Formatted conversation/trajectory history

  Backward-Compat Aliases (until all prompts are rewritten):
    - {{question_context}}      : Alias for task_instr
    - {{final_generated_code}}  : Alias for conversation_history

Supported Operations:
    - ADD: Add new bullet to a section
    - UPDATE: Modify existing bullet content by ID
    - REMOVE: Delete bullet by ID

Return Format:
    - Returns dict with "result" (parsed operations) and "log" (prompt/response/parsed)
    - This enables downstream logging without modifying the core logic
"""

import asyncio
from collections.abc import Callable
from typing import Any

from .utils import (
    extract_json_from_response,
    format_trajectory_as_conversation,
    log_operation,
    normalize_section_name,
    truncate_conversation_history,
)

# Valid sections for FrozenLake playbook
ALLOWED_SECTIONS = {
    "strategies_and_hard_rules",
    "common_mistakes",
    "grid_patterns",
    "verification_checklist",
    "others",
}


class Curator:
    """
    Generates playbook operations based on Reflector analysis.

    Attributes:
        prompt_template: Curator prompt template with {{variable}} placeholders
        llm_call_fn: Async function for LLM calls
        allowed_sections: Set of valid section names for operations
    """

    def __init__(
        self,
        prompt_template: str,
        llm_call_fn: Callable | None = None,
        rollout_engine: Any = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        allowed_sections: set | None = None,
        max_conversation_tokens: int = 8192,
        system_prompt: str = "",
    ):
        """
        Initialize Curator.

        Args:
            prompt_template: Template string with {{variable}} placeholders
            llm_call_fn: Async function for LLM calls (preferred)
            rollout_engine: Alternative: Engine for LLM calls (RolloutEngine or similar)
            model_name: Optional model override
            temperature: Sampling temperature (default 0 for deterministic)
            max_retries: Number of retries for failed LLM calls
            allowed_sections: Set of valid section names (defaults to ALLOWED_SECTIONS)
            max_conversation_tokens: Max tokens for conversation history (default 8192)

        One of llm_call_fn or rollout_engine must be provided.
        """
        self.prompt_template = prompt_template
        self.llm_call_fn = llm_call_fn
        self.rollout_engine = rollout_engine
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.allowed_sections = allowed_sections or ALLOWED_SECTIONS
        self.max_conversation_tokens = max_conversation_tokens
        self.tokenizer = getattr(rollout_engine, "tokenizer", None)
        # Optional system message prepended to every LLM call. ACE-RL TRAINING always
        # sets one (SingleTurnCuratorAgent, rllm/algorithms/ace/agents.py);
        # pass the same string for train-faithful evals. Default "" = legacy user-only.
        self.system_prompt = system_prompt

        if not rollout_engine:
            raise ValueError("Rollout engine must be provided")

    async def generate_operations(self, traj_data: dict, reflection: dict, playbook_snapshot: str) -> dict:
        """
        Generate playbook operations for a single trajectory.

        Args:
            traj_data: Dict containing:
                - trajectory: The trajectory object
                - reward: Final reward value
                - info: Additional info dict
            reflection: Output from Reflector (dict with key_insight, etc.)
            playbook_snapshot: Frozen playbook state for this batch

        Returns:
            Dict with:
                - result: Dict with Curator output:
                    - reasoning: Explanation of decisions
                    - operations: List of operation dicts
                        - type: "ADD", "UPDATE", or "REMOVE"
                        - section: Target section name (for ADD)
                        - content: Bullet content (for ADD/UPDATE)
                        - bullet_id: Target bullet ID (for UPDATE/REMOVE)
                - log: Dict with prompt/response/parsed for logging

            Returns empty operations list with log on error.
        """
        # Initialize log dict
        log = {
            "prompt": "",
            "response": "",
            "parsed": {},
        }

        try:
            # Build prompt using snapshot
            prompt = self._build_prompt(traj_data, reflection, playbook_snapshot)
            log["prompt"] = prompt
            # Per-call receipt: the system message is prepended at message assembly
            # and is otherwise invisible in this log (prompt = user content only).
            log["system_prompt"] = self.system_prompt

            # Make LLM call
            response = await self._llm_call(prompt)
            log["response"] = response

            if not response:
                log_operation("CURATOR", "Empty response from LLM", success=False)
                result = {"reasoning": "No response", "operations": []}
                log["parsed"] = result
                return {"result": result, "log": log}

            print("--- Curator Response ---")
            print(response)

            # Parse and validate JSON response
            result = self._parse_response(response)
            log["parsed"] = result

            # Log result
            num_ops = len(result.get("operations", []))
            if num_ops > 0:
                log_operation("CURATOR", f"Generated {num_ops} operation(s)")
            else:
                log_operation("CURATOR", "No operations generated (redundant or low quality)")

            return {"result": result, "log": log}

        except Exception as e:
            log_operation("CURATOR", f"Error: {e}", success=False)
            log["error_type"] = type(e).__name__
            result = {"reasoning": f"Error: {e}", "operations": []}
            log["parsed"] = result
            return {"result": result, "log": log}

    def _build_prompt(self, traj_data: dict, reflection: dict, playbook_snapshot: str) -> str:
        """
        Build the curator prompt from trajectory data and reflection.

        Supports multiple template styles by providing core variables
        with backward-compat aliases. The template engine will use
        whichever variables are present in the template.

        Args:
            traj_data: Dict containing 'trajectory', 'reward', 'info' keys
            reflection: Reflector output dict (with 'key_insight', etc.)
            playbook_snapshot: Frozen playbook state for this batch

        Returns:
            Filled prompt string ready for LLM call

        Template Variables Provided:
            Core (standardized):
            - task_instr: Task instruction from trajectory
            - current_playbook: The playbook snapshot
            - conversation_history: Formatted conversation history
            - guidebook: Key insights from reflection

            Backward-compat aliases:
            - question_context: Alias for task_instr
            - final_generated_code: Alias for conversation_history
        """
        trajectory = traj_data.get("trajectory")

        # Format conversation history
        conversation_history = format_trajectory_as_conversation(trajectory)

        # Truncate conversation history to fit within limits
        original_len = len(conversation_history)
        if self.tokenizer is not None:
            conversation_history = truncate_conversation_history(
                conversation_history,
                max_tokens=self.max_conversation_tokens,
                tokenizer=self.tokenizer,
            )
        if len(conversation_history) < original_len:
            log_operation("CURATOR", f"Truncated conversation history: {original_len} -> {len(conversation_history)} chars")

        # Extract task instruction
        task_instr = ""
        if trajectory:
            # Try trajectory.task first (set by agent in step 1.1)
            if hasattr(trajectory, "task") and trajectory.task:
                task = trajectory.task
                if isinstance(task, dict):
                    task_instr = task.get("instruction", "")
                elif isinstance(task, str):
                    task_instr = task

            # Fallback: try first step's observation (AppWorld pattern)
            if not task_instr and hasattr(trajectory, "steps") and trajectory.steps:
                first_obs = getattr(trajectory.steps[0], "observation", None)
                if isinstance(first_obs, dict):
                    task_instr = first_obs.get("instruction", "")

        # Extract key_insight from reflection dict (for guidebook variable)
        if isinstance(reflection, dict):
            key_insights = reflection.get("key_insight", "")
        else:
            key_insights = str(reflection)

        # Build template variables
        template_vars = {
            # Core variables (standardized names)
            "current_playbook": playbook_snapshot,
            "guidebook": key_insights,
            "task_instr": task_instr,
            "conversation_history": conversation_history,
            # Backward-compat aliases (until prompts are rewritten in 1.5)
            "question_context": task_instr,
            "final_generated_code": conversation_history,
        }

        # Fill template using .replace() with {{variable}} syntax
        # This avoids conflicts with JSON {} in prompts and aligns with Reflector
        prompt = self.prompt_template
        for var_name, var_value in template_vars.items():
            prompt = prompt.replace("{{" + var_name + "}}", str(var_value))
        return prompt

    async def _llm_call(self, prompt: str) -> str:
        """
        Make async LLM call with retry logic.

        Args:
            prompt: Prompt string to send

        Returns:
            LLM response text
        """
        # Mirror ACE-RL training's message shape: system (if configured) + user.
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            print(f"[CURATOR] system prompt applied: {self.system_prompt[:60]!r}", flush=True)
        else:
            messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.max_retries):
            try:
                # Fall back to rollout_engine interfaces
                if hasattr(self.rollout_engine, "get_model_response"):
                    # rLLM RolloutEngine interface
                    response = await self.rollout_engine.get_model_response(
                        messages,
                        model=self.model_name,
                        temperature=self.temperature,
                    )

                    # Check of truncation
                    finish_reason = getattr(response, "finish_reason", "unknown")
                    completion_tokens = getattr(response, "completion_length", 0)
                    prompt_tokens = getattr(response, "prompt_length", 0)
                    if finish_reason == "length":
                        print(f"⚠️ [CURATOR] TRUNCATED! finish_reason={finish_reason}, tokens: prompt={prompt_tokens}, completion={completion_tokens}")
                    return response.text if hasattr(response, "text") else str(response)

                elif hasattr(self.rollout_engine, "generate"):
                    # ACE-AppWorld style interface
                    response = self.rollout_engine.generate(messages=messages)
                    return response.get("content", "") if isinstance(response, dict) else str(response)

                elif hasattr(self.rollout_engine, "chat"):
                    # OpenAI-style interface
                    response = await self.rollout_engine.chat(messages)
                    return response

                elif callable(self.rollout_engine):
                    # Direct callable
                    response = await self.rollout_engine(messages)
                    return str(response)

                else:
                    raise ValueError("No LLM caller available")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ Curator LLM call failed after {self.max_retries} attempts: {e}")
                    return ""

        return ""

    def _parse_response(self, response: str) -> dict:
        """
        Parse and validate Curator's JSON response.

        Expected JSON format:
        {
            "reasoning": "...",
            "operations": [
                {"type": "ADD", "section": "common_mistakes", "content": "..."},
                {"type": "UPDATE", "bullet_id": "shr-00001", "content": "..."},
                {"type": "REMOVE", "bullet_id": "cm-00002"},
                ...
            ]
        }

        Args:
            response: Raw LLM response string

        Returns:
            Validated dict with reasoning and operations
        """
        parsed = extract_json_from_response(response)

        if not parsed:
            return {
                "reasoning": response[:500] if response else "No response",
                "operations": [],
                "raw_response": response,
            }

        # Extract reasoning
        reasoning = parsed.get("reasoning", "")

        # Extract and validate operations
        operations = parsed.get("operations", [])
        validated_ops = self._validate_operations(operations)

        return {
            "reasoning": reasoning,
            "operations": validated_ops,
        }

    def _validate_operations(self, operations: list) -> list:
        """
        Validate and filter operations.

        Ensures:
        - Operation type is valid (ADD, UPDATE, REMOVE)
        - Section is normalized (for ADD)
        - Content is non-empty and >= 10 chars (for ADD, UPDATE)
        - bullet_id is present (for UPDATE, REMOVE)

        Args:
            operations: Raw operations list from LLM

        Returns:
            Filtered list of valid operations
        """
        if not operations or not isinstance(operations, list):
            return []

        validated = []

        for op in operations:
            if not isinstance(op, dict):
                continue

            op_type = op.get("type", "").upper()

            if op_type == "ADD":
                section = op.get("section", "")
                content = op.get("content", "").strip()

                # Normalize section
                normalized_section = normalize_section_name(section)

                # Validate content
                if not content:
                    print("⚠️ Skipping ADD operation with empty content")
                    continue

                # Check content quality (basic)
                if len(content) < 10:
                    print(f"⚠️ Skipping ADD operation with too short content: {content}")
                    continue

                validated.append(
                    {
                        "type": "ADD",
                        "section": normalized_section,
                        "content": content,
                    }
                )

            elif op_type == "UPDATE":
                bullet_id = (op.get("bullet_id") or op.get("tip_id") or "").strip()
                content = op.get("content", "").strip()

                # Validate bullet_id
                if not bullet_id:
                    print("⚠️ Skipping UPDATE operation with missing bullet_id")
                    continue

                # Validate content
                if not content:
                    print("⚠️ Skipping UPDATE operation with empty content")
                    continue

                # Check content quality (basic)
                if len(content) < 10:
                    print(f"⚠️ Skipping UPDATE operation with too short content: {content}")
                    continue

                validated.append(
                    {
                        "type": "UPDATE",
                        "bullet_id": bullet_id,
                        "content": content,
                    }
                )

            elif op_type == "REMOVE":
                bullet_id = (op.get("bullet_id") or op.get("tip_id") or "").strip()

                # Validate bullet_id
                if not bullet_id:
                    print("⚠️ Skipping REMOVE operation with missing bullet_id")
                    continue

                validated.append(
                    {
                        "type": "REMOVE",
                        "bullet_id": bullet_id,
                    }
                )

        return validated


# =============================================================================
# Utility Functions
# =============================================================================


def prepare_curator_prompt(
    template: str,
    playbook: str,
    reflector_output: Any,
    trajectory: Any,
    max_conversation_tokens: int = 8192,
    tokenizer: Any = None,
) -> str:
    """
    Prepare curator prompt (standalone function).

    Convenience function for direct usage without Curator class.

    Args:
        template: Prompt template with {{variable}} placeholders
        playbook: Current playbook content
        reflector_output: Reflector output dict or string
        trajectory: Trajectory object for conversation history
        max_conversation_tokens: Max tokens for conversation history
        tokenizer: Tokenizer used to enforce the history limit. If omitted, the
            history is returned untruncated for backward-compatible utility use.

    Returns:
        Filled prompt string
    """
    # Format conversation history
    conversation_history = format_trajectory_as_conversation(trajectory)
    if tokenizer is not None:
        conversation_history = truncate_conversation_history(
            conversation_history,
            max_tokens=max_conversation_tokens,
            tokenizer=tokenizer,
        )

    # Extract components from reflector output
    if isinstance(reflector_output, dict):
        guidebook = reflector_output.get("key_insight", "")
    else:
        guidebook = str(reflector_output)

    # Extract task instruction
    task_instr = ""
    if trajectory and hasattr(trajectory, "task"):
        task = trajectory.task
        if isinstance(task, dict):
            task_instr = task.get("instruction", "")
        elif isinstance(task, str):
            task_instr = task

    # Build template variables
    template_vars = {
        # Core variables (standardized names)
        "current_playbook": playbook,
        "guidebook": guidebook,
        "task_instr": task_instr,
        "conversation_history": conversation_history,
        # Backward-compat aliases (until prompts are rewritten)
        "question_context": task_instr,
        "final_generated_code": conversation_history,
    }

    # Fill template using .replace() with {{variable}} syntax
    # This avoids conflicts with JSON {} in prompts and aligns with Reflector
    prompt = template
    for var_name, var_value in template_vars.items():
        prompt = prompt.replace("{{" + var_name + "}}", str(var_value))
    return prompt


def parse_curator_output(response: str) -> dict:
    """
    Parse curator output (standalone function).

    Args:
        response: Raw curator response

    Returns:
        Parsed dict with reasoning and operations
    """
    parsed = extract_json_from_response(response)

    if not parsed:
        return {"reasoning": response[:500] if response else "", "operations": []}

    return {
        "reasoning": parsed.get("reasoning", ""),
        "operations": parsed.get("operations", []),
    }


def validate_curator_operations(operations: list, allowed_sections: set = ALLOWED_SECTIONS) -> list:
    """
    Validate curator operations (standalone function).

    Supports:
    - ADD: Requires section, content (>= 10 chars)
    - UPDATE: Requires bullet_id, content (>= 10 chars)
    - REMOVE: Requires bullet_id

    Args:
        operations: Raw operations list
        allowed_sections: Set of valid section names

    Returns:
        Filtered list of valid operations
    """
    if not operations or not isinstance(operations, list):
        return []

    validated = []

    for op in operations:
        if not isinstance(op, dict):
            continue

        op_type = op.get("type", "").upper()

        if op_type == "ADD":
            section = normalize_section_name(op.get("section", ""))
            content = op.get("content", "").strip()

            if section not in allowed_sections:
                section = "others"

            if content and len(content) >= 10:
                validated.append(
                    {
                        "type": "ADD",
                        "section": section,
                        "content": content,
                    }
                )

        elif op_type == "UPDATE":
            bullet_id = (op.get("bullet_id") or op.get("tip_id") or "").strip()
            content = op.get("content", "").strip()

            if bullet_id and content and len(content) >= 10:
                validated.append(
                    {
                        "type": "UPDATE",
                        "bullet_id": bullet_id,
                        "content": content,
                    }
                )

        elif op_type == "REMOVE":
            bullet_id = (op.get("bullet_id") or op.get("tip_id") or "").strip()

            if bullet_id:
                validated.append(
                    {
                        "type": "REMOVE",
                        "bullet_id": bullet_id,
                    }
                )

    return validated
