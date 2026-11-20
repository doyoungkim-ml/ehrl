"""
==============================================================================
reflector.py
==============================================================================

Reflector component for ACE (Agentic Context Engineering) implementation.

The Reflector analyzes completed trajectories and produces insights for the Curator.
In Pathway i (parallel processing), all Reflector calls see the same playbook snapshot.

Responsibilities:
- Analyze trajectory to determine SUCCESS/FAILED outcome
- Generate failure information if applicable
- Extract key insights for potential playbook updates

Template Engine: Python `.replace()` with `{{variable}}` syntax

Template Variables (via adapter):
- DefaultReflectorAdapter: {{outcome}}, {{failure_info}}, {{playbook}}, {{conversation_history}}
- AppWorldReflectorAdapter: {{task_instr}}, {{trajectory_summary}}, {{playbook}}, {{conversation_history}}
- MBPPReflectorAdapter: {{trajectory_summary}}, {{playbook}}, {{conversation_history}}

Adapter Pattern:
- Adapters provide environment-specific trajectory formatting
- DefaultReflectorAdapter maintains backward compatibility
- Custom adapters can be injected for different environments

Return Format:
- Returns dict with "result" (parsed reflection) and "log" (prompt/response/parsed)
- This enables downstream logging without modifying the core logic
"""

import asyncio
import re
from collections.abc import Callable
from typing import Any

from .prompt_adapter import DefaultReflectorAdapter
from .utils import (
    derive_outcome,
    extract_json_from_response,
    format_trajectory_as_conversation,
    log_operation,
)


class Reflector:
    """
    Analyzes trajectory and produces reflection for Curator.

    In Pathway i: Called in parallel, all see same playbook snapshot.
    This means bullet tagging is based on snapshot, not live state.

    Attributes:
        prompt_template: Reflector prompt template with {{variable}} placeholders
        adapter: ReflectorAdapter for environment-specific formatting
        llm_call_fn: Async function to call LLM (injected for flexibility)
        max_retries: Maximum retry attempts for LLM calls
    """

    def __init__(
        self,
        prompt_template: str,
        adapter: Any = None,
        llm_call_fn: Callable | None = None,
        rollout_engine: Any = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        system_prompt: str = "",
    ):
        """
        Initialize Reflector.

        Args:
            prompt_template: Reflector prompt with {{variable}} placeholders
            adapter: ReflectorAdapter for trajectory formatting.
                    If None, uses DefaultReflectorAdapter (backward compatible).
            llm_call_fn: Async function to call LLM. Signature: async (messages) -> str
                        If None, must be set before calling analyze()
            rollout_engine: Alternative engine for LLM calls (RolloutEngine or similar)
            model_name: Model name for LLM calls
            temperature: Sampling temperature
            max_retries: Maximum retry attempts for LLM calls
            system_prompt: Optional system message prepended to every LLM call.
                    ACE-RL TRAINING always sets one (SingleTurnReflectorAgent,
                    rllm/algorithms/ace/agents.py); pass the same string
                    for train-faithful evals. Default "" = legacy user-only messages.
        """
        self.prompt_template = prompt_template
        self.adapter = adapter or DefaultReflectorAdapter()
        self.llm_call_fn = llm_call_fn
        self.rollout_engine = rollout_engine
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.system_prompt = system_prompt

    # =========================================================================
    # Main Analysis Method
    # =========================================================================

    async def analyze(self, traj_data: dict, playbook_snapshot: str) -> dict:
        """
        Analyze a single trajectory.

        In Pathway i: Uses playbook_snapshot (frozen at batch start),
        not live playbook state.

        Args:
            traj_data: Dict containing:
                - trajectory: The trajectory object
                - reward: Final reward (float)
                - termination_reason: Why trajectory ended (optional)
                - max_turns: Maximum turns allowed (optional)
                - info: Additional info dict (optional)
            playbook_snapshot: Frozen playbook state for this batch

        Returns:
            Dict with:
                - result: Parsed reflection dict (fields depend on environment prompt)
                    At minimum contains: reasoning, outcome, key_insight
                    May also contain environment-specific fields
                - log: Dict with prompt/response/parsed for logging

            Returns default reflection with empty log on failure.
        """
        if not self.rollout_engine:
            raise ValueError("LLM rollout engine not set. Call set_llm_call() first.")

        # Initialize log dict
        log = {
            "prompt": "",
            "response": "",
            "parsed": {},
        }

        try:
            # Build prompt using adapter
            prompt = self._build_prompt(traj_data, playbook_snapshot)
            log["prompt"] = prompt
            # Per-call receipt: the system message is prepended at message assembly
            # and is otherwise invisible in this log (prompt = user content only).
            log["system_prompt"] = self.system_prompt

            # Call LLM with retry
            response = await self._llm_call(prompt)
            log["response"] = response

            if not response:
                log_operation("REFLECTOR", "Empty response from LLM", success=False)
                result = self._get_default_reflection(traj_data)
                log["parsed"] = result
                return {"result": result, "log": log}

            # Parse JSON response
            result = self._parse_response(response)
            # Adapters may correct fields that the environment knows better than the model.
            finalize = getattr(self.adapter, "finalize_reflection", None)
            if result and finalize is not None:
                result = finalize(result, traj_data)
            log["parsed"] = result

            if result:
                log_operation("REFLECTOR", f"Analyzed trajectory: {result.get('outcome', 'UNKNOWN')}")

            return {"result": result, "log": log}

        except Exception as e:
            log_operation("REFLECTOR", f"Analysis failed: {e}", success=False)
            log["error_type"] = type(e).__name__
            result = self._get_default_reflection(traj_data)
            log["parsed"] = result
            return {"result": result, "log": log}

    # =========================================================================
    # Prompt Building
    # =========================================================================

    def _build_prompt(self, traj_data: dict, playbook_snapshot: str) -> str:
        """
        Build reflector prompt from template and trajectory data.

        Delegates to adapter for environment-specific variable formatting.
        Uses .replace() with {{variable}} syntax.

        Args:
            traj_data: Trajectory data dict
            playbook_snapshot: Frozen playbook content

        Returns:
            Filled prompt string
        """
        trajectory = traj_data.get("trajectory")
        reward = traj_data.get("reward", 0.0)

        # Get template variables from adapter
        template_vars = self.adapter.get_template_variables(
            trajectory=trajectory,
            reward=reward,
            playbook=playbook_snapshot,
            termination_reason=traj_data.get("termination_reason"),
            max_turns=traj_data.get("max_turns", 5),
        )

        # Fill template using .replace() for all variables
        prompt = self.prompt_template
        for key, value in template_vars.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

        return prompt

    def _format_trajectory(self, trajectory: Any) -> str:
        """
        Format trajectory as conversation history for the prompt.

        Handles various trajectory formats from rLLM.

        Args:
            trajectory: Trajectory object

        Returns:
            Formatted conversation string
        """
        return format_trajectory_as_conversation(trajectory)

    # =========================================================================
    # LLM Calling
    # =========================================================================

    async def _llm_call(self, prompt: str) -> str:
        """
        Call LLM with retry logic.

        Args:
            prompt: Full prompt string

        Returns:
            LLM response string
        """
        # Mirror ACE-RL training's message shape: system (if configured) + user.
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            print(f"[REFLECTOR] system prompt applied: {self.system_prompt[:60]!r}", flush=True)
        else:
            messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.max_retries):
            try:
                if self.rollout_engine:
                    # Use get_model_response which returns ModelOutput
                    model_output = await self.rollout_engine.get_model_response(
                        messages,
                        temperature=self.temperature,
                    )

                    # Check of truncation
                    finish_reason = getattr(model_output, "finish_reason", "unknown")
                    completion_tokens = getattr(model_output, "completion_length", 0)
                    prompt_tokens = getattr(model_output, "prompt_length", 0)
                    if finish_reason == "length":
                        print(f"⚠️ [REFLECTOR] TRUNCATED! finish_reason={finish_reason}, tokens: prompt={prompt_tokens}, completion={completion_tokens}")

                    # ModelOutput has .text attribute
                    if model_output:
                        return model_output.text if hasattr(model_output, "text") else str(model_output)
                elif self.llm_call_fn:
                    return await self.llm_call_fn(messages)
                else:
                    raise ValueError("No LLM call method available")

            except Exception:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    raise

        return ""

    # =========================================================================
    # Response Parsing
    # =========================================================================

    def _parse_response(self, response: str) -> dict:
        """
        Parse LLM response to extract reflection.

        Handles both JSON and structured text responses.

        Args:
            response: Raw LLM response string

        Returns:
            Parsed reflection dict
        """
        # Try JSON extraction first
        result = extract_json_from_response(response)

        if result:
            return self._validate_reflection(result)

        # Fallback: try to extract from text
        return self._parse_text_response(response)

    def _parse_text_response(self, text: str) -> dict:
        """
        Extract reflection from unstructured text response.

        Fallback parser when JSON extraction fails.
        Environment-agnostic: extracts only the core fields.

        Args:
            text: Raw text response

        Returns:
            Extracted reflection dict
        """
        result = {
            "reasoning": text[:1000] if text else "",
            "outcome": "FAILED",
            "key_insight": "",
        }

        # Try to extract outcome
        if re.search(r"\bSUCCESS\b", text, re.IGNORECASE):
            result["outcome"] = "SUCCESS"

        # Try to extract key insight
        insight_patterns = [
            r"key.?insight[:\s]+(.+?)(?:\n|$)",
            r"insight[:\s]+(.+?)(?:\n|$)",
        ]

        for pattern in insight_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["key_insight"] = match.group(1).strip()
                break

        return result

    def _validate_reflection(self, result: dict) -> dict:
        """
        Validate and normalize reflection result.

        Environment-agnostic validation:
        - Ensures core fields (outcome, reasoning, key_insight) have valid types
        - Preserves any extra fields from environment-specific prompts
          (e.g., error_identification, failure_reason, bullet_tags)
        - Does NOT enforce environment-specific field enums

        Args:
            result: Parsed result dict

        Returns:
            Validated result dict with core fields guaranteed
        """
        if not result:
            return {}

        # Ensure outcome is valid (core field)
        if result.get("outcome") not in ("SUCCESS", "FAILED"):
            result["outcome"] = "FAILED"

        # Ensure key_insight is a string (core field — consumed by Curator)
        if "key_insight" not in result or not isinstance(result.get("key_insight"), str):
            result["key_insight"] = ""

        # Ensure reasoning is a string (core field)
        if "reasoning" not in result or not isinstance(result.get("reasoning"), str):
            result["reasoning"] = ""

        # All other fields (failure_reason, correct_approach, error_identification,
        # root_cause_analysis, bullet_tags, etc.) are preserved as-is from the
        # parsed JSON. They are environment-specific and only used for logging.

        return result

    def _get_default_reflection(self, traj_data: dict) -> dict:
        """
        Get default reflection when analysis fails.

        Returns the minimal 3-field format. Environment-specific prompts
        may produce more fields, but the default only guarantees the core.

        Args:
            traj_data: Trajectory data dict

        Returns:
            Default reflection dict with outcome, reasoning, key_insight
        """
        trajectory = traj_data.get("trajectory")
        reward = traj_data.get("reward", 0.0)

        outcome = derive_outcome(trajectory, reward)

        return {
            "outcome": outcome,
            "reasoning": "Analysis failed - using default values",
            "key_insight": "",
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def load_reflector_prompt(filepath: str) -> str:
    """
    Load reflector prompt template from file.

    Args:
        filepath: Path to prompt template file

    Returns:
        Prompt template string
    """
    with open(filepath) as f:
        return f.read()


def prepare_reflector_prompt(template: str, trajectory: Any, playbook: str, reward: float = 0.0, adapter: Any = None, **kwargs) -> str:
    """
    Prepare reflector prompt from trajectory (standalone function).

    Convenience function for direct usage without Reflector class.

    Args:
        template: Reflector prompt template
        trajectory: Trajectory object
        playbook: Current playbook content
        reward: Final reward value
        adapter: ReflectorAdapter for formatting (optional).
                If None, uses DefaultReflectorAdapter.
        **kwargs: Additional parameters (termination_reason, max_turns, etc.)

    Returns:
        Filled prompt string
    """
    if adapter is None:
        adapter = DefaultReflectorAdapter()

    # Get template variables from adapter
    template_vars = adapter.get_template_variables(trajectory=trajectory, reward=reward, playbook=playbook, **kwargs)

    # Fill template
    prompt = template
    for key, value in template_vars.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

    return prompt


def parse_reflector_output(response: str) -> dict:
    """
    Parse reflector output (standalone function).

    Environment-agnostic: extracts whatever fields are present in
    the JSON response. Guarantees core fields (outcome, reasoning,
    key_insight) with defaults. Extra fields are preserved.

    Args:
        response: Raw reflector response

    Returns:
        Parsed dict with at least outcome, reasoning, key_insight
    """
    parsed = extract_json_from_response(response)

    if not parsed:
        return {
            "outcome": "FAILED",
            "reasoning": response[:1000] if response else "",
            "key_insight": "",
        }

    # Ensure core fields have defaults, preserve everything else
    result = dict(parsed)  # copy all fields from parsed JSON
    result.setdefault("outcome", "FAILED")
    result.setdefault("reasoning", "")
    result.setdefault("key_insight", "")

    # Validate outcome
    if result["outcome"] not in ("SUCCESS", "FAILED"):
        result["outcome"] = "FAILED"

    return result


async def analyze_trajectory(
    trajectory: Any,
    reward: float,
    playbook: str,
    prompt_template: str,
    llm_call_fn: Callable,
    adapter: Any = None,
) -> dict:
    """
    Convenience function to analyze a single trajectory.

    Args:
        trajectory: Trajectory object
        reward: Final reward value
        playbook: Current playbook content
        prompt_template: Reflector prompt template
        llm_call_fn: Async LLM call function
        adapter: ReflectorAdapter (optional)

    Returns:
        Reflection dict
    """
    reflector = Reflector(
        prompt_template=prompt_template,
        adapter=adapter,
        llm_call_fn=llm_call_fn,
    )

    traj_data = {
        "trajectory": trajectory,
        "reward": reward,
    }

    result = await reflector.analyze(traj_data, playbook)
    return result.get("result", {})
