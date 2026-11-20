"""
MBPP Reflector Adapter

Environment-specific adapter for MBPP (Mostly Basic Python Problems) that provides
rich trajectory formatting for the Reflector prompt.

Key Differences from DefaultReflectorAdapter:
- Uses {{trajectory_summary}} instead of {{outcome}} + {{failure_info}}
- Extracts MBPP-specific info (all_passed, turns_used, best_reward) from step.info
- Success detection based on all_passed flag (not reward > 0)
- Provides structured summary optimized for code generation analysis

Termination Reason Logic:
- Engine-level signals (TRUNCATION, PROMPT_TRUNCATION, TIMEOUT, ENV_TIMEOUT) are
  passed through directly from AgentExecutionEngine via trajectory.termination_reason.
  These represent conditions the adapter cannot derive from trajectory data alone.
- For ENV_DONE, MAX_STEPS, or None (missing), the adapter uses heuristics to derive
  a more specific reason (TASK_COMPLETED, MAX_TURNS_REACHED, UNKNOWN).

Usage:
    from examples.mbpp.train.ace_rl.mbpp_adapter import MBPPReflectorAdapter

    adapter = MBPPReflectorAdapter(max_turns=6)
    rule = ACEContextRule(config, rollout_engine=engine, reflector_adapter=adapter)

Template Variables Provided:
    - {{trajectory_summary}}: Rich summary with outcome, turns, termination reason
    - {{conversation_history}}: Formatted episode history
    - {{playbook}}: Current playbook content

Note: Does NOT provide {{outcome}} or {{failure_info}} - use {{trajectory_summary}} instead.
The reflector_prompt.txt for MBPP should use {{trajectory_summary}}.
"""

from typing import Any

# Engine-level termination signals that the adapter cannot derive from
# trajectory data alone. These are passed through directly.
ENGINE_ONLY_SIGNALS = ("TRUNCATION", "PROMPT_TRUNCATION", "TIMEOUT", "ENV_TIMEOUT")


class MBPPReflectorAdapter:
    """
    MBPP-specific adapter for Reflector prompt formatting.

    Extracts rich trajectory information from MBPP environment's step.info
    and formats it for effective reflection on code generation tasks.

    Attributes:
        max_turns: Maximum allowed turns (for trajectory summary context)
    """

    def __init__(
        self,
        max_turns: int = 6,
        evaluation_summary: bool = False,
        verbose_summary: bool = False,
        handoff_protocol: str = "legacy",
        tokenizer: Any = None,
        max_history_tokens: int | None = None,
    ):
        """
        Initialize MBPP adapter.

        Args:
            max_turns: Maximum turns per episode (default: 6).
            evaluation_summary: Use the evaluation prompt's summary layout.
            verbose_summary: Include reward details in the evaluation summary.
            handoff_protocol: ``legacy`` keeps the historical history format;
                ``feedback_v1`` pairs each submission with its test reply,
                including the final one.
            tokenizer: Tokenizer for the optional feedback_v1 history budget.
            max_history_tokens: Optional feedback_v1 history token budget.
        """
        if handoff_protocol not in {"legacy", "feedback_v1"}:
            raise ValueError(f"Unsupported handoff_protocol: {handoff_protocol}")
        self.max_turns = max_turns
        self.evaluation_summary = evaluation_summary
        self.verbose_summary = verbose_summary
        self.handoff_protocol = handoff_protocol
        self.tokenizer = tokenizer
        self.max_history_tokens = max_history_tokens

    def format_conversation_history(self, trajectory: Any) -> str:
        """
        Format trajectory as conversation history.

        Delegates to the standard utility function.

        Args:
            trajectory: Trajectory object with steps

        Returns:
            Formatted conversation string
        """
        # Import inside method to avoid circular dependencies
        from rllm.algorithms.ace.context.utils import format_trajectory_as_conversation, format_trajectory_with_feedback

        if self.handoff_protocol == "feedback_v1":
            return format_trajectory_with_feedback(
                trajectory,
                tokenizer=self.tokenizer,
                max_tokens=self.max_history_tokens,
            )
        return format_trajectory_as_conversation(trajectory)

    def format_trajectory_summary(self, trajectory: Any, reward: float, **kwargs) -> str:
        """
        Format a rich trajectory summary for MBPP.

        Extracts MBPP-specific information from trajectory steps:
        - all_passed: Whether all test cases passed
        - turns_used: Number of turns taken
        - best_reward: Highest reward achieved during episode
        - termination: Why the episode ended (engine signal or heuristic)

        Args:
            trajectory: Trajectory object
            reward: Final reward value
            **kwargs: Additional context (termination_reason, max_turns)

        Returns:
            Structured summary string
        """
        # Extract MBPP-specific info from trajectory
        all_passed = False
        turns_used = 0
        best_reward = reward
        max_turns = kwargs.get("max_turns", self.max_turns)

        # Process trajectory steps to extract info
        if hasattr(trajectory, "steps") and trajectory.steps:
            turns_used = len(trajectory.steps)

            for step in trajectory.steps:
                # Extract from step.info if available
                if hasattr(step, "info") and isinstance(step.info, dict):
                    info = step.info

                    # Check for all_passed flag
                    if info.get("all_passed", False):
                        all_passed = True

                    # Track best reward
                    step_reward = info.get("reward", 0)
                    if step_reward > best_reward:
                        best_reward = step_reward

                # Also check step.reward directly
                if hasattr(step, "reward") and step.reward is not None:
                    if step.reward > best_reward:
                        best_reward = step.reward

        # Determine outcome based on all_passed (not just reward)
        # For MBPP, success means all test cases passed
        outcome = "SUCCESS" if all_passed else "FAILED"

        # Determine termination reason
        # Engine-level signals (TRUNCATION, TIMEOUT, etc.) are passed through
        # directly — the adapter cannot derive these from trajectory data alone.
        # For ENV_DONE / MAX_STEPS / None, use heuristics for more detail.
        engine_reason = kwargs.get("termination_reason")
        if engine_reason in ENGINE_ONLY_SIGNALS:
            termination = engine_reason
        elif all_passed:
            termination = "TASK_COMPLETED"
        elif turns_used >= max_turns and not all_passed:
            termination = "MAX_TURNS_REACHED"
        else:
            termination = "UNKNOWN"

        if self.evaluation_summary:
            lines = [
                f"Outcome: {outcome}",
                f"Turns Used: {turns_used}/{max_turns}",
                f"Termination Reason: {termination}",
            ]
            if self.verbose_summary:
                lines.insert(1, f"All Tests Passed: {all_passed}")
                lines.extend([f"Final Reward: {reward:.2f}", f"Best Reward: {best_reward:.2f}"])
            return "\n".join(lines)

        # Build structured summary
        lines = [
            f"Outcome: {outcome}",
            f"All Tests Passed: {all_passed}",
            f"Turns Used: {turns_used}/{max_turns}",
            f"Final Reward: {reward:.2f}",
            f"Best Reward: {best_reward:.2f}",
            f"Termination: {termination}",
        ]

        return "\n".join(lines)

    def get_template_variables(self, trajectory: Any, reward: float, playbook: str, **kwargs) -> dict[str, str]:
        """
        Get template variables for MBPP Reflector prompt.

        Unlike DefaultReflectorAdapter, this does NOT provide {{outcome}} or
        {{failure_info}}. MBPP prompts should use {{trajectory_summary}} instead.

        Args:
            trajectory: Trajectory object
            reward: Final reward value
            playbook: Current playbook content
            **kwargs: Additional context (termination_reason, max_turns)

        Returns:
            Dict with trajectory_summary, conversation_history, playbook
        """
        conversation_history = self.format_conversation_history(trajectory)
        trajectory_summary = self.format_trajectory_summary(trajectory, reward, **kwargs)

        return {
            "trajectory_summary": trajectory_summary,
            "conversation_history": conversation_history,
            "playbook": playbook,
        }
