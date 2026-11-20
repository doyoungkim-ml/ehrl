"""
AppWorld Reflector Adapter

Environment-specific adapter for AppWorld that provides rich trajectory
formatting for the Reflector prompt.

Key Differences from DefaultReflectorAdapter:
- Uses {{trajectory_summary}} instead of {{outcome}} + {{failure_info}}
- Adds {{task_instr}} extracted from trajectory.task["instruction"]
- Derives 4-field trajectory summary: Outcome, Submitted, Steps Used, Termination
- Termination reason derived via decision tree from trajectory data

Termination Reason Logic:
- Engine-level signals (TRUNCATION, PROMPT_TRUNCATION, TIMEOUT, ENV_TIMEOUT) are
  passed through directly from AgentExecutionEngine via trajectory.termination_reason.
  These represent conditions the adapter cannot derive from trajectory data alone.
- For ENV_DONE, MAX_STEPS, or None (missing), the adapter uses a heuristic decision
  tree based on reward, submitted, num_steps, and max_turns to derive a more specific
  reason (CORRECT_SUBMISSION, WRONG_SUBMISSION, MAX_TURNS_EXHAUSTED, EARLY_TERMINATION).

Usage:
    from examples.appworld.train.ace_rl.appworld_adapter import AppWorldReflectorAdapter

    adapter = AppWorldReflectorAdapter(max_turns=20)
    rule = ACEContextRule(config, rollout_engine=engine, reflector_adapter=adapter)

Template Variables Provided:
    - {{task_instr}}: Task instruction string
    - {{trajectory_summary}}: Rich summary with 4 fields
    - {{conversation_history}}: Formatted episode history
    - {{playbook}}: Current playbook content

Note: Does NOT provide {{outcome}} or {{failure_info}} - use {{trajectory_summary}} instead.
The reflector_prompt.txt for AppWorld should use {{trajectory_summary}}.
"""

from typing import Any

# Engine-level termination signals that the adapter cannot derive from
# trajectory data alone. These are passed through directly.
ENGINE_ONLY_SIGNALS = ("TRUNCATION", "PROMPT_TRUNCATION", "TIMEOUT", "ENV_TIMEOUT")


class AppWorldReflectorAdapter:
    """
    AppWorld-specific adapter for Reflector prompt formatting.

    Provides rich trajectory summary with 4 derived variables:
    - Outcome: SUCCESS or FAILED (from reward)
    - Submitted: whether agent called complete_task (from action strings)
    - Steps Used: number of steps taken
    - Termination: engine signal or derived via decision tree

    Also provides task_instr extracted from trajectory.task["instruction"],
    which is set by AppWorldReactAgentACE._initialize_from_task() (step 1.1).

    Attributes:
        max_turns: Maximum allowed turns (from AppWorldEnv config)
    """

    def __init__(
        self,
        max_turns: int = 20,
        tokenizer: Any = None,
        max_history_tokens: int | None = None,
        handoff_protocol: str = "legacy",
    ):
        """
        Initialize AppWorld adapter.

        Args:
            max_turns: Maximum turns per episode (from AppWorldEnv config).
                      Required for distinguishing MAX_TURNS_EXHAUSTED from
                      EARLY_TERMINATION.
        """
        self.max_turns = max_turns
        self.tokenizer = tokenizer
        self.max_history_tokens = max_history_tokens
        if handoff_protocol not in {"legacy", "feedback_v1"}:
            raise ValueError(f"Unsupported handoff_protocol: {handoff_protocol}")
        self.handoff_protocol = handoff_protocol

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
        from rllm.algorithms.ace.context.utils import (
            format_trajectory_as_conversation,
            format_trajectory_with_feedback,
            truncate_conversation_history,
        )

        if self.handoff_protocol == "feedback_v1":
            return format_trajectory_with_feedback(
                trajectory,
                tokenizer=self.tokenizer,
                max_tokens=self.max_history_tokens,
            )
        history = format_trajectory_as_conversation(trajectory)
        if self.tokenizer is not None and self.max_history_tokens is not None:
            return truncate_conversation_history(
                history,
                max_tokens=self.max_history_tokens,
                tokenizer=self.tokenizer,
            )
        return history

    def _check_submitted(self, trajectory: Any) -> bool:
        """
        Check if agent called complete_task during the episode.

        Scans all step actions for "complete_task" substring.
        This is the only reliable way to detect submission from
        trajectory data, since the engine doesn't persist this info.

        Args:
            trajectory: Trajectory object with steps

        Returns:
            True if agent called complete_task in any step
        """
        if not hasattr(trajectory, "steps") or not trajectory.steps:
            return False

        for step in trajectory.steps:
            action = getattr(step, "action", None)
            if action is None:
                continue
            # action can be Action object or string
            action_str = str(action)
            if "complete_task" in action_str:
                return True

        return False

    def _derive_termination_reason(
        self,
        reward: float,
        submitted: bool,
        num_steps: int,
        max_turns: int,
    ) -> str:
        """
        Derive termination reason via decision tree (heuristic fallback).

        Only called when the engine-level signal is ENV_DONE, MAX_STEPS, or
        None — i.e., when the adapter can infer a more specific reason from
        the trajectory data.

        The decision tree order matters — it's a priority chain:
        1. reward == 1.0 → CORRECT_SUBMISSION
        2. submitted → WRONG_SUBMISSION (called complete_task but wrong answer)
        3. num_steps >= max_turns → MAX_TURNS_EXHAUSTED
        4. else → EARLY_TERMINATION (covers cases where engine-level reasons
           like PROMPT_TRUNCATION or ENV_TIMEOUT were not propagated)

        Args:
            reward: Final reward value
            submitted: Whether agent called complete_task
            num_steps: Number of steps taken
            max_turns: Maximum allowed turns

        Returns:
            One of: CORRECT_SUBMISSION, WRONG_SUBMISSION,
                    MAX_TURNS_EXHAUSTED, EARLY_TERMINATION
        """
        if reward == 1.0:
            return "CORRECT_SUBMISSION"
        elif submitted:
            return "WRONG_SUBMISSION"
        elif num_steps >= max_turns:
            return "MAX_TURNS_EXHAUSTED"
        else:
            # Generic fallback — engine-level termination reasons
            # (PROMPT_TRUNCATION, ENV_TIMEOUT) are lost and not
            # recoverable from trajectory data alone
            return "EARLY_TERMINATION"

    def format_trajectory_summary(self, trajectory: Any, reward: float, **kwargs) -> str:
        """
        Format a rich trajectory summary for AppWorld.

        Produces 4 structured fields:
        - Outcome: SUCCESS or FAILED
        - Submitted: True/False (whether agent called complete_task)
        - Steps Used: N/max_turns
        - Termination: engine signal or derived reason

        Args:
            trajectory: Trajectory object
            reward: Final reward value
            **kwargs: Additional context (termination_reason, max_turns)

        Returns:
            Structured summary string
        """
        max_turns = kwargs.get("max_turns", self.max_turns)

        # Count steps
        num_steps = 0
        if hasattr(trajectory, "steps") and trajectory.steps:
            num_steps = len(trajectory.steps)

        # Derive outcome and submission status
        outcome = "SUCCESS" if reward == 1.0 else "FAILED"
        submitted = self._check_submitted(trajectory)

        # Determine termination reason
        # Engine-level signals (TRUNCATION, TIMEOUT, etc.) are passed through
        # directly — the adapter cannot derive these from trajectory data alone.
        # For ENV_DONE / MAX_STEPS / None, use heuristic decision tree for
        # more specific reasons (CORRECT_SUBMISSION, WRONG_SUBMISSION, etc.).
        engine_reason = kwargs.get("termination_reason")
        if engine_reason in ENGINE_ONLY_SIGNALS:
            termination = engine_reason
        else:
            termination = self._derive_termination_reason(reward, submitted, num_steps, max_turns)

        lines = [
            f"Outcome: {outcome}",
            f"Submitted: {submitted}",
            f"Steps Used: {num_steps}/{max_turns}",
            f"Termination: {termination}",
        ]

        return "\n".join(lines)

    def get_template_variables(self, trajectory: Any, reward: float, playbook: str, **kwargs) -> dict[str, str]:
        """
        Get template variables for AppWorld Reflector prompt.

        Unlike DefaultReflectorAdapter, this does NOT provide {{outcome}} or
        {{failure_info}}. AppWorld prompts should use {{trajectory_summary}} instead.

        Includes {{task_instr}} from trajectory.task, which is set by
        AppWorldReactAgentACE._initialize_from_task() (step 1.1).

        Args:
            trajectory: Trajectory object
            reward: Final reward value
            playbook: Current playbook content
            **kwargs: Additional context (termination_reason, max_turns)

        Returns:
            Dict with task_instr, trajectory_summary, conversation_history, playbook
        """
        # Extract task instruction from trajectory.task (set by agent in 1.1)
        task_instr = ""
        if hasattr(trajectory, "task") and trajectory.task:
            if isinstance(trajectory.task, dict):
                task_instr = trajectory.task.get("instruction", "")
            elif isinstance(trajectory.task, str):
                task_instr = trajectory.task

        conversation_history = self.format_conversation_history(trajectory)
        trajectory_summary = self.format_trajectory_summary(trajectory, reward, **kwargs)

        return {
            "task_instr": task_instr,
            "trajectory_summary": trajectory_summary,
            "conversation_history": conversation_history,
            "playbook": playbook,
        }
