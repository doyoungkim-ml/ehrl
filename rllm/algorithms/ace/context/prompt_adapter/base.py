"""
Base Adapter Classes for ACE Reflector Prompt Formatting

This module defines the ReflectorAdapter protocol and provides the
DefaultReflectorAdapter for backward compatibility with existing environments.

Design:
- ReflectorAdapter: Protocol (interface) that all adapters must implement
- DefaultReflectorAdapter: Replicates legacy behavior for FrozenLake, AppWorld
- Environment-specific adapters extend this pattern (e.g., MBPPReflectorAdapter)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ReflectorAdapter(Protocol):
    """
    Protocol for environment-specific Reflector prompt adapters.

    Adapters transform trajectory data into template variables for the
    Reflector prompt. Different environments may need different variables
    or formatting.

    Methods:
        format_conversation_history: Format trajectory as conversation string
        format_trajectory_summary: Create structured summary of trajectory
        get_template_variables: Return all variables for template substitution
    """

    def format_conversation_history(self, trajectory: Any) -> str:
        """
        Format trajectory as conversation history string.

        Args:
            trajectory: Trajectory object with steps

        Returns:
            Formatted conversation string
        """
        ...

    def format_trajectory_summary(self, trajectory: Any, reward: float, **kwargs) -> str:
        """
        Format a structured summary of the trajectory.

        Args:
            trajectory: Trajectory object
            reward: Final reward value
            **kwargs: Additional context (termination_reason, max_turns, etc.)

        Returns:
            Structured summary string
        """
        ...

    def get_template_variables(self, trajectory: Any, reward: float, playbook: str, **kwargs) -> dict[str, str]:
        """
        Get all template variables for Reflector prompt.

        Args:
            trajectory: Trajectory object
            reward: Final reward value
            playbook: Current playbook content
            **kwargs: Additional context (termination_reason, max_turns, etc.)

        Returns:
            Dict mapping variable names to values (e.g., {"outcome": "SUCCESS"})
        """
        ...


class DefaultReflectorAdapter:
    """
    Default adapter providing backward-compatible behavior.

    Replicates the legacy Reflector formatting used by FrozenLake and AppWorld.
    Provides both legacy variables ({{outcome}}, {{failure_info}}) and new
    variables ({{trajectory_summary}}) for gradual migration.

    Template Variables Provided:
        - {{outcome}}: "SUCCESS" or "FAILED"
        - {{failure_info}}: Human-readable failure explanation
        - {{conversation_history}}: Formatted episode history
        - {{playbook}}: Current playbook content
        - {{trajectory_summary}}: Structured summary (for forward compatibility)
    """

    def format_conversation_history(self, trajectory: Any) -> str:
        """
        Format trajectory as conversation history.

        Uses the existing utility function for consistency.
        """
        # Import inside method to avoid circular dependencies
        from rllm.algorithms.ace.context.utils import format_trajectory_as_conversation

        return format_trajectory_as_conversation(trajectory)

    def format_trajectory_summary(self, trajectory: Any, reward: float, **kwargs) -> str:
        """
        Format a basic trajectory summary.

        Default implementation provides a simple summary.
        Environment-specific adapters can override for richer output.
        """
        # Import inside method to avoid circular dependencies
        from rllm.algorithms.ace.context.utils import derive_outcome

        outcome = derive_outcome(trajectory, reward)

        # Count steps
        num_steps = 0
        if hasattr(trajectory, "steps") and trajectory.steps:
            num_steps = len(trajectory.steps)

        lines = [
            f"Result: {outcome}",
            f"Steps: {num_steps}",
            f"Reward: {reward:.2f}",
        ]

        return "\n".join(lines)

    def get_template_variables(self, trajectory: Any, reward: float, playbook: str, **kwargs) -> dict[str, str]:
        """
        Get template variables with legacy compatibility.

        Provides both old-style variables (outcome, failure_info) and
        new-style variables (trajectory_summary) for backward compatibility.
        """
        # Import inside method to avoid circular dependencies
        from rllm.algorithms.ace.context.utils import derive_outcome, generate_failure_info

        outcome = derive_outcome(trajectory, reward)
        failure_info = generate_failure_info(trajectory, outcome)
        conversation_history = self.format_conversation_history(trajectory)
        trajectory_summary = self.format_trajectory_summary(trajectory, reward, **kwargs)

        return {
            # Legacy variables (for existing prompts)
            "outcome": outcome,
            "failure_info": failure_info,
            # Common variables
            "conversation_history": conversation_history,
            "playbook": playbook,
            # New variable (for forward compatibility)
            "trajectory_summary": trajectory_summary,
        }
