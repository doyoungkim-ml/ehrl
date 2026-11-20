"""Contracts for context that persists across agent episodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.agents.agent import Step, Trajectory


class ContextUpdateLevel(Enum):
    """Required update timing; runners declare which levels they support."""

    STEP = "step"
    TRAJECTORY = "trajectory"
    BATCH = "batch"


class ContextRule(ABC):
    """Provide prompt context and learn from completed agent experiences.

    get_context is a read operation, including during evaluation. Update and
    reset hooks may mutate state; runners skip them when adaptation is off.
    ContextAwareRunner supports batch updates: it calls the trajectory hooks
    in result order, then awaits one batch hook before starting the next batch.
    Step and trajectory update levels require a runner with those capabilities.
    """

    @property
    @abstractmethod
    def update_level(self) -> ContextUpdateLevel:
        """Return the timing required by this rule."""

    @abstractmethod
    def get_context(self) -> str:
        """Read the current prompt context without changing rule state."""

    def get_step_context(self, step_idx: int, previous_steps: list[Step]) -> str:
        """Read step-specific context when supported by the runner."""
        return self.get_context()

    def on_step_complete(self, step: Step, step_idx: int, trajectory_so_far: list[Step]) -> None:
        """Update after a step, for runners with step callback support."""
        return None

    def on_trajectory_complete(self, trajectory: Trajectory) -> None:
        """Consume one completed trajectory before the batch hook."""
        return None

    @abstractmethod
    async def on_batch_complete(self, trajectories: list[Trajectory]) -> None:
        """Update context after all trajectories in a batch complete."""

    def reset_trajectory(self) -> None:
        """Reset transient episode state when supported by the runner."""
        return None

    def reset_batch(self) -> None:
        """Reset transient state before an adapting batch starts."""
        return None

    @abstractmethod
    def save(self, path: str) -> None:
        """Save persistent context state."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore persistent context state."""


class NoContextRule(ContextRule):
    """Run with empty context and no adaptation."""

    @property
    def update_level(self) -> ContextUpdateLevel:
        return ContextUpdateLevel.BATCH

    def get_context(self) -> str:
        return ""

    async def on_batch_complete(self, trajectories: list[Trajectory]) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
