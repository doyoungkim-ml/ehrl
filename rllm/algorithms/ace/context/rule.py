"""
==============================================================================
ace_context_rule.py
==============================================================================

ACE (Agentic Context Engineering) Context Rule implementation.

This module implements ACEContextRule, which manages cross-episode learning
via the Reflector-Curator pipeline.

Execution Flow:
1. Generator: Run tasks sequentially (via AgentExecutionEngine)
2. Reflector: Analyze ALL trajectories (same playbook snapshot)
3. Curator: Generate operations (same playbook snapshot)

Updates:
- Use batch_size=1 for ACE's sequential processing.

Adapter Pattern:
- reflector_adapter parameter allows environment-specific trajectory formatting
- If None, DefaultReflectorAdapter is used (backward compatible)
- Custom adapters (e.g., MBPPReflectorAdapter) provide environment-specific logic

Logging:
- When ace_log_dir is configured, saves Reflector/Curator I/O per task
- Each task gets a list of 2 items: [reflector_log, curator_log]
- Logs include: prompt, response (raw LLM output), parsed (JSON result)
"""

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rllm.agents.agent import Trajectory
from rllm.context.base import ContextRule, ContextUpdateLevel

from .curator import Curator
from .playbook import Playbook
from .reflector import Reflector
from .utils import log_operation


@dataclass
class ACEConfig:
    """
    Configuration for ACE context rule.

    Attributes:
        reflector_prompt_path: Path to reflector prompt template
        curator_prompt_path: Path to curator prompt template
        initial_playbook_path: Path to initial playbook
        save_playbook_path: Path to save updated playbook
        model_name: Model to use for Reflector/Curator (optional)
        temperature: Sampling temperature for LLM calls
        max_retries: Maximum retries for failed LLM calls
        save_interval: Save playbook every N trajectories (0 = only at end)
        ace_log_dir: Directory to save Reflector/Curator logs (None = no logging)
        reflector_only: Ablation that appends key insights without Curator review
    """

    reflector_prompt_path: str
    curator_prompt_path: str
    initial_playbook_path: str
    save_playbook_path: str | None = None
    model_name: str | None = None
    temperature: float = 0.0
    max_retries: int = 3
    save_interval: int = 0  # 0 means only save at end
    ace_log_dir: str | None = None  # Directory for Reflector/Curator logs
    reflector_only: bool = False  # Skip Curator; ADD reflection.key_insight directly


class ACEContextRule(ContextRule):
    """
    ACE context rule with Pathway i (parallel processing).

    Manages cross-episode learning via:
    1. Playbook: Accumulated knowledge injected into Generator prompts
    2. Reflector: Analyzes trajectories to extract insights
    3. Curator: Generates playbook operations from reflections

    Pathway i (Parallel Processing):
    - All Reflector calls run in parallel with same playbook snapshot
    - All Curator calls run in parallel with same playbook snapshot

    This trades potential within-batch redundancy for O(2) LLM round trips
    instead of O(2k) for k trajectories.

    Example:
        config = ACEConfig(
            reflector_prompt_path="prompts/reflector.txt",
            curator_prompt_path="prompts/curator.txt",
            initial_playbook_path="playbooks/initial.txt",
        )

        async def llm_call(messages):
            return await my_llm.chat(messages)

        rule = ACEContextRule(config, llm_call_fn=llm_call)

        # Get context for Generator
        playbook = rule.get_context()

        # After batch completes
        await rule.on_batch_complete(trajectories)

    Environment-Specific Adapter Example:
        from mbpp_adapter import MBPPReflectorAdapter

        adapter = MBPPReflectorAdapter(max_turns=5)
        rule = ACEContextRule(config, rollout_engine=engine, reflector_adapter=adapter)
    """

    def __init__(
        self,
        config: ACEConfig,
        llm_call_fn: Callable | None = None,
        rollout_engine: Any | None = None,
        reflector_adapter: Any | None = None,
        reflector_system_prompt: str = "",
        curator_system_prompt: str = "",
    ):
        """
        Initialize ACE context rule.

        Args:
            config: ACEConfig with paths and parameters
            llm_call_fn: Async function for LLM calls (preferred)
            rollout_engine: Alternative: rLLM RolloutEngine for LLM calls
            reflector_adapter: Optional ReflectorAdapter for environment-specific
                              trajectory formatting. If None, DefaultReflectorAdapter
                              is used (backward compatible).
            reflector_system_prompt / curator_system_prompt: Optional system messages
                              prepended to role LLM calls — set to the ACE-RL training
                              strings for train-faithful evals ("" = legacy user-only).

        One of llm_call_fn or rollout_engine must be provided.
        """
        self.config = config

        # Load prompt templates
        self.reflector_prompt = self._load_file(config.reflector_prompt_path)
        self.curator_prompt = "" if config.reflector_only else self._load_file(config.curator_prompt_path)

        # Initialize playbook
        initial_playbook = self._load_file(config.initial_playbook_path)
        self.playbook = Playbook(initial_playbook)

        # Set up LLM calling
        self.llm_call_fn = llm_call_fn
        self.rollout_engine = rollout_engine

        if not llm_call_fn and not rollout_engine:
            raise ValueError("Either llm_call_fn or rollout_engine must be provided")

        # Initialize Reflector with optional adapter
        self.reflector = Reflector(
            prompt_template=self.reflector_prompt,
            adapter=reflector_adapter,  # Pass adapter to Reflector
            rollout_engine=rollout_engine,
            model_name=config.model_name,
            max_retries=config.max_retries,
            system_prompt=reflector_system_prompt,
        )

        self.curator = None
        if not config.reflector_only:
            self.curator = Curator(
                prompt_template=self.curator_prompt,
                rollout_engine=rollout_engine,
                model_name=config.model_name,
                temperature=config.temperature,
                max_retries=config.max_retries,
                system_prompt=curator_system_prompt,
            )

        # Batch state
        self.pending_trajectories: list[dict] = []
        self.trajectories_processed = 0

        # Stats
        self.stats = {
            "batches_processed": 0,
            "trajectories_processed": 0,
            "operations_applied": 0,
            "operations_skipped": 0,
        }

        # Create log directory if specified
        if config.ace_log_dir:
            os.makedirs(config.ace_log_dir, exist_ok=True)

    # =========================================================================
    # ContextRule Interface
    # =========================================================================

    @property
    def update_level(self) -> ContextUpdateLevel:
        """ACE updates at batch level."""
        return ContextUpdateLevel.BATCH

    def get_context(self) -> str:
        """
        Return current playbook for injection into Generator prompt.

        The playbook contains accumulated knowledge from previous episodes.

        Returns:
            Current playbook content string
        """
        return self.playbook.to_string()

    def on_trajectory_complete(self, trajectory: "Trajectory") -> None:
        """
        Accumulate trajectory for batch processing.

        Called by ContextAwareRunner after each trajectory completes.
        Trajectories are stored until on_batch_complete is called.

        Args:
            trajectory: Completed trajectory object
        """
        reward = self._extract_reward(trajectory)

        traj_data = {
            "trajectory": trajectory,
            "reward": reward,
            "termination_reason": getattr(trajectory, "termination_reason", None),
        }

        self.pending_trajectories.append(traj_data)
        log_operation("ACE", f"Trajectory accumulated (reward={reward:.2f})")

    async def on_batch_complete(self, trajectories: list = None) -> None:
        """
        Process batch using Pathway i (parallel Reflector/Curator).

        Args:
            trajectories: Optional list of trajectories (uses pending if None)
        """
        # Use pending trajectories if none provided
        if trajectories is None:
            trajectories_to_process = self.pending_trajectories
        else:
            trajectories_to_process = [{"trajectory": t, "reward": self._extract_reward(t)} for t in trajectories]

        if not trajectories_to_process:
            log_operation("ACE", "No trajectories to process", success=False)
            return

        log_operation("ACE", f"Processing batch of {len(trajectories_to_process)} trajectories")

        # Snapshot playbook for this batch
        playbook_snapshot = self.playbook.to_string()

        # Phase 1: Parallel Reflector calls
        reflections, reflector_logs = await self._run_reflector_batch(trajectories_to_process, playbook_snapshot)

        # Phase 2: Curator, unless the reflector-only ablation is requested
        if self.config.reflector_only:
            operations_list, curator_logs = self._reflections_to_operations(reflections)
        else:
            operations_list, curator_logs = await self._run_curator_batch(reflections, trajectories_to_process, playbook_snapshot)

        # Phase 3: Apply operations sequentially
        self._apply_operations(operations_list)

        # Phase 4: Save logs if configured
        if self.config.ace_log_dir:
            self._save_batch_logs(reflector_logs, curator_logs, trajectories_to_process, self.stats["batches_processed"])

        # Update stats
        self.stats["batches_processed"] += 1
        self.stats["trajectories_processed"] += len(trajectories_to_process)
        self.trajectories_processed += len(trajectories_to_process)

        # Clear pending trajectories
        self.pending_trajectories = []

        # Save snapshot if configured
        if self.config.save_interval > 0:
            if self.trajectories_processed % self.config.save_interval == 0:
                self._save_snapshot()

        log_operation("ACE", f"Batch complete. Playbook: {self.playbook.get_stats()}")

    def reset_batch(self) -> None:
        """Reset batch state without processing."""
        self.pending_trajectories = []

    def save(self, path: str) -> None:
        """Save current playbook to file."""
        self.playbook.save(path)

    def load(self, path: str) -> None:
        """Load playbook from file."""
        self.playbook.load(path)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _load_file(self, path: str) -> str:
        """Load file contents."""
        with open(path) as f:
            return f.read()

    async def _run_reflector_batch(self, trajectories: list[dict], playbook_snapshot: str) -> tuple[list[dict], list[dict]]:
        """
        Run Reflector on all trajectories in parallel.

        All calls see the same playbook snapshot.

        Returns:
            Tuple of (reflections, logs):
            - reflections: List of reflection result dicts
            - logs: List of log dicts with prompt/response/parsed
        """
        tasks = [self.reflector.analyze(traj_data, playbook_snapshot) for traj_data in trajectories]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate results and logs, handle exceptions
        reflections = []
        logs = []

        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                log_operation("REFLECTOR", f"Failed for trajectory {i}: {r}", success=False)
                reflections.append({})
                logs.append({"prompt": "", "response": "", "parsed": {}, "error": str(r)})
            else:
                # New format: r = {"result": ..., "log": ...}
                reflections.append(r.get("result", {}))
                logs.append(r.get("log", {"prompt": "", "response": "", "parsed": {}}))

        return reflections, logs

    async def _run_curator_batch(self, reflections: list[dict], trajectories: list[dict], playbook_snapshot: str) -> tuple[list[dict], list[dict]]:
        """
        Run Curator on all reflections in parallel.

        All calls see the same playbook snapshot.

        Returns:
            Tuple of (operations_list, logs):
            - operations_list: List of operation result dicts
            - logs: List of log dicts with prompt/response/parsed (aligned with trajectories)
        """
        # Track which indices have valid reflections
        valid_indices = []
        tasks = []

        for i, (reflection, traj_data) in enumerate(zip(reflections, trajectories, strict=False)):
            if reflection:  # Skip empty reflections
                valid_indices.append(i)
                tasks.append(
                    self.curator.generate_operations(
                        traj_data,  # First arg: traj_data dict
                        reflection,  # Second arg: reflection dict
                        playbook_snapshot,  # Third arg: playbook snapshot string
                    )
                )

        # Initialize logs for all trajectories (including skipped ones)
        all_logs = [{"prompt": "", "response": "", "parsed": {}, "skipped": True} for _ in trajectories]

        if not tasks:
            return [], all_logs

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and update logs for valid indices
        operations_list = []

        for idx, (valid_idx, raw) in enumerate(zip(valid_indices, raw_results, strict=False)):
            if isinstance(raw, Exception):
                log_operation("CURATOR", f"Failed for reflection {idx}: {raw}", success=False)
                all_logs[valid_idx] = {"prompt": "", "response": "", "parsed": {}, "error": str(raw)}
            else:
                # New format: raw = {"result": ..., "log": ...}
                result = raw.get("result", {})
                log = raw.get("log", {"prompt": "", "response": "", "parsed": {}})
                log["skipped"] = False

                if result:
                    operations_list.append(result)

                all_logs[valid_idx] = log

        return operations_list, all_logs

    def _reflections_to_operations(self, reflections: list[dict]) -> tuple[list[dict], list[dict]]:
        """Append each nonempty insight in the reflector-only ablation.

        This bypasses Curator review, deduplication, and UPDATE/REMOVE choices.
        Section playbooks use "others"; flat playbooks ignore the section.
        Logs retain one entry per reflection, including empty or failed ones.
        """
        operations_list: list[dict] = []
        curator_logs: list[dict] = []

        for reflection in reflections:
            insight = reflection.get("key_insight") if isinstance(reflection, dict) else None
            key_insight = insight.strip() if isinstance(insight, str) else ""
            log = {"prompt": "", "response": "", "parsed": {}, "skipped": True, "reason": "reflector_only"}
            curator_logs.append(log)
            if not key_insight:
                log["empty_key_insight"] = True
                continue

            synthesized = {"operations": [{"type": "ADD", "section": "others", "content": key_insight}]}
            operations_list.append(synthesized)
            log["parsed"] = synthesized
            log["synthesized_op"] = synthesized

        return operations_list, curator_logs

    def _apply_operations(self, operations_list: list[dict]) -> None:
        """
        Apply operations sequentially; Curator review owns deduplication.
        """
        for operations in operations_list:
            if operations:
                applied = self.playbook.apply_operations(operations)
                self.stats["operations_applied"] += applied

    def _save_batch_logs(self, reflector_logs: list[dict], curator_logs: list[dict], trajectories: list[dict], batch_idx: int) -> None:
        """
        Save Reflector and Curator logs for each task in the batch.

        Creates a JSON file per task with structure:
        [
            {reflector log},
            {curator log}
        ]

        Args:
            reflector_logs: List of Reflector log dicts
            curator_logs: List of Curator log dicts
            trajectories: List of trajectory dicts
            batch_idx: Current batch index
        """
        batch_dir = os.path.join(self.config.ace_log_dir, f"batch_{batch_idx:04d}")
        os.makedirs(batch_dir, exist_ok=True)

        for i, (ref_log, cur_log) in enumerate(zip(reflector_logs, curator_logs, strict=False)):
            # Combine as list of 2 items
            task_log = [ref_log, cur_log]

            # Save to JSON file
            log_path = os.path.join(batch_dir, f"task_{i:04d}.json")
            with open(log_path, "w") as f:
                json.dump(task_log, f, indent=2, default=str)

        # Also save a batch summary
        summary = {
            "batch_idx": batch_idx,
            "num_tasks": len(trajectories),
            "reflector_logs_count": len(reflector_logs),
            "curator_logs_count": len(curator_logs),
            "curator_skipped": sum(1 for log in curator_logs if log.get("skipped", False)),
        }
        summary_path = os.path.join(batch_dir, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        log_operation("ACE", f"Saved logs for batch {batch_idx} to {batch_dir}")

    def _extract_reward(self, trajectory) -> float:
        """Extract reward from trajectory."""
        if hasattr(trajectory, "reward"):
            return float(trajectory.reward)

        if hasattr(trajectory, "steps") and trajectory.steps:
            last_step = trajectory.steps[-1]
            if hasattr(last_step, "reward"):
                return float(last_step.reward)

        return 0.0

    def _save_snapshot(self) -> None:
        """Save playbook snapshot."""
        if self.config.save_playbook_path:
            base, ext = os.path.splitext(self.config.save_playbook_path)
            snapshot_path = f"{base}_{self.trajectories_processed}{ext}"
            self.playbook.save(snapshot_path)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            **self.stats,
            "playbook_bullets": self.playbook.get_stats()["total_bullets"],
        }

    def get_playbook_stats(self) -> dict:
        """Get playbook statistics."""
        return self.playbook.get_stats()


# =============================================================================
# Factory Function
# =============================================================================


def create_ace_rule(
    reflector_prompt_path: str,
    curator_prompt_path: str,
    initial_playbook_path: str,
    llm_call_fn: Callable = None,
    rollout_engine: Any = None,
    reflector_adapter: Any = None,
    save_playbook_path: str | None = None,
    ace_log_dir: str | None = None,
    **kwargs,
) -> ACEContextRule:
    """
    Factory function to create ACE context rule.

    Args:
        reflector_prompt_path: Path to reflector prompt template
        curator_prompt_path: Path to curator prompt template
        initial_playbook_path: Path to initial playbook
        llm_call_fn: Async function for LLM calls
        rollout_engine: Alternative: rLLM RolloutEngine for LLM calls
        reflector_adapter: Optional ReflectorAdapter for environment-specific
                          trajectory formatting
        save_playbook_path: Optional path to save updated playbook
        ace_log_dir: Optional directory for Reflector/Curator logs
        **kwargs: Additional config options

    Returns:
        Configured ACEContextRule instance
    """
    config = ACEConfig(
        reflector_prompt_path=reflector_prompt_path,
        curator_prompt_path=curator_prompt_path,
        initial_playbook_path=initial_playbook_path,
        save_playbook_path=save_playbook_path,
        ace_log_dir=ace_log_dir,
        **kwargs,
    )

    return ACEContextRule(
        config,
        llm_call_fn=llm_call_fn,
        rollout_engine=rollout_engine,
        reflector_adapter=reflector_adapter,
    )
