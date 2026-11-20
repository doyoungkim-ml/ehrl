"""
ACE+RL Trainer: Extends AgentPPOTrainer to train policy, reflector, and curator
using GRPO policy gradient.

The rollout stage of GRPO is replaced with the full ACE pipeline:
  1. ACT:     Run policy rollouts with current playbook
  2. REFLECT: Generate insights from trajectories (single-turn LLM calls)
  3. CURATE:  Generate candidate playbooks from insights (single-turn LLM calls)
  4. EVALUATE: Score each candidate playbook
  5. SELECT:  Install best candidate
  6. COMBINE: Merge all trajectory dicts for GRPO update

Reflector/curator generations go through the same VeRL engine as policy rollouts,
enabling GRPO gradient updates for all three components.

"""

import asyncio
import copy
import importlib
import json
import logging
import os
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

from rllm.algorithms.ace.context.playbook import Playbook
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from rllm.utils.training_reward import raw_reward_metrics

logger = logging.getLogger(__name__)


def _coerce_bool(value, default: bool = False) -> bool:
    """Parse Hydra/env boolean-like values without depending on OmegaConf types."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return default


_VALID_REFLECTION_SOURCE_SELECTIONS = ("min_reward", "random")


def _coerce_reflection_source_selection(value) -> str:
    """Validate how singlemode picks the one policy rollout it reflects on."""
    selection = "random" if value is None else str(value).strip().lower()
    if selection not in _VALID_REFLECTION_SOURCE_SELECTIONS:
        raise ValueError(f"Unsupported ace.reflection_source_selection={value!r}; expected one of {list(_VALID_REFLECTION_SOURCE_SELECTIONS)}")
    return selection


def _reflection_source_reward(trajectory: dict) -> float:
    """Trajectory reward used to rank reflection sources; a missing reward counts as 0."""
    reward = trajectory.get("trajectory_reward")
    return 0.0 if reward is None else float(reward)


_VALID_POLICY_TRAIN_SOURCES = {"seed", "selected_candidate", "both", "seed_and_candidates"}
_POLICY_TRAIN_SOURCE_IDS = {
    "seed": 0,
    "selected_candidate": 1,
    "both": 2,
    "fallback_seed": 3,
    "seed_and_candidates": 4,
}


def _coerce_policy_train_source(value, default: str = "seed") -> str:
    """Normalize the ACE policy-training rollout source selector."""
    source = str(default if value is None else value).strip().lower().replace("-", "_")
    if not source:
        source = default
    if source not in _VALID_POLICY_TRAIN_SOURCES:
        raise ValueError(f"Unsupported ace.policy_train_source={source!r}; expected one of {sorted(_VALID_POLICY_TRAIN_SOURCES)}")
    return source


def _import_object(import_path: str):
    """Import ``module.attr`` paths from Hydra/env config values."""
    module_name, sep, attr_name = str(import_path).rpartition(".")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Expected a fully-qualified import path like 'pkg.module.Class', got {import_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _render_template(template: str, variables: dict) -> str:
    """Simple Jinja2-style template rendering using string replacement."""
    result = template
    for key, value in variables.items():
        result = result.replace("{{ " + key + " }}", str(value))
        result = result.replace("{{" + key + "}}", str(value))
    return result


def _parse_json_response(text: str):
    """Parse JSON from LLM response text, handling markdown code blocks."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    import re

    patterns = [
        r"```json\s*\n?(.*?)```",
        r"```\s*\n?(.*?)```",
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    return {}


def _truncate_response_text(text: str, limit: int = 240) -> str:
    text = (text or "").strip().replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _coerce_reflection_payload(payload, *, response_text: str = "", context: str = "reflector") -> dict:
    """Normalize reflector output to a single JSON object."""
    if isinstance(payload, dict):
        nested = payload.get("reflection")
        if isinstance(nested, dict):
            return nested
        return payload

    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, dict):
                logger.warning(
                    "[%s] Expected a JSON object but received a list; using dict element %s/%s. response=%s",
                    context,
                    idx + 1,
                    len(payload),
                    _truncate_response_text(response_text),
                )
                nested = item.get("reflection")
                if isinstance(nested, dict):
                    return nested
                return item
        logger.warning(
            "[%s] Expected a JSON object but received a list without any dict elements; ignoring response. response=%s",
            context,
            _truncate_response_text(response_text),
        )
        return {}

    if payload not in (None, ""):
        logger.warning(
            "[%s] Expected a JSON object but received %s; ignoring response. response=%s",
            context,
            type(payload).__name__,
            _truncate_response_text(response_text),
        )
    return {}


def _save_playbook_snapshot(
    playbook_text: str,
    snapshot_dir: str,
    global_step: int,
) -> str:
    """Save a playbook snapshot to disk."""
    os.makedirs(snapshot_dir, exist_ok=True)
    filename = f"playbook_step{global_step:04d}.txt"
    path = os.path.join(snapshot_dir, filename)
    with open(path, "w") as f:
        f.write(f"# Playbook Snapshot - Step {global_step}\n")
        f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"# Size: {len(playbook_text)} characters\n")
        f.write("#" + "=" * 60 + "\n\n")
        f.write(playbook_text)
    return path


def _concatenate_insights(insights: list[dict]) -> str:
    """Merge insights into single text block for curator (same as mbpp_ace)."""
    merged = []
    for i, insight_data in enumerate(insights):
        reflection = insight_data.get("reflection", {})
        task_idx = insight_data.get("task_idx", i)
        if not reflection:
            continue
        if not isinstance(reflection, dict):
            logger.warning(
                "[curator] Skipping malformed reflection for task %s: expected dict, got %s",
                task_idx,
                type(reflection).__name__,
            )
            continue
        merged.append(f"--- Insight {i + 1} (Task {task_idx}) ---")
        merged.append(f"Outcome: {reflection.get('outcome', 'UNKNOWN')}")
        reasoning = reflection.get("reasoning", "")
        if reasoning:
            merged.append(f"Reasoning: {reasoning}")
        key_insight = reflection.get("key_insight", "")
        if key_insight:
            merged.append(f"Key Insight: {key_insight}")
        failure_reason = reflection.get("failure_reason")
        if failure_reason:
            merged.append(f"Failure Reason: {failure_reason}")
        correct_approach = reflection.get("correct_approach")
        if correct_approach:
            merged.append(f"Correct Approach: {correct_approach}")
        merged.append("")
    return "\n".join(merged)


def _get_playbook_token_length(playbook_text: str, tokenizer) -> int:
    """Get playbook token length using the model tokenizer."""
    token_ids = tokenizer.encode(playbook_text, add_special_tokens=False)
    return len(token_ids)


class ACERLAgentPPOTrainer(AgentPPOTrainer):
    """Extends AgentPPOTrainer to run ACE pipeline during GRPO rollout.

    The generate_agent_trajectory() method is overridden to:
    1. Run policy rollouts with playbook-injected system prompts
    2. Generate reflector insights (single-turn LLM calls via VeRL engine)
    3. Generate curator operations (single-turn LLM calls via VeRL engine)
    4. Evaluate candidate playbooks
    5. Select best candidate and update playbook
    6. Combine all trajectory dicts with appropriate rewards and UIDs

    All three types of generations (policy, reflector, curator) go through
    the same VeRL engine and participate in GRPO gradient updates.
    """

    def __init__(self, *args, ace_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ace_config = ace_config or {}
        self._current_batch = None  # Reference to fit_agent()'s batch for in-place expansion

    @staticmethod
    def _parse_candidate_eval_rollouts(value) -> int:
        """Return per-candidate ace_evaluate rollout cap; 0 means use all."""
        if value is None:
            return 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "none", "null", "auto", "all"}:
                return 0
            value = normalized
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning(
                "[EVALUATE] Invalid ace.candidate_eval_rollouts=%r; using all policy rollouts",
                value,
            )
            return 0
        return max(0, parsed)

    def init_workers(self):
        super().init_workers()
        self._init_ace_components()

    def init_envs_and_agents(self, batch):
        """Override to store batch reference for later in-place expansion.

        The parent's fit_agent() calls batch.union(final_gen_batch_output) which
        requires matching row counts. In ACE+RL, generate_agent_trajectory() returns
        more rows (policy + reflector + curator). We store the batch reference here
        so generate_agent_trajectory() can expand it in-place before returning.
        """
        self._current_batch = batch
        return super().init_envs_and_agents(batch)

    def _init_ace_components(self):
        """Initialize ACE components: prompts, playbook, adapter."""
        ace = self.ace_config

        self.handoff_protocol = str(ace.get("handoff_protocol", "legacy")).strip().lower()
        if self.handoff_protocol not in {"legacy", "feedback_v1"}:
            raise ValueError(f"Unsupported ace.handoff_protocol={self.handoff_protocol!r}")
        if self.handoff_protocol == "feedback_v1" and not _coerce_bool(self.config.rllm.agent.get("preserve_terminal_feedback", False)):
            raise ValueError("feedback_v1 requires rllm.agent.preserve_terminal_feedback=true")

        # Benchmark assets belong to the caller, outside the algorithm package.
        for name in ("prompts_dir", "initial_playbook", "reflector_adapter_class"):
            if not ace.get(name):
                raise ValueError(f"ACE requires explicit ace.{name}; supply benchmark configuration at the entrypoint")
        for name in ("reflector_system_prompt", "curator_system_prompt"):
            if name not in ace:
                raise ValueError(f"ACE requires explicit ace.{name}; supply benchmark configuration at the entrypoint")
        prompts_dir = os.path.abspath(ace["prompts_dir"])
        initial_playbook_path = os.path.abspath(ace["initial_playbook"])
        base_prompt_path = os.path.join(prompts_dir, "base_system_prompt.txt")
        reflector_prompt_path = os.path.join(prompts_dir, "reflector_prompt.txt")
        handoff_reflector_path = os.path.join(prompts_dir, "reflector_prompt_handoff_v1.txt")
        if self.handoff_protocol == "feedback_v1" and os.path.isfile(handoff_reflector_path):
            reflector_prompt_path = handoff_reflector_path
        curator_prompt_name = "curator_prompt_handoff_v1.txt" if self.handoff_protocol == "feedback_v1" else "curator_prompt.txt"
        curator_prompt_path = os.path.join(prompts_dir, curator_prompt_name)
        self.prompt_paths = {
            "prompts_dir": prompts_dir,
            "base_system_prompt": base_prompt_path,
            "reflector_prompt": reflector_prompt_path,
            "curator_prompt": curator_prompt_path,
            "initial_playbook": initial_playbook_path,
        }

        # Load prompt templates
        with open(base_prompt_path) as f:
            self.base_system_prompt_template = f.read()
        with open(reflector_prompt_path) as f:
            self.reflector_prompt_template = f.read()
        with open(curator_prompt_path) as f:
            self.curator_prompt_template = f.read()

        # Load initial playbook
        with open(initial_playbook_path) as f:
            initial_playbook_text = f.read()
        self.initial_playbook_text = initial_playbook_text
        self.playbook = Playbook(initial_playbook_text)

        # Create environment-specific adapter for trajectory formatting.
        adapter_cls = _import_object(ace["reflector_adapter_class"])
        adapter_kwargs = {"max_turns": ace.get("max_turns", 5)}
        adapter_kwargs.update(ace.get("reflector_adapter_kwargs") or {})
        if "handoff_protocol" in adapter_kwargs and str(adapter_kwargs["handoff_protocol"]).strip().lower() != self.handoff_protocol:
            raise ValueError("reflector_adapter_kwargs.handoff_protocol conflicts with ace.handoff_protocol")
        if self.handoff_protocol == "feedback_v1":
            adapter_kwargs["handoff_protocol"] = self.handoff_protocol
        # Token-aware adapters opt in explicitly; class names carry no behavior.
        if _coerce_bool(ace.get("reflector_adapter_uses_tokenizer", False)):
            adapter_kwargs.update(
                tokenizer=self.tokenizer,
                max_history_tokens=ace.get("reflector_max_history_tokens"),
            )
        self.reflector_adapter = adapter_cls(**adapter_kwargs)
        self.reflector_system_prompt = ace["reflector_system_prompt"]
        self.curator_system_prompt = ace["curator_system_prompt"]

        # ACE hyperparameters
        self.g_ref = ace.get("g_ref", 4)
        self.g_cur = ace.get("g_cur", 4)
        self.singlemode_enabled = _coerce_bool(ace.get("singlemode", False))
        self.singlemode_curator_one_group = _coerce_bool(ace.get("singlemode_curator_one_group", False))
        self.reflection_source_selection = _coerce_reflection_source_selection(ace.get("reflection_source_selection", "random"))
        self.policy_train_source = _coerce_policy_train_source(ace.get("policy_train_source", "seed"))
        self.guidebook_aggregation_level = str(ace.get("guidebook_aggregation_level", "task") or "task").strip().lower()
        valid_guidebook_aggregation_levels = {"task", "single"}
        if self.guidebook_aggregation_level not in valid_guidebook_aggregation_levels:
            raise ValueError(f"Unsupported ace.guidebook_aggregation_level={self.guidebook_aggregation_level!r}; expected one of {sorted(valid_guidebook_aggregation_levels)}")
        if self.singlemode_enabled and self.guidebook_aggregation_level != "single":
            logger.info("[SINGLEMODE] Forcing guidebook_aggregation_level=single so one selected rollout yields per-insight guidebooks per task.")
            self.guidebook_aggregation_level = "single"
        if self.singlemode_curator_one_group and not self.singlemode_enabled:
            logger.info("[SINGLEMODE] singlemode_curator_one_group=True is only active when ace.singlemode=True; curator sampling will keep all insight groups.")
        self.selection_mode = ace.get("selection_mode", "best_rejection")
        valid_selection_modes = {
            "random",
            "best",
            "best_rejection",
            "best_rejection_update",
        }
        if self.selection_mode not in valid_selection_modes:
            raise ValueError(f"Unsupported ace.selection_mode={self.selection_mode!r}; expected one of {sorted(valid_selection_modes)}")
        self.auxiliary_update_gate = str(ace.get("auxiliary_update_gate", "none")).strip().lower()
        if self.auxiliary_update_gate not in {"none", "strict_improvement"}:
            raise ValueError(f"Unsupported ace.auxiliary_update_gate={self.auxiliary_update_gate!r}; expected 'none' or 'strict_improvement'")
        self.ace_temperature = ace.get("ace_temperature", 0.7)
        self.training_mode = "standard"
        self.policy_only_phase_enabled = False
        self.role_reduced_sampling = False
        self.flush_playbook_every_n_steps = int(ace.get("flush_playbook_every_n_steps", 30) or 0)
        self.flush_playbook_every_n_epochs = int(ace.get("flush_playbook_every_n_epochs", 0) or 0)
        self._last_playbook_flush_step = 0
        self.playbook_snapshot_dir = ace.get("playbook_snapshot_dir")
        self.chat_completion_dir = ace.get("chat_completion_dir")
        self.ace_log_dir = ace.get("ace_log_dir")
        self.max_playbook_length = ace.get("max_playbook_length")
        self.candidate_eval_batch_size = max(1, int(ace.get("candidate_eval_batch_size", 4) or 1))
        self.candidate_eval_rollouts = self._parse_candidate_eval_rollouts(ace.get("candidate_eval_rollouts", 0))
        try:
            self.candidate_eval_max_parallel_agents = int(ace.get("candidate_eval_max_parallel_agents", 256) or 0)
        except (TypeError, ValueError):
            self.candidate_eval_max_parallel_agents = 256
        if self.candidate_eval_max_parallel_agents < 0:
            self.candidate_eval_max_parallel_agents = 0
        self.component_token_limits = {
            "policy": {
                "max_prompt_length": int(ace.get("policy_max_prompt_length", self.config.data.max_prompt_length)),
                "max_response_length": int(ace.get("policy_max_response_length", self.config.data.max_response_length)),
            },
            "reflector": {
                "max_prompt_length": int(ace.get("reflector_max_prompt_length", self.config.data.max_prompt_length)),
                "max_response_length": int(ace.get("reflector_max_response_length", self.config.data.max_response_length)),
            },
            "curator": {
                "max_prompt_length": int(ace.get("curator_max_prompt_length", self.config.data.max_prompt_length)),
                "max_response_length": int(ace.get("curator_max_response_length", self.config.data.max_response_length)),
            },
        }

        self._steps_per_epoch = 0

        # Store original agents and envs for restoring after reflector/curator
        self._original_agents = None
        self._original_envs = None
        self._validation_policy_trajectory_cache = None
        self._last_candidate_eval_trajectories = None

        max_component_prompt = max(limits["max_prompt_length"] for limits in self.component_token_limits.values())
        max_component_response = max(limits["max_response_length"] for limits in self.component_token_limits.values())
        if self.config.data.max_prompt_length < max_component_prompt:
            logger.info(
                "[TOKLEN] batch prompt padding width=%s exceeds data.max_prompt_length=%s to preserve ACE component prompts; generation limits are unchanged.",
                max_component_prompt,
                self.config.data.max_prompt_length,
            )
        if self.config.data.max_response_length < max_component_response:
            logger.info(
                "[TOKLEN] batch response padding width=%s exceeds data.max_response_length=%s to preserve ACE component responses; generation limits are unchanged.",
                max_component_response,
                self.config.data.max_response_length,
            )

        if self.max_playbook_length is not None:
            initial_len = _get_playbook_token_length(self._get_current_playbook_text(), self.tokenizer)
            if initial_len > self.max_playbook_length:
                logger.info(
                    "[PLAYBOOK] Initial playbook exceeds max_playbook_length: %s > %s tokens",
                    initial_len,
                    self.max_playbook_length,
                )

        logger.info(
            f"ACE+RL initialized: g_ref={self.g_ref}, g_cur={self.g_cur}, "
            f"mode={self.training_mode}, selection={self.selection_mode}, "
            f"guidebook_aggregation_level={self.guidebook_aggregation_level}, "
            f"singlemode={self.singlemode_enabled}, "
            f"singlemode_curator_one_group={self.singlemode_curator_one_group}, "
            f"reflection_source_selection={self.reflection_source_selection}, "
            f"policy_train_source={self.policy_train_source}, "
            f"flush_every_n_steps={self.flush_playbook_every_n_steps}, "
            f"flush_every_n_epochs={self.flush_playbook_every_n_epochs}, "
            f"max_playbook_length={self.max_playbook_length}, "
            f"candidate_eval_batch_size={self.candidate_eval_batch_size}, "
            f"candidate_eval_rollouts={self.candidate_eval_rollouts or 'all'}, "
            f"candidate_eval_max_parallel_agents={self.candidate_eval_max_parallel_agents or 'auto'}"
        )
        logger.info(
            "[TOKLEN] component_limits policy=%s/%s reflector=%s/%s curator=%s/%s",
            self.component_token_limits["policy"]["max_prompt_length"],
            self.component_token_limits["policy"]["max_response_length"],
            self.component_token_limits["reflector"]["max_prompt_length"],
            self.component_token_limits["reflector"]["max_response_length"],
            self.component_token_limits["curator"]["max_prompt_length"],
            self.component_token_limits["curator"]["max_response_length"],
        )
        logger.info(
            "[ARTIFACTS] playbook_snapshots=%s chat_completions=%s ace_logs=%s",
            self.playbook_snapshot_dir,
            self.chat_completion_dir,
            self.ace_log_dir,
        )

    def _get_current_playbook_text(self) -> str:
        """Get current playbook as text."""
        return self.playbook.to_string()

    def _pop_validation_policy_trajectory_cache(self):
        cached = self._validation_policy_trajectory_cache
        self._validation_policy_trajectory_cache = None
        return cached

    def _store_validation_policy_trajectory_cache(self, trajectories):
        self._validation_policy_trajectory_cache = trajectories if trajectories else None

    def _clear_validation_policy_trajectory_cache(self):
        self._validation_policy_trajectory_cache = None

    def _make_guidebook_uid(
        self,
        *,
        task_idx: int,
        insight_idx: int,
        traj_idx: int | None = None,
        task_id: str | None = None,
    ) -> str:
        """Return guidebook grouping uid for reflector/curator alignment.

        ``task`` preserves the historical behavior: one guidebook aggregates the
        same insight sample index across all policy rollouts for a task.
        ``single`` creates one guidebook from one reflector insight.
        """
        prefix = f"{task_id}::" if task_id is not None else ""
        if self.guidebook_aggregation_level == "single":
            if traj_idx is None:
                raise ValueError("single guidebook aggregation requires traj_idx")
            return f"{prefix}gb_t{task_idx}_traj{traj_idx}_i{insight_idx}"
        return f"{prefix}gb_t{task_idx}_i{insight_idx}"

    def _uses_single_guidebook_aggregation(self) -> bool:
        return self.guidebook_aggregation_level == "single"

    def _select_reflection_source_indices(self, task_group: list[dict]) -> list[int]:
        """Pick policy rollout indices that will feed reflector/curator phases.

        Singlemode reflects on one uniformly random rollout per task by default.
        ``reflection_source_selection=min_reward`` chooses the lowest reward,
        with exact ties broken uniformly at random among the minima.
        """
        if not task_group:
            return []
        if not self.singlemode_enabled:
            return list(range(len(task_group)))
        if getattr(self, "reflection_source_selection", "random") == "random":
            return [random.randrange(len(task_group))]
        rewards = [_reflection_source_reward(trajectory) for trajectory in task_group]
        lowest = min(rewards)
        return [random.choice([index for index, reward in enumerate(rewards) if reward == lowest])]

    def _should_use_singlemode_curator_one_group(self) -> bool:
        """Whether only one sampled insight receives the full curator GRPO group."""
        return bool(self.singlemode_enabled and getattr(self, "singlemode_curator_one_group", False))

    def _expand_curator_prompt_specs_for_sampling(
        self,
        prompt_specs: list[dict],
        effective_g_cur: int,
    ) -> list[dict]:
        """Expand curator prompt specs into sampled curator rows.

        Historical single-guidebook sampling gives every reflector insight its
        own curator group: ``g_ref * g_cur`` curator calls per selected policy
        rollout.  The simpler singlemode variant keeps all reflector insights as
        candidate sources, but chooses one insight per task to receive the full
        curator GRPO group and samples each remaining insight once.
        """
        prompt_specs = [dict(spec) for spec in (prompt_specs or [])]
        if not prompt_specs:
            return []

        grouped_samples = max(1, int(effective_g_cur or 1))
        one_group_mode = self._should_use_singlemode_curator_one_group()
        selected_prompt_idx = random.randrange(len(prompt_specs)) if one_group_mode else None

        expanded: list[dict] = []
        for prompt_idx, spec in enumerate(prompt_specs):
            is_grouped_prompt = prompt_idx == selected_prompt_idx if one_group_mode else True
            sample_count = grouped_samples if is_grouped_prompt else 1
            for sample_idx in range(sample_count):
                expanded_spec = dict(spec)
                expanded_spec["curator_prompt_idx"] = prompt_idx
                expanded_spec["curator_sample_idx"] = sample_idx
                expanded_spec["curator_samples_for_prompt"] = sample_count
                expanded_spec["curator_grouped"] = is_grouped_prompt
                expanded_spec["curator_selected_for_group"] = is_grouped_prompt
                expanded_spec["curator_group_sampling_mode"] = "single_group_per_task" if one_group_mode else "all_groups"
                expanded.append(expanded_spec)
        return expanded

    @staticmethod
    def _get_candidate_seed_reward(candidate: dict, seed_policy_groups) -> float | None:
        """Return the seed rollout reward that produced a rollout-aligned candidate.

        Rollout-aligned candidates are generated from one particular policy rollout,
        identified by ``task_idx``/``traj_idx`` in the shared-playbook path and
        ``task_id``/``traj_idx`` in the local-playbook path.
        """
        try:
            if isinstance(seed_policy_groups, dict):
                seed_group = seed_policy_groups.get(candidate.get("task_id"), [])
            else:
                task_idx = int(candidate["task_idx"])
                seed_group = seed_policy_groups[task_idx]
            traj_idx = int(candidate["traj_idx"])
            return float(seed_group[traj_idx].get("trajectory_reward", 0.0))
        except (IndexError, KeyError, TypeError, ValueError):
            return None

    def _get_best_rejection_update_candidate_indices(
        self,
        candidate_playbooks: list[dict],
        candidate_rewards: list[float],
        seed_policy_groups,
        candidate_indices: list[int] | None = None,
    ) -> list[int]:
        """Keep candidates whose mean evaluation reward matches their seed rollout.

        A candidate without an unambiguous source rollout is excluded: accepting it
        would bypass the per-rollout comparison required by
        ``best_rejection_update``.
        """
        if candidate_indices is None:
            candidate_indices = list(range(len(candidate_playbooks)))

        surviving_indices = []
        for candidate_idx in candidate_indices:
            if candidate_idx >= len(candidate_rewards):
                continue
            seed_reward = self._get_candidate_seed_reward(
                candidate_playbooks[candidate_idx],
                seed_policy_groups,
            )
            if seed_reward is None:
                logger.warning(
                    "[ACE][SELECT] Cannot match candidate %s to its seed rollout; excluding it from best_rejection_update selection.",
                    candidate_idx,
                )
                continue
            if float(candidate_rewards[candidate_idx]) >= seed_reward:
                surviving_indices.append(candidate_idx)
        return surviving_indices

    @staticmethod
    def _select_best_rejection_update_candidate_index(candidate_rewards: list[float], candidate_indices: list[int]) -> int:
        """Pick a maximum from nonempty survivors, randomizing only exact ties."""
        best_reward = max(float(candidate_rewards[idx]) for idx in candidate_indices)
        best_indices = [idx for idx in candidate_indices if float(candidate_rewards[idx]) == best_reward]
        return best_indices[0] if len(best_indices) == 1 else random.choice(best_indices)

    def _should_capture_selected_candidate_policy_trajectories(self) -> bool:
        return self.policy_train_source in {"selected_candidate", "both", "seed_and_candidates"}

    @staticmethod
    def _compute_rollout_token_totals(
        policy_trajectories=None,
        reflector_trajectories=None,
        curator_trajectories=None,
        candidate_eval_token_metrics=None,
        ppo_trajectories=None,
    ) -> dict:
        """Compute generated-token totals and PPO-token totals for one step.

        Candidate evaluation normally does not retain full trajectories when
        policy training uses seed samples. Its metrics belong in generated
        prompt/completion totals, but not in the PPO gradient-token total.
        """

        def _sum_token(traj_list, key):
            total = 0
            for t in traj_list:
                m = t.get("metrics", {}) if isinstance(t, dict) else {}
                v = m.get(key, 0)
                if v is not None and v > 0:
                    total += int(v)
            return total

        eval_metric_stubs = [{"metrics": metrics} for metrics in (candidate_eval_token_metrics or [])]

        all_rollout_trajs = list(policy_trajectories or []) + list(reflector_trajectories or []) + list(curator_trajectories or []) + eval_metric_stubs
        if ppo_trajectories is None:
            ppo_trajectories = list(policy_trajectories or []) + list(reflector_trajectories or []) + list(curator_trajectories or [])

        return {
            "rollout/total_prompt_tokens": _sum_token(all_rollout_trajs, "total_prompt_tokens"),
            "rollout/total_completion_tokens": _sum_token(all_rollout_trajs, "total_completion_tokens"),
            "rollout/total_gradient_update_tokens": _sum_token(ppo_trajectories, "total_gradient_update_tokens"),
            "rollout/total_trajectories": len(all_rollout_trajs),
        }

    @staticmethod
    def _curator_mutation_metrics(records) -> dict:
        reports = [row["mutation"] for row in records if isinstance(row.get("mutation"), dict)]
        return {
            "ace/curator_mutation_requests": sum(row["requested_count"] for row in reports),
            "ace/curator_applied_operations": sum(row["applied_count"] for row in reports),
            "ace/curator_changed_proposals": sum(bool(row["changed"]) for row in reports),
            "ace/curator_unchanged_proposals": sum(not row["changed"] for row in reports),
            "ace/curator_normalized_aliases": sum(row["normalized_tip_id_count"] for row in reports),
        }

    def _uses_auxiliary_update_gate(self, meta_info=None) -> bool:
        """Only standard training admission changes; validation/delayed ACE do not."""
        return (
            getattr(self, "auxiliary_update_gate", "none") == "strict_improvement"
            and getattr(self, "training_mode", "standard") in {"standard", "iterative"}
            and not bool(meta_info and meta_info.get("validate", False))
        )

    def _selected_playbook_improvement(
        self,
        selected_idx: int,
        selected_reward: float,
        current_reward: float,
        candidate_playbooks: list[dict],
        curator_records: list[dict],
        seed_policy_groups,
    ) -> tuple[bool, float | None]:
        """Compare the installed candidate with the baseline used for selection.

        Rejection-update selection compares each candidate to its own source
        rollout, whose provenance lives in the curator record. Other selection
        modes use the pre-update policy mean. A tie admits no auxiliary rows.
        This observes selection; it never changes the installed playbook.
        """
        if selected_idx < 0 or selected_idx >= len(candidate_playbooks):
            return False, None
        baseline = float(current_reward)
        if self.selection_mode == "best_rejection_update":
            curator_idx = candidate_playbooks[selected_idx].get("curator_idx")
            if not isinstance(curator_idx, int) or not 0 <= curator_idx < len(curator_records):
                return False, None
            baseline = self._get_candidate_seed_reward(curator_records[curator_idx], seed_policy_groups)
        return baseline is not None and float(selected_reward) > baseline, baseline

    @staticmethod
    def _prepare_auxiliary_training_rows(
        reflector_trajectories: list[dict],
        curator_trajectories: list[dict],
        all_reflections: list[dict],
        curator_records: list[dict],
        curator_to_guidebook: list[str],
        *,
        admitted_task_ids: set | None = None,
    ) -> tuple[list[dict], list[dict], list[str], list[str]]:
        """Filter whole task groups, retaining original row/UID correspondence.

        ``None`` preserves all rows. An empty set excludes all auxiliaries.
        Generated trajectories and provenance lists remain intact for artifacts
        and token accounting, including failed/invalid candidates in each group.
        """
        ref_indices = [
            i
            for i in range(len(reflector_trajectories))
            if admitted_task_ids is None or (i < len(all_reflections) and all_reflections[i].get("task_id") in admitted_task_ids)
        ]
        cur_indices = [
            i
            for i in range(len(curator_trajectories))
            if admitted_task_ids is None or (i < len(curator_records) and curator_records[i].get("task_id") in admitted_task_ids)
        ]
        reflector_uids = [all_reflections[i].get("reflector_uid", f"ref_{i}") if i < len(all_reflections) else f"ref_{i}" for i in ref_indices]
        curator_uids = [f"cur_{curator_to_guidebook[i]}" if i < len(curator_to_guidebook) else f"cur_gb_{i}" for i in cur_indices]
        return [reflector_trajectories[i] for i in ref_indices], [curator_trajectories[i] for i in cur_indices], reflector_uids, curator_uids

    @staticmethod
    def _auxiliary_training_metrics(gate_active, admitted_tasks, total_tasks, policy_rows, reflector_uids, curator_uids) -> dict:
        """Admission counts precede generic all-equal-group filtering/padding."""
        return {
            "ace/auxiliary_update_gate_active": int(gate_active),
            "ace/auxiliary_tasks_admitted": admitted_tasks,
            "ace/auxiliary_tasks_dropped": total_tasks - admitted_tasks,
            "ace/policy_training_rows": policy_rows,
            "ace/reflector_training_rows": len(reflector_uids),
            "ace/curator_training_rows": len(curator_uids),
            "ace/reflector_training_groups": len(set(reflector_uids)),
            "ace/curator_training_groups": len(set(curator_uids)),
        }

    def _policy_train_source_metric_id(self, source: str | None = None) -> int:
        return _POLICY_TRAIN_SOURCE_IDS.get(source or self.policy_train_source, -1)

    def _select_policy_trajectories_for_training(
        self,
        seed_policy_trajectories: list[dict],
        selected_policy_trajectories: list[dict] | None,
        *,
        expected_policy_count: int | None = None,
        log_prefix: str = "[ACE][SELECT]",
    ) -> tuple[list[dict], str]:
        """Choose policy rows for GRPO according to ace.policy_train_source.

        seed              -> train policy on the current/pre-update playbook rows.
        selected_candidate-> train policy on rows collected under the selected
                             updated playbook, falling back to seed if missing.
        both              -> train on seed rows plus selected updated-playbook rows.

        ``seed_and_candidates`` reuses every candidate-scoring rollout as a
        policy row in the seed group; only the local standard trainer builds
        those rows, so this selector rejects it.
        """
        if self.policy_train_source == "seed_and_candidates":
            raise ValueError("ace.policy_train_source=seed_and_candidates is supported only by the local standard ACE-RL trainer")
        seed_policy_trajectories = seed_policy_trajectories or []
        selected_policy_trajectories = selected_policy_trajectories or []
        seed_rows = [copy.deepcopy(traj) for traj in seed_policy_trajectories]

        if self.policy_train_source == "seed":
            return seed_rows, "seed"

        if expected_policy_count is None:
            expected_policy_count = len(seed_policy_trajectories)

        if selected_policy_trajectories and len(selected_policy_trajectories) == expected_policy_count:
            selected_rows = [copy.deepcopy(traj) for traj in selected_policy_trajectories]
            if self.policy_train_source == "selected_candidate":
                return selected_rows, "selected_candidate"
            return seed_rows + selected_rows, "both"

        logger.warning(
            "%s policy_train_source=%s requested selected-candidate policy rollouts (expected=%s got=%s); falling back to seed policy trajectories.",
            log_prefix,
            self.policy_train_source,
            expected_policy_count,
            len(selected_policy_trajectories),
        )
        return seed_rows, "fallback_seed"

    def _append_policy_and_component_rows_to_batch(
        self,
        batch,
        *,
        extra_policy_count: int = 0,
        component_extra_uids: list[str] | None = None,
        extra_policy_base_indices: list[int] | None = None,
        share_extra_policy_uids: bool = False,
    ) -> None:
        """Append non-tensor metadata rows to match extra generated trajectories.

        Existing parent batches already contain seed policy rows. ``both`` adds
        selected-candidate policy rows before reflector/curator rows, so duplicate
        the seed metadata for those extra policy rows and then pad component rows.
        ``extra_policy_base_indices`` names the seed row each extra policy row
        copies; ``share_extra_policy_uids`` keeps that row's uid so the extra
        row joins the seed GRPO group instead of forming its own group.
        """
        if batch is None:
            return
        extra_policy_count = max(0, int(extra_policy_count or 0))
        component_extra_uids = list(component_extra_uids or [])
        n_component = len(component_extra_uids)
        n_extra = extra_policy_count + n_component
        if n_extra <= 0:
            return

        if "uid" in batch.non_tensor_batch:
            current_size = len(batch.non_tensor_batch["uid"])
        else:
            first_key = next(iter(batch.non_tensor_batch.keys()))
            current_size = len(batch.non_tensor_batch[first_key])
        combined_size = current_size + n_extra

        if extra_policy_base_indices is None:
            extra_policy_base_indices = list(range(extra_policy_count))
        if len(extra_policy_base_indices) != extra_policy_count:
            raise ValueError(f"Expected {extra_policy_count} extra policy base indices, got {len(extra_policy_base_indices)}")
        uid_values = batch.non_tensor_batch.get("uid")
        policy_uids = []
        for i, base_index in enumerate(extra_policy_base_indices):
            base_uid = str(uid_values[base_index % len(uid_values)]) if uid_values is not None and len(uid_values) else f"policy_{i}"
            policy_uids.append(base_uid if share_extra_policy_uids else f"{base_uid}::policy_selected_{i}")
        extra_uids = np.array(policy_uids + component_extra_uids, dtype=object)

        def _empty_padding_like(original: np.ndarray, count: int) -> np.ndarray:
            shape = (count,) + tuple(original.shape[1:])
            if original.dtype.kind in ("U", "S", "O"):
                padding = np.empty(shape, dtype=original.dtype)
                padding[...] = ""
                return padding
            return np.zeros(shape, dtype=original.dtype)

        def _policy_padding_like(original: np.ndarray, count: int) -> np.ndarray:
            if not len(original):
                return _empty_padding_like(original, count)
            # Object vectors commonly store per-row arrays/lists.  np.array([...],
            # dtype=object) can still promote those into a 2-D object matrix, so
            # allocate a vector explicitly and assign element-by-element.
            if original.dtype.kind == "O" and original.ndim == 1:
                padding = np.empty((count,), dtype=original.dtype)
            else:
                padding = np.empty((count,) + tuple(original.shape[1:]), dtype=original.dtype)
            for i, base_index in enumerate(extra_policy_base_indices):
                padding[i] = copy.deepcopy(original[base_index % len(original)])
            return padding

        for key in list(batch.non_tensor_batch.keys()):
            original = batch.non_tensor_batch[key]
            parts = [original]
            if key == "uid":
                parts.append(extra_uids)
            else:
                if extra_policy_count:
                    parts.append(_policy_padding_like(original, extra_policy_count))
                if n_component:
                    parts.append(_empty_padding_like(original, n_component))
            batch.non_tensor_batch[key] = np.concatenate(parts, axis=0)

        from tensordict import TensorDict as TD

        batch.batch = TD({}, batch_size=[combined_size])

    def _get_active_component(self) -> str:
        return "all"

    def _should_run_policy_only_phase(self, active_component: str) -> bool:
        if active_component == "policy" and self.policy_train_source != "seed":
            return False
        return bool(self.policy_only_phase_enabled and active_component == "policy")

    def _should_use_role_reduced_sampling(self) -> bool:
        return bool(self.role_reduced_sampling)

    def _should_use_policy_only_validation(self, meta_info=None) -> bool:
        return True

    def _get_validation_rollout_kwargs(self):
        rollout_cfg = getattr(self.config.actor_rollout_ref, "rollout", None)
        if rollout_cfg is None:
            return {}
        val_kwargs = rollout_cfg.get("val_kwargs", None)
        return val_kwargs or {}

    def _get_eval_g_ref(self, meta_info=None) -> int:
        if meta_info and meta_info.get("validate", False):
            val_kwargs = self._get_validation_rollout_kwargs()
            return max(1, int(val_kwargs.get("g_ref", self.g_ref) or self.g_ref))
        return max(1, int(self.g_ref or 1))

    def _get_eval_g_cur(self, meta_info=None) -> int:
        if meta_info and meta_info.get("validate", False):
            val_kwargs = self._get_validation_rollout_kwargs()
            return max(1, int(val_kwargs.get("g_cur", self.g_cur) or self.g_cur))
        return max(1, int(self.g_cur or 1))

    def _advance_component_schedule(self):
        return

    def _artifact_mode(self, meta_info=None) -> str:
        return "eval" if bool(meta_info and meta_info.get("validate", False)) else "train"

    def _get_playbook_snapshot_dir(self, meta_info=None) -> str | None:
        if not self.playbook_snapshot_dir:
            return None
        if bool(meta_info and meta_info.get("validate", False)):
            return None
        return self.playbook_snapshot_dir

    def _get_chat_completion_dir(self, meta_info=None) -> str | None:
        if not self.chat_completion_dir:
            return None
        return os.path.join(self.chat_completion_dir, self._artifact_mode(meta_info))

    def _get_ace_log_dir(self, meta_info=None) -> str | None:
        if not self.ace_log_dir:
            return None
        return os.path.join(self.ace_log_dir, self._artifact_mode(meta_info))

    def _json_default(self, value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Playbook):
            return str(value)
        return str(value)

    def _write_json(self, path: str, payload: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=self._json_default)

    def _save_policy_chat_completions(self, policy_trajectories: list[dict], bsz: int, meta_info=None):
        chat_dir = self._get_chat_completion_dir(meta_info)
        if not chat_dir:
            return None

        payload = {
            "step": int(self.global_steps),
            "mode": self._artifact_mode(meta_info),
            "num_trajectories": len(policy_trajectories),
            "trajectories": [],
        }
        for i, traj in enumerate(policy_trajectories):
            payload["trajectories"].append(
                {
                    "idx": i,
                    "task_idx": i % bsz if bsz > 0 else i,
                    "trajectory_reward": float(traj.get("trajectory_reward", 0.0)),
                    "step_rewards": [float(x) for x in traj.get("step_rewards", [])],
                    "termination_reason": traj.get("termination_reason"),
                    "chat_completions": traj.get("chat_completions", []),
                    "response_text": self._decode_trajectory_response(traj),
                }
            )

        path = os.path.join(chat_dir, f"policy_step{self.global_steps:06d}.json")
        self._write_json(path, payload)
        return path

    def _save_ace_step_log(self, payload: dict, meta_info=None):
        ace_dir = self._get_ace_log_dir(meta_info)
        if not ace_dir:
            return None
        path = os.path.join(ace_dir, f"ace_step{self.global_steps:06d}.json")
        self._write_json(path, payload)
        return path

    def _save_playbook_snapshot_artifact(self, meta_info=None):
        snapshot_dir = self._get_playbook_snapshot_dir(meta_info)
        if not snapshot_dir:
            return None
        return _save_playbook_snapshot(
            self._get_current_playbook_text(),
            snapshot_dir,
            self.global_steps,
        )

    def _set_component_token_limits(self, component: str):
        """Apply component-specific prompt/response limits to the execution engine."""
        limits = self.component_token_limits[component]
        max_prompt_length = limits["max_prompt_length"]
        max_response_length = limits["max_response_length"]

        self.agent_execution_engine.max_prompt_length = max_prompt_length
        self.agent_execution_engine.max_response_length = max_response_length

        rollout_engine = getattr(self.agent_execution_engine, "rollout_engine", None)
        if rollout_engine is not None:
            if hasattr(rollout_engine, "max_prompt_length"):
                rollout_engine.max_prompt_length = max_prompt_length
            if hasattr(rollout_engine, "max_response_length"):
                rollout_engine.max_response_length = max_response_length

        logger.info(
            "[TOKLEN] step=%s component=%s using max_prompt_length=%s max_response_length=%s",
            self.global_steps,
            component,
            max_prompt_length,
            max_response_length,
        )

    def _render_base_system_prompt(self, playbook_text: str) -> str:
        """Render the base system prompt with the current playbook."""
        return _render_template(self.base_system_prompt_template, {"playbook": playbook_text})

    def _apply_playbook_prompt_to_agent(
        self,
        agent,
        *,
        playbook_text: str,
        rendered_prompt: str,
    ) -> None:
        """Apply the current playbook prompt to one policy agent.

        MBPP-style agents read a rendered ``SYSTEM_PROMPT`` during ``reset``.
        AppWorld-style agents need the unrendered Jinja template plus the
        playbook so task/user fields can be rendered after env reset.  Agents
        can opt in to that richer path via ``set_ace_playbook_prompt``.
        """
        setter = getattr(agent, "set_ace_playbook_prompt", None)
        if callable(setter):
            setter(
                template=self.base_system_prompt_template,
                playbook=playbook_text,
                rendered_prompt=rendered_prompt,
            )
            return

        agent.SYSTEM_PROMPT = rendered_prompt
        # Also update if already initialized. The execution engine will call
        # reset() next, but candidate-eval helpers may reuse fresh instances.
        if hasattr(agent, "messages") and agent.messages:
            agent.messages[0] = {"role": "system", "content": rendered_prompt}

    def _inject_playbook_into_agents(self, agents, playbook_text: str):
        """Inject the current playbook into agent prompts."""
        rendered_prompt = self._render_base_system_prompt(playbook_text)
        for agent in agents:
            self._apply_playbook_prompt_to_agent(
                agent,
                playbook_text=playbook_text,
                rendered_prompt=rendered_prompt,
            )

    def _get_full_agent_args(self) -> dict:
        return dict(self.config.rllm.agent.get("agent_args", {})) | dict(self.agent_args)

    def _get_full_env_args(self) -> dict:
        return dict(self.config.rllm.env.get("env_args", {})) | dict(self.env_args)

    def _get_candidate_eval_source_envs(self, source_envs) -> list:
        """Return source envs to clone for one candidate during ace_evaluate."""
        selected_envs = list(source_envs or [])
        rollout_cap = self._parse_candidate_eval_rollouts(getattr(self, "candidate_eval_rollouts", 0))
        if rollout_cap > 0:
            selected_envs = selected_envs[:rollout_cap]
        return selected_envs

    def _build_candidate_eval_envs_and_agents(self, candidate_batch: list[tuple[int, dict]]) -> tuple[list, list, list[dict]]:
        """Clone policy envs/agents so multiple candidates can be scored in one rollout pass."""
        if not candidate_batch or not self._original_envs:
            return [], [], []

        full_agent_args = self._get_full_agent_args()
        base_env_args = self._get_full_env_args()
        eval_envs = []
        eval_agents = []
        eval_specs = []

        for candidate_idx, candidate in candidate_batch:
            candidate_agents = []
            candidate_envs = []
            source_envs = self._get_candidate_eval_source_envs(self._original_envs)
            for rollout_idx, source_env in enumerate(source_envs):
                task_payload = copy.deepcopy(getattr(source_env, "task", None))
                if not isinstance(task_payload, dict):
                    raise TypeError(f"Candidate evaluation requires env.task to be a dict payload, but got {type(task_payload).__name__}")
                candidate_envs.append(self.env_class.from_dict({**task_payload, **base_env_args}))
                candidate_agents.append(self.agent_class(**full_agent_args))
                eval_specs.append(
                    {
                        "candidate_idx": candidate_idx,
                        "guidebook_uid": candidate["guidebook_uid"],
                        "rollout_idx": rollout_idx,
                    }
                )

            self._inject_playbook_into_agents(
                candidate_agents,
                candidate["playbook"].to_string(),
            )
            eval_envs.extend(candidate_envs)
            eval_agents.extend(candidate_agents)

        return eval_envs, eval_agents, eval_specs

    def _get_candidate_eval_parallel_agents(self, base_parallel_agents: int, num_eval_envs: int) -> int:
        target_parallel_agents = max(base_parallel_agents, num_eval_envs)
        if self.candidate_eval_max_parallel_agents > 0:
            target_parallel_agents = min(
                target_parallel_agents,
                self.candidate_eval_max_parallel_agents,
            )
            target_parallel_agents = max(base_parallel_agents, target_parallel_agents)
        return max(1, target_parallel_agents)

    def _evaluate_candidate_pool(
        self,
        candidate_playbooks: list[dict],
        build_envs_and_agents,
        *,
        capture_eval_trajectories: bool,
        timing_raw=None,
        meta_info=None,
    ) -> tuple[list[float], list[dict], dict[int, list[dict]], list[dict]]:
        """Score candidates in one replenished rollout pass at the existing cap.

        Logical candidate batches still determine capacity and record metadata,
        but no longer create barriers between trajectories. Environment builders
        construct lightweight objects; scarce resources are acquired by reset()
        inside the execution engine's semaphore.
        """
        if not candidate_playbooks:
            return [], [], {}, []
        if not self._original_envs or not self._original_agents:
            raise RuntimeError("Candidate evaluation requires original policy envs and agents")
        if timing_raw is None:
            timing_raw = {}

        engine = self.agent_execution_engine
        original_parallel_agents = int(engine.n_parallel_agents)
        batch_size = max(1, self.candidate_eval_batch_size)
        indexed_candidates = list(enumerate(candidate_playbooks))
        eval_envs, eval_agents, eval_specs = [], [], []
        batch_candidates_by_index = {}
        eval_parallel_agents = 0
        completed_indices = set()
        generation_started = False
        externally_managed_rollout_engine = False
        rollout_engine = getattr(engine, "rollout_engine", None)
        rewards_by_candidate = {}
        trajectories_by_candidate = {}
        token_metrics_by_index = {}
        candidate_rewards = [0.0 for _ in candidate_playbooks]

        try:
            for batch_start in range(0, len(indexed_candidates), batch_size):
                candidate_batch = indexed_candidates[batch_start : batch_start + batch_size]
                envs, agents, specs = build_envs_and_agents(candidate_batch)
                eval_envs.extend(envs)
                eval_agents.extend(agents)
                eval_specs.extend(specs)
                if not (len(envs) == len(agents) == len(specs)):
                    raise ValueError("Candidate evaluation requires one agent and specification per environment")
                if not envs:
                    continue
                # The old pass could run at most the number of rows in its
                # chunk, even when the configured base parallelism was larger.
                chunk_parallel = self._get_candidate_eval_parallel_agents(original_parallel_agents, len(envs))
                eval_parallel_agents = max(eval_parallel_agents, min(chunk_parallel, len(envs)))
                for candidate_idx, _ in candidate_batch:
                    batch_candidates_by_index[candidate_idx] = len(candidate_batch)

            if not eval_envs:
                return candidate_rewards, [], {}, []

            manage_rollout_engine_state = True
            if getattr(engine, "engine_name", None) == "verl" and rollout_engine is not None:
                if getattr(self, "_rollout_engine_external_management_depth", 0) > 0:
                    manage_rollout_engine_state = False
                else:
                    try:
                        asyncio.run(rollout_engine.wake_up())
                        manage_rollout_engine_state = False
                        externally_managed_rollout_engine = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[EVALUATE] Failed to pre-wake rollout engine for candidate scoring: %s", exc)

            logger.info(
                "[EVALUATE] candidate_pool=%s eval_rollouts=%s parallel_agents=%s",
                len(candidate_playbooks),
                len(eval_envs),
                eval_parallel_agents,
            )
            engine.n_parallel_agents = eval_parallel_agents
            engine.update_envs_and_agents(eval_envs, eval_agents)
            invalid_result = None
            generation_started = True
            for traj in self.generate_agent_trajectories_async(
                timing_raw=timing_raw,
                meta_info=meta_info,
                mode="Token",
                manage_rollout_engine_state=manage_rollout_engine_state,
            ):
                eval_idx = traj.get("idx")
                if not isinstance(eval_idx, int) or not 0 <= eval_idx < len(eval_specs) or eval_idx in completed_indices:
                    # Drain the producer before restoring its mutable engine
                    # bindings; never abandon still-running trajectories here.
                    invalid_result = invalid_result or f"Invalid or duplicate candidate evaluation index: {eval_idx!r}"
                    continue
                completed_indices.add(eval_idx)
                spec = eval_specs[eval_idx]
                candidate_idx = int(spec["candidate_idx"])
                rollout_idx = int(spec.get("rollout_idx", 0))
                rewards_by_candidate.setdefault(candidate_idx, []).append((rollout_idx, float(traj.get("trajectory_reward", 0.0))))
                token_metrics_by_index[eval_idx] = dict(traj.get("metrics", {}))
                if capture_eval_trajectories:
                    trajectories_by_candidate.setdefault(candidate_idx, []).append((rollout_idx, copy.deepcopy(traj)))

            if invalid_result is not None:
                raise ValueError(invalid_result)
            if len(completed_indices) != len(eval_specs):
                raise ValueError(f"Expected {len(eval_specs)} candidate evaluations but received {len(completed_indices)}")
        finally:
            # Async cancellation does not stop an already-running environment
            # operation in the engine's custom executor. Join it before closing
            # its resources or restoring the mutable environment bindings.
            executor = getattr(engine, "executor", None)
            if generation_started and executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            engine.n_parallel_agents = original_parallel_agents
            engine.update_envs_and_agents(self._original_envs, self._original_agents)
            # Successful trajectories close their environments before yielding.
            # Release resources left by a failed or cancelled candidate pass.
            for eval_idx, env in enumerate(eval_envs):
                if eval_idx not in completed_indices:
                    try:
                        env.close()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[EVALUATE] Failed to close candidate environment %s: %s", eval_idx, exc)
            if externally_managed_rollout_engine:
                try:
                    asyncio.run(rollout_engine.sleep())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[EVALUATE] Failed to put rollout engine back to sleep after candidate scoring: %s", exc)

        candidate_eval_records = []
        for candidate_idx, logical_batch_size in batch_candidates_by_index.items():
            candidate = candidate_playbooks[candidate_idx]
            eval_rewards = [reward for _, reward in sorted(rewards_by_candidate.get(candidate_idx, []))]
            mean_reward = float(np.mean(eval_rewards)) if eval_rewards else 0.0
            candidate_rewards[candidate_idx] = mean_reward
            candidate_eval_records.append(
                {
                    "candidate_idx": candidate_idx,
                    "guidebook_uid": candidate["guidebook_uid"],
                    "reward": mean_reward,
                    "eval_rollouts": len(eval_rewards),
                    "eval_batch_candidates": logical_batch_size,
                    "eval_parallel_agents": eval_parallel_agents,
                }
            )
        ordered_trajectories = {candidate_idx: [traj for _, traj in sorted(entries, key=lambda item: item[0])] for candidate_idx, entries in sorted(trajectories_by_candidate.items())}
        token_metrics = [token_metrics_by_index[idx] for idx in range(len(eval_specs))]
        return candidate_rewards, candidate_eval_records, ordered_trajectories, token_metrics

    def _evaluate_candidate_playbooks(
        self,
        candidate_playbooks: list[dict],
        timing_raw=None,
        meta_info=None,
    ) -> tuple[list[float], list[dict]]:
        capture_eval_trajectories = bool(meta_info and meta_info.get("validate", False)) or self._should_capture_selected_candidate_policy_trajectories()
        self._last_candidate_eval_trajectories = None
        rewards, records, trajectories, token_metrics = self._evaluate_candidate_pool(
            candidate_playbooks,
            self._build_candidate_eval_envs_and_agents,
            capture_eval_trajectories=capture_eval_trajectories,
            timing_raw=timing_raw,
            meta_info=meta_info,
        )
        if capture_eval_trajectories:
            self._last_candidate_eval_trajectories = trajectories
        self._last_candidate_eval_token_metrics = token_metrics
        return rewards, records

    def _build_curator_prompt(self, merged_insights: str, playbook_text: str, conversation_history: str = "", *, task_instr: str = "") -> str:
        """Build a curator prompt with insights and current playbook."""
        variables = {
            "current_playbook": playbook_text,
            "guidebook": merged_insights,
            "conversation_history": conversation_history,
            "task_instr": task_instr,
        }
        return _render_template(self.curator_prompt_template, variables)

    def _flush_playbook_now(self, *, step: int, reason: str) -> None:
        self.playbook = Playbook(self.initial_playbook_text)
        self._last_playbook_flush_step = step
        logger.info(
            "[FLUSH] Reset playbook to initial template at step=%s reason=%s",
            step,
            reason,
        )

    def _maybe_flush_playbook(self):
        """Periodically reset the global playbook to the initial template."""
        if self.flush_playbook_every_n_epochs <= 0:
            return

        if self.global_steps <= 1:
            return

        if self._steps_per_epoch <= 0:
            try:
                self._steps_per_epoch = len(self.train_dataloader)
            except Exception:
                self._steps_per_epoch = 0

        if self._steps_per_epoch <= 0:
            return

        # global_steps is incremented after each batch in parent trainer.
        # At epoch boundaries, first batch satisfies this condition.
        is_epoch_start = ((self.global_steps - 1) % self._steps_per_epoch) == 0
        if not is_epoch_start:
            return

        current_epoch = ((self.global_steps - 1) // self._steps_per_epoch) + 1
        if current_epoch <= 1:
            return

        if ((current_epoch - 1) % self.flush_playbook_every_n_epochs) != 0:
            return

        self._flush_playbook_now(
            step=self.global_steps,
            reason=f"epoch_boundary_{current_epoch}",
        )

    def _after_scheduled_events(
        self,
        *,
        step: int,
        epoch: int | None = None,
        batch_idx: int | None = None,
        validation_ran: bool = False,
        checkpoint_saved: bool = False,
    ) -> None:
        super()._after_scheduled_events(
            step=step,
            epoch=epoch,
            batch_idx=batch_idx,
            validation_ran=validation_ran,
            checkpoint_saved=checkpoint_saved,
        )
        if self.flush_playbook_every_n_steps <= 0:
            return
        if step <= 0 or self._last_playbook_flush_step == step:
            return
        if step % self.flush_playbook_every_n_steps != 0:
            return

        reason = "post_validation_step" if validation_ran else "post_step"
        self._flush_playbook_now(step=step, reason=reason)

    def _generate_policy_only_trajectory(self, timing_raw=None, meta_info=None):
        """Run policy rollouts only (no ACE adaptation), used for validation."""
        from verl.utils.debug import marked_timer

        if timing_raw is None:
            timing_raw = {}

        playbook_text = self._get_current_playbook_text()

        with marked_timer("ace_act", timing_raw):
            self._set_component_token_limits("policy")
            self._inject_playbook_into_agents(self.agent_execution_engine.agents, playbook_text)
            policy_trajectories = []
            gen = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
            for traj in gen:
                traj["_component_type"] = "policy"
                policy_trajectories.append(traj)
            policy_trajectories.sort(key=lambda x: x["idx"])
            actor_reward_metrics = raw_reward_metrics(policy_trajectories, expected_count=len(self.agent_execution_engine.envs))
            rollout_n = int(self.config.actor_rollout_ref.rollout.n or 1)
            bsz = len(policy_trajectories) // rollout_n if rollout_n > 0 else len(policy_trajectories)
            if bsz <= 0:
                bsz = len(policy_trajectories)
            chat_log_path = self._save_policy_chat_completions(
                policy_trajectories,
                bsz=bsz,
                meta_info=meta_info,
            )

        with marked_timer("transform_trajectory", timing_raw):
            final_gen_batch_output, transform_metrics = self._transform_agent_trajectories(policy_trajectories)

        rewards = [t.get("trajectory_reward", 0.0) for t in policy_trajectories]
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        metrics = dict(transform_metrics)
        if not (meta_info and meta_info.get("validate", False)):
            metrics.update(actor_reward_metrics)

        # Rollout totals: ALL tokens generated this step (validation, pre-rejection)
        def _sum_token_local(traj_list, key):
            total = 0
            for t in traj_list:
                m = t.get("metrics", {}) if isinstance(t, dict) else {}
                v = m.get(key, 0)
                if v is not None and v > 0:
                    total += int(v)
            return total

        metrics["rollout/total_prompt_tokens"] = _sum_token_local(policy_trajectories, "total_prompt_tokens")
        metrics["rollout/total_completion_tokens"] = _sum_token_local(policy_trajectories, "total_completion_tokens")
        metrics["rollout/total_gradient_update_tokens"] = _sum_token_local(policy_trajectories, "total_gradient_update_tokens")
        metrics["rollout/total_trajectories"] = len(policy_trajectories)
        metrics["ace/policy_rollouts"] = len(policy_trajectories)
        metrics["ace/reflector_rollouts"] = 0
        metrics["ace/curator_rollouts"] = 0
        metrics["ace/total_rollouts"] = len(policy_trajectories)
        metrics["ace/candidates"] = 0
        metrics["ace/current_reward"] = avg_reward
        metrics["ace/selected_reward"] = avg_reward
        metrics["ace/best_candidate_reward"] = avg_reward
        metrics["ace/reject_candidate"] = 0
        metrics["ace/active_component"] = hash("policy_only_eval") % 1000
        metrics["ace/validate_policy_only"] = 1
        pb_stats = self.playbook.get_stats()
        metrics["playbook/size"] = len(playbook_text)
        metrics["playbook/bullets"] = pb_stats.get("total_bullets", 0)
        self._save_ace_step_log(
            {
                "step": int(self.global_steps),
                "mode": self._artifact_mode(meta_info),
                "policy_only": True,
                "chat_completions_path": chat_log_path,
                "metrics": metrics,
            },
            meta_info=meta_info,
        )

        return final_gen_batch_output, metrics

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """Run ACE rollout for training and policy-only rollout for validation."""
        from verl.utils.debug import marked_timer

        if timing_raw is None:
            timing_raw = {}

        is_validation = bool(meta_info and meta_info.get("validate", False))
        self._last_candidate_eval_trajectories = None
        self._last_candidate_eval_token_metrics = []
        if is_validation and self._should_use_policy_only_validation(meta_info):
            self._clear_validation_policy_trajectory_cache()
            return self._generate_policy_only_trajectory(timing_raw=timing_raw, meta_info=meta_info)

        self._maybe_flush_playbook()

        active_component = self._get_active_component()
        if self._should_run_policy_only_phase(active_component):
            final_gen_batch_output, metrics = self._generate_policy_only_trajectory(timing_raw=timing_raw, meta_info=meta_info)
            metrics["ace/active_component"] = hash("policy") % 1000
            metrics["ace/validate_policy_only"] = 0
            self._advance_component_schedule()
            return final_gen_batch_output, metrics

        playbook_text = self._get_current_playbook_text()
        playbook_before_text = playbook_text
        n = self.config.actor_rollout_ref.rollout.n  # g_act / rollouts per task

        if self._should_use_role_reduced_sampling():
            effective_g_ref = self._get_eval_g_ref(meta_info) if active_component in ("all", "reflector") else 1
            effective_g_cur = self._get_eval_g_cur(meta_info) if active_component in ("all", "curator") else 1
        else:
            effective_g_ref = self._get_eval_g_ref(meta_info)
            effective_g_cur = self._get_eval_g_cur(meta_info)

        # ================================================================
        # Phase 1: ACT - Policy rollouts with playbook
        # ================================================================
        reused_validation_rollout = False
        cached_policy_trajectories = self._pop_validation_policy_trajectory_cache() if is_validation else None
        with marked_timer("ace_act", timing_raw):
            self._set_component_token_limits("policy")
            self._inject_playbook_into_agents(self.agent_execution_engine.agents, playbook_text)

            if cached_policy_trajectories:
                policy_trajectories = [copy.deepcopy(traj) for traj in cached_policy_trajectories]
                policy_trajectories.sort(key=lambda x: x.get("idx", 0))
                for idx, traj in enumerate(policy_trajectories):
                    traj["idx"] = idx
                reused_validation_rollout = True
                logger.info(
                    "[ACT] Reusing %s cached validation policy rollouts",
                    len(policy_trajectories),
                )
            else:
                policy_trajectories = []
                gen = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for traj in gen:
                    policy_trajectories.append(traj)
                policy_trajectories.sort(key=lambda x: x["idx"])

        actor_reward_metrics = raw_reward_metrics(policy_trajectories, expected_count=len(self.agent_execution_engine.envs))
        bsz = len(policy_trajectories) // n if n > 0 else len(policy_trajectories)
        if bsz == 0:
            bsz = len(policy_trajectories)
        policy_chat_log_path = self._save_policy_chat_completions(
            policy_trajectories,
            bsz=bsz,
            meta_info=meta_info,
        )

        logger.info(f"[ACT] Generated {len(policy_trajectories)} policy rollouts (bsz={bsz}, n={n})")

        task_groups = [[] for _ in range(bsz)]
        for i, traj in enumerate(policy_trajectories):
            task_idx = i % bsz
            task_groups[task_idx].append(traj)

        reflection_source_indices_by_task = [self._select_reflection_source_indices(task_group) for task_group in task_groups]

        current_rewards = [t["trajectory_reward"] for t in policy_trajectories]
        current_avg_reward = float(np.mean(current_rewards)) if current_rewards else 0.0

        # Save base policy agents/envs for temporary reflector/curator swaps.
        self._original_agents = self.agent_execution_engine.agents
        self._original_envs = self.agent_execution_engine.envs

        # ================================================================
        # Phase 2: REFLECT - Generate insights from all trajectories
        # ================================================================
        reflector_trajectories = []
        all_reflections = []  # ordered to match reflector_trajectories
        reflector_records = []
        reflections_by_task = [[[{} for _ in range(effective_g_ref)] for _ in task_group] for task_group in task_groups]

        with marked_timer("ace_reflect", timing_raw):
            self._set_component_token_limits("reflector")
            reflector_prompts = []
            for task_idx, group in enumerate(task_groups):
                for traj_idx in reflection_source_indices_by_task[task_idx]:
                    traj_dict = group[traj_idx]
                    prompt_text = self._build_reflector_prompt_from_dict(traj_dict, playbook_text, task_idx)
                    for insight_idx in range(effective_g_ref):
                        guidebook_uid = self._make_guidebook_uid(
                            task_idx=task_idx,
                            traj_idx=traj_idx,
                            insight_idx=insight_idx,
                        )
                        reflector_prompts.append(
                            {
                                "prompt_text": prompt_text,
                                "task_id": f"ref_t{task_idx}_traj{traj_idx}_ins{insight_idx}",
                                "task_idx": task_idx,
                                "traj_idx": traj_idx,
                                "insight_idx": insight_idx,
                                "guidebook_uid": guidebook_uid,
                                "reflector_uid": f"ref_t{task_idx}_traj{traj_idx}",
                            }
                        )

            if reflector_prompts:
                from rllm.algorithms.ace.agents import SingleTurnReflectorAgent
                from rllm.algorithms.ace.environments import SingleTurnReflectorEnv

                ref_agents = []
                ref_envs = []
                for rp in reflector_prompts:
                    ref_agents.append(
                        SingleTurnReflectorAgent(
                            system_prompt=self.reflector_system_prompt,
                        )
                    )
                    ref_envs.append(
                        SingleTurnReflectorEnv(
                            prompt_text=rp["prompt_text"],
                            task_id=rp["task_id"],
                        )
                    )

                self.agent_execution_engine.update_envs_and_agents(ref_envs, ref_agents)

                ref_gen = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for traj in ref_gen:
                    reflector_trajectories.append(traj)
                reflector_trajectories.sort(key=lambda x: x["idx"])

                for i, traj in enumerate(reflector_trajectories):
                    response_text = self._decode_trajectory_response(traj)
                    reflection = _coerce_reflection_payload(
                        _parse_json_response(response_text),
                        response_text=response_text,
                        context="reflector",
                    )
                    rp = reflector_prompts[i]

                    task_idx = rp["task_idx"]
                    traj_idx = rp["traj_idx"]
                    insight_idx = rp["insight_idx"]
                    if task_idx < len(reflections_by_task) and traj_idx < len(reflections_by_task[task_idx]):
                        reflections_by_task[task_idx][traj_idx][insight_idx] = reflection

                    reward = 0.0
                    if task_idx < len(task_groups) and traj_idx < len(task_groups[task_idx]):
                        reward = float(task_groups[task_idx][traj_idx].get("trajectory_reward", 0.0))

                    all_reflections.append(
                        {
                            "task_idx": task_idx,
                            "traj_idx": traj_idx,
                            "insight_idx": insight_idx,
                            "guidebook_uid": rp["guidebook_uid"],
                            "reflector_uid": rp["reflector_uid"],
                            "reflection": reflection,
                            "reward": reward,
                        }
                    )
                    reflector_records.append(
                        {
                            "task_id": rp["task_id"],
                            "task_idx": task_idx,
                            "traj_idx": traj_idx,
                            "insight_idx": insight_idx,
                            "guidebook_uid": rp["guidebook_uid"],
                            "prompt_text": rp["prompt_text"],
                            "response_text": response_text,
                            "parsed_reflection": reflection,
                            "seed_reward": reward,
                        }
                    )

                self.agent_execution_engine.update_envs_and_agents(self._original_envs, self._original_agents)

        valid_reflections = sum(1 for r in all_reflections if r.get("reflection"))
        logger.info(f"[REFLECT] Got {valid_reflections}/{len(all_reflections)} valid reflections")

        # ================================================================
        # Phase 3: CURATE - Per-task guidebooks (same task, different trajectories)
        # ================================================================
        curator_trajectories = []
        candidate_playbooks = []
        curator_to_guidebook = []  # curator traj index -> guidebook uid
        curator_records = []

        with marked_timer("ace_curate", timing_raw):
            self._set_component_token_limits("curator")
            curator_prompts = []
            for task_idx, group in enumerate(task_groups):
                if not group:
                    continue
                task_curator_specs = []
                if self._uses_single_guidebook_aggregation():
                    for traj_idx in reflection_source_indices_by_task[task_idx]:
                        for insight_idx in range(effective_g_ref):
                            guidebook_uid = self._make_guidebook_uid(
                                task_idx=task_idx,
                                traj_idx=traj_idx,
                                insight_idx=insight_idx,
                            )
                            merged_insights = _concatenate_insights(
                                [
                                    {
                                        "task_idx": task_idx,
                                        "reflection": reflections_by_task[task_idx][traj_idx][insight_idx],
                                    }
                                ]
                            )
                            if not merged_insights.strip():
                                continue

                            task_curator_specs.append(
                                {
                                    "prompt_text": self._build_curator_prompt(
                                        merged_insights,
                                        playbook_text,
                                        task_instr=self._get_policy_task_instruction(group[traj_idx], task_idx),
                                    ),
                                    "task_id": f"cur_{guidebook_uid}",
                                    "guidebook_uid": guidebook_uid,
                                    "task_idx": task_idx,
                                    "traj_idx": traj_idx,
                                    "insight_idx": insight_idx,
                                }
                            )
                else:
                    for insight_idx in range(effective_g_ref):
                        guidebook_uid = self._make_guidebook_uid(
                            task_idx=task_idx,
                            insight_idx=insight_idx,
                        )
                        guidebook_insights = []
                        for traj_idx, _ in enumerate(group):
                            guidebook_insights.append(
                                {
                                    "task_idx": task_idx,
                                    "reflection": reflections_by_task[task_idx][traj_idx][insight_idx],
                                }
                            )

                        merged_insights = _concatenate_insights(guidebook_insights)
                        if not merged_insights.strip():
                            continue

                        task_curator_specs.append(
                            {
                                "prompt_text": self._build_curator_prompt(
                                    merged_insights,
                                    playbook_text,
                                    task_instr=self._get_policy_task_instruction(group[0], task_idx),
                                ),
                                "task_id": f"cur_{guidebook_uid}",
                                "guidebook_uid": guidebook_uid,
                                "task_idx": task_idx,
                                "insight_idx": insight_idx,
                            }
                        )

                curator_prompts.extend(
                    self._expand_curator_prompt_specs_for_sampling(
                        task_curator_specs,
                        effective_g_cur,
                    )
                )

            if curator_prompts:
                from rllm.algorithms.ace.agents import SingleTurnCuratorAgent
                from rllm.algorithms.ace.environments import SingleTurnCuratorEnv

                cur_agents = []
                cur_envs = []
                for cp in curator_prompts:
                    cur_agents.append(
                        SingleTurnCuratorAgent(
                            system_prompt=self.curator_system_prompt,
                        )
                    )
                    cur_envs.append(
                        SingleTurnCuratorEnv(
                            prompt_text=cp["prompt_text"],
                            task_id=cp["task_id"],
                        )
                    )

                self.agent_execution_engine.update_envs_and_agents(cur_envs, cur_agents)

                cur_gen = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for traj in cur_gen:
                    curator_trajectories.append(traj)
                curator_trajectories.sort(key=lambda x: x["idx"])

                for i, traj in enumerate(curator_trajectories):
                    response_text = self._decode_trajectory_response(traj)
                    ops = _parse_json_response(response_text)
                    cp = curator_prompts[i]
                    curator_to_guidebook.append(cp["guidebook_uid"])
                    record = {
                        "task_id": cp["task_id"],
                        "task_idx": cp.get("task_idx"),
                        "traj_idx": cp.get("traj_idx"),
                        "insight_idx": cp.get("insight_idx"),
                        "guidebook_uid": cp["guidebook_uid"],
                        "prompt_text": cp["prompt_text"],
                        "response_text": response_text,
                        "parsed_operations": ops,
                        "curator_prompt_idx": cp.get("curator_prompt_idx", 0),
                        "curator_sample_idx": cp.get("curator_sample_idx", 0),
                        "curator_samples_for_prompt": cp.get("curator_samples_for_prompt", 1),
                        "curator_grouped": bool(cp.get("curator_grouped", False)),
                        "curator_group_sampling_mode": cp.get("curator_group_sampling_mode", "all_groups"),
                        "accepted": False,
                    }

                    # Curator may return either {"operations": [...]} or directly [...].
                    if isinstance(ops, list):
                        ops_payload = {"operations": ops}
                    elif isinstance(ops, dict):
                        ops_payload = ops
                    else:
                        ops_payload = {}

                    if ops_payload.get("operations"):
                        candidate = Playbook(playbook_text)
                        try:
                            if getattr(self, "handoff_protocol", "legacy") == "feedback_v1":
                                record["mutation"] = candidate.apply_operations_with_report(ops_payload)
                                if not record["mutation"]["changed"]:
                                    record["reject_reason"] = "unchanged_playbook"
                                    record["candidate_playbook"] = candidate.to_string()
                                    curator_records.append(record)
                                    continue
                            else:
                                candidate.apply_operations(ops_payload)
                            if self.max_playbook_length is not None:
                                candidate_len = _get_playbook_token_length(candidate.to_string(), self.tokenizer)
                                record["candidate_length"] = candidate_len
                                if candidate_len > self.max_playbook_length:
                                    logger.info(
                                        "[CURATE] Rejected candidate: %s > %s tokens",
                                        candidate_len,
                                        self.max_playbook_length,
                                    )
                                    record["reject_reason"] = "max_playbook_length"
                                    record["candidate_playbook"] = candidate.to_string()
                                    curator_records.append(record)
                                    continue
                            candidate_playbooks.append(
                                {
                                    "playbook": candidate,
                                    "curator_idx": i,
                                    "guidebook_uid": cp["guidebook_uid"],
                                    "curator_grouped": bool(cp.get("curator_grouped", False)),
                                    "curator_sample_idx": cp.get("curator_sample_idx", 0),
                                    "curator_samples_for_prompt": cp.get("curator_samples_for_prompt", 1),
                                }
                            )
                            record["accepted"] = True
                            record["candidate_index"] = len(candidate_playbooks) - 1
                            record["candidate_playbook"] = candidate.to_string()
                        except Exception as e:
                            record["error"] = str(e)
                            logger.warning(f"Playbook apply_operations failed: {e}")
                    curator_records.append(record)

                self.agent_execution_engine.update_envs_and_agents(self._original_envs, self._original_agents)

        logger.info(f"[CURATE] Generated {len(candidate_playbooks)} valid candidates")

        # ================================================================
        # Phase 4: EVALUATE - Score each candidate playbook globally
        # ================================================================
        candidate_rewards = []
        candidate_eval_records = []
        if candidate_playbooks:
            with marked_timer("ace_evaluate", timing_raw):
                self._set_component_token_limits("policy")
                candidate_rewards, candidate_eval_records = self._evaluate_candidate_playbooks(
                    candidate_playbooks,
                    timing_raw=timing_raw,
                    meta_info=meta_info,
                )

        # ================================================================
        # Phase 5: SELECT - Pick best candidate playbook
        # ================================================================
        policy_trajectories_for_training = [copy.deepcopy(traj) for traj in policy_trajectories]
        policy_train_source = "seed"
        selected_reward = current_avg_reward
        selected_idx = -1
        reject_candidate = False
        if candidate_playbooks and candidate_rewards:
            with marked_timer("ace_select", timing_raw):
                best_candidate_reward = float(np.max(candidate_rewards))
                candidate_indices = list(range(min(len(candidate_playbooks), len(candidate_rewards))))
                surviving_candidate_indices = candidate_indices
                if self.selection_mode == "best_rejection":
                    reject_candidate = best_candidate_reward < current_avg_reward
                elif self.selection_mode == "best_rejection_update":
                    # Standard candidates retain curator_idx, while the source
                    # task/rollout indices live in the corresponding record.
                    surviving_candidate_indices = self._get_best_rejection_update_candidate_indices(
                        [curator_records[cand["curator_idx"]] for cand in candidate_playbooks],
                        candidate_rewards,
                        task_groups,
                        candidate_indices,
                    )
                    reject_candidate = not surviving_candidate_indices
                if reject_candidate:
                    logger.info(
                        "[SELECT] Rejected all candidates: current_reward=%.3f best_candidate_reward=%.3f",
                        current_avg_reward,
                        best_candidate_reward,
                    )
                else:
                    if self.selection_mode == "best_rejection_update":
                        best_idx = self._select_best_rejection_update_candidate_index(candidate_rewards, surviving_candidate_indices)
                    elif self.selection_mode in {"best", "best_rejection"}:
                        best_idx = max(
                            surviving_candidate_indices,
                            key=lambda idx: float(candidate_rewards[idx]),
                        )
                    else:
                        best_idx = random.randint(0, len(candidate_playbooks) - 1)

                    selected = candidate_playbooks[best_idx]
                    selected_idx = best_idx
                    selected_reward = candidate_rewards[best_idx]
                    self.playbook = selected["playbook"]
                    selected_eval_trajectories = (self._last_candidate_eval_trajectories or {}).get(selected_idx)
                    policy_trajectories_for_training, policy_train_source = self._select_policy_trajectories_for_training(
                        policy_trajectories,
                        selected_eval_trajectories,
                        expected_policy_count=len(policy_trajectories),
                        log_prefix="[SINGLEMODE][SELECT]",
                    )
                    logger.info(f"[SELECT] Candidate {best_idx} (mode={self.selection_mode}, reward={selected_reward:.3f})")
        else:
            best_candidate_reward = current_avg_reward

        if is_validation:
            next_validation_policy_trajectories = None
            if not reject_candidate and selected_idx >= 0:
                selected_eval_trajectories = (self._last_candidate_eval_trajectories or {}).get(selected_idx)
                if selected_eval_trajectories:
                    next_validation_policy_trajectories = [copy.deepcopy(traj) for traj in selected_eval_trajectories]
                else:
                    logger.warning(
                        "[VALIDATION-REUSE] Missing cached eval trajectories for selected candidate %s; next local eval epoch will rerun policy rollout.",
                        selected_idx,
                    )
            elif policy_trajectories:
                next_validation_policy_trajectories = [copy.deepcopy(traj) for traj in policy_trajectories]
            self._store_validation_policy_trajectory_cache(next_validation_policy_trajectories)
        else:
            self._clear_validation_policy_trajectory_cache()

        snapshot_path = self._save_playbook_snapshot_artifact(meta_info)

        # Restore policy agents with updated playbook for next step.
        if self._original_agents is not None:
            self._inject_playbook_into_agents(self._original_agents, self._get_current_playbook_text())
            self.agent_execution_engine.update_envs_and_agents(self._original_envs, self._original_agents)

        # ================================================================
        # Phase 6: ASSIGN REWARDS to reflector/curator trajectories
        # ================================================================
        # Rejection-update rejects the playbook, not the evaluated auxiliary credit.
        if reject_candidate and self.selection_mode != "best_rejection_update":
            for traj in reflector_trajectories:
                traj["trajectory_reward"] = 0.0
            for traj in curator_trajectories:
                traj["trajectory_reward"] = 0.0
        else:
            guidebook_rewards = {}  # guidebook_uid -> list of candidate rewards
            for ci, cand in enumerate(candidate_playbooks):
                gb_uid = cand["guidebook_uid"]
                if gb_uid not in guidebook_rewards:
                    guidebook_rewards[gb_uid] = []
                if ci < len(candidate_rewards):
                    guidebook_rewards[gb_uid].append(candidate_rewards[ci])

            for i, traj in enumerate(reflector_trajectories):
                if i < len(all_reflections):
                    gb_uid = all_reflections[i]["guidebook_uid"]
                    gb_rewards = guidebook_rewards.get(gb_uid, [])
                    ref_reward = float(np.mean(gb_rewards)) if gb_rewards else 0.0
                else:
                    ref_reward = 0.0
                traj["trajectory_reward"] = ref_reward

            curator_idx_to_reward = {}
            for ci, cand in enumerate(candidate_playbooks):
                if ci < len(candidate_rewards):
                    curator_idx_to_reward[cand["curator_idx"]] = candidate_rewards[ci]

            for i, traj in enumerate(curator_trajectories):
                traj["trajectory_reward"] = curator_idx_to_reward.get(i, 0.0)

        # ================================================================
        # Phase 7: COMBINE & TRANSFORM
        # ================================================================
        for traj in policy_trajectories_for_training:
            traj["_component_type"] = "policy"
        for traj in reflector_trajectories:
            traj["_component_type"] = "reflector"
        for traj in curator_trajectories:
            traj["_component_type"] = "curator"

        auxiliary_gate_active = self._uses_auxiliary_update_gate(meta_info)
        if active_component != "all":
            for traj in reflector_trajectories:
                if active_component not in ("all", "reflector"):
                    traj["trajectory_reward"] = 0.0
            for traj in curator_trajectories:
                if active_component not in ("all", "curator"):
                    traj["trajectory_reward"] = 0.0
            for traj in policy_trajectories_for_training:
                if not auxiliary_gate_active and active_component not in ("all", "policy"):
                    traj["trajectory_reward"] = 0.0

        self._advance_component_schedule()

        auxiliary_improved, auxiliary_baseline = self._selected_playbook_improvement(
            selected_idx, selected_reward, current_avg_reward, candidate_playbooks, curator_records, task_groups
        )
        auxiliary_admitted = not auxiliary_gate_active or auxiliary_improved
        training_reflector_trajectories, training_curator_trajectories, reflector_uids, curator_uids = self._prepare_auxiliary_training_rows(
            reflector_trajectories,
            curator_trajectories,
            all_reflections,
            curator_records,
            curator_to_guidebook,
            admitted_task_ids=None if auxiliary_admitted else set(),
        )
        all_trajectories = policy_trajectories_for_training + training_reflector_trajectories + training_curator_trajectories
        for i, traj in enumerate(all_trajectories):
            traj["idx"] = i

        with marked_timer("transform_trajectory", timing_raw):
            final_gen_batch_output, transform_metrics = self._transform_agent_trajectories(all_trajectories)

        # Keep turn/* metrics policy-only even when training with mixed ACE trajectories.
        policy_step_rewards = []
        for traj in policy_trajectories_for_training:
            step_rewards = traj.get("step_rewards", [])
            if isinstance(step_rewards, list | tuple):
                policy_step_rewards.append([float(r) for r in step_rewards])
            else:
                policy_step_rewards.append([])
        self._current_step_rewards_list = policy_step_rewards if policy_step_rewards and any(len(sr) > 0 for sr in policy_step_rewards) else None

        # Expand parent batch for union() compatibility.
        n_policy = len(policy_trajectories_for_training)
        n_ref = len(reflector_trajectories)
        n_cur = len(curator_trajectories)
        extra_policy_count = max(0, n_policy - len(policy_trajectories))

        self._append_policy_and_component_rows_to_batch(
            self._current_batch,
            extra_policy_count=extra_policy_count,
            component_extra_uids=reflector_uids + curator_uids,
        )

        metrics = {}
        metrics.update(transform_metrics)
        if not is_validation:
            metrics.update(actor_reward_metrics)
            metrics.update(self._auxiliary_training_metrics(auxiliary_gate_active, int(auxiliary_admitted), 1, n_policy, reflector_uids, curator_uids))
        metrics.update(
            self._compute_rollout_token_totals(
                policy_trajectories=policy_trajectories,
                reflector_trajectories=reflector_trajectories,
                curator_trajectories=curator_trajectories,
                candidate_eval_token_metrics=self._last_candidate_eval_token_metrics,
                ppo_trajectories=all_trajectories,
            )
        )
        if getattr(self, "handoff_protocol", "legacy") == "feedback_v1":
            metrics.update(self._curator_mutation_metrics(curator_records))
        metrics["ace/policy_rollouts"] = n_policy
        metrics["ace/reflector_rollouts"] = n_ref
        metrics["ace/curator_rollouts"] = n_cur
        metrics["ace/total_rollouts"] = len(all_trajectories)
        metrics["ace/candidates"] = len(candidate_playbooks)
        metrics["ace/singlemode"] = int(self.singlemode_enabled)
        metrics["ace/singlemode_curator_one_group"] = int(self._should_use_singlemode_curator_one_group())
        metrics["ace/guidebook_aggregation_single"] = int(self.guidebook_aggregation_level == "single")
        metrics["ace/current_reward"] = current_avg_reward
        metrics["ace/selected_reward"] = selected_reward
        metrics["ace/best_candidate_reward"] = best_candidate_reward
        metrics["ace/reject_candidate"] = int(reject_candidate)
        metrics["ace/active_component"] = hash(active_component) % 1000
        metrics["ace/policy_train_source_id"] = self._policy_train_source_metric_id(policy_train_source)
        metrics["ace/guidebooks"] = len(reflector_records) if self.guidebook_aggregation_level == "single" else bsz * effective_g_ref
        metrics["ace/reused_validation_policy_rollout"] = int(reused_validation_rollout)
        playbook_after_text = self._get_current_playbook_text()
        metrics["playbook/size"] = len(playbook_after_text)
        pb_stats = self.playbook.get_stats()
        metrics["playbook/bullets"] = pb_stats.get("total_bullets", 0)
        self._save_ace_step_log(
            {
                "step": int(self.global_steps),
                "mode": self._artifact_mode(meta_info),
                "active_component": active_component,
                "counts": {
                    "policy_rollouts": n_policy,
                    "reflector_rollouts": n_ref,
                    "curator_rollouts": n_cur,
                    "guidebooks": metrics["ace/guidebooks"],
                    "candidates": len(candidate_playbooks),
                    "guidebook_aggregation_level": self.guidebook_aggregation_level,
                    "singlemode": self.singlemode_enabled,
                    "reflection_source_selection": getattr(self, "reflection_source_selection", "random"),
                    "singlemode_curator_one_group": self._should_use_singlemode_curator_one_group(),
                },
                "paths": {
                    "policy_chat_completions": policy_chat_log_path,
                    "playbook_snapshot": snapshot_path,
                },
                "selection": {
                    "selected_candidate_idx": selected_idx,
                    "selected_reward": selected_reward,
                    "best_candidate_reward": best_candidate_reward,
                    "reject_candidate": bool(reject_candidate),
                    "selection_mode": self.selection_mode,
                    "policy_train_source": policy_train_source,
                    "auxiliary_update_gate_active": auxiliary_gate_active,
                    "auxiliary_update_baseline": auxiliary_baseline,
                    "auxiliary_update_improved": auxiliary_improved,
                    "auxiliary_training_admitted": auxiliary_admitted,
                },
                "playbook": {
                    "before": playbook_before_text,
                    "after": playbook_after_text,
                },
                "prompt_paths": self.prompt_paths,
                "reflector": reflector_records,
                "curator": curator_records,
                "evaluation": candidate_eval_records,
            },
            meta_info=meta_info,
        )

        return final_gen_batch_output, metrics

    def _get_policy_task_for_trajectory(self, traj_dict: dict, task_idx: int):
        """Recover the completed policy agent's task without changing environment identity."""
        engine = getattr(self, "agent_execution_engine", None)
        envs = getattr(self, "_original_envs", None) or getattr(engine, "envs", [])
        agents = getattr(self, "_original_agents", None) or getattr(engine, "agents", [])
        candidate_indices = []
        try:
            candidate_indices.append(int(traj_dict.get("idx")))
        except (TypeError, ValueError):
            pass
        try:
            candidate_indices.append(int(task_idx))
        except (TypeError, ValueError):
            pass

        for idx in candidate_indices:
            env_task = getattr(envs[idx], "task", None) if 0 <= idx < len(envs) else None
            if 0 <= idx < len(agents):
                agent = agents[idx]
                task = getattr(getattr(agent, "_trajectory", None), "task", None)
                instruction = getattr(agent, "task_instruction", None)
                if isinstance(task, dict) or instruction:
                    # AppWorld dataset rows hold only task_id. Its agent records
                    # the original instruction after the environment resets.
                    payload = dict(env_task) if isinstance(env_task, dict) else {}
                    if isinstance(task, dict):
                        payload.update(task)
                    if instruction and not payload.get("instruction"):
                        payload["instruction"] = instruction
                    if payload:
                        return payload
                elif task:
                    return task
            if env_task:
                return env_task
        return {}

    def _get_policy_task_instruction(self, traj_dict: dict, task_idx: int) -> str:
        """Get the original task text for task-aware role templates."""
        task = self._get_policy_task_for_trajectory(traj_dict, task_idx)
        if isinstance(task, dict):
            return str(task.get("instruction", "") or "")
        return task if isinstance(task, str) else ""

    def _trajectory_steps_from_chat(
        self,
        chat_completions,
        step_rewards: list,
        *,
        all_passed: bool = False,
        terminal_feedback: dict | None = None,
    ) -> list[SimpleNamespace]:
        """Reconstruct lightweight Step-like objects for ACE reflectors."""
        if not isinstance(chat_completions, list):
            return []

        steps: list[SimpleNamespace] = []
        pending_observation = None
        for msg in chat_completions:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            content = msg.get("content", "")
            if role == "user":
                # The environment's reply to the previous action, including a terminal reply.
                if steps and steps[-1].next_observation is None:
                    steps[-1].next_observation = content
                pending_observation = content
                continue
            if role != "assistant":
                continue

            step_idx = len(steps)
            reward = step_rewards[step_idx] if step_idx < len(step_rewards) else None
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
                    reward=reward,
                    done=False,
                    info=info,
                )
            )
            pending_observation = None
        if steps and isinstance(terminal_feedback, dict):
            steps[-1].info["terminal_feedback"] = terminal_feedback
        return steps

    def _build_reflector_prompt_from_dict(self, traj_dict: dict, playbook_text: str, task_idx: int) -> str:
        """Build a reflector prompt from a trajectory dict.

        Since VeRL token rollouts return dictionaries rather than Trajectory
        instances, reconstruct a lightweight trajectory and delegate
        environment-specific fields to the configured reflector adapter.
        """
        chat_completions = traj_dict.get("chat_completions", [])
        step_rewards = list(traj_dict.get("step_rewards", []) or [])
        reward = float(traj_dict.get("trajectory_reward", 0.0) or 0.0)
        all_passed = bool(traj_dict.get("all_passed", reward == 1.0))
        lightweight_trajectory = SimpleNamespace(
            steps=self._trajectory_steps_from_chat(
                chat_completions,
                step_rewards,
                all_passed=all_passed,
                terminal_feedback=traj_dict.get("terminal_feedback"),
            ),
            reward=reward,
            task=self._get_policy_task_for_trajectory(traj_dict, task_idx),
            info={
                "step_rewards": step_rewards,
                "termination_reason": traj_dict.get("termination_reason"),
                "all_passed": all_passed,
            },
        )
        if isinstance(traj_dict.get("terminal_feedback"), dict):
            lightweight_trajectory.info["terminal_feedback"] = traj_dict["terminal_feedback"]

        try:
            variables = self.reflector_adapter.get_template_variables(
                lightweight_trajectory,
                reward,
                playbook_text,
                termination_reason=traj_dict.get("termination_reason"),
                max_turns=self.ace_config.get("max_turns"),
            )
        except Exception as exc:  # noqa: BLE001
            if getattr(self, "handoff_protocol", "legacy") == "feedback_v1":
                raise ValueError("feedback_v1 reflector formatting failed; refusing to discard diagnostic feedback") from exc
            logger.warning(
                "[REFLECT] Adapter %s failed to format trajectory; falling back to generic prompt vars: %s",
                type(self.reflector_adapter).__name__,
                exc,
            )
            if isinstance(chat_completions, list):
                conversation_lines = []
                for msg in chat_completions:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        conversation_lines.append(f"[{role.upper()}]: {content}")
                conversation_history = "\n\n".join(conversation_lines)
            else:
                conversation_history = str(chat_completions)
            turns_used = len(step_rewards) if step_rewards else 1
            best_reward = max(step_rewards) if step_rewards else reward
            variables = {
                "conversation_history": conversation_history,
                "trajectory_summary": "\n".join(
                    [
                        f"Outcome: {'SUCCESS' if all_passed else 'FAILED'}",
                        f"All Tests Passed: {all_passed}",
                        f"Turns Used: {turns_used}",
                        f"Final Reward: {reward:.2f}",
                        f"Best Reward: {best_reward:.2f}",
                    ]
                ),
                "playbook": playbook_text,
            }

        variables.setdefault("playbook", playbook_text)

        def render() -> str:
            return _render_template(self.reflector_prompt_template, variables)

        prompt = render()
        reflector_limit = self.component_token_limits["reflector"]["max_prompt_length"]

        # The role's engine counts the fully chat-templated prompt (including
        # the system message), so use the same parser here rather than a
        # character estimate. This is a final guard in addition to the
        # adapter's history budget, which also covers custom history budgets.
        def prompt_token_count(user_prompt: str) -> int:
            messages = [
                {"role": "system", "content": self.reflector_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            rollout_engine = self.agent_execution_engine.rollout_engine
            parser = getattr(rollout_engine, "chat_parser", None)
            if parser is not None:
                formatted = parser.parse(
                    messages,
                    add_generation_prompt=True,
                    is_first_msg=True,
                    accumulate_reasoning=False,
                )
                return len(self.tokenizer.encode(formatted, add_special_tokens=False))
            return len(self.tokenizer.encode(user_prompt, add_special_tokens=False))

        prompt_tokens = prompt_token_count(prompt)
        if prompt_tokens <= reflector_limit:
            return prompt

        history = variables.get("conversation_history", "")
        if not history:
            raise ValueError(f"Reflector prompt exceeds its token budget without conversation history to truncate: {prompt_tokens} > {reflector_limit}")

        from rllm.algorithms.ace.context.utils import truncate_conversation_history

        # Calculate the history allocation after rendering every other prompt
        # field, then verify against the exact chat-templated count. Reducing
        # the allocation in the loop absorbs token-boundary/template effects.
        variables["conversation_history"] = ""
        fixed_prompt_tokens = prompt_token_count(render())
        if fixed_prompt_tokens > reflector_limit:
            raise ValueError(f"Reflector prompt's non-history fields exceed its token budget: {fixed_prompt_tokens} > {reflector_limit}")
        history_budget = reflector_limit - fixed_prompt_tokens
        while True:
            variables["conversation_history"] = truncate_conversation_history(
                history,
                max_tokens=history_budget,
                tokenizer=self.tokenizer,
            )
            prompt = render()
            prompt_tokens = prompt_token_count(prompt)
            if prompt_tokens <= reflector_limit:
                logger.info(
                    "[REFLECT] Truncated trajectory history to %s tokens; reflector prompt=%s/%s tokens",
                    history_budget,
                    prompt_tokens,
                    reflector_limit,
                )
                return prompt
            # The standalone-history allocation can differ slightly from the
            # rendered prompt because tokenization spans the replacement edge.
            history_budget -= max(1, prompt_tokens - reflector_limit)

    def _decode_trajectory_response(self, traj_dict: dict) -> str:
        """Decode the response tokens from a trajectory dict back to text."""
        response_tokens = traj_dict.get("response_tokens")
        if response_tokens is not None:
            if isinstance(response_tokens, torch.Tensor):
                token_ids = response_tokens.tolist()
            else:
                token_ids = list(response_tokens)
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return ""

    def fit_agent(self):
        """Override to add playbook logging."""
        try:
            super().fit_agent()
        finally:
            # Print final playbook
            final_text = self._get_current_playbook_text()
            print(f"\n{'=' * 70}")
            print("FINAL PLAYBOOK")
            print(f"{'=' * 70}")
            print(final_text)
            print(f"{'=' * 70}\n")
