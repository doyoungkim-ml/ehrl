"""Local ACE+RL trainer.

This variant maintains one playbook per task (task_id) and updates it when the
same task is seen during training. Larger outer batches are sliced back into
per-task inner ACE updates so each task-specific playbook evolves independently.
"""

import asyncio
import copy
import hashlib
import logging
import os
import random
import uuid
from contextlib import contextmanager

import numpy as np

from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.trainer_base import (
    ACERLAgentPPOTrainer,
    _coerce_reflection_payload,
    _concatenate_insights,
    _get_playbook_token_length,
    _parse_json_response,
)
from rllm.utils.compute_pass_at_k import (
    compute_validation_turn_metrics,
    normalize_validation_data_source,
)
from rllm.utils.training_reward import raw_reward_metrics

logger = logging.getLogger(__name__)


def _safe_task_id_fragment(value: str) -> str:
    fragment = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    fragment = fragment.strip("_")
    return fragment or "task"


def _fallback_task_id_from_payload(task: dict) -> str:
    """Derive a stable per-task id for legacy datasets missing task_id."""
    if not isinstance(task, dict):
        return "unknown"

    raw_task_id = task.get("task_id")
    if raw_task_id:
        return str(raw_task_id)

    # Mastermind's prepared task payloads use uid rather than task_id.
    # This is the dataset identity, not the trainer's per-rollout batch uid.
    raw_uid = task.get("uid")
    if raw_uid:
        return str(raw_uid)

    game_file = task.get("game_file")
    if not game_file:
        return "unknown"

    normalized_game_file = str(game_file)
    alfworld_data = os.environ.get("ALFWORLD_DATA")
    if alfworld_data:
        try:
            normalized_game_file = os.path.relpath(normalized_game_file, alfworld_data)
        except ValueError:
            pass
    else:
        marker = f"{os.sep}json_2.1.1{os.sep}"
        if marker in normalized_game_file:
            normalized_game_file = normalized_game_file.split(marker, 1)[1]

    stable_key = normalized_game_file.replace(os.sep, "/")
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:16]
    task_type = _safe_task_id_fragment(str(task.get("task_type", "alfworld")))
    return f"alfworld_{task_type}_{digest}"


def _save_playbook_snapshot(
    playbook_text: str,
    snapshot_dir: str,
    global_step: int,
    local_eval_epoch: int | None = None,
) -> str:
    """Save a task-local playbook snapshot to disk."""
    os.makedirs(snapshot_dir, exist_ok=True)
    suffix = f"_iter_{local_eval_epoch - 1}" if local_eval_epoch is not None else ""
    filename = f"playbook_step{global_step:04d}{suffix}.txt"
    path = os.path.join(snapshot_dir, filename)
    with open(path, "w") as f:
        f.write(f"# Playbook Snapshot - Step {global_step}\n")
        if local_eval_epoch is not None:
            f.write(f"# Local Eval Epoch: {local_eval_epoch}\n")
        f.write(f"# Size: {len(playbook_text)} characters\n")
        f.write("#" + "=" * 60 + "\n\n")
        f.write(playbook_text)
    return path


def _save_task_playbook_snapshot(
    playbook_text: str,
    base_dir: str,
    task_id: str,
    step: int,
    epoch: int,
    is_eval: bool,
    local_eval_epoch: int | None = None,
) -> str:
    mode_dir = "eval" if is_eval else "train"
    task_dir = os.path.join(base_dir, mode_dir, f"task_{task_id}")
    os.makedirs(task_dir, exist_ok=True)
    suffix = f"_iter_{local_eval_epoch - 1}" if local_eval_epoch is not None else ""
    path = os.path.join(task_dir, f"epoch{epoch:04d}_step{step:06d}{suffix}.txt")
    with open(path, "w") as f:
        f.write("# Local ACE-RL Task Playbook Snapshot\n")
        f.write(f"# task_id: {task_id}\n")
        f.write(f"# mode: {mode_dir}\n")
        f.write(f"# epoch: {epoch}\n")
        f.write(f"# step: {step}\n")
        if local_eval_epoch is not None:
            f.write(f"# local_eval_epoch: {local_eval_epoch}\n")
        f.write(f"# size: {len(playbook_text)} chars\n")
        f.write("#" + "=" * 60 + "\n\n")
        f.write(playbook_text)
    return path


class LocalACERLAgentPPOTrainer(ACERLAgentPPOTrainer):
    """Local ACE-RL trainer with task-specific playbooks."""

    def _iter_training_epoch(self):
        from rllm.algorithms.ace.checkpoint import preserve_controller_rng

        # StatefulDataLoader consumes a worker-base-seed draw even when it is
        # reconstructing an existing iterator. Keep this bookkeeping separate
        # from controller randomness, in both normal and resumed epochs.
        with preserve_controller_rng():
            return iter(self.train_dataloader)

    def _save_checkpoint(self):
        from rllm.algorithms.ace.checkpoint import save_ace_state

        if self.config.trainer.get("default_hdfs_dir") is not None:
            raise NotImplementedError("Local ACE state checkpoints currently support local/shared filesystems only")
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        # Persist controller state before VeRL can publish the latest marker.
        # A failed sidecar write must not advertise a weights-only checkpoint.
        save_ace_state(self, checkpoint_dir)
        return super()._save_checkpoint()

    def _load_checkpoint(self):
        from rllm.algorithms.ace.checkpoint import load_ace_state

        result = super()._load_checkpoint()
        if self.config.trainer.resume_mode == "disable" or (result == 0 and self.global_steps == 0):
            return result
        if not isinstance(getattr(self, "_dataloader_resume_iterator_finished", None), bool):
            raise RuntimeError("Local ACE resume requires the VeRL full dataloader-state restore fix and a checkpoint with data.pt; see docs/ace-checkpoint-resume.md.")
        if self.config.trainer.resume_mode == "resume_path":
            checkpoint_dir = self.config.trainer.resume_from_path
        else:
            # Use the step actually loaded by VeRL, not a second read of a
            # latest marker that another writer could have advanced meanwhile.
            checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        load_ace_state(self, checkpoint_dir)
        return result

    def _init_ace_components(self):
        super()._init_ace_components()
        self.local_train_playbooks = {}
        self.local_eval_playbooks = {}
        self.local_train_task_seen_counts = {}
        self._current_task_seen_counts = {}
        self._current_task_flush_info = {}
        self.local_eval_epochs = int(self.ace_config.get("local_eval_epochs", 3) or 3)
        self.flush_playbook_every_n_steps = int(self.ace_config.get("flush_playbook_every_n_steps", 3) or 0)
        self.flush_playbook_every_n_epochs = int(self.ace_config.get("flush_playbook_every_n_epochs", 0) or 0)
        self.local_artifact_root = self.ace_config.get("local_task_artifact_root")
        self.local_playbook_log_dir = self.ace_config.get("local_playbook_log_dir") or self.playbook_snapshot_dir
        self._in_local_eval = False
        self._epoch_task_data = []
        self._current_epoch = 1
        self._artifact_task_id = None
        self._artifact_task_ids = []
        self._last_policy_chat_paths = {}
        self._suppress_inner_flush = False
        self._current_validation_policy_tasks = None
        self._current_validation_policy_sources = None
        self._local_eval_policy_trajectory_cache = {}
        self._manage_rollout_engine_state_default = None
        self._rollout_engine_external_management_depth = 0
        self._current_local_eval_epoch = None
        logger.info(
            f"Local ACE+RL enabled: local_eval_epochs={self.local_eval_epochs}, "
            f"task_flush_every_n_seen={self.flush_playbook_every_n_steps}, "
            f"flush_every_n_epochs={self.flush_playbook_every_n_epochs}, "
            f"local_playbook_log_dir={self.local_playbook_log_dir}, "
            f"local_task_artifact_root={self.local_artifact_root}"
        )

    def _estimate_epoch(self) -> int:
        if hasattr(self, "_training_epoch"):
            return self._training_epoch + 1
        if self._steps_per_epoch <= 0:
            try:
                self._steps_per_epoch = len(self.train_dataloader)
            except Exception:
                self._steps_per_epoch = 0
        if self._steps_per_epoch <= 0:
            return 1
        return ((self.global_steps - 1) // self._steps_per_epoch) + 1

    def _extract_single_task_id(self) -> str:
        envs = getattr(self.agent_execution_engine, "envs", None) or []
        if not envs:
            return "unknown"
        tid = getattr(envs[0], "task_id", None)
        if tid:
            return str(tid)
        task = getattr(envs[0], "task", None) or {}
        return _fallback_task_id_from_payload(task)

    def _extract_batch_task_ids(self) -> list[str]:
        """Extract unique task_ids from current env batch in first-seen order."""
        envs = getattr(self.agent_execution_engine, "envs", None) or []
        if not envs:
            return ["unknown"]

        ordered = []
        seen = set()
        for env in envs:
            tid = getattr(env, "task_id", None)
            if not tid:
                task = getattr(env, "task", None) or {}
                tid = _fallback_task_id_from_payload(task)
            tid = str(tid)
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)

        return ordered or ["unknown"]

    def _task_id_from_env(self, env) -> str:
        tid = getattr(env, "task_id", None)
        if tid:
            return str(tid)
        task = getattr(env, "task", None) or {}
        return _fallback_task_id_from_payload(task)

    def _build_task_index_groups(self) -> tuple[list[str], dict[str, list[int]]]:
        envs = getattr(self.agent_execution_engine, "envs", None) or []
        ordered_task_ids: list[str] = []
        task_to_indices: dict[str, list[int]] = {}
        for idx, env in enumerate(envs):
            task_id = self._task_id_from_env(env)
            if task_id not in task_to_indices:
                task_to_indices[task_id] = []
                ordered_task_ids.append(task_id)
            task_to_indices[task_id].append(idx)
        return ordered_task_ids, task_to_indices

    def _capture_component_schedule_state(self):
        if hasattr(self, "_iter_role_index") and hasattr(self, "_iter_role_step"):
            return {
                "iter_role_index": int(self._iter_role_index),
                "iter_role_step": int(self._iter_role_step),
            }
        return None

    def _restore_component_schedule_state(self, state):
        if not state:
            return
        if "iter_role_index" in state and hasattr(self, "_iter_role_index"):
            self._iter_role_index = int(state["iter_role_index"])
        if "iter_role_step" in state and hasattr(self, "_iter_role_step"):
            self._iter_role_step = int(state["iter_role_step"])

    def _get_verl_rollout_engine(self):
        if getattr(self.agent_execution_engine, "engine_name", None) != "verl":
            return None
        return getattr(self.agent_execution_engine, "rollout_engine", None)

    @contextmanager
    def _keep_rollout_engine_awake(self, reason: str):
        """Keep the VeRL rollout engine awake across all local ACE rollout phases.

        ``free_cache_engine`` must still free the rollout replicas before PPO,
        but sleeping between ACE's policy, reflector, curator, and candidate
        evaluation passes turns one hand-off into many expensive hand-offs.
        This scope owns a single wake/sleep pair for the whole ACE generation
        sequence; the normal trainer hand-off sleeps it again before PPO as a
        safe no-op if needed.
        """
        rollout_engine = self._get_verl_rollout_engine()
        previous_default = self._manage_rollout_engine_state_default
        using_existing_scope = self._rollout_engine_external_management_depth > 0
        started_external_scope = False

        if rollout_engine is None:
            try:
                yield False
            finally:
                self._manage_rollout_engine_state_default = previous_default
            return

        if not using_existing_scope:
            try:
                asyncio.run(rollout_engine.wake_up())
                started_external_scope = True
                logger.info(
                    "[LOCAL-ACE] keeping rollout engine awake across %s",
                    reason,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[LOCAL-ACE] Failed to pre-wake rollout engine for %s: %s",
                    reason,
                    exc,
                )
                raise RuntimeError(f"Failed to pre-wake rollout engine for {reason}") from exc

        scope_active = using_existing_scope or started_external_scope
        if scope_active:
            self._rollout_engine_external_management_depth += 1
            self._manage_rollout_engine_state_default = False
        try:
            yield started_external_scope
        finally:
            if scope_active:
                self._rollout_engine_external_management_depth = max(0, self._rollout_engine_external_management_depth - 1)
            self._manage_rollout_engine_state_default = previous_default
            if started_external_scope:
                try:
                    asyncio.run(rollout_engine.sleep())
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[LOCAL-ACE] Failed to put rollout engine back to sleep after %s: %s",
                        reason,
                        exc,
                    )

    def generate_agent_trajectories_async(
        self,
        timing_raw=None,
        meta_info=None,
        mode="Token",
        **trajectory_generator_kwargs,
    ):
        if "manage_rollout_engine_state" not in trajectory_generator_kwargs:
            default_manage = self._manage_rollout_engine_state_default
            if default_manage is not None:
                trajectory_generator_kwargs["manage_rollout_engine_state"] = default_manage
        return super().generate_agent_trajectories_async(
            timing_raw=timing_raw,
            meta_info=meta_info,
            mode=mode,
            **trajectory_generator_kwargs,
        )

    def _aggregate_task_metrics(self, metrics_by_task: list[dict]) -> dict:
        if not metrics_by_task:
            return {}

        sum_keys = {
            "train/actor_reward_sum",
            "train/actor_reward_count",
            "train/actor_reward_invalid_count",
            "ace/policy_rollouts",
            "ace/reflector_rollouts",
            "ace/curator_rollouts",
            "ace/candidate_eval_rollouts",
            "ace/total_rollouts",
            "ace/candidates",
            "ace/guidebooks",
            "ace/auxiliary_tasks_admitted",
            "ace/auxiliary_tasks_dropped",
            "ace/policy_training_rows",
            "ace/reflector_training_rows",
            "ace/curator_training_rows",
            "ace/reflector_training_groups",
            "ace/curator_training_groups",
        }
        first_keys = {
            "ace/active_component",
            "ace/auxiliary_update_gate_active",
            "ace/validate_policy_only",
            "ace/guidebook_aggregation_single",
            "ace/singlemode",
            "ace/singlemode_curator_one_group",
        }

        aggregated = {}
        all_keys = set().union(*(m.keys() for m in metrics_by_task))
        for key in all_keys:
            values = [m[key] for m in metrics_by_task if key in m]
            if not values:
                continue

            if key in first_keys:
                aggregated[key] = values[0]
                continue

            numeric_values = [float(v) for v in values if isinstance(v, int | float | np.integer | np.floating)]
            if len(numeric_values) != len(values):
                aggregated[key] = values[0]
                continue

            if key in sum_keys:
                total = float(np.sum(numeric_values))
                aggregated[key] = int(total) if all(float(v).is_integer() for v in numeric_values) else total
            else:
                aggregated[key] = float(np.mean(numeric_values))

        return aggregated

    def _get_task_playbook(self, task_id: str, is_eval: bool) -> Playbook:
        store = self.local_eval_playbooks if is_eval else self.local_train_playbooks
        if task_id not in store:
            store[task_id] = Playbook(self.initial_playbook_text)
        return store[task_id]

    def _prepare_task_playbooks_for_step(
        self,
        task_ids: list[str],
        *,
        is_eval: bool,
    ) -> dict[str, dict]:
        """Count train sightings and reset task-local playbooks on interval.

        Local ACE playbooks are scoped by task, so step-based flushing must be
        keyed by how many training times a given task has appeared instead of by
        global trainer step. The reset happens before the interval encounter is
        rolled out, so sighting counts 3, 6, ... use a fresh task playbook when
        ``flush_playbook_every_n_steps=3``.
        """
        self._current_task_seen_counts = {}
        self._current_task_flush_info = {}

        if is_eval:
            return {}

        flushed: dict[str, dict] = {}
        for raw_task_id in task_ids:
            task_id = str(raw_task_id)
            seen_count = int(self.local_train_task_seen_counts.get(task_id, 0)) + 1
            self.local_train_task_seen_counts[task_id] = seen_count
            self._current_task_seen_counts[task_id] = seen_count

            if self.flush_playbook_every_n_steps <= 0:
                continue
            if seen_count % self.flush_playbook_every_n_steps != 0:
                continue

            previous_playbook = self.local_train_playbooks.get(task_id)
            previous_size = len(previous_playbook.to_string()) if previous_playbook is not None else 0
            self.local_train_playbooks[task_id] = Playbook(self.initial_playbook_text)
            flushed[task_id] = {
                "task_seen_count": seen_count,
                "interval": self.flush_playbook_every_n_steps,
                "previous_size": previous_size,
                "reason": f"task_seen_{seen_count}",
            }
            logger.info(
                "[FLUSH][TASK] Reset local train playbook task_id=%s seen_count=%s interval=%s previous_size=%s global_step=%s",
                task_id,
                seen_count,
                self.flush_playbook_every_n_steps,
                previous_size,
                self.global_steps,
            )

        self._current_task_flush_info = flushed
        return flushed

    def _task_flush_payload(self, task_id: str, *, is_eval: bool = False) -> dict:
        if is_eval:
            return {
                "flushed": False,
                "interval": int(self.flush_playbook_every_n_steps),
                "mode": "eval",
            }
        payload = dict(self._current_task_flush_info.get(str(task_id), {}))
        seen_count = self.local_train_task_seen_counts.get(str(task_id))
        if seen_count is not None:
            payload.setdefault("task_seen_count", int(seen_count))
        payload.setdefault("interval", int(self.flush_playbook_every_n_steps))
        payload.setdefault("flushed", str(task_id) in self._current_task_flush_info)
        return payload

    def _add_task_flush_metrics(
        self,
        metrics: dict,
        task_ids: list[str],
        *,
        is_eval: bool,
    ) -> None:
        if is_eval:
            return
        counts = [int(self.local_train_task_seen_counts.get(str(task_id), 0)) for task_id in task_ids]
        if counts:
            metrics["local/task_seen_count_mean"] = float(np.mean(counts))
            metrics["local/task_seen_count_max"] = float(np.max(counts))
        metrics["local/task_playbook_flush_interval"] = int(self.flush_playbook_every_n_steps)
        metrics["local/task_playbook_flushes"] = sum(1 for task_id in task_ids if str(task_id) in self._current_task_flush_info)

    def _inject_task_playbooks_into_agents(
        self,
        task_to_indices: dict[str, list[int]],
        *,
        is_eval: bool,
    ) -> dict[str, Playbook]:
        """Inject each task's playbook prompt into the matching agent replicas."""
        agents = getattr(self.agent_execution_engine, "agents", None) or []
        playbooks_by_task: dict[str, Playbook] = {}

        for task_id, indices in task_to_indices.items():
            playbook = self._get_task_playbook(task_id, is_eval=is_eval)
            playbooks_by_task[task_id] = playbook
            playbook_text = playbook.to_string()
            rendered_prompt = self._render_base_system_prompt(playbook_text)
            for idx in indices:
                if idx >= len(agents):
                    continue
                self._apply_playbook_prompt_to_agent(
                    agents[idx],
                    playbook_text=playbook_text,
                    rendered_prompt=rendered_prompt,
                )

        return playbooks_by_task

    def _group_trajectories_by_task(
        self,
        trajectories: list[dict],
        task_to_indices: dict[str, list[int]],
    ) -> dict[str, list[dict]]:
        grouped = {task_id: [] for task_id in task_to_indices}
        idx_to_task_id = {int(idx): task_id for task_id, indices in task_to_indices.items() for idx in indices}

        for traj in trajectories:
            task_id = idx_to_task_id.get(int(traj.get("idx", -1)))
            if task_id is not None:
                grouped[task_id].append(traj)

        return grouped

    def _ordered_grouped_entries(
        self,
        trajectories_by_task: dict[str, list[dict]],
        task_to_indices: dict[str, list[int]],
    ) -> list[tuple[int, int, int, str, dict]]:
        """Order rows by round, then env index: seed rows first, extra rows after."""
        ordered_entries = []
        for task_id, indices in task_to_indices.items():
            task_trajectories = trajectories_by_task.get(task_id, [])
            if not indices:
                continue
            for local_idx, traj in enumerate(task_trajectories):
                source_round = local_idx // len(indices)
                base_index = int(indices[local_idx % len(indices)])
                ordered_entries.append((source_round, base_index, local_idx, task_id, traj))

        ordered_entries.sort(key=lambda item: (item[0], item[1], item[2]))
        return ordered_entries

    def _flatten_grouped_trajectories_in_env_order(
        self,
        trajectories_by_task: dict[str, list[dict]],
        task_to_indices: dict[str, list[int]],
    ) -> list[dict]:
        flattened = []
        for new_idx, (_, _, _, _, traj) in enumerate(self._ordered_grouped_entries(trajectories_by_task, task_to_indices)):
            traj["idx"] = new_idx
            flattened.append(traj)
        return flattened

    def _flatten_grouped_task_ids_in_env_order(
        self,
        trajectories_by_task: dict[str, list[dict]],
        task_to_indices: dict[str, list[int]],
    ) -> list[str]:
        return [task_id for _, _, _, task_id, _ in self._ordered_grouped_entries(trajectories_by_task, task_to_indices)]

    def _flatten_grouped_base_indices_in_env_order(
        self,
        trajectories_by_task: dict[str, list[dict]],
        task_to_indices: dict[str, list[int]],
    ) -> list[int]:
        """Env index of the seed row that each flattened row belongs to."""
        return [base_index for _, base_index, _, _, _ in self._ordered_grouped_entries(trajectories_by_task, task_to_indices)]

    def _save_task_policy_chat_logs(
        self,
        grouped_policy_trajectories: dict[str, list[dict]],
        meta_info=None,
    ) -> dict[str, str]:
        prev_task_id = self._artifact_task_id
        paths: dict[str, str] = {}
        self._last_policy_chat_paths = {}
        try:
            for task_id, task_trajectories in grouped_policy_trajectories.items():
                if not task_trajectories:
                    continue
                self._artifact_task_id = task_id
                path = super()._save_policy_chat_completions(
                    task_trajectories,
                    bsz=1,
                    meta_info=meta_info,
                )
                if path:
                    paths[task_id] = path
        finally:
            self._artifact_task_id = prev_task_id
        self._last_policy_chat_paths = dict(paths)
        return paths

    def _save_task_ace_logs(
        self,
        task_payloads: dict[str, dict],
        meta_info=None,
    ) -> dict[str, str]:
        prev_task_id = self._artifact_task_id
        paths: dict[str, str] = {}
        try:
            for task_id, payload in task_payloads.items():
                self._artifact_task_id = task_id
                path = super()._save_ace_step_log(payload, meta_info=meta_info)
                if path:
                    paths[task_id] = path
        finally:
            self._artifact_task_id = prev_task_id
        return paths

    def _build_local_candidate_eval_envs_and_agents(
        self,
        candidate_batch: list[tuple[int, dict]],
        original_envs_by_task: dict[str, list],
    ) -> tuple[list, list, list[dict]]:
        if not candidate_batch:
            return [], [], []

        full_agent_args = self._get_full_agent_args()
        base_env_args = self._get_full_env_args()
        eval_envs = []
        eval_agents = []
        eval_specs = []

        for candidate_idx, candidate in candidate_batch:
            task_id = str(candidate.get("task_id", "unknown"))
            source_envs = self._get_candidate_eval_source_envs(original_envs_by_task.get(task_id, []))
            if not source_envs:
                continue

            candidate_agents = []
            candidate_envs = []
            for rollout_idx, source_env in enumerate(source_envs):
                task_payload = copy.deepcopy(getattr(source_env, "task", None))
                if not isinstance(task_payload, dict):
                    raise TypeError(f"Local candidate evaluation requires env.task to be a dict payload, but got {type(task_payload).__name__}")
                candidate_envs.append(self.env_class.from_dict({**task_payload, **base_env_args}))
                candidate_agents.append(self.agent_class(**full_agent_args))
                eval_specs.append(
                    {
                        "candidate_idx": candidate_idx,
                        "task_id": task_id,
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

    def _evaluate_local_candidate_playbooks(
        self,
        candidate_playbooks: list[dict],
        original_envs_by_task: dict[str, list],
        *,
        capture_eval_trajectories: bool,
        timing_raw=None,
        meta_info=None,
    ) -> tuple[list[float], list[dict], dict[int, list[dict]], list[dict]]:
        self._last_candidate_eval_trajectories = None
        rewards, records, trajectories, token_metrics = self._evaluate_candidate_pool(
            candidate_playbooks,
            lambda candidates: self._build_local_candidate_eval_envs_and_agents(candidates, original_envs_by_task),
            capture_eval_trajectories=capture_eval_trajectories,
            timing_raw=timing_raw,
            meta_info=meta_info,
        )
        for record in records:
            record["task_id"] = candidate_playbooks[record["candidate_idx"]]["task_id"]
        self._last_candidate_eval_trajectories = trajectories if capture_eval_trajectories else None
        return rewards, records, trajectories, token_metrics

    def _store_local_eval_policy_trajectory_cache_by_task(
        self,
        trajectories_by_task: dict[str, list[dict]],
    ):
        for task_id, task_trajectories in trajectories_by_task.items():
            if not task_trajectories:
                self._local_eval_policy_trajectory_cache.pop(task_id, None)
                continue
            cached = [copy.deepcopy(traj) for traj in task_trajectories]
            for idx, traj in enumerate(cached):
                traj["idx"] = idx
            self._local_eval_policy_trajectory_cache[task_id] = cached

    def _pop_local_eval_policy_trajectory_cache_by_task(
        self,
        expected_counts: dict[str, int],
    ) -> dict[str, list[dict]]:
        """Reuse cached policy rows only when they cover the requested batch.

        Candidate evaluation may use fewer samples than ``val_kwargs.n``.
        Those selected-candidate rows cannot replace a full validation policy
        batch: renumbering them against the full env indices also changes task
        ownership. Regenerate the requested batch under the updated playbooks
        instead of duplicating samples or changing validation sample counts.
        """
        cached = {task_id: self._local_eval_policy_trajectory_cache.pop(task_id, None) for task_id in expected_counts}
        mismatched = {task_id: (len(cached[task_id] or []), expected_count) for task_id, expected_count in expected_counts.items() if len(cached[task_id] or []) != expected_count}
        if mismatched:
            if any(cached.values()):
                logger.info(
                    "[LOCAL-ACE] Regenerating validation policy rollouts; cached/requested sample counts differ: %s",
                    mismatched,
                )
            return {}
        return cached

    def _get_local_task_root(self, task_id: str | None) -> str | None:
        if not self.local_artifact_root or not task_id:
            return None
        return os.path.join(self.local_artifact_root, f"task_{task_id}")

    def _get_playbook_snapshot_dir(self, meta_info=None) -> str | None:
        task_root = self._get_local_task_root(self._artifact_task_id)
        if task_root:
            return os.path.join(task_root, "playbooks", self._artifact_mode(meta_info))
        return super()._get_playbook_snapshot_dir(meta_info)

    def _get_chat_completion_dir(self, meta_info=None) -> str | None:
        task_root = self._get_local_task_root(self._artifact_task_id)
        if task_root:
            return os.path.join(task_root, "chat_completions", self._artifact_mode(meta_info))
        return super()._get_chat_completion_dir(meta_info)

    def _get_ace_log_dir(self, meta_info=None) -> str | None:
        task_root = self._get_local_task_root(self._artifact_task_id)
        if task_root:
            return os.path.join(task_root, "ace_logs", self._artifact_mode(meta_info))
        return super()._get_ace_log_dir(meta_info)

    def _artifact_target_task_ids(self) -> list[str]:
        task_ids = getattr(self, "_artifact_task_ids", None) or []
        if task_ids:
            ordered = []
            seen = set()
            for tid in task_ids:
                tid = str(tid)
                if tid and tid not in seen:
                    seen.add(tid)
                    ordered.append(tid)
            if ordered:
                return ordered
        if self._artifact_task_id:
            return [str(self._artifact_task_id)]
        return []

    def _save_policy_chat_completions(self, policy_trajectories: list[dict], bsz: int, meta_info=None):
        task_ids = self._artifact_target_task_ids()
        self._last_policy_chat_paths = {}
        if len(task_ids) <= 1:
            path = super()._save_policy_chat_completions(policy_trajectories, bsz, meta_info=meta_info)
            if path and task_ids:
                self._last_policy_chat_paths[task_ids[0]] = path
            return path

        prev_task_id = self._artifact_task_id
        first_path = None
        try:
            for task_id in task_ids:
                self._artifact_task_id = task_id
                path = super()._save_policy_chat_completions(policy_trajectories, bsz, meta_info=meta_info)
                if path:
                    self._last_policy_chat_paths[task_id] = path
                    if first_path is None:
                        first_path = path
        finally:
            self._artifact_task_id = prev_task_id
        return first_path

    def _save_ace_step_log(self, payload: dict, meta_info=None):
        task_ids = self._artifact_target_task_ids()
        if len(task_ids) <= 1:
            return super()._save_ace_step_log(payload, meta_info=meta_info)

        prev_task_id = self._artifact_task_id
        first_path = None
        try:
            for task_id in task_ids:
                task_payload = dict(payload) if isinstance(payload, dict) else payload
                if isinstance(task_payload, dict):
                    chat_path = self._last_policy_chat_paths.get(task_id)
                    if chat_path:
                        if "chat_completions_path" in task_payload:
                            task_payload["chat_completions_path"] = chat_path
                        artifacts = task_payload.get("artifacts")
                        if isinstance(artifacts, dict) and "policy_chat_completions" in artifacts:
                            updated_artifacts = dict(artifacts)
                            updated_artifacts["policy_chat_completions"] = chat_path
                            task_payload["artifacts"] = updated_artifacts

                self._artifact_task_id = task_id
                path = super()._save_ace_step_log(task_payload, meta_info=meta_info)
                if first_path is None and path:
                    first_path = path
        finally:
            self._artifact_task_id = prev_task_id
        return first_path

    def _save_local_snapshot(self, task_id: str, playbook: Playbook, is_eval: bool):
        local_eval_epoch = self._current_local_eval_epoch if is_eval else None
        task_root = self._get_local_task_root(task_id)
        if task_root:
            snapshot_dir = os.path.join(task_root, "playbooks", "eval" if is_eval else "train")
            return _save_playbook_snapshot(
                playbook_text=playbook.to_string(),
                snapshot_dir=snapshot_dir,
                global_step=self.global_steps,
                local_eval_epoch=local_eval_epoch,
            )
        if not self.local_playbook_log_dir:
            return None
        epoch = self._estimate_epoch()
        return _save_task_playbook_snapshot(
            playbook_text=playbook.to_string(),
            base_dir=self.local_playbook_log_dir,
            task_id=task_id,
            step=self.global_steps,
            epoch=epoch,
            is_eval=is_eval,
            local_eval_epoch=local_eval_epoch,
        )

    def _generate_multi_task_batched_full_ace(
        self,
        task_ids: list[str],
        task_to_indices: dict[str, list[int]],
        timing_raw=None,
        meta_info=None,
    ):
        from verl.utils.debug import marked_timer

        from rllm.algorithms.ace.agents import (
            SingleTurnCuratorAgent,
            SingleTurnReflectorAgent,
        )
        from rllm.algorithms.ace.environments import (
            SingleTurnCuratorEnv,
            SingleTurnReflectorEnv,
        )

        is_validation = bool(meta_info and meta_info.get("validate", False))
        if timing_raw is None:
            timing_raw = {}

        parent_batch = self._current_batch
        original_envs = list(getattr(self.agent_execution_engine, "envs", None) or [])
        original_agents = list(getattr(self.agent_execution_engine, "agents", None) or [])
        if not original_envs or len(original_envs) != len(original_agents):
            return None
        self._prepare_task_playbooks_for_step(task_ids, is_eval=is_validation)

        active_component = self._get_active_component()
        if self._should_use_role_reduced_sampling():
            effective_g_ref = self._get_eval_g_ref(meta_info) if active_component in ("all", "reflector") else 1
            effective_g_cur = self._get_eval_g_cur(meta_info) if active_component in ("all", "curator") else 1
        else:
            effective_g_ref = self._get_eval_g_ref(meta_info)
            effective_g_cur = self._get_eval_g_cur(meta_info)

        playbooks_by_task = {task_id: self._get_task_playbook(task_id=task_id, is_eval=is_validation) for task_id in task_ids}
        playbook_before_text_by_task = {task_id: playbook.to_string() for task_id, playbook in playbooks_by_task.items()}
        original_envs_by_task = {task_id: [original_envs[i] for i in task_to_indices.get(task_id, [])] for task_id in task_ids}

        schedule_state = self._capture_component_schedule_state()
        prev_task_id = self._artifact_task_id
        prev_task_ids = list(self._artifact_task_ids)
        prev_suppress_flush = self._suppress_inner_flush

        self._artifact_task_id = task_ids[0]
        self._artifact_task_ids = list(task_ids)
        self._last_candidate_eval_trajectories = None

        try:
            with self._keep_rollout_engine_awake(reason=(f"local batched {'validation' if is_validation else 'training'} ACE step with {len(task_ids)} task(s)")):
                if is_validation:
                    self._prime_local_validation_policy_cache(
                        task_ids=task_ids,
                        task_to_indices=task_to_indices,
                        timing_raw=timing_raw,
                        meta_info=meta_info,
                    )

                self._suppress_inner_flush = True

                reused_validation_rollout = False
                cached_policy_by_task = {}
                if is_validation:
                    cached_policy_by_task = self._pop_local_eval_policy_trajectory_cache_by_task({task_id: len(task_to_indices[task_id]) for task_id in task_ids})

                with marked_timer("ace_act", timing_raw):
                    self._set_component_token_limits("policy")
                    self._inject_task_playbooks_into_agents(
                        task_to_indices,
                        is_eval=is_validation,
                    )

                    if is_validation and all(cached_policy_by_task.get(task_id) for task_id in task_ids):
                        reused_validation_rollout = True
                        policy_trajectories = self._flatten_grouped_trajectories_in_env_order(
                            {task_id: [copy.deepcopy(traj) for traj in cached_policy_by_task.get(task_id, [])] for task_id in task_ids},
                            task_to_indices,
                        )
                        logger.info(
                            "[LOCAL-ACE] Reusing cached validation policy rollouts for %s tasks",
                            len(task_ids),
                        )
                    else:
                        policy_trajectories = []
                        gen = self.generate_agent_trajectories_async(
                            timing_raw=timing_raw,
                            meta_info=meta_info,
                            mode="Token",
                        )
                        for traj in gen:
                            policy_trajectories.append(traj)
                        policy_trajectories.sort(key=lambda x: x["idx"])

                actor_reward_metrics = raw_reward_metrics(policy_trajectories, expected_count=len(original_envs))
                grouped_policy_trajectories = self._group_trajectories_by_task(
                    policy_trajectories,
                    task_to_indices,
                )
                policy_chat_paths = self._save_task_policy_chat_logs(
                    grouped_policy_trajectories,
                    meta_info=meta_info,
                )

                current_rewards_by_task = {}
                for task_id in task_ids:
                    task_rewards = [float(traj.get("trajectory_reward", 0.0)) for traj in grouped_policy_trajectories.get(task_id, [])]
                    current_rewards_by_task[task_id] = float(np.mean(task_rewards)) if task_rewards else 0.0
                current_rewards = [float(traj.get("trajectory_reward", 0.0)) for traj in policy_trajectories]
                current_avg_reward = float(np.mean(current_rewards)) if current_rewards else 0.0

                self._original_agents = original_agents
                self._original_envs = original_envs

                reflector_trajectories = []
                all_reflections = []
                reflector_records = []
                reflections_by_task = {task_id: [[{} for _ in range(effective_g_ref)] for _ in grouped_policy_trajectories.get(task_id, [])] for task_id in task_ids}
                reflection_source_indices_by_task = {task_id: self._select_reflection_source_indices(grouped_policy_trajectories.get(task_id, [])) for task_id in task_ids}

                with marked_timer("ace_reflect", timing_raw):
                    self._set_component_token_limits("reflector")
                    reflector_prompts = []
                    for task_idx, task_id in enumerate(task_ids):
                        task_playbook_text = playbook_before_text_by_task[task_id]
                        task_group = grouped_policy_trajectories.get(task_id, [])
                        for traj_idx in reflection_source_indices_by_task.get(task_id, []):
                            traj_dict = task_group[traj_idx]
                            prompt_text = self._build_reflector_prompt_from_dict(
                                traj_dict,
                                task_playbook_text,
                                task_idx,
                            )
                            for insight_idx in range(effective_g_ref):
                                guidebook_uid = self._make_guidebook_uid(
                                    task_id=task_id,
                                    task_idx=task_idx,
                                    traj_idx=traj_idx,
                                    insight_idx=insight_idx,
                                )
                                reflector_prompts.append(
                                    {
                                        "prompt_text": prompt_text,
                                        "env_task_id": f"ref::{task_id}::traj{traj_idx}::ins{insight_idx}",
                                        "task_id": task_id,
                                        "task_idx": task_idx,
                                        "traj_idx": traj_idx,
                                        "insight_idx": insight_idx,
                                        "guidebook_uid": guidebook_uid,
                                        "reflector_uid": f"{task_id}::ref_t{task_idx}_traj{traj_idx}",
                                    }
                                )

                    if reflector_prompts:
                        ref_agents = [
                            SingleTurnReflectorAgent(
                                system_prompt=self.reflector_system_prompt,
                            )
                            for _ in reflector_prompts
                        ]
                        ref_envs = [
                            SingleTurnReflectorEnv(
                                prompt_text=prompt["prompt_text"],
                                task_id=prompt["env_task_id"],
                            )
                            for prompt in reflector_prompts
                        ]
                        self.agent_execution_engine.update_envs_and_agents(ref_envs, ref_agents)
                        try:
                            ref_gen = self.generate_agent_trajectories_async(
                                timing_raw=timing_raw,
                                meta_info=meta_info,
                                mode="Token",
                            )
                            for traj in ref_gen:
                                reflector_trajectories.append(traj)
                        finally:
                            self.agent_execution_engine.update_envs_and_agents(
                                self._original_envs,
                                self._original_agents,
                            )

                    reflector_trajectories.sort(key=lambda x: x["idx"])
                    for i, traj in enumerate(reflector_trajectories):
                        if i >= len(reflector_prompts):
                            break
                        response_text = self._decode_trajectory_response(traj)
                        reflection = _coerce_reflection_payload(
                            _parse_json_response(response_text),
                            response_text=response_text,
                            context="local.reflector",
                        )
                        prompt = reflector_prompts[i]
                        task_id = prompt["task_id"]
                        traj_idx = int(prompt["traj_idx"])
                        insight_idx = int(prompt["insight_idx"])
                        if task_id in reflections_by_task and traj_idx < len(reflections_by_task[task_id]) and insight_idx < len(reflections_by_task[task_id][traj_idx]):
                            reflections_by_task[task_id][traj_idx][insight_idx] = reflection

                        reward = 0.0
                        task_group = grouped_policy_trajectories.get(task_id, [])
                        if traj_idx < len(task_group):
                            reward = float(task_group[traj_idx].get("trajectory_reward", 0.0))

                        all_reflections.append(
                            {
                                "task_id": task_id,
                                "task_idx": int(prompt["task_idx"]),
                                "traj_idx": traj_idx,
                                "insight_idx": insight_idx,
                                "guidebook_uid": prompt["guidebook_uid"],
                                "reflector_uid": prompt["reflector_uid"],
                                "reflection": reflection,
                                "reward": reward,
                            }
                        )
                        reflector_records.append(
                            {
                                "task_id": task_id,
                                "task_idx": int(prompt["task_idx"]),
                                "traj_idx": traj_idx,
                                "insight_idx": insight_idx,
                                "guidebook_uid": prompt["guidebook_uid"],
                                "prompt_text": prompt["prompt_text"],
                                "response_text": response_text,
                                "parsed_reflection": reflection,
                                "seed_reward": reward,
                            }
                        )

                curator_trajectories = []
                candidate_playbooks = []
                curator_to_guidebook = []
                curator_records = []

                with marked_timer("ace_curate", timing_raw):
                    self._set_component_token_limits("curator")
                    curator_prompts = []
                    for task_idx, task_id in enumerate(task_ids):
                        task_group = grouped_policy_trajectories.get(task_id, [])
                        if not task_group:
                            continue
                        task_playbook_text = playbook_before_text_by_task[task_id]
                        task_curator_specs = []
                        if self._uses_single_guidebook_aggregation():
                            for traj_idx in reflection_source_indices_by_task.get(task_id, []):
                                for insight_idx in range(effective_g_ref):
                                    guidebook_uid = self._make_guidebook_uid(
                                        task_id=task_id,
                                        task_idx=task_idx,
                                        traj_idx=traj_idx,
                                        insight_idx=insight_idx,
                                    )
                                    merged_insights = _concatenate_insights(
                                        [
                                            {
                                                "task_idx": task_idx,
                                                "reflection": reflections_by_task[task_id][traj_idx][insight_idx],
                                            }
                                        ]
                                    )
                                    if not merged_insights.strip():
                                        continue

                                    task_curator_specs.append(
                                        {
                                            "prompt_text": self._build_curator_prompt(
                                                merged_insights,
                                                task_playbook_text,
                                                task_instr=self._get_policy_task_instruction(task_group[traj_idx], task_idx),
                                            ),
                                            "env_task_id": f"cur::{guidebook_uid}",
                                            "guidebook_uid": guidebook_uid,
                                            "task_id": task_id,
                                            "task_idx": task_idx,
                                            "traj_idx": traj_idx,
                                            "insight_idx": insight_idx,
                                        }
                                    )
                        else:
                            for insight_idx in range(effective_g_ref):
                                guidebook_uid = self._make_guidebook_uid(
                                    task_id=task_id,
                                    task_idx=task_idx,
                                    insight_idx=insight_idx,
                                )
                                guidebook_insights = []
                                for traj_idx, _ in enumerate(task_group):
                                    guidebook_insights.append(
                                        {
                                            "task_idx": task_idx,
                                            "reflection": reflections_by_task[task_id][traj_idx][insight_idx],
                                        }
                                    )

                                merged_insights = _concatenate_insights(guidebook_insights)
                                if not merged_insights.strip():
                                    continue

                                task_curator_specs.append(
                                    {
                                        "prompt_text": self._build_curator_prompt(
                                            merged_insights,
                                            task_playbook_text,
                                            task_instr=self._get_policy_task_instruction(task_group[0], task_idx),
                                        ),
                                        "env_task_id": f"cur::{guidebook_uid}",
                                        "guidebook_uid": guidebook_uid,
                                        "task_id": task_id,
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
                        cur_agents = [
                            SingleTurnCuratorAgent(
                                system_prompt=self.curator_system_prompt,
                            )
                            for _ in curator_prompts
                        ]
                        cur_envs = [
                            SingleTurnCuratorEnv(
                                prompt_text=prompt["prompt_text"],
                                task_id=prompt["env_task_id"],
                            )
                            for prompt in curator_prompts
                        ]
                        self.agent_execution_engine.update_envs_and_agents(cur_envs, cur_agents)
                        try:
                            cur_gen = self.generate_agent_trajectories_async(
                                timing_raw=timing_raw,
                                meta_info=meta_info,
                                mode="Token",
                            )
                            for traj in cur_gen:
                                curator_trajectories.append(traj)
                        finally:
                            self.agent_execution_engine.update_envs_and_agents(
                                self._original_envs,
                                self._original_agents,
                            )

                    curator_trajectories.sort(key=lambda x: x["idx"])
                    for i, traj in enumerate(curator_trajectories):
                        if i >= len(curator_prompts):
                            break
                        response_text = self._decode_trajectory_response(traj)
                        ops = _parse_json_response(response_text)
                        prompt = curator_prompts[i]
                        task_id = prompt["task_id"]
                        curator_to_guidebook.append(prompt["guidebook_uid"])
                        record = {
                            "task_id": task_id,
                            "task_idx": int(prompt["task_idx"]),
                            "traj_idx": prompt.get("traj_idx"),
                            "insight_idx": prompt.get("insight_idx"),
                            "guidebook_uid": prompt["guidebook_uid"],
                            "prompt_text": prompt["prompt_text"],
                            "response_text": response_text,
                            "parsed_operations": ops,
                            "curator_prompt_idx": prompt.get("curator_prompt_idx", 0),
                            "curator_sample_idx": prompt.get("curator_sample_idx", 0),
                            "curator_samples_for_prompt": prompt.get("curator_samples_for_prompt", 1),
                            "curator_grouped": bool(prompt.get("curator_grouped", False)),
                            "curator_group_sampling_mode": prompt.get("curator_group_sampling_mode", "all_groups"),
                            "accepted": False,
                        }

                        if isinstance(ops, list):
                            ops_payload = {"operations": ops}
                        elif isinstance(ops, dict):
                            ops_payload = ops
                        else:
                            ops_payload = {}

                        if ops_payload.get("operations"):
                            candidate = Playbook(playbook_before_text_by_task[task_id])
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
                                candidate_text = candidate.to_string()
                                if self.max_playbook_length is not None:
                                    candidate_len = _get_playbook_token_length(
                                        candidate_text,
                                        self.tokenizer,
                                    )
                                    record["candidate_length"] = candidate_len
                                    if candidate_len > self.max_playbook_length:
                                        record["reject_reason"] = "max_playbook_length"
                                        record["candidate_playbook"] = candidate_text
                                        curator_records.append(record)
                                        continue

                                candidate_playbooks.append(
                                    {
                                        "playbook": candidate,
                                        "curator_idx": i,
                                        "guidebook_uid": prompt["guidebook_uid"],
                                        "task_id": task_id,
                                        "curator_grouped": bool(prompt.get("curator_grouped", False)),
                                        "curator_sample_idx": prompt.get("curator_sample_idx", 0),
                                        "curator_samples_for_prompt": prompt.get("curator_samples_for_prompt", 1),
                                    }
                                )
                                record["accepted"] = True
                                record["candidate_index"] = len(candidate_playbooks) - 1
                                record["candidate_playbook"] = candidate_text
                            except Exception as exc:  # noqa: BLE001
                                record["error"] = str(exc)
                                logger.warning("[LOCAL-ACE][CURATE] apply_operations failed: %s", exc)
                        curator_records.append(record)

                candidate_rewards = []
                candidate_eval_records = []
                candidate_eval_trajectories = {}
                candidate_eval_token_metrics = []
                if candidate_playbooks:
                    with marked_timer("ace_evaluate", timing_raw):
                        self._set_component_token_limits("policy")
                        (
                            candidate_rewards,
                            candidate_eval_records,
                            candidate_eval_trajectories,
                            candidate_eval_token_metrics,
                        ) = self._evaluate_local_candidate_playbooks(
                            candidate_playbooks,
                            original_envs_by_task=original_envs_by_task,
                            capture_eval_trajectories=(is_validation or self._should_capture_selected_candidate_policy_trajectories()),
                            timing_raw=timing_raw,
                            meta_info=meta_info,
                        )

                selected_idx_by_task = {}
                selected_reward_by_task = {}
                best_candidate_reward_by_task = {}
                reject_candidate_by_task = {}
                playbook_after_text_by_task = {}
                snapshot_paths = {}
                next_validation_trajectories = {}
                grouped_policy_for_training_by_task = {}
                policy_train_source_by_task = {}
                task_playbook_sizes = []
                task_playbook_bullets = []
                task_metric_pairs = []
                store = self.local_eval_playbooks if is_validation else self.local_train_playbooks

                for task_id in task_ids:
                    task_current_reward = current_rewards_by_task.get(task_id, 0.0)
                    task_candidate_indices = [idx for idx, candidate in enumerate(candidate_playbooks) if candidate.get("task_id") == task_id]
                    selected_idx = -1
                    selected_reward = task_current_reward
                    best_candidate_reward = task_current_reward
                    reject_candidate = False
                    updated_playbook_text = playbook_before_text_by_task[task_id]
                    seed_policy_for_task = grouped_policy_trajectories.get(task_id, [])
                    policy_trajectories_for_training = [copy.deepcopy(traj) for traj in seed_policy_for_task]
                    policy_train_source_by_task[task_id] = "seed"

                    if task_candidate_indices:
                        task_candidate_rewards = [float(candidate_rewards[idx]) for idx in task_candidate_indices if idx < len(candidate_rewards)]
                        if task_candidate_rewards:
                            best_candidate_reward = float(np.max(task_candidate_rewards))
                            surviving_candidate_indices = task_candidate_indices
                            if self.selection_mode == "best_rejection":
                                reject_candidate = best_candidate_reward < task_current_reward
                            elif self.selection_mode == "best_rejection_update":
                                # Source rollout indices are stored in curator records,
                                # not in the standard candidate dictionaries.
                                surviving_candidate_indices = self._get_best_rejection_update_candidate_indices(
                                    [curator_records[cand["curator_idx"]] for cand in candidate_playbooks],
                                    candidate_rewards,
                                    grouped_policy_trajectories,
                                    task_candidate_indices,
                                )
                                reject_candidate = not surviving_candidate_indices
                            if not reject_candidate:
                                if self.selection_mode == "best_rejection_update":
                                    selected_idx = self._select_best_rejection_update_candidate_index(candidate_rewards, surviving_candidate_indices)
                                elif self.selection_mode in {"best", "best_rejection"}:
                                    selected_idx = max(
                                        surviving_candidate_indices,
                                        key=lambda idx: float(candidate_rewards[idx]),
                                    )
                                else:
                                    selected_idx = random.choice(task_candidate_indices)
                                selected_reward = float(candidate_rewards[selected_idx])
                                updated_playbook_text = candidate_playbooks[selected_idx]["playbook"].to_string()
                                selected_eval_trajectories = candidate_eval_trajectories.get(
                                    selected_idx,
                                    [],
                                )
                                expected_policy = len(seed_policy_for_task)
                                if self.policy_train_source != "seed_and_candidates":
                                    (
                                        policy_trajectories_for_training,
                                        policy_train_source_by_task[task_id],
                                    ) = self._select_policy_trajectories_for_training(
                                        seed_policy_for_task,
                                        selected_eval_trajectories,
                                        expected_policy_count=expected_policy,
                                        log_prefix=f"[LOCAL-ACE][SELECT][task={task_id}]",
                                    )

                    if self.policy_train_source == "seed_and_candidates" and not is_validation:
                        # Candidate-scoring rollouts are ordinary actor rollouts
                        # on this task; train them in the seed GRPO group.
                        policy_trajectories_for_training.extend(
                            copy.deepcopy(traj) for candidate_idx in task_candidate_indices for traj in candidate_eval_trajectories.get(candidate_idx, [])
                        )
                        policy_train_source_by_task[task_id] = "seed_and_candidates"

                    grouped_policy_for_training_by_task[task_id] = policy_trajectories_for_training
                    store[task_id] = Playbook(updated_playbook_text)
                    playbook_after_text_by_task[task_id] = updated_playbook_text
                    snapshot_paths[task_id] = self._save_local_snapshot(
                        task_id=task_id,
                        playbook=store[task_id],
                        is_eval=is_validation,
                    )
                    pb_stats = store[task_id].get_stats()
                    task_playbook_sizes.append(len(updated_playbook_text))
                    task_playbook_bullets.append(pb_stats.get("total_bullets", 0))
                    selected_idx_by_task[task_id] = selected_idx
                    selected_reward_by_task[task_id] = selected_reward
                    best_candidate_reward_by_task[task_id] = best_candidate_reward
                    reject_candidate_by_task[task_id] = reject_candidate
                    task_metric_pairs.append(
                        (
                            task_id,
                            {
                                "ace/current_reward": task_current_reward,
                                "ace/selected_reward": selected_reward,
                            },
                        )
                    )

                    if is_validation:
                        if not reject_candidate and selected_idx >= 0:
                            selected_eval_trajectories = candidate_eval_trajectories.get(selected_idx, [])
                            if selected_eval_trajectories:
                                next_validation_trajectories[task_id] = [copy.deepcopy(traj) for traj in selected_eval_trajectories]
                            else:
                                logger.warning(
                                    "[LOCAL-VALIDATION-REUSE] Missing eval trajectories for selected candidate %s on task %s; reusing current policy rollouts.",
                                    selected_idx,
                                    task_id,
                                )
                                next_validation_trajectories[task_id] = [copy.deepcopy(traj) for traj in grouped_policy_trajectories.get(task_id, [])]
                        else:
                            next_validation_trajectories[task_id] = [copy.deepcopy(traj) for traj in grouped_policy_trajectories.get(task_id, [])]

                if is_validation:
                    self._store_local_eval_policy_trajectory_cache_by_task(next_validation_trajectories)
                else:
                    self._clear_validation_policy_trajectory_cache()

                policy_trajectories_for_training = self._flatten_grouped_trajectories_in_env_order(
                    grouped_policy_for_training_by_task,
                    task_to_indices,
                )
                policy_step_reward_tasks = self._flatten_grouped_task_ids_in_env_order(
                    grouped_policy_for_training_by_task,
                    task_to_indices,
                )

                if self._original_agents is not None:
                    self.agent_execution_engine.update_envs_and_agents(
                        self._original_envs,
                        self._original_agents,
                    )
                    self._inject_task_playbooks_into_agents(
                        task_to_indices,
                        is_eval=is_validation,
                    )

                guidebook_rewards = {}
                curator_idx_to_reward = {}
                for candidate_idx, candidate in enumerate(candidate_playbooks):
                    task_id = candidate["task_id"]
                    if reject_candidate_by_task.get(task_id) and self.selection_mode != "best_rejection_update":
                        continue
                    if candidate_idx < len(candidate_rewards):
                        guidebook_rewards.setdefault(candidate["guidebook_uid"], []).append(float(candidate_rewards[candidate_idx]))
                        curator_idx_to_reward[candidate["curator_idx"]] = float(candidate_rewards[candidate_idx])

                for i, traj in enumerate(reflector_trajectories):
                    if i >= len(all_reflections):
                        traj["trajectory_reward"] = 0.0
                        continue
                    task_id = all_reflections[i]["task_id"]
                    if reject_candidate_by_task.get(task_id) and self.selection_mode != "best_rejection_update":
                        traj["trajectory_reward"] = 0.0
                        continue
                    guidebook_uid = all_reflections[i]["guidebook_uid"]
                    rewards = guidebook_rewards.get(guidebook_uid, [])
                    traj["trajectory_reward"] = float(np.mean(rewards)) if rewards else 0.0

                for i, traj in enumerate(curator_trajectories):
                    if i >= len(curator_records):
                        traj["trajectory_reward"] = 0.0
                        continue
                    task_id = curator_records[i]["task_id"]
                    if reject_candidate_by_task.get(task_id) and self.selection_mode != "best_rejection_update":
                        traj["trajectory_reward"] = 0.0
                        continue
                    traj["trajectory_reward"] = curator_idx_to_reward.get(i, 0.0)

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

                if schedule_state is not None:
                    self._restore_component_schedule_state(schedule_state)
                    if not is_validation:
                        self._advance_component_schedule()

                auxiliary_improvement_by_task = {
                    task_id: self._selected_playbook_improvement(
                        selected_idx_by_task[task_id],
                        selected_reward_by_task[task_id],
                        current_rewards_by_task.get(task_id, 0.0),
                        candidate_playbooks,
                        curator_records,
                        grouped_policy_trajectories,
                    )
                    for task_id in task_ids
                }
                auxiliary_admitted_task_ids = {task_id for task_id in task_ids if not auxiliary_gate_active or auxiliary_improvement_by_task[task_id][0]}
                training_reflector_trajectories, training_curator_trajectories, reflector_uids, curator_uids = self._prepare_auxiliary_training_rows(
                    reflector_trajectories,
                    curator_trajectories,
                    all_reflections,
                    curator_records,
                    curator_to_guidebook,
                    admitted_task_ids=auxiliary_admitted_task_ids if auxiliary_gate_active else None,
                )
                all_trajectories = policy_trajectories_for_training + training_reflector_trajectories + training_curator_trajectories
                for idx, traj in enumerate(all_trajectories):
                    traj["idx"] = idx

                with marked_timer("transform_trajectory", timing_raw):
                    final_gen_batch_output, transform_metrics = self._transform_agent_trajectories(all_trajectories)

                policy_step_rewards = []
                for traj in policy_trajectories_for_training:
                    step_rewards = traj.get("step_rewards", [])
                    if isinstance(step_rewards, list | tuple):
                        policy_step_rewards.append([float(r) for r in step_rewards])
                    else:
                        policy_step_rewards.append([])
                self._current_step_rewards_list = policy_step_rewards if policy_step_rewards and any(len(sr) > 0 for sr in policy_step_rewards) else None
                self._current_step_reward_tasks = policy_step_reward_tasks if self._current_step_rewards_list is not None and len(policy_step_reward_tasks) == len(policy_step_rewards) else None

                if is_validation and parent_batch is not None:
                    self._current_validation_policy_tasks = list(parent_batch.non_tensor_batch.get("extra_info", [])[: len(policy_trajectories_for_training)])
                    self._current_validation_policy_sources = list(parent_batch.non_tensor_batch.get("data_source", [])[: len(policy_trajectories_for_training)])
                else:
                    self._current_validation_policy_tasks = None
                    self._current_validation_policy_sources = None

                n_policy = len(policy_trajectories_for_training)
                extra_policy_count = max(0, n_policy - len(policy_trajectories))
                share_policy_groups = self.policy_train_source == "seed_and_candidates" and not is_validation

                self._append_policy_and_component_rows_to_batch(
                    parent_batch,
                    extra_policy_count=extra_policy_count,
                    component_extra_uids=reflector_uids + curator_uids,
                    extra_policy_base_indices=(
                        self._flatten_grouped_base_indices_in_env_order(grouped_policy_for_training_by_task, task_to_indices)[len(policy_trajectories) :] if share_policy_groups else None
                    ),
                    share_extra_policy_uids=share_policy_groups,
                )

                metrics = dict(transform_metrics)
                if not is_validation:
                    metrics.update(actor_reward_metrics)
                    metrics.update(
                        self._auxiliary_training_metrics(
                            auxiliary_gate_active, len(auxiliary_admitted_task_ids), len(task_ids), n_policy, reflector_uids, curator_uids
                        )
                    )
                # Prompt/completion totals cover all generated rollouts;
                # gradient tokens cover only rows retained for PPO.
                metrics.update(
                    self._compute_rollout_token_totals(
                        policy_trajectories=policy_trajectories,
                        reflector_trajectories=reflector_trajectories,
                        curator_trajectories=curator_trajectories,
                        candidate_eval_token_metrics=candidate_eval_token_metrics,
                        ppo_trajectories=all_trajectories,
                    )
                )
                if getattr(self, "handoff_protocol", "legacy") == "feedback_v1":
                    metrics.update(self._curator_mutation_metrics(curator_records))
                metrics["ace/policy_rollouts"] = len(policy_trajectories)
                metrics["ace/reflector_rollouts"] = len(reflector_trajectories)
                metrics["ace/curator_rollouts"] = len(curator_trajectories)
                metrics["ace/candidate_eval_rollouts"] = len(candidate_eval_token_metrics)
                metrics["ace/extra_policy_training_rows"] = extra_policy_count
                # This remains the number of rollout rows used for training.
                metrics["ace/total_rollouts"] = len(all_trajectories)
                metrics["ace/candidates"] = len(candidate_playbooks)
                metrics["ace/singlemode"] = int(self.singlemode_enabled)
                metrics["ace/singlemode_curator_one_group"] = int(self._should_use_singlemode_curator_one_group())
                metrics["ace/guidebooks"] = len(reflector_records) if self.guidebook_aggregation_level == "single" else len(task_ids) * effective_g_ref
                metrics["ace/guidebook_aggregation_single"] = int(self.guidebook_aggregation_level == "single")
                metrics["ace/current_reward"] = float(np.mean(list(current_rewards_by_task.values()))) if current_rewards_by_task else current_avg_reward
                metrics["ace/selected_reward"] = float(np.mean(list(selected_reward_by_task.values()))) if selected_reward_by_task else current_avg_reward
                metrics["ace/best_candidate_reward"] = float(np.mean(list(best_candidate_reward_by_task.values()))) if best_candidate_reward_by_task else current_avg_reward
                metrics["ace/reject_candidate"] = float(np.mean([int(v) for v in reject_candidate_by_task.values()])) if reject_candidate_by_task else 0.0
                metrics["ace/active_component"] = hash(active_component) % 1000
                policy_source_ids = [self._policy_train_source_metric_id(policy_train_source_by_task.get(task_id, "seed")) for task_id in task_ids]
                metrics["ace/policy_train_source_id"] = float(np.mean(policy_source_ids)) if policy_source_ids else -1.0
                metrics["ace/validate_policy_only"] = 0
                metrics["ace/reused_validation_policy_rollout"] = int(reused_validation_rollout)
                metrics["playbook/size"] = float(np.mean(task_playbook_sizes)) if task_playbook_sizes else 0.0
                metrics["playbook/bullets"] = float(np.mean(task_playbook_bullets)) if task_playbook_bullets else 0.0
                metrics["local/task_id_hash"] = hash(task_ids[0]) % 1000000
                metrics["local/task_batch_size"] = len(task_ids)
                metrics["local/task_playbook_size"] = float(np.mean(task_playbook_sizes)) if task_playbook_sizes else 0.0
                metrics["local/task_playbook_count"] = len(self.local_eval_playbooks) if is_validation else len(self.local_train_playbooks)
                self._add_task_flush_metrics(metrics, task_ids, is_eval=is_validation)

                task_payloads = {}
                for task_id in task_ids:
                    task_candidates = [candidate for candidate in candidate_playbooks if candidate.get("task_id") == task_id]
                    task_payloads[task_id] = {
                        "step": int(self.global_steps),
                        "mode": self._artifact_mode(meta_info),
                        "active_component": active_component,
                        "counts": {
                            "policy_rollouts": len(grouped_policy_trajectories.get(task_id, [])),
                            "reflector_rollouts": sum(1 for record in reflector_records if record.get("task_id") == task_id),
                            "curator_rollouts": sum(1 for record in curator_records if record.get("task_id") == task_id),
                            "guidebooks": (
                                sum(1 for record in reflector_records if record.get("task_id") == task_id)
                                if self.guidebook_aggregation_level == "single"
                                else effective_g_ref
                                if grouped_policy_trajectories.get(task_id)
                                else 0
                            ),
                            "candidates": len(task_candidates),
                            "guidebook_aggregation_level": self.guidebook_aggregation_level,
                            "singlemode": self.singlemode_enabled,
                            "reflection_source_selection": getattr(self, "reflection_source_selection", "random"),
                            "singlemode_curator_one_group": self._should_use_singlemode_curator_one_group(),
                        },
                        "paths": {
                            "policy_chat_completions": policy_chat_paths.get(task_id),
                            "playbook_snapshot": snapshot_paths.get(task_id),
                        },
                        "selection": {
                            "selected_candidate_idx": selected_idx_by_task.get(task_id, -1),
                            "selected_reward": selected_reward_by_task.get(task_id, current_rewards_by_task.get(task_id, 0.0)),
                            "best_candidate_reward": best_candidate_reward_by_task.get(task_id, current_rewards_by_task.get(task_id, 0.0)),
                            "reject_candidate": bool(reject_candidate_by_task.get(task_id, False)),
                            "selection_mode": self.selection_mode,
                            "policy_train_source": policy_train_source_by_task.get(task_id, "seed"),
                            "auxiliary_update_gate_active": auxiliary_gate_active,
                            "auxiliary_update_baseline": auxiliary_improvement_by_task[task_id][1],
                            "auxiliary_update_improved": auxiliary_improvement_by_task[task_id][0],
                            "auxiliary_training_admitted": task_id in auxiliary_admitted_task_ids,
                        },
                        "playbook": {
                            "before": playbook_before_text_by_task.get(task_id, ""),
                            "after": playbook_after_text_by_task.get(task_id, ""),
                        },
                        "playbook_flush": self._task_flush_payload(
                            task_id,
                            is_eval=is_validation,
                        ),
                        "prompt_paths": self.prompt_paths,
                        "reflector": [record for record in reflector_records if record.get("task_id") == task_id],
                        "curator": [record for record in curator_records if record.get("task_id") == task_id],
                        "evaluation": [record for record in candidate_eval_records if record.get("task_id") == task_id],
                    }
                self._save_task_ace_logs(task_payloads, meta_info=meta_info)

                if not is_validation:
                    epoch = self._estimate_epoch()
                    if epoch > self._current_epoch:
                        self._flush_epoch_table()
                        self._current_epoch = epoch

                    for task_id, task_metrics in task_metric_pairs:
                        playbook = self.local_train_playbooks.get(task_id)
                        if playbook is None:
                            continue
                        pb_stats = playbook.get_stats()
                        self._epoch_task_data.append(
                            {
                                "epoch": epoch,
                                "task_id": task_id,
                                "reward": task_metrics.get("ace/current_reward", 0.0),
                                "selected_reward": task_metrics.get("ace/selected_reward", 0.0),
                                "playbook_size": len(playbook.to_string()),
                                "playbook_bullets": pb_stats.get("total_bullets", 0),
                            }
                        )

                if task_ids:
                    self.playbook = Playbook(playbook_after_text_by_task.get(task_ids[0], self.initial_playbook_text))

                return final_gen_batch_output, metrics
        finally:
            self.agent_execution_engine.update_envs_and_agents(original_envs, original_agents)
            self._artifact_task_id = prev_task_id
            self._artifact_task_ids = prev_task_ids
            self._suppress_inner_flush = prev_suppress_flush

    def _should_use_policy_only_validation(self, meta_info=None) -> bool:
        return not self._in_local_eval

    def _pop_validation_policy_trajectory_cache(self):
        if self._in_local_eval:
            task_id = str(self._artifact_task_id) if self._artifact_task_id is not None else None
            if not task_id:
                return None
            envs = getattr(self.agent_execution_engine, "envs", None) or []
            return self._pop_local_eval_policy_trajectory_cache_by_task({task_id: len(envs)}).get(task_id)
        return super()._pop_validation_policy_trajectory_cache()

    def _store_validation_policy_trajectory_cache(self, trajectories):
        if self._in_local_eval:
            task_id = str(self._artifact_task_id) if self._artifact_task_id is not None else None
            if task_id:
                if trajectories:
                    self._local_eval_policy_trajectory_cache[task_id] = trajectories
                else:
                    self._local_eval_policy_trajectory_cache.pop(task_id, None)
                return
        super()._store_validation_policy_trajectory_cache(trajectories)

    def _clear_validation_policy_trajectory_cache(self):
        self._local_eval_policy_trajectory_cache = {}
        super()._clear_validation_policy_trajectory_cache()

    def _should_run_policy_only_phase(self, active_component: str) -> bool:
        if self._in_local_eval:
            return False
        return super()._should_run_policy_only_phase(active_component)

    def _should_use_role_reduced_sampling(self) -> bool:
        if self._in_local_eval:
            return False
        return super()._should_use_role_reduced_sampling()

    def _flush_playbook_now(self, *, step: int, reason: str) -> None:
        super()._flush_playbook_now(step=step, reason=reason)
        self.local_train_playbooks = {}
        self.local_eval_playbooks = {}
        self.local_train_task_seen_counts = {}
        self._current_task_seen_counts = {}
        self._current_task_flush_info = {}
        logger.info(
            "[FLUSH] Cleared local task-playbook caches at step=%s reason=%s",
            step,
            reason,
        )

    def _maybe_flush_playbook(self):
        if self._suppress_inner_flush:
            return

        super()._maybe_flush_playbook()

    def _flush_epoch_table(self):
        """Build and log a wandb.Table from accumulated epoch task data."""
        if not self._epoch_task_data:
            return
        try:
            import wandb

            table_columns = [
                "epoch",
                "task_id",
                "reward",
                "selected_reward",
                "playbook_size",
                "playbook_bullets",
            ]
            task_table = wandb.Table(columns=table_columns)
            for row in self._epoch_task_data:
                task_table.add_data(*[row[c] for c in table_columns])
            wandb.log({"epoch/task_table": task_table})
            logger.info(f"[EPOCH TABLE] Logged table with {len(self._epoch_task_data)} rows for epoch {self._epoch_task_data[0]['epoch']}")
        except Exception as e:
            logger.warning(f"Failed to log epoch task table: {e}")
        self._epoch_task_data = []

    def _aggregate_validation_batch_metrics(self, metrics_by_batch: list[dict]) -> dict:
        if not metrics_by_batch:
            return {}

        sum_keys = {
            "ace/policy_rollouts",
            "ace/reflector_rollouts",
            "ace/curator_rollouts",
            "ace/total_rollouts",
            "ace/candidates",
            "ace/guidebooks",
            "ace/auxiliary_tasks_admitted",
            "ace/auxiliary_tasks_dropped",
            "ace/policy_training_rows",
            "ace/reflector_training_rows",
            "ace/curator_training_rows",
            "ace/reflector_training_groups",
            "ace/curator_training_groups",
        }
        first_keys = {
            "ace/active_component",
            "ace/auxiliary_update_gate_active",
            "ace/validate_policy_only",
            "ace/guidebook_aggregation_single",
            "ace/singlemode",
        }

        aggregated = {}
        all_keys = set().union(*(metrics.keys() for metrics in metrics_by_batch))
        for key in all_keys:
            weighted_values = []
            weights = []
            for metrics in metrics_by_batch:
                if key not in metrics:
                    continue
                weight = float(metrics.get("local/task_batch_size", 1) or 1)
                weighted_values.append(metrics[key])
                weights.append(weight)

            if not weighted_values:
                continue
            if key in first_keys:
                aggregated[key] = weighted_values[0]
                continue

            numeric_values = [float(value) for value in weighted_values if isinstance(value, int | float | np.integer | np.floating)]
            if len(numeric_values) != len(weighted_values):
                aggregated[key] = weighted_values[0]
                continue

            if key in sum_keys:
                total = float(np.sum(numeric_values))
                aggregated[key] = int(total) if all(float(v).is_integer() for v in numeric_values) else total
                continue

            total_weight = float(np.sum(weights)) if weights else 0.0
            if total_weight <= 0:
                aggregated[key] = float(np.mean(numeric_values))
            else:
                aggregated[key] = float(np.sum([value * weight for value, weight in zip(numeric_values, weights, strict=False)]) / total_weight)

        return aggregated

    def _build_validation_turn_rows(self, step_rewards_list: list[list[float]]) -> tuple[list[dict], dict]:
        if not step_rewards_list:
            return [], {}

        actual_max_turns = max(len(rewards) for rewards in step_rewards_list)
        if actual_max_turns <= 0:
            return [], {}

        cutoff_mode = self.config.rllm.get("turn_level_cutoff_mode", "max")
        max_steps = self.config.rllm.agent.get("max_steps", None)
        max_turns = max_steps if max_steps is not None else actual_max_turns

        processed_rewards = []
        original_lengths = [len(rewards) for rewards in step_rewards_list]
        for rewards in step_rewards_list:
            if len(rewards) >= max_turns:
                processed_rewards.append(rewards[:max_turns])
                continue

            extended = list(rewards)
            if cutoff_mode == "max":
                fill_value = max(rewards) if rewards else 0.0
            elif cutoff_mode == "last":
                fill_value = rewards[-1] if rewards else 0.0
            else:
                fill_value = 0.0
            extended.extend([fill_value] * (max_turns - len(rewards)))
            processed_rewards.append(extended)

        turn_rows = []
        turn_rewards = []
        cumulative_rewards = []
        max_rewards = []
        min_rewards = []
        for turn_idx in range(max_turns):
            values = [rewards[turn_idx] for rewards in processed_rewards]
            turn_avg = float(np.mean(values)) if values else 0.0
            turn_rewards.append(turn_avg)
            cumulative_rewards.append(turn_avg + (cumulative_rewards[-1] if cumulative_rewards else 0.0))
            trajectory_max = [max(rewards[: turn_idx + 1]) for rewards in processed_rewards]
            trajectory_min = [min(rewards[: turn_idx + 1]) for rewards in processed_rewards]
            max_rewards.append(float(np.mean(trajectory_max)) if trajectory_max else 0.0)
            min_rewards.append(float(np.mean(trajectory_min)) if trajectory_min else 0.0)
            turn_rows.append(
                {
                    "turn": turn_idx,
                    "avg_reward": turn_rewards[-1],
                    "cumulative_reward": cumulative_rewards[-1],
                    "max_reward": max_rewards[-1],
                    "min_reward": min_rewards[-1],
                    "n_trajectories": sum(1 for length in original_lengths if turn_idx < length),
                }
            )

        summary = {}
        if turn_rows:
            summary = {
                "final_avg_reward": turn_rewards[-1],
                "final_cumulative_reward": cumulative_rewards[-1],
                "final_max_reward": max_rewards[-1],
                "best_avg_reward": max(turn_rewards),
                "best_max_reward": max(max_rewards),
            }
        return turn_rows, summary

    def _extract_validation_pass_at_k(self, metric_dict: dict) -> float:
        candidates = [float(value) for key, value in metric_dict.items() if key.endswith("/pass_at_k_max_reward") and isinstance(value, int | float | np.integer | np.floating)]
        return float(np.mean(candidates)) if candidates else 0.0

    def _get_wandb_logger(self, logger=None):
        wandb_logger = None
        try:
            if logger is not None and hasattr(logger, "logger") and isinstance(logger.logger, dict):
                wandb_logger = logger.logger.get("wandb")
        except Exception:
            wandb_logger = None

        if wandb_logger is not None:
            return wandb_logger

        try:
            import wandb

            if wandb.run is not None:
                return wandb
        except Exception:
            return None

        return None

    def _log_validation_turn_table(self, logger=None, step=0, rows: list[dict] | None = None):
        rows = rows or []
        if not rows:
            return

        wandb_logger = self._get_wandb_logger(logger)
        if wandb_logger is None:
            return

        import wandb

        base_columns = [
            "turn",
            "avg_reward",
            "cumulative_reward",
            "max_reward",
            "min_reward",
            "n_trajectories",
        ]
        extra_columns = []
        seen = set(base_columns)
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    extra_columns.append(key)

        table = wandb.Table(columns=base_columns + extra_columns)
        for row in rows:
            table.add_data(*([row.get(key) for key in base_columns] + [row.get(key) for key in extra_columns]))
        wandb_logger.log({"val/turn_table": table}, step=step)

    def _run_local_validation_epoch(self):
        from verl import DataProto

        step_rewards_by_source = {}
        tasks_by_source = {}
        batch_metrics = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            removable_prompt_keys = [key for key in ("input_ids", "attention_mask", "position_ids") if key in test_batch.batch.keys()]
            if removable_prompt_keys:
                test_batch.pop(removable_prompt_keys)
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            self.init_envs_and_agents(test_batch)

            generate_metrics = {}
            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)
            else:
                test_output_gen_batch, generate_metrics = self.generate_agent_trajectory(meta_info=test_batch.meta_info)
                if generate_metrics:
                    batch_metrics.append(generate_metrics)

            test_batch = test_batch.union(test_output_gen_batch)

            current_step_rewards = getattr(self, "_current_step_rewards_list", None)
            if current_step_rewards is None:
                reward_tensor = test_batch.batch["token_level_scores"]
                current_step_rewards = [[reward.item()] for reward in reward_tensor.sum(-1).cpu()]

            policy_sources = getattr(self, "_current_validation_policy_sources", None)
            policy_tasks = getattr(self, "_current_validation_policy_tasks", None)

            if policy_sources is not None and policy_tasks is not None and len(policy_sources) == len(current_step_rewards) and len(policy_tasks) == len(current_step_rewards):
                data_sources = policy_sources
                extra_info = policy_tasks
            else:
                expected_len = len(current_step_rewards)
                batch_size = len(test_batch.non_tensor_batch["uid"])
                raw_sources = test_batch.non_tensor_batch.get(
                    "data_source",
                    ["unknown"] * batch_size,
                )
                raw_tasks = test_batch.non_tensor_batch.get(
                    "extra_info",
                    [None] * batch_size,
                )
                data_sources = list(raw_sources[:expected_len])
                extra_info = list(raw_tasks[:expected_len])

            for data_source, task, step_rewards in zip(
                data_sources,
                extra_info,
                current_step_rewards,
                strict=False,
            ):
                normalized_source = normalize_validation_data_source(data_source)
                if normalized_source not in step_rewards_by_source:
                    step_rewards_by_source[normalized_source] = []
                    tasks_by_source[normalized_source] = []
                step_rewards_by_source[normalized_source].append(step_rewards)
                tasks_by_source[normalized_source].append(task)

            self._current_step_rewards_list = None
            self._current_validation_policy_sources = None
            self._current_validation_policy_tasks = None

        metric_dict = {}
        all_step_rewards = []
        for data_source, step_rewards_list in step_rewards_by_source.items():
            metric_dict.update(
                compute_validation_turn_metrics(
                    step_rewards_list=step_rewards_list,
                    tasks=tasks_by_source[data_source],
                    cutoff_mode=self.config.rllm.get("turn_level_cutoff_mode", "max"),
                    max_steps=self.config.rllm.agent.get("max_steps", None),
                    data_source=data_source,
                )
            )
            all_step_rewards.extend(step_rewards_list)

        turn_rows, turn_summary = self._build_validation_turn_rows(all_step_rewards)
        ace_metrics = self._aggregate_validation_batch_metrics(batch_metrics)
        ace_metrics["pass_at_k_max_reward"] = self._extract_validation_pass_at_k(metric_dict)
        return metric_dict, ace_metrics, turn_rows, turn_summary

    def _maybe_generate_batched_policy_only_training_step(self, timing_raw=None, meta_info=None):
        """Run policy-only local training as one batched rollout across tasks."""
        from verl.utils.debug import marked_timer

        if timing_raw is None:
            timing_raw = {}

        is_validation = bool(meta_info and meta_info.get("validate", False))
        if is_validation:
            return None

        active_component = self._get_active_component()
        if not self._should_run_policy_only_phase(active_component):
            return None

        task_ids, task_to_indices = self._build_task_index_groups()
        if len(task_ids) <= 1:
            return None

        logger.info(
            "[LOCAL-ACE] batched policy-only fast path for %s tasks (env_count=%s)",
            len(task_ids),
            len(getattr(self.agent_execution_engine, "envs", None) or []),
        )

        self._prepare_task_playbooks_for_step(task_ids, is_eval=False)

        prev_task_id = self._artifact_task_id
        prev_task_ids = list(self._artifact_task_ids)
        chat_log_path = None

        self._artifact_task_id = task_ids[0]
        self._artifact_task_ids = list(task_ids)
        try:
            with self._keep_rollout_engine_awake(reason=f"local policy-only training step with {len(task_ids)} task(s)"):
                with marked_timer("ace_act", timing_raw):
                    self._set_component_token_limits("policy")
                    playbooks_by_task = self._inject_task_playbooks_into_agents(
                        task_to_indices,
                        is_eval=False,
                    )

                    policy_trajectories = []
                    gen = self.generate_agent_trajectories_async(
                        timing_raw=timing_raw,
                        meta_info=meta_info,
                        mode="Token",
                    )
                    for traj in gen:
                        traj["_component_type"] = "policy"
                        policy_trajectories.append(traj)
                    policy_trajectories.sort(key=lambda x: x["idx"])
                    actor_reward_metrics = raw_reward_metrics(
                        policy_trajectories, expected_count=len(self.agent_execution_engine.envs)
                    )

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

            grouped_policy_trajectories = self._group_trajectories_by_task(
                policy_trajectories,
                task_to_indices,
            )

            rewards = [float(traj.get("trajectory_reward", 0.0)) for traj in policy_trajectories]
            avg_reward = float(np.mean(rewards)) if rewards else 0.0

            metrics = dict(transform_metrics)
            metrics.update(actor_reward_metrics)
            # Rollout totals: ALL tokens generated this step (including rejected)
            metrics.update(
                self._compute_rollout_token_totals(
                    policy_trajectories=policy_trajectories,
                )
            )
            metrics["ace/policy_rollouts"] = len(policy_trajectories)
            metrics["ace/reflector_rollouts"] = 0
            metrics["ace/curator_rollouts"] = 0
            metrics["ace/total_rollouts"] = len(policy_trajectories)
            metrics["ace/candidates"] = 0
            metrics["ace/current_reward"] = avg_reward
            metrics["ace/selected_reward"] = avg_reward
            metrics["ace/best_candidate_reward"] = avg_reward
            metrics["ace/reject_candidate"] = 0
            metrics["ace/active_component"] = hash("policy") % 1000
            metrics["ace/validate_policy_only"] = 0
            metrics["ace/singlemode"] = int(self.singlemode_enabled)
            metrics["ace/guidebook_aggregation_single"] = int(self.guidebook_aggregation_level == "single")

            task_metric_pairs = []
            playbook_sizes = []
            playbook_bullets = []
            for task_id in task_ids:
                task_playbook = playbooks_by_task[task_id]
                task_playbook_text = task_playbook.to_string()
                pb_stats = task_playbook.get_stats()
                task_rewards = [float(traj.get("trajectory_reward", 0.0)) for traj in grouped_policy_trajectories.get(task_id, [])]
                task_avg_reward = float(np.mean(task_rewards)) if task_rewards else 0.0
                playbook_sizes.append(len(task_playbook_text))
                playbook_bullets.append(pb_stats.get("total_bullets", 0))
                task_metric_pairs.append(
                    (
                        task_id,
                        {
                            "ace/current_reward": task_avg_reward,
                            "ace/selected_reward": task_avg_reward,
                        },
                    )
                )
                self._save_local_snapshot(
                    task_id=task_id,
                    playbook=task_playbook,
                    is_eval=False,
                )

            metrics["playbook/size"] = float(np.mean(playbook_sizes)) if playbook_sizes else 0.0
            metrics["playbook/bullets"] = float(np.mean(playbook_bullets)) if playbook_bullets else 0.0
            metrics["local/task_id_hash"] = hash(task_ids[0]) % 1000000
            metrics["local/task_batch_size"] = len(task_ids)
            metrics["local/task_playbook_size"] = float(np.mean(playbook_sizes)) if playbook_sizes else 0.0
            metrics["local/task_playbook_count"] = len(self.local_train_playbooks)
            self._add_task_flush_metrics(metrics, task_ids, is_eval=False)

            self._save_ace_step_log(
                {
                    "step": int(self.global_steps),
                    "mode": self._artifact_mode(meta_info),
                    "active_component": "policy",
                    "policy_only": True,
                    "counts": {
                        "policy_rollouts": len(policy_trajectories),
                        "reflector_rollouts": 0,
                        "curator_rollouts": 0,
                        "guidebooks": 0,
                        "candidates": 0,
                    },
                    "paths": {
                        "policy_chat_completions": chat_log_path,
                    },
                    "selection": {
                        "selected_candidate_idx": -1,
                        "selected_reward": avg_reward,
                        "best_candidate_reward": avg_reward,
                        "reject_candidate": False,
                        "selection_mode": self.selection_mode,
                    },
                    "playbook_flush": {task_id: self._task_flush_payload(task_id) for task_id in task_ids},
                    "metrics": metrics,
                },
                meta_info=meta_info,
            )

            epoch = self._estimate_epoch()
            if epoch > self._current_epoch:
                self._flush_epoch_table()
                self._current_epoch = epoch

            for task_id, task_metrics in task_metric_pairs:
                playbook = self.local_train_playbooks.get(task_id)
                pb_stats = playbook.get_stats() if playbook is not None else {}
                self._epoch_task_data.append(
                    {
                        "epoch": epoch,
                        "task_id": task_id,
                        "reward": task_metrics.get("ace/current_reward", 0.0),
                        "selected_reward": task_metrics.get("ace/selected_reward", 0.0),
                        "playbook_size": len(playbook.to_string()) if playbook is not None else 0,
                        "playbook_bullets": pb_stats.get("total_bullets", 0),
                    }
                )

            self._advance_component_schedule()
            return final_gen_batch_output, metrics
        finally:
            self._artifact_task_id = prev_task_id
            self._artifact_task_ids = prev_task_ids

    def _prime_local_validation_policy_cache(
        self,
        task_ids: list[str],
        task_to_indices: dict[str, list[int]],
        timing_raw=None,
        meta_info=None,
    ):
        """Batch the initial local-eval policy rollout so later per-task ACE reuses it."""
        from verl.utils.debug import marked_timer

        if not self._in_local_eval:
            return
        if not (meta_info and meta_info.get("validate", False)):
            return

        missing_task_ids = [task_id for task_id in task_ids if task_id not in self._local_eval_policy_trajectory_cache]
        if not missing_task_ids:
            return
        if len(missing_task_ids) != len(task_ids):
            return

        logger.info(
            "[LOCAL-ACE] priming local validation policy cache for %s tasks",
            len(missing_task_ids),
        )

        if timing_raw is None:
            timing_raw = {}

        with marked_timer("ace_act", timing_raw):
            self._set_component_token_limits("policy")
            self._inject_task_playbooks_into_agents(
                {task_id: task_to_indices[task_id] for task_id in missing_task_ids},
                is_eval=True,
            )

            policy_trajectories = []
            gen = self.generate_agent_trajectories_async(
                timing_raw=timing_raw,
                meta_info=meta_info,
                mode="Token",
            )
            for traj in gen:
                policy_trajectories.append(traj)
            policy_trajectories.sort(key=lambda x: x["idx"])

        grouped_policy_trajectories = self._group_trajectories_by_task(
            policy_trajectories,
            {task_id: task_to_indices[task_id] for task_id in missing_task_ids},
        )
        for task_id, task_trajectories in grouped_policy_trajectories.items():
            self._local_eval_policy_trajectory_cache[task_id] = task_trajectories

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        is_validation = bool(meta_info and meta_info.get("validate", False))

        # During validation, use local ACE only when explicitly requested.
        if is_validation and not self._in_local_eval:
            return self._generate_policy_only_trajectory(timing_raw=timing_raw, meta_info=meta_info)

        fast_path_result = self._maybe_generate_batched_policy_only_training_step(
            timing_raw=timing_raw,
            meta_info=meta_info,
        )
        if fast_path_result is not None:
            return fast_path_result

        n = self.config.actor_rollout_ref.rollout.n
        envs = getattr(self.agent_execution_engine, "envs", None) or []
        env_count = len(envs)
        if env_count > n and (env_count % n) != 0:
            logger.warning(f"Local ACE-RL expects env_count to be a multiple of rollout.n. Got env_count={env_count}, rollout.n={n}; behavior may deviate.")

        task_ids, task_to_indices = self._build_task_index_groups()
        if not task_ids:
            return ACERLAgentPPOTrainer.generate_agent_trajectory(self, timing_raw=timing_raw, meta_info=meta_info)

        if len(task_ids) > 1:
            logger.info(
                "Local ACE-RL per-task ACE update: %s tasks in this step (env_count=%s, rollout.n=%s)",
                len(task_ids),
                env_count,
                n,
            )

            batched_result = self._generate_multi_task_batched_full_ace(
                task_ids=task_ids,
                task_to_indices=task_to_indices,
                timing_raw=timing_raw,
                meta_info=meta_info,
            )
            if batched_result is not None:
                return batched_result

        self._prepare_task_playbooks_for_step(task_ids, is_eval=is_validation)

        parent_batch = self._current_batch
        original_envs = list(getattr(self.agent_execution_engine, "envs", None) or [])
        original_agents = list(getattr(self.agent_execution_engine, "agents", None) or [])
        if not original_envs or len(original_envs) != len(original_agents):
            logger.warning(
                "Unexpected env/agent state (envs=%s, agents=%s). Falling back to base ACE call.",
                len(original_envs),
                len(original_agents),
            )
            return ACERLAgentPPOTrainer.generate_agent_trajectory(self, timing_raw=timing_raw, meta_info=meta_info)

        from verl import DataProto

        schedule_state = self._capture_component_schedule_state()
        prev_task_id = self._artifact_task_id
        prev_task_ids = list(self._artifact_task_ids)
        prev_current_batch = self._current_batch
        prev_suppress_flush = self._suppress_inner_flush

        task_outputs: list[DataProto] = []
        expanded_sub_batches: list[DataProto] = []
        task_metric_pairs: list[tuple[str, dict]] = []
        combined_step_rewards: list[list[float]] = []
        combined_step_reward_tasks: list[object] = []
        playbook_size_sum = 0
        validation_policy_tasks: list[object] = []
        validation_policy_sources: list[object] = []

        with self._keep_rollout_engine_awake(reason=(f"local {'validation' if is_validation else 'training'} step with {len(task_ids)} task(s)")):
            if is_validation and len(task_ids) > 1:
                self._prime_local_validation_policy_cache(
                    task_ids=task_ids,
                    task_to_indices=task_to_indices,
                    timing_raw=timing_raw,
                    meta_info=meta_info,
                )
            self._suppress_inner_flush = True
            try:
                for task_id in task_ids:
                    indices = task_to_indices.get(task_id, [])
                    if not indices:
                        continue

                    if schedule_state is not None:
                        self._restore_component_schedule_state(schedule_state)

                    sub_envs = [original_envs[i] for i in indices]
                    sub_agents = [original_agents[i] for i in indices]
                    self.agent_execution_engine.update_envs_and_agents(sub_envs, sub_agents)

                    sub_batch = None
                    if parent_batch is not None:
                        sub_batch = parent_batch.select_idxs(np.array(indices, dtype=np.int64))
                        self._current_batch = sub_batch
                    else:
                        self._current_batch = None

                    policy_task_entries = []
                    policy_source_entries = []
                    if is_validation and sub_batch is not None:
                        raw_tasks = sub_batch.non_tensor_batch.get("extra_info", [])
                        raw_sources = sub_batch.non_tensor_batch.get("data_source", [])
                        policy_task_entries = list(raw_tasks)
                        policy_source_entries = list(raw_sources)

                    current_task_playbook = self._get_task_playbook(task_id=task_id, is_eval=is_validation)
                    self.playbook = Playbook(current_task_playbook.to_string())
                    self._last_policy_chat_paths = {}
                    self._artifact_task_id = task_id
                    self._artifact_task_ids = [task_id]

                    task_output, task_metrics = ACERLAgentPPOTrainer.generate_agent_trajectory(self, timing_raw=timing_raw, meta_info=meta_info)
                    task_outputs.append(task_output)
                    task_metric_pairs.append((task_id, task_metrics))

                    if sub_batch is not None:
                        expanded_sub_batches.append(sub_batch)

                    task_step_rewards = getattr(self, "_current_step_rewards_list", None) or []
                    task_step_reward_tasks = getattr(self, "_current_step_reward_tasks", None)
                    for rewards in task_step_rewards:
                        combined_step_rewards.append([float(r) for r in rewards])
                    if task_step_reward_tasks is not None and len(task_step_reward_tasks) == len(task_step_rewards):
                        combined_step_reward_tasks.extend(task_step_reward_tasks)
                    else:
                        combined_step_reward_tasks.extend([task_id] * len(task_step_rewards))
                    if is_validation and task_step_rewards:
                        policy_count = len(task_step_rewards)
                        if policy_source_entries:
                            validation_policy_sources.extend(policy_source_entries[:policy_count])
                        else:
                            validation_policy_sources.extend(["unknown"] * policy_count)
                        if policy_task_entries:
                            validation_policy_tasks.extend(policy_task_entries[:policy_count])
                        else:
                            validation_policy_tasks.extend([None] * policy_count)

                    updated_playbook = Playbook(self.playbook.to_string())
                    playbook_size_sum += len(updated_playbook.to_string())
                    store = self.local_eval_playbooks if is_validation else self.local_train_playbooks
                    store[task_id] = Playbook(updated_playbook.to_string())
                    self._save_local_snapshot(task_id=task_id, playbook=store[task_id], is_eval=is_validation)

            finally:
                self.agent_execution_engine.update_envs_and_agents(original_envs, original_agents)
                self._artifact_task_id = prev_task_id
                self._artifact_task_ids = prev_task_ids
                self._current_batch = prev_current_batch
                self._suppress_inner_flush = prev_suppress_flush

        if schedule_state is not None:
            # Inner calls may advance the iterative schedule. Restore it for
            # validation so eval never changes training state, and advance only
            # once for the outer training step.
            self._restore_component_schedule_state(schedule_state)
            if not is_validation:
                self._advance_component_schedule()

        if not task_outputs:
            return ACERLAgentPPOTrainer.generate_agent_trajectory(self, timing_raw=timing_raw, meta_info=meta_info)

        output = task_outputs[0] if len(task_outputs) == 1 else DataProto.concat(task_outputs)

        # Replace the current batch in-place with concatenated expanded sub-batches
        # so parent fit_agent() can union() with mixed ACE trajectories.
        if parent_batch is not None and expanded_sub_batches:
            expanded_batch = expanded_sub_batches[0] if len(expanded_sub_batches) == 1 else DataProto.concat(expanded_sub_batches)
            parent_batch.non_tensor_batch = expanded_batch.non_tensor_batch
            parent_batch.batch = expanded_batch.batch

        metrics = self._aggregate_task_metrics([m for _, m in task_metric_pairs])

        # Keep turn/* metrics policy-only across all tasks in the outer batch.
        self._current_step_rewards_list = combined_step_rewards if combined_step_rewards and any(len(sr) > 0 for sr in combined_step_rewards) else None
        self._current_step_reward_tasks = combined_step_reward_tasks if self._current_step_rewards_list is not None and len(combined_step_reward_tasks) == len(combined_step_rewards) else None
        if is_validation and validation_policy_tasks and validation_policy_sources:
            self._current_validation_policy_tasks = validation_policy_tasks
            self._current_validation_policy_sources = validation_policy_sources
        else:
            self._current_validation_policy_tasks = None
            self._current_validation_policy_sources = None

        primary_task_id = task_ids[0]
        mean_playbook_size = (playbook_size_sum / len(task_metric_pairs)) if task_metric_pairs else 0.0
        metrics["local/task_id_hash"] = hash(primary_task_id) % 1000000
        metrics["local/task_batch_size"] = len(task_metric_pairs)
        metrics["local/task_playbook_size"] = float(mean_playbook_size)
        metrics["local/task_playbook_count"] = len(self.local_eval_playbooks) if is_validation else len(self.local_train_playbooks)
        self._add_task_flush_metrics(metrics, task_ids, is_eval=is_validation)

        # Accumulate per-task data for epoch table (training only).
        if not is_validation:
            epoch = self._estimate_epoch()
            if epoch > self._current_epoch:
                self._flush_epoch_table()
                self._current_epoch = epoch

            for task_id, task_metrics in task_metric_pairs:
                playbook = self.local_train_playbooks.get(task_id, self.playbook)
                pb_stats = playbook.get_stats()
                self._epoch_task_data.append(
                    {
                        "epoch": epoch,
                        "task_id": task_id,
                        "reward": task_metrics.get("ace/current_reward", 0.0),
                        "selected_reward": task_metrics.get("ace/selected_reward", 0.0),
                        "playbook_size": len(playbook.to_string()),
                        "playbook_bullets": pb_stats.get("total_bullets", 0),
                    }
                )

        return output, metrics

    def _validate_agent(self, logger=None, step=0):
        from rllm.algorithms.ace.checkpoint import preserve_ace_training_state

        # Validation on resume must not perturb the next training decision.
        with preserve_ace_training_state(self):
            return self._validate_agent_impl(logger=logger, step=step)

    def _validate_agent_impl(self, logger=None, step=0):
        self.local_eval_playbooks = {}
        self._in_local_eval = True
        self._clear_validation_policy_trajectory_cache()
        schedule_state = self._capture_component_schedule_state()
        total_local_eval_epochs = max(1, self.local_eval_epochs)
        combined_turn_rows = []
        metrics = {}

        try:
            with self._phase_scope(
                "local_validation",
                step=step,
                local_eval_epochs=total_local_eval_epochs,
            ):
                for local_epoch in range(1, total_local_eval_epochs + 1):
                    self._current_local_eval_epoch = local_epoch
                    with self._phase_scope(
                        "local_validation_epoch",
                        step=step,
                        local_epoch=local_epoch,
                        local_eval_epochs=total_local_eval_epochs,
                    ):
                        metrics, epoch_ace_metrics, turn_rows, _ = self._run_local_validation_epoch()
                    table_metric_keys = ["pass_at_k_max_reward"] + sorted(key for key in epoch_ace_metrics if key.startswith("ace/") or key.startswith("playbook/"))
                    for row in turn_rows:
                        local_turn = int(row["turn"])
                        enriched = dict(row)
                        enriched["turn"] = ((local_epoch - 1) * len(turn_rows)) + local_turn
                        enriched["local_turn"] = local_turn
                        enriched["local_eval_epoch"] = local_epoch
                        enriched["local_eval_epoch_total"] = total_local_eval_epochs
                        for key in table_metric_keys:
                            if key in epoch_ace_metrics:
                                enriched[key] = epoch_ace_metrics[key]
                        combined_turn_rows.append(enriched)

                metrics["val/local_ace_epoch"] = total_local_eval_epochs
                metrics["val/local_ace_eval_epochs"] = total_local_eval_epochs
                self._log_validation_turn_table(logger=logger, step=step, rows=combined_turn_rows)
                return metrics
        finally:
            if schedule_state is not None:
                self._restore_component_schedule_state(schedule_state)
            self._current_validation_policy_tasks = None
            self._current_validation_policy_sources = None
            self._clear_validation_policy_trajectory_cache()
            self._current_local_eval_epoch = None
            self._in_local_eval = False

    def fit_agent(self):
        try:
            super().fit_agent()
        finally:
            # Flush last epoch's table.
            if self._epoch_task_data:
                self._flush_epoch_table()
