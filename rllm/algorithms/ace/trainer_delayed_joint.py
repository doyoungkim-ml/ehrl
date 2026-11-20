"""Local delayed-joint ACE+RL trainer."""

import copy
import logging
import random

import numpy as np

from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.trainer import LocalACERLAgentPPOTrainer
from rllm.algorithms.ace.trainer_delayed_joint_base import (
    ACERLDelayedJointAgentPPOTrainer,
)
from rllm.utils.training_reward import raw_reward_metrics

logger = logging.getLogger(__name__)


class LocalACERLDelayedJointAgentPPOTrainer(
    LocalACERLAgentPPOTrainer,
    ACERLDelayedJointAgentPPOTrainer,
):
    """Local ACE+RL with per-task playbooks and delayed-joint rollout logic."""

    def _generate_multi_task_batched_delayed_joint(
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
            with self._keep_rollout_engine_awake(reason=(f"local batched delayed-joint {'validation' if is_validation else 'training'} step with {len(task_ids)} task(s)")):
                self._suppress_inner_flush = True

                with marked_timer("ace_act", timing_raw):
                    self._set_component_token_limits("policy")
                    self._inject_task_playbooks_into_agents(
                        task_to_indices,
                        is_eval=is_validation,
                    )
                    seed_policy_trajectories = self._collect_token_rollouts(
                        timing_raw=timing_raw,
                        meta_info=meta_info,
                    )

                actor_reward_metrics = raw_reward_metrics(seed_policy_trajectories, expected_count=len(original_envs))
                if not seed_policy_trajectories:
                    raise RuntimeError("[LOCAL-ACE-DELAYED-JOINT] No seed policy trajectories collected; joint update cannot proceed.")

                grouped_seed_policy = self._group_trajectories_by_task(
                    seed_policy_trajectories,
                    task_to_indices,
                )
                current_rewards_by_task = {}
                for task_id in task_ids:
                    task_rewards = [float(traj.get("trajectory_reward", 0.0)) for traj in grouped_seed_policy.get(task_id, [])]
                    current_rewards_by_task[task_id] = float(np.mean(task_rewards)) if task_rewards else 0.0
                current_rewards = [float(traj.get("trajectory_reward", 0.0)) for traj in seed_policy_trajectories]
                current_avg_reward = float(np.mean(current_rewards)) if current_rewards else 0.0

                self._original_agents = original_agents
                self._original_envs = original_envs

                reflector_prompts = []
                for task_idx, task_id in enumerate(task_ids):
                    task_playbook_text = playbook_before_text_by_task[task_id]
                    for traj_idx, traj in enumerate(grouped_seed_policy.get(task_id, [])):
                        reflector_prompts.append(
                            {
                                "task_id": task_id,
                                "task_idx": task_idx,
                                "traj_idx": traj_idx,
                                "env_task_id": f"ref::{task_id}::traj{traj_idx}",
                                "task_uid": task_id,
                                "prompt_text": self._build_reflector_prompt_from_dict(
                                    traj,
                                    task_playbook_text,
                                    task_idx,
                                ),
                            }
                        )

                reflector_trajectories = []
                reflector_records = []
                if reflector_prompts:
                    with marked_timer("ace_reflect", timing_raw):
                        self._set_component_token_limits("reflector")
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
                            reflector_trajectories = self._collect_token_rollouts(
                                timing_raw=timing_raw,
                                meta_info=meta_info,
                            )
                        finally:
                            self.agent_execution_engine.update_envs_and_agents(
                                self._original_envs,
                                self._original_agents,
                            )

                    for i, traj in enumerate(reflector_trajectories):
                        if i >= len(reflector_prompts):
                            break
                        prompt = reflector_prompts[i]
                        response_text = self._decode_trajectory_response(traj)
                        reflector_records.append(
                            {
                                "task_id": prompt["task_id"],
                                "task_idx": prompt["task_idx"],
                                "task_uid": prompt["task_uid"],
                                "traj_idx": prompt["traj_idx"],
                                "task_id_env": prompt["env_task_id"],
                                "prompt_text": prompt["prompt_text"],
                                "response_text": response_text,
                                "reflection": self._coerce_reflection_payload_for_delayed_joint(response_text),
                            }
                        )

                curator_prompts = []
                for reflector_idx, record in enumerate(reflector_records):
                    insight_text = self._build_delayed_joint_insight_text(record)
                    if not insight_text.strip():
                        continue
                    task_id = record["task_id"]
                    curator_prompts.append(
                        {
                            "task_id": task_id,
                            "task_idx": record["task_idx"],
                            "task_uid": record["task_uid"],
                            "traj_idx": record["traj_idx"],
                            "reflector_idx": reflector_idx,
                            "env_task_id": f"cur::{task_id}::traj{record['traj_idx']}",
                            "guidebook_uid": f"{task_id}::traj{record['traj_idx']}",
                            "prompt_text": self._build_curator_prompt(
                                insight_text,
                                playbook_before_text_by_task[task_id],
                                task_instr=self._get_policy_task_instruction(
                                    grouped_seed_policy[task_id][record["traj_idx"]], record["task_idx"]
                                ),
                            ),
                        }
                    )

                curator_trajectories = []
                curator_records = []
                candidate_playbooks = []
                if curator_prompts:
                    with marked_timer("ace_curate", timing_raw):
                        self._set_component_token_limits("curator")
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
                            curator_trajectories = self._collect_token_rollouts(
                                timing_raw=timing_raw,
                                meta_info=meta_info,
                            )
                        finally:
                            self.agent_execution_engine.update_envs_and_agents(
                                self._original_envs,
                                self._original_agents,
                            )

                    for i, traj in enumerate(curator_trajectories):
                        if i >= len(curator_prompts):
                            break
                        prompt = curator_prompts[i]
                        response_text = self._decode_trajectory_response(traj)
                        ops = self._parse_curator_operations_for_delayed_joint(response_text)
                        task_id = prompt["task_id"]
                        record = {
                            "task_id": task_id,
                            "task_idx": prompt["task_idx"],
                            "task_uid": prompt["task_uid"],
                            "traj_idx": prompt["traj_idx"],
                            "reflector_idx": prompt["reflector_idx"],
                            "task_id_env": prompt["env_task_id"],
                            "guidebook_uid": prompt["guidebook_uid"],
                            "prompt_text": prompt["prompt_text"],
                            "response_text": response_text,
                            "parsed_operations": ops,
                            "accepted": False,
                        }

                        if ops.get("operations"):
                            candidate = Playbook(playbook_before_text_by_task[task_id])
                            try:
                                candidate.apply_operations(ops)
                                candidate_text = candidate.to_string()
                                if self.max_playbook_length is not None:
                                    candidate_len = self._get_candidate_playbook_length(candidate_text)
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
                                        "reflector_idx": prompt["reflector_idx"],
                                        "task_id": task_id,
                                        "task_uid": prompt["task_uid"],
                                        "traj_idx": prompt["traj_idx"],
                                        "guidebook_uid": prompt["guidebook_uid"],
                                    }
                                )
                                record["accepted"] = True
                                record["candidate_index"] = len(candidate_playbooks) - 1
                                record["candidate_playbook"] = candidate_text
                            except Exception as exc:  # noqa: BLE001
                                record["error"] = str(exc)
                                logger.warning("[LOCAL-ACE-DELAYED-JOINT][CURATE] apply_operations failed: %s", exc)
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
                task_playbook_sizes = []
                task_playbook_bullets = []
                grouped_policy_for_training_by_task = {}
                policy_train_source_by_task = {}
                store = self.local_eval_playbooks if is_validation else self.local_train_playbooks

                for task_id in task_ids:
                    task_seed_policy = grouped_seed_policy.get(task_id, [])
                    task_current_reward = current_rewards_by_task.get(task_id, 0.0)
                    task_candidate_indices = [idx for idx, candidate in enumerate(candidate_playbooks) if candidate.get("task_id") == task_id]
                    selected_idx = -1
                    selected_reward = task_current_reward
                    best_candidate_reward = task_current_reward
                    reject_candidate = False
                    updated_playbook_text = playbook_before_text_by_task[task_id]
                    policy_trajectories_for_training = [copy.deepcopy(traj) for traj in task_seed_policy]
                    policy_train_source_by_task[task_id] = "seed"

                    task_candidate_indices = [idx for idx in task_candidate_indices if idx < len(candidate_rewards)]
                    if task_candidate_indices:
                        task_candidate_rewards = [float(candidate_rewards[idx]) for idx in task_candidate_indices]
                        best_candidate_reward = float(np.max(task_candidate_rewards))
                        surviving_candidate_indices = task_candidate_indices
                        if self.selection_mode == "best_rejection":
                            reject_candidate = best_candidate_reward < task_current_reward
                        elif self.selection_mode == "best_rejection_update":
                            surviving_candidate_indices = self._get_best_rejection_update_candidate_indices(
                                candidate_playbooks,
                                candidate_rewards,
                                grouped_seed_policy,
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
                            selected_policy = candidate_eval_trajectories.get(selected_idx, [])
                            (
                                policy_trajectories_for_training,
                                policy_train_source_by_task[task_id],
                            ) = self._select_policy_trajectories_for_training(
                                task_seed_policy,
                                selected_policy,
                                expected_policy_count=len(task_seed_policy),
                                log_prefix=f"[LOCAL-ACE-DELAYED-JOINT][SELECT][task={task_id}]",
                            )

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

                reflector_idx_to_reward = {}
                curator_idx_to_reward = {}
                for candidate_idx, candidate in enumerate(candidate_playbooks):
                    task_id = candidate["task_id"]
                    if reject_candidate_by_task.get(task_id) and self.selection_mode != "best_rejection_update":
                        continue
                    if candidate_idx >= len(candidate_rewards):
                        continue
                    reward = float(candidate_rewards[candidate_idx])
                    reflector_idx_to_reward[int(candidate["reflector_idx"])] = reward
                    curator_idx_to_reward[int(candidate["curator_idx"])] = reward

                for i, traj in enumerate(reflector_trajectories):
                    task_id = reflector_records[i]["task_id"] if i < len(reflector_records) else None
                    if task_id and reject_candidate_by_task.get(task_id) and self.selection_mode != "best_rejection_update":
                        traj["trajectory_reward"] = 0.0
                    else:
                        traj["trajectory_reward"] = reflector_idx_to_reward.get(i, 0.0)
                for i, traj in enumerate(curator_trajectories):
                    task_id = curator_records[i]["task_id"] if i < len(curator_records) else None
                    if task_id and reject_candidate_by_task.get(task_id) and self.selection_mode != "best_rejection_update":
                        traj["trajectory_reward"] = 0.0
                    else:
                        traj["trajectory_reward"] = curator_idx_to_reward.get(i, 0.0)

                for traj in policy_trajectories_for_training:
                    traj["_component_type"] = "policy"
                for traj in reflector_trajectories:
                    traj["_component_type"] = "reflector"
                for traj in curator_trajectories:
                    traj["_component_type"] = "curator"

                all_trajectories = policy_trajectories_for_training + reflector_trajectories + curator_trajectories
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

                self._expand_current_batch_for_component_rows(
                    reflector_records=reflector_records,
                    curator_records=curator_records,
                    extra_policy_count=max(
                        0,
                        len(policy_trajectories_for_training) - len(seed_policy_trajectories),
                    ),
                )

                policy_chat_paths = self._save_task_policy_chat_logs(
                    grouped_policy_for_training_by_task,
                    meta_info=meta_info,
                )

                metrics = dict(transform_metrics)
                if not is_validation:
                    metrics.update(actor_reward_metrics)
                metrics.update(
                    self._compute_rollout_token_totals(
                        policy_trajectories=seed_policy_trajectories,
                        reflector_trajectories=reflector_trajectories,
                        curator_trajectories=curator_trajectories,
                        candidate_eval_token_metrics=candidate_eval_token_metrics,
                        ppo_trajectories=all_trajectories,
                    )
                )
                metrics["ace/policy_rollouts"] = len(policy_trajectories_for_training)
                metrics["ace/reflector_rollouts"] = len(reflector_trajectories)
                metrics["ace/curator_rollouts"] = len(curator_trajectories)
                metrics["ace/candidate_eval_rollouts"] = len(candidate_eval_token_metrics)
                metrics["ace/total_rollouts"] = len(all_trajectories)
                metrics["ace/candidates"] = len(candidate_playbooks)
                metrics["ace/guidebooks"] = len(reflector_records)
                metrics["ace/current_reward"] = float(np.mean(list(current_rewards_by_task.values()))) if current_rewards_by_task else current_avg_reward
                metrics["ace/selected_reward"] = float(np.mean(list(selected_reward_by_task.values()))) if selected_reward_by_task else current_avg_reward
                metrics["ace/best_candidate_reward"] = float(np.mean(list(best_candidate_reward_by_task.values()))) if best_candidate_reward_by_task else current_avg_reward
                metrics["ace/reject_candidate"] = float(np.mean([int(v) for v in reject_candidate_by_task.values()])) if reject_candidate_by_task else 0.0
                metrics["ace/selected_candidate_idx"] = float(np.mean(list(selected_idx_by_task.values()))) if selected_idx_by_task else -1.0
                metrics["ace/active_component"] = hash("all") % 1000
                policy_source_ids = [self._policy_train_source_metric_id(policy_train_source_by_task.get(task_id, "seed")) for task_id in task_ids]
                metrics["ace/policy_train_source_id"] = float(np.mean(policy_source_ids)) if policy_source_ids else -1.0
                metrics["ace/validate_policy_only"] = 0
                metrics["playbook/size"] = float(np.mean(task_playbook_sizes)) if task_playbook_sizes else 0.0
                metrics["playbook/bullets"] = float(np.mean(task_playbook_bullets)) if task_playbook_bullets else 0.0
                metrics["local/task_id_hash"] = hash(task_ids[0]) % 1000000
                metrics["local/task_batch_size"] = len(task_ids)
                metrics["local/task_playbook_size"] = float(np.mean(task_playbook_sizes)) if task_playbook_sizes else 0.0
                metrics["local/task_playbook_count"] = len(self.local_eval_playbooks) if is_validation else len(self.local_train_playbooks)
                self._add_task_flush_metrics(metrics, task_ids, is_eval=is_validation)

                task_payloads = {}
                for task_id in task_ids:
                    task_payloads[task_id] = {
                        "step": int(self.global_steps),
                        "mode": self._artifact_mode(meta_info),
                        "training_mode": self.training_mode,
                        "chat_completions_path": policy_chat_paths.get(task_id),
                        "artifacts": {
                            "policy_chat_completions": policy_chat_paths.get(task_id),
                            "playbook_snapshot": snapshot_paths.get(task_id),
                        },
                        "metrics": metrics,
                        "playbook": {
                            "before": playbook_before_text_by_task.get(task_id, ""),
                            "after": playbook_after_text_by_task.get(task_id, ""),
                        },
                        "playbook_flush": self._task_flush_payload(
                            task_id,
                            is_eval=is_validation,
                        ),
                        "reflector": [record for record in reflector_records if record.get("task_id") == task_id],
                        "curator": [record for record in curator_records if record.get("task_id") == task_id],
                        "candidate_evaluations": [record for record in candidate_eval_records if record.get("task_id") == task_id],
                        "selection": {
                            "selected_candidate_idx": selected_idx_by_task.get(task_id, -1),
                            "selected_reward": selected_reward_by_task.get(task_id, current_rewards_by_task.get(task_id, 0.0)),
                            "best_candidate_reward": best_candidate_reward_by_task.get(task_id, current_rewards_by_task.get(task_id, 0.0)),
                            "reject_candidate": bool(reject_candidate_by_task.get(task_id, False)),
                            "selection_mode": self.selection_mode,
                            "policy_train_source": policy_train_source_by_task.get(task_id, "seed"),
                        },
                    }
                self._save_task_ace_logs(task_payloads, meta_info=meta_info)

                if not is_validation:
                    epoch = self._estimate_epoch()
                    if epoch > self._current_epoch:
                        self._flush_epoch_table()
                        self._current_epoch = epoch

                    for task_id in task_ids:
                        playbook = self.local_train_playbooks.get(task_id)
                        if playbook is None:
                            continue
                        pb_stats = playbook.get_stats()
                        self._epoch_task_data.append(
                            {
                                "epoch": epoch,
                                "task_id": task_id,
                                "reward": current_rewards_by_task.get(task_id, 0.0),
                                "selected_reward": selected_reward_by_task.get(task_id, current_rewards_by_task.get(task_id, 0.0)),
                                "playbook_size": len(playbook.to_string()),
                                "playbook_bullets": pb_stats.get("total_bullets", 0),
                            }
                        )

                if schedule_state is not None:
                    self._restore_component_schedule_state(schedule_state)
                if task_ids:
                    self.playbook = Playbook(playbook_after_text_by_task.get(task_ids[0], self.initial_playbook_text))
                return final_gen_batch_output, metrics
        finally:
            self.agent_execution_engine.update_envs_and_agents(original_envs, original_agents)
            self._artifact_task_id = prev_task_id
            self._artifact_task_ids = prev_task_ids
            self._suppress_inner_flush = prev_suppress_flush

    def _coerce_reflection_payload_for_delayed_joint(self, response_text: str):
        from rllm.algorithms.ace.trainer import (
            _coerce_reflection_payload,
            _parse_json_response,
        )

        return _coerce_reflection_payload(
            _parse_json_response(response_text),
            response_text=response_text,
            context="reflector.delayed_joint.local",
        )

    def _build_delayed_joint_insight_text(self, record: dict) -> str:
        from rllm.algorithms.ace.trainer import _concatenate_insights

        return _concatenate_insights(
            [
                {
                    "task_idx": record["task_idx"],
                    "reflection": record.get("reflection", {}),
                }
            ]
        )

    def _parse_curator_operations_for_delayed_joint(self, response_text: str) -> dict:
        from rllm.algorithms.ace.trainer import _parse_json_response

        ops = _parse_json_response(response_text)
        if isinstance(ops, list):
            return {"operations": ops}
        if isinstance(ops, dict):
            return ops
        return {}

    def _get_candidate_playbook_length(self, playbook_text: str) -> int:
        from rllm.algorithms.ace.trainer import _get_playbook_token_length

        return _get_playbook_token_length(playbook_text, self.tokenizer)

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        is_validation = bool(meta_info and meta_info.get("validate", False))

        if is_validation and not self._in_local_eval:
            return self._generate_policy_only_trajectory(timing_raw=timing_raw, meta_info=meta_info)

        n = self.config.actor_rollout_ref.rollout.n
        envs = getattr(self.agent_execution_engine, "envs", None) or []
        env_count = len(envs)
        if env_count > n and (env_count % n) != 0:
            logger.warning(
                "Local delayed-joint ACE-RL expects env_count to be a multiple of rollout.n. Got env_count=%s, rollout.n=%s; behavior may deviate.",
                env_count,
                n,
            )

        task_ids, task_to_indices = self._build_task_index_groups()
        if not task_ids:
            return ACERLDelayedJointAgentPPOTrainer.generate_agent_trajectory(self, timing_raw=timing_raw, meta_info=meta_info)

        if len(task_ids) > 1:
            batched_result = self._generate_multi_task_batched_delayed_joint(
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
            return ACERLDelayedJointAgentPPOTrainer.generate_agent_trajectory(self, timing_raw=timing_raw, meta_info=meta_info)

        from verl import DataProto

        schedule_state = self._capture_component_schedule_state()
        prev_task_id = self._artifact_task_id
        prev_task_ids = list(self._artifact_task_ids)
        prev_current_batch = self._current_batch
        prev_suppress_flush = self._suppress_inner_flush

        task_outputs = []
        expanded_sub_batches = []
        task_metric_pairs = []
        combined_step_rewards = []
        combined_step_reward_tasks = []
        playbook_size_sum = 0
        validation_policy_tasks = []
        validation_policy_sources = []

        with self._keep_rollout_engine_awake(reason=(f"local delayed-joint {'validation' if is_validation else 'training'} step with {len(task_ids)} task(s)")):
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

                    task_output, task_metrics = ACERLDelayedJointAgentPPOTrainer.generate_agent_trajectory(self, timing_raw=timing_raw, meta_info=meta_info)
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
            self._restore_component_schedule_state(schedule_state)

        if not task_outputs:
            return ACERLDelayedJointAgentPPOTrainer.generate_agent_trajectory(self, timing_raw=timing_raw, meta_info=meta_info)

        output = task_outputs[0] if len(task_outputs) == 1 else DataProto.concat(task_outputs)

        if parent_batch is not None and expanded_sub_batches:
            expanded_batch = expanded_sub_batches[0] if len(expanded_sub_batches) == 1 else DataProto.concat(expanded_sub_batches)
            parent_batch.non_tensor_batch = expanded_batch.non_tensor_batch
            parent_batch.batch = expanded_batch.batch

        metrics = self._aggregate_task_metrics([m for _, m in task_metric_pairs])
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
