"""Delayed-joint ACE+RL trainer with rollout-aligned component sampling.

Flow per training step:
1. Sample multiple policy rollouts with the current playbook.
2. For each policy rollout, sample exactly one reflector insight.
3. For each insight, sample exactly one curator response to produce one
   candidate playbook.
4. Evaluate every candidate playbook on the current batch.
5. Update policy on the selected candidate's policy rollouts and assign the
   delayed candidate reward back to the aligned reflector/curator rows.

This removes the separate reflector/curator harness sampling dimensions
(`g_ref`, `g_cur`) from candidate construction while keeping the same playbook
selection logic as the standard trainer.
"""

import copy
import logging
import random

import numpy as np

from rllm.algorithms.ace.agents import (
    SingleTurnCuratorAgent,
    SingleTurnReflectorAgent,
)
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.environments import (
    SingleTurnCuratorEnv,
    SingleTurnReflectorEnv,
)
from rllm.algorithms.ace.trainer_base import (
    ACERLAgentPPOTrainer,
    _coerce_reflection_payload,
    _concatenate_insights,
    _get_playbook_token_length,
    _parse_json_response,
)
from rllm.utils.training_reward import raw_reward_metrics

logger = logging.getLogger(__name__)


class ACERLDelayedJointAgentPPOTrainer(ACERLAgentPPOTrainer):
    """Joint ACE+RL trainer with delayed policy updates and rollout-aligned ACE."""

    def _init_ace_components(self):
        super()._init_ace_components()
        if self.policy_train_source == "seed_and_candidates":
            raise ValueError("ace.policy_train_source=seed_and_candidates is not supported by delayed-joint ACE-RL")
        self.training_mode = "delayed_joint"
        self.auxiliary_update_gate = "none"
        logger.info(
            "[ACE-DELAYED-JOINT] rollout_aligned=True selection=%s configured_g_ref=%s configured_g_cur=%s",
            self.selection_mode,
            self.g_ref,
            self.g_cur,
        )

    def _collect_token_rollouts(self, timing_raw=None, meta_info=None) -> list[dict]:
        trajectories = []
        gen = self.generate_agent_trajectories_async(
            timing_raw=timing_raw,
            meta_info=meta_info,
            mode="Token",
        )
        for traj in gen:
            trajectories.append(traj)
        trajectories.sort(key=lambda x: x["idx"])
        return trajectories

    def _group_policy_rollouts(self, policy_trajectories: list[dict]) -> tuple[int, list[list[dict]]]:
        rollout_n = int(self.config.actor_rollout_ref.rollout.n or 1)
        bsz = len(policy_trajectories) // rollout_n if rollout_n > 0 else len(policy_trajectories)
        if bsz <= 0:
            bsz = len(policy_trajectories)

        task_groups = [[] for _ in range(bsz)]
        for i, traj in enumerate(policy_trajectories):
            task_idx = i % bsz if bsz > 0 else i
            task_groups[task_idx].append(traj)
        return bsz, task_groups

    def _get_task_base_uids(self, bsz: int) -> list[str]:
        default_uids = [f"task_{i}" for i in range(bsz)]
        if bsz <= 0 or self._current_batch is None:
            return default_uids

        uid_values = self._current_batch.non_tensor_batch.get("uid")
        if uid_values is None or len(uid_values) == 0:
            return default_uids

        base_uids = list(default_uids)
        for task_idx in range(min(bsz, len(uid_values))):
            base_uids[task_idx] = str(uid_values[task_idx])
        return base_uids

    def _collect_policy_rollouts(self, playbook_text: str, timing_raw=None, meta_info=None) -> list[dict]:
        from verl.utils.debug import marked_timer

        with marked_timer("ace_act", timing_raw):
            self._set_component_token_limits("policy")
            self._inject_playbook_into_agents(
                self.agent_execution_engine.agents,
                playbook_text,
            )
            return self._collect_token_rollouts(timing_raw=timing_raw, meta_info=meta_info)

    def _collect_rollout_aligned_reflector_rollouts(
        self,
        task_groups: list[list[dict]],
        task_base_uids: list[str],
        playbook_text: str,
        timing_raw=None,
        meta_info=None,
    ) -> tuple[list[dict], list[dict]]:
        from verl.utils.debug import marked_timer

        reflector_prompts = []
        for task_idx, group in enumerate(task_groups):
            task_uid = task_base_uids[task_idx] if task_idx < len(task_base_uids) else f"task_{task_idx}"
            for traj_idx, traj in enumerate(group):
                reflector_prompts.append(
                    {
                        "task_idx": task_idx,
                        "task_uid": task_uid,
                        "traj_idx": traj_idx,
                        "task_id": f"ref::{task_uid}::traj{traj_idx}",
                        "task_instr": self._get_policy_task_instruction(traj, task_idx),
                        "prompt_text": self._build_reflector_prompt_from_dict(traj, playbook_text, task_idx),
                    }
                )

        reflector_trajectories = []
        reflector_records = []
        if not reflector_prompts:
            return reflector_trajectories, reflector_records

        with marked_timer("ace_reflect", timing_raw):
            self._set_component_token_limits("reflector")

            ref_agents = [SingleTurnReflectorAgent(system_prompt=self.reflector_system_prompt) for _ in reflector_prompts]
            ref_envs = [
                SingleTurnReflectorEnv(
                    prompt_text=prompt["prompt_text"],
                    task_id=prompt["task_id"],
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
                    "task_idx": prompt["task_idx"],
                    "task_uid": prompt["task_uid"],
                    "traj_idx": prompt["traj_idx"],
                    "task_id": prompt["task_id"],
                    "task_instr": prompt["task_instr"],
                    "prompt_text": prompt["prompt_text"],
                    "response_text": response_text,
                    "reflection": _coerce_reflection_payload(
                        _parse_json_response(response_text),
                        response_text=response_text,
                        context="reflector.delayed_joint",
                    ),
                }
            )

        return reflector_trajectories, reflector_records

    def _collect_rollout_aligned_curator_rollouts(
        self,
        reflector_records: list[dict],
        playbook_text: str,
        timing_raw=None,
        meta_info=None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        from verl.utils.debug import marked_timer

        curator_prompts = []
        for reflector_idx, record in enumerate(reflector_records):
            insight_text = _concatenate_insights(
                [
                    {
                        "task_idx": record["task_idx"],
                        "reflection": record.get("reflection", {}),
                    }
                ]
            )
            if not insight_text.strip():
                continue

            curator_prompts.append(
                {
                    "task_idx": record["task_idx"],
                    "task_uid": record["task_uid"],
                    "traj_idx": record["traj_idx"],
                    "reflector_idx": reflector_idx,
                    "task_id": f"cur::{record['task_uid']}::traj{record['traj_idx']}",
                    "guidebook_uid": f"{record['task_uid']}::traj{record['traj_idx']}",
                    "prompt_text": self._build_curator_prompt(insight_text, playbook_text, task_instr=record["task_instr"]),
                }
            )

        curator_trajectories = []
        curator_records = []
        candidate_playbooks = []
        if not curator_prompts:
            return curator_trajectories, curator_records, candidate_playbooks

        with marked_timer("ace_curate", timing_raw):
            self._set_component_token_limits("curator")

            cur_agents = [SingleTurnCuratorAgent(system_prompt=self.curator_system_prompt) for _ in curator_prompts]
            cur_envs = [
                SingleTurnCuratorEnv(
                    prompt_text=prompt["prompt_text"],
                    task_id=prompt["task_id"],
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
            ops = _parse_json_response(response_text)
            if isinstance(ops, list):
                ops_payload = {"operations": ops}
            elif isinstance(ops, dict):
                ops_payload = ops
            else:
                ops_payload = {}

            record = {
                "task_idx": prompt["task_idx"],
                "task_uid": prompt["task_uid"],
                "traj_idx": prompt["traj_idx"],
                "reflector_idx": prompt["reflector_idx"],
                "task_id": prompt["task_id"],
                "guidebook_uid": prompt["guidebook_uid"],
                "prompt_text": prompt["prompt_text"],
                "response_text": response_text,
                "parsed_operations": ops,
                "accepted": False,
            }

            if ops_payload.get("operations"):
                candidate = Playbook(playbook_text)
                try:
                    candidate.apply_operations(ops_payload)
                    candidate_text = candidate.to_string()
                    if self.max_playbook_length is not None:
                        candidate_len = _get_playbook_token_length(candidate_text, self.tokenizer)
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
                            "task_idx": prompt["task_idx"],
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
                    logger.warning("[ACE-DELAYED-JOINT][CURATE] apply_operations failed: %s", exc)

            curator_records.append(record)

        return curator_trajectories, curator_records, candidate_playbooks

    def _evaluate_candidate_playbooks_with_trajectories(
        self,
        candidate_playbooks: list[dict],
        timing_raw=None,
        meta_info=None,
    ) -> tuple[list[float], list[dict], list[list[dict]]]:
        rewards, records, trajectories, _ = self._evaluate_candidate_pool(
            candidate_playbooks,
            self._build_candidate_eval_envs_and_agents,
            capture_eval_trajectories=True,
            timing_raw=timing_raw,
            meta_info=meta_info,
        )
        for record in records:
            candidate = candidate_playbooks[record["candidate_idx"]]
            record.update({key: candidate[key] for key in ("task_idx", "task_uid", "traj_idx")})
        return rewards, records, [trajectories.get(idx, []) for idx in range(len(candidate_playbooks))]

    def _assign_rollout_aligned_aux_rewards(
        self,
        candidate_playbooks: list[dict],
        candidate_rewards: list[float],
        reflector_trajectories: list[dict],
        curator_trajectories: list[dict],
    ):
        reflector_idx_to_reward = {}
        curator_idx_to_reward = {}
        for candidate_idx, candidate in enumerate(candidate_playbooks):
            if candidate_idx >= len(candidate_rewards):
                continue
            reward = float(candidate_rewards[candidate_idx])
            reflector_idx_to_reward[int(candidate["reflector_idx"])] = reward
            curator_idx_to_reward[int(candidate["curator_idx"])] = reward

        for i, traj in enumerate(reflector_trajectories):
            traj["trajectory_reward"] = reflector_idx_to_reward.get(i, 0.0)
        for i, traj in enumerate(curator_trajectories):
            traj["trajectory_reward"] = curator_idx_to_reward.get(i, 0.0)

    def _expand_current_batch_for_component_rows(
        self,
        reflector_records: list[dict],
        curator_records: list[dict],
        extra_policy_count: int = 0,
    ):
        reflector_uids = [f"ref::{record['task_uid']}" for record in reflector_records]
        curator_uids = [f"cur::{record['task_uid']}" for record in curator_records]
        self._append_policy_and_component_rows_to_batch(
            self._current_batch,
            extra_policy_count=extra_policy_count,
            component_extra_uids=reflector_uids + curator_uids,
        )

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        from verl.utils.debug import marked_timer

        if timing_raw is None:
            timing_raw = {}

        is_validation = bool(meta_info and meta_info.get("validate", False))
        self._last_candidate_eval_trajectories = None

        if is_validation and self._should_use_policy_only_validation(meta_info):
            self._clear_validation_policy_trajectory_cache()
            return self._generate_policy_only_trajectory(
                timing_raw=timing_raw,
                meta_info=meta_info,
            )

        self._maybe_flush_playbook()

        playbook_before_text = self._get_current_playbook_text()
        seed_policy_trajectories = self._collect_policy_rollouts(
            playbook_text=playbook_before_text,
            timing_raw=timing_raw,
            meta_info=meta_info,
        )
        actor_reward_metrics = raw_reward_metrics(seed_policy_trajectories, expected_count=len(self.agent_execution_engine.envs))
        if not seed_policy_trajectories:
            raise RuntimeError("[ACE-DELAYED-JOINT] No seed policy trajectories collected; joint update cannot proceed.")

        bsz, task_groups = self._group_policy_rollouts(seed_policy_trajectories)
        task_base_uids = self._get_task_base_uids(bsz)
        current_rewards = [float(t.get("trajectory_reward", 0.0)) for t in seed_policy_trajectories]
        current_avg_reward = float(np.mean(current_rewards)) if current_rewards else 0.0

        self._original_agents = self.agent_execution_engine.agents
        self._original_envs = self.agent_execution_engine.envs

        reflector_trajectories, reflector_records = self._collect_rollout_aligned_reflector_rollouts(
            task_groups=task_groups,
            task_base_uids=task_base_uids,
            playbook_text=playbook_before_text,
            timing_raw=timing_raw,
            meta_info=meta_info,
        )
        curator_trajectories, curator_records, candidate_playbooks = self._collect_rollout_aligned_curator_rollouts(
            reflector_records=reflector_records,
            playbook_text=playbook_before_text,
            timing_raw=timing_raw,
            meta_info=meta_info,
        )

        candidate_rewards = []
        candidate_eval_records = []
        candidate_policy_trajectories = []
        if candidate_playbooks:
            with marked_timer("ace_evaluate", timing_raw):
                self._set_component_token_limits("policy")
                (
                    candidate_rewards,
                    candidate_eval_records,
                    candidate_policy_trajectories,
                ) = self._evaluate_candidate_playbooks_with_trajectories(
                    candidate_playbooks=candidate_playbooks,
                    timing_raw=timing_raw,
                    meta_info=meta_info,
                )

        policy_trajectories_for_training = [copy.deepcopy(traj) for traj in seed_policy_trajectories]
        policy_train_source = "seed"
        selected_candidate_idx = -1
        selected_reward = current_avg_reward
        best_candidate_reward = current_avg_reward
        reject_candidate = False

        if candidate_playbooks and candidate_rewards:
            best_candidate_reward = float(np.max(candidate_rewards))
            candidate_indices = list(range(min(len(candidate_playbooks), len(candidate_rewards))))
            surviving_candidate_indices = candidate_indices
            if self.selection_mode == "best_rejection":
                reject_candidate = best_candidate_reward < current_avg_reward
            elif self.selection_mode == "best_rejection_update":
                surviving_candidate_indices = self._get_best_rejection_update_candidate_indices(
                    candidate_playbooks,
                    candidate_rewards,
                    task_groups,
                    candidate_indices,
                )
                reject_candidate = not surviving_candidate_indices
            if not reject_candidate:
                if self.selection_mode == "best_rejection_update":
                    selected_candidate_idx = self._select_best_rejection_update_candidate_index(candidate_rewards, surviving_candidate_indices)
                elif self.selection_mode in {"best", "best_rejection"}:
                    selected_candidate_idx = max(
                        surviving_candidate_indices,
                        key=lambda idx: float(candidate_rewards[idx]),
                    )
                else:
                    selected_candidate_idx = random.choice(candidate_indices)

                selected_candidate = candidate_playbooks[selected_candidate_idx]
                selected_reward = float(candidate_rewards[selected_candidate_idx])
                self.playbook = selected_candidate["playbook"]

                selected_policy_trajectories = candidate_policy_trajectories[selected_candidate_idx] if selected_candidate_idx < len(candidate_policy_trajectories) else []
                policy_trajectories_for_training, policy_train_source = self._select_policy_trajectories_for_training(
                    seed_policy_trajectories,
                    selected_policy_trajectories,
                    expected_policy_count=len(seed_policy_trajectories),
                    log_prefix="[ACE-DELAYED-JOINT][SELECT]",
                )

        snapshot_path = self._save_playbook_snapshot_artifact(meta_info)

        if self._original_agents is not None:
            self._inject_playbook_into_agents(
                self._original_agents,
                self._get_current_playbook_text(),
            )
            self.agent_execution_engine.update_envs_and_agents(
                self._original_envs,
                self._original_agents,
            )

        if reject_candidate and self.selection_mode != "best_rejection_update":
            for traj in reflector_trajectories:
                traj["trajectory_reward"] = 0.0
            for traj in curator_trajectories:
                traj["trajectory_reward"] = 0.0
        else:
            self._assign_rollout_aligned_aux_rewards(
                candidate_playbooks=candidate_playbooks,
                candidate_rewards=candidate_rewards,
                reflector_trajectories=reflector_trajectories,
                curator_trajectories=curator_trajectories,
            )

        for traj in policy_trajectories_for_training:
            traj["_component_type"] = "policy"
        for traj in reflector_trajectories:
            traj["_component_type"] = "reflector"
        for traj in curator_trajectories:
            traj["_component_type"] = "curator"

        all_trajectories = policy_trajectories_for_training + reflector_trajectories + curator_trajectories
        for i, traj in enumerate(all_trajectories):
            traj["idx"] = i

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

        self._expand_current_batch_for_component_rows(
            reflector_records=reflector_records,
            curator_records=curator_records,
            extra_policy_count=max(
                0,
                len(policy_trajectories_for_training) - len(seed_policy_trajectories),
            ),
        )

        policy_chat_log_path = self._save_policy_chat_completions(
            policy_trajectories_for_training,
            bsz=bsz,
            meta_info=meta_info,
        )

        metrics = {}
        metrics.update(transform_metrics)
        if not is_validation:
            metrics.update(actor_reward_metrics)
        metrics["ace/policy_rollouts"] = len(policy_trajectories_for_training)
        metrics["ace/reflector_rollouts"] = len(reflector_trajectories)
        metrics["ace/curator_rollouts"] = len(curator_trajectories)
        metrics["ace/total_rollouts"] = len(all_trajectories)
        metrics["ace/candidates"] = len(candidate_playbooks)
        metrics["ace/guidebooks"] = len(reflector_records)
        metrics["ace/current_reward"] = current_avg_reward
        metrics["ace/selected_reward"] = selected_reward
        metrics["ace/best_candidate_reward"] = best_candidate_reward
        metrics["ace/reject_candidate"] = int(reject_candidate)
        metrics["ace/selected_candidate_idx"] = selected_candidate_idx
        metrics["ace/active_component"] = hash("all") % 1000
        metrics["ace/validate_policy_only"] = 0
        playbook_after_text = self._get_current_playbook_text()
        metrics["playbook/size"] = len(playbook_after_text)
        metrics["playbook/bullets"] = self.playbook.get_stats().get("total_bullets", 0)

        self._save_ace_step_log(
            {
                "step": int(self.global_steps),
                "mode": self._artifact_mode(meta_info),
                "training_mode": self.training_mode,
                "chat_completions_path": policy_chat_log_path,
                "artifacts": {
                    "policy_chat_completions": policy_chat_log_path,
                    "playbook_snapshot": snapshot_path,
                },
                "metrics": metrics,
                "playbook": {
                    "before": playbook_before_text,
                    "after": playbook_after_text,
                },
                "reflector": reflector_records,
                "curator": curator_records,
                "candidate_evaluations": candidate_eval_records,
            },
            meta_info=meta_info,
        )

        return final_gen_batch_output, metrics
