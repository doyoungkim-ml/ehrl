"""ACE chains trained jointly with a learned critic and ordinary per-row GAE.

A training example is one initial actor attempt, one reflection, one curation,
and one downstream attempt. All four generated segments receive the downstream
verifier reward. Each task may sample multiple independent complete chains,
without candidate competition or reward-group normalization.
"""

import copy
import logging

import numpy as np
from omegaconf import OmegaConf

from rllm.algorithms.ace.agents import SingleTurnCuratorAgent, SingleTurnReflectorAgent
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.environments import SingleTurnCuratorEnv, SingleTurnReflectorEnv
from rllm.algorithms.ace.trainer import LocalACERLAgentPPOTrainer
from rllm.algorithms.ace.trainer_base import (
    _coerce_reflection_payload,
    _concatenate_insights,
    _get_playbook_token_length,
    _parse_json_response,
)
from rllm.utils.training_reward import raw_reward_metrics

logger = logging.getLogger(__name__)


class LocalACEJointPPOTrainer(LocalACERLAgentPPOTrainer):
    """Collect complete, unfiltered ACE chains and update all roles together."""

    def _init_ace_components(self):
        super()._init_ace_components()
        self.training_mode = "joint_ppo"
        self.auxiliary_update_gate = "none"
        if OmegaConf.select(self.config, "algorithm.adv_estimator") != "gae":
            raise ValueError("Joint PPO requires algorithm.adv_estimator=gae")
        if OmegaConf.select(self.config, "critic.enable") is False or getattr(self, "use_critic", None) is False:
            raise ValueError("Joint PPO requires an enabled learned critic")
        for path in ("rllm.mask_truncated_samples", "rllm.agent.overlong_filter"):
            if OmegaConf.select(self.config, path, default=False):
                raise ValueError(f"Joint PPO requires {path}=false; every generated segment is trained")
        if int(self.config.actor_rollout_ref.rollout.n) < 1:
            raise ValueError("Joint PPO requires rollout.n >= 1")
        settings = {
            "ace.g_ref": self.g_ref,
            "ace.g_cur": self.g_cur,
            "ace.candidate_eval_rollouts": self.candidate_eval_rollouts,
        }
        for name, value in settings.items():
            if int(value) != 1:
                raise ValueError(f"Joint PPO requires {name}=1, got {value!r}")
        if self.config.rllm.rejection_sample.enable:
            raise ValueError("Joint PPO requires rejection_sample.enable=false; every chain is trained")
        if self.config.rllm.stepwise_advantage.enable:
            raise ValueError("Joint PPO requires stepwise_advantage.enable=false and ordinary token-level GAE")

    def _collect_joint_phase(self, component, envs, agents, *, timing_raw, meta_info):
        """Preserve exact environment ownership even when async results arrive out of order."""
        from verl.utils.debug import marked_timer

        engine = self.agent_execution_engine
        previous_envs, previous_agents = engine.envs, engine.agents
        try:
            engine.update_envs_and_agents(envs, agents)
            self._set_component_token_limits(component)
            with marked_timer(f"ace_{component}", timing_raw):
                rows = list(self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token"))
            indices = [int(row["idx"]) for row in rows]
            if len(rows) != len(envs) or sorted(indices) != list(range(len(envs))):
                raise RuntimeError(f"Joint PPO {component} expected one trajectory for each of {len(envs)} environments; received indices {indices}")
            return sorted(rows, key=lambda row: int(row["idx"]))
        finally:
            engine.update_envs_and_agents(previous_envs, previous_agents)

    def _joint_updated_playbook(self, before_playbook, response_text):
        """Invalid/no-op curation leaves the chain intact and retains its original playbook."""
        payload = _parse_json_response(response_text)
        operations = {"operations": payload} if isinstance(payload, list) else payload
        record = {"parsed_operations": operations, "applied_operations": 0, "playbook_changed": False}
        original = copy.deepcopy(before_playbook)
        if not isinstance(operations, dict) or not isinstance(operations.get("operations"), list):
            record["update_status"] = "invalid_operations"
            return original, record
        updated = copy.deepcopy(before_playbook)
        try:
            applied = updated.apply_operations(operations)
            if not applied:
                record["update_status"] = "no_op"
                return original, record
            if self.max_playbook_length is not None:
                length = _get_playbook_token_length(updated.to_string(), self.tokenizer)
                record["candidate_length"] = length
                if length > self.max_playbook_length:
                    record["update_status"] = "max_playbook_length"
                    return original, record
        except Exception as exc:  # noqa: BLE001
            # apply_operations may have mutated its copy before raising.
            record.update(update_status="invalid_operations", error=str(exc))
            logger.warning("[JOINT-PPO] Curation failed; continuing with original playbook: %s", exc)
            return original, record
        record.update(applied_operations=applied, playbook_changed=updated.to_string() != before_playbook.to_string(), update_status="applied")
        return updated, record

    def _expand_joint_batch(self, trajectories):
        """Give every generated segment the metadata of its own task, plus chain provenance."""
        from tensordict import TensorDict

        parent = self._current_batch
        if parent is None:
            return
        source_indices = [row["_chain_task_index"] for row in trajectories]
        for key, original in list(parent.non_tensor_batch.items()):
            expanded = np.empty((len(trajectories),) + original.shape[1:], dtype=original.dtype)
            for index, source_index in enumerate(source_indices):
                expanded[index] = copy.deepcopy(original[source_index])
            parent.non_tensor_batch[key] = expanded
        parent.non_tensor_batch["uid"] = np.array([f"{row['_chain_id']}::{row['_chain_segment']}" for row in trajectories], dtype=object)
        parent.non_tensor_batch["chain_id"] = np.array([row["_chain_id"] for row in trajectories], dtype=object)
        parent.non_tensor_batch["chain_segment"] = np.array([row["_chain_segment"] for row in trajectories], dtype=object)
        parent.non_tensor_batch["component_type"] = np.array([row["_component_type"] for row in trajectories], dtype=object)
        parent.batch = TensorDict({}, batch_size=[len(trajectories)])

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        # Keep the established ACE evaluation visits/sample counts. The chain
        # contract below is the training collector, not a new evaluation protocol.
        if meta_info and meta_info.get("validate", False):
            return super().generate_agent_trajectory(timing_raw=timing_raw, meta_info=meta_info)

        engine = self.agent_execution_engine
        original_envs = list(engine.envs or [])
        original_agents = list(engine.agents or [])
        task_ids, task_to_indices = self._build_task_index_groups()
        group_size = int(self.config.actor_rollout_ref.rollout.n)
        if group_size < 1 or not task_ids or any(len(indices) != group_size for indices in task_to_indices.values()):
            raise ValueError("Joint PPO requires exactly rollout.n initial policy environments per task")
        chain_tasks = [self._task_id_from_env(env) for env in original_envs]
        chain_slots = {index: slot for indices in task_to_indices.values() for slot, index in enumerate(indices)}
        chain_count = len(chain_tasks)
        if len(original_envs) != len(original_agents):
            raise ValueError("Joint PPO requires one agent per policy environment")
        if self._current_batch is not None and len(self._current_batch) != len(original_envs):
            raise ValueError("Joint PPO parent batch must match the initial policy environments")
        timing_raw = {} if timing_raw is None else timing_raw
        old_artifact_id, old_artifact_ids = self._artifact_task_id, self._artifact_task_ids
        old_suppress = self._suppress_inner_flush
        old_original_envs = getattr(self, "_original_envs", None)
        old_original_agents = getattr(self, "_original_agents", None)
        self._artifact_task_id, self._artifact_task_ids = task_ids[0], list(task_ids)
        self._original_envs, self._original_agents = original_envs, original_agents
        self._last_candidate_eval_trajectories = None
        try:
            with self._keep_rollout_engine_awake(reason=f"joint PPO chain collection for {len(task_ids)} tasks x {group_size} chains"):
                self._suppress_inner_flush = True
                self._prepare_task_playbooks_for_step(task_ids, is_eval=False)
                before_playbooks = {task_id: copy.deepcopy(self._get_task_playbook(task_id, is_eval=False)) for task_id in task_ids}
                before = {task_id: playbook.to_string() for task_id, playbook in before_playbooks.items()}
                self._inject_task_playbooks_into_agents(task_to_indices, is_eval=False)
                seed = self._collect_joint_phase("policy", original_envs, original_agents, timing_raw=timing_raw, meta_info=meta_info)
                actor_reward_metrics = raw_reward_metrics(seed, expected_count=chain_count)
                seed_rewards = [float(row["trajectory_reward"]) for row in seed]

                ref_records = [
                    {
                        "task_id": task_id,
                        "task_idx": index,
                        "chain_slot": chain_slots[index],
                        "prompt_text": self._build_reflector_prompt_from_dict(seed[index], before[task_id], index),
                    }
                    for index, task_id in enumerate(chain_tasks)
                ]
                reflectors = self._collect_joint_phase(
                    "reflector",
                    [SingleTurnReflectorEnv(prompt_text=record["prompt_text"], task_id=f"ref::{record['task_id']}::{record['chain_slot']}") for record in ref_records],
                    [SingleTurnReflectorAgent(system_prompt=self.reflector_system_prompt) for _ in ref_records],
                    timing_raw=timing_raw,
                    meta_info=meta_info,
                )
                cur_records = []
                for record, trajectory in zip(ref_records, reflectors, strict=True):
                    response = self._decode_trajectory_response(trajectory)
                    record["response_text"] = response
                    record["reflection"] = _coerce_reflection_payload(_parse_json_response(response), response_text=response, context="joint_ppo.reflector")
                    insight = _concatenate_insights([record])
                    # Raw malformed reflection is still evidence available to the
                    # curator; an empty reflection must not skip a chain segment.
                    if not insight.strip():
                        insight = response.strip() or "The reflector returned no usable insight. Preserve the playbook unless an update is justified."
                    cur_records.append({
                        "task_id": record["task_id"], "task_idx": record["task_idx"], "chain_slot": record["chain_slot"],
                        "prompt_text": self._build_curator_prompt(
                            insight,
                            before[record["task_id"]],
                            task_instr=self._get_policy_task_instruction(seed[record["task_idx"]], record["task_idx"]),
                        ),
                    })
                curators = self._collect_joint_phase(
                    "curator",
                    [SingleTurnCuratorEnv(prompt_text=record["prompt_text"], task_id=f"cur::{record['task_id']}::{record['chain_slot']}") for record in cur_records],
                    [SingleTurnCuratorAgent(system_prompt=self.curator_system_prompt) for _ in cur_records],
                    timing_raw=timing_raw,
                    meta_info=meta_info,
                )
                updated_playbooks = []
                for record, trajectory in zip(cur_records, curators, strict=True):
                    response = self._decode_trajectory_response(trajectory)
                    record["response_text"] = response
                    updated, update_record = self._joint_updated_playbook(before_playbooks[record["task_id"]], response)
                    updated_playbooks.append(updated)
                    record.update(update_record)

                # Reuse ACE's environment reconstruction, but collect one direct
                # continuation per chain. No candidate scoring or selection runs.
                downstream_envs, downstream_agents, specs = self._build_local_candidate_eval_envs_and_agents(
                    [(index, {"task_id": task_id, "guidebook_uid": f"chain::{task_id}::{chain_slots[index]}", "playbook": updated_playbooks[index]}) for index, task_id in enumerate(chain_tasks)],
                    {task_id: [original_envs[task_to_indices[task_id][0]]] for task_id in task_ids},
                )
                if len(specs) != chain_count or [spec["task_id"] for spec in specs] != chain_tasks or [spec["candidate_idx"] for spec in specs] != list(range(chain_count)):
                    raise RuntimeError("Joint PPO must reconstruct exactly one downstream policy environment per chain")
                downstream = self._collect_joint_phase("policy", downstream_envs, downstream_agents, timing_raw=timing_raw, meta_info=meta_info)
                terminal_rewards = [float(row["trajectory_reward"]) for row in downstream]

                all_trajectories = []
                parent_uids = self._current_batch.non_tensor_batch.get("uid", []) if self._current_batch is not None else []
                # Segment order is explicit metadata; ordinary GAE is per row.
                phases = (
                    ("initial_actor", "policy", seed),
                    ("reflector", "reflector", reflectors),
                    ("curator", "curator", curators),
                    ("downstream_actor", "policy", downstream),
                )
                for segment, component, trajectories in phases:
                    for task_index, (task_id, trajectory) in enumerate(zip(chain_tasks, trajectories, strict=True)):
                        row = copy.deepcopy(trajectory)
                        row["_verifier_reward"] = float(trajectory["trajectory_reward"])
                        row["trajectory_reward"] = terminal_rewards[task_index]
                        row["_component_type"] = component
                        row["_chain_segment"] = segment
                        row["_chain_task_index"] = task_index
                        row["_chain_id"] = str(parent_uids[task_index]) if len(parent_uids) else f"{self.global_steps}::{task_id}"
                        if group_size > 1:
                            row["_chain_id"] += f"::rollout-{chain_slots[task_index]}"
                        row["idx"] = len(all_trajectories)
                        all_trajectories.append(row)
                output, transform_metrics = self._transform_agent_trajectories(all_trajectories)
                if len(output) != 4 * chain_count:
                    raise RuntimeError("Joint PPO transformation must retain every generated chain segment")
                self._expand_joint_batch(all_trajectories)

                # Metrics describe the actual downstream actor outcome, while
                # PPO uses the shared chain reward on every generated role row.
                step_rewards = [list(map(float, trajectory.get("step_rewards", []))) for trajectory in downstream]
                self._current_step_rewards_list = step_rewards if any(step_rewards) else None
                self._current_step_reward_tasks = list(chain_tasks) if self._current_step_rewards_list is not None else None
                self._current_validation_policy_tasks = None
                self._current_validation_policy_sources = None
                grouped_policy = {task_id: [row for index in indices for row in (seed[index], downstream[index])] for task_id, indices in task_to_indices.items()}
                chat_paths = self._save_task_policy_chat_logs(grouped_policy, meta_info=meta_info)
                snapshots = {}
                # Persist the first chain, independent of rewards; train every branch.
                persisted_playbooks = {task_id: updated_playbooks[task_to_indices[task_id][0]] for task_id in task_ids}
                for task_id in task_ids:
                    self.local_train_playbooks[task_id] = copy.deepcopy(persisted_playbooks[task_id])
                    snapshots[task_id] = self._save_local_snapshot(task_id, self.local_train_playbooks[task_id], is_eval=False)
                self.playbook = copy.deepcopy(persisted_playbooks[task_ids[0]])
                self._inject_task_playbooks_into_agents(task_to_indices, is_eval=False)

                metrics = dict(transform_metrics)
                metrics.update(actor_reward_metrics)
                metrics.update(
                    self._compute_rollout_token_totals(
                        policy_trajectories=seed + downstream,
                        reflector_trajectories=reflectors,
                        curator_trajectories=curators,
                        ppo_trajectories=all_trajectories,
                    )
                )
                # Count loss-eligible tokens after trajectory assembly/masking.
                # Like the inherited counter, this is per collection step; critic
                # warmup may suppress the actor update after this batch is built.
                metrics["rollout/total_gradient_update_tokens"] = int(output.batch["response_mask"].sum().item())
                metrics.update({
                    "ace/policy_rollouts": len(seed) + len(downstream),
                    "ace/reflector_rollouts": len(reflectors),
                    "ace/curator_rollouts": len(curators),
                    "ace/total_rollouts": len(all_trajectories),
                    "ace/current_reward": float(np.mean(seed_rewards)),
                    "ace/selected_reward": float(np.mean(terminal_rewards)),
                    "joint_ppo/chains": chain_count,
                    "joint_ppo/chains_per_task": group_size,
                    "joint_ppo/downstream_reward": float(np.mean(terminal_rewards)),
                    "joint_ppo/playbook_updates": sum(record["playbook_changed"] for record in cur_records),
                    "ace/active_component": hash("all") % 1000,
                    "ace/policy_train_source_id": self._policy_train_source_metric_id("both"),
                    "local/task_batch_size": len(task_ids),
                    "local/task_playbook_count": len(self.local_train_playbooks),
                    "playbook/size": float(np.mean([len(playbook.to_string()) for playbook in persisted_playbooks.values()])),
                    "playbook/bullets": float(np.mean([playbook.get_stats().get("total_bullets", 0) for playbook in persisted_playbooks.values()])),
                })
                self._add_task_flush_metrics(metrics, task_ids, is_eval=False)
                payloads = {}
                for task_id, indices in task_to_indices.items():
                    chains = []
                    for index in indices:
                        chain_id = all_trajectories[index]["_chain_id"]
                        chains.append({
                            "id": chain_id,
                            "slot": chain_slots[index],
                            "initial_reward": seed_rewards[index],
                            "terminal_reward": terminal_rewards[index],
                            "playbook_after": updated_playbooks[index].to_string(),
                            "segments": [row["_chain_segment"] for row in all_trajectories if row["_chain_id"] == chain_id],
                        })
                    payloads[task_id] = {
                        "step": int(self.global_steps), "mode": self._artifact_mode(meta_info), "training_mode": self.training_mode,
                        "chat_completions_path": chat_paths.get(task_id),
                        "artifacts": {"policy_chat_completions": chat_paths.get(task_id), "playbook_snapshot": snapshots[task_id]},
                        "metrics": metrics,
                        "playbook": {"before": before[task_id], "after": persisted_playbooks[task_id].to_string()},
                        "playbook_flush": self._task_flush_payload(task_id, is_eval=False),
                        "reflector": [ref_records[index] for index in indices], "curator": [cur_records[index] for index in indices],
                        "chains": chains,
                        "persisted_chain_id": chains[0]["id"],
                        "persistence_rule": "first_chain",
                    }
                    if group_size == 1:
                        payloads[task_id]["chain"] = chains[0]
                self._save_task_ace_logs(payloads, meta_info=meta_info)
                epoch = self._estimate_epoch()
                if epoch > self._current_epoch:
                    self._flush_epoch_table()
                    self._current_epoch = epoch
                for task_id in task_ids:
                    playbook = persisted_playbooks[task_id]
                    self._epoch_task_data.append(
                        {
                            "epoch": epoch,
                            "task_id": task_id,
                            "reward": float(np.mean([seed_rewards[index] for index in task_to_indices[task_id]])),
                            "selected_reward": float(np.mean([terminal_rewards[index] for index in task_to_indices[task_id]])),
                            "playbook_size": len(playbook.to_string()),
                            "playbook_bullets": playbook.get_stats().get("total_bullets", 0),
                        }
                    )
                return output, metrics
        finally:
            engine.update_envs_and_agents(original_envs, original_agents)
            self._set_component_token_limits("policy")
            self._original_envs, self._original_agents = old_original_envs, old_original_agents
            self._artifact_task_id, self._artifact_task_ids = old_artifact_id, old_artifact_ids
            self._suppress_inner_flush = old_suppress
