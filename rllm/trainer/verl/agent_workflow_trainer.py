import asyncio
import copy
import json
import math
import os
import threading
import uuid
from collections import Counter, defaultdict
from functools import reduce
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    agg_loss,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer

try:
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
except ModuleNotFoundError:
    from verl.workers.roles.utils.padding import left_right_2_no_padding, no_padding_2_padding

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.trainer.utils.metric_normalization import normalize_actor_metric_keys
from rllm.trainer.verl.rejection_gate import group_rejection_verdict
from rllm.trainer.verl.rollout_correction import apply_rollout_correction, rollout_correction_settings
from rllm.utils.compute_pass_at_k import compute_pass_at_k_max_reward_from_step_rewards, compute_turn_level_metrics
from rllm.utils.episode_logger import EpisodeLogger
from rllm.workflows.workflow import TerminationReason

# Cross-benchmark wandb alignment (2026-07-03): map the workflow path's
# TerminationReason values (lowercase) onto the agent+env path's termination
# vocabulary (AgentExecutionEngine, uppercase) so batch/term_* keys are shared
# with appworld/alfworld/mbpp runs. Semantic verification per pair:
#   env_done                     -> ENV_DONE          (env returned done=True)
#   max_prompt_length_exceeded   -> PROMPT_TRUNCATION (prompt over cap; engine :272/:290)
#   max_response_length_exceeded -> TRUNCATION        (response budget hit;  engine :403)
#   max_turns_exceeded           -> MAX_STEPS         (turn cap;             engine :432)
#   timeout                      -> TIMEOUT
#   error / unknown              -> ERROR / UNKNOWN   (workflow-only, no agent-path collision)
_TERMINATION_TO_AGENT_VOCAB = {
    "env_done": "ENV_DONE",
    "env_execution_uncertain": "ENV_EXECUTION_UNCERTAIN",
    "max_prompt_length_exceeded": "PROMPT_TRUNCATION",
    "max_response_length_exceeded": "TRUNCATION",
    "max_turns_exceeded": "MAX_STEPS",
    "timeout": "TIMEOUT",
    "error": "ERROR",
    "unknown": "UNKNOWN",
}


class AgentWorkflowPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        workflow_class=None,
        workflow_args=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self._validate_config()
        self._rollout_correction = rollout_correction_settings(self.config)

        # Initialize teacher engine if distillation is enabled
        self.distill_enabled = self.config.rllm.get("distill", {}).get("enable", False)
        if self.distill_enabled:
            print("Distillation is enabled, will ignore rewards returned in episodes.")
        self.teacher_engine = None
        self.teacher_tokenizer = None
        if self.distill_enabled:
            from transformers import AutoTokenizer

            from rllm.engine.rollout.openai_engine import OpenAIEngine

            teacher_rollout_args = self.config.rllm.distill.get("teacher_rollout_args", {})
            teacher_model = teacher_rollout_args.get("model", "")
            if not teacher_model:
                raise ValueError("model must be specified in rllm.distill.teacher_rollout_args when distillation is enabled")

            self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model)
            self.teacher_engine = OpenAIEngine(
                **teacher_rollout_args,
                tokenizer=self.teacher_tokenizer,
            )

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _validate_config(self):
        assert self.workflow_class is not None, "workflow_class is required for agent workflow trainer"
        assert self.config.actor_rollout_ref.hybrid_engine is True, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"
        assert self.use_rm is False, "Reward models are not supported. Rewards should be assigned using a reward function in the workflow or environment."
        if self.config.rllm.rejection_sample.multiplier != 1:
            assert self.config.rllm.rejection_sample.enable is True, "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        # TODO: revisit whether this is now supported by Verl
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX is not supported yet")

    def init_workers(self):
        super().init_workers()

        rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # Create episode logger if enabled in config
        episode_logger = None
        if self.config.trainer.get("log_episodes", False):
            # Get episode log directory from config, default to "logs/my_project/my_experiment"
            episode_log_dir = self.config.trainer.get("episode_log_dir", f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}")
            episode_logger = EpisodeLogger(base_dir=episode_log_dir, subdirectory="episodes")

        # Persistent rollout-error log (written regardless of log_episodes):
        # per-attempt failures otherwise only print inside the Ray actor and
        # vanish with the node-local Ray temp dir when the job ends.
        error_log_dir = self.config.trainer.get("episode_log_dir", f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}")

        self.agent_execution_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            retry_limit=self.config.rllm.workflow.retry_limit,
            # False: a task failing all retries drops its rollout from the batch
            # (existing dropped_episodes path) instead of killing the run
            raise_on_error=self.config.rllm.workflow.get("raise_on_error", True),
            episode_logger=episode_logger,
            error_log_path=os.path.join(error_log_dir, "rollout_errors.jsonl"),
        )

        # init workflow workers
        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()

    def _maybe_restore_rollout_after_skipped_batch(self, timing_raw=None):
        """Restore HYBRID weights/cache without publishing an optimizer update."""
        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        if not self.config.actor_rollout_ref.rollout.free_cache_engine or checkpoint_manager is None:
            return
        if timing_raw is None:
            timing_raw = {}
        print(f"Refilling training batch at step {self.global_steps}; restoring unchanged rollout weights/cache.", flush=True)
        with marked_timer("restore_rollout_after_skip", timing_raw):
            # None preserves the previously published version, including fresh
            # step zero and the first wave after loading a checkpoint.
            checkpoint_manager.update_weights()

    def _load_checkpoint(self):
        from rllm.utils.training_reward import restore_training_reward

        self._workflow_checkpoint_loaded = False
        result = super()._load_checkpoint()
        if self.config.trainer.resume_mode == "disable" or (result == 0 and self.global_steps == 0):
            restore_training_reward(self, False)
            return result
        if not isinstance(getattr(self, "_dataloader_resume_iterator_finished", None), bool):
            raise RuntimeError("Workflow resume requires the pinned VeRL full dataloader-state restore and a checkpoint with data.pt.")
        if self.config.trainer.resume_mode == "resume_path":
            checkpoint_dir = Path(self.config.trainer.resume_from_path)
        else:
            # Resolve from the step actually loaded, never a second marker read.
            checkpoint_dir = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        epoch_path = checkpoint_dir / "epoch_state.json"
        try:
            with epoch_path.open() as stream:
                state = json.load(stream)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid workflow epoch checkpoint: {epoch_path}") from exc
        if (
            not isinstance(state, dict)
            or type(state.get("epoch")) is not int
            or state["epoch"] < 0
            or type(state.get("global_step")) is not int
            or state["global_step"] != self.global_steps
        ):
            raise ValueError(f"Invalid workflow epoch checkpoint: {epoch_path}")
        self._training_epoch = state["epoch"]
        restore_training_reward(self, True)
        if self._training_epoch != state["epoch"]:
            raise ValueError("Training reward checkpoint epoch disagrees with workflow epoch")
        self._workflow_checkpoint_loaded = True
        print(f"Workflow resume: restored epoch {self._training_epoch} from {epoch_path}", flush=True)
        return result

    def _save_checkpoint(self):
        from rllm.utils.run_state import atomic_json
        from rllm.utils.training_reward import save_training_reward

        if self.config.trainer.get("default_hdfs_dir") is not None:
            raise NotImplementedError("Workflow epoch checkpoints support local/shared filesystems only")
        if type(getattr(self, "_training_epoch", None)) is not int or self._training_epoch < 0:
            raise ValueError("Workflow checkpoint requires the actual training epoch")
        checkpoint_dir = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        # A sidecar failure must not let the parent advertise the checkpoint.
        atomic_json(checkpoint_dir / "epoch_state.json", {"epoch": self._training_epoch, "global_step": self.global_steps})
        save_training_reward(self)
        return super()._save_checkpoint()

    def _get_training_start_epoch(self):
        # A finished StatefulDataLoader starts the next epoch on iter(); a
        # last-batch/pre-StopIteration snapshot must consume its empty tail.
        return getattr(self, "_training_epoch", 0) + int(getattr(self, "_dataloader_resume_iterator_finished", None) is True)

    def _iter_training_epoch(self):
        return iter(self.train_dataloader)

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from rllm.utils.run_state import finish_data_exhausted, prepare_training_run
        from rllm.utils.tracking import Tracking
        from rllm.utils.training_reward import complete_training_reward_epoch, initialize_training_reward, workflow_reward_metrics

        prepare_training_run(self.config)
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        start_epoch = self._get_training_start_epoch()
        initialize_training_reward(self, start_epoch)
        getattr(self, "checkpoint_manager", None) and self.checkpoint_manager.update_weights(self.global_steps)

        # Workflow updates are zero-based. A saved N already includes update N;
        # fresh training still begins at zero, while a resume begins at N + 1.
        next_step = self.global_steps + int(self._workflow_checkpoint_loaded)
        if next_step >= self.total_training_steps and not self.config.trainer.get("val_only", False):
            print(f"[run] Training already complete at step {self.global_steps}", flush=True)
            return

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=start_epoch)
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        self.global_steps = next_step

        batch = None
        solve_none = 0
        solve_all = 0
        solve_partial = 0
        solve_all_mixed_reward = 0
        kept_groups = 0
        num_tasks = 0
        termination_counts = Counter()
        workflow_metrics = defaultdict(list)
        metrics = {}
        timing_raw = {}
        # Cross-benchmark aligned accumulators (traj/*, batch/term_*, turn/*):
        # per-TRAJECTORY raw values, matching agent_ppo_trainer definitions.
        aligned_traj_stats = defaultdict(list)  # llm_time/env_time/reward_time/total_time/steps
        aligned_term_reasons = []  # one entry per attempted episode (incl. dropped)
        aligned_step_rewards = []  # per-trajectory step-reward sequences
        aligned_step_reward_tasks = []  # task id per entry of aligned_step_rewards (pass@k grouping)

        print(
            f"[kailin-debug] fit start: len(train_dataloader)={len(self.train_dataloader)} "
            f"total_training_steps={self.total_training_steps} "
            f"total_epochs={self.config.trainer.total_epochs} "
            f"test_freq={self.config.trainer.test_freq} "
            f"val_before_train={self.config.trainer.get('val_before_train', True)} "
            f"global_steps={self.global_steps}",
            flush=True,
        )

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self._training_epoch = epoch
            self._training_reward_tracker.begin_epoch(epoch)
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self._iter_training_epoch():
                print(f"[kailin-debug] >>> training step START: global_steps={self.global_steps}", flush=True)
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_tasks += len(new_batch.batch)

                new_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n)
                expected_reward_count = len(new_batch)

                # Update training step in engine for episode logging
                self.agent_execution_engine.set_training_step(self.global_steps, mode="train", epoch=epoch)

                with marked_timer("step", timing_raw):
                    # generate trajectories
                    final_gen_batch_output = self.generate_trajectories(batch=new_batch, timing_raw=timing_raw)
                    getattr(self, "checkpoint_manager", None) and self.checkpoint_manager.sleep_replicas()

                    # need to repeat to make shape match
                    repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                    new_batch = new_batch.sample_level_repeat(repeat_counts)
                    final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                    new_batch = new_batch.union(final_gen_batch_output)

                    # rejection sampling
                    # we do rejection sampling at the episode level instead of the traj/step level
                    # mode "correct" (default) keeps groups with mixed is_correct flags;
                    # mode "variance" keeps groups whose rewards are not all equal — for
                    # graded rewards the flags binarize at reward>0 and would discard
                    # groups like [0.2, 0.2, 1.0, 0.2] that GRPO can learn from.
                    rejection_mode = self.config.rllm.rejection_sample.get("mode", "correct")
                    uids = new_batch.non_tensor_batch["task_ids"]
                    unique_uids = np.unique(uids)
                    is_correct = new_batch.non_tensor_batch["is_correct"]
                    # every row of a trajectory carries the full trajectory reward on its
                    # last response token (transform._build_step_and_trajectory_rewards),
                    # so the per-row sum recovers that row's trajectory reward exactly
                    if "traj_rewards" in new_batch.batch.keys():
                        row_rewards = new_batch.batch["traj_rewards"].sum(-1).cpu().numpy()
                    else:
                        assert rejection_mode == "correct", "rejection_sample.mode=variance requires traj_rewards in the batch"
                        row_rewards = np.zeros(len(uids), dtype=np.float64)  # census-only fallback
                    drop_uids = set()

                    for uid in unique_uids:
                        candidate_rows = uids == uid
                        keep, bucket, mixed_reward = group_rejection_verdict(
                            rewards=row_rewards[candidate_rows],
                            flags=is_correct[candidate_rows],
                            mode=rejection_mode,
                        )
                        if bucket == "solve_none":
                            solve_none += 1
                        elif bucket == "solve_all":
                            solve_all += 1
                            if mixed_reward:
                                solve_all_mixed_reward += 1
                        else:
                            solve_partial += 1
                        if keep:
                            kept_groups += 1
                        else:
                            drop_uids.add(uid)

                    # Build a view with a single item per episode_id for metrics/logging
                    seen_episodes = set()
                    episode_unique_idxs = []
                    for i, episode_id in enumerate(new_batch.non_tensor_batch["episode_ids"]):
                        if episode_id not in seen_episodes:
                            seen_episodes.add(episode_id)
                            episode_unique_idxs.append(i)
                    episode_unique_batch = new_batch.select_idxs(episode_unique_idxs)
                    # Observe all actor episodes before rejection/refill/padding.
                    # Ambiguous multiple trajectories or dropped episodes make
                    # the epoch ineligible, rather than becoming zero rewards.
                    reward_metrics = workflow_reward_metrics(new_batch, expected_reward_count)
                    self._training_reward_tracker.observe(reward_metrics, expected_count=expected_reward_count)
                    for key, value in reward_metrics.items():
                        metrics[key] = metrics.get(key, 0) + value

                    # Include termination reasons from episodes that were dropped before producing any steps.
                    # These are reported via meta_info from AgentWorkflowEngine.transform_results_for_verl.
                    dropped_episodes = final_gen_batch_output.meta_info.get("dropped_episodes", [])
                    for ep in dropped_episodes:
                        termination_counts.update([ep.get("termination_reason", "unknown")])
                        aligned_term_reasons.append(ep.get("termination_reason", "unknown"))
                    # Consumed. The per-wave list rode into new_batch via the union
                    # above, and DataProto.concat asserts meta equality across
                    # refetch waves — a wave with drops next to one without kills
                    # the run (job 15130156). Same pattern as the repeat_counts pop.
                    new_batch.meta_info.pop("dropped_episodes", None)

                    # log metrics returned by workflows. Reserved "_"-prefixed keys carry
                    # raw per-trajectory data for the cross-benchmark traj/* + turn/*
                    # metrics (attached in Workflow.postprocess_episode / TimingTrackingMixin);
                    # they are routed to the aligned accumulators, NOT batch/<key>-averaged.
                    episode_task_ids = episode_unique_batch.non_tensor_batch["task_ids"]
                    for ep_task_id, metric_dict in zip(episode_task_ids, episode_unique_batch.non_tensor_batch["metrics"], strict=False):
                        for key, value in metric_dict.items():
                            if not key.startswith("_"):
                                workflow_metrics[key].append(value)
                            elif key == "_traj_steps":
                                aligned_traj_stats["steps"].extend(value)
                            elif key == "_step_rewards":
                                aligned_step_rewards.extend(value)
                                aligned_step_reward_tasks.extend([ep_task_id] * len(value))
                            elif key in ("_llm_time", "_env_time", "_reward_time", "_total_time"):
                                aligned_traj_stats[key[1:]].append(value)

                    # collect and log termination reasons
                    termination_reasons = episode_unique_batch.non_tensor_batch["termination_reasons"]
                    termination_counts.update(termination_reasons)
                    aligned_term_reasons.extend(termination_reasons)

                    # critic/full-score/*: complete pre-rejection training scores, same
                    # definition and same gating as agent_ppo_trainer (rejection enabled;
                    # per-sequence sum of token-level trajectory rewards; last gen
                    # iteration of the step wins, mirroring the agent path's overwrite).
                    if self.config.rllm.rejection_sample.enable and "traj_rewards" in episode_unique_batch.batch.keys():
                        full_sequence_score = episode_unique_batch.batch["traj_rewards"].sum(-1)
                        metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                        metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                        metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                    # If no valid samples remain, skip this batch and get a new one
                    # if len(drop_uids) == len(unique_uids):
                    #     print("No valid samples remain, skipping batch")
                    #     continue

                    if not self.config.rllm.rejection_sample.enable:
                        batch = new_batch
                    else:
                        rejection_mask = np.isin(uids, list(drop_uids))
                        new_batch = new_batch[~rejection_mask]
                        if batch is None:
                            batch = new_batch
                        else:
                            # The transform includes rollout_log_probs only when EVERY
                            # episode in the wave returned logprobs, so refetch waves
                            # can disagree on the key and TensorDict refuses the
                            # mixed-schema concat (jobs 15088440/15128800). In this
                            # trainer the key only feeds key-guarded debug metrics
                            # (and the distill path, which requires it uniformly), so
                            # on mismatch drop it from the side that has it.
                            if ("rollout_log_probs" in batch.batch.keys()) != ("rollout_log_probs" in new_batch.batch.keys()):
                                if "rollout_log_probs" in batch.batch.keys():
                                    del batch.batch["rollout_log_probs"]
                                else:
                                    del new_batch.batch["rollout_log_probs"]
                            batch = DataProto.concat([batch, new_batch])

                        # kept_groups counts actual keeps (census buckets are
                        # reward-based and may diverge from the keep decision
                        # when a workflow overrides is_correct)
                        if kept_groups < self.config.data.train_batch_size:
                            self._maybe_restore_rollout_after_skipped_batch(timing_raw=timing_raw)
                            continue
                        else:
                            # randomly select bsz task uids from batch, then filter batch to only contain these tasks
                            # TODO: add heuristic for selecting train_batch_size uids
                            uids = batch.non_tensor_batch["task_ids"]
                            unique_uids = np.unique(uids)
                            assert len(unique_uids) >= self.config.data.train_batch_size, "Not enough unique uids to sample from"
                            selected_uids = np.random.choice(unique_uids, size=self.config.data.train_batch_size, replace=False)
                            selected_mask = np.isin(uids, selected_uids)
                            batch = batch[selected_mask]

                    if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                        # need to make sure both number of last steps (number of uids) and number of total steps in the batch
                        # (batch size after processing) are both multiples of world size

                        # first we split the batch in two: one with only the last steps of each trajectory and the other with the remaining steps
                        is_last_step = batch.non_tensor_batch["is_last_step"]
                        valid_last_step_indices = np.where(is_last_step == True)[0]
                        not_last_step_indices = np.where(is_last_step == False)[0]
                        last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                        non_last_step_batch = batch.select_idxs(not_last_step_indices)

                        # round down last_step_batch to make sure its multiple of world size
                        num_trainer_replicas = self.actor_rollout_wg.world_size
                        max_batch_size = (last_step_batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas

                        size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                        size_mask[:max_batch_size] = True
                        last_step_batch = last_step_batch[size_mask]  # filtered last steps

                        # now we go through all the non_last_step_batch and keep everything that has same trajectory_id that exists in the filtered last steps
                        valid_last_step_trajectory_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_trajectory_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_mask = np.isin(non_last_step_trajectory_ids, valid_last_step_trajectory_ids)
                        non_last_step_batch = non_last_step_batch[non_last_step_mask]

                        # concatenate then pad
                        batch = DataProto.concat([last_step_batch, non_last_step_batch])
                        batch = self._pad_dataproto_to_world_size(batch)

                    else:
                        # then we just pad the batch size to a multiple of world size
                        batch = self._pad_dataproto_to_world_size(batch=batch)

                    # Balance the number of valid tokens across DP ranks BEFORE compute operations.
                    # This must happen before compute_log_prob to prevent NCCL desync when workers
                    # process micro-batches with uneven token distributions.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                        images_seqlens_all = []
                        for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                            if "image_grid_thw" not in multi_modal_input.keys():
                                continue
                            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                        batch.meta_info["images_seqlens"] = images_seqlens_all
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    # recompute old_log_probs
                    # Mirror verl's RayPPOTrainer._compute_old_log_prob: the new EngineWorker
                    # path (use_legacy_worker_impl == "disable") takes a TensorDict in no-padding
                    # format; the legacy AsyncActorRolloutRefWorker takes a DataProto directly.
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if self.use_legacy_worker_impl == "disable":
                            batch_td = batch.to_tensordict()
                            batch_td = left_right_2_no_padding(batch_td)
                            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False, temperature=self.config.actor_rollout_ref.rollout.temperature)
                            old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch_td)
                            # New worker returns TensorDict in no-padding format
                            entropy = tu.get(old_log_prob_output, "entropy")
                            log_probs = tu.get(old_log_prob_output, "log_probs")
                            entropy = no_padding_2_padding(entropy, batch_td)
                            log_probs = no_padding_2_padding(log_probs, batch_td)
                            old_log_prob = DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()}))
                        else:
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        rollout_correction = getattr(self, "_rollout_correction", None)
                        if rollout_correction is not None:
                            # Inference padding rows are removed before advantages.
                            batch, correction_metrics = apply_rollout_correction(batch, rollout_correction, excluded_rows=batch.non_tensor_batch.get("is_pad_step"))
                            metrics.update(correction_metrics)
                        elif "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            debug_metrics = calculate_debug_metrics(batch)
                            metrics.update(debug_metrics)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if self.use_legacy_worker_impl == "disable":
                                # Reuse batch_td from old_log_prob, but override the metadata for
                                # the ref step (verl's RayPPOTrainer._compute_ref_log_prob pattern).
                                # When ref_in_actor (LoRA), we use the actor worker's compute_log_prob
                                # with no_lora_adapter=True so the base model is used for ref.
                                ref_metadata = {"calculate_entropy": False, "compute_loss": False}
                                if self.ref_in_actor:
                                    ref_metadata["no_lora_adapter"] = True
                                tu.assign_non_tensor(batch_td, **ref_metadata)
                                if self.ref_in_actor:
                                    ref_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch_td)
                                else:
                                    ref_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
                                ref_lp = tu.get(ref_log_prob_output, "log_probs")
                                ref_lp = no_padding_2_padding(ref_lp, batch_td)
                                ref_log_prob = DataProto.from_tensordict(tu.get_tensordict({"ref_log_prob": ref_lp.float()}))
                            else:
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            # Dispatch through RayPPOTrainer._compute_values: the new EngineWorker
                            # path (use_legacy_worker_impl == "disable", verl default) exposes
                            # infer_batch on a no-padding TensorDict, the legacy CriticWorker
                            # exposes compute_values on a DataProto.
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # step_ids is safe to always use for advantage computation
                        # if we're not using computing advantages stepwise (i.e., for cumulative agents or single turn workflows)
                        # then step_ids == trajectory_ids
                        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "per_step":
                            batch.batch["token_level_scores"] = batch.batch["step_rewards"]
                        else:
                            batch.batch["token_level_scores"] = batch.batch["traj_rewards"]

                        if self.distill_enabled:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                            batch = self._remove_padding(batch)
                            distill_advantages = asyncio.run_coroutine_threadsafe(self._compute_distill_advantages(batch), self._loop).result()
                            batch.batch["advantages"] = distill_advantages
                            batch.batch["returns"] = distill_advantages
                        else:
                            # apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                                # last_step_batch contains no padded steps as it was rounded down (not padded) to a multiple of world size
                            else:
                                batch = self._remove_padding(batch)  # compute advantages over non-padded steps only

                            # compute advantages, executed on the driver process
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # Merging the separated out steps using the advantage from last steps
                                self._stepwise_advantage_broadcast(batch, non_last_step_batch)
                                batch = DataProto.concat([batch, non_last_step_batch])

                    # remove invalid items filtered out due to compact filtering
                    is_valid = batch.non_tensor_batch["is_valid"]
                    valid_idxs = np.where(is_valid == True)[0]
                    batch = batch.select_idxs(valid_idxs)

                    # for backward compatibility
                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    # re-pad batch size to world size for gradient update
                    batch = self._pad_dataproto_to_world_size(batch=batch)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_batch = self._pad_workflow_ppo_update_batch(
                                batch,
                                worker_group=self.critic_wg,
                                global_mini_batch_size=self.config.critic.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n,
                                metrics=metrics,
                                metric_prefix="update_critic",
                            )
                            critic_output = self._update_critic(critic_batch)  # impl-dispatching helper (see _compute_values above)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup: no actor update, but update_weights() must
                        # still run -- in the free_cache_engine path it is what wakes the
                        # rollout replicas put to sleep after generation (verl
                        # RayPPOTrainer.fit has the same branch). Without it the next
                        # generate runs on unmapped weight memory -> vLLM "CUDA error: an
                        # illegal memory access" in the first embedding kernel (appworld PPO
                        # smoke jobs 16318811 / 16321138 / 16323852, 2026-08-25).
                        with marked_timer("update_weights", timing_raw):
                            getattr(self, "checkpoint_manager", None) and self.checkpoint_manager.update_weights(self.global_steps)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Mirror verl's RayPPOTrainer._update_actor: meta_info goes onto the
                        # DataProto for the legacy worker; the new EngineWorker path needs a
                        # TensorDict in no-padding format with training metadata set via
                        # tu.assign_non_tensor.
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        if self.use_legacy_worker_impl == "disable":
                            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
                            update_batch = self._prepare_actor_update_batch(batch, mini_batch_size=ppo_mini_batch_size, metrics=metrics).to_tensordict()
                            update_batch = left_right_2_no_padding(update_batch)
                            tu.assign_non_tensor(
                                update_batch,
                                calculate_entropy=(self.config.actor_rollout_ref.actor.entropy_coeff != 0.0),
                                global_batch_size=ppo_mini_batch_size,
                                mini_batch_size=ppo_mini_batch_size,
                                epochs=self.config.actor_rollout_ref.actor.ppo_epochs,
                                seed=self.config.actor_rollout_ref.actor.data_loader_seed,
                                dataloader_kwargs={"shuffle": self.config.actor_rollout_ref.actor.shuffle},
                            )
                        else:
                            update_batch = batch

                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self.actor_rollout_wg.update_actor(update_batch)

                        # save checkpoint
                        if (
                            self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0
                        ) or (self.config.trainer.get("save_at_end", False) and self.global_steps + 1 >= self.total_training_steps):
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            getattr(self, "checkpoint_manager", None) and self.checkpoint_manager.update_weights(self.global_steps)

                        # verl 0.7.1 new worker returns TensorDict; extract metrics
                        if isinstance(actor_output, TensorDict):
                            actor_output_metrics = normalize_actor_metric_keys(reduce_metrics(tu.get(actor_output, "metrics")))
                        else:
                            actor_output_metrics = normalize_actor_metric_keys(reduce_metrics(actor_output.meta_info["metrics"]))
                        metrics.update(actor_output_metrics)

                    # Visualize some sample trajectories
                    if batch is not None and len(batch) > 0:
                        # Randomly select a few samples to visualize
                        batch_size = len(batch)
                        num_samples = min(2, batch_size)  # Visualize up to 2 samples
                        if num_samples > 0:
                            sample_indices = np.random.choice(batch_size, size=num_samples, replace=False)
                            for idx in sample_indices:
                                self.visualize_trajectory_last_step(batch, sample_idx=idx, max_samples=1)

                # validate
                _do_perstep_val = self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0
                print(
                    f"[kailin-debug] step {self.global_steps}: per-step validation -> {'RUN' if _do_perstep_val else 'SKIP'} (test_freq={self.config.trainer.test_freq})",
                    flush=True,
                )
                if _do_perstep_val:
                    with marked_timer("testing", timing_raw, color="green"):
                        self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=epoch)
                        val_metrics: dict = self._validate_agent()
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Add token tracking metrics
                if "total_prompt_tokens" in batch.non_tensor_batch:
                    token_prompt = batch.non_tensor_batch["total_prompt_tokens"]
                    token_completion = batch.non_tensor_batch["total_completion_tokens"]
                    token_gradient = batch.non_tensor_batch["total_gradient_update_tokens"]
                    metrics.update(
                        {
                            "token/total_prompt_tokens_mean": np.mean(token_prompt),
                            "token/total_prompt_tokens_max": np.max(token_prompt),
                            "token/total_prompt_tokens_min": np.min(token_prompt),
                            "token/total_completion_tokens_mean": np.mean(token_completion),
                            "token/total_completion_tokens_max": np.max(token_completion),
                            "token/total_completion_tokens_min": np.min(token_completion),
                            "token/total_gradient_update_tokens_mean": np.mean(token_gradient),
                            "token/total_gradient_update_tokens_max": np.max(token_gradient),
                            "token/total_gradient_update_tokens_min": np.min(token_gradient),
                        }
                    )

                metrics["batch/solve_none"] = solve_none / num_tasks
                metrics["batch/solve_all"] = solve_all / num_tasks
                metrics["batch/solve_partial"] = solve_partial / num_tasks
                # solve_all groups with unequal rewards: signal the legacy "correct"
                # gate discards and the "variance" gate keeps (logged in both modes)
                metrics["batch/solve_all_mixed_reward"] = solve_all_mixed_reward / num_tasks

                for key, value in workflow_metrics.items():
                    metrics[f"batch/{key}"] = np.mean(value)

                # Denominator should be the number of attempted tasks (including those dropped before 1st step),
                # otherwise metrics are biased/inflated.
                for r in TerminationReason:
                    metrics[f"batch/{r.value}"] = termination_counts[r.value] / max(1, num_tasks)

                metrics["batch/num_tasks"] = num_tasks

                # ---- Cross-benchmark aligned metrics (2026-07-03) ------------------
                # Same keys + same definitions as agent_ppo_trainer (appworld/alfworld/
                # mbpp) so mini_swe runs are directly comparable in wandb. The legacy
                # batch/<reason> and batch/<key> metrics above are kept for dashboard
                # continuity ("acceptable new metrics" direction is fine; renames are not).

                # traj/<k>_{mean,min,max}: verbatim aggregation from agent_ppo_trainer
                # (per-TRAJECTORY values; None/negative filtered as "not measured").
                for k, v_list in aligned_traj_stats.items():
                    v_list = [v for v in v_list if v is not None and v >= 0]
                    if not v_list:
                        continue
                    v_list = np.array(v_list)
                    metrics.update(
                        {
                            f"traj/{k}_mean": v_list.mean(),
                            f"traj/{k}_min": v_list.min(),
                            f"traj/{k}_max": v_list.max(),
                        }
                    )

                # batch/term_<REASON>: fractions of TRAJECTORIES (agent-path convention;
                # the legacy batch/<reason> above divides by tasks instead — with
                # rollout.n > 1 the two intentionally differ).
                _term_counts = Counter(_TERMINATION_TO_AGENT_VOCAB.get(r, r) for r in aligned_term_reasons)
                _n_traj = max(1, len(aligned_term_reasons))
                for _reason, _cnt in _term_counts.items():
                    metrics[f"batch/term_{_reason}"] = _cnt / _n_traj

                # turn/* + train/turn_table + train/pass_at_k_max_reward: identical call
                # to agent_ppo_trainer._consume_current_step_reward_metrics. Sequences
                # are per-trajectory step rewards with the graded terminal reward last,
                # so with cutoff_mode="max" the turn curve reads "reward achieved within
                # <= k turns", exactly as in appworld.
                if aligned_step_rewards and any(len(sr) > 0 for sr in aligned_step_rewards):
                    _gamma = self.config.algorithm.get("gamma", 1.0)
                    _cutoff_mode = self.config.rllm.get("turn_level_cutoff_mode", "max")
                    _max_steps = OmegaConf.select(self.config, "rllm.workflow.workflow_args.max_steps")
                    train_pass_at_k_max_reward = compute_pass_at_k_max_reward_from_step_rewards(
                        aligned_step_rewards,
                        tasks=aligned_step_reward_tasks if aligned_step_reward_tasks else None,
                    )
                    metrics.update(
                        compute_turn_level_metrics(
                            aligned_step_rewards,
                            logger=logger,
                            gamma=_gamma,
                            step=self.global_steps,
                            cutoff_mode=_cutoff_mode,
                            max_steps=_max_steps,
                            prefix="turn",
                            table_prefix="train",
                            table_extra_values={
                                "pass_at_k_max_reward": train_pass_at_k_max_reward,
                            },
                        )
                    )
                    metrics["train/pass_at_k_max_reward"] = train_pass_at_k_max_reward

                # timing_s/collect_trajectory: agent-path name for the rollout phase
                # (workflow path calls it generate_trajectories; same measured span).
                if "timing_s/generate_trajectories" in metrics:
                    metrics["timing_s/collect_trajectory"] = metrics["timing_s/generate_trajectories"]

                # TODO: make a canonical logger that supports various backend
                logger.log(data=normalize_actor_metric_keys(metrics), step=self.global_steps)

                batch = None
                solve_none = 0
                solve_all = 0
                solve_partial = 0
                solve_all_mixed_reward = 0
                kept_groups = 0
                num_tasks = 0
                termination_counts = Counter()
                workflow_metrics = defaultdict(list)
                metrics = {}
                timing_raw = {}
                aligned_traj_stats = defaultdict(list)
                aligned_term_reasons = []
                aligned_step_rewards = []
                aligned_step_reward_tasks = []

                self.global_steps += 1
                print(
                    f"[kailin-debug] <<< step done: global_steps={self.global_steps} "
                    f"total_training_steps={self.total_training_steps} "
                    f"will_break(end training)={self.global_steps >= self.total_training_steps}",
                    flush=True,
                )

                if self.global_steps >= self.total_training_steps and not self.config.trainer.get("val_only", False):
                    print(
                        f"[kailin-debug] reached total_training_steps -> FINAL validation: {'RUN' if self.val_reward_fn is not None and self.config.trainer.get('val_after_train', True) else 'SKIP'}",
                        flush=True,
                    )
                    # perform validation after training
                    if self.val_reward_fn is not None and self.config.trainer.get("val_after_train", True):
                        self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=epoch)
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)

                    try:
                        logger.finish()
                    except Exception:
                        pass  # skip errors during cleanup

                    return

            complete_training_reward_epoch(self)

        finish_data_exhausted(self)

    def _validate_agent(self):
        is_correct_lst = []
        data_source_lst = []
        uid_lst = []
        workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))
        batches_for_distill = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            test_batch.meta_info = {"validate": True}

            # Keep a mapping from task_id -> data_source so we can account for dropped episodes.
            base_data_sources = test_batch.non_tensor_batch.get("data_source", None)
            if base_data_sources is None:
                base_data_sources = ["unknown"] * len(test_batch)
            task_to_source = dict(zip(test_batch.non_tensor_batch["task_ids"].tolist(), base_data_sources, strict=False))

            test_output_gen_batch = self.generate_trajectories(batch=test_batch)

            # Account for dropped episodes in validation metrics to avoid inflated pass@k.
            dropped_episodes = test_output_gen_batch.meta_info.get("dropped_episodes", [])
            for ep in dropped_episodes:
                uid = ep.get("task_id")
                if uid is None:
                    continue
                is_correct_lst.append(False)
                uid_lst.append(uid)
                data_source_lst.append(task_to_source.get(uid, "unknown"))

            repeat_counts = test_output_gen_batch.meta_info["repeat_counts"]
            # need to repeat to make shape match
            test_batch = test_batch.sample_level_repeat(repeat_counts)
            test_output_gen_batch.meta_info.pop("repeat_counts", None)  # no longer needed after this
            test_batch = test_batch.union(test_output_gen_batch)

            if self.distill_enabled:
                batches_for_distill.append(test_batch)

            seen_episodes = set()
            selected_idxs = []
            for i, episode_id in enumerate(test_batch.non_tensor_batch["episode_ids"]):
                if episode_id not in seen_episodes:
                    seen_episodes.add(episode_id)
                    selected_idxs.append(i)
            test_batch = test_batch.select_idxs(selected_idxs)

            is_correct_lst.extend(test_batch.non_tensor_batch["is_correct"])
            uid_lst.extend(test_batch.non_tensor_batch["task_ids"])

            data_sources = test_batch.non_tensor_batch.get("data_source", None)
            if data_sources is None:
                data_sources = ["unknown"] * len(test_batch)
            data_source_lst.extend(data_sources)

            # Collect workflow metrics per episode and data source
            for i, data_source in enumerate(data_sources):
                episode_metrics = test_batch.non_tensor_batch["metrics"][i]
                if episode_metrics is not None:
                    for key, value in episode_metrics.items():
                        # "_"-prefixed keys carry raw per-trajectory data (lists,
                        # e.g. _traj_steps / _step_rewards), not averageable scalars.
                        # Skip them here so float() never sees a list; mirrors the
                        # reduction guard below and the training-path routing (~L302).
                        # TODO(kailin): port the training-path aligned-accumulator
                        # routing (lines ~302-310) into this val loop to also emit
                        # val/traj/* and val/turn/* metrics; skipped for now.
                        if key.startswith("_"):
                            continue
                        workflow_metrics_by_source[data_source][key].append(float(value))

        metrics = {}
        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)
        data_source_array = np.array(data_source_lst)

        for data_source in np.unique(data_source_array):
            pass_rates = defaultdict(list)

            data_source_mask = data_source_array == data_source
            is_correct_data_source = is_correct_array[data_source_mask]
            uids_data_source = uid_array[data_source_mask]

            for is_correct, uid in zip(is_correct_data_source, uids_data_source, strict=False):
                pass_rates[uid].append(is_correct)

            metrics[f"val/{data_source}/pass@1"] = np.mean(is_correct_data_source)
            metrics[f"val/{data_source}/pass@{n_val_samples}"] = np.mean([1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()])

            # Add workflow metrics for this data source
            if data_source in workflow_metrics_by_source:
                for key, values in workflow_metrics_by_source[data_source].items():
                    if key.startswith("_"):
                        continue  # reserved raw per-trajectory data, not averageable
                    if values:  # Only add if we have values
                        metrics[f"val/{data_source}/{key}"] = np.mean(values)

        # Compute distillation metrics if enabled
        if self.distill_enabled and batches_for_distill:
            try:
                # Concatenate all validation batches
                combined_batch = DataProto.concat(batches_for_distill)
                combined_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                if self.use_legacy_worker_impl == "disable":
                    # New worker path: convert to TensorDict + no-padding, then convert back.
                    cb_td = combined_batch.to_tensordict()
                    cb_td = left_right_2_no_padding(cb_td)
                    tu.assign_non_tensor(cb_td, calculate_entropy=True, compute_loss=False, temperature=self.config.actor_rollout_ref.rollout.temperature)
                    old_log_prob_output = self.actor_rollout_wg.compute_log_prob(cb_td)
                    log_probs = tu.get(old_log_prob_output, "log_probs")
                    log_probs = no_padding_2_padding(log_probs, cb_td)
                    old_log_prob = DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": log_probs.float()}))
                else:
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(combined_batch)
                combined_batch = combined_batch.union(old_log_prob)

                # Compute distillation advantages
                distill_advantages = asyncio.run_coroutine_threadsafe(self._compute_distill_advantages(combined_batch), self._loop).result()

                # Extract distillation metrics
                response_mask = combined_batch.batch["response_mask"]
                valid_advantages = distill_advantages[response_mask.bool()]
                if len(valid_advantages) > 0:
                    metrics["val/distill/mean_advantage"] = valid_advantages.mean().item()
                    metrics["val/distill/std_advantage"] = valid_advantages.std().item()
                    metrics["val/distill/min_advantage"] = valid_advantages.min().item()
                    metrics["val/distill/max_advantage"] = valid_advantages.max().item()
            except Exception as e:
                print(f"Warning: Failed to compute distillation metrics during validation: {e}")
                import traceback

                traceback.print_exc()

        return metrics

    async def _compute_distill_advantages(self, batch: DataProto) -> torch.Tensor:
        """
        Compute distillation advantages by querying teacher and aligning logprobs.

        For each sample in the batch:
        1. Extract student completion_ids, student logprobs, and (if needed) chat_completions
        2. Query teacher for logprobs on the same completion
        3. Align teacher logprobs to student tokens (byte-level alignment if different tokenizers)
        4. Compute per-token advantages: teacher_logprob - student_logprob

        Args:
            batch: DataProto containing student responses, logprobs, and optionally chat_completions
                  (chat_completions only needed when shared_tokenizer=False)

        Returns:
            Tensor of shape (batch_size, max_response_length) with per-token advantages.
            Padded positions have advantages = 0.0
        """
        shared_tokenizer = self.config.rllm.distill.shared_tokenizer

        prompts = batch.batch["prompts"]  # (batch_size, max_prompt_length)
        responses = batch.batch["responses"]  # (batch_size, max_response_length)
        response_mask = batch.batch["response_mask"]  # (batch_size, max_response_length)
        attention_mask = batch.batch["attention_mask"]  # (batch_size, max_prompt_length + max_response_length)
        rollout_log_probs = batch.batch["rollout_log_probs"]  # (batch_size, max_response_length)

        batch_size, max_prompt_length = prompts.shape
        _, max_response_length = responses.shape
        advantages = torch.zeros((batch_size, max_response_length), dtype=torch.float32)

        # Only need chat_completions and chat_parser when tokenizers differ
        if not shared_tokenizer:
            from rllm.trainer.distill import align_teacher_logprobs

            chat_completions = batch.non_tensor_batch.get("chat_completions", None)
            if chat_completions is None:
                raise ValueError("chat_completions not found in batch, cannot perform distillation.")

            if not hasattr(self.teacher_engine, "chat_parser") or self.teacher_engine.chat_parser is None:
                raise ValueError("Teacher engine does not have a chat_parser.")
            teacher_chat_parser = self.teacher_engine.chat_parser

        async def get_teacher_logprobs(prompt: str | list[int], prompt_length: int, sample_idx: int) -> list[float]:
            """Query teacher for logprobs
            Note: We assume the teacher engine is vLLM, not SGLang.
            This is because SGLang does not support prompt_logprobs through the completions endpoint.
            Though we could support this through the echo and logprobs params if needed.
            You can still use SGLang to serve the policy, but you should ensure the temperature is 1.0 and the top_p is 1.0."""

            teacher_resp = await self.teacher_engine.completion(
                prompt,
                max_tokens=1,
                extra_body={"prompt_logprobs": 1},
            )
            if not teacher_resp.prompt_logprobs:
                raise ValueError(f"Teacher missing prompt_logprobs for sample {sample_idx}.")

            return teacher_resp.prompt_logprobs[prompt_length:]

        async def process_sample(sample_idx: int) -> None:
            """Process a single sample: query teacher, align, compute advantages."""
            try:
                student_prompt_length = attention_mask[sample_idx, :max_prompt_length].sum().item()
                if student_prompt_length == 0:
                    raise ValueError(f"Sample {sample_idx} has no valid prompt tokens.")

                student_response_length = attention_mask[sample_idx, -max_response_length:].sum().item()
                if student_response_length == 0:
                    raise ValueError(f"Sample {sample_idx} has no valid response tokens.")

                student_prompt_ids = prompts[sample_idx, -student_prompt_length:].tolist()
                student_response_ids = responses[sample_idx, :student_response_length].tolist()
                student_logprobs = rollout_log_probs[sample_idx, :student_response_length].tolist()

                if shared_tokenizer:
                    # Fast path: student and teacher use the same tokenizer
                    # Directly use student token IDs for teacher query
                    teacher_ids = student_prompt_ids + student_response_ids
                    aligned_teacher_logprobs = await get_teacher_logprobs(teacher_ids, student_prompt_length, sample_idx)

                else:
                    # Slow path: different tokenizers, need byte-level alignment
                    sample_chat_completions = chat_completions[sample_idx]
                    if sample_chat_completions is None or len(sample_chat_completions) == 0:
                        raise ValueError(f"Sample {sample_idx} has no chat_completions for distillation.")

                    teacher_prompt_messages = sample_chat_completions[:-1]
                    teacher_completion_messages = sample_chat_completions[-1:]

                    reasoning_str = teacher_completion_messages[0].get("reasoning", "")
                    content_str = teacher_completion_messages[0].get("content", "")
                    if not reasoning_str and not content_str:
                        raise ValueError(f"Sample {sample_idx} has no reasoning or content in teacher completion message.")

                    # Build teacher prompt and completion
                    teacher_prompt = teacher_chat_parser.parse(
                        teacher_prompt_messages,
                        is_first_msg=True,
                        add_generation_prompt=True,
                        tools=[],
                        accumulate_reasoning=False,
                    )
                    teacher_prompt_ids = self.teacher_tokenizer.encode(teacher_prompt, add_special_tokens=False)

                    teacher_completion = teacher_chat_parser.parse(
                        teacher_completion_messages,
                        is_first_msg=False,
                        add_generation_prompt=False,
                        tools=[],
                        accumulate_reasoning=True,
                    )
                    if teacher_completion.startswith(teacher_chat_parser.generation_prompt):
                        teacher_completion = teacher_completion[len(teacher_chat_parser.generation_prompt) :]
                    teacher_completion_ids = self.teacher_tokenizer.encode(teacher_completion, add_special_tokens=False)

                    teacher_full_prompt = teacher_prompt + teacher_completion
                    teacher_prompt_length = len(teacher_prompt_ids)
                    teacher_logprobs = await get_teacher_logprobs(teacher_full_prompt, teacher_prompt_length, sample_idx)

                    # Align teacher logprobs to student tokens using fast byte-level alignment algorithm
                    aligned_teacher_logprobs = align_teacher_logprobs(
                        student_ids=student_response_ids,
                        student_tokenizer=self.tokenizer,
                        teacher_ids=teacher_completion_ids,
                        teacher_tokenizer=self.teacher_tokenizer,
                        teacher_logprobs=teacher_logprobs,
                        student_logprobs=student_logprobs,
                        reasoning_str=reasoning_str,
                        content_str=content_str,
                    )

                    # Visualize first sample for debugging alignment
                    if sample_idx == 0:
                        from rllm.trainer.distill import visualize_alignment

                        visualize_alignment(
                            student_ids=student_response_ids,
                            student_tokenizer=self.tokenizer,
                            teacher_ids=teacher_completion_ids,
                            teacher_tokenizer=self.teacher_tokenizer,
                            teacher_logprobs=teacher_logprobs,
                            student_logprobs=student_logprobs,
                            reasoning_str=reasoning_str,
                            content_str=content_str,
                            max_tokens=150,
                        )

                # reverse kl: teacher_logprob - student_logprob
                sample_advantages = [t_lp - s_lp for t_lp, s_lp in zip(aligned_teacher_logprobs, student_logprobs, strict=False)]

                advantages[sample_idx, :student_response_length] = torch.tensor(sample_advantages, dtype=torch.float32)

            except Exception as e:
                print(f"Error processing sample {sample_idx} for distillation: {e}")
                import traceback

                traceback.print_exc()
                batch.non_tensor_batch["is_valid"][sample_idx] = False  # drop the item from the batch

        await asyncio.gather(*[process_sample(i) for i in range(batch_size)])
        advantages *= response_mask.float()
        return advantages

    def generate_trajectories(self, batch, timing_raw=None, **kwargs):
        """
        Generates trajectories asynchronously using the agent execution engine's excute tasks method.
        Post-processing is done in the engine as well.

        Args:
            batch: The input batch for trajectory generation
            timing_raw: Dictionary to store timing information for profiling
            **kwargs: Additional arguments to pass to trajectory_generator

        Returns:
            list: List of collected processed trajectories
        """
        if timing_raw is None:
            timing_raw = {}

        with marked_timer("generate_trajectories", timing_raw, color="red"):
            coro = self.agent_execution_engine.execute_tasks_verl(batch, **kwargs)
            final_gen_batch_output = asyncio.run_coroutine_threadsafe(coro, self._loop).result()

        return final_gen_batch_output

    def _stepwise_advantage_broadcast(self, last_step_batch, non_last_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps within the same episode and trajectory.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants

        src_traj_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
        src_eps_ids = last_step_batch.non_tensor_batch["episode_ids"]
        src_steps = last_step_batch.non_tensor_batch["step_nums"]
        src_mask = last_step_batch.batch["response_mask"]
        src_advantages = last_step_batch.batch["advantages"]

        tgt_traj_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
        tgt_eps_ids = non_last_step_batch.non_tensor_batch["episode_ids"]
        tgt_mask = non_last_step_batch.batch["response_mask"]

        # Build id -> scalar advantage
        traj_ep_to_scalar_adv = {}
        for i, (traj_id, eps_id) in enumerate(zip(src_traj_ids, src_eps_ids, strict=False)):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

        # Create new tensor for non_last_step_batch with per-token assignment
        scalar_rows = torch.stack(
            [torch.full_like(tgt_mask[i], fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)], dtype=torch.float32) for i, (traj_id, eps_id) in enumerate(zip(tgt_traj_ids, tgt_eps_ids, strict=False))]
        )  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        non_last_step_batch.batch["advantages"] = final_advantage
        non_last_step_batch.batch["returns"] = final_advantage

    def _prepare_actor_update_batch(self, batch: DataProto, *, mini_batch_size: int, metrics: dict | None = None) -> DataProto:
        """Make partial token-mean minibatches transportable without training copies.

        Workflow prefix collapse and validity filtering produce variable row counts.
        The pinned EngineWorker requires a full local minibatch on every DP rank.
        Extend each existing rank shard separately: padding the global tail would
        regroup a 70-row, two-rank update into 38+32 real rows instead of 64+6.
        """
        worker_group = self.actor_rollout_wg
        if "actor" not in worker_group._dispatch_info:
            worker_group._dispatch_info["actor"] = worker_group._query_dispatch_info("actor")
        dp_mapping = worker_group._dispatch_info["actor"]
        dp_size = max(dp_mapping) + 1
        if len(dp_mapping) != worker_group.world_size or set(dp_mapping) != set(range(dp_size)):
            raise ValueError("Invalid actor data-parallel dispatch mapping")
        if mini_batch_size <= 0 or mini_batch_size % dp_size:
            raise ValueError("Actor mini_batch_size must be positive and divisible by its data-parallel size")
        if len(batch) == 0 or len(batch) % dp_size:
            raise ValueError("Actor batch must be nonempty and already divisible by its data-parallel size")

        local_rows = len(batch) // dp_size
        local_mini_batch_size = mini_batch_size // dp_size
        local_pad_size = (-local_rows) % local_mini_batch_size
        old_pad_mask = batch.non_tensor_batch.get("is_pad_step", np.zeros(len(batch), dtype=bool))
        if not local_pad_size and not np.any(old_pad_mask):
            return batch

        actor_config = self.config.actor_rollout_ref.actor
        if actor_config.loss_agg_mode != "token-mean":
            raise ValueError("Partial workflow actor minibatches require token-mean loss; padding other aggregations can change loss weights")
        if local_pad_size and actor_config.shuffle:
            raise ValueError("Partial workflow actor minibatches with shuffle=True cannot be padded without changing minibatch membership")

        shards = []
        for rank in range(dp_size):
            # Advanced selection also isolates the payload from the batch used for
            # episode metrics and critic updates. Never erase its original masks.
            shard = batch.select_idxs(np.arange(rank * local_rows, (rank + 1) * local_rows))
            shard, _ = pad_dataproto_to_divisor(shard, local_mini_batch_size)
            pad_mask = shard.non_tensor_batch.get("is_pad_step", np.zeros(len(shard), dtype=bool)).copy()
            pad_mask[local_rows:] = True
            shard.non_tensor_batch["is_pad_step"] = pad_mask
            for key in ("is_valid", "is_last_step"):
                if key in shard.non_tensor_batch:
                    shard.non_tensor_batch[key] = shard.non_tensor_batch[key].copy()
                    shard.non_tensor_batch[key][pad_mask] = False
            for key in ("response_mask", "advantages", "returns"):
                if key in shard.batch:
                    tensor = shard.batch[key].clone()
                    tensor[torch.as_tensor(pad_mask, device=tensor.device)] = 0
                    shard.batch[key] = tensor
            # Keep input/attention tokens for valid model forwards. The pinned
            # no-padding converter derives loss_mask directly from response_mask.
            shards.append(shard)

        for start in range(0, len(shards[0]), local_mini_batch_size):
            if not any(bool(shard.batch["response_mask"][start : start + local_mini_batch_size].any()) for shard in shards):
                raise ValueError("Workflow actor minibatch has no loss tokens across its data-parallel ranks")

        payload = DataProto.concat(shards)
        added_rows = local_pad_size * dp_size
        masked_rows = int(np.count_nonzero(payload.non_tensor_batch["is_pad_step"]))
        if metrics is not None:
            metrics.update(
                {
                    "batch/update_actor_prepad_size": len(batch),
                    "batch/update_actor_pad_size": added_rows,
                    "batch/update_actor_masked_pad_size": masked_rows,
                    "batch/update_actor_pad_divisor": mini_batch_size,
                    "batch/update_actor_dp_size": dp_size,
                }
            )
        print(f"[actor-batch] input_rows={len(batch)} added_transport_rows={added_rows} masked_transport_rows={masked_rows} dp_size={dp_size} global_mini_batch_size={mini_batch_size}", flush=True)
        return payload

    def _pad_workflow_ppo_update_batch(self, batch, *, worker_group, global_mini_batch_size, metrics=None, metric_prefix="update"):
        """Complete PPO worker minibatches without repeating any training target.

        Dropped workflow episodes can leave 62 rows of an expected 64. Padding
        only to two ranks leaves 31 rows per rank, while the pinned worker's
        iterator requires full 32-row minibatches. Strip any inference padding,
        then make a separate worker input divisible by both dispatch world size
        and the configured global minibatch. Synthetic rows retain executable
        inputs, but their action loss masks are zero for actor, critic and KL.
        """
        mini_batch_size = int(global_mini_batch_size)
        world_size = int(getattr(worker_group, "world_size", 1) or 1)
        if mini_batch_size <= 0 or world_size <= 0:
            raise ValueError("Workflow PPO update requires positive minibatch and worker sizes")
        original = self._remove_padding(batch)
        original_size = len(original)
        if not original_size:
            raise ValueError("Workflow PPO update has no valid rows after removing padding")
        if not original.batch["response_mask"].any():
            raise ValueError("Workflow PPO update has no action tokens after filtering")
        divisor = math.lcm(world_size, mini_batch_size)
        worker_batch, pad_size = pad_dataproto_to_divisor(original, divisor)
        # Keep metrics/metadata on the original batch independent of worker-only
        # padding, including when the transport returns its input unchanged.
        worker_batch = copy.copy(worker_batch)
        worker_batch.meta_info = copy.deepcopy(worker_batch.meta_info)
        if pad_size:
            response_mask = worker_batch.batch["response_mask"].clone()
            response_mask[original_size:] = 0
            worker_batch.batch["response_mask"] = response_mask
            worker_batch.non_tensor_batch["is_last_step"][original_size:] = False
            worker_batch.non_tensor_batch["is_pad_step"][original_size:] = True
            worker_batch.non_tensor_batch["is_valid"][original_size:] = False
        worker_batch.meta_info["global_token_num"] = worker_batch.batch["attention_mask"].sum(dim=-1).tolist()
        if metrics is not None:
            metrics[f"batch/{metric_prefix}_prepad_size"] = original_size
            metrics[f"batch/{metric_prefix}_pad_size"] = pad_size
            metrics[f"batch/{metric_prefix}_pad_divisor"] = divisor
        return worker_batch

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if hasattr(self, "rollout_wg") and self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        batch = self._remove_padding(batch)  # Remove any padded steps from the batch (just in case)
        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True
            batch.non_tensor_batch["is_valid"][idx] = False

        return batch

    def _remove_padding(self, batch):
        """Removes padded steps from the batch"""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
        return batch

    def shutdown(self):
        """A cleanup method to gracefully stop the background event loop."""
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
        if hasattr(self, "_loop") and self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join()

    def visualize_trajectory_last_step(self, tensor_batch, sample_idx=0, max_samples=1):
        """
        Visualize last steps from a workflow rollout using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        # Select only last steps if stepwise-advantage is enabled
        if "is_last_step" in tensor_batch.non_tensor_batch:
            is_last = tensor_batch.non_tensor_batch["is_last_step"]
            if is_last is not None and len(is_last) == len(tensor_batch):
                tensor_batch = tensor_batch[is_last]

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key="response_mask",
            reward_key="step_rewards" if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "per_step" else "traj_rewards",
            show_workflow_metadata=True,
        )
