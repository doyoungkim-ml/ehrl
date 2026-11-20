import asyncio
import gc
import json
import math
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

try:
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
except ModuleNotFoundError:
    from verl.workers.roles.utils.padding import left_right_2_no_padding, no_padding_2_padding

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm.trainer.utils import compute_token_level_kl_coefs, get_kl_schedule_summary
from rllm.trainer.utils.metric_normalization import add_actor_update_flop_metrics, normalize_actor_metric_keys
from rllm.trainer.verl.advantages import compute_advantage_with_overrides
from rllm.trainer.verl.padding import prepare_masked_actor_batch
from rllm.trainer.verl.rejection_gate import REWARD_EQUALITY_ATOL
from rllm.trainer.verl.rollout_correction import (
    ROLLOUT_LOG_PROB_MASK_KEY,
    ROLLOUT_LOG_PROBS_KEY,
    apply_rollout_correction,
    pad_rollout_rows,
    rollout_correction_settings,
    trajectory_rollout_tensors,
)
from rllm.utils.compute_pass_at_k import (
    compute_pass_at_k_max_reward_from_step_rewards,
    compute_turn_level_metrics,
    compute_validation_turn_metrics,
    normalize_validation_data_source,
)


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
        )

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}
        self._current_step_rewards_list = None
        self._current_step_reward_tasks = None
        self._logged_turn_level_kl_summary = False

        assert self.config.actor_rollout_ref.hybrid_engine, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"
        # Validate before any worker starts; trajectories then carry sampler log-probs.
        self._rollout_correction = rollout_correction_settings(self.config)

        if self.config.rllm.stepwise_advantage.enable:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    @staticmethod
    def _phase_log_value(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _emit_phase_log(self, phase: str, event: str, **fields) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        rendered = " ".join(f"{key}={self._phase_log_value(value)}" for key, value in fields.items() if value is not None)
        suffix = f" {rendered}" if rendered else ""
        print(f"[phase {timestamp}] {phase} {event}{suffix}", flush=True)

    @staticmethod
    def _trim_host_memory() -> None:
        """Best-effort release of Python/glibc host memory between large PPO steps."""
        gc.collect()
        if os.name != "posix":
            return
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            malloc_trim = getattr(libc, "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except Exception:
            # malloc_trim is Linux/glibc-specific; cleanup should never affect training.
            pass

    @contextmanager
    def _phase_scope(self, phase: str, **fields):
        start = time.monotonic()
        self._emit_phase_log(phase, "start", **fields)
        try:
            yield
        except Exception as exc:
            self._emit_phase_log(
                phase,
                "error",
                duration_s=time.monotonic() - start,
                error=type(exc).__name__,
                **fields,
            )
            raise
        else:
            self._emit_phase_log(
                phase,
                "end",
                duration_s=time.monotonic() - start,
                **fields,
            )

    def _maybe_sleep_rollout_replicas_after_training_generation(self, timing_raw=None) -> None:
        """Sleep hybrid rollout replicas after training generation when free-cache mode is enabled.

        In the naive checkpoint-engine path, `verl`'s trainer-side `update_weights()`
        expects the rollout servers to already be in sleep mode before it resumes
        weights/kv-cache. The workflow and SDK trainers do this explicitly; the
        direct agent PPO path needs the same hand-off.
        """
        if not self.config.actor_rollout_ref.rollout.free_cache_engine:
            return
        if timing_raw is None:
            timing_raw = {}
        with marked_timer("sleep_replicas", timing_raw):
            getattr(self, "checkpoint_manager", None) and self.checkpoint_manager.sleep_replicas()

    def _maybe_restore_rollout_after_skipped_batch(self, timing_raw=None) -> None:
        """Restore unchanged weights/cache when a skipped batch bypasses PPO sync.

        Keep the normal pre-training sleep for GPU memory headroom. HYBRID
        level-2 sleep releases weights too, so a bare replica wake-up is not
        enough: use the same checkpoint-manager handoff as a completed update.
        global_steps is the next optimizer step here; do not publish that
        version or advance any training counters for a rejected batch.
        """
        if not self.config.actor_rollout_ref.rollout.free_cache_engine:
            return
        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        if checkpoint_manager is None:
            return
        if timing_raw is None:
            timing_raw = {}
        print(f"Skipped training batch at step {self.global_steps}; restoring unchanged rollout weights/cache.", flush=True)
        with marked_timer("restore_rollout_after_skip", timing_raw):
            checkpoint_manager.update_weights(self.global_steps - 1)

    def init_workers(self):
        super().init_workers()

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = engine_args.pop("n_parallel_agents", None) or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        print(f"n_parallel_agents: {n_parallel_agents}")

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            trajectory_hard_timeout=self.config.rllm.agent.get("trajectory_hard_timeout", 7200),
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        assert self.agent_class is not None and self.env_class is not None, "Agent and environment classes must be provided"
        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.rllm.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.rllm.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def _consume_current_step_reward_metrics(self, logger, rollout_uids) -> dict:
        current_step_rewards = self._current_step_rewards_list
        current_step_reward_tasks = getattr(self, "_current_step_reward_tasks", None)
        self._current_step_rewards_list = None
        self._current_step_reward_tasks = None

        if not current_step_rewards or not any(len(step_rewards) > 0 for step_rewards in current_step_rewards):
            return {}

        gamma = self.config.algorithm.get("gamma", 1.0)
        cutoff_mode = self.config.rllm.get("turn_level_cutoff_mode", "max")
        max_steps = self.config.rllm.agent.get("max_steps", None)
        if current_step_reward_tasks is not None and len(current_step_reward_tasks) == len(current_step_rewards):
            current_task_groups = list(current_step_reward_tasks)
        else:
            current_task_groups = rollout_uids[: len(current_step_rewards)] if rollout_uids else None
        train_pass_at_k_max_reward = compute_pass_at_k_max_reward_from_step_rewards(
            current_step_rewards,
            tasks=current_task_groups,
        )

        metrics = compute_turn_level_metrics(
            current_step_rewards,
            logger=logger,
            gamma=gamma,
            step=self.global_steps,
            cutoff_mode=cutoff_mode,
            max_steps=max_steps,
            prefix="turn",
            table_prefix="train",
            table_extra_values={
                "pass_at_k_max_reward": train_pass_at_k_max_reward,
            },
        )
        metrics["train/pass_at_k_max_reward"] = train_pass_at_k_max_reward
        return metrics

    def _maybe_attach_turn_level_kl(self, batch: DataProto, metrics: dict) -> None:
        turn_level_kl_config = self.config.rllm.get("turn_level_kl", {})
        if not turn_level_kl_config.get("enable", False):
            return
        if "step_indices" not in batch.non_tensor_batch:
            return

        step_indices = batch.non_tensor_batch["step_indices"]
        response_mask = batch.batch.get("response_mask", batch.batch["attention_mask"][:, -self.config.data.max_response_length :])
        response_lengths = response_mask.sum(dim=-1).cpu().numpy()

        token_kl_coefs = compute_token_level_kl_coefs(
            step_indices=step_indices,
            response_lengths=response_lengths,
            max_response_length=self.config.data.max_response_length,
            schedule=turn_level_kl_config.get("schedule", "linear"),
            kl_min=turn_level_kl_config.get("kl_min", 0.0),
            kl_step=turn_level_kl_config.get("kl_step", 0.1),
            kl_max=turn_level_kl_config.get("kl_max", 1.0),
            max_turns=turn_level_kl_config.get("max_turns", 5),
        )
        batch.batch["token_level_kl_coefs"] = token_kl_coefs.to(batch.batch["attention_mask"].device)

        if not self._logged_turn_level_kl_summary:
            print(
                get_kl_schedule_summary(
                    schedule=turn_level_kl_config.get("schedule", "linear"),
                    kl_min=turn_level_kl_config.get("kl_min", 0.0),
                    kl_step=turn_level_kl_config.get("kl_step", 0.1),
                    kl_max=turn_level_kl_config.get("kl_max", 1.0),
                    max_turns=turn_level_kl_config.get("max_turns", 5),
                )
            )
            self._logged_turn_level_kl_summary = True

        unique_steps = np.unique(step_indices)
        for step_idx in unique_steps[: min(5, len(unique_steps))]:
            step_mask = step_indices == step_idx
            if step_mask.any():
                metrics[f"kl/turn_{step_idx}_coef"] = token_kl_coefs[step_mask].mean().item()

    @staticmethod
    def _slice_padded_batch_tensor(tensor: torch.Tensor, original_batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == original_batch_size:
            return tensor
        return tensor[:original_batch_size]

    def _release_critic_device_memory(self) -> None:
        """Empty the CUDA allocator cache on the critic workers (new worker impl only).

        See rllm.trainer.verl.train_agent_ppo.CriticTrainingWorker. No-op on the legacy
        CriticWorker, which manages its own offload/empty_cache.
        """
        if getattr(self, "use_legacy_worker_impl", "auto") != "disable":
            return
        fn = getattr(self.critic_wg, "release_device_memory", None)
        if fn is not None:
            fn()

    def _pad_batch_for_critic(self, batch: DataProto, *, metrics: dict | None = None, metric_prefix: str = "critic", size_divisor: int | None = None) -> tuple[DataProto, int]:
        """Pad a DataProto to a multiple of the critic worker group's world size
        (and of `size_divisor`, when given).

        RayPPOTrainer._compute_values / _update_critic hand the batch to the critic
        worker group, whose dispatch chunks it across DP ranks and needs the row
        count divisible by world_size. The critic UPDATE additionally requires the
        batch to be a multiple of the critic's global ppo mini-batch
        (critic.ppo_mini_batch_size * rollout.n; verl asserts
        `batch % mini_batch == 0` per rank): with rejection sampling the kept batch
        can be a single 8-trajectory group -> job 16339596 died on
        `AssertionError: 4 % 32 != 0`. Pass size_divisor=<global mini-batch> for
        the update so the batch is padded with duplicate rows, exactly what the
        actor update already does via _build_no_padding_batch_td(size_divisor=...).
        Returns (padded_batch, pad_size); pad_size is 0 when no padding was needed.
        """
        # The new EngineWorker forward (verl transformer_impl.prepare_model_inputs) reads
        # micro_batch["temperature"] unconditionally, value model included. verl's own fit
        # loop sets batch.meta_info["temperature"] before compute_values and
        # DataProto.to_tensordict() carries meta_info into the TensorDict; this trainer
        # only sets it later, inside the old_log_prob block (after compute_values) —
        # job 16317215 died on exactly that KeyError. Set it here so both critic calls see it.
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        world_size = getattr(self.critic_wg, "world_size", 0) or 0
        divisor = size_divisor or 1
        if world_size > 1:
            divisor = math.lcm(divisor, world_size)
        if divisor > 1 and len(batch) % divisor != 0:
            batch_for_worker, pad_size = pad_dataproto_to_divisor(batch, divisor)
            batch_for_worker.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            if metrics is not None:
                metrics[f"batch/{metric_prefix}_prepad_size"] = len(batch)
                metrics[f"batch/{metric_prefix}_pad_size"] = pad_size
                metrics[f"batch/{metric_prefix}_pad_divisor"] = divisor
            return batch_for_worker, pad_size
        return batch, 0

    def _build_no_padding_batch_td(
        self,
        batch: DataProto,
        worker_group,
        *,
        non_tensor_metadata: dict,
        metrics: dict | None = None,
        metric_prefix: str | None = None,
        size_divisor: int | None = None,
    ) -> tuple[TensorDict, int]:
        batch_for_worker = batch
        if metric_prefix == "update_actor" and batch.meta_info.get("mask_training_padding", False):
            batch_for_worker = prepare_masked_actor_batch(
                batch,
                worker_group=worker_group,
                mini_batch_size=size_divisor,
                shuffle=self.config.actor_rollout_ref.actor.shuffle,
                metrics=metrics,
            )
            batch_td = left_right_2_no_padding(batch_for_worker.to_tensordict())
            tu.assign_non_tensor(batch_td, **non_tensor_metadata)
            return batch_td, len(batch_for_worker) - len(batch)
        world_size = getattr(worker_group, "world_size", 0) or 0
        divisor = size_divisor or 1
        if world_size > 1:
            divisor = math.lcm(divisor, world_size)

        if divisor > 1 and len(batch) % divisor != 0:
            batch_for_worker, pad_size = pad_dataproto_to_divisor(batch, divisor)
            if metrics is not None:
                key = metric_prefix or "worker_group_call"
                metrics[f"batch/{key}_prepad_size"] = len(batch)
                metrics[f"batch/{key}_pad_size"] = pad_size
                metrics[f"batch/{key}_pad_divisor"] = divisor
        else:
            pad_size = 0

        batch_td = batch_for_worker.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, **non_tensor_metadata)
        return batch_td, pad_size

    def _augment_accepted_training_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Optional algorithm records after rejection, before probability scoring.

        The default preserves the original training population. Critic-free
        algorithms can append independently scored records here without exposing
        them to the ordinary reward-group rejection gate. Subclasses own pending
        record cleanup when a generation round is skipped.
        """
        return batch, {}

    def _compute_training_advantages(self, batch: DataProto) -> DataProto:
        return compute_advantage_with_overrides(
            batch,
            compute_fn=compute_advantage,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

    def _iter_training_epoch(self):
        return iter(self.train_dataloader)

    def _load_checkpoint(self):
        from rllm.utils.training_reward import restore_training_reward

        result = super()._load_checkpoint()
        loaded = self.config.trainer.resume_mode != "disable" and not (result == 0 and self.global_steps == 0)
        restore_training_reward(self, loaded)
        return result

    def _save_checkpoint(self):
        from rllm.utils.training_reward import save_training_reward

        # Write monitoring state before VeRL can publish its checkpoint marker.
        save_training_reward(self)
        return super()._save_checkpoint()

    def _get_training_start_epoch(self):
        iterator_finished = getattr(self, "_dataloader_resume_iterator_finished", None)
        if hasattr(self, "_training_epoch"):
            # An already-finished StatefulDataLoader resets itself on iter().
            return self._training_epoch + int(iterator_finished is True)
        # Legacy checkpoints lack a data epoch. This estimate is exact only
        # when no batches were skipped. A restored pre-StopIteration boundary
        # needs an empty tail iteration before advancing to the next epoch.
        steps_per_epoch = len(self.train_dataloader)
        completed_steps = self.global_steps - int(iterator_finished is False)
        return max(0, completed_steps // steps_per_epoch) if steps_per_epoch else 0

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from rllm.utils.run_state import finish_data_exhausted, prepare_training_run
        from rllm.utils.tracking import Tracking
        from rllm.utils.training_reward import complete_training_reward_epoch, initialize_training_reward

        prepare_training_run(self.config)
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        start_epoch = self._get_training_start_epoch()
        initialize_training_reward(self, start_epoch)
        getattr(self, "checkpoint_manager", None) and self.checkpoint_manager.update_weights(self.global_steps)

        if self.global_steps >= self.total_training_steps and not self.config.trainer.get("val_only", False):
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
            val_metrics = self._validate_agent(logger=logger, step=self.global_steps)
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            # Optimizer steps can diverge from consumed batches after rejection.
            # ACE checkpoints persist the actual data epoch, not that estimate.
            self._training_epoch = epoch
            self._training_reward_tracker.begin_epoch(epoch)
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self._iter_training_epoch():
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                rollout_uids = list(batch.non_tensor_batch["uid"])

                metrics = {}
                timing_raw = {}

                with marked_timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    if self.config.rllm.stepwise_advantage.enable:
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        metrics.update(final_gen_batch_output.meta_info.pop("training_reward_metrics", {}))
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                        metrics.update(self._consume_current_step_reward_metrics(logger=logger, rollout_uids=rollout_uids))
                    else:
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                        batch = batch.union(final_gen_batch_output)
                        metrics.update(generate_metrics)
                        metrics.update(self._consume_current_step_reward_metrics(logger=logger, rollout_uids=rollout_uids))

                    self._training_reward_tracker.observe(metrics, expected_count=len(rollout_uids))
                    self._maybe_sleep_rollout_replicas_after_training_generation(timing_raw=timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            # Go through RayPPOTrainer._compute_values, which dispatches on
                            # use_legacy_worker_impl: the new EngineWorker critic (verl default
                            # "disable") exposes infer_batch on a no-padding TensorDict, the
                            # legacy CriticWorker exposes compute_values on a DataProto. The
                            # batch is not world-size padded on this path (only the actor calls
                            # pad, via _build_no_padding_batch_td), so pad/slice here the same way.
                            if getattr(self, "use_legacy_worker_impl", "auto") == "disable":
                                batch_for_critic, critic_pad_size = self._pad_batch_for_critic(batch, metrics=metrics, metric_prefix="compute_values")
                                values = self._compute_values(batch_for_critic)
                                if critic_pad_size:
                                    values = values.select_idxs(list(range(len(batch))))
                                batch = batch.union(values)
                                self._release_critic_device_memory()
                            else:
                                # Legacy CriticWorker: unchanged upstream call.
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Rejection sampling based on rewards. Keep the legacy
                        # threshold gate by default; graded-reward runs can opt
                        # into the workflow trainer's reward-variance rule.
                        rejection_mode = getattr(self.config.rllm.rejection_sample, "mode", "correct")
                        if rejection_mode not in ("correct", "variance"):
                            raise ValueError(f"Unknown rejection_sample.mode: {rejection_mode!r} (expected 'correct' or 'variance')")
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                            if rejection_mode == "variance":
                                # Match the shared gate's float64 comparison and
                                # tolerance without changing the solve_* census.
                                group_rewards = uid_rewards.to(torch.float64)
                                valid_mask[uid_mask] = bool(group_rewards.max() - group_rewards.min() > REWARD_EQUALITY_ATOL)

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.rllm.rejection_sample.enable:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.rllm.stepwise_advantage.enable:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                self._maybe_restore_rollout_after_skipped_batch(timing_raw=timing_raw)
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    self._maybe_restore_rollout_after_skipped_batch(timing_raw=timing_raw)
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    self._maybe_restore_rollout_after_skipped_batch(timing_raw=timing_raw)
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        batch, additional_metrics = self._augment_accepted_training_batch(batch)
                        metrics.update(additional_metrics)
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            if self.use_legacy_worker_impl == "disable":
                                original_batch_size = len(batch)
                                batch_td, _ = self._build_no_padding_batch_td(
                                    batch,
                                    self.actor_rollout_wg,
                                    non_tensor_metadata={
                                        "calculate_entropy": True,
                                        "compute_loss": False,
                                        "temperature": self.config.actor_rollout_ref.rollout.temperature,
                                    },
                                    metrics=metrics,
                                    metric_prefix="old_log_prob",
                                )
                                old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch_td)
                                entropy = tu.get(old_log_prob_output, "entropy")
                                log_probs = tu.get(old_log_prob_output, "log_probs")
                                entropy = no_padding_2_padding(entropy, batch_td)
                                log_probs = no_padding_2_padding(log_probs, batch_td)
                                entropy = self._slice_padded_batch_tensor(entropy, original_batch_size)
                                log_probs = self._slice_padded_batch_tensor(log_probs, original_batch_size)
                                old_log_prob = DataProto.from_tensordict(
                                    tu.get_tensordict(
                                        {
                                            "old_log_probs": log_probs.float(),
                                            "entropys": entropy.float(),
                                        }
                                    )
                                )
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
                                batch, correction_metrics = apply_rollout_correction(batch, rollout_correction)
                                metrics.update(correction_metrics)
                            elif "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw):
                                if self.use_legacy_worker_impl == "disable":
                                    original_batch_size = len(batch)
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
                                    ref_lp = self._slice_padded_batch_tensor(ref_lp, original_batch_size)
                                    ref_log_prob = DataProto.from_tensordict(tu.get_tensordict({"ref_log_prob": ref_lp.float()}))
                                else:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.rllm.stepwise_advantage.enable:
                            if self.config.rllm.stepwise_advantage.mode == "per_step":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.rllm.stepwise_advantage.mode} not supported")

                        # compute advantages, executed on the driver process
                        batch = self._compute_training_advantages(batch)

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    if "ace_component" in batch.non_tensor_batch:
                        from rllm.trainer.verl.role_credit import role_credit_metrics

                        metrics.update(role_credit_metrics(batch))

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # Ragged mixed-record updates partition and mask-pad at the
                    # actor boundary. The inherited equal-size rebalance would
                    # reject their logical row count before that transport step.
                    if not batch.meta_info.get("mask_training_padding", False):
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            # Same impl dispatch + padding as compute_values above. Padded rows
                            # are duplicates of real rows (pad_dataproto_to_divisor), the same
                            # approximation the actor update already makes on this path.
                            if getattr(self, "use_legacy_worker_impl", "auto") == "disable":
                                critic_mini_batch_size = self.config.critic.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
                                batch_for_critic, _ = self._pad_batch_for_critic(batch, metrics=metrics, metric_prefix="update_critic", size_divisor=critic_mini_batch_size)
                                critic_output = self._update_critic(batch_for_critic)
                                self._release_critic_device_memory()
                            else:
                                # Legacy CriticWorker: unchanged upstream call.
                                critic_output = self.critic_wg.update_critic(batch)
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
                        self._maybe_attach_turn_level_kl(batch, metrics)

                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        if self.use_legacy_worker_impl == "disable":
                            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
                            update_batch, _ = self._build_no_padding_batch_td(
                                batch,
                                self.actor_rollout_wg,
                                non_tensor_metadata={
                                    "calculate_entropy": (self.config.actor_rollout_ref.actor.entropy_coeff != 0.0),
                                    "global_batch_size": ppo_mini_batch_size,
                                    "mini_batch_size": ppo_mini_batch_size,
                                    "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
                                    "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
                                    "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
                                },
                                metrics=metrics,
                                metric_prefix="update_actor",
                                size_divisor=ppo_mini_batch_size,
                            )
                        else:
                            update_batch = batch

                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(update_batch)

                        if (
                            self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0
                        ) or (self.config.trainer.get("save_at_end", False) and self.global_steps + 1 >= self.total_training_steps):
                            with marked_timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw):
                            getattr(self, "checkpoint_manager", None) and self.checkpoint_manager.update_weights(self.global_steps)

                        if isinstance(actor_output, TensorDict):
                            actor_output_metrics = normalize_actor_metric_keys(reduce_metrics(tu.get(actor_output, "metrics")))
                        else:
                            actor_output_metrics = normalize_actor_metric_keys(reduce_metrics(actor_output.meta_info["metrics"]))
                        metrics.update(actor_output_metrics)

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    with marked_timer("testing", timing_raw):
                        val_metrics: dict = self._validate_agent(logger=logger, step=self.global_steps)
                    metrics.update(val_metrics)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(
                    data=add_actor_update_flop_metrics(
                        metrics,
                        n_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                    ),
                    step=self.global_steps,
                )

                batch = None
                final_gen_batch_output = None
                generate_metrics = None
                values = None
                reward_tensor = None
                old_log_prob = None
                ref_log_prob = None
                actor_output = None
                update_batch = None
                self._trim_host_memory()

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps and not self.config.trainer.get("val_only", False):
                    # perform validation after training
                    if self.val_reward_fn is not None and self.config.trainer.get("val_after_train", True):
                        val_metrics = self._validate_agent(logger=logger, step=self.global_steps)
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

            # Reached only after StopIteration, never a partial max-step return.
            complete_training_reward_epoch(self)

        finish_data_exhausted(self)

    def _validate_agent(self, logger=None, step=0):
        step_rewards_by_source = {}
        tasks_by_source = {}

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            self.init_envs_and_agents(test_batch)

            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                # for validation, we only need the last step
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            test_batch = test_batch.union(test_output_gen_batch)
            data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch.batch["input_ids"]))
            tasks = test_batch.non_tensor_batch.get("extra_info", [None] * len(test_batch.batch["input_ids"]))

            current_step_rewards = self._current_step_rewards_list
            if current_step_rewards is None:
                reward_tensor = test_batch.batch["token_level_scores"]
                current_step_rewards = [[reward.item()] for reward in reward_tensor.sum(-1).cpu()]

            for data_source, task, step_rewards in zip(data_sources, tasks, current_step_rewards, strict=False):
                normalized_source = normalize_validation_data_source(data_source)
                if normalized_source not in step_rewards_by_source:
                    step_rewards_by_source[normalized_source] = []
                    tasks_by_source[normalized_source] = []
                step_rewards_by_source[normalized_source].append(step_rewards)
                tasks_by_source[normalized_source].append(task)

            self._current_step_rewards_list = None

        metric_dict = {}
        cutoff_mode = self.config.rllm.get("turn_level_cutoff_mode", "max")
        max_steps = self.config.rllm.agent.get("max_steps", None)

        all_step_rewards = []
        for data_source, step_rewards_list in step_rewards_by_source.items():
            tasks = tasks_by_source[data_source]
            all_step_rewards.extend(step_rewards_list)
            metric_dict.update(
                compute_validation_turn_metrics(
                    step_rewards_list=step_rewards_list,
                    tasks=tasks,
                    cutoff_mode=cutoff_mode,
                    max_steps=max_steps,
                    data_source=data_source,
                )
            )

        if logger is not None and all_step_rewards:
            compute_turn_level_metrics(
                all_step_rewards,
                logger=logger,
                gamma=1.0,
                step=step,
                cutoff_mode=cutoff_mode,
                max_steps=max_steps,
                prefix="val_turn",
                table_prefix="val",
            )

        return metric_dict

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        from rllm.utils.training_reward import raw_reward_metrics

        if timing_raw is None:
            timing_raw = {}
        with marked_timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.async_rollout_mode:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                raise ValueError("Only async rollout mode is supported")
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        # Rollout totals: ALL tokens generated this step (pre-rejection)
        def _sum_token(traj_list, key):
            total = 0
            for t in traj_list:
                m = t.get("metrics", {}) if isinstance(t, dict) else {}
                v = m.get(key, 0)
                if v is not None and v > 0:
                    total += int(v)
            return total

        rollout_metrics = {
            "rollout/total_prompt_tokens": _sum_token(trajectories, "total_prompt_tokens"),
            "rollout/total_completion_tokens": _sum_token(trajectories, "total_completion_tokens"),
            "rollout/total_gradient_update_tokens": _sum_token(trajectories, "total_gradient_update_tokens"),
            "rollout/total_trajectories": len(trajectories),
        }
        rollout_metrics.update(raw_reward_metrics(trajectories))

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
            metrics.update(rollout_metrics)
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        from rllm.utils.training_reward import raw_reward_metrics

        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with marked_timer("collect_trajectory", timing_raw):
            steps = []
            gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
            for _, trajectory in enumerate(gen_seq_generator):
                steps.append(trajectory)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])
        reward_metrics = raw_reward_metrics(steps, expected_count=len(uids))

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids)
            final_gen_batch_output.meta_info["training_reward_metrics"] = reward_metrics
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        metrics = {}
        step_rewards_list = []
        traj_log_meta = []  # per-trajectory id/reward/outcome for enriched & structured logs

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, (
                f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty.Please check make sure environment is working and the config is correct."
            )
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            chat_completions.append(traj["chat_completions"])
            traj_metrics.append(traj["metrics"])
            step_rewards_list.append(traj.get("step_rewards", []))
            traj_log_meta.append(
                {
                    "task_idx": traj.get("idx"),
                    "reward": traj["trajectory_reward"],
                    "all_passed": traj.get("all_passed"),
                    "termination_reason": traj.get("termination_reason"),
                    **({"terminal_feedback": traj["terminal_feedback"]} if isinstance(traj.get("terminal_feedback"), dict) else {}),
                    **{key: value for key, value in traj.items() if key.startswith("execution_") or key in {"execute_retry_attempted", "limit_termination_reason"}},
                }
            )

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
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

        # Add token tracking metrics from trajectory info
        token_metrics = {
            "total_prompt_tokens": [],
            "total_completion_tokens": [],
            "total_gradient_update_tokens": [],
        }
        for traj in trajectories:
            metrics_dict = traj.get("metrics", {})
            for tm_key in token_metrics:
                if tm_key in metrics_dict:
                    token_metrics[tm_key].append(metrics_dict[tm_key])

        for tm_key, v_list in token_metrics.items():
            if v_list:
                v_list = np.array(v_list)
                metrics.update(
                    {
                        f"token/{tm_key}_mean": v_list.mean(),
                        f"token/{tm_key}_min": v_list.min(),
                        f"token/{tm_key}_max": v_list.max(),
                    }
                )

        # Log how trajectories ended: for each termination reason (e.g. ENV_DONE),
        # the fraction of trajectories in the batch that ended that way.
        from collections import Counter as _Counter

        _term_counts = _Counter((m.get("termination_reason") or "unknown") for m in traj_log_meta)
        _n_traj = max(1, len(traj_log_meta))
        for _reason, _cnt in _term_counts.items():
            metrics[f"batch/term_{_reason}"] = _cnt / _n_traj

        # Persist trajectory chat_completions; trajectory_log_dir/trajectory_log_format
        # are optional ("enriched" adds per-trajectory outcome metadata to each line).
        tcfg = self.config.trainer
        managed_run = os.environ.get("RLLM_RUN_DIR")
        log_root = tcfg.get("trajectory_log_dir", None) or (os.path.join(managed_run, "episodes") if managed_run else tcfg.default_local_dir)
        log_format = tcfg.get("trajectory_log_format", "raw")
        cc_dir = os.path.join(log_root, "chat_completions")
        os.makedirs(cc_dir, exist_ok=True)
        with open(os.path.join(cc_dir, f"{self.global_steps}.jsonl"), "w") as f:
            for chat_completion, meta in zip(chat_completions, traj_log_meta, strict=False):
                if log_format == "enriched":
                    f.write(json.dumps({**meta, "chat_completions": chat_completion}) + "\n")
                else:
                    f.write(json.dumps(chat_completion) + "\n")
        # Optional per-task dump: one JSON per (task, step) holding all rollout.n
        # samples for that task. One file per task per step -> off by default.
        if tcfg.get("log_trajectories_structured", False):
            by_task: dict = {}
            for chat_completion, meta in zip(chat_completions, traj_log_meta, strict=False):
                by_task.setdefault(meta["task_idx"], []).append({**meta, "chat_completions": chat_completion})
            for task_idx, trajs in by_task.items():
                task_dir = os.path.join(log_root, "trajectories", f"task_{task_idx}")
                os.makedirs(task_dir, exist_ok=True)
                with open(os.path.join(task_dir, f"policy_step{self.global_steps:06d}.json"), "w") as f:
                    json.dump(
                        {"step": self.global_steps, "task_idx": task_idx, "num_samples": len(trajs), "trajectories": trajs},
                        f,
                        indent=2,
                    )

        # ACE transforms policy and auxiliary roles together or in separate
        # calls before concatenation. Use the same configured widths for every
        # call; the engine may already have restored the policy generation caps.
        max_prompt_length = self.config.data.max_prompt_length
        max_response_length = self.config.data.max_response_length
        component_token_limits = getattr(self, "component_token_limits", None)
        if component_token_limits:
            max_prompt_length = max(max_prompt_length, max(limits["max_prompt_length"] for limits in component_token_limits.values()))
            max_response_length = max(max_response_length, max(limits["max_response_length"] for limits in component_token_limits.values()))
            for name, token_rows, width in (
                ("prompt", all_initial_tokens_list, max_prompt_length),
                ("response", all_response_tokens_list, max_response_length),
            ):
                actual_length = max(len(tokens) for tokens in token_rows)
                if actual_length > width:
                    raise ValueError(f"ACE trajectory {name} length {actual_length} exceeds configured batch width {width}; refusing to truncate generated tokens")

        # left pad prompts
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_initial_tokens_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_response_tokens_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token (e.g., eos token)
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, score in enumerate(traj_scores):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= score_batch.shape[1]:
                score_batch[i, resp_len - 1] = score

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "response_mask": traj_mask,
        }
        if getattr(self, "_rollout_correction", None) is not None:
            # Align sampler log-probs with the right-padded, capped response tokens.
            rollout_rows = [trajectory_rollout_tensors(traj, len(traj["response_tokens"])) for traj in trajectories]
            tensor_batch[ROLLOUT_LOG_PROBS_KEY] = pad_rollout_rows([row[0] for row in rollout_rows], max_response_length, torch.float32)
            tensor_batch[ROLLOUT_LOG_PROB_MASK_KEY] = pad_rollout_rows([row[1] for row in rollout_rows], max_response_length, torch.long) * (traj_mask > 0).long()

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))
        self._current_step_rewards_list = step_rewards_list if step_rewards_list and any(len(sr) > 0 for sr in step_rewards_list) else None
        self._current_step_reward_tasks = None

        if getattr(self, "ace_config", {}).get("log_role_credit", False):
            non_tensors = {"ace_component": np.array([traj.get("_component_type", "policy") for traj in trajectories], dtype=object)}
            return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensors), metrics
        return DataProto.from_dict(tensors=tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="response_mask"):
        """
        Visualize the trajectory from tensor_batch using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key=mask_key,
            reward_key="token_level_scores",
            show_workflow_metadata=False,
        )

    def generate_agent_trajectories_async(
        self,
        timing_raw=None,
        meta_info=None,
        mode="Token",
        **trajectory_generator_kwargs,
    ):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()
        done_sentinel = object()
        error_sentinel = object()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(
                    timing_raw=timing_raw,
                    mode=mode,
                    meta_info=meta_info,
                    **trajectory_generator_kwargs,
                ):
                    queue.put(item)

            try:
                asyncio.run(consume())
            except Exception as exc:  # noqa: BLE001
                queue.put((error_sentinel, exc, traceback.format_exc()))
            finally:
                queue.put(done_sentinel)

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is done_sentinel:
                break
            if isinstance(item, tuple) and len(item) == 3 and item[0] is error_sentinel:
                _, exc, formatted_traceback = item
                raise RuntimeError(formatted_traceback) from exc
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray):
        from verl.utils.torch_functional import pad_sequence_to_length

        overlong_filter = self.config.rllm.agent.get("overlong_filter", False)
        overlong_reasons = {"TRUNCATION", "MAX_STEPS", "TIMEOUT"}

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        all_steps_step_indices = []  # 0-indexed step index within each trajectory
        all_steps_masked_out = []  # whether this step should be masked out due to overlong filter
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]
            termination_reason = episode.get("limit_termination_reason") or episode.get("termination_reason")

            # Mask out overlong trajectories
            masked_out = overlong_filter and termination_reason in overlong_reasons

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])
            all_steps_step_indices.extend([i for i in range(len(episode_steps))])
            all_steps_masked_out.extend([masked_out for _ in range(len(episode_steps))])

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_prompts_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_responses_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = attention_mask[:, max_prompt_length:]
        # apply overlong filter by zeroing out masked trajectories
        if overlong_filter:
            overlong_mask = torch.tensor(all_steps_masked_out, dtype=torch.bool).unsqueeze(1)
            traj_mask = traj_mask * (~overlong_mask).long()

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token of each step
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                resp_len = response_lengths[step_index]
                if resp_len > 0 and resp_len <= score_batch.shape[1]:
                    score_batch[step_index, resp_len - 1] = traj_score
                    mc_return_batch[step_index, resp_len - 1] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "response_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
            "step_indices": np.array(all_steps_step_indices),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)

        step_rewards_list = []
        for episode in steps:
            step_rewards = episode.get("step_rewards", [])
            if not step_rewards:
                step_rewards = episode.get("mc_returns", [])
            step_rewards_list.append([float(reward) for reward in step_rewards])

        self._current_step_rewards_list = step_rewards_list if step_rewards_list and any(len(sr) > 0 for sr in step_rewards_list) else None
        self._current_step_reward_tasks = None
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        if batch.meta_info.get("mask_training_padding", False):
            # Inference dispatch pads/slices privately; actor update constructs
            # its own masked payload. Metrics retain the actual algorithm rows.
            return batch
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
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            if "is_last_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_last_step"][idx] = False
            if "is_pad_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch

    def shutdown(self):
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
