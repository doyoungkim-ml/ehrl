"""Original ERL collection bound to the shared VeRL optimization backend.

The executable reference is Microsoft's Hotpot full-objective recipe at the
package UPSTREAM_COMMIT. This trainer has no ACE inheritance or dependencies.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from verl import DataProto

from rllm.algorithms.erl import UPSTREAM_COMMIT
from rllm.algorithms.erl.agent import PromptOverrideAgent
from rllm.algorithms.erl.checkpoint import load_state, preserve_controller_rng, save_state
from rllm.algorithms.erl.controller import ERLController, UpdaterPromptOverflow
from rllm.algorithms.erl.replay import pack_replay_tensors, validate_replay_trajectory
from rllm.algorithms.erl.upstream import DEFAULT_UPDATER_SYSTEM_PROMPT, extract_prompt_from_response
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from rllm.trainer.verl.rollout_correction import usable_step_log_probs
from rllm.utils.training_reward import raw_reward_metrics


def _import(path):
    module, _, attribute = path.rpartition(".")
    return getattr(importlib.import_module(module), attribute)


class OriginalERLAgentPPOTrainer(AgentPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.erl_config = dict(self.config.rllm.get("erl", {}))
        if self.erl_config.get("upstream_commit", UPSTREAM_COMMIT) != UPSTREAM_COMMIT:
            raise ValueError("ERL upstream_commit does not match the implemented reference")
        if self.erl_config.get("validation_mode", "plain") != "plain":
            raise ValueError("Original ERL supports plain validation only")
        if self.use_critic or self.config.rllm.stepwise_advantage.enable:
            raise ValueError("Original ERL requires trajectory GRPO/RAFT without a critic")
        if self.config.rllm.rejection_sample.enable:
            raise ValueError("Original ERL does not use reward-group rejection sampling")
        estimator = getattr(self.config.algorithm.adv_estimator, "value", self.config.algorithm.adv_estimator)
        if str(estimator).lower() != "grpo":
            raise ValueError("Original ERL requires algorithm.adv_estimator=grpo with RAFT row overrides")
        if getattr(self, "use_legacy_worker_impl", "auto") != "disable":
            raise ValueError("Original ERL requires the current worker implementation for unbiased masked batch padding")
        if self.config.actor_rollout_ref.actor.get("shuffle", False):
            raise ValueError("Original ERL mixed-record normalization requires actor.shuffle=False")
        self._success = _import(self.erl_config["success_adapter"]) if self.erl_config.get("success_adapter") else None
        self._attempt_runner = _import(self.erl_config["attempt_runner"]) if self.erl_config.get("attempt_runner") else None
        self.updater_max_response = int(self.erl_config.get("updater_max_response_length", 8192))
        base_prompt, response = self.config.data.max_prompt_length, self.config.data.max_response_length
        self.updater_max_prompt = int(self.erl_config.get("updater_max_prompt_length", base_prompt))
        self.retry_max_prompt = int(self.erl_config.get("retry_max_prompt_length", base_prompt))
        self.record_prompt_width = max(base_prompt, self.updater_max_prompt, self.retry_max_prompt)
        self.record_response_width = max(response, self.updater_max_response)
        needed_context = max(self.updater_max_prompt + self.updater_max_response, self.retry_max_prompt + response, base_prompt + response)
        model_context = self.config.actor_rollout_ref.rollout.get("max_model_len") or base_prompt + response
        if min(self.updater_max_prompt, self.updater_max_response, self.retry_max_prompt) <= 0 or needed_context > model_context:
            raise ValueError(f"ERL role caps need max_model_len >= {needed_context}; configured {model_context}")
        self._erl_pending = None
        self._erl_state_signature = {
            "upstream_commit": UPSTREAM_COMMIT,
            "erl": self.erl_config,
            "policy_prompt_length": base_prompt,
            "policy_response_length": response,
            "model_context": model_context,
        }

    def init_workers(self):
        super().init_workers()
        self.erl_controller = ERLController(self.agent_execution_engine.n_parallel_agents)

    def init_envs_and_agents(self, batch):
        self._erl_source_batch = batch
        return super().init_envs_and_agents(batch)

    def _iter_training_epoch(self):
        with preserve_controller_rng():
            return iter(self.train_dataloader)

    def _save_checkpoint(self):
        if self.config.trainer.get("default_hdfs_dir") is not None:
            raise NotImplementedError("ERL controller checkpoints require a local/shared filesystem")
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        save_state(self, folder)
        return super()._save_checkpoint()

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        if self.config.trainer.resume_mode == "disable" or (result == 0 and self.global_steps == 0):
            return result
        if not isinstance(getattr(self, "_dataloader_resume_iterator_finished", None), bool):
            raise RuntimeError("ERL resume requires the pinned VeRL full dataloader-state restoration and data.pt")
        folder = self.config.trainer.resume_from_path if self.config.trainer.resume_mode == "resume_path" else Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        load_state(self, folder)
        return result

    def _validate_agent(self, logger=None, step=0):
        # First attempts exclusively, with the launcher's vanilla sampling.
        # The base meta_info['do_sample'] field is not consumed by VerlEngine;
        # val_kwargs.do_sample controls temperature in that engine.
        with preserve_controller_rng():
            return super()._validate_agent(logger=logger, step=step)

    def _succeeded(self, trajectory):
        if self._success:
            return bool(self._success(trajectory))
        return float(trajectory["trajectory_reward"]) >= float(self.erl_config.get("success_reward_threshold", 1.0))

    async def _run_attempt(self, index, prompt, meta_info):
        engine = self.agent_execution_engine
        agent = engine.agents[index]
        if prompt is not None:
            engine.agents[index] = PromptOverrideAgent(agent, prompt)
        # Role limits must not mutate the shared engine while another episode
        # is generating a plain first attempt.
        local_engine = copy.copy(engine)
        local_engine.max_prompt_length = self.retry_max_prompt if prompt is not None else self.config.data.max_prompt_length
        try:
            if self._attempt_runner is None:
                result = await local_engine.run_agent_trajectory_with_retry(index, mode="Token", meta_info=meta_info)
            else:
                result = await self._attempt_runner(self, local_engine, index, meta_info)
            validate_replay_trajectory(result)
            if result.get("metrics", {}).get("token_mismatch", 0):
                raise RuntimeError("ERL received a non-cumulative sampled trajectory; refusing silently masked internalization")
            return result
        finally:
            engine.agents[index] = agent

    async def _update_prompt(self, index, state, current_prompt):
        engine = self.agent_execution_engine
        messages = [{"role": "system", "content": DEFAULT_UPDATER_SYSTEM_PROMPT}, {"role": "user", "content": state}]
        prompt_text = engine.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=engine.accumulate_reasoning)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > self.updater_max_prompt:
            raise UpdaterPromptOverflow(f"ERL updater prompt overflow at row {index}: {len(prompt_ids)} > {self.updater_max_prompt}")
        output = await engine.rollout_engine.get_model_response(
            messages,
            application_id=f"erl-updater-{self.global_steps}-{index}",
            temperature=float(self.erl_config.get("updater_temperature", 0.7)),
            top_p=float(self.erl_config.get("updater_top_p", 0.9)),
            max_tokens=self.updater_max_response,
            enforce_max_prompt_length=False,
            accumulate_reasoning=engine.accumulate_reasoning,
        )
        if not output.prompt_ids or not output.completion_ids:
            raise ValueError("ERL updater requires original sampled prompt/completion IDs")
        content = output.content or output.text or ""
        improved = extract_prompt_from_response(content) or current_prompt
        # Updater sampling (its own temperature/top-p) is usually not the training
        # distribution; such rows keep weight 1 under rollout correction.
        updater_log_probs = usable_step_log_probs(
            {"logprobs": getattr(output, "logprobs", None), "completion_ids": output.completion_ids, "logprobs_match_training": getattr(output, "logprobs_match_training", False)}
        )
        trajectory = {
            "idx": index,
            "prompt_tokens": torch.tensor(output.prompt_ids, dtype=torch.long),
            "response_tokens": torch.tensor(output.completion_ids, dtype=torch.long),
            "response_masks": torch.ones(len(output.completion_ids), dtype=torch.long),
            "rollout_log_probs": torch.tensor(updater_log_probs or [0.0] * len(output.completion_ids), dtype=torch.float32),
            "rollout_log_prob_mask": torch.full((len(output.completion_ids),), int(updater_log_probs is not None), dtype=torch.long),
            "trajectory_reward": 0.0,
            "step_rewards": [0.0],
            "chat_completions": messages + [{"role": "assistant", "content": content, "raw_completion": output.text, "reasoning": output.reasoning}],
            "termination_reason": output.finish_reason,
            "metrics": {
                "steps": 1,
                "total_prompt_tokens": len(output.prompt_ids),
                "total_completion_tokens": len(output.completion_ids),
                "total_gradient_update_tokens": len(output.completion_ids),
            },
        }
        validate_replay_trajectory(trajectory)
        return improved, trajectory

    async def _collect(self, meta_info):
        engine = self.agent_execution_engine
        if getattr(engine.executor, "_shutdown", False):
            engine.executor = ThreadPoolExecutor(max_workers=engine.max_env_workers)
        await engine.rollout_engine.wake_up()
        try:
            return await self.erl_controller.collect(
                len(engine.envs),
                attempt=lambda index, prompt: self._run_attempt(index, prompt, meta_info),
                update=self._update_prompt,
                succeeded=self._succeeded,
            )
        finally:
            await engine.rollout_engine.sleep()

    def _pack(self, trajectories):
        return DataProto.from_dict(
            tensors=pack_replay_tensors(
                trajectories,
                max_prompt_length=self.record_prompt_width,
                max_response_length=self.record_response_width,
                pad_token_id=self.tokenizer.pad_token_id,
                include_rollout_log_probs=getattr(self, "_rollout_correction", None) is not None,
            )
        )

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        if meta_info and meta_info.get("validate", False):
            # Calling the plain method avoids memory, updater, retry and RAFT.
            # MiniSWE uses its canonical workflow attempt adapter here too.
            if self._attempt_runner is None:
                return super().generate_agent_trajectory(timing_raw=timing_raw, meta_info=meta_info)
            return self._generate_workflow_validation(meta_info)
        with ThreadPoolExecutor(max_workers=1) as executor:
            collection = executor.submit(lambda: asyncio.run(self._collect(meta_info))).result()
        actor_reward_metrics = raw_reward_metrics(collection.first, expected_count=len(self.agent_execution_engine.envs))
        self._erl_pending = collection
        self._current_step_rewards_list = [row.get("step_rewards", []) for row in collection.first]
        self._current_step_reward_tasks = None
        totals = {
            f"rollout/{name}": sum(int(row.get("metrics", {}).get(name, 0) or 0) for row in collection.generated)
            for name in ("total_prompt_tokens", "total_completion_tokens", "total_gradient_update_tokens")
        }
        totals.update(actor_reward_metrics)
        totals["rollout/total_trajectories"] = len(collection.generated)
        totals["erl/updater_prompt_overflow_rows"] = len(collection.dropped)
        totals["erl/first_attempt_rows"] = len(collection.first)
        totals["erl/first_success_rate"] = sum(self._succeeded(row) for row in collection.first) / len(collection.first)
        self._log_collection(collection)
        return self._pack(collection.first), totals

    def _log_collection(self, collection):
        episode_root = self.config.trainer.get("episode_log_dir") or Path(self.config.trainer.default_local_dir).parent / "episodes"
        folder = Path(episode_root) / "erl"
        folder.mkdir(parents=True, exist_ok=True)
        rows = [(index, "first", row) for index, row in enumerate(collection.first)] + collection.additional
        with (folder / f"step_{self.global_steps}.jsonl").open("w", encoding="utf-8") as output:
            for index, kind, row in rows:
                output.write(
                    json.dumps(
                        {
                            "row": index,
                            "record_type": kind,
                            "dropped": kind == "first" and index in collection.dropped,
                            "reward": row["trajectory_reward"],
                            "termination_reason": row.get("termination_reason"),
                            "chat_completions": row["chat_completions"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _augment_accepted_training_batch(self, batch):
        collection, self._erl_pending = self._erl_pending, None
        if collection is None or len(collection.first) != len(batch):
            raise RuntimeError("ERL first-attempt batch was rejected, rounded, or lost before mixed-objective packing")
        batch.non_tensor_batch["adv_estimator"] = np.full(len(batch), "grpo", dtype=object)
        batch.non_tensor_batch["erl_record_type"] = np.full(len(batch), "first", dtype=object)
        original_uids = batch.non_tensor_batch["uid"].copy()
        batch.non_tensor_batch["uid"] = np.asarray([f"first::{uid}" for uid in original_uids], dtype=object)
        batch.meta_info["mask_training_padding"] = True
        # Packed sampler log-probs stay aligned per row when rollout correction
        # is enabled; every other copied probability would be stale.
        rollout_keys = ("rollout_log_probs", "rollout_log_prob_mask") if getattr(self, "_rollout_correction", None) is not None else ()
        for name in ("rollout_log_probs", "rollout_log_prob_mask", "old_log_probs", "ref_log_prob"):
            if name in batch.batch and name not in rollout_keys:
                batch.batch.pop(name)
        # A rollout dropped for updater overflow leaves the batch whole,
        # first attempt included, as the release's empty episode does.
        kept = batch[torch.tensor([index not in collection.dropped for index in range(len(batch))])] if collection.dropped else batch
        if not collection.additional:
            return kept, {"erl/contextual_retry_rows": 0, "erl/internalization_rows": 0, "erl/updater_rows": 0}
        indices = [row[0] for row in collection.additional]
        extra = self._pack([row[2] for row in collection.additional])
        # RLHFDataset carries source tensors such as its required dummy_tensor
        # through the initial union. Expand those aligned dataset fields too;
        # freshly packed token/reward/mask tensors remain authoritative.
        for key in batch.batch.keys():
            if key not in extra.batch:
                extra.batch[key] = batch.batch[key][indices].clone()
        extra.non_tensor_batch = {key: copy.deepcopy(value[indices]) for key, value in batch.non_tensor_batch.items()}
        extra.non_tensor_batch["uid"] = np.asarray([f"{kind}::{original_uids[index]}" for index, kind, _ in collection.additional], dtype=object)
        extra.non_tensor_batch["adv_estimator"] = np.asarray(["raft" if kind == "second_distilled" else "grpo" for _, kind, _ in collection.additional], dtype=object)
        extra.non_tensor_batch["erl_record_type"] = np.asarray([kind for _, kind, _ in collection.additional], dtype=object)
        extra.meta_info = copy.deepcopy(batch.meta_info)
        count = len(collection.additional) // 3
        return DataProto.concat([kept, extra]), {
            "erl/contextual_retry_rows": count,
            "erl/internalization_rows": count,
            "erl/updater_rows": count,
            "erl/positive_internalization_rows": sum(row["trajectory_reward"] > 0 for _, kind, row in collection.additional if kind == "second_distilled"),
        }

    def _generate_workflow_validation(self, meta_info):
        async def collect():
            engine = self.agent_execution_engine
            if getattr(engine.executor, "_shutdown", False):
                engine.executor = ThreadPoolExecutor(max_workers=engine.max_env_workers)
            semaphore = asyncio.Semaphore(engine.n_parallel_agents)

            async def one(index):
                async with semaphore:
                    return await self._run_attempt(index, None, meta_info)

            await engine.rollout_engine.wake_up()
            try:
                return await asyncio.gather(*(one(index) for index in range(len(engine.envs))))
            finally:
                await engine.rollout_engine.sleep()

        with ThreadPoolExecutor(max_workers=1) as executor:
            rows = executor.submit(lambda: asyncio.run(collect())).result()
        self._current_step_rewards_list = [row.get("step_rewards", []) for row in rows]
        return self._pack(rows), {}
