"""Use the canonical MiniSweWorkflow/SWEAgent protocol for every ERL attempt."""

from __future__ import annotations

import asyncio

from omegaconf import OmegaConf


def succeeded(trajectory):
    return bool(trajectory.get("all_passed", False))


async def run_attempt(trainer, engine, index, meta_info):
    from examples.mini_swe_bench.workflow import MiniSweWorkflow
    from rllm.engine.rollout.rollout_engine import is_fatal_rollout_engine_error
    from rllm.workflows.workflow import TerminationReason

    agent, env = engine.agents[index], engine.envs[index]
    arguments = OmegaConf.to_container(trainer.config.rllm.workflow.get("workflow_args", {})) or {}
    # The actual benchmark workflow owns cumulative-token stopping and reward
    # mapping. The outer engine supplies only model workers and serialization.
    workflow = MiniSweWorkflow(
        rollout_engine=engine.rollout_engine,
        executor=engine.executor,
        agent_cls=lambda **unused: agent,
        env_cls=lambda **unused: env,
        max_steps=engine.max_steps,
        timeout=arguments.get("timeout", engine.trajectory_timeout),
        **{key: arguments[key] for key in ("reward_timeout", "reward_mode", "frac3_lambda", "frac3_floor") if key in arguments},
    )
    try:
        episode = None
        for retry in range(engine.retry_limit):
            episode = await workflow.run_with_termination_handling(
                task=env.task,
                uid=f"erl-swe-{trainer.global_steps}-{index}-{retry}",
                validate=bool(meta_info and meta_info.get("validate", False)),
            )
            if episode.termination_reason != TerminationReason.ERROR:
                break
            error = episode.info.get("error", {})
            exception = RuntimeError(error.get("error_message", "MiniSWE workflow failed"))
            if is_fatal_rollout_engine_error(exception) or retry + 1 == engine.retry_limit:
                raise exception
        if episode is None or len(episode.trajectories) != 1:
            raise ValueError("ERL MiniSWE attempt requires one nonempty solver trajectory")
        trajectory = episode.trajectories[0]
        steps = []
        prompt_total = completion_total = 0
        for step in trajectory.steps:
            output = step.model_output
            if output is None or not output.prompt_ids or not output.completion_ids:
                raise ValueError("ERL MiniSWE requires original model-output IDs for every sampled step")
            prompt_total += len(output.prompt_ids)
            completion_total += len(output.completion_ids)
            # Sampler log-probs let rollout correction weight each exact completion.
            sampled = {"logprobs": getattr(output, "logprobs", None), "logprobs_match_training": getattr(output, "logprobs_match_training", False)}
            steps.append({"prompt_ids": list(output.prompt_ids), "completion_ids": list(output.completion_ids), **sampled})
        if not steps:
            raise ValueError("ERL MiniSWE returned no sampled actions")
        if len(steps[0]["prompt_ids"]) > engine.max_prompt_length:
            raise ValueError("ERL MiniSWE initial prompt exceeds its explicit role cap")
        prompt, response, mask, aligned, rollout_log_probs, rollout_log_prob_mask = engine.assemble_steps(steps, return_rollout_log_probs=True)
        step_rewards = [float(step.reward or 0.0) for step in trajectory.steps]
        step_rewards[-1] = float(trajectory.reward)
        return {
            "idx": index,
            "prompt_tokens": prompt,
            "response_tokens": response,
            "response_masks": mask,
            "rollout_log_probs": rollout_log_probs,
            "rollout_log_prob_mask": rollout_log_prob_mask,
            "trajectory_reward": float(trajectory.reward),
            "step_rewards": step_rewards,
            "all_passed": bool(episode.is_correct),
            "chat_completions": trajectory.steps[-1].chat_completions,
            "termination_reason": episode.termination_reason.value,
            "metrics": {
                "steps": len(steps),
                "token_mismatch": float(not aligned),
                "total_prompt_tokens": prompt_total,
                "total_completion_tokens": completion_total,
                "total_gradient_update_tokens": completion_total,
            },
        }
    finally:
        await asyncio.get_running_loop().run_in_executor(engine.executor, env.close)
