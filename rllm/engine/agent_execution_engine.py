import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
except ImportError as err:
    raise ImportError("AgentExecutionEngine requires extra dependencies. Install with: pip install rllm[train]") from err

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.engine.rollout.rollout_engine import ModelOutput, is_fatal_rollout_engine_error
from rllm.engine.terminal_feedback import sanitize_terminal_feedback
from rllm.environments.base.base_env import BaseEnv, UncertainExecutionError, _iter_exception_chain
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.parser import ChatTemplateParser
from rllm.trainer.verl.rollout_correction import ROLLOUT_LOG_PROB_MASK_KEY, ROLLOUT_LOG_PROBS_KEY, usable_step_log_probs
from rllm.utils import colorful_print
from rllm.utils.provider_seed import provider_seed_scope, task_provider_seed

logger = logging.getLogger(__name__)


class RetryableTrajectoryTerminationError(RuntimeError):
    """Raised when a trajectory should be retried based on its termination reason."""


class AgentExecutionEngine:
    def __init__(
        self,
        engine_name="openai",
        tokenizer=None,
        rollout_engine=None,
        chat_parser=None,
        n_parallel_agents=128,  # The number of active agents
        trajectory_timeout=None,
        trajectory_hard_timeout=7200,
        gamma=0.2,
        api_retries=3,
        retry_limit=3,
        retry_on_termination_reasons=None,
        max_steps=5,
        max_response_length=8192,
        max_prompt_length=1024,
        config=None,
        agent_class=None,
        env_class=None,
        agent_args=None,
        rollout_engine_args=None,
        env_args=None,
        max_workers=64,  # The number of concurrent env operations
        enforce_max_prompt_length=False,  # If enabled, applies max_prompt check per step
        overlong_filter=False,  # Filter for overlong trajectories (i.e. TRUNCATION, MAX_STEPS, TIMEOUT)
        **kwargs,
    ):
        if agent_args is None:
            agent_args = {}
        if rollout_engine_args is None:
            rollout_engine_args = {}
        if env_args is None:
            env_args = {}

        self.config = config
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.n_parallel_agents = n_parallel_agents
        self.max_env_workers = max_workers
        self.overlong_filter = overlong_filter

        # For interaction
        self.gamma = gamma
        self.retry_limit = retry_limit
        if retry_on_termination_reasons is None:
            retry_on_termination_reasons = ("ENV_TIMEOUT", "ENV_ERROR")
        self.retry_on_termination_reasons = {str(reason) for reason in retry_on_termination_reasons if str(reason)}
        self.max_steps = max_steps
        self.max_response_length = max_response_length
        self.max_prompt_length = max_prompt_length
        self.enforce_max_prompt_length = enforce_max_prompt_length
        rllm_config = self.config.get("rllm", {}) if self.config is not None else {}
        agent_config = rllm_config.get("agent", {}) if rllm_config is not None else {}
        self.disable_thinking = rllm_config.get("disable_thinking", False) if rllm_config is not None else False
        self.accumulate_reasoning = rllm_config.get("accumulate_reasoning", False) if rllm_config is not None else False
        self.preserve_terminal_feedback = bool(agent_config.get("preserve_terminal_feedback", False))
        self.trajectory_reward_aggregation = str(
            agent_config.get(
                "trajectory_reward_aggregation",
                rllm_config.get("trajectory_reward_aggregation", "sum") if rllm_config is not None else "sum",
            )
            or "sum"
        )

        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_class = env_class
        self.env_args = env_args

        self.agents = [None for _ in range(n_parallel_agents)]
        self.envs = [None for _ in range(n_parallel_agents)]

        self.trajectory_timeout = trajectory_timeout
        if not trajectory_timeout:
            self.trajectory_timeout = int(1e9)
        self.trajectory_hard_timeout = 7200.0 if trajectory_hard_timeout is None else float(trajectory_hard_timeout)

        if env_class is not None:
            assert env_class.is_multithread_safe(), "Environment must be multithread safe for async engine"

        if chat_parser is None:
            self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=self.disable_thinking)
        else:
            self.chat_parser = chat_parser

        self.rollout_engine_args = rollout_engine_args
        self.sampling_params = kwargs.get("sampling_params", {})  # for openai api requests

        assert self.engine_name in ["openai", "verl"], "Currently only openai and verl are supported as rollout engine"
        if self.engine_name == "openai":
            from rllm.engine.rollout.openai_engine import OpenAIEngine

            self.rollout_engine = OpenAIEngine(
                **rollout_engine_args,
                api_retries=api_retries,
                tokenizer=self.tokenizer,
                chat_parser=self.chat_parser,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                disable_thinking=self.disable_thinking,
            )
        elif self.engine_name == "verl":
            from rllm.engine.rollout.verl_engine import VerlEngine

            self.rollout_engine = VerlEngine(
                config=self.config,
                rollout_manager=rollout_engine,
                tokenizer=self.tokenizer,
                disable_thinking=self.disable_thinking,
            )

        # Create a thread pool executor for environment interactions (i.e. step, reset, close)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def get_model_response(self, prompt, application_id, **kwargs) -> ModelOutput:
        """
        Compute model response asynchronously based on the engine type.

        This function is multithread safe and routes the request to the appropriate
        engine-specific handler.

        Args:
            prompt: The input prompt to send to the model
            application_id: Unique identifier for the application
            **kwargs: Additional arguments to pass to the model

        Returns:
            The model's response as a ModelOutput object

        Raises:
            NotImplementedError: If the engine type is not supported
        """

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        if self.engine_name == "openai":
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, enforce_max_prompt_length=False, **sampling_params)
            return output
        elif self.engine_name == "verl":
            meta_data = sampling_params.pop("meta_info", {})
            validate = meta_data.get("validate", False)
            output = await self.rollout_engine.get_model_response(prompt, application_id=application_id, validate=validate, enforce_max_prompt_length=False, **sampling_params)
            return output
        else:
            raise NotImplementedError(f"Engine type '{self.engine_name}' not supported")

    def update_envs_and_agents(self, envs, agents):
        """
        Update the environments and agents.

        Args:
            envs: List of environments to use
            agents: List of agents to use
        """
        assert len(agents) == len(envs), f"Number of agents must equal to number of environments but received, {len(agents)} and {len(envs)}"
        self.envs = envs
        # For keeping track of the environment index in the batch.
        for idx, env in enumerate(envs):
            env.idx = idx
        self.agents = agents

    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        provider_identity = kwargs.pop("_evaluation_provider_identity", None) or task_provider_seed(agent.trajectory.task)
        # env_id = env.env_id

        termination_reason = None
        execution_info = {}
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # Token tracking for gradient update accounting
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_gradient_update_tokens = 0

        # for step return
        episode_steps = []

        # Reset environment with the task using the executor
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        # Update agent internal state from environment.
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(
            messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=True,
            contains_generation_msg=True,
            accumulate_reasoning=self.accumulate_reasoning,
        )
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            # Get action from agent
            prompt_messages = agent.chat_completions.copy()
            # Max remaining tokens left for the response
            # For enforced max prompt at each step, no need to deduct here
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            try:
                with provider_seed_scope(provider_identity, call=step_idx):
                    model_output = await self.get_model_response(prompt_messages, application_id, **kwargs)
            except Exception as gen_exc:
                # A multi-turn prompt can outgrow the model context; end the episode
                # via PROMPT_TRUNCATION instead of letting the retry loop raise and
                # crash the training step.
                if "maximum context length" in str(gen_exc).lower():
                    termination_reason = "PROMPT_TRUNCATION"
                    break
                raise
            response = model_output.text
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            # Update steps
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
                "prompt_ids": model_output.prompt_ids,
                "completion_ids": model_output.completion_ids,
                "logprobs": model_output.logprobs,
                "logprobs_match_training": model_output.logprobs_match_training,
            }
            episode_steps.append(prompt_response_pair)

            # Track token counts
            total_prompt_tokens += len(model_output.prompt_ids) if model_output.prompt_ids else 0
            total_completion_tokens += len(model_output.completion_ids) if model_output.completion_ids else 0
            total_gradient_update_tokens += model_output.tokens_for_gradient_update

            # Update agent with model response. Pass the structured ModelOutput
            # as well as raw text so agents that preserve reasoning/history can
            # reconstruct Qwen thinking turns losslessly while still parsing
            # executable actions from the visible/raw response text.
            action: Action = agent.update_from_model(response, model_output=model_output)
            action = action.action

            # Take step in environment using the executor
            start_time = time.time()

            try:
                next_observation, reward, done, info = await asyncio.wait_for(loop.run_in_executor(self.executor, env.step, action), timeout=(self.trajectory_timeout - total_time))
            except asyncio.TimeoutError as error:
                uncertain = self._execution_uncertainty_error(idx, "TrajectoryStepTimeout")
                if uncertain is not None:
                    raise uncertain from error
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(
                        f"Warning: Trajectory {idx} completed due to: {termination_reason}"
                        " before able to perform 1 complete action. This might cause"
                        " unexpected behavior. Consider increasing trajectory timeout limit.\n",
                        "red",
                    )
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len
            if info.get("execution_uncertain"):
                execution_info = {key: value for key, value in info.items() if key.startswith("execution_") or key == "execute_retry_attempted"}

            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks(
                    [assistant_message],
                    tokenizer=self.tokenizer,
                    parser=self.chat_parser,
                    contains_first_msg=False,
                    contains_generation_msg=False,
                    accumulate_reasoning=self.accumulate_reasoning,
                )
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                    env_messages,
                    tokenizer=self.tokenizer,
                    parser=self.chat_parser,
                    contains_first_msg=False,
                    contains_generation_msg=True,
                    accumulate_reasoning=self.accumulate_reasoning,
                )

            # Update response token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                # Truncation length
                truncation_length = self.max_response_length - response_token_len
                # Truncate the response and masks
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    # Edge case where the response is exactly the max response length.
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                # handle returning
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            # Check if episode is done
            if done:
                termination_reason = info.get("termination", "ENV_DONE")
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True

        # Preserve the existing length/time masks, but never report an
        # uncertain stateful action as an ordinary cutoff or completion.
        if execution_info:
            if termination_reason != UncertainExecutionError.termination_reason:
                execution_info["limit_termination_reason"] = termination_reason
            termination_reason = UncertainExecutionError.termination_reason

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
            reward_time = time.time() - start_time
            cur_step.reward = reward
        # Closing environment using the executor.
        await loop.run_in_executor(self.executor, env.close)
        if termination_reason:
            if reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        trajectory: Trajectory = agent.trajectory
        # Aggregate final trajectory statistics
        compute_trajectory_reward(
            trajectory,
            aggregation=self.trajectory_reward_aggregation,
        )
        compute_mc_return(trajectory, gamma=self.gamma)

        all_passed = any(isinstance(getattr(step, "info", None), dict) and step.info.get("all_passed", False) for step in trajectory.steps)
        step_rewards = [step.reward for step in trajectory.steps]
        trajectory.info["all_passed"] = all_passed
        trajectory.info["step_rewards"] = step_rewards
        trajectory.info["termination_reason"] = termination_reason
        trajectory.info.update(execution_info)

        # Add token tracking metrics to trajectory info
        trajectory.info["total_prompt_tokens"] = total_prompt_tokens
        trajectory.info["total_completion_tokens"] = total_completion_tokens
        trajectory.info["total_gradient_update_tokens"] = total_gradient_update_tokens

        if mode == "Text":
            return trajectory
        elif mode == "Token":
            prompt_tokens, response_tokens, response_masks, is_valid_trajectory, rollout_log_probs, rollout_log_prob_mask = self.assemble_steps(episode_steps, return_rollout_log_probs=True)
            token_result = {
                **execution_info,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_masks": response_masks,
                ROLLOUT_LOG_PROBS_KEY: rollout_log_probs,
                ROLLOUT_LOG_PROB_MASK_KEY: rollout_log_prob_mask,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "chat_completions": agent.chat_completions,
                "step_rewards": step_rewards,
                "all_passed": all_passed,
                "termination_reason": termination_reason,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                    "token_mismatch": 0.0 if is_valid_trajectory else 1.0,
                    # Token tracking for gradient update accounting
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_gradient_update_tokens": total_gradient_update_tokens,
                },
            }
            if getattr(self, "preserve_terminal_feedback", False) and trajectory.steps:
                token_result["terminal_feedback"] = sanitize_terminal_feedback(trajectory.steps[-1].info)
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                **execution_info,
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
                "step_rewards": step_rewards[: len(episode_steps)],
                "all_passed": all_passed,
                "termination_reason": termination_reason,
            }
            return steps_result
        else:
            raise ValueError(f"Mode {mode} not supported")

    def assemble_steps(self, steps: list[dict], return_rollout_log_probs: bool = False):
        """
        Transform step-by-step results into trajectory format for training.
        The assemble is aggresive, if steps is not cumulative, the response_masks is set to all 0s.

        Each step_result contains:
        - steps: List of {"prompt": str, "response": str, "prompt_ids": list, "completion_ids": list}

        For training, we need to assemble the full conversation sequence where:
        - prompt_tokens: Initial prompt (first step's prompt_ids)
        - response_tokens: All subsequent conversation (completion_ids + next step's prompt_ids)
        - response_masks: Mask indicating which tokens contribute to loss (only completion_ids)

        With ``return_rollout_log_probs`` it also returns the sampler's per-token
        log-probabilities aligned to ``response_tokens`` and their availability
        mask. A completion's log-probabilities are available only when its exact
        sampled context equals the assembled training context: a turn whose
        generated Qwen ``<think>\\n`` sentinel is dropped, and every turn after a
        text-equivalent retokenization, is marked unavailable (weight 1).
        """

        # Start with initial prompt from first step
        initial_prompt_ids = steps[0]["prompt_ids"]
        accumulated_sequence = initial_prompt_ids.copy()
        response_tokens = []
        response_masks = []
        # Parallel to response_tokens: sampler log-probabilities and whether each
        # is valid for the token's assembled training context.
        rollout_log_probs: list[float] = []
        rollout_log_prob_mask: list[int] = []
        context_diverged = False
        previous_completion_span = (0, 0)
        is_valid_trajectory = True

        def completion_log_probs(step: dict, count: int) -> tuple[list[float], list[int]]:
            values = None if context_diverged else usable_step_log_probs(step)
            if values is None:
                return [0.0] * count, [0] * count
            return values, [1] * count

        for i, step in enumerate(steps):
            current_prompt_ids = step["prompt_ids"]
            current_completion_ids = step["completion_ids"]

            if i == 0:
                # First step: just add completion
                response_tokens.extend(current_completion_ids)
                response_masks.extend([1] * len(current_completion_ids))  # completion contributes to loss
                values, available = completion_log_probs(step, len(current_completion_ids))
                rollout_log_probs.extend(values)
                rollout_log_prob_mask.extend(available)
                previous_completion_span = (0, len(current_completion_ids))
                accumulated_sequence.extend(current_completion_ids)
            else:
                prompt_prefix_len = len(accumulated_sequence)
                if current_prompt_ids[:prompt_prefix_len] != accumulated_sequence:
                    # Find the first differing position
                    prefix = current_prompt_ids[:prompt_prefix_len]
                    diff_pos = None
                    for i, (expected, actual) in enumerate(zip(accumulated_sequence, prefix, strict=False)):
                        if expected != actual:
                            diff_pos = i
                            break

                    # Qwen thinking history can drop exactly the generated
                    # <think>\n sentinel from the next-turn prompt even when the
                    # rest of the prompt is token-identical. Preserve the usable
                    # trajectory by deleting only that exact sentinel from the
                    # previous response span if, and only if, all following tokens
                    # align. Do not suppress arbitrary mismatches.
                    qwen_think_sentinel = [151667, 198]
                    if diff_pos is not None and accumulated_sequence[diff_pos : diff_pos + 2] == qwen_think_sentinel:
                        candidate_accumulated = accumulated_sequence[:diff_pos] + accumulated_sequence[diff_pos + 2 :]
                        if current_prompt_ids[: len(candidate_accumulated)] == candidate_accumulated:
                            response_pos = diff_pos - len(initial_prompt_ids)
                            if response_pos >= 0 and response_tokens[response_pos : response_pos + 2] == qwen_think_sentinel:
                                logger.info(
                                    "When assemble steps, normalized missing Qwen <think>\\n sentinel at position %s by removing ids %s from response span.",
                                    diff_pos,
                                    qwen_think_sentinel,
                                )
                                del response_tokens[response_pos : response_pos + 2]
                                del response_masks[response_pos : response_pos + 2]
                                del rollout_log_probs[response_pos : response_pos + 2]
                                del rollout_log_prob_mask[response_pos : response_pos + 2]
                                # The turn that generated the dropped sentinel was sampled
                                # with it in context; the trainer scores it without.
                                span_start, span_end = previous_completion_span
                                if response_pos < span_start:
                                    span_start -= 2
                                span_end -= 2
                                for position in range(span_start, max(span_start, span_end)):
                                    rollout_log_prob_mask[position] = 0
                                accumulated_sequence = candidate_accumulated
                                prompt_prefix_len = len(accumulated_sequence)
                                prefix = current_prompt_ids[:prompt_prefix_len]
                                diff_pos = None

                    # Qwen/BPE decode->reencode can change token boundaries without
                    # changing the exact text. First try same-token-count decoded
                    # equality; if token counts shifted, search a small window around
                    # the expected boundary for a text-equivalent prefix. Fail closed
                    # for any decoded-text difference, catching real lost sentinels,
                    # newlines, or content.
                    tokenizer = self.tokenizer
                    assert tokenizer is not None
                    decoded_accumulated = tokenizer.decode(accumulated_sequence, skip_special_tokens=False)
                    decoded_prefix = tokenizer.decode(prefix, skip_special_tokens=False)
                    matched_text_equivalent_prefix = decoded_accumulated == decoded_prefix

                    if not matched_text_equivalent_prefix:
                        search_radius = 256
                        start = max(0, len(accumulated_sequence) - search_radius)
                        end = min(len(current_prompt_ids), len(accumulated_sequence) + search_radius)
                        for candidate_len in range(start, end + 1):
                            if candidate_len == len(accumulated_sequence):
                                continue
                            candidate_text = tokenizer.decode(current_prompt_ids[:candidate_len], skip_special_tokens=False)
                            if candidate_text == decoded_accumulated:
                                prompt_prefix_len = candidate_len
                                matched_text_equivalent_prefix = True
                                break

                    if matched_text_equivalent_prefix:
                        if diff_pos is not None:
                            logger.info(
                                "When assemble steps, accepted text-equivalent retokenized prefix at position %s. Expected ids: %s, got ids: %s, prefix_len=%s->%s.",
                                diff_pos,
                                accumulated_sequence[diff_pos : diff_pos + 5],
                                prefix[diff_pos : diff_pos + 5],
                                len(accumulated_sequence),
                                prompt_prefix_len,
                            )
                    else:
                        if diff_pos is not None:
                            logger.warning(
                                "When assemble steps, detect the trajectory not accumulative at position %s. "
                                "Expected ids: %s decoded=%r, got ids: %s decoded=%r. Setting response_masks to all 0s. "
                                "This is likely due to retokenization or lossy history reconstruction.",
                                diff_pos,
                                accumulated_sequence[diff_pos : diff_pos + 5],
                                tokenizer.decode(accumulated_sequence[diff_pos : diff_pos + 20], skip_special_tokens=False),
                                prefix[diff_pos : diff_pos + 5],
                                tokenizer.decode(prefix[diff_pos : diff_pos + 20], skip_special_tokens=False),
                            )
                        else:
                            logger.warning(f"When assemble steps, detect length mismatch. Expected length: {len(accumulated_sequence)}, Got length: {len(prefix)}. Setting response_masks to all 0s.")

                        is_valid_trajectory = False
                        break

                # A text-equivalent but token-different prefix means the sampler saw
                # other context IDs than the trainer from this turn onward.
                if current_prompt_ids[:prompt_prefix_len] != accumulated_sequence:
                    context_diverged = True
                gap = len(current_prompt_ids) - prompt_prefix_len
                response_tokens.extend(current_prompt_ids[prompt_prefix_len:] + current_completion_ids)
                response_masks.extend([0] * gap + [1] * len(current_completion_ids))  # completion contributes to loss
                values, available = completion_log_probs(step, len(current_completion_ids))
                rollout_log_probs.extend([0.0] * gap + values)
                rollout_log_prob_mask.extend([0] * gap + available)
                previous_completion_span = (len(response_tokens) - len(current_completion_ids), len(response_tokens))
                accumulated_sequence = current_prompt_ids + current_completion_ids

        assert len(response_masks) == len(response_tokens)

        # The rollout loop enforces max_response_length on its re-tokenized view
        # of the trajectory, while this assembly uses the rollout engine's raw
        # prompt/completion ids; the two counts can differ by a few tokens, so a
        # TRUNCATION-terminated trajectory could come back slightly wider than
        # the configured response budget. Apply the same budget to the assembled
        # sequence so it never exceeds the training batch width.
        max_response_length = getattr(self, "max_response_length", None)
        if max_response_length is not None and len(response_tokens) > max_response_length:
            logger.info(
                "When assemble steps, assembled response length %s exceeds max_response_length %s; trimming the trailing %s token(s).",
                len(response_tokens),
                max_response_length,
                len(response_tokens) - max_response_length,
            )
            response_tokens = response_tokens[:max_response_length]
            response_masks = response_masks[:max_response_length]
            rollout_log_probs = rollout_log_probs[:max_response_length]
            rollout_log_prob_mask = rollout_log_prob_mask[:max_response_length]

        prompt_tokens = torch.tensor(initial_prompt_ids, dtype=torch.long)
        response_tokens = torch.tensor(response_tokens, dtype=torch.long)
        response_masks = torch.tensor(response_masks, dtype=torch.long)

        if self.config.rllm.filter_token_mismatch:
            response_masks = response_masks * int(is_valid_trajectory)

        if not return_rollout_log_probs:
            return prompt_tokens, response_tokens, response_masks, is_valid_trajectory
        assert len(rollout_log_probs) == len(rollout_log_prob_mask) == len(response_tokens)
        rollout_log_prob_mask = torch.tensor(rollout_log_prob_mask, dtype=torch.long) * (response_masks > 0).long()
        return prompt_tokens, response_tokens, response_masks, is_valid_trajectory, torch.tensor(rollout_log_probs, dtype=torch.float32), rollout_log_prob_mask

    @staticmethod
    def _get_result_termination_reason(result):
        if isinstance(result, dict):
            return result.get("termination_reason")
        termination_reason = getattr(result, "termination_reason", None)
        if termination_reason is not None:
            return termination_reason
        info = getattr(result, "info", None)
        if isinstance(info, dict):
            return info.get("termination_reason")
        return None

    def _execution_uncertainty_error(self, idx, error_type):
        if idx >= len(self.envs) or self.envs[idx] is None:
            return None
        env = self.envs[idx]
        if getattr(env, "execution_in_flight", False) or getattr(env, "execution_uncertainty", None):
            marker = getattr(env, "mark_execution_uncertain", None)
            if callable(marker):
                return marker(error_type)
        return None

    def _is_retryable_trajectory_exception(self, idx, exc: BaseException) -> bool:
        if any(isinstance(current, UncertainExecutionError) for current in _iter_exception_chain(exc)):
            return False
        if is_fatal_rollout_engine_error(exc):
            return False
        if idx >= len(self.envs):
            return False
        env = self.envs[idx]
        if env is None:
            return False
        classifier = getattr(env, "is_retryable_error", None)
        if not callable(classifier):
            return False
        try:
            return bool(classifier(exc))
        except Exception as classifier_exc:  # noqa: BLE001 - retry classifier must not hide original error.
            logger.warning(
                "Error classifying trajectory %s exception %s as retryable: %s",
                idx,
                type(exc).__name__,
                classifier_exc,
            )
            return False

    def _cleanup_env_after_failed_attempt(self, idx) -> None:
        if idx >= len(self.envs):
            return
        env = self.envs[idx]
        if env is None:
            return
        close = getattr(env, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort before retry.
            logger.warning("Error closing env %s after retryable trajectory failure: %s", idx, exc)

    async def run_agent_trajectory_with_retry(self, idx, seed=0, mode="Text", **kwargs):
        last_retryable_error = None
        agents = getattr(self, "agents", ())
        provider_identity = task_provider_seed(agents[idx].trajectory.task) if agents else None
        if provider_identity is not None:
            kwargs["_evaluation_provider_identity"] = provider_identity
        for attempt_idx in range(self.retry_limit):
            try:
                application_id = str(uuid.uuid4())
                result = await asyncio.wait_for(
                    self.run_agent_trajectory_async(
                        idx,
                        application_id=application_id,
                        seed=seed,
                        mode=mode,
                        **kwargs,
                    ),
                    timeout=self.trajectory_hard_timeout,
                )
                termination_reason = self._get_result_termination_reason(result)
                # Even a broad/custom retry configuration must not replay a
                # stateful action whose execution outcome is unknown.
                if termination_reason == UncertainExecutionError.termination_reason:
                    return result
                if termination_reason in self.retry_on_termination_reasons:
                    raise RetryableTrajectoryTerminationError(f"Trajectory {idx} ended with retryable termination {termination_reason} (attempt {attempt_idx + 1}/{self.retry_limit})")
                return result
            except asyncio.TimeoutError as exc:
                uncertain = self._execution_uncertainty_error(idx, "TrajectoryHardTimeout")
                if uncertain is not None:
                    raise uncertain from exc
                last_retryable_error = exc
                self._cleanup_env_after_failed_attempt(idx)
                colorful_print(
                    f"Trajectory {idx} exceeded hard timeout {self.trajectory_hard_timeout}s (attempt {attempt_idx + 1}/{self.retry_limit}); retrying.\n",
                    "red",
                )
            except RetryableTrajectoryTerminationError as exc:
                last_retryable_error = exc
                self._cleanup_env_after_failed_attempt(idx)
                colorful_print(f"{exc}; retrying.\n", "yellow")
            except Exception as exc:
                if is_fatal_rollout_engine_error(exc):
                    colorful_print(
                        f"Fatal rollout engine error while running trajectory {idx}; aborting the trajectory batch instead of retrying this environment.\n",
                        "red",
                    )
                    raise
                if not self._is_retryable_trajectory_exception(idx, exc):
                    raise
                last_retryable_error = exc
                self._cleanup_env_after_failed_attempt(idx)
                colorful_print(
                    f"Trajectory {idx} hit retryable environment error {type(exc).__name__}: {exc} (attempt {attempt_idx + 1}/{self.retry_limit}); retrying.\n",
                    "yellow",
                )
        final_error = Exception(f"Trajectory {idx} cannot complete after {self.retry_limit} attempts. Last hard timeout was {self.trajectory_hard_timeout}s. Please check the log message")
        if last_retryable_error is not None:
            raise final_error from last_retryable_error
        raise final_error

    async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", manage_rollout_engine_state=True, **kwargs):
        if timing_raw is None:
            timing_raw = {}
        assert all(env is not None and isinstance(env, BaseEnv) for env in self.envs), "All environments must be inheriting from BaseEnv"
        assert all(env.is_multithread_safe() for env in self.envs), "All environments must be multithread safe for async engine"  # type: ignore
        max_concurrency = self.n_parallel_agents

        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

        try:
            if self.engine_name == "verl" and manage_rollout_engine_state:
                # TODO(mbpp): if free_cache_engine + rollout data parallelism becomes unstable,
                # revive the old serialized wake/sleep path instead of parallel transitions.
                await self.rollout_engine.wake_up()  # type: ignore

            semaphore = asyncio.Semaphore(self.n_parallel_agents)

            async def launch_one_trajectory_task(env_idx: int):
                async with semaphore:
                    try:
                        result = await self.run_agent_trajectory_with_retry(
                            idx=env_idx,
                            seed=reset_seed,
                            mode=mode,
                            **kwargs,
                        )
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        raise e
                    return result

            # Create all N conceptual tasks. Their execution will be throttled by the semaphore
            # and the availability of agent/env indices. Keep task handles so a fatal
            # shared rollout-engine failure can cancel outstanding work immediately.
            tasks_to_run = [asyncio.create_task(launch_one_trajectory_task(i)) for i in range(len(self.envs))]

            tasks_completed = 0
            for coro in asyncio.as_completed(tasks_to_run):
                try:
                    result = await coro
                    tasks_completed += 1
                    colorful_print(f"Number of Trajectories {tasks_completed}/{len(self.envs)} completed", "cyan")
                    yield result
                except Exception as e:
                    if is_fatal_rollout_engine_error(e):
                        for task in tasks_to_run:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks_to_run, return_exceptions=True)
                    raise e
        finally:
            if self.engine_name == "verl" and manage_rollout_engine_state:
                try:
                    await self.rollout_engine.sleep()  # type: ignore
                except Exception as exc:
                    logger.warning("Error sleeping rollout engine during trajectory cleanup: %s", exc)
            self.executor.shutdown(wait=False, cancel_futures=True)

    async def execute_tasks(self, tasks: list[dict]):
        """
        Run asynchronous interactions between the agent and environment where each agent
        has its own environment instance and can proceed independently.

        Args:
            tasks: List of tasks to process
            max_concurrent: Maximum number of concurrent tasks to process (defaults to self.n_parallel_agents)

        Returns:
            A list of trajectories, one for each task.
        """
        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)

        max_concurrent = self.n_parallel_agents

        # Initialize results list to store trajectories for all tasks
        all_trajectories = {}

        # Create a queue of tasks to process
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        # Track completed trajectories
        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                # Get an available index
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert self.agents[index] is not None and isinstance(self.agents[index], BaseAgent), "Agent is not initalized or not inheriting from BaseAgent"
                    self.agents[index].trajectory.task = task  # type: ignore
                    res = await self.run_agent_trajectory_async(index, application_id=task_id)
                    res.task = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    # Put the index back in the queue when done
                    await index_queue.put(index)

        # Run all tasks concurrently
        results = await asyncio.gather(*[sem_wrapper(task_id, task) for task_id, task in task_queue])

        all_trajectories = {task_id: trajectory for task_id, trajectory in results}
        ordered_trajectories = [all_trajectories[i] for i in range(len(all_trajectories))]

        self.executor.shutdown(wait=False, cancel_futures=True)

        return ordered_trajectories

    def shutdown(self):
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown()
            self.executor = None


class AsyncAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
