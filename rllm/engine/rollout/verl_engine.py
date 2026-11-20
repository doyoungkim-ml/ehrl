import inspect
import uuid

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import (
    FatalRolloutEngineError,
    ModelOutput,
    RolloutEngine,
    is_fatal_rollout_engine_error,
)
from rllm.parser import ChatTemplateParser
from rllm.trainer.verl.rollout_correction import sampling_matches_training
from rllm.workflows import TerminationEvent, TerminationReason


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        # Reconstruct the server manager across VeRL API versions.
        # Newer VeRL accepts servers=zip(addresses, handles) and optional
        # load_balancer_handle; old/reverted VeRL accepts server_handles only.
        server_manager_params = inspect.signature(AsyncLLMServerManager).parameters
        self._supports_per_turn_max_tokens = "servers" in server_manager_params
        if self._supports_per_turn_max_tokens:
            servers = zip(rollout_manager.server_addresses, rollout_manager.server_handles, strict=True)
            kwargs = {"servers": servers}
            if "load_balancer_handle" in server_manager_params:
                kwargs["load_balancer_handle"] = getattr(rollout_manager, "global_load_balancer", None)
            self.server_manager = AsyncLLMServerManager(config, **kwargs)
        else:
            self.server_manager = AsyncLLMServerManager(config, rollout_manager.server_handles)

        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
            logprobs=1,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
            logprobs=1,
        )

        # Trainer log-probabilities are recomputed at the rollout temperature.
        self.training_logprob_temperature = float(self.train_sampling_params["temperature"])

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        is_validation = bool(self.validate or validate)
        sampling_params = self.val_sampling_params.copy() if is_validation else self.train_sampling_params.copy()
        sampling_params.update(kwargs)
        logprobs_match_training = not is_validation and sampling_matches_training(sampling_params, self.training_logprob_temperature)

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))
        # Starting from newer VeRL, per-turn max_tokens can be passed in sampling_params.
        # Old/reverted VeRL computes max_tokens itself inside vLLMHttpServer.generate()
        # and would pass duplicate max_tokens to vLLM SamplingParams if we include it here.
        if self._supports_per_turn_max_tokens:
            sampling_params["max_tokens"] = max_tokens

        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        request_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # list[int]

        if any(msg.get("images", None) is not None and msg["role"] == "user" for msg in messages) and self.processor is not None:
            image_data = self.chat_parser.process_image_data(messages)  # list[PIL.Image.Image]
            model_inputs = self.processor(text=[prompt], images=image_data)
            prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
            model_inputs.pop("attention_mask")
            multi_modal_inputs = dict(model_inputs)
        else:
            image_data = None
            multi_modal_inputs = None
            prompt_ids = request_prompt_ids

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        try:
            token_output: TokenOutput = await self.server_manager.generate(
                request_id=application_id,
                prompt_ids=request_prompt_ids,
                image_data=image_data,
                sampling_params=sampling_params,
            )  # type: ignore
        except Exception as exc:
            if is_fatal_rollout_engine_error(exc):
                raise FatalRolloutEngineError("Shared VeRL/vLLM rollout engine died during generation; aborting the trajectory batch instead of retrying individual environments.") from exc
            raise
        completion_ids: list[int] = token_output.token_ids
        logprobs: list[float] = token_output.log_probs

        finish_reason = token_output.stop_reason
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)

        # For GRPO, all generated tokens are used for gradient update
        # For ACE workflows, this may be overridden by the workflow/agent
        tokens_for_gradient_update = len(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
            tokens_for_gradient_update=tokens_for_gradient_update,
            logprobs_match_training=bool(logprobs_match_training and logprobs is not None and len(logprobs) == len(completion_ids)),
        )
