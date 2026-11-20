from dataclasses import dataclass

from rllm.tools.tool_base import ToolCall


class FatalRolloutEngineError(RuntimeError):
    """Raised when a shared rollout/model-serving backend has died.

    Per-trajectory retries are unsafe for these errors because all pending
    trajectories share the same rollout engine instance.
    """


def _iter_exception_chain(exc: BaseException):
    """Yield an exception and its explicit/implicit causes once each."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_fatal_rollout_engine_error(exc: BaseException) -> bool:
    """Return whether an exception means the shared rollout engine is dead.

    Ray often wraps vLLM failures as ``RayTaskError(EngineDeadError)`` without
    preserving the original exception type in the Python cause chain, so this
    intentionally checks both the concrete type and the rendered exception text.
    """

    fatal_fragments = (
        "enginedeaderror",
        "enginecore encountered an issue",
    )
    for current in _iter_exception_chain(exc):
        if isinstance(current, FatalRolloutEngineError):
            return True
        text = f"{type(current).__module__}.{type(current).__name__}: {current}".lower()
        if any(fragment in text for fragment in fatal_fragments):
            return True
    return False


@dataclass
class ModelOutput:
    text: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    multi_modal_inputs: dict[str, list] | None = None
    logprobs: list[float] | None = None  # completion logprobs
    prompt_logprobs: list[float] | None = None  # prompt logprobs aligned to prompt_ids
    prompt_length: int = 0
    completion_length: int = 0
    finish_reason: str | None = None
    # Token tracking for gradient update accounting
    # For GRPO: tokens_for_gradient_update = completion_length (all generated tokens)
    # For ACE: tokens_for_gradient_update may be less than completion_length
    # (e.g., curator/reflector tokens not used for gradient update)
    tokens_for_gradient_update: int = 0
    # True only when `logprobs` are the sampler's per-token log-probabilities
    # under the training sampling distribution (see rllm.trainer.verl.rollout_correction).
    logprobs_match_training: bool = False

    def to_dict(self):
        return {
            "text": self.text,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls] if self.tool_calls else [],
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "multi_modal_inputs": self.multi_modal_inputs,
            "logprobs": self.logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "prompt_length": self.prompt_length,
            "completion_length": self.completion_length,
            "finish_reason": self.finish_reason,
            "tokens_for_gradient_update": self.tokens_for_gradient_update,
            "logprobs_match_training": self.logprobs_match_training,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            text=data.get("text"),
            content=data.get("content"),
            reasoning=data.get("reasoning"),
            tool_calls=[ToolCall(**tool_call) for tool_call in data.get("tool_calls", [])] if data.get("tool_calls") else None,
            prompt_ids=data.get("prompt_ids"),
            completion_ids=data.get("completion_ids"),
            multi_modal_inputs=data.get("multi_modal_inputs"),
            logprobs=data.get("logprobs"),
            prompt_logprobs=data.get("prompt_logprobs"),
            prompt_length=data.get("prompt_length", 0),
            completion_length=data.get("completion_length", 0),
            finish_reason=data.get("finish_reason"),
            tokens_for_gradient_update=data.get("tokens_for_gradient_update", 0),
            logprobs_match_training=bool(data.get("logprobs_match_training", False)),
        )


class RolloutEngine:
    def __init__(self, *args, **kwargs):
        pass

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        raise NotImplementedError("get_model_response is not implemented")

    async def wake_up(self):
        pass

    async def sleep(self):
        pass
