"""Exercise real OpenAI SDK timeout handling without network requests.

Run with --noconftest in the prepared CPU environment to avoid dependency stubs.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from rllm.engine.rollout.openai_engine import OpenAIEngine

REAL_ASYNC_OPENAI = openai.AsyncOpenAI
TIMEOUT_ENV = "RLLM_OPENAI_REQUEST_TIMEOUT_S"
pytestmark = pytest.mark.skipif(
    not hasattr(openai, "__version__") or not all(hasattr(httpx, name) for name in ("Timeout", "MockTransport", "AsyncClient")),
    reason="Real OpenAI/httpx dependencies required; run in the prepared CPU environment with --noconftest",
)


@pytest.fixture(autouse=True)
def clear_timeout_env(monkeypatch):
    monkeypatch.delenv(TIMEOUT_ENV, raising=False)


def run_request(monkeypatch, endpoint, *, engine_kwargs=None, sampling_timeout=None, call_kwargs=None):
    requests = []
    client_kwargs = []

    def respond(request):
        requests.append(request)
        choice = {"index": 0, "finish_reason": "stop"}
        if endpoint == "completion":
            choice.update(text="answer", token_ids=[7, 8, 9], logprobs=None)
        else:
            choice["message"] = {"role": "assistant", "content": "answer"}
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "text_completion" if endpoint == "completion" else "chat.completion",
                "created": 0,
                "model": "dummy",
                "choices": [choice],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        )

    def create_client(**kwargs):
        client_kwargs.append(kwargs)
        return REAL_ASYNC_OPENAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai, "AsyncOpenAI", create_client)
    sampling = {"temperature": 0.7, "top_p": 0.8, "seed": 42, "max_tokens": 32768}
    if sampling_timeout is not None:
        sampling["timeout"] = sampling_timeout
    original_sampling = sampling.copy()
    parser = SimpleNamespace(
        parse=lambda *_args, **_kwargs: [11, 12],
        parse_completion=lambda _ids: {"content": "answer", "reasoning": "", "tool_calls": []},
    )
    engine = OpenAIEngine(
        model="dummy",
        tokenizer=SimpleNamespace() if endpoint == "completion" else None,
        chat_parser=parser,
        api_key="unit-test",
        base_url="http://unit.test/v1",
        max_prompt_length=8192,
        max_response_length=32768,
        max_model_length=40000,
        sampling_params=sampling,
        **(engine_kwargs or {}),
    )
    messages = [{"role": "user", "content": "test"}]

    async def invoke():
        try:
            return await engine.get_model_response(messages, **(call_kwargs or {}))
        finally:
            await engine.client.close()

    result = asyncio.run(invoke())
    assert result.content == "answer"
    assert result.completion_length == 3
    assert engine.api_retries == 3
    assert engine.client.max_retries == openai.DEFAULT_MAX_RETRIES
    assert len(requests) == 1
    request = requests[0]
    expected_body = {"model": "dummy", "temperature": 0.7, "top_p": 0.8, "seed": 42, "max_tokens": 32768}
    expected_body.update({"prompt": [11, 12]} if endpoint == "completion" else {"messages": messages})
    assert json.loads(request.content) == expected_body
    assert request.url.path == ("/v1/completions" if endpoint == "completion" else "/v1/chat/completions")
    assert sampling == original_sampling
    return engine, client_kwargs[0], request.extensions["timeout"]


@pytest.mark.parametrize("endpoint", ["completion", "chat"])
@pytest.mark.parametrize(
    ("env_value", "explicit", "expected"),
    [(None, None, 3600.0), ("4800.5", None, 4800.5), ("4800", 2400.0, 2400.0), ("invalid", 2400.0, 2400.0)],
)
def test_timeout_default_and_precedence_reach_http_transport(monkeypatch, endpoint, env_value, explicit, expected):
    if env_value is not None:
        monkeypatch.setenv(TIMEOUT_ENV, env_value)
    engine, client_kwargs, transport_timeout = run_request(monkeypatch, endpoint, engine_kwargs={"request_timeout_s": explicit})
    assert engine.request_timeout_s == expected
    assert client_kwargs["timeout"] == expected
    assert transport_timeout == httpx.Timeout(expected).as_dict()


@pytest.mark.parametrize("endpoint", ["completion", "chat"])
@pytest.mark.parametrize("override", [125.0, None, "structured"])
def test_per_call_sdk_timeout_overrides_engine_and_sampling_defaults(monkeypatch, endpoint, override):
    if override == "structured":
        override = httpx.Timeout(90.0, connect=5.0)
    _, client_kwargs, transport_timeout = run_request(
        monkeypatch, endpoint, engine_kwargs={"request_timeout_s": 3600}, sampling_timeout=500, call_kwargs={"timeout": override}
    )
    assert client_kwargs["timeout"] == 3600
    assert transport_timeout == httpx.Timeout(override).as_dict()


@pytest.mark.parametrize("endpoint", ["completion", "chat"])
def test_existing_sampling_timeout_override_is_preserved(monkeypatch, endpoint):
    _, client_kwargs, transport_timeout = run_request(monkeypatch, endpoint, sampling_timeout=750.0)
    assert client_kwargs["timeout"] == 3600
    assert transport_timeout == httpx.Timeout(750.0).as_dict()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf"), "", "invalid", True, False, object()])
def test_invalid_explicit_timeout_fails_before_client_creation(monkeypatch, value):
    def unexpected_client(**_kwargs):
        pytest.fail("invalid timeout reached client construction")

    monkeypatch.setattr(openai, "AsyncOpenAI", unexpected_client)
    monkeypatch.setenv(TIMEOUT_ENV, "3600")
    with pytest.raises(ValueError, match="positive finite number"):
        OpenAIEngine(request_timeout_s=value)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "", "invalid"])
def test_invalid_environment_timeout_fails_before_client_creation(monkeypatch, value):
    def unexpected_client(**_kwargs):
        pytest.fail("invalid timeout reached client construction")

    monkeypatch.setattr(openai, "AsyncOpenAI", unexpected_client)
    monkeypatch.setenv(TIMEOUT_ENV, value)
    with pytest.raises(ValueError, match="positive finite number"):
        OpenAIEngine()
