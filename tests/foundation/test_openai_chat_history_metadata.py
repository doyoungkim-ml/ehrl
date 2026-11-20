"""Keep local Qwen history metadata out of chat requests, using the real SDK."""

import asyncio
import copy
import json

import httpx
import openai
from PIL import Image

from rllm.engine.rollout.openai_engine import OpenAIEngine


def test_chat_request_excludes_only_internal_history_fields(monkeypatch):
    real_client = openai.AsyncOpenAI
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "history-metadata-test",
                "object": "chat.completion",
                "created": 0,
                "model": "dummy",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    def create_client(**kwargs):
        return real_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai, "AsyncOpenAI", create_client)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {
            "role": "assistant",
            "name": "solver",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            "refusal": None,
            "audio": {"id": "audio_1"},
            "raw_completion": "raw sampled text\n<|im_end|>",
            "reasoning": "local parser metadata",
            "reasoning_content": "provider extension kept",
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    original = copy.deepcopy(messages)
    expected = [{key: value for key, value in message.items() if key not in {"raw_completion", "reasoning"}} for message in messages]
    engine = OpenAIEngine(model="dummy", api_key="unit-test", base_url="http://unit.test/v1", api_retries=1, request_timeout_s=30)

    async def invoke():
        try:
            return await engine.get_model_response(messages)
        finally:
            await engine.client.close()

    output = asyncio.run(invoke())
    assert output.content == "done"
    assert len(requests) == 1
    assert requests[0]["messages"] == expected
    assert messages == original
    assert engine._convert_messages_to_openai_format(messages)[1] is not messages[1]


def test_image_conversion_keeps_raw_history_local_and_input_unchanged():
    image = Image.new("RGB", (1, 1), color="white")
    message = {"role": "user", "content": "task", "images": [image], "raw_completion": "raw", "reasoning": "local"}
    engine = object.__new__(OpenAIEngine)
    converted = engine._convert_messages_to_openai_format([message])
    assert set(converted[0]) == {"role", "content"}
    assert converted[0]["content"][0] == {"type": "text", "text": "task"}
    assert converted[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert message == {"role": "user", "content": "task", "images": [image], "raw_completion": "raw", "reasoning": "local"}
