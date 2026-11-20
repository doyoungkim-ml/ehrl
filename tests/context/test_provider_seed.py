"""Seed receipts must agree with real OpenAI SDK request bodies."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.utils.provider_seed import apply_provider_seed, derive_provider_seed, provider_seed_identity, provider_seed_scope, provider_usage_totals, record_provider_request, record_provider_seeds


@pytest.mark.parametrize("endpoint", ["completion", "chat"])
def test_sdk_receipt_seed_and_hash_match_wire_body(monkeypatch, endpoint):
    original = openai.AsyncOpenAI
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        choice = {"index": 0, "finish_reason": "stop"}
        if endpoint == "completion":
            choice.update(text="answer", token_ids=[7], logprobs=None)
        else:
            choice["message"] = {"role": "assistant", "content": "answer"}
        return httpx.Response(200, json={"id": "test", "object": "text_completion" if endpoint == "completion" else "chat.completion", "created": 0, "model": "dummy", "choices": [choice], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: original(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    parser = SimpleNamespace(parse=lambda *_args, **_kwargs: [11, 12], parse_completion=lambda _ids: {"content": "answer", "reasoning": "", "tool_calls": []})
    engine = OpenAIEngine(model="dummy", tokenizer=SimpleNamespace() if endpoint == "completion" else None, chat_parser=parser, api_key="test", base_url="http://unit.test/v1", sampling_params={"temperature": 0.7, "seed": 42, "extra_body": {"seed": 43}})
    identity = provider_seed_identity({"scheme": "sha256_v1", "namespace": "handoff/appworld", "evaluation_seed": 711}, "alpha", 2, call=3)

    async def run():
        try:
            with record_provider_seeds() as records:
                with provider_seed_scope(identity):
                    await engine.get_model_response([{"role": "user", "content": "test"}])
                return records
        finally:
            await engine.client.close()

    records = asyncio.run(run())
    assert len(requests) == len(records) == 1
    assert requests[0]["seed"] == records[0]["seed"] == derive_provider_seed(identity)
    digest = hashlib.sha256(json.dumps(requests[0], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    assert records[0]["request_sha256"] == digest
    assert records[0]["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    assert engine.sampling_params["seed"] == 42
    assert engine.sampling_params["extra_body"]["seed"] == 43


def test_seed_schedule_distinguishes_replicate_role_task_visit_and_turn():
    config = {"scheme": "sha256_v1", "namespace": "handoff/appworld", "evaluation_seed": 711}
    identities = [provider_seed_identity(config, "alpha", 0)]
    identities += [provider_seed_identity({**config, "evaluation_seed": 712}, "alpha", 0)]
    identities += [provider_seed_identity(config, "beta", 0), provider_seed_identity(config, "alpha", 1)]
    identities += [provider_seed_identity(config, "alpha", 0, role=role) for role in ("reflector", "curator")]
    identities += [provider_seed_identity(config, "alpha", 0, call=1)]
    assert len(set(map(derive_provider_seed, identities))) == len(identities)


def test_usage_includes_returned_response_before_parse_failure_and_retry(monkeypatch):
    original = openai.AsyncOpenAI
    wire = []

    def respond(request):
        wire.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "text_completion", "created": 0, "model": "dummy", "choices": [{"index": 0, "finish_reason": "stop", "text": "answer", "token_ids": [7], "logprobs": None}], "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}})

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: original(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))))
    parse_calls = []

    def parse(ids):
        parse_calls.append(ids)
        if len(parse_calls) == 1:
            raise ValueError("retry after a returned provider response")
        return {"content": "answer", "reasoning": "", "tool_calls": []}

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    parser = SimpleNamespace(parse=lambda *_args, **_kwargs: [11], parse_completion=parse)
    engine = OpenAIEngine(model="dummy", tokenizer=SimpleNamespace(), chat_parser=parser, api_key="test", base_url="http://unit.test/v1", api_retries=2)
    identity = provider_seed_identity({"scheme": "sha256_v1", "namespace": "handoff/appworld", "evaluation_seed": 711}, "alpha", 0, role="curator")

    async def run():
        try:
            with record_provider_seeds() as records:
                with provider_seed_scope(identity):
                    await engine.get_model_response([{"role": "user", "content": "test"}])
                return records
        finally:
            await engine.client.close()

    records = asyncio.run(run())
    assert len(records) == len(wire) == 2
    assert records[0]["seed"] == records[1]["seed"]
    usage = provider_usage_totals(records)
    assert usage["curator"]["prompt_tokens"] == 26
    assert usage["curator"]["completion_tokens"] == 10
    assert usage["total"]["usage_unknown_requests"] == 0


def test_dispatch_without_response_keeps_total_usage_unknown():
    identity = provider_seed_identity({"scheme": "sha256_v1", "namespace": "handoff/appworld", "evaluation_seed": 711}, "alpha", 0)
    with record_provider_seeds() as records, provider_seed_scope(identity):
        sampling = {}
        apply_provider_seed(sampling)
        record_provider_request(endpoint="completion", model="dummy", sampling_params=sampling, payload={"model": "dummy", **sampling})
    totals = provider_usage_totals(records)
    assert totals["total"]["usage_unknown_requests"] == 1
    assert totals["total"]["prompt_tokens"] is totals["total"]["completion_tokens"] is None
    assert totals["total"]["observed_completion_tokens"] == 0
