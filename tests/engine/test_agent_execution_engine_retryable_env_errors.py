from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

from rllm.environments.base.base_env import BaseEnv


class RetryableResetError(RuntimeError):
    pass


class NonRetryableResetError(RuntimeError):
    pass


class FakeRetryableEnv(BaseEnv):
    def __init__(self) -> None:
        self.close_calls = 0

    def reset(self):
        return {}, {}

    def step(self, action):
        return {}, 0.0, True, {}

    def close(self):
        self.close_calls += 1

    @staticmethod
    def from_dict(info: dict):
        return FakeRetryableEnv()

    @staticmethod
    def is_retryable_error(exc: BaseException) -> bool:
        return isinstance(exc, RetryableResetError)


def _load_agent_execution_engine(monkeypatch):
    # The local test environment does not install torch; the retry loop tested
    # here does not touch tensor APIs, so a minimal import stub is enough.
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    sys.modules.pop("rllm.engine.agent_execution_engine", None)
    module = importlib.import_module("rllm.engine.agent_execution_engine")
    return module.AgentExecutionEngine


def _make_engine(monkeypatch):
    agent_execution_engine = _load_agent_execution_engine(monkeypatch)
    engine = agent_execution_engine.__new__(agent_execution_engine)
    engine.retry_limit = 3
    engine.retry_on_termination_reasons = {"ENV_TIMEOUT", "ENV_ERROR"}
    engine.trajectory_hard_timeout = 7200.0
    env = FakeRetryableEnv()
    engine.envs = [env]
    return engine, env


def test_agent_execution_engine_retries_env_classified_reset_errors(monkeypatch):
    engine, env = _make_engine(monkeypatch)
    attempts = 0

    async def fake_run_agent_trajectory_async(idx, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RetryableResetError("reset timed out")
        return {"termination_reason": "ENV_DONE"}

    engine.run_agent_trajectory_async = fake_run_agent_trajectory_async

    result = asyncio.run(engine.run_agent_trajectory_with_retry(0))

    assert result == {"termination_reason": "ENV_DONE"}
    assert attempts == 3
    assert env.close_calls == 2


def test_agent_execution_engine_does_not_retry_unclassified_reset_errors(monkeypatch):
    engine, env = _make_engine(monkeypatch)
    attempts = 0

    async def fake_run_agent_trajectory_async(idx, **kwargs):
        nonlocal attempts
        attempts += 1
        raise NonRetryableResetError("bad task config")

    engine.run_agent_trajectory_async = fake_run_agent_trajectory_async

    with pytest.raises(NonRetryableResetError):
        asyncio.run(engine.run_agent_trajectory_with_retry(0))

    assert attempts == 1
    assert env.close_calls == 0


def test_appworld_marks_http_timeouts_retryable():
    requests = pytest.importorskip("requests")
    from rllm.environments.appworld.appworld_env import AppWorldEnv

    assert AppWorldEnv.is_retryable_error(requests.exceptions.ReadTimeout("read timed out"))
    assert AppWorldEnv.is_retryable_error(requests.exceptions.ConnectionError("connection reset"))
    assert not AppWorldEnv.is_retryable_error(ValueError("task_id missing"))
