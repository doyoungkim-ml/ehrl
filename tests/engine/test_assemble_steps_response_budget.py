from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from rllm.engine.agent_execution_engine import AgentExecutionEngine  # noqa: E402


def _make_engine(max_response_length: int) -> AgentExecutionEngine:
    engine = AgentExecutionEngine.__new__(AgentExecutionEngine)
    engine.max_response_length = max_response_length
    engine.tokenizer = None
    engine.config = types.SimpleNamespace(rllm=types.SimpleNamespace(filter_token_mismatch=True))
    return engine


def _steps() -> list[dict]:
    # Two turns whose raw ids accumulate exactly: prompt [1,2], completion [3,4],
    # env observation [5], completion [6,7,8] -> six response tokens.
    return [
        {"prompt_ids": [1, 2], "completion_ids": [3, 4]},
        {"prompt_ids": [1, 2, 3, 4, 5], "completion_ids": [6, 7, 8]},
    ]


def test_assemble_steps_trims_assembled_response_to_budget():
    engine = _make_engine(max_response_length=5)

    prompt_tokens, response_tokens, response_masks, is_valid = engine.assemble_steps(_steps())

    assert is_valid
    assert prompt_tokens.tolist() == [1, 2]
    assert response_tokens.tolist() == [3, 4, 5, 6, 7]
    assert response_masks.tolist() == [1, 1, 0, 1, 1]


def test_assemble_steps_keeps_response_within_budget_untouched():
    engine = _make_engine(max_response_length=6)

    _, response_tokens, response_masks, is_valid = engine.assemble_steps(_steps())

    assert is_valid
    assert response_tokens.tolist() == [3, 4, 5, 6, 7, 8]
    assert response_masks.tolist() == [1, 1, 0, 1, 1, 1]
