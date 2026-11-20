"""Sampler log-probabilities align with assembled multi-turn training tokens.

The invariant checked for every available token at response position j: the
trainer's context (prompt + response[:j]) equals the sampled step's
prompt_ids + completion_ids[:k], the token is completion_ids[k], and its
log-probability is that step's logprobs[k].
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from rllm.engine.agent_execution_engine import AgentExecutionEngine  # noqa: E402


class PieceTokenizer:
    """decode() joins fixed text pieces, so different IDs can spell equal text."""

    pieces = {20: "ab", 21: "a", 22: "b", 151667: "<think>", 198: "\n"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces.get(token, f"<{token}>") for token in ids)


def engine(max_response_length=100, filter_token_mismatch=True):
    subject = AgentExecutionEngine.__new__(AgentExecutionEngine)
    subject.max_response_length = max_response_length
    subject.tokenizer = PieceTokenizer()
    subject.config = types.SimpleNamespace(rllm=types.SimpleNamespace(filter_token_mismatch=filter_token_mismatch))
    return subject


def step(prompt, completion, *, matched=True, base=0.0):
    return {"prompt_ids": prompt, "completion_ids": completion, "logprobs": [base - 0.01 * (i + 1) for i in range(len(completion))], "logprobs_match_training": matched}


def assemble(steps, **kwargs):
    prompt, response, masks, valid, log_probs, available = engine(**kwargs).assemble_steps(steps, return_rollout_log_probs=True)
    assert log_probs.shape == available.shape == response.shape == masks.shape
    assert ((available == 0) | (masks == 1)).all(), "available tokens must be trained action tokens"
    return prompt.tolist(), response.tolist(), masks.tolist(), valid, log_probs.tolist(), available.tolist()


def assert_exact_contexts(prompt, response, log_probs, available, steps):
    for j, flag in enumerate(available):
        if not flag:
            continue
        context = prompt + response[:j]
        matches = [
            s
            for s in steps
            if context[: len(s["prompt_ids"])] == s["prompt_ids"]
            and 0 <= len(context) - len(s["prompt_ids"]) < len(s["completion_ids"])
            and context[len(s["prompt_ids"]) :] == s["completion_ids"][: len(context) - len(s["prompt_ids"])]
        ]
        assert matches, f"available token {j} has no exactly matching sampled context"
        k = len(context) - len(matches[0]["prompt_ids"])
        assert response[j] == matches[0]["completion_ids"][k]
        assert log_probs[j] == pytest.approx(matches[0]["logprobs"][k])


def test_env_and_template_tokens_are_masked_and_completions_align():
    steps = [step([1, 2], [3, 4], base=-1), step([1, 2, 3, 4, 5, 9], [6, 7, 8], base=-2)]
    prompt, response, masks, valid, log_probs, available = assemble(steps)
    assert valid and prompt == [1, 2]
    assert response == [3, 4, 5, 9, 6, 7, 8]
    assert masks == [1, 1, 0, 0, 1, 1, 1]
    assert available == masks
    assert log_probs[2:4] == [0.0, 0.0]
    assert_exact_contexts(prompt, response, log_probs, available, steps)


def test_turn_sampled_with_other_parameters_keeps_weight_one():
    steps = [step([1], [3, 4]), step([1, 3, 4, 5], [6], matched=False), step([1, 3, 4, 5, 6, 7], [8])]
    prompt, response, masks, _, log_probs, available = assemble(steps)
    assert masks == [1, 1, 0, 1, 0, 1]
    assert available == [1, 1, 0, 0, 0, 1]
    assert_exact_contexts(prompt, response, log_probs, available, steps)


def test_logprob_count_mismatch_is_unavailable():
    broken = step([1], [3, 4])
    broken["logprobs"] = broken["logprobs"][:1]
    _, _, masks, _, _, available = assemble([broken])
    assert masks == [1, 1] and available == [0, 0]


def test_dropped_qwen_think_sentinel_invalidates_only_that_turn():
    # The next prompt omits the generated <think>\n; the trainer deletes it from
    # the response, so the rest of that completion is scored without it.
    steps = [step([1, 2], [151667, 198, 10, 11]), step([1, 2, 10, 11, 5], [12, 13])]
    prompt, response, masks, valid, log_probs, available = assemble(steps)
    assert valid
    assert response == [10, 11, 5, 12, 13]
    assert masks == [1, 1, 0, 1, 1]
    assert available == [0, 0, 0, 1, 1]
    assert_exact_contexts(prompt, response, log_probs, available, steps)


def test_retokenized_history_disables_every_later_turn():
    # [20] and [21, 22] decode to the same text; the sampler saw [21, 22].
    steps = [step([1], [20, 7]), step([1, 21, 22, 7, 5], [8, 9]), step([1, 21, 22, 7, 5, 8, 9, 6], [30])]
    prompt, response, masks, valid, log_probs, available = assemble(steps)
    assert valid
    assert response == [20, 7, 5, 8, 9, 6, 30]
    assert masks == [1, 1, 0, 1, 1, 0, 1]
    assert available == [1, 1, 0, 0, 0, 0, 0]
    assert_exact_contexts(prompt, response, log_probs, available, steps)


def test_non_cumulative_trajectory_has_no_available_tokens():
    steps = [step([1], [3]), step([2, 9], [4])]
    _, _, masks, valid, _, available = assemble(steps)
    assert not valid and masks == [0] and available == [0]


def test_response_budget_trims_log_probs_with_tokens():
    steps = [step([1, 2], [3, 4]), step([1, 2, 3, 4, 5], [6, 7, 8])]
    prompt, response, masks, _, log_probs, available = assemble(steps, max_response_length=5)
    assert response == [3, 4, 5, 6, 7]
    assert available == [1, 1, 0, 1, 1]
    assert_exact_contexts(prompt, response, log_probs, available, steps)


def test_default_return_signature_is_unchanged():
    result = engine().assemble_steps([step([1], [2])])
    assert len(result) == 4
