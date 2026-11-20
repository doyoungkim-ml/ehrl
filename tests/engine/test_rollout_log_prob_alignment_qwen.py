"""Rollout log-prob alignment with the real Qwen3 tokenizer and chat template.

Requires a locally cached Qwen3 tokenizer (all Qwen3 sizes share it); the test
never downloads. It replays the VerlEngine request path: prompt IDs are the
encoded rLLM chat-parser prompt and completion IDs are the exact raw
completion, then checks that every available sampler log-prob sits on the
token and context it was sampled with.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from rllm.agents.swe_agent import SWEAgent  # noqa: E402
from rllm.engine.rollout.rollout_engine import ModelOutput  # noqa: E402
from rllm.parser import QwenChatTemplateParser  # noqa: E402

from .test_rollout_log_prob_alignment import assert_exact_contexts, engine  # noqa: E402

TOKENIZER = os.environ.get("RLLM_TEST_QWEN3_TOKENIZER", "Qwen/Qwen3-1.7B")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        pytest.skip(f"{TOKENIZER} tokenizer is not cached locally")


def sampled_step(tokenizer, parser, messages, raw_completion, accumulate):
    prompt = parser.parse(messages, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=accumulate)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(raw_completion, add_special_tokens=False)
    return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": [-0.001 * (i + 1) for i in range(len(completion_ids))], "logprobs_match_training": True}


def conversation(tokenizer, accumulate, disable_thinking=False):
    parser = QwenChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
    messages = [{"role": "system", "content": "Solve the task."}, {"role": "user", "content": "Write add(a, b)."}]
    raws = [
        "First, the signature.\n</think>\n\n```python\ndef add(a, b):\n    return a + b\n```<|im_end|>",
        "The test passed.\n</think>\n\nDone.<|im_end|>",
    ]
    if disable_thinking:
        raws = ["```python\ndef add(a, b):\n    return a + b\n```<|im_end|>", "Done.<|im_end|>"]
    steps = []
    for index, raw in enumerate(raws):
        steps.append(sampled_step(tokenizer, parser, messages, raw, accumulate))
        content = raw.split("</think>\n\n")[-1].removesuffix("<|im_end|>")
        messages = messages + [{"role": "assistant", "content": content, "raw_completion": raw}, {"role": "user", "content": f"Observation {index}: all tests passed."}]
    return steps


@pytest.mark.parametrize("disable_thinking", [False, True])
def test_exact_raw_history_keeps_every_completion_token_available(tokenizer, disable_thinking):
    steps = conversation(tokenizer, accumulate=True, disable_thinking=disable_thinking)
    subject = engine()
    subject.tokenizer = tokenizer
    prompt, response, masks, valid, log_probs, available = subject.assemble_steps(steps, return_rollout_log_probs=True)
    assert valid
    assert available.tolist() == masks.tolist()
    assert int(available.sum()) == sum(len(s["completion_ids"]) for s in steps)
    # Template tokens between turns (<|im_start|>user ... <|im_start|>assistant\n<think>\n) are masked.
    gap = response.tolist()[len(steps[0]["completion_ids"]) : len(response) - len(steps[1]["completion_ids"])]
    assert gap and all(flag == 0 for flag in available.tolist()[len(steps[0]["completion_ids"]) : len(steps[0]["completion_ids"]) + len(gap)])
    assert_exact_contexts(prompt.tolist(), response.tolist(), log_probs.tolist(), available.tolist(), steps)


def test_lossy_history_never_reuses_sampler_log_probs(tokenizer):
    # Dropping the thinking text from history changes the next prompt.
    steps = conversation(tokenizer, accumulate=False)
    subject = engine()
    subject.tokenizer = tokenizer
    prompt, response, masks, valid, log_probs, available = subject.assemble_steps(steps, return_rollout_log_probs=True)
    later = available.tolist()[len(steps[0]["completion_ids"]) :]
    assert not any(later)
    assert_exact_contexts(prompt.tolist(), response.tolist(), log_probs.tolist(), available.tolist(), steps)


@pytest.mark.parametrize(
    "raw, expected_action, exact_token_prefix",
    [
        (
            "Inspect the handler.\n<function=execute_bash>\n<parameter=command>pwd</parameter>\n</function>\n<|im_end|>",
            "<function=execute_bash>\n<parameter=command>pwd</parameter>\n</function>",
            True,
        ),
        (
            "\n  <function=execute_bash><parameter=command>pwd</parameter></function>  <|im_end|>",
            "<function=execute_bash><parameter=command>pwd</parameter></function>",
            False,
        ),
    ],
)
def test_swe_thinking_disabled_history_preserves_sampled_whitespace(tokenizer, raw, expected_action, exact_token_prefix):
    # The live MiniSWE strict-gate run disables thinking and reasoning
    # accumulation. Its completions can end in </function>\n<|im_end|>;
    # action parsing strips that newline, but the next sampled prompt must not.
    parser = QwenChatTemplateParser(tokenizer, disable_thinking=True)
    agent = SWEAgent(scaffold="sweagent")
    agent.update_from_env("Inspect the repository.", 0.0, False, {})
    first = sampled_step(tokenizer, parser, agent.chat_completions, raw, accumulate=False)
    output = ModelOutput(text=raw, **parser.parse_completion(first["completion_ids"]))
    action = agent.update_from_model(raw, model_output=output)
    assert action.action == expected_action
    assert agent.chat_completions[-1]["raw_completion"] == raw
    assert agent.chat_completions[-1]["content"] != raw
    agent.update_from_env("/testbed", 0.0, False, {})
    second = sampled_step(tokenizer, parser, agent.chat_completions, "<function=submit></function><|im_end|>", accumulate=False)
    subject = engine(max_response_length=32768)
    subject.tokenizer = tokenizer
    prompt, response, masks, valid, log_probs, available = subject.assemble_steps([first, second], return_rollout_log_probs=True)
    assert valid
    if exact_token_prefix:
        assert available.tolist() == masks.tolist()
    assert int(masks.sum()) == len(first["completion_ids"]) + len(second["completion_ids"])
    assert_exact_contexts(prompt.tolist(), response.tolist(), log_probs.tolist(), available.tolist(), [first, second])
