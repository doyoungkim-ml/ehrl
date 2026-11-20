"""Offline regression of the real Qwen parser's exact raw completion history.

The tokenizer is a tiny text fixture. These tests protect reconstruction behavior;
actual model tokenizer/prefix equivalence remains a separate integration check.
"""

import pytest

from rllm.parser.chat_template_parser import QwenChatTemplateParser


class TextTokenizer:
    bos_token = None
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        return "<assistant>" if add_generation_prompt else ""

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return tokens


@pytest.mark.parametrize("disable_thinking", [False, True])
@pytest.mark.parametrize(
    "raw",
    [
        "\n  deliberate reasoning  \n</think>\n\n action  ",
        "<think>\n<think>\nreasoning without close",
        "unfinished reasoning\n",
        "answer<|im_end|>",
        "answer<|im_end|>\n",
        "",
    ],
)
def test_history_keeps_raw_tokens_whitespace_and_prefill(raw, disable_thinking):
    parser = QwenChatTemplateParser(TextTokenizer(), disable_thinking=disable_thinking)
    message = {"role": "assistant", "content": "filtered action", "reasoning": "trimmed", "raw_completion": raw}
    expected_terminal = "" if raw.endswith(("<|im_end|>", "<|im_end|>\n")) else "<|im_end|>\n"
    reconstructed = parser.parse_assistant(message, accumulate_reasoning=True)
    assert reconstructed == parser.generation_prompt + raw + expected_terminal
    assert message["raw_completion"] == raw


def test_filtered_action_can_differ_from_exact_history():
    parser = QwenChatTemplateParser(TextTokenizer())
    raw = "  reasoning\n</think>\n\n execute() <|im_end|>"
    parsed = parser.parse_completion(raw)
    assert parsed["content"] == "execute()"
    history = parser.parse_assistant({"raw_completion": raw, **parsed}, accumulate_reasoning=True)
    assert history == parser.generation_prompt + raw


def test_unfinished_reasoning_does_not_become_environment_action():
    parser = QwenChatTemplateParser(TextTokenizer())
    parsed = parser.parse_completion("  continue thinking\n")
    assert parsed["content"] == ""
    assert parsed["reasoning"] == "continue thinking"


def test_raw_history_is_opt_in():
    parser = QwenChatTemplateParser(TextTokenizer())
    result = parser.parse_assistant({"raw_completion": "private reasoning", "content": "action"}, accumulate_reasoning=False)
    assert result == parser.assistant_token + "action" + parser.eot_token


@pytest.mark.parametrize("raw", ["\n  action  \n<|im_end|>", "action\n<|im_end|>\n", "", "unfinished action\n"])
def test_thinking_disabled_history_keeps_raw_content_without_reasoning_accumulation(raw):
    parser = QwenChatTemplateParser(TextTokenizer(), disable_thinking=True)
    message = {"role": "assistant", "content": "action", "raw_completion": raw}
    expected_terminal = "" if raw.endswith(("<|im_end|>", "<|im_end|>\n")) else parser.eot_token
    assert parser.parse_assistant(message, accumulate_reasoning=False) == parser.generation_prompt + raw + expected_terminal


@pytest.mark.parametrize("disable_thinking", [False, True])
def test_malformed_tool_call_does_not_crash_or_become_action_and_keeps_raw_history(disable_thinking):
    parser = QwenChatTemplateParser(TextTokenizer(), disable_thinking=disable_thinking)
    content = '\n  <tool_call>{"arguments": {"code": "print(1)"}}</tool_call>  <|im_end|>\n'
    raw = content if disable_thinking else "reasoning\n</think>" + content
    parsed = parser.parse_completion(raw)
    assert parsed["content"] == ""
    assert parsed["tool_calls"] == []
    history = parser.parse_assistant({"raw_completion": raw, **parsed}, accumulate_reasoning=True)
    assert history == parser.generation_prompt + raw


def test_malformed_tool_block_keeps_surrounding_appworld_code():
    parser = QwenChatTemplateParser(TextTokenizer(), disable_thinking=True)
    raw = '<tool_call>{"arguments": {"code": "print(1)"}}</tool_call>\n```python\nprint(2)\n```<|im_end|>'
    parsed = parser.parse_completion(raw)
    assert parsed["content"] == "```python\nprint(2)\n```"
    assert parsed["tool_calls"] == []
