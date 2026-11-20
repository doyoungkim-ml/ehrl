"""Real agent-to-parser-to-assembly coverage with an offline text tokenizer.

These fixtures test history preservation, not a model tokenizer or live rollout.
Every sampled prompt is encoded from the agent's current production history.
"""

from types import SimpleNamespace

import pytest

from rllm.agents.agent import make_assistant_message
from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.agents.appworld_react_agents_ace import AppWorldReactAgentACE
from rllm.agents.mastermind_agent import MastermindAgent
from rllm.agents.mbpp_agent import MBPPAgent
from rllm.agents.swe_agent import SWEAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout import ModelOutput
from rllm.parser.chat_template_parser import QwenChatTemplateParser


class TextTokenizer:
    """One character per token; no model files, downloads, or BPE claims."""

    bos_token = None
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        return "<assistant>" if add_generation_prompt else ""

    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, tokens, *, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(chr(token) for token in tokens)


AGENTS = ("appworld", "appworld_ace", "mbpp", "mastermind", "miniswe")


def initialized_agent(name, accumulate_thinking=False):
    if name in ("appworld", "appworld_ace"):
        cls = AppWorldReactAgentACE if name == "appworld_ace" else AppWorldReactAgent
        agent = cls(accumulate_thinking=accumulate_thinking)
        agent.update_from_env({"instruction": "Print ok"}, 0.0, False, {})
        return agent, "```python\nprint('ok')\n```", "print('ok')"
    if name == "mbpp":
        agent = MBPPAgent(accumulate_thinking=accumulate_thinking)
        agent.update_from_env({"task_description": "Print ok"}, 0.0, False, {})
        content = "```python\nprint('ok')\n```"
        return agent, content, content
    if name == "mastermind":
        agent = MastermindAgent(use_accumulate_thinking=accumulate_thinking)
        agent.update_from_env({"instructions": "Guess", "code_length": 4, "num_digits": 6}, 0.0, False, {})
        return agent, "```1234```", "1234"
    agent = SWEAgent(scaffold="sweagent")
    agent.update_from_env("Inspect the repository.", 0.0, False, {})
    content = "<function=execute_bash><parameter=command>pwd</parameter></function>"
    return agent, content, content


def sampled_step(tokenizer, parser, messages, raw, *, accumulate_reasoning=False):
    prompt = parser.parse(messages, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=accumulate_reasoning)
    completion_ids = tokenizer.encode(raw, add_special_tokens=False)
    return {
        "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False),
        "completion_ids": completion_ids,
        "logprobs": [-0.001 * (index + 1) for index in range(len(completion_ids))],
        "logprobs_match_training": True,
    }


@pytest.mark.parametrize("name", AGENTS)
@pytest.mark.parametrize("suffix", ["\n<|im_end|>", "\n<|im_end|>\n", "\n"])
def test_thinking_disabled_real_agent_history_preserves_sampled_contexts(name, suffix):
    tokenizer = TextTokenizer()
    parser = QwenChatTemplateParser(tokenizer, disable_thinking=True)
    agent, content, expected_action = initialized_agent(name)
    raw = "\n  " + content + suffix
    first = sampled_step(tokenizer, parser, agent.chat_completions, raw)
    output = ModelOutput(text=raw, **parser.parse_completion(first["completion_ids"]))
    action = agent.update_from_model(raw, model_output=output)
    assert action.action == expected_action
    message = agent.chat_completions[-1]
    assert message["content"] == content
    assert message["raw_completion"] == raw
    assert agent.trajectory.steps[-1].model_response == content
    snapshot = agent.trajectory.steps[-1].chat_completions
    assert snapshot[-1]["raw_completion"] == raw

    agent.update_from_env("Observation: continue.", 0.0, False, {})
    second = sampled_step(tokenizer, parser, agent.chat_completions, content + "<|im_end|>")
    # Freshly rendered context must contain the exact first sampled completion.
    sampled_prefix = first["prompt_ids"] + first["completion_ids"]
    assert second["prompt_ids"][: len(sampled_prefix)] == sampled_prefix
    subject = AgentExecutionEngine.__new__(AgentExecutionEngine)
    subject.tokenizer = tokenizer
    subject.max_response_length = 32768
    subject.config = SimpleNamespace(rllm=SimpleNamespace(filter_token_mismatch=True))
    prompt, response, masks, valid, logprobs, available = subject.assemble_steps([first, second], return_rollout_log_probs=True)
    assert valid
    assert int(masks.sum()) == len(first["completion_ids"]) + len(second["completion_ids"])
    assert available.tolist() == masks.tolist()
    for step in (first, second):
        start = len(step["prompt_ids"]) - len(prompt)
        assert prompt.tolist() + response[:start].tolist() == step["prompt_ids"]
        assert response[start : start + len(step["completion_ids"])].tolist() == step["completion_ids"]
        assert logprobs[start : start + len(step["completion_ids"])].tolist() == pytest.approx(step["logprobs"])
    agent.reset()
    assert snapshot[-1]["raw_completion"] == raw


@pytest.mark.parametrize("name", AGENTS)
@pytest.mark.parametrize("accumulate_reasoning", [False, True])
def test_thinking_enabled_parser_still_controls_reasoning_visibility(name, accumulate_reasoning):
    tokenizer = TextTokenizer()
    parser = QwenChatTemplateParser(tokenizer, disable_thinking=False)
    agent, content, expected_action = initialized_agent(name, accumulate_thinking=accumulate_reasoning)
    raw = "private reasoning\n</think>\n\n" + content + "\n<|im_end|>"
    output = ModelOutput(text=raw, **parser.parse_completion(tokenizer.encode(raw)))
    assert agent.update_from_model(raw, model_output=output).action == expected_action
    message = agent.chat_completions[-1]
    assert message["raw_completion"] == raw
    rendered = parser.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
    if accumulate_reasoning:
        assert rendered == parser.generation_prompt + raw
    else:
        assert "private reasoning" not in rendered
        assert rendered == parser.parse_assistant({"role": "assistant", "content": content}, accumulate_reasoning=False)


@pytest.mark.parametrize("name", AGENTS[:-1])
def test_legacy_string_response_keeps_filtered_action_and_plain_history(name):
    agent, content, expected_action = initialized_agent(name)
    assert agent.update_from_model(content + "<|im_end|>").action == expected_action
    assert agent.chat_completions[-1] == {"role": "assistant", "content": content}


@pytest.mark.parametrize("tool", ['{"name":"execute","arguments":{"code":"ignored"}}', '{"arguments":{"code":"ignored"}}'])
def test_appworld_action_filters_tool_blocks_but_history_keeps_exact_raw_completion(tool):
    tokenizer = TextTokenizer()
    parser = QwenChatTemplateParser(tokenizer, disable_thinking=True)
    agent, _, _ = initialized_agent("appworld_ace")
    raw = '<tool_call>' + tool + '</tool_call>\n```python\nprint(2)\n```\n<|im_end|>'
    output = ModelOutput(text=raw, **parser.parse_completion(tokenizer.encode(raw)))
    assert agent.update_from_model(raw, model_output=output).action == "print(2)"
    message = agent.chat_completions[-1]
    assert message["content"] == "```python\nprint(2)\n```"
    assert parser.parse_assistant(message, accumulate_reasoning=False) == parser.generation_prompt + raw


def test_shared_message_helper_retains_its_explicit_preservation_contract():
    output = ModelOutput(text=" raw<|im_end|>", content="visible", reasoning="thought")
    assert make_assistant_message("visible", output) == {"role": "assistant", "content": "visible"}
    assert make_assistant_message("visible", output, preserve_raw_completion=True)["raw_completion"] == output.text
    assert make_assistant_message("visible", None, preserve_raw_completion=True) == {"role": "assistant", "content": "visible"}
