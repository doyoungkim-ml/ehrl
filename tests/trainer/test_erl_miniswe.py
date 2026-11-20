"""Execute the ERL adapter with the real MiniSweWorkflow and SWEAgent on CPU."""

import ast
import asyncio
import copy
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_erl_original import ROOT

from examples.mini_swe_bench.train.erl.adapter import run_attempt
from rllm.agents.swe_agent import SWEAgent
from rllm.algorithms.erl.agent import PromptOverrideAgent
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.environments.base.base_env import BaseEnv


class TextTokenizer:
    @staticmethod
    def encode(text, **kwargs):
        return list(text.encode())

    @staticmethod
    def decode(tokens, **kwargs):
        return bytes(tokens).decode()


class Parser:
    @staticmethod
    def parse(messages, **kwargs):
        text = ""
        for message in messages:
            role = message["role"]
            raw = message.get("raw_completion")
            text += f"<{role}>" + (raw if raw else message["content"] + f"</{role}>")
        return text + "<assistant>"


class Rollout:
    tokenizer = TextTokenizer()
    chat_parser = Parser()
    max_response_length = 20000

    def __init__(self):
        self.requests = []

    async def get_model_response(self, messages, **kwargs):
        self.requests.append((copy.deepcopy(messages), kwargs))
        action = "<function=bash><parameter=command>echo hi</parameter></function>" if len(self.requests) == 1 else "<function=submit></function>"
        prompt = self.tokenizer.encode(self.chat_parser.parse(messages))
        response = self.tokenizer.encode(action + "</assistant>")
        return ModelOutput(text=action + "</assistant>", content=action, prompt_ids=prompt, completion_ids=response, finish_reason="stop", tokens_for_gradient_update=len(response))


class Environment(BaseEnv):
    def __init__(self):
        self.task = {"instance": {"problem_statement": "Repair a bug"}}
        self.closed = 0

    def reset(self, task=None):
        self.step_count = 0
        return "Repair a bug", {"max_steps": 2}

    def step(self, action):
        self.step_count += 1
        if self.step_count == 1:
            return "observed output", 0, False, {"max_steps": 2}
        return "submitted", 0.2, True, {"max_steps": 2, "git_diff": "+fixed", "reward_reason": "tests_failed", "test_counts": {"ftp_passed": 3, "ftp_total": 4, "ptp_passed": 2, "ptp_total": 3}}

    def close(self):
        self.closed += 1

    @staticmethod
    def is_multithread_safe():
        return True

    @staticmethod
    def from_dict(payload):
        return Environment()


def real_assembler():
    path = ROOT / "rllm/engine/agent_execution_engine.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "AgentExecutionEngine")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "assemble_steps")
    from rllm.trainer.verl.rollout_correction import usable_step_log_probs

    namespace = {"torch": torch, "logger": logging.getLogger(__name__), "usable_step_log_probs": usable_step_log_probs}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["assemble_steps"]


@pytest.mark.parametrize("override", [None, "a revised solver prompt"])
def test_real_workflow_keeps_plain_scaffold_grading_cumulative_budget_and_tokens(override):
    agent = SWEAgent(scaffold="sweagent")
    if override:
        agent = PromptOverrideAgent(agent, override)
    env = Environment()
    rollout = Rollout()
    subject = SimpleNamespace(global_steps=1, config=OmegaConf.create({"rllm": {"workflow": {"workflow_args": {"reward_mode": "frac3", "frac3_lambda": 0.1}}}}))
    with ThreadPoolExecutor(max_workers=2) as executor:
        engine = SimpleNamespace(
            agents=[agent],
            envs=[env],
            executor=executor,
            rollout_engine=rollout,
            max_steps=2,
            trajectory_timeout=10,
            retry_limit=1,
            max_response_length=20000,
            max_prompt_length=20000,
            tokenizer=TextTokenizer(),
            config=OmegaConf.create({"rllm": {"filter_token_mismatch": True}}),
        )
        engine.assemble_steps = lambda steps, **options: real_assembler()(engine, steps, **options)
        result = asyncio.run(run_attempt(subject, engine, 0, {"validate": True}))
    assert len(rollout.requests) == 2
    assert all(kwargs["validate"] is True and kwargs["accumulate_reasoning"] is True for _, kwargs in rollout.requests)
    assert rollout.requests[1][1]["max_tokens"] < rollout.requests[0][1]["max_tokens"]
    assert result["trajectory_reward"] == pytest.approx(0.65)  # actual MiniSweRewardMixin frac3
    assert result["all_passed"] is False
    assert result["metrics"]["token_mismatch"] == 0
    assert 0 < result["response_masks"].sum() < len(result["response_masks"])
    assert env.closed == 1
    if override:
        assert rollout.requests[0][0][0]["content"] == override
    else:
        assert rollout.requests[0][0][0]["content"] == agent.system_prompt
    assert "Repair a bug" in rollout.requests[0][0][1]["content"]
