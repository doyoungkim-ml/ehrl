"""CPU integration of AppWorld uncertainty with real rollout methods and tensors."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch
from omegaconf import OmegaConf

from rllm.agents.agent import Episode
from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.engine import agent_execution_engine as engine_module
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.environments.appworld import appworld_env as env_module
from rllm.environments.appworld.appworld_env import AppWorldEnv
from rllm.environments.base.base_env import UncertainExecutionError
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

ROOT = Path(__file__).resolve().parents[2]


def response(output=None, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = "http://unused.test/execute"
    result._content = json.dumps({"output": output}).encode()
    return result


@pytest.fixture
def engine_case(monkeypatch):
    counts = {"initialize": 0, "execute": 0, "close": 0, "generation": 0}

    def post(url, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        counts[endpoint] += 1
        if endpoint == "initialize":
            return response({"instruction": "Synthetic task", "supervisor": {}})
        if endpoint == "execute":
            return response(status=500)
        assert endpoint == "close", "Uncertain actions must never be graded as completed"
        return response()

    monkeypatch.setattr(env_module.requests, "post", post)
    monkeypatch.setattr(engine_module, "convert_messages_to_tokens_and_masks", lambda *args, **kwargs: ([1], [1]))
    engine = AgentExecutionEngine.__new__(AgentExecutionEngine)
    engine.agents = [AppWorldReactAgent()]
    env = AppWorldEnv(task={"task_id": "synthetic"})
    env.idx = 0
    engine.envs = [env]
    engine.executor = ThreadPoolExecutor(max_workers=1)
    engine.max_steps = 30
    engine.max_prompt_length = 100
    engine.max_response_length = 100
    engine.trajectory_timeout = 30.0
    engine.trajectory_hard_timeout = 60.0
    engine.retry_limit = 3
    # Explicitly broad configurations must still not retry uncertain execution.
    engine.retry_on_termination_reasons = {"ENV_TIMEOUT", "ENV_ERROR", "ENV_EXECUTION_UNCERTAIN"}
    engine.enforce_max_prompt_length = False
    engine.accumulate_reasoning = False
    engine.overlong_filter = True
    engine.gamma = 0.0
    engine.trajectory_reward_aggregation = "sum"
    engine.config = OmegaConf.create({"rllm": {"filter_token_mismatch": False}})
    engine.chat_parser = SimpleNamespace(parse=lambda *args, **kwargs: "synthetic prompt")
    engine.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2], pad_token_id=0)

    async def generate(*args, **kwargs):
        counts["generation"] += 1
        return ModelOutput(text="Code:\n```python\nstate += 1\n```", prompt_ids=[1], completion_ids=[2, 3], tokens_for_gradient_update=2)

    engine.get_model_response = generate
    yield engine, env, counts
    engine.executor.shutdown(wait=True)


@pytest.mark.parametrize("mode", ["Text", "Token", "Step"])
@pytest.mark.parametrize("competing_limit", [None, "TRUNCATION", "TIMEOUT"])
def test_uncertainty_survives_eval_and_training_outputs_without_retry(engine_case, monkeypatch, mode, competing_limit):
    engine, env, counts = engine_case
    if competing_limit == "TRUNCATION":
        engine.max_response_length = 1
    elif competing_limit == "TIMEOUT":
        engine.trajectory_timeout = 2.0
        times = iter([0.0, 1.0, 1.0, 3.0])
        monkeypatch.setattr(engine_module, "time", SimpleNamespace(time=lambda: next(times)))

    result = asyncio.run(engine.run_agent_trajectory_with_retry(0, mode=mode))
    info = result.info if mode == "Text" else result
    assert info["termination_reason"] == "ENV_EXECUTION_UNCERTAIN"
    assert info["execution_uncertain"] is True
    assert info["execution_outcome"] == "unknown"
    assert info["execute_retry_attempted"] is False
    assert info["execution_http_status"] == 500
    assert info.get("limit_termination_reason") == competing_limit
    assert counts["execute"] == counts["initialize"] == counts["generation"] == 1
    assert env.done
    if mode == "Text":
        assert result.reward == 0 and len(result.steps) == 1
        assert result.steps[0].info["execution_uncertain"]
    else:
        assert result["trajectory_reward"] == 0 and result["all_passed"] is False
    if mode == "Token" and competing_limit is None:
        # The policy remains eligible for the existing training/selection rules.
        assert result["response_masks"].tolist() == [1, 1]


@pytest.mark.parametrize("deadline,error_type", [("soft", "TrajectoryStepTimeout"), ("hard", "TrajectoryHardTimeout")])
def test_deadline_while_execute_is_in_flight_fails_without_replay(engine_case, monkeypatch, deadline, error_type):
    engine, env, counts = engine_case
    entered = threading.Event()
    release = threading.Event()
    original_post = env_module.requests.post

    def blocked_post(url, **kwargs):
        if not url.endswith("/execute"):
            return original_post(url, **kwargs)
        counts["execute"] += 1
        entered.set()
        assert release.wait(3), "Test failed to release the synthetic server"
        return response("action was applied before the response arrived")

    monkeypatch.setattr(env_module.requests, "post", blocked_post)
    if deadline == "soft":
        engine.trajectory_timeout = 0.02
    else:
        engine.trajectory_hard_timeout = 0.02

    async def run():
        task = asyncio.create_task(engine.run_agent_trajectory_with_retry(0, mode="Text"))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            with pytest.raises(UncertainExecutionError) as caught:
                await task
            assert caught.value.info["execution_error_type"] == error_type
            assert caught.value.info["execution_outcome"] == "unknown"
        finally:
            release.set()

    asyncio.run(run())
    engine.executor.shutdown(wait=True)
    assert counts["execute"] == counts["initialize"] == counts["generation"] == 1
    assert env.execution_uncertainty["execution_error_type"] == error_type


def trainer_prefix(function_name, return_names):
    """Execute the actual trainer logging/filter block without GPU/Ray imports.

    The remaining tensor-batch/padding backend is outside this regression's
    scope. Real torch tensors and actual JSON-file writes are used here.
    """
    path = ROOT / "rllm/trainer/verl/agent_ppo_trainer.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "AgentPPOTrainer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "max_prompt_length" for t in n.targets))
    fn.body = [n for n in fn.body[:end] if not isinstance(n, ast.ImportFrom) or n.module != "verl.utils.torch_functional"]
    fn.body.append(ast.Return(ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Load()) for name in return_names], ctx=ast.Load())))
    namespace = {"torch": torch, "np": np, "os": os, "json": json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[function_name]


def test_trainer_persists_uncertainty_and_existing_zero_reward(engine_case, tmp_path):
    engine, _, _ = engine_case
    trajectory = asyncio.run(engine.run_agent_trajectory_with_retry(0, mode="Token"))
    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"default_local_dir": str(tmp_path), "trajectory_log_format": "enriched", "log_trajectories_structured": True}}),
        global_steps=1,
    )
    metrics, masks, rewards = trainer_prefix("_transform_agent_trajectories", ["metrics", "all_masks_list", "traj_scores"])(trainer, [trajectory])
    assert metrics["batch/term_ENV_EXECUTION_UNCERTAIN"] == 1.0
    assert rewards == [0.0] and masks[0].tolist() == [1, 1]
    enriched = json.loads((tmp_path / "chat_completions/1.jsonl").read_text())
    structured = json.loads((tmp_path / "trajectories/task_0/policy_step000001.json").read_text())["trajectories"][0]
    for saved in [enriched, structured]:
        assert saved["execution_uncertain"] is True
        assert saved["execution_outcome"] == "unknown"
        assert saved["execution_error_type"] == "HTTPError"
        assert saved["execute_retry_attempted"] is False
        assert saved["termination_reason"] == "ENV_EXECUTION_UNCERTAIN"
        assert saved["reward"] == 0.0


@pytest.mark.parametrize("overlong_filter", [True, False])
@pytest.mark.parametrize("competing_limit", [None, "TRUNCATION", "TIMEOUT"])
def test_stepwise_trainer_keeps_existing_overlong_filter(engine_case, overlong_filter, competing_limit):
    engine, _, _ = engine_case
    episode = asyncio.run(engine.run_agent_trajectory_with_retry(0, mode="Step"))
    if competing_limit:
        episode["limit_termination_reason"] = competing_limit
    trainer = SimpleNamespace(config=OmegaConf.create({"rllm": {"agent": {"overlong_filter": overlong_filter}}}), tokenizer=engine.tokenizer)
    (masked,) = trainer_prefix("_transform_agent_steps", ["all_steps_masked_out"])(trainer, [episode], ["synthetic"])
    assert masked == [bool(overlong_filter and competing_limit)]


class UncertainWorkflow(Workflow):
    async def run(self, task, uid, **kwargs):
        self.uid, self.task = uid, task
        observation, info = self.env.reset()
        self.agent.reset()
        self.agent.update_from_env(observation, 0.0, False, info)
        self.agent.update_from_model("Code:\n```python\nstate += 1\n```")
        observation, reward, done, info = self.env.step("state += 1")
        self.agent.update_from_env(observation, reward, done, info)
        # Generic workflows currently raise ENV_DONE for any done result.
        raise TerminationEvent(TerminationReason.ENV_DONE)


def test_workflow_postprocessing_preserves_flag_and_wrapper_does_not_retry(engine_case):
    engine, env, counts = engine_case
    workflow = UncertainWorkflow(rollout_engine=None, executor=engine.executor)
    workflow.agent = AppWorldReactAgent()
    workflow.env = env
    wrapper = AgentWorkflowEngine.__new__(AgentWorkflowEngine)
    wrapper.retry_limit = 3
    wrapper.raise_on_error = True
    wrapper.workflow_queue = asyncio.Queue()
    wrapper.workflow_queue.put_nowait(workflow)
    _, _, episode = asyncio.run(wrapper.process_task_with_retry({}, "synthetic", 0))
    assert episode.termination_reason == TerminationReason.ENV_EXECUTION_UNCERTAIN
    assert episode.info["execution_uncertain"] and episode.info["execution_outcome"] == "unknown"
    assert episode.trajectories[0].info["execution_uncertain"]
    assert not episode.is_correct and episode.trajectories[0].reward == 0
    assert counts["execute"] == counts["initialize"] == 1
    # The new enum survives the existing serialized episode interface.
    assert Episode.from_dict(episode.to_dict()).termination_reason == TerminationReason.ENV_EXECUTION_UNCERTAIN
