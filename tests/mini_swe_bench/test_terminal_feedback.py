"""Real Token engine → ACE reconstruction → MiniSWE reflector feedback on CPU.

Only model generation, tokenization and external container/grading are doubles.
The environment's submit step, agent, engine and reconstruction remain real.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from examples.mini_swe_bench.train.ace_rl.adapter import MiniSweReflectorAdapter
from examples.mini_swe_bench.train.ace_rl.agent import MiniSweACEAgent
from examples.mini_swe_bench.train.ace_rl.env import MiniSweACEEnv
from rllm.engine import agent_execution_engine as engine_module
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.engine.terminal_feedback import sanitize_terminal_feedback

ROOT = Path(__file__).resolve().parents[2]


def reconstruction_controller(adapter):
    """Execute complete production prompt methods, excluding GPU imports only."""
    path = ROOT / "rllm/algorithms/ace/trainer_base.py"
    tree = ast.parse(path.read_text())
    render = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_render_template")
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ACERLAgentPPOTrainer")
    cls.bases = []
    names = {"_trajectory_steps_from_chat", "_build_reflector_prompt_from_dict"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"SimpleNamespace": SimpleNamespace, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[render, cls], type_ignores=[]), str(path), "exec"), namespace)
    trainer = namespace["ACERLAgentPPOTrainer"]()
    trainer.reflector_adapter = adapter
    trainer.reflector_prompt_template = "{{trajectory_summary}}"
    trainer.reflector_system_prompt = "reflect"
    trainer.ace_config = {}
    trainer.component_token_limits = {"reflector": {"max_prompt_length": 100_000}}
    trainer.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    trainer.agent_execution_engine = SimpleNamespace(rollout_engine=SimpleNamespace(chat_parser=None))
    trainer._get_policy_task_for_trajectory = lambda *_: {"problem_statement": "Repair simplify()"}
    return trainer


@pytest.fixture
def token_rollout(monkeypatch):
    """Generate terminal info through the actual submit implementation."""
    engines = []

    def run(reason, *, patch="diff --git a/PRIVATE_PATCH b/PRIVATE_PATCH", counts=None, preserve=True):
        env = MiniSweACEEnv(task={"instance": {"instance_id": "sympy__sympy-1", "problem_statement": "Repair simplify()"}}, reward_mode="binary")
        env.idx = 0
        env._last_test_counts = counts or {}
        env._last_tests_status = {"FAIL_TO_PASS": {"failure": ["PRIVATE_GOLD_TEST"]}}
        monkeypatch.setattr(env, "reset", lambda: ("Repair simplify()", {}))
        monkeypatch.setattr(env, "_get_git_diff", lambda: patch)
        monkeypatch.setattr(env, "_compute_reward", lambda: (1.0 if reason == "tests_passed" else 0.0, reason))
        monkeypatch.setattr(engine_module, "convert_messages_to_tokens_and_masks", lambda *args, **kwargs: ([1], [1]))
        engine = AgentExecutionEngine.__new__(AgentExecutionEngine)
        engine.agents = [MiniSweACEAgent()]
        engine.envs = [env]
        engine.executor = ThreadPoolExecutor(max_workers=1)
        engines.append(engine)
        engine.max_steps = 30
        engine.max_prompt_length = 100
        engine.max_response_length = 100
        engine.trajectory_timeout = 30.0
        engine.enforce_max_prompt_length = False
        engine.accumulate_reasoning = False
        engine.overlong_filter = False
        engine.gamma = 0.0
        engine.trajectory_reward_aggregation = "sum"
        engine.preserve_terminal_feedback = preserve
        engine.config = OmegaConf.create({"rllm": {"filter_token_mismatch": False}})
        engine.chat_parser = SimpleNamespace(parse=lambda *args, **kwargs: "prompt")
        engine.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2], pad_token_id=0)

        async def generate(*args, **kwargs):
            return ModelOutput(text="<function=submit></function>", prompt_ids=[1], completion_ids=[2], tokens_for_gradient_update=1)

        engine.get_model_response = generate
        result = asyncio.run(engine.run_agent_trajectory_async(0, "test", mode="Token"))
        return result, engine.agents[0].trajectory

    yield run
    for engine in engines:
        engine.executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "reason,patch,counts,verdict",
    [
        ("tests_failed (ftp_fail=2, ptp_fail=0)", "patch", {"ftp_passed": 1, "ftp_total": 3, "ptp_passed": 4, "ptp_total": 4}, "WRONG_FIX"),
        ("tests_failed (ftp_fail=0, ptp_fail=1)", "patch", {"ftp_passed": 3, "ftp_total": 3, "ptp_passed": 3, "ptp_total": 4}, "REGRESSION_ONLY"),
        ("tests_failed (ftp_fail=2, ptp_fail=0)", "", {"ftp_passed": 1, "ftp_total": 3}, "NO_PATCH"),
        ("get_eval_report_failed: Error PRIVATE_GOLD_TEST", "patch", {}, "OTHER"),
        ("tests_passed", "patch", {"ftp_passed": 3, "ftp_total": 3, "ptp_passed": 4, "ptp_total": 4}, "RESOLVED"),
    ],
)
def test_actual_token_transport_reaches_reconstructed_reflector(token_rollout, reason, patch, counts, verdict):
    row, native = token_rollout(reason, patch=patch, counts=counts)
    assert row["termination_reason"] == "ENV_DONE"
    feedback = row["terminal_feedback"]
    assert feedback["submitted"] is True
    assert feedback["patch_present"] is bool(patch)
    assert feedback["grading_error"] is reason.startswith("get_eval_report_failed")
    serialized = json.dumps(feedback)
    assert "PRIVATE_GOLD_TEST" not in serialized and "git_diff" not in serialized and "tests_status" not in serialized
    assert len(serialized) < 700
    adapter = MiniSweReflectorAdapter(handoff_protocol="feedback_v1")
    prompt = reconstruction_controller(adapter)._build_reflector_prompt_from_dict(row, "", 0)
    assert f"Verdict: {verdict}" in prompt
    assert "Submit Attempted: YES" in prompt
    assert "Patch Produced: " + ("yes" if patch else "no") in prompt
    assert prompt == adapter.format_trajectory_summary(native, row["trajectory_reward"])
    original_reply = native.steps[-1].next_observation
    history = adapter.format_conversation_history(native)
    assert "PRIVATE_GOLD_TEST" not in history
    assert native.steps[-1].next_observation == original_reply
    trainer = reconstruction_controller(adapter)
    trainer.reflector_prompt_template = "{{trajectory_summary}}\n{{conversation_history}}"
    assert "PRIVATE_GOLD_TEST" not in trainer._build_reflector_prompt_from_dict(row, "", 0)
    if counts:
        assert f"FAIL_TO_PASS: {counts['ftp_passed']} passed, {counts['ftp_total'] - counts['ftp_passed']} failed" in prompt


def test_legacy_token_payload_and_prompt_remain_without_feedback(token_rollout):
    row, _ = token_rollout("tests_failed (ftp_fail=1, ptp_fail=0)", preserve=False)
    assert "terminal_feedback" not in row
    prompt = reconstruction_controller(MiniSweReflectorAdapter())._build_reflector_prompt_from_dict(row, "", 0)
    assert "Verdict: NOT_RESOLVED" in prompt
    assert "Patch Produced: unknown" in prompt and "Test Results:" not in prompt


def test_feedback_has_fixed_labels_and_counts_missing_cases_as_failed():
    result = sanitize_terminal_feedback(
        {
            "reward_reason": "tests_failed (ftp_fail=0, ptp_fail=0) PRIVATE_GOLD_TEST",
            "test_counts": {"ftp_passed": 1, "ftp_total": 3, "ptp_passed": 4, "ptp_total": 4, "PRIVATE_GOLD_TEST": 1},
            "git_diff": "PRIVATE_PATCH" * 10000,
            "tests_status": {"PRIVATE_GOLD_TEST": "PRIVATE_CONTENT"},
        }
    )
    assert result["reward_reason"] == "tests_failed (ftp_fail=2, ptp_fail=0)"
    assert sanitize_terminal_feedback(result) == result
    assert "PRIVATE" not in json.dumps(result)
    assert len(json.dumps(result)) < 400


def test_campaign_prompt_retains_correct_tips_after_success():
    prompts = ROOT / "examples/mini_swe_bench/train/ace_rl/prompts"
    legacy = (prompts / "curator_prompt.txt").read_text()
    repaired = (prompts / "curator_prompt_handoff_v1.txt").read_text()
    assert "clearly mastered? If yes, consider REMOVE" in legacy
    assert "clearly mastered" not in repaired
    assert "Preserve tips that helped the successful attempt" in repaired
    assert "A success does not establish that a fresh Generator attempt will remember the fact" in repaired
    assert "exact duplicate retained elsewhere" in repaired
    reflector = (prompts / "reflector_prompt_handoff_v1.txt").read_text()
    assert "Preserve correct or helpful tips, including tips used in a successful attempt" in reflector
    assert "identify the retained tip that carries the same fact" in reflector
    assert "Never request removal merely because a tip was followed, validated, or appears mastered" in reflector
