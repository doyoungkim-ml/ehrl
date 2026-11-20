"""CPU contracts for the packaged ACE boundary and shared prompt-response roles.

Trainer initialization executes its real method body without importing GPU workers.
The agent, environments, Playbook, and entrypoint configuration are real classes.
"""

import ast
import importlib
import logging
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rllm.agents.single_turn_agent import SingleTurnAgent
from rllm.algorithms.ace import entrypoints
from rllm.algorithms.ace.checkpoint import _pack_rng, preserve_controller_rng
from rllm.algorithms.ace.agents import SingleTurnCuratorAgent, SingleTurnReflectorAgent
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.environments import SingleTurnCuratorEnv, SingleTurnReflectorEnv
from rllm.environments.base.prompt_response_env import SingleTurnEnv

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("agent_cls", [SingleTurnAgent, SingleTurnReflectorAgent, SingleTurnCuratorAgent])
def test_single_turn_roles_preserve_raw_qwen_history_and_step_snapshot(agent_cls):
    agent = agent_cls(system_prompt="role instructions", max_steps=99)
    agent.update_from_env({"task_description": "inspect this trajectory"}, 0, False, {})
    raw = '<think>\nkeep exact whitespace  \n</think>\n{"result": true}<|im_end|>'
    output = SimpleNamespace(content='{"result": true}', reasoning="keep exact whitespace  \n", text=raw)
    action = agent.update_from_model(raw, model_output=output)
    assert agent.max_steps == 1
    assert action.action == raw
    assert agent.chat_completions[-1] == {
        "role": "assistant",
        "content": output.content,
        "reasoning": output.reasoning,
        "raw_completion": raw,
    }
    step = agent.trajectory.steps[0]
    assert step.model_response == raw
    assert step.chat_completions[-1]["raw_completion"] == raw
    agent.update_from_env({}, 0.75, True, {"task_id": "sample"})
    assert step.reward == 0.75 and step.done
    assert len(step.chat_completions) == 3  # Later environment updates are detached.
    agent.reset()
    assert not agent.trajectory.steps
    assert step.chat_completions[-1]["raw_completion"] == raw


def test_single_turn_without_thinking_keeps_existing_content_policy():
    agent = SingleTurnAgent(accumulate_thinking=False)
    agent.update_from_env("prompt", 0, False, {})
    assert agent.update_from_model("<think>reason</think>\nanswer").action == "\nanswer"
    assert agent.chat_completions[-1] == {"role": "assistant", "content": "\nanswer"}


@pytest.mark.parametrize("env_cls", [SingleTurnReflectorEnv, SingleTurnCuratorEnv])
def test_ace_role_envs_preserve_metadata_and_factory_type(env_cls):
    env = env_cls.from_dict({"prompt_text": "prompt", "task_id": "task"})
    assert isinstance(env, env_cls)
    observation, info = env.reset()
    assert observation == {"task_id": "task", "task_description": "prompt", "num_tests": 0, "done": False}
    assert info == {"task_id": "task"}
    assert env.step("response") == ({}, 0.0, True, {"task_id": "task", "all_passed": False})
    assert env.done
    env.reset()
    assert not env.done


def test_generic_prompt_env_does_not_assume_coding_reward_fields():
    env = SingleTurnEnv(prompt_text="judge this solution", task_id="sample")
    assert "num_tests" not in env.reset()[0]
    assert env.step("response") == ({}, 0.0, True, {"task_id": "sample"})


def test_ace_binding_resolves_overrides_and_preserves_benchmark_defaults(monkeypatch):
    config = OmegaConf.create({"rllm": {"ace": {"g_ref": "${mul:2,3}", "prompts_dir": "", "reflector_adapter_uses_tokenizer": False}}})
    defaults = {"g_ref": 4, "prompts_dir": "/benchmark/prompts", "reflector_adapter_uses_tokenizer": True}
    captured = {}
    monkeypatch.setattr(entrypoints, "run_agent_task", lambda runner, **kwargs: captured.update(kwargs))
    entrypoints.run_local_ace_rl_task(
        object(),
        config=config,
        trainer_class_path="custom.Trainer",
        default_agent_class_path="custom.Agent",
        default_env_class_path="custom.Env",
        ace_defaults=defaults,
    )
    assert captured["trainer_kwargs"]["ace_config"] == {
        "controller_seed": None,
        "g_ref": 6,
        "prompts_dir": "/benchmark/prompts",
        "reflector_adapter_uses_tokenizer": False,
    }
    assert defaults["g_ref"] == 4
    assert defaults["reflector_adapter_uses_tokenizer"] is True
    assert OmegaConf.to_container(config.rllm.ace, resolve=True) == captured["trainer_kwargs"]["ace_config"]


def _seeded_binding(monkeypatch, seed, construct):
    config = OmegaConf.create({"rllm": {"ace": {"controller_seed": seed}}})
    monkeypatch.setattr(entrypoints, "run_agent_task", construct)
    entrypoints.run_local_ace_rl_task(object(), config=config, trainer_class_path="custom.Trainer", default_agent_class_path="custom.Agent", default_env_class_path="custom.Env")
    return config


def test_controller_seed_none_preserves_python_numpy_and_cpu_torch_rng(monkeypatch):
    with preserve_controller_rng():
        before = _pack_rng()
        captured = []
        config = _seeded_binding(monkeypatch, None, lambda *_args, **_kwargs: captured.append(_pack_rng()))
        assert captured == [before]
        assert _pack_rng() == before
        assert config.rllm.ace.controller_seed is None


@pytest.mark.parametrize("seed", [0, 42, 2**32 - 1])
def test_controller_seed_controls_first_driver_draws_before_trainer_construction(monkeypatch, seed):
    def cuda_seed_is_not_allowed(*_args, **_kwargs):
        pytest.fail("Controller initialization must not seed CUDA")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seed_is_not_allowed)
    expected = (random.Random(seed).random(), float(np.random.RandomState(seed).random_sample()), torch.rand(4, generator=torch.Generator().manual_seed(seed)).tolist())
    captured = []

    def construct(_runner, **kwargs):
        captured.append((random.random(), float(np.random.random()), torch.rand(4).tolist()))
        assert kwargs["config"].rllm.ace.controller_seed == seed
        assert kwargs["trainer_kwargs"]["ace_config"]["controller_seed"] == seed

    with preserve_controller_rng():
        _seeded_binding(monkeypatch, seed, construct)
        random.random(), np.random.random(), torch.rand(3)
        _seeded_binding(monkeypatch, seed, construct)
    assert captured == [expected, expected]


@pytest.mark.parametrize("seed", [True, False, -1, 1.5, "42", 2**32])
def test_invalid_controller_seed_fails_before_rng_or_trainer_mutation(monkeypatch, seed):
    def construct(*_args, **_kwargs):
        pytest.fail("Invalid seed reached trainer construction")

    with preserve_controller_rng():
        before = _pack_rng()
        with pytest.raises(ValueError, match="ace.controller_seed"):
            _seeded_binding(monkeypatch, seed, construct)
        assert _pack_rng() == before


def _benchmark_defaults(benchmark):
    path = ROOT / f"examples/{benchmark}/train/ace_rl/training_entrypoints.py"
    if not path.exists():
        path = path.with_name("train.py")
    tree = ast.parse(path.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ACE_DEFAULTS" for t in n.targets))
    namespace = {"_BENCHMARK_DIR": path.parent, "_HARNESS_DIR": path.parent, "_ASSETS": path.parent, "HARNESS_DIR": path.parent}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ACE_DEFAULTS"]


def _initialize(ace_config, tokenizer, *, preserve_terminal_feedback=False):
    path = ROOT / "rllm/algorithms/ace/trainer_base.py"
    tree = ast.parse(path.read_text())
    helpers = {"_coerce_bool", "_coerce_policy_train_source", "_coerce_reflection_source_selection", "_import_object"}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in helpers]
    trainer_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    trainer_cls.bases = []
    trainer_cls.body = [n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_parse_candidate_eval_rollouts", "_init_ace_components"}]
    selected.append(trainer_cls)
    namespace = {
        "os": os,
        "importlib": importlib,
        "Playbook": Playbook,
        "logger": logging.getLogger(__name__),
        "_VALID_POLICY_TRAIN_SOURCES": {"seed", "selected_candidate", "both"},
        "_VALID_REFLECTION_SOURCE_SELECTIONS": ("min_reward", "random"),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    trainer = namespace["ACERLAgentPPOTrainer"]()
    trainer.ace_config = ace_config
    trainer.tokenizer = tokenizer
    trainer.config = SimpleNamespace(
        data=SimpleNamespace(max_prompt_length=256, max_response_length=256),
        rllm=SimpleNamespace(agent={"preserve_terminal_feedback": preserve_terminal_feedback}),
    )
    trainer._init_ace_components()
    return trainer


@pytest.mark.parametrize("benchmark", ["mbpp", "appworld", "mastermind", "mini_swe_bench"])
def test_benchmark_assets_initialize_real_adapter_without_implicit_package_paths(benchmark):
    defaults = _benchmark_defaults(benchmark)
    tokenizer = object()
    trainer = _initialize(defaults, tokenizer)
    assert trainer.initial_playbook_text == Path(defaults["initial_playbook"]).read_text()
    assert trainer.base_system_prompt_template == (Path(defaults["prompts_dir"]) / "base_system_prompt.txt").read_text()
    assert (
        type(trainer.reflector_adapter).__name__
        == {
            "mbpp": "MBPPReflectorAdapter",
            "appworld": "AppWorldReflectorAdapter",
            "mastermind": "MastermindReflectorAdapter",
            "mini_swe_bench": "MiniSweReflectorAdapter",
        }[benchmark]
    )
    if benchmark == "appworld":
        assert trainer.reflector_adapter.tokenizer is tokenizer
    assert trainer.reflector_system_prompt == defaults["reflector_system_prompt"]
    assert trainer.curator_system_prompt == defaults["curator_system_prompt"]
    assert trainer.g_ref == 4 and trainer.g_cur == 4
    assert trainer.policy_train_source == "seed"


@pytest.mark.parametrize("missing", ["prompts_dir", "initial_playbook", "reflector_adapter_class", "reflector_system_prompt", "curator_system_prompt"])
def test_missing_benchmark_contract_fails_before_worker_generation(missing):
    defaults = _benchmark_defaults("mbpp")
    del defaults[missing]
    with pytest.raises(ValueError, match=f"ace.{missing}"):
        _initialize(defaults, object())


def test_tokenizer_injection_is_explicit_and_independent_of_adapter_module_name(monkeypatch):
    class ArbitraryAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    original_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(Adapter=ArbitraryAdapter) if name == "custom.adapter" else original_import(name))
    defaults = _benchmark_defaults("mbpp")
    defaults.update({"reflector_adapter_class": "custom.adapter.Adapter", "reflector_adapter_uses_tokenizer": True, "reflector_adapter_kwargs": {"custom": 7}, "reflector_max_history_tokens": 123})
    tokenizer = object()
    adapter = _initialize(defaults, tokenizer).reflector_adapter
    assert adapter.kwargs == {"max_turns": 5, "custom": 7, "tokenizer": tokenizer, "max_history_tokens": 123}
    defaults["reflector_adapter_uses_tokenizer"] = "false"
    assert _initialize(defaults, tokenizer).reflector_adapter.kwargs == {"max_turns": 5, "custom": 7}


def test_explicit_empty_role_prompt_and_benchmark_override_are_preserved():
    defaults = _benchmark_defaults("mbpp")
    defaults.update({"reflector_system_prompt": "", "curator_system_prompt": "Custom domain instructions"})
    trainer = _initialize(defaults, object())
    assert trainer.reflector_system_prompt == ""
    assert trainer.curator_system_prompt == "Custom domain instructions"
