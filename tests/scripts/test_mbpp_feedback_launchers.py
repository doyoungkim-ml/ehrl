"""MBPP_FEEDBACK selects hidden test feedback with its own training and evaluation identity."""

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rllm.utils.run_state import training_identity

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("ace_rl", "grpo", "ppo", "erl", "delayed_joint_ppo")
spec = importlib.util.spec_from_file_location("evaluation_runtime", ROOT / "scripts/evaluation_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def base_env(**env):
    clean_env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    clean_env.update(RLLM_DIR=str(ROOT), PYTHON_BIN=sys.executable, **env)
    return clean_env


def train(method, *overrides, **env):
    launcher = ROOT / "examples/mbpp/train" / method / "train.slurm"
    environment = base_env(RLLM_DRY_RUN="1", MODEL_VARIANT="nonthinking", **env)
    return subprocess.run(["bash", str(launcher), *overrides], cwd=ROOT, env=environment, text=True, capture_output=True)


def experiment_name(argv):
    return next(arg.split("=", 1)[1] for arg in argv if arg.startswith("trainer.experiment_name="))


def evaluate(method, **env):
    launcher = ROOT / f"examples/mbpp/eval/{method}/eval.slurm"
    return subprocess.run(["bash", str(launcher), "--dry-run"], cwd="/tmp", env=base_env(**env), capture_output=True, text=True)


def parse_evaluator(argv):
    module = argv[argv.index("-m") + 1]
    tree = ast.parse((ROOT / (module.replace(".", "/") + ".py")).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), module, "exec"), namespace)
    original = sys.argv
    try:
        sys.argv = [module, *argv[argv.index("-m") + 2 :]]
        return namespace["parse_args"]()
    finally:
        sys.argv = original


@pytest.mark.parametrize("method", METHODS)
def test_hidden_training_passes_feedback_and_tags_its_identity(method, tmp_path):
    full = train(method)
    assert full.returncode == 0, full.stderr
    full_argv = json.loads(full.stdout)
    assert not any("env_args.feedback" in arg for arg in full_argv)
    assert "hiddenfb" not in experiment_name(full_argv)

    run_dir = tmp_path / f"{method}_hiddenfb"
    hidden = train(method, MBPP_FEEDBACK="hidden", RUN_DIR=str(run_dir))
    assert hidden.returncode == 0, hidden.stderr
    argv = json.loads(hidden.stdout)
    assert argv.count("+rllm.env.env_args.feedback=hidden") == 1
    assert experiment_name(argv) == experiment_name(full_argv) + "_hiddenfb"
    assert f"trainer.default_local_dir={run_dir}/checkpoints" in argv
    # Apart from the feedback argument, names and run paths, the command is unchanged.
    def settings(command):
        varying = ("env_args.feedback", "experiment_name", "default_local_dir", "episode_log_dir", "artifact_root")
        return [arg for arg in command if not any(key in arg for key in varying)]

    assert settings(argv) == settings(full_argv)


def test_explicit_experiment_name_is_tagged_once(tmp_path):
    run_dir = str(tmp_path / "x_hiddenfb")
    for name, expected in (("custom", "custom_hiddenfb"), ("custom_hiddenfb", "custom_hiddenfb")):
        result = train("grpo", MBPP_FEEDBACK="hidden", RUN_DIR=run_dir, EXPERIMENT_NAME=name)
        assert result.returncode == 0, result.stderr
        assert experiment_name(json.loads(result.stdout)) == expected


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"MBPP_FEEDBACK": "hidden"}, "set its own RUN_DIR"),
        ({"MBPP_FEEDBACK": "hidden", "RUN_DIR": "/tmp/rllm-mbpp-old-run"}, "contains hiddenfb"),
        ({"MBPP_FEEDBACK": "partial"}, "must be full or hidden"),
        ({"MBPP_FEEDBACK": "Hidden", "RUN_DIR": "/tmp/x_hiddenfb"}, "must be full or hidden"),
    ],
)
def test_invalid_training_feedback_settings_fail_before_runtime(method, settings, message):
    result = train(method, **settings)
    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize("override", ("+rllm.env.env_args.feedback=hidden", "rllm.env.env_args.feedback=full"))
def test_feedback_cannot_bypass_the_launcher_identity_with_a_hydra_override(override):
    result = train("grpo", override)
    assert result.returncode != 0 and "MBPP_FEEDBACK" in result.stderr


def test_hidden_feedback_is_part_of_the_recorded_training_identity():
    full = {"rllm": {"env": {"env_args": {"max_turns": 6, "timeout": 10}}}}
    hidden = {"rllm": {"env": {"env_args": {"max_turns": 6, "timeout": 10, "feedback": "hidden"}}}}
    assert training_identity(hidden)["rllm"]["env"]["env_args"]["feedback"] == "hidden"
    assert training_identity(full) != training_identity(hidden)


@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_hidden_evaluation_uses_its_own_output_command_and_record(method, tmp_path):
    preset = "visits8" if method == "ace" else "pass8"
    root = tmp_path / "qwen3-0.6b-nonthinking"
    full = evaluate(method, RUN_DIR=str(root), STEP="20")
    assert full.returncode == 0, full.stderr
    full_item = json.loads(full.stdout)["evaluations"][0]
    assert "--feedback" not in full_item["evaluation"] and "mbpp_feedback" not in full_item
    assert parse_evaluator(full_item["evaluation"]).feedback == "full"
    assert full_item["run_dir"] == str(root / f"eval/{method}/{preset}/step_20")

    hidden = evaluate(method, RUN_DIR=str(root), STEP="20", MBPP_FEEDBACK="hidden")
    assert hidden.returncode == 0, hidden.stderr
    item = json.loads(hidden.stdout)["evaluations"][0]
    assert parse_evaluator(item["evaluation"]).feedback == "hidden"
    assert item["mbpp_feedback"] == "hidden"
    assert item["run_dir"] == str(root / f"eval/{method}/{preset}-hiddenfb/step_20")
    assert item["metrics_file"].startswith(item["run_dir"])
    # The campaign monitor resolves the same directory from a plan entry's eval_env.
    layout = runtime.evaluation_paths("mbpp", method, {"RUN_DIR": str(root), "STEP": "20", "PRESET": preset, "MBPP_FEEDBACK": "hidden"}, ROOT)
    assert layout["evaluations"][0]["run_dir"] == item["run_dir"]
    assert not root.exists()


def test_evaluation_requires_explicit_feedback_for_a_hidden_training_run(tmp_path):
    root = tmp_path / "qwen3-0.6b-nonthinking"
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"training_identity": {"rllm": {"env": {"env_args": {"feedback": "hidden"}}}}}))
    result = evaluate("vanilla", RUN_DIR=str(root), STEP="20")
    assert result.returncode != 0 and "set MBPP_FEEDBACK explicitly" in result.stderr
    for feedback, suffix in (("hidden", "-hiddenfb"), ("full", "")):
        result = evaluate("vanilla", RUN_DIR=str(root), STEP="20", MBPP_FEEDBACK=feedback)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["evaluations"][0]["run_dir"] == str(root / f"eval/vanilla/pass8{suffix}/step_20")


def test_invalid_or_misapplied_evaluation_feedback_fails():
    result = evaluate("ace", MODEL_NAME="Qwen/Qwen3-0.6B", MBPP_FEEDBACK="partial")
    assert result.returncode != 0 and "MBPP_FEEDBACK must be full or hidden" in result.stderr
    with pytest.raises(ValueError, match="applies only to MBPP"):
        runtime.evaluation_paths("mini_swe_bench", "ace", {"MODEL_NAME": "Qwen/Qwen3-4B", "MBPP_FEEDBACK": "hidden"}, ROOT)
