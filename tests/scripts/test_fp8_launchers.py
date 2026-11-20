"""ROLLOUT_QUANTIZATION and EVAL_QUANTIZATION reach every maintained command."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("mbpp", "mastermind", "appworld", "mini_swe_bench")
METHODS = ("grpo", "ace_rl", "ppo", "erl", "delayed_joint_ppo")
FP8_TRAINING_ARGS = [
    "actor_rollout_ref.rollout.quantization=fp8",
    "actor_rollout_ref.rollout.calculate_log_probs=True",
    "algorithm.rollout_correction.rollout_is=token",
    "algorithm.rollout_correction.rollout_is_threshold=2.0",
]


def clean_env(**overrides):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(ROOT), RLLM_DRY_RUN="1", PYTHON_BIN=sys.executable)
    env.update(overrides)
    return env


def training_argv(benchmark, method, *overrides, check=True, **env):
    result = subprocess.run(["bash", str(ROOT / "examples" / benchmark / "train" / method / "train.slurm"), *overrides], cwd=ROOT, env=clean_env(**env), text=True, capture_output=True, check=check)
    return json.loads(result.stdout) if check else result


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("method", METHODS)
def test_fp8_rollout_knob_reaches_every_training_method(benchmark, method):
    default = training_argv(benchmark, method)
    assert training_argv(benchmark, method, ROLLOUT_QUANTIZATION="none") == default
    assert not any("quantization" in arg or "rollout_correction" in arg for arg in default)
    fp8 = training_argv(benchmark, method, ROLLOUT_QUANTIZATION="fp8")
    assert fp8 == default + FP8_TRAINING_ARGS


def test_fp8_rollout_knob_also_reaches_the_delayed_ace_preset():
    assert training_argv("mbpp", "ace_rl", PRESET="delayed_oneeval_rollout8", ROLLOUT_QUANTIZATION="fp8")[-4:] == FP8_TRAINING_ARGS


def test_unknown_rollout_quantization_fails_before_launch():
    result = training_argv("mbpp", "grpo", check=False, ROLLOUT_QUANTIZATION="int8")
    assert result.returncode != 0 and "ROLLOUT_QUANTIZATION must be none or fp8" in result.stderr


@pytest.mark.parametrize("benchmark,method", [("mbpp", "grpo"), ("appworld", "ace_rl"), ("mini_swe_bench", "grpo"), ("mastermind", "erl"), ("mastermind", "ppo")])
def test_fp8_overrides_compose_and_validate_against_pinned_verl(benchmark, method):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from rllm.trainer.verl.rollout_correction import rollout_correction_settings

    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE to validate the actual pinned Hydra schema")
    argv = training_argv(benchmark, method, ROLLOUT_QUANTIZATION="fp8")
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    search = f"hydra.searchpath=[file://{source}/verl/trainer/config]"
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        cfg = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [search])
        baseline = compose(config_name="agent_ppo_trainer", overrides=training_argv(benchmark, method)[argv.index("-m") + 2 :] + [search])
    assert cfg.actor_rollout_ref.rollout.quantization == "fp8"
    assert cfg.actor_rollout_ref.rollout.calculate_log_probs is True
    assert cfg.actor_rollout_ref.rollout.logprobs_mode == "processed_logprobs"
    assert cfg.algorithm.rollout_correction.rollout_is == "token"
    assert rollout_correction_settings(cfg) == {"rollout_is": "token", "rollout_is_threshold": 2.0, "quantization": "fp8"}
    assert rollout_correction_settings(baseline) is None
    assert baseline.actor_rollout_ref.rollout.quantization is None
    # run.json's training identity separates FP8 runs from their BF16 parents.
    from rllm.utils.run_state import training_identity

    fp8_identity = training_identity(OmegaConf.to_container(cfg, resolve=True))
    bf16_identity = training_identity(OmegaConf.to_container(baseline, resolve=True))
    assert {key for key in fp8_identity if fp8_identity[key] != bf16_identity[key]} == {"sampling", "algorithm"}
    assert fp8_identity["sampling"]["quantization"] == "fp8"
    assert fp8_identity["algorithm"]["rollout_correction"]["rollout_is"] == "token"


def evaluation(benchmark, method, tmp_path, **env):
    env = clean_env(RUNS_BASE=str(tmp_path / "runs"), **env)
    env.pop("PYTHON_BIN")
    result = subprocess.run(["bash", str(ROOT / "examples" / benchmark / "eval" / method / "eval.slurm"), "--dry-run"], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    return result


def served(benchmark, method, tmp_path, **env):
    """(server argv, conversion argv or None, output directory, evaluated model)."""
    result = evaluation(benchmark, method, tmp_path, **env)
    assert result.returncode == 0, result.stderr
    if benchmark in {"appworld", "mbpp"}:
        item = json.loads(result.stdout)["evaluations"][0]
        return item["server"], item.get("quantization_source"), item["run_dir"], item["model"]
    lines = [shlex.split(line) for line in result.stdout.splitlines() if line.strip()]
    server = next(line for line in lines if "vllm.entrypoints.openai.api_server" in line)
    conversion = next((line for line in lines if any(arg.endswith("convert_fp8_checkpoint.py") for arg in line)), None)
    evaluator = lines[-1]
    output = next(evaluator[i + 1] for i, arg in enumerate(evaluator) if arg in {"--playbook_snapshot_dir", "--run-dir"})
    return server, conversion, output, server[server.index("--model") + 1]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_fp8_evaluation_serves_a_converted_base_model_in_separate_outputs(tmp_path, benchmark, method):
    model = "Qwen/Qwen3-4B" if benchmark in {"appworld", "mini_swe_bench"} else "Qwen/Qwen3-0.6B"
    variant = {"MODEL_VARIANT": "thinking"} if benchmark == "appworld" else {}
    bf16_server, _, bf16_output, bf16_model = served(benchmark, method, tmp_path, MODEL_NAME=model, **variant)
    server, conversion, output, served_model = served(benchmark, method, tmp_path, MODEL_NAME=model, EVAL_QUANTIZATION="fp8", **variant)
    assert "--quantization" not in bf16_server and bf16_model == model
    assert server[server.index("--quantization") + 1] == "fp8"
    assert server[server.index("--dtype") + 1] == "bfloat16"
    assert served_model.endswith("-fp8") and served_model != bf16_model
    assert output != bf16_output and "-fp8" in output
    if isinstance(conversion, list):
        assert conversion[conversion.index("--hub-model") + 1] == model
        assert conversion[conversion.index("--target-dir") + 1] == served_model
    else:
        assert conversion == model


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_fp8_checkpoint_evaluation_converts_the_merged_step(tmp_path, benchmark):
    run = tmp_path / "train-run-thinking"
    run.mkdir()
    variant = {"MODEL_VARIANT": "thinking"} if benchmark == "appworld" else {}
    server, conversion, output, model = served(benchmark, "vanilla", tmp_path, RUN_DIR=str(run), STEP="7", EVAL_QUANTIZATION="fp8", **variant)
    merged = str(run / "checkpoints_merged" / "global_step_7")
    assert model == merged + "-fp8"
    assert server[server.index("--quantization") + 1] == "fp8"
    assert output == str(run / "eval" / "vanilla" / "pass8-fp8" / "step_7")
    if isinstance(conversion, list):
        assert conversion[conversion.index("--source-dir") + 1] == merged
    else:
        assert conversion == merged


def test_invalid_or_api_fp8_evaluation_fails_before_startup(tmp_path):
    assert evaluation("mbpp", "vanilla", tmp_path, EVAL_QUANTIZATION="int4").returncode != 0
    assert evaluation("mbpp", "vanilla", tmp_path, EVAL_QUANTIZATION="fp8", PRESET="api").returncode != 0
    assert evaluation("mini_swe_bench", "vanilla", tmp_path, EVAL_QUANTIZATION="int4").returncode != 0


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("kernel", [None, "replica"])
def test_fp8_conversion_kernel_is_explicit(tmp_path, benchmark, kernel):
    model = "Qwen/Qwen3-4B" if benchmark in {"appworld", "mini_swe_bench"} else "Qwen/Qwen3-0.6B"
    env = {"MODEL_NAME": model, "EVAL_QUANTIZATION": "fp8", "EVAL_FP8_MIN_CAPABILITY": "9.0", **({"MODEL_VARIANT": "thinking"} if benchmark == "appworld" else {})}
    if kernel:
        env["EVAL_FP8_KERNEL"] = kernel
    result = evaluation(benchmark, "vanilla", tmp_path, **env)
    assert result.returncode == 0, result.stderr
    expected = kernel or "verl"
    if benchmark in {"appworld", "mbpp"}:
        import importlib.util

        spec = importlib.util.spec_from_file_location("evaluation_runtime_fp8", ROOT / "scripts/evaluation_runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        item = json.loads(result.stdout)["evaluations"][0]
        assert item["quantization_kernel"] == expected
        command = runtime.fp8_conversion_command("python", ROOT, "hub", item, env)
    else:
        command = next(shlex.split(line) for line in result.stdout.splitlines() if "convert_fp8_checkpoint.py" in line)
    assert command[command.index("--kernel") + 1] == expected
    assert command[command.index("--min-capability") + 1] == "9.0"


@pytest.mark.parametrize("benchmark", ["mbpp", "mini_swe_bench"])
def test_unknown_fp8_kernel_fails_before_startup(tmp_path, benchmark):
    result = evaluation(benchmark, "vanilla", tmp_path, EVAL_QUANTIZATION="fp8", EVAL_FP8_KERNEL="pytorch")
    assert result.returncode != 0 and "EVAL_FP8_KERNEL must be verl or replica" in result.stderr
