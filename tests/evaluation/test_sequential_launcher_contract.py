"""The same command contract reaches all maintained benchmark evaluators."""

import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# One policy turn budget per benchmark, shared by ACE and vanilla evaluation.
MAX_TURNS = {"mbpp": 6, "mastermind": 5, "appworld": 10, "mini_swe_bench": 10}


def preview(benchmark, method, **overrides):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(ROOT), PYTHON_BIN=sys.executable, MODEL_NAME="Qwen/Qwen3-0.6B",
               MODEL_VARIANT="thinking", DISABLE_THINKING="False")
    env.update(overrides)
    return subprocess.run(["bash", str(ROOT / f"examples/{benchmark}/eval/{method}/eval.slurm"), "--dry-run"],
                          env=env, text=True, capture_output=True, check=True)


@pytest.mark.parametrize("benchmark", ["appworld", "mbpp", "mini_swe_bench", "mastermind"])
@pytest.mark.parametrize("method", ["ace", "vanilla"])
def test_default_presets_keep_all_eight_visits_or_parallel_samples(benchmark, method, monkeypatch):
    # ACE visits8: eight measured visits, one sample each; vanilla pass8: one visit, eight samples.
    visits, samples = (8, 1) if method == "ace" else (1, 8)
    result = preview(benchmark, method)
    if benchmark in {"appworld", "mbpp"}:
        item = json.loads(result.stdout)["evaluations"][0]
        command = item["evaluation"]
        assert item["preset"] == ("visits8" if method == "ace" else "pass8")
        assert item["local_eval_protocol"] == "sequential_adaptation_v1"
        assert not item["final_policy_eval"]
    else:
        command = shlex.split(result.stdout.splitlines()[-1])
    module = importlib.import_module(command[command.index("-m") + 1])
    monkeypatch.setattr(sys, "argv", [module.__file__, *command[command.index("-m") + 2:]])
    args = module.parse_args()
    assert args.resume is True
    assert args.g_act == samples
    assert args.max_turns == MAX_TURNS[benchmark]
    if benchmark != "mastermind" or method != "vanilla":
        assert args.local_eval_epochs == visits
        assert args.local_eval_protocol == "sequential_adaptation_v1"
        assert (args.g_ref, args.g_cur) == ((1, 1) if method == "ace" else (0, 0))
        assert not getattr(args, "freeze_solved", False)


@pytest.mark.parametrize("benchmark", ["appworld", "mbpp", "mini_swe_bench", "mastermind"])
def test_fixed_budget_ace_rejects_success_skipping(benchmark):
    with pytest.raises(subprocess.CalledProcessError):
        preview(benchmark, "ace", FREEZE_SOLVED="True")


@pytest.mark.parametrize("benchmark", ["appworld", "mbpp", "mini_swe_bench", "mastermind"])
def test_passk_requires_enough_capacity_for_a_complete_parallel_sample_group(benchmark):
    with pytest.raises(subprocess.CalledProcessError):
        preview(benchmark, "vanilla", G_ACT="8", N_PARALLEL_AGENTS="4", PARALLEL="4", N_SERVERS="4")
