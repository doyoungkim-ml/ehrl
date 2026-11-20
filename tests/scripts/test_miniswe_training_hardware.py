"""Check the MiniSWEBench hardware gate before external training dependencies."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "family,gpus,expected_error",
    [
        ("h100", "2", None),
        ("h200", "2", None),
        ("l40s", "2", "requires H100 or H200 GPUs (detected: l40s)"),
        ("a100", "2", "requires H100 or H200 GPUs (detected: a100)"),
        ("hopper", "2", "requires H100 or H200 GPUs (detected: hopper)"),
        ("h100", "1", "requires two GPUs"),
        ("h200", "4", "requires two GPUs"),
    ],
)
def test_runtime_accepts_only_two_identified_hopper_gpus(tmp_path, family, gpus, expected_error):
    repo = tmp_path / "source"
    (repo / "scripts").mkdir(parents=True)
    # Keep the real benchmark startup and stub shared GPU/container setup only.
    (repo / "scripts/training_runtime.sh").write_text("set -euo pipefail\n")
    dependency_marker = tmp_path / "dependency-check"
    script = """source "$COMMON_SCRIPT"
apptainer() { :; }
check_grading_imports() { printf '%s' "$*" > "$DEPENDENCY_MARKER"; }
start_runtime
printf '%s' "$RLLM_GPU_FAMILY"
"""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(
        COMMON_SCRIPT=str(ROOT / "examples/mini_swe_bench/train/common.sh"),
        RLLM_DIR=str(repo),
        RLLM_GPU_FAMILY=family,
        N_GPUS_PER_NODE=gpus,
        PYTHON_BIN="check_grading_imports",
        DEPENDENCY_MARKER=str(dependency_marker),
        CACHE_DIR=str(tmp_path / "cache"),
        TMPDIR=str(tmp_path / "tmp"),
    )
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
    if expected_error is not None:
        assert result.returncode == 1
        assert expected_error in result.stderr
        assert not dependency_marker.exists()
        assert not (tmp_path / "cache").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == family
        assert "from swegym.harness.grading import get_eval_report" in dependency_marker.read_text()
        assert "from swebench.harness.grading import get_eval_report" in dependency_marker.read_text()
        assert (tmp_path / "cache/apptainer").is_dir()
        assert (tmp_path / "cache/sif").is_dir()
        assert (tmp_path / "tmp/apptainer").is_dir()
