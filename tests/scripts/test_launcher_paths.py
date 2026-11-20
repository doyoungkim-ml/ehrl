"""Prepared benchmark paths resolve from a checkout without private config."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "checkout"
    repo.mkdir()
    for name in ("scripts", "examples", "rllm"):
        (repo / name).symlink_to(ROOT / name, target_is_directory=True)
    (repo / ".site.env").write_text(
        "export APPWORLD_DATA_DIR=/wrong/data\n"
        "export APPWORLD_SERVER_VENV=/wrong/venv\n"
        "export APPTAINER_SIF_DIR=/wrong/sif\n"
    )
    return repo


def launcher_env(repo: Path, launcher: Path, tmp_path: Path, names: tuple[str, ...], *, training: bool) -> list[str]:
    script = tmp_path / "capture.sh"
    script.write_text("#!/bin/bash\nprintf '%s\\n' " + " ".join(f'"${{{name}:-unset}}"' for name in names) + "\n")
    script.chmod(0o755)
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(repo), PYTHON_BIN=sys.executable, RLLM_DRY_RUN="1")
    if training or "mini_swe_bench" in str(launcher):
        command = ["bash", "-c", 'launcher=$1; capture=$2; set --; source "$launcher" >/dev/null; "$capture"', "launcher-path-test", str(launcher), str(script)]
    else:
        env["LAUNCHER_PYTHON"] = str(script)
        command = ["bash", str(launcher), "--dry-run"]
    result = subprocess.run(command, env=env, cwd=repo, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()


@pytest.mark.parametrize("training", (True, False))
def test_appworld_uses_prepared_sibling_data_and_checkout_server_venv(tmp_path, training):
    repo = checkout(tmp_path)
    (tmp_path / "appworld_data" / "data").mkdir(parents=True)
    launcher = ROOT / f"examples/appworld/{'train/ace_rl/train' if training else 'eval/ace/eval'}.slurm"
    values = launcher_env(repo, launcher, tmp_path, ("APPWORLD_DATA_DIR", "APPWORLD_SERVER_VENV"), training=training)
    assert values == [str(tmp_path / "appworld_data"), str(repo / ".venv_appworld_server")]


@pytest.mark.parametrize("training", (True, False))
def test_miniswe_uses_checkout_data_and_image_cache(tmp_path, training):
    repo = checkout(tmp_path)
    launcher = ROOT / f"examples/mini_swe_bench/{'train/ace_rl/train' if training else 'eval/ace/eval'}.slurm"
    values = launcher_env(repo, launcher, tmp_path, ("MINISW_DATA_DIR", "APPTAINER_SIF_DIR", "APPTAINER_CACHEDIR"), training=training)
    assert values == [
        str(repo / "rllm/data/datasets/swegym"),
        str(repo / ".cache/mini_swe_bench/sif"),
        str(repo / ".cache/mini_swe_bench/apptainer"),
    ]


@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_appworld_eval_finds_existing_split_export(tmp_path, method):
    repo = tmp_path / "checkout"
    (repo / "examples/appworld/eval" / method).mkdir(parents=True)
    (repo / "examples/appworld/eval/support/task-ids").mkdir(parents=True)
    (repo / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    for relative in (f"eval/{method}/presets.sh", "eval/common.sh", "server"):
        target = repo / "examples/appworld" / relative
        target.symlink_to(ROOT / "examples/appworld" / relative)
    support = repo / "examples/appworld/eval/support/task-ids"
    (support / "appworld_dev_task_ids.json").write_text('["dev-task"]')
    capture = tmp_path / "capture.sh"
    capture.write_text('#!/bin/bash\nprintf "%s\\n" "$APPWORLD_TASK_IDS_DIR"\n')
    capture.chmod(0o755)
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(repo), LAUNCHER_PYTHON=str(capture), TEST_SPLIT="dev", RLLM_AUTO_SETUP="0")
    launcher = ROOT / f"examples/appworld/eval/{method}/eval.slurm"
    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(support)

    standard = repo / "examples/appworld/data"
    standard.mkdir()
    (standard / "appworld_dev_task_ids.json").write_text('["dev-task"]')
    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(standard)


def test_eval_flashinfer_workspace_is_job_local_and_overridable(tmp_path):
    repo = checkout(tmp_path)
    capture = tmp_path / "capture.sh"
    capture.write_text('#!/bin/bash\nprintf "%s\\n" "$FLASHINFER_WORKSPACE_BASE"\n')
    capture.chmod(0o755)
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(repo), LAUNCHER_PYTHON=str(capture), SLURM_TMPDIR=str(tmp_path / "scratch"), SLURM_JOB_ID="123")
    launcher = ROOT / "examples/appworld/eval/ace/eval.slurm"
    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "scratch/rllm-flashinfer-123")
    env["FLASHINFER_WORKSPACE_BASE"] = str(tmp_path / "explicit")
    result = subprocess.run(["bash", str(launcher)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "explicit")
