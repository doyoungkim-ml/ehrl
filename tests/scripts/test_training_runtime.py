"""Shared runtime cleanup preserves diagnostics and leaves other jobs alone."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("status", [0, 17])
def test_cleanup_is_scoped_to_created_directory_and_keeps_failure_logs(tmp_path, status):
    repo = tmp_path / "source"
    (repo / "rllm").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts/gpu_venv.sh").write_text("""rllm_resolve_gpu_env() {
export RLLM_GPU_FAMILY=h200 VENV_DIR=/unused
export RLLM_CACHE_DIR="$1/cache"
}
""")
    (repo / "scripts/gpu_runtime.sh").write_text('rllm_configure_gpu_runtime() { RUNTIME_LD_LIBRARY_PATH=""; }\n')
    for name in ("verify_verl_compat.py", "patch_verl_zmq_ipc.py"):
        (repo / "scripts" / name).write_text("# Fake only the external compatibility preflight.\n")
    other_job = tmp_path / "other-job"
    other_job.mkdir()
    sentinel = other_job / "keep.txt"
    sentinel.write_text("other job logs")
    path_record = tmp_path / "created-path"
    script = """source "$RUNTIME_SCRIPT"
printf '%s' "$RLLM_JOB_TMP" > "$PATH_RECORD"
mkdir -p "$RAY_TMPDIR/session_latest/logs"
printf 'root cause' > "$RAY_TMPDIR/session_latest/logs/worker.err"
exit "$EXIT_STATUS"
"""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(
        RUNTIME_SCRIPT=str(ROOT / "scripts/training_runtime.sh"),
        RLLM_DIR=str(repo),
        BENCHMARK="mastermind",
        TRAINING_METHOD="grpo",
        PRESET="grpo",
        MODEL_NAME="Qwen/Qwen3-0.6B",
        MODEL_VARIANT="nonthinking",
        PYTHON_BIN=sys.executable,
        SLURM_JOB_ID="123",
        LOCAL_TMP_ROOT=str(tmp_path),
        PATH_RECORD=str(path_record),
        EXIT_STATUS=str(status),
        RLLM_GPU_KEEPALIVE="0",
    )
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
    assert result.returncode == status, result.stderr
    assert sentinel.read_text() == "other job logs"
    assert not Path(path_record.read_text()).exists()
    failure_log = repo / "examples/mastermind/train/grpo/runs/grpo/qwen3-0.6b-nonthinking/logs/ray-123/worker.err"
    assert failure_log.exists() == bool(status)
    if status:
        assert failure_log.read_text() == "root cause"


def test_missing_gpu_environment_is_built_once_under_a_lock_and_retried_after_failure(tmp_path):
    repo = tmp_path / "source"
    (repo / "scripts").mkdir(parents=True)
    calls = tmp_path / "setup-calls"
    # Stand in for the long real setup: record the call, then build or fail.
    (repo / "scripts/setup_qwen35.sh").write_text(f"""echo "recreate=$RECREATE_VENV" >> {calls}
[[ -z ${{FAIL_SETUP:-}} ]] || exit 3
mkdir -p "$VENV_DIR/bin" && ln -sf "$(command -v python3)" "$VENV_DIR/bin/python"
""")
    script = 'source "$GPU_VENV"; rllm_ensure_gpu_env "$RLLM_DIR"'
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    venv = tmp_path / "venv"
    env.update(GPU_VENV=str(ROOT / "scripts/gpu_venv.sh"), RLLM_DIR=str(repo), VENV_DIR=str(venv), RLLM_GPU_ENV_KEY="l40s", RLLM_GPU_FAMILY="l40s")

    failed = subprocess.run(["bash", "-c", script], env={**env, "FAIL_SETUP": "1"}, capture_output=True, text=True)
    assert failed.returncode != 0 and (tmp_path / "venv.setup-incomplete").exists()
    built = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    assert not (tmp_path / "venv.setup-incomplete").exists()
    ready = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert ready.returncode == 0 and "preparing" not in ready.stderr
    assert calls.read_text().split() == ["recreate=0", "recreate=1"]

    disabled = subprocess.run(["bash", "-c", script], env={**env, "VENV_DIR": str(tmp_path / "absent"), "RLLM_AUTO_SETUP": "0"}, capture_output=True, text=True)
    assert disabled.returncode != 0 and "RLLM_AUTO_SETUP" in disabled.stderr


def _stub_checkout(tmp_path):
    repo = tmp_path / "source"
    (repo / "rllm").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts/gpu_venv.sh").write_text('rllm_resolve_gpu_env() { export RLLM_GPU_FAMILY=h200 VENV_DIR=/unused RLLM_CACHE_DIR="$1/cache"; }\n')
    (repo / "scripts/gpu_runtime.sh").write_text('rllm_configure_gpu_runtime() { RUNTIME_LD_LIBRARY_PATH=""; }\n')
    for name in ("verify_verl_compat.py", "patch_verl_zmq_ipc.py"):
        (repo / "scripts" / name).write_text("# Fake only the external compatibility preflight.\n")
    launcher = repo / "examples/mbpp/train/grpo/train.slurm"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\n#SBATCH --job-name=mbpp-grpo\n")
    # Stand-ins for Slurm and the trainer: record calls, never submit or train.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "squeue").write_text(f"""#!/usr/bin/env bash
echo "squeue $*" >> {calls}
printf '%s\\n' 'campaign-train|gpu-part|lab-acct|normal|(null)|2-00:00:00|h200|gres/gpu:2|500G|campaign mbpp|grpo'
""")
    (bin_dir / "sbatch").write_text(f"""#!/usr/bin/env bash
echo "sbatch part=$TRAIN_CHAIN_PART/$TRAIN_CHAIN_PARTS resume=$TRAINER_RESUME_MODE from=${{TRAINER_RESUME_FROM_PATH:-}} slurm=${{SLURM_JOB_ID:-}} $*" >> {calls}
count=$(grep -c '^sbatch' {calls})
echo "$((1000 + count));cluster"
""")
    (bin_dir / "python-stub").write_text(f"""#!/usr/bin/env bash
if [[ "$1 $2" == "-m rllm.utils.run_state" ]]; then exit 0; fi
if [[ "$1 $2" == "-u -m" ]]; then echo "trained $3" >> {calls}; exit 0; fi
exec {sys.executable} "$@"
""")
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        RUNTIME_SCRIPT=str(ROOT / "scripts/training_runtime.sh"),
        RLLM_DIR=str(repo),
        BENCHMARK="mbpp",
        TRAINING_METHOD="grpo",
        PRESET="grpo",
        PROJECT_NAME="mbpp",
        MODEL_NAME="Qwen/Qwen3-0.6B",
        MODEL_VARIANT="nonthinking",
        PYTHON_BIN=str(bin_dir / "python-stub"),
        SLURM_JOB_ID="77",
        SLURM_JOB_NAME="campaign-train",
        LOCAL_TMP_ROOT=str(tmp_path),
        RLLM_GPU_KEEPALIVE="0",
        TRAINER_RESUME_MODE="disable",
        TRAINER_RESUME_FROM_PATH="/unused/global_step_5",
    )
    run_dir = repo / "examples/mbpp/train/grpo/runs/grpo/qwen3-0.6b-nonthinking"
    return env, calls, run_dir


CHAIN_SCRIPT = """RLLM_TRAIN_ARGS=("trainer.resume_mode=disable" "+rllm.note=two words")
source "$RUNTIME_SCRIPT"
"""


def test_first_part_submits_the_chain_with_parent_identity_and_ledger(tmp_path):
    env, calls, run_dir = _stub_checkout(tmp_path)
    result = subprocess.run(["bash", "-c", CHAIN_SCRIPT], env={**env, "TRAIN_CHAIN_PARTS": "3"}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    submissions = [line for line in calls.read_text().splitlines() if line.startswith("sbatch")]
    launcher = env["RLLM_DIR"] + "/examples/mbpp/train/grpo/train.slurm"
    common = (
        "--parsable --job-name=campaign-train --comment=campaign mbpp|grpo --partition=gpu-part --account=lab-acct --qos=normal "
        "--time=2-00:00:00 --mem=500G --constraint=h200 --gres=gpu:2"
    )
    assert submissions == [
        f"sbatch part=2/3 resume=auto from= slurm= {common} --dependency=afterany:77 {launcher} +rllm.note=two words",
        f"sbatch part=3/3 resume=auto from= slurm= {common} --dependency=afterany:1001 {launcher} +rllm.note=two words",
    ]
    ledger = json.loads((run_dir / "train_jobs.json").read_text())["jobs"]
    assert [(job["job_id"], job["part"], job["parts"], job["dependency"]) for job in ledger] == [
        ("77", 1, 3, None), ("1001", 2, 3, "afterany:77"), ("1002", 3, 3, "afterany:1001"),
    ]
    assert ledger[0]["started_at"] and ledger[1]["submitted_at"] and ledger[0]["job_name"] == "campaign-train"


def test_later_parts_never_resubmit_and_record_their_start(tmp_path):
    env, calls, run_dir = _stub_checkout(tmp_path)
    result = subprocess.run(["bash", "-c", CHAIN_SCRIPT], env={**env, "TRAIN_CHAIN_PARTS": "3", "TRAIN_CHAIN_PART": "2", "SLURM_JOB_ID": "1001"}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not [line for line in calls.read_text().splitlines() if line.startswith("sbatch")]
    ledger = json.loads((run_dir / "train_jobs.json").read_text())["jobs"]
    assert [(job["job_id"], job["part"]) for job in ledger] == [("1001", 2)]


def test_single_part_runs_submit_nothing(tmp_path):
    env, calls, _ = _stub_checkout(tmp_path)
    result = subprocess.run(["bash", "-c", CHAIN_SCRIPT], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not calls.exists() or "sbatch" not in calls.read_text()


@pytest.mark.parametrize("parts,part", [("0", "1"), ("2", "3"), ("x", "1")])
def test_invalid_chain_settings_fail_before_runtime(tmp_path, parts, part):
    env, _, run_dir = _stub_checkout(tmp_path)
    result = subprocess.run(["bash", "-c", CHAIN_SCRIPT], env={**env, "TRAIN_CHAIN_PARTS": parts, "TRAIN_CHAIN_PART": part}, text=True, capture_output=True)
    assert result.returncode == 2 and "TRAIN_CHAIN_PARTS" in result.stderr
    assert not run_dir.exists()


def test_dry_run_prints_chain_commands_without_submitting(tmp_path):
    env, calls, run_dir = _stub_checkout(tmp_path)
    env = {**env, "RLLM_DRY_RUN": "1", "TRAIN_CHAIN_PARTS": "2"}
    env.pop("SLURM_JOB_ID")
    result = subprocess.run(["bash", "-c", CHAIN_SCRIPT], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "[chain] part 2/2 would submit:" in result.stderr
    assert "--dependency=afterany:\\<part-1-job-id\\>" in result.stderr and "TRAINER_RESUME_MODE=auto" in result.stderr
    assert not calls.exists() and not run_dir.exists()


@pytest.mark.parametrize("saved,limit,trains", [("300", "301", False), ("301", "301", False), ("290", "301", True)])
def test_launch_skips_training_once_the_step_limit_was_reached(tmp_path, saved, limit, trains):
    env, calls, run_dir = _stub_checkout(tmp_path)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "latest_checkpointed_iteration.txt").write_text(saved)
    script = CHAIN_SCRIPT + 'rllm_training_launch examples.mbpp.train.grpo.train "trainer.total_training_steps=$LIMIT"\n'
    result = subprocess.run(["bash", "-c", script], env={**env, "TRAINER_RESUME_MODE": "auto", "LIMIT": limit}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert ("trained examples.mbpp.train.grpo.train" in calls.read_text()) == trains
    assert ("already complete" in result.stdout) != trains
