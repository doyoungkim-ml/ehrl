"""Shared launcher keepalive owns only its child and preserves cleanup semantics."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime_fixture(tmp_path):
    repo = tmp_path / "source"
    (repo / "rllm").mkdir(parents=True)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "gpu_venv.sh").write_text("""rllm_resolve_gpu_env() {
export RLLM_GPU_FAMILY=h200 VENV_DIR=/unused
export RLLM_CACHE_DIR="$1/cache"
}
""")
    (scripts / "gpu_runtime.sh").write_text('rllm_configure_gpu_runtime() { RUNTIME_LD_LIBRARY_PATH=""; }\n')
    for name in ("verify_verl_compat.py", "patch_verl_zmq_ipc.py"):
        (scripts / name).write_text("# External compatibility preflight is not under test.\n")
    (scripts / "gpu_keepalive.py").write_text("""import json, os, signal, time
from pathlib import Path
keys = ["CUDA_VISIBLE_DEVICES", "RLLM_GPU_KEEPALIVE_INTERVAL_SEC", "RLLM_GPU_KEEPALIVE_WORK_SEC", "RLLM_GPU_KEEPALIVE_SIZE", "RLLM_GPU_KEEPALIVE_MODE", "RLLM_GPU_KEEPALIVE_TARGET_UTIL"]
def stop(*_):
    Path(os.environ["STOP_RECORD"]).write_text("stopped")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path(os.environ["READY_RECORD"]).write_text(json.dumps({"pid": os.getpid(), **{k: os.environ.get(k) for k in keys}}))
while True:
    time.sleep(0.01)
""")
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(
        RUNTIME_SCRIPT=str(ROOT / "scripts/training_runtime.sh"),
        RLLM_DIR=str(repo),
        BENCHMARK="mini_swe_bench",
        TRAINING_METHOD="grpo",
        PRESET="grpo",
        MODEL_NAME="Qwen/Qwen3-0.6B",
        MODEL_VARIANT="nonthinking",
        PYTHON_BIN=sys.executable,
        SLURM_JOB_ID="123",
        LOCAL_TMP_ROOT=str(tmp_path),
        READY_RECORD=str(tmp_path / "helper-ready.json"),
        STOP_RECORD=str(tmp_path / "helper-stopped"),
        PATH_RECORD=str(tmp_path / "created-path"),
        CUDA_VISIBLE_DEVICES="3,5",
    )
    return repo, env


@pytest.mark.parametrize("ending,status,appworld", [("exit 0", 0, False), ("exit 17", 17, False), ("kill -TERM $$", 143, False), ("kill -INT $$", 130, False), ("exit 17", 17, True)])
def test_keepalive_is_owned_idempotent_and_preserves_exit_status(runtime_fixture, tmp_path, ending, status, appworld):
    repo, env = runtime_fixture
    chain = ""
    if appworld:
        sidecars = (ROOT / "examples/appworld/train/sidecars.sh").read_text()
        chain = sidecars[sidecars.index("APPWORLD_PIDS=()") : sidecars.index("# Probe a job-specific block")]
    script = (
        """source "$RUNTIME_SCRIPT"
for (( i=0; i<300; i++ )); do [[ -s "$READY_RECORD" ]] && break; sleep 0.01; done
[[ -s "$READY_RECORD" ]]
first_pid=$RLLM_GPU_KEEPALIVE_PID
rllm_start_gpu_keepalive
[[ "$first_pid" == "$RLLM_GPU_KEEPALIVE_PID" ]]
printf '%s' "$RLLM_JOB_TMP" > "$PATH_RECORD"
mkdir -p "$RAY_TMPDIR/session_latest/logs"
printf 'original failure' > "$RAY_TMPDIR/session_latest/logs/worker.err"
"""
        + chain
        + ending
        + "\n"
    )
    other_directory = tmp_path / "other-job"
    other_directory.mkdir()
    sentinel = other_directory / "keep.txt"
    sentinel.write_text("unrelated data")
    other_process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, timeout=20)
        assert result.returncode == status, result.stderr
        assert other_process.poll() is None
        assert sentinel.read_text() == "unrelated data"
        assert Path(env["STOP_RECORD"]).read_text() == "stopped"
        ready = json.loads(Path(env["READY_RECORD"]).read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(ready["pid"], 0)
        assert ready["CUDA_VISIBLE_DEVICES"] == "3,5"
        assert [ready[key] for key in ("RLLM_GPU_KEEPALIVE_INTERVAL_SEC", "RLLM_GPU_KEEPALIVE_WORK_SEC", "RLLM_GPU_KEEPALIVE_SIZE")] == ["5.0", "0.5", "4096"]
        assert [ready[key] for key in ("RLLM_GPU_KEEPALIVE_MODE", "RLLM_GPU_KEEPALIVE_TARGET_UTIL")] == ["fill", "85"]
        assert not Path(Path(env["PATH_RECORD"]).read_text()).exists()
        failure_log = repo / "examples/mini_swe_bench/train/grpo/runs/grpo/qwen3-0.6b-nonthinking/logs/ray-123/worker.err"
        assert failure_log.exists() == bool(status)
        if status:
            assert failure_log.read_text() == "original failure"
    finally:
        other_process.terminate()
        other_process.wait(timeout=5)


def test_appworld_cleanup_kills_a_server_that_ignores_sigterm(runtime_fixture, tmp_path):
    _, env = runtime_fixture
    env["APPWORLD_SERVER_STOP_TIMEOUT"] = "1"
    sidecars = (ROOT / "examples/appworld/train/sidecars.sh").read_text()
    chain = sidecars[sidecars.index("APPWORLD_PIDS=()") : sidecars.index("# Probe a job-specific block")]
    pids = tmp_path / "server-pids"
    script = (
        """source "$RUNTIME_SCRIPT"
for (( i=0; i<300; i++ )); do [[ -s "$READY_RECORD" ]] && break; sleep 0.01; done
"""
        + chain
        + f"""python3 -c 'import time; time.sleep(60)' &
APPWORLD_PIDS+=("$!")
python3 -c 'import signal, time; signal.signal(signal.SIGTERM, lambda *_: None); open("{tmp_path}/stubborn-ready", "w").close(); time.sleep(60)' &
APPWORLD_PIDS+=("$!")
for (( i=0; i<300; i++ )); do [[ -e "{tmp_path}/stubborn-ready" ]] && break; sleep 0.01; done
printf '%s ' "${{APPWORLD_PIDS[@]}}" > "{pids}"
exit 17
"""
    )
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 17, result.stderr
    for pid in map(int, pids.read_text().split()):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert Path(env["STOP_RECORD"]).read_text() == "stopped"


@pytest.mark.parametrize("setting", [{"RLLM_GPU_KEEPALIVE": "0"}, {"RLLM_DISABLE_GPU_KEEPALIVE": "1"}])
def test_explicit_disable_does_not_start_a_helper(runtime_fixture, setting):
    _, env = runtime_fixture
    env.update(setting)
    result = subprocess.run(["bash", "-c", 'source "$RUNTIME_SCRIPT"; rllm_start_gpu_keepalive; [[ -z "$RLLM_GPU_KEEPALIVE_PID" ]]'], env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not Path(env["READY_RECORD"]).exists()
    assert not Path(env["STOP_RECORD"]).exists()


def test_explicit_keepalive_timing_and_size_are_preserved(runtime_fixture):
    _, env = runtime_fixture
    expected = {"RLLM_GPU_KEEPALIVE_INTERVAL_SEC": "8", "RLLM_GPU_KEEPALIVE_WORK_SEC": "0.2", "RLLM_GPU_KEEPALIVE_SIZE": "512"}
    env.update(expected)
    script = """source "$RUNTIME_SCRIPT"
for (( i=0; i<300; i++ )); do [[ -s "$READY_RECORD" ]] && break; sleep 0.01; done
[[ -s "$READY_RECORD" ]]
"""
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    ready = json.loads(Path(env["READY_RECORD"]).read_text())
    assert {key: ready[key] for key in expected} == expected
    assert Path(env["STOP_RECORD"]).read_text() == "stopped"


def test_same_run_rejects_a_second_writer_before_runtime_setup(runtime_fixture):
    _, env = runtime_fixture
    env["RLLM_GPU_KEEPALIVE"] = "0"
    script = 'source "$RUNTIME_SCRIPT"; bash -c \'source "$RUNTIME_SCRIPT"\''
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 1
    assert "Another writer already owns RUN_DIR" in result.stdout + result.stderr


@pytest.mark.parametrize("checkpoint", ("fresh", "saved", "stale_marker"))
def test_same_launch_preflights_checkpoint_and_exports_one_run_home(runtime_fixture, tmp_path, checkpoint):
    repo, env = runtime_fixture
    run = tmp_path / "experiment"
    result_path = tmp_path / "trainer-started.json"
    (repo / "fake_training.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "keys = ['RUN_DIR', 'RLLM_RUN_DIR', 'WANDB_DIR', 'WANDB_ENTITY', 'TASK_ARTIFACT_ROOT', 'EPISODE_LOG_DIR']\n"
        "Path(os.environ['RESULT_PATH']).write_text(json.dumps({k: os.environ[k] for k in keys}))\n"
    )
    env.update(RUN_DIR=str(run), RLLM_GPU_KEEPALIVE="0", PROJECT_NAME="mini_swe_bench", PYTHONPATH=str(ROOT), RESULT_PATH=str(result_path), WANDB_ENTITY="caller-entity")
    if checkpoint != "fresh":
        root = run / "checkpoints"
        root.mkdir(parents=True)
        (root / "latest_checkpointed_iteration.txt").write_text("5")
        if checkpoint == "saved":
            saved = root / "global_step_5"
            (saved / "actor").mkdir(parents=True)
            (saved / "data.pt").write_bytes(b"fixture loader state")
            for kind in ("model", "optim", "extra_state"):
                (saved / "actor" / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"fixture shard")
    script = '''source "$RUNTIME_SCRIPT"
rllm_training_launch fake_training "trainer.default_local_dir=$CHECKPOINT_DIR" "trainer.resume_mode=auto"
'''
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, timeout=10)
    if checkpoint == "stale_marker":
        assert result.returncode != 0
        assert not result_path.exists()
        assert "Incomplete checkpoint" in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        resolved = json.loads(result_path.read_text())
        assert resolved == {
            "RUN_DIR": str(run), "RLLM_RUN_DIR": str(run), "WANDB_DIR": str(run),
            "WANDB_ENTITY": "caller-entity", "TASK_ARTIFACT_ROOT": str(run / "task_artifacts"),
            "EPISODE_LOG_DIR": str(run / "episodes"),
        }
        assert ("Resume checkpoint:" in result.stdout) == (checkpoint == "saved")
    assert (run / "logs/train-123.log").read_text() == result.stdout
