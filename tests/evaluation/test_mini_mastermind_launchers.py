"""CPU coverage of evaluation command plans, parser parity, and server ownership."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


def launch(benchmark, mode, tmp_path, **overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("RLLM_", "MODEL_", "SLURM_"))}
    env.update(RLLM_DRY_RUN="1", RUNS_BASE=str(tmp_path / "runs"))
    env.update({key: str(value) for key, value in overrides.items()})
    return subprocess.run(
        ["bash", str(REPO / "examples" / benchmark / "eval" / mode / "eval.slurm")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("benchmark", "mode", "preset", "expected"),
    [
        # (max_turns, max_response_length, max_model_length, g_act, g_ref, g_cur, local_eval_epochs)
        ("mini_swe_bench", "ace", None, (10, 32768, 40000, 1, 1, 1, 8)),
        ("mini_swe_bench", "ace", "visits8", (10, 32768, 40000, 1, 1, 1, 8)),
        ("mini_swe_bench", "vanilla", None, (10, 32768, 40960, 8, 0, 0, 1)),
        ("mini_swe_bench", "vanilla", "pass8", (10, 32768, 40960, 8, 0, 0, 1)),
        ("mastermind", "ace", None, None),
        ("mastermind", "ace", "visits8", None),
        ("mastermind", "vanilla", None, None),
        ("mastermind", "vanilla", "pass8", None),
    ],
)
def test_dry_run_matches_real_parser_without_output_writes(tmp_path, benchmark, mode, preset, expected):
    result = launch(benchmark, mode, tmp_path, **({"PRESET": preset} if preset else {}))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    command = shlex.split(lines[-1])
    assert all("--hf-overrides" not in shlex.split(line) for line in lines)
    module = importlib.import_module(command[2])
    args = module.parse_args(command[3:])
    assert not (tmp_path / "runs").exists()
    default_preset = "visits8" if mode == "ace" else "pass8"
    if expected:
        assert (args.max_turns, args.max_response_length, args.max_model_length, args.g_act, args.g_ref, args.g_cur, args.local_eval_epochs) == expected
        assert args.task_concurrency == (8 if mode == "ace" else 1)
        assert len(lines) == 2  # local model server, then evaluator
        assert f"/runs/{default_preset}/" in args.metrics_file
        prompt = Path(args.base_system_prompt_filename)
        if not prompt.is_absolute():
            prompt = Path(args.prompts_dir) / prompt
        assert prompt.is_file()
    elif mode == "ace":
        assert (args.disable_thinking, args.gen_temperature, args.gen_top_p) == (False, 0.7, 1.0)
        assert Path(args.prompts_dir) == REPO / "examples/mastermind/train/ace_rl/prompts"
        assert args.ace_model is None
        assert (args.max_turns, args.local_eval_epochs, args.g_act, args.g_ref, args.g_cur, args.task_concurrency) == (5, 8, 1, 1, 1, 8)
        assert f"/runs/{default_preset}/" in args.metrics_file
    else:
        assert (args.model_variant, args.temperature, args.top_p) == ("nonthinking", 0.7, 1.0)
        assert (args.max_turns, args.g_act, args.parallel) == (5, 8, 8)
        assert f"/runs/{default_preset}/" in str(args.run_dir) + "/"


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind"])
@pytest.mark.parametrize("mode", ["ace", "vanilla"])
@pytest.mark.parametrize(("value", "expected"), [("False", False), ("0", False), ("TRUE", True), ("1", True)])
def test_serving_head_override_reaches_real_shell_command(tmp_path, benchmark, mode, value, expected):
    result = launch(benchmark, mode, tmp_path, EVAL_TIE_WORD_EMBEDDINGS=value)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    server = next(command for command in commands if "vllm.entrypoints.openai.api_server" in command)
    assert server[server.index("--hf-overrides") + 1] == json.dumps({"tie_word_embeddings": expected}, separators=(",", ":"))
    assert "--hf-overrides" not in commands[-1]
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind"])
@pytest.mark.parametrize("value", ["", "yes", "2", " false "])
def test_invalid_serving_head_override_fails_before_shell_runtime(tmp_path, benchmark, value):
    result = launch(benchmark, "ace", tmp_path, EVAL_TIE_WORD_EMBEDDINGS=value, RLLM_DRY_RUN="0", PYTHON_BIN="/missing/evaluation-python")
    assert result.returncode != 0 and "EVAL_TIE_WORD_EMBEDDINGS must be" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind"])
def test_api_model_rejects_shell_serving_head_override(tmp_path, benchmark):
    result = launch(benchmark, "ace", tmp_path, MODEL_NAME="gemini-2.5-pro", MODEL_SOURCE="api", EVAL_TIE_WORD_EMBEDDINGS="False")
    assert result.returncode != 0 and "EVAL_TIE_WORD_EMBEDDINGS requires" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind"])
@pytest.mark.parametrize("mode", ["ace", "vanilla"])
@pytest.mark.parametrize("preset", ["standard", "extended", "api", "gemini", "unknown", "other"])
def test_removed_and_unknown_presets_are_rejected(tmp_path, benchmark, mode, preset):
    preset = {"other": "pass8" if mode == "ace" else "visits8"}.get(preset, preset)
    result = launch(benchmark, mode, tmp_path, PRESET=preset)
    assert result.returncode != 0
    assert "supports only PRESET" in result.stderr
    assert not (tmp_path / "runs").exists()


def test_checkpoint_loop_and_ablation_flags_reach_parser(tmp_path):
    result = launch(
        "mini_swe_bench",
        "ace",
        tmp_path,
        CKPT_BASE="/models/checkpoints with spaces",
        STEPS="5 10",
        MAX_TASKS="3",
        TASK_OFFSET="2",
        REFLECTOR_ONLY="true",
        REFLECTOR_SIMPLIFIED_HISTORY="true",
        DISABLE_THINKING="True",
        ACCUMULATE_THINKING="True",
        TASK_LIST="/datasets/selected tasks.csv",
    )
    assert result.returncode == 0, result.stderr
    from examples.mini_swe_bench.eval.ace.evaluate import parse_args

    commands = [shlex.split(line) for line in result.stdout.splitlines() if "examples.mini_swe_bench.eval" in line]
    assert len(commands) == 2
    for command, step in zip(commands, [5, 10], strict=True):
        args = parse_args(command[3:])
        assert args.model_name == f"/models/checkpoints with spaces/global_step_{step}"
        assert args.reflector_only and args.reflector_simplified_history
        assert args.disable_thinking and args.accumulate_thinking
        assert (args.max_tasks, args.task_offset, args.task_list) == (3, 2, "/datasets/selected tasks.csv")
        assert f"metrics_step{step}.json" in args.metrics_file
        assert args.wandb_step == step


@pytest.mark.parametrize("mode", ["ace", "vanilla"])
def test_miniswe_task_concurrency_is_explicit_and_keeps_budget(tmp_path, mode):
    from examples.mini_swe_bench.eval.ace.evaluate import parse_args

    serial = launch("mini_swe_bench", mode, tmp_path, TASK_CONCURRENCY="1", N_PARALLEL_AGENTS="32")
    concurrent = launch("mini_swe_bench", mode, tmp_path, TASK_CONCURRENCY="4", N_PARALLEL_AGENTS="32")
    explicit_serial = launch("mini_swe_bench", mode, tmp_path, TASK_CONCURRENCY="1", N_PARALLEL_AGENTS="32")
    for result in (serial, concurrent, explicit_serial):
        assert result.returncode == 0, result.stderr
    serial_command = shlex.split(serial.stdout.splitlines()[-1])
    concurrent_command = shlex.split(concurrent.stdout.splitlines()[-1])
    explicit_command = shlex.split(explicit_serial.stdout.splitlines()[-1])
    assert "--task_concurrency" not in serial_command

    def without_server(command):
        # Dry runs pick a free port per launch; the recorded command masks it too.
        return [token for token in command if not token.startswith("http://127.0.0.1:")]

    assert without_server(serial_command) == without_server(explicit_command)
    serial_args = parse_args(serial_command[3:])
    args = parse_args(concurrent_command[3:])
    assert (serial_args.task_concurrency, args.task_concurrency) == (1, 4)
    assert args.n_parallel_agents == serial_args.n_parallel_agents == 32
    for field in ("g_act", "g_ref", "g_cur", "local_eval_epochs", "seed", "max_turns", "max_response_length", "max_model_length", "max_tasks", "ace_temperature"):
        assert getattr(args, field) == getattr(serial_args, field)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"TASK_CONCURRENCY": "0"},
        {"TASK_CONCURRENCY": "9"},
        {"TASK_CONCURRENCY": "4", "N_PARALLEL_AGENTS": "2"},
        {"TASK_CONCURRENCY": "4", "G_ACT": "3"},
    ],
)
def test_miniswe_invalid_task_concurrency_fails_before_startup(tmp_path, settings):
    result = launch("mini_swe_bench", "ace", tmp_path, **settings)
    assert result.returncode != 0
    assert "TASK_CONCURRENCY" in result.stderr
    assert not (tmp_path / "runs").exists()


def test_mastermind_task_concurrency_is_explicit_and_keeps_budget(tmp_path):
    from examples.mastermind.eval.ace.evaluate import parse_args

    serial = launch("mastermind", "ace", tmp_path, TASK_CONCURRENCY="1")
    concurrent = launch("mastermind", "ace", tmp_path, TASK_CONCURRENCY="4")
    explicit_serial = launch("mastermind", "ace", tmp_path, TASK_CONCURRENCY="1")
    for result in (serial, concurrent, explicit_serial):
        assert result.returncode == 0, result.stderr
    serial_command = shlex.split(serial.stdout.splitlines()[-1])
    concurrent_command = shlex.split(concurrent.stdout.splitlines()[-1])
    explicit_command = shlex.split(explicit_serial.stdout.splitlines()[-1])
    assert "--task_concurrency" not in serial_command

    def without_server(command):
        return [token for token in command if not token.startswith("http://127.0.0.1:")]

    assert without_server(serial_command) == without_server(explicit_command)
    serial_args = parse_args(serial_command[3:])
    args = parse_args(concurrent_command[3:])
    assert (serial_args.task_concurrency, args.task_concurrency) == (1, 4)
    assert args.n_parallel_agents == serial_args.n_parallel_agents == 8
    for field in ("g_act", "g_ref", "g_cur", "local_eval_epochs", "seed", "max_turns", "max_model_length", "ace_temperature"):
        assert getattr(args, field) == getattr(serial_args, field)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("settings", [{"TASK_CONCURRENCY": "0"}, {"TASK_CONCURRENCY": "9"}, {"TASK_CONCURRENCY": "4", "PARALLEL": "2"}, {"TASK_CONCURRENCY": "4", "G_ACT": "3"}])
def test_mastermind_invalid_task_concurrency_fails_before_startup(tmp_path, settings):
    result = launch("mastermind", "ace", tmp_path, **settings)
    assert result.returncode != 0
    assert "TASK_CONCURRENCY" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "module",
    ["examples.mbpp.eval.ace.mbpp_ace_utils", "examples.mastermind.eval.ace.mastermind_ace_utils"],
)
def test_small_benchmark_worker_args_isolate_engines_within_capacity(module):
    prepare_local_worker_args = importlib.import_module(module).prepare_local_worker_args

    args = SimpleNamespace(n_parallel_agents=8, g_act=2, max_turns=30, task_concurrency=4)
    workers = [prepare_local_worker_args(args, index, 4) for index in range(4)]
    assert [worker.n_parallel_agents for worker in workers] == [2, 2, 2, 2]
    assert all(worker is not args and worker.max_turns == 30 for worker in workers)
    assert args.n_parallel_agents == 8
    for worker_index, worker_count in ((0, 5), (4, 4), (-1, 4), (0, 0)):
        with pytest.raises(ValueError):
            prepare_local_worker_args(args, worker_index, worker_count)
    assert prepare_local_worker_args(SimpleNamespace(n_parallel_agents=1, g_act=1), 0, 1).n_parallel_agents == 1


def test_miniswe_worker_args_isolate_engines_within_capacity():
    from examples.mini_swe_bench.eval.ace.mini_swe_bench_ace_utils import prepare_local_worker_args

    args = SimpleNamespace(n_parallel_agents=8, g_act=2, max_turns=30, task_concurrency=4)
    workers = [prepare_local_worker_args(args, index, 4) for index in range(4)]
    assert [worker.n_parallel_agents for worker in workers] == [2, 2, 2, 2]
    assert all(worker is not args and worker.max_turns == 30 for worker in workers)
    assert args.n_parallel_agents == 8  # The original namespace keeps the shared capacity.
    for worker_index, worker_count in ((0, 5), (4, 4), (-1, 4), (0, 0)):
        with pytest.raises(ValueError):
            prepare_local_worker_args(args, worker_index, worker_count)
    single = prepare_local_worker_args(SimpleNamespace(n_parallel_agents=1, g_act=1), 0, 1)
    assert single.n_parallel_agents == 1


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind"])
def test_invalid_task_slice_fails_before_startup(tmp_path, benchmark):
    result = launch(benchmark, "ace", tmp_path, TASK_OFFSET="-1")
    assert result.returncode != 0
    assert "TASK_OFFSET" in result.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("filename", "digest"),
    [
        ("base_system_prompt.txt", "31fc6447104f378292473f971621240fd187b529e42621f86e1047f9eddec289"),
        ("reflector_prompt.txt", "fd26200d6c6d9a4ca385ab1e4fbd8a5a6ffa859f60c1574e21449fb06502068f"),
        ("curator_prompt.txt", "7a79cb06efaffa7918ab65fc3075d721e99826968fe230bcb5ad2fb8ebed54ba"),
    ],
)
def test_mastermind_training_and_evaluation_share_pinned_prompts(filename, digest):
    # Changing these changes ACE-RL training; update the digest deliberately.
    content = (REPO / "examples/mastermind/train/ace_rl/prompts" / filename).read_bytes()
    assert hashlib.sha256(content).hexdigest() == digest
    assert not (REPO / "examples/mastermind/eval/ace/prompts").exists()


def test_miniswe_evaluation_prompt_is_the_tool_contract_plus_the_training_prompt():
    from rllm.agents.system_prompts import SWEAGENT_SYSTEM_PROMPT

    prompts = REPO / "examples/mini_swe_bench/train/ace_rl/prompts"
    expected = SWEAGENT_SYSTEM_PROMPT.rstrip() + "\n\n" + (prompts / "base_system_prompt.txt").read_text()
    assert (prompts / "base_system_prompt_tools.txt").read_text() == expected
    from rllm.algorithms.ace.evaluation import evaluation_role_prompt_paths

    role_paths = evaluation_role_prompt_paths(str(prompts))
    names = sorted(["base_system_prompt.txt", "base_system_prompt_tools.txt", *(Path(path).name for path in role_paths)])
    assert names == ["base_system_prompt.txt", "base_system_prompt_tools.txt", "curator_prompt.txt", "reflector_prompt.txt"]


def test_miniswe_feedback_protocol_selects_only_repaired_role_assets():
    from rllm.algorithms.ace.evaluation import evaluation_role_prompt_paths

    prompts = REPO / "examples/mini_swe_bench/train/ace_rl/prompts"
    paths = evaluation_role_prompt_paths(str(prompts), "feedback_v1")
    assert [Path(path).name for path in paths] == ["reflector_prompt_handoff_v1.txt", "curator_prompt_handoff_v1.txt"]
    assert all(Path(path).is_file() for path in paths)


def test_miniswe_training_reconstruction_feeds_the_same_summary():
    from examples.mini_swe_bench.train.ace_rl.adapter import MiniSweReflectorAdapter
    from rllm.algorithms.ace.context.training_inputs import steps_from_training_chat

    view = "Look.\n<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/testbed/a.py</parameter>\n</function>"
    edit = "Fix.\n<function=str_replace_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/a.py</parameter>\n<parameter=old_str>x</parameter>\n</function>"
    chat = [{"role": "system", "content": "s"}, {"role": "user", "content": "issue"}]
    for action, reply in [
        (view, "1 x"),
        ("Again.\n" + view, "1 x"),
        (edit, "Error: old_str not found in file"),
        ("<function=submit></function>", "Submitting solution. Episode complete. Result: tests_failed"),
    ]:
        chat += [{"role": "assistant", "content": action}, {"role": "user", "content": reply}]
    steps = steps_from_training_chat(chat, [0, 0, 0, 0.2])
    assert steps[-1].next_observation.startswith("Submitting solution")
    trajectory = SimpleNamespace(steps=steps, task={"instance_id": "a__b-1", "problem_statement": "p"}, info={"all_passed": False, "termination_reason": "ENV_DONE"})
    values = MiniSweReflectorAdapter(max_turns=30).get_template_variables(trajectory, 0.2, "")
    lines = values["trajectory_summary"].splitlines()
    assert lines[:2] == ["Outcome: Failed", "Verdict: NOT_RESOLVED — the issue was not resolved; the grader's per-test report is not available for this attempt."]
    assert "Edit Outcomes: str_replace succeeded 0, failed 1" in lines
    assert any(line.startswith("Repeated Actions: 1 turn(s)") for line in lines)
    assert "Patch Produced: unknown" in lines
    assert values["outcome"] == "FAILED" and values["instance_id"] == "a__b-1"
    assert "old_str not found" in values["conversation_history"]


def test_vanilla_prompt_has_tools_without_playbook():
    text = (REPO / "examples/mini_swe_bench/eval/vanilla/prompts/base_system_prompt.txt").read_text()
    assert "playbook" not in text.lower()
    assert all(tool in text for tool in ("execute_bash", "str_replace_editor", "submit"))


def test_csv_filter_rejects_missing_column_and_empty_ids(tmp_path):
    from examples.mini_swe_bench.eval.ace.evaluate import _load_task_list

    csv = tmp_path / "tasks.csv"
    csv.write_text("instance_id\na\nb\na\n")
    assert _load_task_list(str(csv)) == {"a", "b"}
    csv.write_text("id\na\n")
    with pytest.raises(ValueError, match="instance_id"):
        _load_task_list(str(csv))
    csv.write_text("instance_id\n")
    with pytest.raises(ValueError, match="no instance_id"):
        _load_task_list(str(csv))


def test_mastermind_logging_preserves_summary_and_explicit_output(tmp_path, monkeypatch):
    from examples.mastermind.eval import evaluate

    async def fake_evaluate(args):
        return {"num_tasks": 2, "success_count": 1, "success_rate": 0.5, "avg_reward": 0.75, "trajectories": [{"task_id": "a"}, {"task_id": "b"}]}

    monkeypatch.setattr(evaluate, "evaluate", fake_evaluate)
    output = tmp_path / "copy.json"
    evaluate.main(
        ["--base-url", "http://example/v1", "--runs-base", str(tmp_path), "--run-tag", "trial", "--step", "5", "--limit", "20", "--offset", "98", "--save-trajectories", "--output", str(output)]
    )
    run_dir = tmp_path / "trial_step5_tasks98-99"
    assert json.loads(output.read_text()) == json.loads((run_dir / "metrics_step5.json").read_text())
    assert len(json.loads((run_dir / "trajectories.json").read_text())) == 2
    assert evaluate.task_slice(SimpleNamespace(offset=98, limit=20), 2) == "98-99"


@pytest.fixture
def fake_python(tmp_path):
    path = tmp_path / "python"
    path.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
args = sys.argv[1:]
if args and args[0].endswith("gpu_runtime.py"):
    raise SystemExit(0)
if args[:2] == ["-m", "vllm.entrypoints.openai.api_server"]:
    port = int(args[args.index("--port") + 1])
    model = args[args.index("--served-model-name") + 1]
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    Path(os.environ["FAKE_PIDS"]).write_text(json.dumps([os.getpid(), child.pid]))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = json.dumps({"data": [{"id": model}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
if args[:2] == ["-m", "examples.mastermind.eval.evaluate"]:
    raise SystemExit(7)
os.execv(sys.executable, [sys.executable, *args])
"""
    )
    path.chmod(0o755)
    return path


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def test_evaluation_failure_cleans_only_its_server_group(tmp_path, fake_python):
    # An unrelated process must survive the same failure that stops our server workers.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    pids_file = tmp_path / "pids.json"
    try:
        result = launch(
            "mastermind",
            "vanilla",
            tmp_path,
            RLLM_DRY_RUN="0",
            PYTHON_BIN=fake_python,
            VLLM_PORT=port,
            FAKE_PIDS=pids_file,
            VLLM_POLL_SECONDS="0.02",
            VLLM_STARTUP_ATTEMPTS="100",
        )
        assert result.returncode == 7, result.stderr
        pids = json.loads(pids_file.read_text())
        deadline = time.monotonic() + 3
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not any(alive(pid) for pid in pids)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_existing_output_is_preserved_without_starting_server(tmp_path, fake_python):
    run_dir = tmp_path / "runs/pass8/Qwen3-0.6B"
    run_dir.mkdir(parents=True)
    marker = run_dir / "metrics_step0.json"
    marker.write_text("existing metrics")
    pids_file = tmp_path / "pids.json"
    result = launch("mastermind", "vanilla", tmp_path, RLLM_DRY_RUN="0", PYTHON_BIN=fake_python, FAKE_PIDS=pids_file)
    assert result.returncode != 0
    assert "not empty" in result.stderr
    assert marker.read_text() == "existing metrics"
    assert not pids_file.exists()


def resume_guard(tmp_path, script, *, resume="0"):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(REPO), RUN_DIR=str(tmp_path / "run"), RESUME=resume)
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", 'source "$RLLM_DIR/scripts/evaluation_server.sh"; ' + script],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("directory_exists", [False, True])
def test_resume_zero_empty_directory_accepts_its_lock(tmp_path, directory_exists):
    run_dir = tmp_path / "run"
    if directory_exists:
        run_dir.mkdir()
    result = resume_guard(tmp_path, 'eval_check_resume_dir; mkdir -p "$RUN_DIR"; eval_lock_run; eval_check_resume_dir; eval_unlock_run')
    assert result.returncode == 0, result.stderr
    assert [path.name for path in run_dir.iterdir()] == [".evaluation.lock"]


@pytest.mark.parametrize("filename", ["metrics_step0.json", "evaluation_config.json", ".ace_events", "..outputs"])
def test_resume_zero_preserves_real_existing_output(tmp_path, filename):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / filename
    marker.write_text("existing output")
    result = resume_guard(tmp_path, "eval_check_resume_dir")
    assert result.returncode == 1 and "Run directory is not empty" in result.stderr
    assert marker.read_text() == "existing output"
    assert not (run_dir / ".evaluation.lock").exists()


def test_resume_zero_rechecks_real_output_after_lock(tmp_path):
    result = resume_guard(
        tmp_path,
        'eval_check_resume_dir; mkdir -p "$RUN_DIR"; eval_lock_run; '
        'printf "new output" > "$RUN_DIR/metrics_step0.json"; '
        'if eval_check_resume_dir; then exit 9; fi; eval_unlock_run',
    )
    assert result.returncode == 0, result.stderr
    assert "Run directory is not empty" in result.stderr
    assert (tmp_path / "run/metrics_step0.json").read_text() == "new output"


def test_resume_zero_concurrent_lock_is_still_rejected(tmp_path):
    result = resume_guard(
        tmp_path,
        'eval_check_resume_dir; mkdir -p "$RUN_DIR"; eval_lock_run; eval_check_resume_dir; '
        'if bash -euo pipefail -c \'source "$RLLM_DIR/scripts/evaluation_server.sh"; eval_check_resume_dir; eval_lock_run\'; '
        'then exit 9; fi; eval_unlock_run',
    )
    assert result.returncode == 0, result.stderr
    assert "Evaluation is already running" in result.stderr


@pytest.mark.parametrize("recorded", [False, True])
def test_resume_one_keeps_valid_existing_outputs(tmp_path, recorded):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    contents = {"vllm.log": "startup log", "wandb_run_id.txt": "run id"}
    if recorded:
        contents.update({"evaluation_config.json": "{}", "metrics_step0.json": "existing metrics"})
    for name, body in contents.items():
        (run_dir / name).write_text(body)
    result = resume_guard(tmp_path, "eval_check_resume_dir; eval_lock_run; eval_check_resume_dir; eval_unlock_run", resume="1")
    assert result.returncode == 0, result.stderr
    assert all((run_dir / name).read_text() == body for name, body in contents.items())


def test_resume_one_keeps_unconfigured_output_rejection(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "metrics_step0.json"
    marker.write_text("existing metrics")
    result = resume_guard(tmp_path, "eval_check_resume_dir", resume="1")
    assert result.returncode == 1 and "has no evaluation configuration" in result.stderr
    assert marker.read_text() == "existing metrics"


def test_port_conflict_does_not_attach_to_another_server(tmp_path, fake_python):
    pids_file = tmp_path / "pids.json"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        result = launch(
            "mastermind",
            "vanilla",
            tmp_path,
            RLLM_DRY_RUN="0",
            PYTHON_BIN=fake_python,
            VLLM_PORT=sock.getsockname()[1],
            FAKE_PIDS=pids_file,
        )
    assert result.returncode != 0
    assert "Address already in use" in result.stderr
    assert not pids_file.exists()


def test_miniswe_reflector_summary_keeps_its_active_contract():
    from examples.mini_swe_bench.train.ace_rl.adapter import MiniSweReflectorAdapter

    view = "<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/testbed/a.py</parameter>\n</function>"
    steps = [SimpleNamespace(action=view, info={}, next_observation="1 import os") for _ in range(3)]
    tests = {"FAIL_TO_PASS": {"success": [], "failure": ["t1"]}, "PASS_TO_PASS": {"success": ["t2"], "failure": []}}
    steps.append(SimpleNamespace(action="<function=submit></function>", info={"max_steps": 20, "reward_reason": "tests_failed (ftp_fail=1)", "git_diff": "", "tests_status": tests}))
    summary = MiniSweReflectorAdapter(max_turns=30).format_trajectory_summary(SimpleNamespace(steps=steps, info={}), 0.0)
    lines = summary.splitlines()
    assert lines[:4] == ["Outcome: Failed", "Verdict: NO_PATCH — no patch was produced; the repository was left unchanged.", "Submit Attempted: YES", "Turns Used: 4/20"]
    assert "First Successful Edit: never" in lines
    assert any(line.startswith("Repeated Actions: 2 turn(s)") and "starts at turn 1" in line for line in lines)
    assert "Patch Produced: no" in lines
    assert any(line.startswith("  FAIL_TO_PASS: 0 passed, 1 failed") for line in lines)


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind", "mbpp", "appworld"])
@pytest.mark.parametrize("mode", ["ace", "vanilla"])
def test_evaluation_launchers_do_not_export_shell_functions(benchmark, mode):
    # Exported functions reach container shells, which cannot import the heredoc in eval_start_server.
    text = (REPO / "examples" / benchmark / "eval" / mode / "eval.slurm").read_text()
    assert "set +a\n# set -a also exported the functions above; child shells (e.g. MiniSWE containers) must not import them.\nexport -fn $(compgen -A function)\n" in text


def test_trajectory_aggregation_rejects_mixed_checkpoints_before_wandb(tmp_path, monkeypatch):
    from examples.mini_swe_bench.eval.helpers import aggregate_trajectory_shapes as aggregate

    runs = tmp_path / "runs"
    runs.mkdir()
    for step in [0, 30]:
        (runs / f"fixture_step{step}_32768_30_dtFalse_atFalse_tasks100").mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["aggregate", "--model-tag", "fixture", "--runs-dir", str(runs), "--out-dir", str(output), "--wandb-entity", "fixture"])
    assert aggregate.main() == 2
    assert not output.exists()
    command = aggregate.build_resubmit_cmd(32768, 30, "10-19", 5, at=True, dt=True)
    assert "ACCUMULATE_THINKING=True" in command and "DISABLE_THINKING=True" in command
    assert "TASK_OFFSET=10" in command and "MAX_TASKS=10" in command


@pytest.mark.parametrize(("problem", "expected"), [(None, 0), ("missing", 1), ("error", 1), ("disagree", 1)])
def test_regrading_gate_fails_on_missing_or_invalid_coverage(tmp_path, monkeypatch, problem, expected):
    from examples.mini_swe_bench.eval.helpers import regrade_ab_swegym as regrade

    rows = [{"instance_id": str(index), "cls": "resolved", "agree": True, "stock": {"elapsed": 1.0}, "transformed": {"elapsed": 1.0}, "stock_matches_original": True} for index in range(2)]
    if problem == "missing":
        rows.pop()
    elif problem == "error":
        rows[-1]["stock"] = {"error": "fixture failure"}
        rows[-1]["agree"] = False
    elif problem == "disagree":
        rows[-1]["agree"] = False
    monkeypatch.setattr(regrade, "collect_samples", lambda n, experiment_dirs: rows)
    monkeypatch.setattr(regrade, "run_sample", lambda sample: sample)
    monkeypatch.setattr(sys, "argv", ["regrade", "--n", "2", "--workers", "1", "--experiments", str(tmp_path / "input"), "--out", str(tmp_path / "output")])
    assert regrade.main() == expected


def test_wandb_ids_are_distinct_per_checkpoint_and_stable_on_resume(tmp_path):
    first, second = tmp_path / "step5", tmp_path / "step10"
    first.mkdir()
    second.mkdir()
    script = """
set -euo pipefail
source "$RLLM_DIR/scripts/evaluation_server.sh"
RUN_DIR="$FIRST_RUN"
eval_prepare_tracking
first_id=$(<"$RUN_DIR/wandb_run_id.txt")
RUN_DIR="$SECOND_RUN"
eval_prepare_tracking
RUN_DIR="$FIRST_RUN"
eval_prepare_tracking
"""
    env = {**os.environ, "RLLM_DIR": str(REPO), "PYTHON_BIN": sys.executable, "RLLM_DRY_RUN": "0", "WANDB_PROJECT": "fixture", "FIRST_RUN": str(first), "SECOND_RUN": str(second)}
    env.pop("WANDB_RUN_ID", None)
    env.pop("REQUESTED_WANDB_RUN_ID", None)
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert (first / "wandb_run_id.txt").read_text() != (second / "wandb_run_id.txt").read_text()


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind"])
@pytest.mark.parametrize("method", ["ace", "vanilla"])
def test_training_run_has_exact_checkpoint_and_output(tmp_path, benchmark, method):
    root = tmp_path / "training run"
    result = launch(benchmark, method, tmp_path, RUN_DIR=root, STEP=215)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    merge, server, evaluation = commands
    assert merge[merge.index("--source-dir") + 1] == str(root / "checkpoints/global_step_215")
    assert merge[merge.index("--target-dir") + 1] == str(root / "checkpoints_merged/global_step_215")
    assert server[server.index("--model") + 1] == str(root / "checkpoints_merged/global_step_215")
    output_flag = "--run-dir" if benchmark == "mastermind" and method == "vanilla" else "--metrics_file"
    expected_output = root / f"eval/{method}/{'visits8' if method == 'ace' else 'pass8'}/step_215"
    if output_flag == "--metrics_file":
        expected_output /= "metrics_step215.json"
    assert evaluation[evaluation.index(output_flag) + 1] == str(expected_output)
    module = importlib.import_module(evaluation[2])
    module.parse_args(evaluation[3:])
    assert not root.exists()


@pytest.mark.parametrize("method", ["ace", "vanilla"])
def test_miniswe_api_model_uses_one_entrypoint(tmp_path, method):
    result = launch("mini_swe_bench", method, tmp_path, MODEL_NAME="gemini-2.5-pro")
    assert result.returncode == 0, result.stderr
    commands = result.stdout.splitlines()
    assert len(commands) == 1 and "vllm" not in commands[0]
    preset = "visits8" if method == "ace" else "pass8"
    assert f"/{preset}/gemini-2.5-pro/metrics_step0.json" in commands[0]


def test_tracking_helper_keeps_caller_entity_and_evaluation_output(tmp_path):
    script = 'source "$RLLM_DIR/scripts/evaluation_server.sh"; eval_prepare_tracking; printf "%s\\n" "$WANDB_ENTITY" "$WANDB_DIR"'
    env = {**os.environ, "RLLM_DIR": str(REPO), "RUN_DIR": str(tmp_path), "WANDB_PROJECT": "fixture", "WANDB_ENTITY": "personal-account", "PYTHON_BIN": sys.executable, "RLLM_DRY_RUN": "0"}
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ["personal-account", str(tmp_path)]


def test_cleanup_removes_only_owned_temporary_directories(tmp_path):
    owned, external = tmp_path / "owned", tmp_path / "external"
    owned.mkdir()
    external.mkdir()
    (owned / "scratch").write_text("temporary")
    (external / "evidence").write_text("preserve")
    script = 'source "$RLLM_DIR/scripts/evaluation_server.sh"; EVAL_OWNED_TEMP_DIRS=("$OWNED"); APPTAINER_TMPDIR="$EXTERNAL"; eval_cleanup'
    env = {**os.environ, "RLLM_DIR": str(REPO), "OWNED": str(owned), "EXTERNAL": str(external)}
    subprocess.run(["bash", "-euo", "pipefail", "-c", script], env=env, capture_output=True, text=True, check=True)
    assert not owned.exists()
    assert (external / "evidence").read_text() == "preserve"


@pytest.fixture
def shell_keepalive_env(tmp_path):
    scripts = tmp_path / "source/scripts"
    scripts.mkdir(parents=True)
    (scripts / "gpu_keepalive.py").write_text('''import json, os, signal, time
from pathlib import Path
def stop(*_):
    Path(os.environ["STOP_RECORD"]).write_text("stopped")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path(os.environ["READY_RECORD"]).write_text(json.dumps({"pid": os.getpid(), "devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "size": os.environ.get("RLLM_GPU_KEEPALIVE_SIZE")}))
print("helper started", flush=True)
while True:
    time.sleep(0.01)
''')
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(
        EVAL_SERVER_SCRIPT=str(REPO / "scripts/evaluation_server.sh"), RLLM_DIR=str(scripts.parent),
        PYTHON_BIN=sys.executable, RUN_DIR=str(tmp_path / "outputs/step_10"), MODEL_PATH="unused", MODEL_SOURCE="hub",
        SLURM_JOB_ID="123", READY_RECORD=str(tmp_path / "ready.json"), STOP_RECORD=str(tmp_path / "stopped"),
        CUDA_VISIBLE_DEVICES="3,5", RLLM_GPU_KEEPALIVE_SIZE="512",
    )
    return env


@pytest.mark.parametrize("ending,status", [("exit 0", 0), ("exit 17", 17), ("kill -TERM $$", 143), ("kill -INT $$", 130)])
def test_shell_keepalive_starts_before_merge_is_owned_and_preserves_exit_status(shell_keepalive_env, ending, status):
    env = shell_keepalive_env
    script = '''set -euo pipefail
source "$EVAL_SERVER_SCRIPT"
eval_merge_bf16_checkpoint() {
    for (( i=0; i<300; i++ )); do [[ -s "$READY_RECORD" ]] && break; sleep 0.01; done
    [[ -s "$READY_RECORD" ]]
    kill -0 "$EVAL_GPU_KEEPALIVE_PID"
}
eval_merge_checkpoint
first_pid=$EVAL_GPU_KEEPALIVE_PID
eval_merge_checkpoint
[[ "$first_pid" == "$EVAL_GPU_KEEPALIVE_PID" ]]
''' + ending + "\n"
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # A stale inherited PID is never accepted as ownership of another job.
        env["EVAL_GPU_KEEPALIVE_PID"] = str(other.pid)
        result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == status, result.stderr
        assert other.poll() is None
        ready = json.loads(Path(env["READY_RECORD"]).read_text())
        assert ready["devices"] == "3,5" and ready["size"] == "512"
        assert Path(env["STOP_RECORD"]).read_text() == "stopped"
        with pytest.raises(ProcessLookupError):
            os.kill(ready["pid"], 0)
        log = Path(env["RUN_DIR"]).parent / "gpu-keepalive-123.log"
        assert log.read_text().strip() == "helper started"
        assert not Path(env["RUN_DIR"]).exists()
    finally:
        other.terminate()
        other.wait(timeout=5)


@pytest.mark.parametrize("overrides", [
    {"RLLM_GPU_KEEPALIVE": "0"}, {"RLLM_DISABLE_GPU_KEEPALIVE": "1"},
    {"RLLM_DRY_RUN": "1"}, {"MODEL_SOURCE": "api"}, {"SLURM_JOB_ID": ""},
])
def test_shell_keepalive_skips_disabled_api_dry_run_and_nonallocation(shell_keepalive_env, overrides):
    env = {**shell_keepalive_env, **overrides}
    script = '''set -euo pipefail
source "$EVAL_SERVER_SCRIPT"
eval_merge_bf16_checkpoint() { :; }
eval_merge_checkpoint
[[ -z "$EVAL_GPU_KEEPALIVE_PID" ]]
eval_cleanup
'''
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert not Path(env["READY_RECORD"]).exists()
    assert not Path(env["RUN_DIR"]).parent.exists()


def test_resume_config_allows_new_port_but_rejects_changed_science(tmp_path):
    script = """
set -euo pipefail
source "$RLLM_DIR/scripts/evaluation_server.sh"
COMMAND=(python -m evaluator --base_url "$BASE_URL" --max_response_length "$BUDGET")
[[ ${RESUME:-0} != 1 ]] || COMMAND+=(--resume)
eval_record_config
"""
    env = {
        **os.environ,
        "RLLM_DIR": str(REPO),
        "RUN_DIR": str(tmp_path),
        "BENCHMARK": "mini_swe_bench",
        "EVAL_MODE": "ace",
        "PRESET": "visits8",
        "STEP": "5",
        "MODEL_NAME": "Qwen/example",
        "MODEL_PATH": "Qwen/example",
        "PYTHON_BIN": sys.executable,
        "BASE_URL": "http://127.0.0.1:10000/v1",
        "BUDGET": "32768",
    }
    subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, check=True)
    before = (tmp_path / "evaluation_config.json").read_bytes()
    env.update(BASE_URL="http://127.0.0.1:10001/v1", RESUME="1")
    subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True, check=True)
    env["BUDGET"] = "16384"
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "configuration differs" in result.stderr
    assert (tmp_path / "evaluation_config.json").read_bytes() == before


@pytest.mark.parametrize("value", [None, "False", "True"])
@pytest.mark.parametrize("completed", [False, True])
def test_shell_config_binds_serving_head_override_for_partial_and_completed_output(tmp_path, value, completed):
    script = '''
set -euo pipefail
source "$RLLM_DIR/scripts/evaluation_server.sh"
COMMAND=(python -m evaluator --base_url http://127.0.0.1:10000/v1)
eval_record_config
'''
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(REPO), RUN_DIR=str(tmp_path), BENCHMARK="mastermind", EVAL_MODE="ace", PRESET="visits8", STEP="300", MODEL_NAME="Qwen/example", MODEL_PATH="Qwen/example", PYTHON_BIN=sys.executable)
    if value is not None:
        env["EVAL_TIE_WORD_EMBEDDINGS"] = value
    subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)
    record = tmp_path / "evaluation_config.json"
    before = record.read_bytes()
    saved = json.loads(before)
    if value is None:
        assert "hf_overrides" not in saved
    else:
        assert saved["hf_overrides"] == {"tie_word_embeddings": value == "True"}
        env["EVAL_TIE_WORD_EMBEDDINGS"] = "1" if value == "True" else "0"
    if completed:
        (tmp_path / "evaluation_complete.json").write_text("{}")
    # Equivalent boolean spellings retain the exact existing config.
    subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)
    for changed in (None, "False", "True"):
        if changed == value:
            continue
        env.pop("EVAL_TIE_WORD_EMBEDDINGS", None)
        if changed is not None:
            env["EVAL_TIE_WORD_EMBEDDINGS"] = changed
        result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        assert result.returncode != 0 and "configuration differs" in result.stderr
        assert record.read_bytes() == before


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind"])
def test_disagreeing_step_inputs_fail_before_output(tmp_path, benchmark):
    result = launch(benchmark, "ace", tmp_path, RUN_DIR=tmp_path / "training", STEP="5", STEPS="10")
    assert result.returncode != 0 and "STEP and STEPS disagree" in result.stderr
    assert not (tmp_path / "training").exists()


@pytest.mark.parametrize("benchmark", ["mini_swe_bench", "mastermind"])
def test_ace_budget_override_keeps_the_preset_output(tmp_path, benchmark):
    result = launch(benchmark, "ace", tmp_path, LOCAL_EVAL_EPOCHS="4")
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    args = importlib.import_module(command[2]).parse_args(command[3:])
    assert args.local_eval_epochs == 4
    assert str(tmp_path / "runs/visits8") in args.metrics_file
    assert not (tmp_path / "runs").exists()


def test_completed_mastermind_launch_skips_server_and_evaluator(tmp_path):
    (tmp_path / 'evaluation_complete.json').write_text('{}')
    script = '''
set -euo pipefail
source "$RLLM_DIR/examples/mastermind/eval/common.sh"
for fn in eval_setup_python eval_check_resume_dir eval_lock_run eval_merge_checkpoint eval_prepare_tracking eval_select_server_url eval_record_config eval_unlock_run; do
    eval "$fn() { :; }"
done
eval_complete_run() { echo validated; }
eval_start_server() { echo unexpected-server >&2; return 77; }
EVAL_MODULE=unused
EVAL_ARGS=()
BASE_URL=http://localhost/v1
launch
'''
    env = {**os.environ, 'RLLM_DIR': str(REPO), 'RUN_DIR': str(tmp_path), 'PYTHON_BIN': sys.executable,
           'MODEL_SOURCE': 'hub', 'TASK_OFFSET': '0', 'TASK_LIMIT': '', 'RLLM_DRY_RUN': '0'}
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'validated'
