"""Evaluation commands and output/process lifecycle, without a GPU or API calls."""

import ast
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("evaluation_runtime", ROOT / "scripts/evaluation_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def command(benchmark, method, **overrides):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(ROOT), PYTHON_BIN=sys.executable, **overrides)
    result = subprocess.run(["bash", str(ROOT / f"examples/{benchmark}/eval/{method}/eval.slurm"), "--dry-run"], cwd="/tmp", env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def parse_evaluator(argv):
    # Execute the actual parser function alone: importing GPU engines would mask
    # launcher/parser mismatches behind missing optional dependencies.
    import argparse

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


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_method_matrix_matches_real_parser_and_preserves_budgets(benchmark, method, tmp_path):
    destination = tmp_path / "uncreated"
    run = command(benchmark, method, RUNS_BASE=str(destination))
    assert not destination.exists()
    item = run["evaluations"][0]
    parsed = parse_evaluator(item["evaluation"])
    assert run["preset"] == item["preset"] == ("visits8" if method == "ace" else "pass8")
    assert parsed.local_eval_epochs == item["expected_epochs"] == (8 if method == "ace" else 1)
    assert parsed.g_act == item["g_act"] == (1 if method == "ace" else 8)
    assert (parsed.g_ref, parsed.g_cur) == ((1, 1) if method == "ace" else (0, 0))
    assert run["task_concurrency"] == parsed.task_concurrency == (8 if method == "ace" else 1)
    assert parsed.n_parallel_agents == 8
    assert parsed.local_eval_protocol == "sequential_adaptation_v1"
    assert parsed.gen_temperature == 0.7 and parsed.gen_top_p == 1
    assert parsed.max_model_length == 40960
    assert parsed.max_turns == (6 if benchmark == "mbpp" else 10)
    assert not parsed.disable_thinking
    if benchmark == "mbpp":
        assert parsed.base_system_prompt_filename == ("base_system_prompt.txt" if method == "ace" else "base_system_prompt_v1.txt")
        assert parsed.gen_max_response_length == 8192
    if benchmark == "appworld":
        assert parsed.max_response_length == 32768
        assert parsed.reflector_max_history_tokens == 20000
    assert "--host" in item["server"] and item["server"][item["server"].index("--host") + 1] == "127.0.0.1"
    assert "--hf-overrides" not in item["server"] and "hf_overrides" not in item


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
@pytest.mark.parametrize("method", ("ace", "vanilla"))
@pytest.mark.parametrize(("value", "expected"), (("False", False), ("0", False), ("TRUE", True), ("1", True)))
def test_serving_head_override_reaches_real_command(benchmark, method, value, expected, tmp_path):
    output = tmp_path / "uncreated"
    item = command(benchmark, method, EVAL_TIE_WORD_EMBEDDINGS=value, OUTPUT_DIR=str(output))["evaluations"][0]
    assert item["hf_overrides"] == {"tie_word_embeddings": expected}
    assert json.loads(item["server"][item["server"].index("--hf-overrides") + 1]) == item["hf_overrides"]
    assert "--hf-overrides" not in item["evaluation"]
    assert not output.exists()


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
@pytest.mark.parametrize("value", ("", "yes", "2", " false "))
def test_invalid_serving_head_override_fails_before_runtime(benchmark, value, tmp_path):
    output = tmp_path / "uncreated"
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(ROOT), PYTHON_BIN="/missing/evaluation-python", OUTPUT_DIR=str(output), EVAL_TIE_WORD_EMBEDDINGS=value)
    result = subprocess.run(["bash", str(ROOT / f"examples/{benchmark}/eval/ace/eval.slurm")], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "EVAL_TIE_WORD_EMBEDDINGS must be" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
def test_api_model_rejects_serving_head_override(benchmark, tmp_path):
    output = tmp_path / "uncreated"
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    env.update(RLLM_DIR=str(ROOT), PYTHON_BIN="/missing/evaluation-python", MODEL_NAME="gemini-2.5-pro", EVAL_TIE_WORD_EMBEDDINGS="False", OUTPUT_DIR=str(output))
    result = subprocess.run(["bash", str(ROOT / f"examples/{benchmark}/eval/ace/eval.slurm")], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "EVAL_TIE_WORD_EMBEDDINGS requires" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_default_preset_is_the_methods_eight_sample_protocol(benchmark, method, tmp_path):
    default = command(benchmark, method, RUN_DIR=str(tmp_path / "qwen3-4b-thinking"), STEP="5")
    preset = "visits8" if method == "ace" else "pass8"
    explicit = command(benchmark, method, RUN_DIR=str(tmp_path / "qwen3-4b-thinking"), STEP="5", PRESET=preset)
    assert default["preset"] == preset
    assert default["evaluations"][0]["run_dir"] == str(tmp_path / f"qwen3-4b-thinking/eval/{method}/{preset}/step_5")
    assert default["evaluations"][0]["evaluation"] == explicit["evaluations"][0]["evaluation"]


@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_lane_suffix_names_a_separate_preset_directory(method, tmp_path):
    preset = "visits8" if method == "ace" else "pass8"
    env = {"RUN_DIR": str(tmp_path / "run"), "STEP": "10", "EVAL_LANE_SUFFIX": "matched-v1"}
    layout = runtime.evaluation_paths("mini_swe_bench", method, env, ROOT)
    assert layout["evaluations"][0]["run_dir"] == str(tmp_path / f"run/eval/{method}/{preset}-matched-v1/step_10")
    for value in ("../x", "a/b", "-x", "a b"):
        with pytest.raises(ValueError, match="EVAL_LANE_SUFFIX"):
            runtime.evaluation_paths("mini_swe_bench", method, {**env, "EVAL_LANE_SUFFIX": value}, ROOT)


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
@pytest.mark.parametrize("method", ("ace", "vanilla"))
@pytest.mark.parametrize("preset", ("standard", "extended", "baseline", "api", "gemini", "unknown", "other"))
def test_removed_and_unknown_presets_are_rejected(benchmark, method, preset):
    preset = {"other": "pass8" if method == "ace" else "visits8"}.get(preset, preset)
    with pytest.raises(subprocess.CalledProcessError):
        command(benchmark, method, PRESET=preset)
    with pytest.raises(ValueError, match="supports only PRESET"):
        runtime.evaluation_paths(benchmark, method, {"PRESET": preset, "MODEL_NAME": "Qwen/Qwen3-0.6B"}, ROOT)


@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_api_model_uses_shared_launcher_without_model_server(method):
    run = command("mbpp", method, MODEL_NAME="gemini-2.5-pro")
    item = run["evaluations"][0]
    parsed = parse_evaluator(item["evaluation"])
    assert run["model_source"] == "api"
    assert item["server"] == []
    assert parsed.base_url == ""
    assert item["run_dir"].endswith(f"eval/{method}/results/{run['preset']}/gemini-2.5-pro")


def test_checkpoint_series_keeps_steps_chunks_and_whitespace(tmp_path):
    run = command("appworld", "ace", CKPT_BASE=str(tmp_path / "model dir"), STEPS="10 20", MODEL_VARIANT="nonthinking", TASK_OFFSET="19", MAX_TASKS="19")
    assert len(run["evaluations"]) == 2
    for step, item in zip((10, 20), run["evaluations"], strict=True):
        parsed = parse_evaluator(item["evaluation"])
        assert parsed.model_name == str(tmp_path / "model dir" / f"global_step_{step}")
        assert parsed.task_offset == 19 and parsed.max_tasks == 19
        assert parsed.disable_thinking and not parsed.accumulate_thinking and not parsed.accumulate_reasoning
        assert parsed.wandb_step == step


@pytest.mark.parametrize("values", ({"MODEL_NAME": "my_checkpoint"}, {"STEPS": "1 2"}, {"TASK_OFFSET": "-1"}, {"MAX_TASKS": "-1"}, {"RESUME": "invalid"}, {"PRESET": "unknown"}))
def test_invalid_configuration_fails_before_start(values):
    with pytest.raises(subprocess.CalledProcessError):
        command("mbpp", "ace", **values)


def execute_fixture(tmp_path, program, *, resume=False):
    output = tmp_path / "run"
    metrics = output / "metrics_step0.json"
    command_args = [sys.executable, "-c", program, str(metrics)]
    item = {"step": 0, "expected_epochs": 1, "model": "gemini-test", "run_dir": str(output), "metrics_file": str(metrics), "evaluation": command_args, "server": [], "resume": resume}
    run = {"repo": str(ROOT), "model_source": "api", "benchmark": "mbpp", "evaluations": [item]}
    return run, item


@pytest.mark.parametrize("tied", (None, False, True))
def test_success_requires_metrics_and_matching_completion_record(tmp_path, tied):
    payload = {
        "model_name": "gemini-test",
        "evaluations": [
            {
                "step": 0,
                "local_eval_epochs": 1,
                "metrics": {"success_rate": 0, "avg_reward": 0, "max_reward_mean": 0},
                "local_epochs": [{"local_eval_epoch": 1, "metrics": {"n_solved": 0}, "tasks": [{"task_id": "one", "reward": 0}]}],
            }
        ],
    }
    program = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(" + repr(json.dumps(payload)) + ")"
    run, item = execute_fixture(tmp_path, program)
    if tied is not None:
        item["hf_overrides"] = {"tie_word_embeddings": tied}
    assert runtime.execute(run, dict(os.environ)) == 0
    assert runtime.check_outputs(item)
    assert runtime.execute(run, dict(os.environ)) == 0
    for filename in ("evaluation_config.json", "evaluation_complete.json"):
        saved = json.loads((Path(item["run_dir"]) / filename).read_text())
        if tied is None:
            assert "hf_overrides" not in saved
        else:
            assert saved["hf_overrides"] == item["hf_overrides"]
    for changed in (None, False, True):
        if changed is tied:
            continue
        altered = {key: value for key, value in item.items() if key != "hf_overrides"}
        if changed is not None:
            altered["hf_overrides"] = {"tie_word_embeddings": changed}
        with pytest.raises(ValueError, match="different configuration"):
            runtime.check_outputs(altered)
        with pytest.raises(ValueError, match="completion receipt disagrees"):
            runtime.write_completion(altered)
    altered = dict(item, evaluation=[*item["evaluation"], "changed"])
    with pytest.raises(ValueError, match="different configuration"):
        runtime.check_outputs(altered)


@pytest.mark.parametrize("tied", (None, False, True))
def test_partial_output_requires_unchanged_serving_head_override(tmp_path, tied):
    _, item = execute_fixture(tmp_path, "pass", resume=True)
    if tied is not None:
        item["hf_overrides"] = {"tie_word_embeddings": tied}
    output = Path(item["run_dir"])
    output.mkdir()
    record = output / "evaluation_config.json"
    record.write_text(json.dumps({**item, "evaluation": runtime.recorded_command(item["evaluation"])}))
    original = record.read_bytes()
    assert runtime.check_outputs(item) is False
    for changed in (None, False, True):
        if changed is tied:
            continue
        altered = {key: value for key, value in item.items() if key != "hf_overrides"}
        if changed is not None:
            altered["hf_overrides"] = {"tie_word_embeddings": changed}
        with pytest.raises(ValueError, match="different evaluation"):
            runtime.check_outputs(altered)
    assert record.read_bytes() == original


@pytest.mark.parametrize("program", ("raise SystemExit(7)", "pass"))
def test_failed_evaluation_is_nonzero_and_preserves_evidence(tmp_path, program):
    run, item = execute_fixture(tmp_path, program)
    with pytest.raises(RuntimeError):
        runtime.execute(run, dict(os.environ))
    output = Path(item["run_dir"])
    assert (output / "evaluation.log").exists()
    assert not (output / "evaluation_complete.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        runtime.execute(run, dict(os.environ))
    assert (output / "evaluation_config.json").exists()


def test_empty_partial_output_retries_in_same_directory(tmp_path):
    output = tmp_path / "run"
    marker = tmp_path / "ready"
    program = f"import pathlib,sys; p=pathlib.Path({str(marker)!r}); p.exists() or sys.exit(7); pathlib.Path(sys.argv[1]).write_text({json.dumps({'model_name': 'gemini-test', 'evaluations': [{'step': 0, 'local_eval_epochs': 1, 'metrics': {'success_rate': 0, 'avg_reward': 0, 'max_reward_mean': 0}, 'local_epochs': [{'local_eval_epoch': 1, 'metrics': {'n_solved': 0}, 'tasks': [{'task_id': 'one', 'reward': 0}]}]}]})!r})"
    run, item = execute_fixture(tmp_path, program, resume=True)
    with pytest.raises(RuntimeError, match="Evaluator failed"):
        runtime.execute(run, dict(os.environ))
    assert (output / "evaluation_config.json").exists()
    marker.write_text("ready")
    assert runtime.execute(run, dict(os.environ)) == 0
    assert runtime.check_outputs(item)


def test_startup_only_retry_can_change_interpreter_but_completed_tasks_cannot(tmp_path):
    run, item = execute_fixture(tmp_path, "pass", resume=True)
    output = Path(item["run_dir"])
    output.mkdir()
    previous = dict(item, evaluation=["/old/python", *item["evaluation"][1:]])
    (output / "evaluation_config.json").write_text(json.dumps({**previous, "evaluation": runtime.recorded_command(previous["evaluation"])}))
    assert runtime.check_outputs(item) is False
    (output / "task_one/completed_visits").mkdir(parents=True)
    (output / "task_one/completed_visits/step000000_iter_0.json").write_text("{}")
    with pytest.raises(ValueError, match="different evaluation"):
        runtime.check_outputs(item)


def test_appworld_sidecar_startup_logs_do_not_strand_empty_output(tmp_path):
    _, item = execute_fixture(tmp_path, "pass", resume=True)
    output = Path(item["run_dir"])
    (output / "service_logs").mkdir(parents=True)
    (output / "service_logs/appworld.log").write_text("startup failed")
    assert runtime.check_outputs(item) is False
    (output / "unrelated.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        runtime.check_outputs(item)


def test_existing_outputs_are_never_deleted(tmp_path):
    run, item = execute_fixture(tmp_path, "pass")
    output = Path(item["run_dir"])
    output.mkdir()
    evidence = output / "trajectory.json"
    evidence.write_text("keep")
    with pytest.raises(ValueError):
        runtime.execute(run, dict(os.environ))
    assert evidence.read_text() == "keep"


def test_busy_port_is_refused():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(OSError):
            runtime.free_port(str(listener.getsockname()[1]))


def test_owned_process_stops_on_exception(tmp_path):
    with pytest.raises(RuntimeError, match="test failure"):
        with runtime.owned_process([sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ), tmp_path / "child.log") as child:
            assert child.poll() is None
            raise RuntimeError("test failure")
    assert child.poll() is not None


def test_completion_identity_ignores_random_service_ports_but_keeps_sampling():
    first = ["python", "--base_url", "http://127.0.0.1:20001/v1", "--gen_temperature", "0.7"]
    second = ["python", "--base_url", "http://127.0.0.1:20002/v1", "--gen_temperature", "0.7"]
    assert runtime.recorded_command(first) == runtime.recorded_command(second)
    second[-1] = "0.8"
    assert runtime.recorded_command(first) != runtime.recorded_command(second)


def test_partial_and_single_task_selections_do_not_claim_all_tasks():
    suffix = command("appworld", "ace", TASK_OFFSET="19")["evaluations"][0]
    single = command("appworld", "ace", TASK_ID="task_example")["evaluations"][0]
    assert suffix["task_label"] == "from19"
    assert single["task_label"].startswith("id_")
    with pytest.raises(subprocess.CalledProcessError):
        command("appworld", "ace", TASK_ID="task_example", MAX_TASKS="2")


def test_appworld_checkpoint_requires_explicit_thinking_configuration(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        command("appworld", "ace", CKPT_BASE=str(tmp_path), STEPS="10")


def test_appworld_pool_preserves_parallelism_and_external_server_is_single_task():
    result = command("appworld", "ace", N_PARALLEL_AGENTS="4", TASK_CONCURRENCY="4")
    args = parse_evaluator(result["evaluations"][0]["evaluation"])
    assert result["appworld_servers"] == args.n_servers == args.n_parallel_agents == 4
    with pytest.raises(subprocess.CalledProcessError):
        command("appworld", "ace", APPWORLD_URL="http://localhost:9000")
    result = command("appworld", "ace", APPWORLD_URL="http://localhost:9000", N_PARALLEL_AGENTS="1", TASK_CONCURRENCY="1")
    assert result["appworld_servers"] == 1


def test_raw_overrides_cannot_override_planned_identity():
    with pytest.raises(ValueError, match="environment variables"):
        runtime.plan("mbpp", "ace", {}, ["--wandb_step", "999"])


@pytest.mark.parametrize("payload", ({}, {"model_name": "different"}, {"model_name": "gemini-test", "evaluations": []}))
def test_empty_or_wrong_model_metrics_cannot_complete(tmp_path, payload):
    run, item = execute_fixture(tmp_path, "pass")
    Path(item["run_dir"]).mkdir()
    Path(item["metrics_file"]).write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        runtime.validate_metrics(item)


def test_completion_is_rechecked_after_a_competing_launcher_finishes(tmp_path, monkeypatch):
    payload = {
        "model_name": "gemini-test",
        "evaluations": [
            {
                "step": 0,
                "local_eval_epochs": 1,
                "metrics": {"success_rate": 0, "avg_reward": 0, "max_reward_mean": 0},
                "local_epochs": [{"local_eval_epoch": 1, "metrics": {"n_solved": 0}, "tasks": [{"task_id": "one", "reward": 0}]}],
            }
        ],
    }
    tally = tmp_path / "executions"
    program = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(" + repr(json.dumps(payload)) + "); "
    program += f"open({str(tally)!r}, 'a').write('ran\\n')"
    run, item = execute_fixture(tmp_path, program)
    original = runtime.check_outputs
    first = True

    def compete(item, *, ignore_lock=False):
        nonlocal first
        if first:
            first = False
            absent = original(item)
            # Another launcher completes after this one's initial check.
            runtime.execute(run, dict(os.environ))
            return absent
        return original(item, ignore_lock=ignore_lock)

    monkeypatch.setattr(runtime, "check_outputs", compete)
    runtime.execute(run, dict(os.environ))
    assert tally.read_text().splitlines() == ["ran"]


def test_setup_checks_verl_before_environment_mutation(tmp_path):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    unpatched = tmp_path / "unpatched-verl"
    unpatched.mkdir()
    env.update(VENV_DIR=str(tmp_path / "never-created"), RLLM_VERL_SOURCE=str(unpatched), GPU_FAMILY="l40s")
    result = subprocess.run(["bash", str(ROOT / "scripts/setup_qwen35.sh")], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "never-created").exists()


def test_managed_appworld_sidecars_own_roots_and_bind_loopback(tmp_path, monkeypatch):
    shared = tmp_path / "shared-appworld"
    (shared / "data").mkdir(parents=True)
    (shared / "data" / "task.txt").write_text("shared task input")
    existing = shared / "experiments" / "outputs" / "ace_training" / "same-task"
    existing.mkdir(parents=True)
    (existing / "result.json").write_text("keep existing experiment")

    # Run the launcher's actual sidecar command in child interpreters. The fake
    # server writes a repeated task into AppWorld's real root-relative layout.
    modules = tmp_path / "modules"
    package = modules / "appworld"
    (package / "serve").mkdir(parents=True)
    (package / "__init__.py").write_text("configured_root = None\ndef update_root(root):\n    global configured_root\n    configured_root = root\n")
    (package / "serve" / "__init__.py").write_text("")
    (package / "serve" / "environment.py").write_text(
        "from appworld import configured_root\n"
        "assert configured_root is not None, 'update_root must precede server imports'\n"
        "app = object()\n"
        "def run(port):\n"
        "    raise AssertionError('Do not delegate to the public server wrapper')\n"
    )
    (modules / "uvicorn.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "def run(app, *, host, port):\n"
        "    assert host == '127.0.0.1', 'Managed server must bind loopback'\n"
        "    root = Path(os.environ['APPWORLD_ROOT'])\n"
        "    output = root / 'experiments' / 'outputs' / 'ace_training' / 'same-task'\n"
        "    output.mkdir(parents=True)\n"
        "    payload = {'port': port, 'task': (root / 'data' / 'task.txt').read_text()}\n"
        "    (output / 'result.json').write_text(json.dumps(payload))\n"
        "    assert (root / 'data').is_symlink()\n"
        "    with Path(os.environ['SIDECAR_ROOTS']).open('a') as stream: stream.write(str(root) + '\\n')\n"
        "    print('sidecar evidence retained', flush=True)\n"
    )

    def sidecar_ready(process, url, **kwargs):
        assert url in {"http://127.0.0.1:19000", "http://127.0.0.1:19001"}
        assert process.wait(timeout=10) == 0

    monkeypatch.setattr(runtime, "wait_ready", sidecar_ready)
    env = dict(
        os.environ,
        APPWORLD_DATA_DIR=str(shared),
        APPWORLD_SERVER_VENV=str(Path(sys.executable).parent.parent),
        APPWORLD_PORT="19000",
        SLURM_TMPDIR=str(tmp_path / "runtime"),
        SIDECAR_ROOTS=str(tmp_path / "sidecar_roots.txt"),
        PYTHONPATH=str(modules),
    )
    # Two launches in the same parent process also exercise runtime uniqueness;
    # reuse of the PID or ports must not reuse a previous sidecar's writable root.
    for job in ("first", "second"):
        run, _ = execute_fixture(tmp_path / job, "raise SystemExit(7)")
        run.update(benchmark="appworld", appworld_servers=2)
        with pytest.raises(RuntimeError, match="Evaluator failed with status 7"):
            runtime.execute(run, env)

    roots = [Path(line) for line in (tmp_path / "sidecar_roots.txt").read_text().splitlines()]
    assert len(roots) == 4
    assert len({root.parent for root in roots}) == 2
    assert all(not root.parent.exists() for root in roots)
    for job in ("first", "second"):
        for port in (19000, 19001):
            assert (tmp_path / job / "run" / "service_logs" / f"appworld-{port}.log").read_text() == "sidecar evidence retained\n"
    assert (existing / "result.json").read_text() == "keep existing experiment"
    assert (shared / "data" / "task.txt").read_text() == "shared task input"
    assert sorted(path.name for path in shared.iterdir()) == ["data", "experiments"]


def test_gpu_keepalive_is_owned_only_for_gpu_evaluations_inside_allocations(tmp_path, monkeypatch):
    import contextlib

    started = []
    real_owned_process = runtime.owned_process

    @contextlib.contextmanager
    def recording(command, env, log):
        started.append((command, log.name))
        with real_owned_process(command, env, log) as process:
            yield process

    monkeypatch.setattr(runtime, "owned_process", recording)
    keepalive = [sys.executable, str(ROOT / "scripts/gpu_keepalive.py")]

    def attempt(name, model_source, **env_overrides):
        (tmp_path / name).mkdir()
        run, _ = execute_fixture(tmp_path / name, "pass")
        run["model_source"] = model_source
        env = {key: value for key, value in os.environ.items() if key not in {"RLLM_GPU_KEEPALIVE", "RLLM_DISABLE_GPU_KEEPALIVE"}}
        env.update(PYTHON_BIN=sys.executable, **env_overrides)
        started.clear()
        with pytest.raises(RuntimeError, match="without writing metrics"):
            runtime.execute(run, env)
        return [command for command, _ in started], [log for _, log in started]

    commands, logs = attempt("gpu-in-slurm", "hub", SLURM_JOB_ID="123")
    assert commands[0] == keepalive and logs[0] == "gpu-keepalive-123.log"
    assert (tmp_path / "gpu-in-slurm" / "gpu-keepalive-123.log").is_file()
    assert not (tmp_path / "gpu-in-slurm" / "run" / "gpu-keepalive-123.log").exists()
    commands, _ = attempt("api-in-slurm", "api", SLURM_JOB_ID="123")
    assert keepalive not in commands
    env_without_slurm = {key: value for key, value in os.environ.items() if key != "SLURM_JOB_ID"}
    monkeypatch.setattr(os, "environ", env_without_slurm)
    commands, _ = attempt("gpu-outside-slurm", "hub")
    assert keepalive not in commands
    commands, _ = attempt("gpu-disabled", "hub", SLURM_JOB_ID="123", RLLM_GPU_KEEPALIVE="0")
    assert keepalive not in commands


def test_owned_vllm_does_not_inherit_unrelated_server_auth(tmp_path, monkeypatch):
    run, item = execute_fixture(tmp_path, "raise SystemExit(7)")
    auth_presence = tmp_path / "server-auth.json"
    item["server"] = [
        sys.executable,
        "-c",
        "import json,os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps('VLLM_API_KEY' in os.environ))",
        str(auth_presence),
    ]
    run.update(model_source="hub", base_url="http://127.0.0.1:19002/v1")

    def server_ready(process, url, **kwargs):
        assert url == "http://127.0.0.1:19002/v1/models"
        assert process.wait(timeout=10) == 0

    monkeypatch.setattr(runtime, "wait_ready", server_ready)
    env = dict(os.environ, VLLM_API_KEY="test-only-inherited-key")
    with pytest.raises(RuntimeError, match="Evaluator failed with status 7"):
        runtime.execute(run, env)
    assert json.loads(auth_presence.read_text()) is False
    assert env["VLLM_API_KEY"] == "test-only-inherited-key"


@pytest.mark.parametrize("method", ("ace", "vanilla"))
@pytest.mark.parametrize("selection", ({"EVAL_PROTOCOL": "training_input_parity"}, {"TRAINING_INPUT_PARITY": "True"}))
def test_appworld_training_input_parity_protocol_is_explicit_and_preserves_method_budget(method, selection):
    result = command("appworld", method, **selection)
    item = result["evaluations"][0]
    parsed = parse_evaluator(item["evaluation"])
    assert result["protocol"] == item["protocol"] == "training_input_parity"
    preset = "visits8" if method == "ace" else "pass8"
    assert result["preset"] == preset
    assert item["run_dir"].endswith(f"eval/{method}/results/{preset}/Qwen3-4B")
    assert parsed.training_input_parity
    assert (parsed.g_act, parsed.g_ref, parsed.g_cur) == ((1, 1, 1) if method == "ace" else (8, 0, 0))
    assert parsed.local_eval_epochs == (8 if method == "ace" else 1)
    assert parsed.max_turns == 10
    assert parsed.selection_mode == "best_rejection"
    assert parsed.max_model_length == 32768
    assert item["server"][item["server"].index("--max-model-len") + 1] == "32768"
    assert (parsed.reflector_max_prompt_length, parsed.reflector_max_response_length, parsed.reflector_max_history_tokens) == (30720, 2048, 20000)
    assert (parsed.curator_max_prompt_length, parsed.curator_max_response_length) == (8192, 4096)
    assert (parsed.gen_temperature, parsed.gen_top_p, parsed.ace_temperature) == (0.7, 1, 0.7)
    assert parsed.trajectory_hard_timeout == 7200
    assert "--trajectory_hard_timeout" in item["evaluation"]


def test_training_input_parity_keeps_explicit_numeric_overrides_and_identity():
    result = command(
        "appworld",
        "ace",
        EVAL_PROTOCOL="training_input_parity",
        MAX_MODEL_LEN="49152",
        REFLECTOR_MAX_PROMPT_LENGTH="30000",
        GEN_TEMPERATURE="0.5",
        ACE_TEMPERATURE="0.6",
        TRAJECTORY_HARD_TIMEOUT="3600",
    )
    parsed = parse_evaluator(result["evaluations"][0]["evaluation"])
    assert parsed.max_model_length == 49152
    assert parsed.reflector_max_prompt_length == 30000
    assert (parsed.gen_temperature, parsed.ace_temperature, parsed.trajectory_hard_timeout) == (0.5, 0.6, 3600)
    legacy = command("appworld", "ace")
    assert legacy["protocol"] == "legacy"
    assert "--training_input_parity" not in legacy["evaluations"][0]["evaluation"]
    assert runtime.recorded_command(result["evaluations"][0]["evaluation"]) != runtime.recorded_command(legacy["evaluations"][0]["evaluation"])


@pytest.mark.parametrize(
    "settings",
    (
        {"EVAL_PROTOCOL": "unknown"},
        {"EVAL_PROTOCOL": "legacy", "TRAINING_INPUT_PARITY": "True"},
        {"EVAL_PROTOCOL": "training_input_parity", "TRAINING_INPUT_PARITY": "False"},
        {"EVAL_PROTOCOL": "training_input_parity", "PRESET": "standard"},
        {"EVAL_PROTOCOL": "training_input_parity", "MODEL_NAME": "gpt-test"},
        {"EVAL_PROTOCOL": "training_input_parity", "TRAJECTORY_HARD_TIMEOUT": "0"},
    ),
)
def test_invalid_or_incompatible_parity_selection_fails_before_start(settings):
    with pytest.raises(subprocess.CalledProcessError):
        command("appworld", "ace", **settings)


@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_appworld_task_concurrency_is_explicit_and_keeps_budget(method):
    capacity = {"N_PARALLEL_AGENTS": "32", "N_SERVERS": "32"}
    serial = command("appworld", method, EVAL_PROTOCOL="training_input_parity", TASK_CONCURRENCY="1", **capacity)
    concurrent = command("appworld", method, EVAL_PROTOCOL="training_input_parity", TASK_CONCURRENCY="4", **capacity)
    serial_args = parse_evaluator(serial["evaluations"][0]["evaluation"])
    args = parse_evaluator(concurrent["evaluations"][0]["evaluation"])
    assert serial["task_concurrency"] == serial_args.task_concurrency == 1
    assert concurrent["task_concurrency"] == args.task_concurrency == 4
    assert args.n_servers == args.n_parallel_agents == 32
    for field in ("g_act", "g_ref", "g_cur", "local_eval_epochs", "seed", "max_model_length", "gen_temperature", "reflector_max_history_tokens"):
        assert getattr(args, field) == getattr(serial_args, field)
    assert runtime.recorded_command(serial["evaluations"][0]["evaluation"]) != runtime.recorded_command(concurrent["evaluations"][0]["evaluation"])
    assert "--task_concurrency" not in serial["evaluations"][0]["evaluation"]


def test_appworld_concurrent_pass8_reserves_sixteen_servers():
    planned = command("appworld", "vanilla", TASK_CONCURRENCY="2", G_ACT="8", N_PARALLEL_AGENTS="16", N_SERVERS="16")
    args = parse_evaluator(planned["evaluations"][0]["evaluation"])
    assert (args.task_concurrency, args.g_act, args.n_parallel_agents, args.n_servers) == (2, 8, 16, 16)
    assert (args.g_ref, args.g_cur, args.local_eval_epochs) == (0, 0, 1)


@pytest.mark.parametrize(
    "benchmark,settings",
    [
        ("appworld", {"TASK_CONCURRENCY": "0"}),
        ("appworld", {"TASK_CONCURRENCY": "4", "N_SERVERS": "2", "N_PARALLEL_AGENTS": "2"}),
        ("appworld", {"TASK_CONCURRENCY": "4", "G_ACT": "3"}),
        ("appworld", {"TASK_CONCURRENCY": "2", "G_ACT": "8", "N_PARALLEL_AGENTS": "16", "N_SERVERS": "15"}),
        ("appworld", {"TASK_CONCURRENCY": "4", "APPWORLD_URL": "http://localhost:9000", "N_PARALLEL_AGENTS": "1"}),
        ("mbpp", {"TASK_CONCURRENCY": "4", "N_PARALLEL_AGENTS": "2"}),
        ("mbpp", {"TASK_CONCURRENCY": "4", "G_ACT": "2", "N_PARALLEL_AGENTS": "7"}),
        ("mbpp", {"TASK_CONCURRENCY": "0"}),
    ],
)
def test_invalid_task_concurrency_fails_before_services_start(benchmark, settings):
    with pytest.raises(subprocess.CalledProcessError):
        command(benchmark, "ace", **settings)


@pytest.mark.parametrize("benchmark", ("mbpp",))
@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_small_benchmark_task_concurrency_is_explicit_and_keeps_budget(benchmark, method):
    serial = command(benchmark, method, TASK_CONCURRENCY="1", N_PARALLEL_AGENTS="32")
    concurrent = command(benchmark, method, TASK_CONCURRENCY="4", N_PARALLEL_AGENTS="32")
    serial_args = parse_evaluator(serial["evaluations"][0]["evaluation"])
    args = parse_evaluator(concurrent["evaluations"][0]["evaluation"])
    assert serial["task_concurrency"] == serial_args.task_concurrency == 1
    assert concurrent["task_concurrency"] == args.task_concurrency == 4
    assert "--task_concurrency" not in serial["evaluations"][0]["evaluation"]
    assert args.n_parallel_agents == serial_args.n_parallel_agents  # workers share the launcher's capacity
    for field in ("g_act", "g_ref", "g_cur", "local_eval_epochs", "seed", "max_model_length", "gen_temperature", "max_turns"):
        assert getattr(args, field) == getattr(serial_args, field)


def test_final_policy_defaults_and_reuse_controls_match_parser(tmp_path):
    item = command("appworld", "ace")["evaluations"][0]
    assert item["final_policy_eval"] is False
    assert parse_evaluator(item["evaluation"]).final_policy_eval is False
    assert item["run_dir"].endswith("eval/ace/results/visits8/Qwen3-4B")
    legacy = command("appworld", "ace", FINAL_POLICY_EVAL="False")["evaluations"][0]
    assert not legacy["final_policy_eval"]
    assert not parse_evaluator(legacy["evaluation"]).final_policy_eval
    vanilla = command("appworld", "vanilla")["evaluations"][0]
    assert not vanilla["final_policy_eval"]
    source = tmp_path / "completed run"
    replay = command("appworld", "ace", LOCAL_EVAL_PROTOCOL="legacy_search_v1", FINAL_POLICY_FROM_RUN_DIR=str(source))["evaluations"][0]
    args = parse_evaluator(replay["evaluation"])
    assert args.final_policy_from_run_dir == str(source)
    assert args.local_eval_epochs == 8
    assert not source.exists(), "Planning must not open or create source/output artifacts"
    receipt = tmp_path / "verified source receipt.json"
    verified = command("appworld", "ace", LOCAL_EVAL_PROTOCOL="legacy_search_v1", FINAL_POLICY_FROM_RUN_DIR=str(source), FINAL_POLICY_SOURCE_MANIFEST=str(receipt))["evaluations"][0]
    assert verified["final_policy_source_manifest"] == str(receipt)
    assert parse_evaluator(verified["evaluation"]).final_policy_source_manifest == str(receipt)
    assert not receipt.exists(), "Planning must not open or create provenance receipts"


@pytest.mark.parametrize(
    "benchmark,method,settings",
    [
        ("appworld", "ace", {"FINAL_POLICY_EVAL": "False", "FINAL_POLICY_FROM_RUN_DIR": "/source"}),
        ("appworld", "vanilla", {"FINAL_POLICY_EVAL": "True"}),
        ("mbpp", "ace", {"FINAL_POLICY_EVAL": "True"}),
        ("appworld", "ace", {"FINAL_POLICY_SOURCE_MANIFEST": "/receipt.json"}),
        ("appworld", "vanilla", {"FINAL_POLICY_SOURCE_MANIFEST": "/receipt.json"}),
        ("appworld", "ace", {"FINAL_POLICY_FROM_RUN_DIR": "/source", "CKPT_BASE": "/checkpoints", "STEPS": "10 20", "MODEL_VARIANT": "thinking"}),
    ],
)
def test_invalid_final_policy_controls_fail_in_dry_run(benchmark, method, settings):
    with pytest.raises(subprocess.CalledProcessError):
        command(benchmark, method, **settings)


def test_final_policy_completion_requires_full_separate_actor_pass(tmp_path):
    import copy

    metrics = {"success_rate": 0.5, "avg_reward": 0.5, "max_reward_mean": 1.0}
    tasks = [{"task_id": "a", "reward": 0}, {"task_id": "b", "reward": 1}]
    final = {
        "kind": "final_policy_only",
        "after_local_eval_epoch": 2,
        "policy_visit": 3,
        "adaptation_applied": False,
        "tasks": tasks,
        "metrics": metrics,
        "work": {"policy_rollouts": 2, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0},
    }
    entry = {
        "step": 10,
        "local_eval_epochs": 2,
        "tasks": tasks,
        "local_epochs": [{"local_eval_epoch": i, "tasks": tasks, "metrics": metrics} for i in (1, 2)],
        "metrics": {**metrics, **{f"final_policy/{k}": v for k, v in metrics.items()}},
        "final_policy_evaluation": final,
    }
    payload = {"model_name": "same-model", "evaluations": [entry]}
    path = tmp_path / "metrics_step10.json"
    item = {"metrics_file": str(path), "model": "same-model", "step": 10, "expected_epochs": 2, "final_policy_eval": True, "g_act": 1}
    path.write_text(json.dumps(payload))
    assert runtime.validate_metrics(item)
    mutations = [
        lambda e: e.pop("final_policy_evaluation"),
        lambda e: e["final_policy_evaluation"].update(policy_visit=2),
        lambda e: e["final_policy_evaluation"]["tasks"].pop(),
        lambda e: e["final_policy_evaluation"]["tasks"].append(dict(e["final_policy_evaluation"]["tasks"][0])),
        lambda e: e["final_policy_evaluation"]["work"].update(policy_rollouts=1),
        lambda e: e["final_policy_evaluation"]["work"].update(reflector_rollouts=1),
        lambda e: e["metrics"].update({"final_policy/avg_reward": 0.9}),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(payload)
        mutate(changed["evaluations"][0])
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            runtime.validate_metrics(item)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unrequested"):
        runtime.validate_metrics({**item, "final_policy_eval": False})


def partial_rows_final_metrics(tmp_path, observed_ids):
    metrics = {"success_rate": 0.5, "avg_reward": 0.5, "max_reward_mean": 1.0}
    tasks = [{"task_id": "a", "reward": 0}, {"task_id": "b", "reward": 1}]
    observed = [{"task_id": identifier, "reward": 0} for identifier in observed_ids]
    entry = {
        "step": 10,
        "local_eval_epochs": 2,
        "evaluation_task_ids": ["a", "b"],
        "tasks": observed,
        "local_epochs": [{"local_eval_epoch": i, "tasks": observed, "metrics": {**metrics, "n_solved": 2}} for i in (1, 2)],
        "metrics": {**metrics, **{f"final_policy/{key}": value for key, value in metrics.items()}},
        "final_policy_evaluation": {
            "kind": "final_policy_only",
            "after_local_eval_epoch": 2,
            "policy_visit": 3,
            "adaptation_applied": False,
            "tasks": tasks,
            "metrics": metrics,
            "work": {"policy_rollouts": 2, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0},
        },
    }
    path = tmp_path / "metrics_step10.json"
    payload = {"model_name": "same-model", "evaluations": [entry]}
    path.write_text(json.dumps(payload))
    item = {"metrics_file": str(path), "model": "same-model", "step": 10, "expected_epochs": 2, "final_policy_eval": True, "g_act": 1}
    return path, payload, item


@pytest.mark.parametrize("observed_ids", [[], ["b"]])
def test_final_policy_completion_uses_declared_full_task_set_when_rows_omit_frozen_tasks(tmp_path, observed_ids):
    _, _, item = partial_rows_final_metrics(tmp_path, observed_ids)
    assert runtime.validate_metrics(item)


@pytest.mark.parametrize("declared", [None, "a,b", [], ["a", "a"], ["a", None], ["a", " "]])
def test_final_policy_completion_rejects_invalid_declared_task_ids(tmp_path, declared):
    path, payload, item = partial_rows_final_metrics(tmp_path, [])
    payload["evaluations"][0]["evaluation_task_ids"] = declared
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="evaluation_task_ids"):
        runtime.validate_metrics(item)


@pytest.mark.parametrize("declared", [["a"], ["a", "b", "c"], ["a", "c"]])
def test_final_policy_completion_requires_exact_declared_final_coverage(tmp_path, declared):
    path, payload, item = partial_rows_final_metrics(tmp_path, [])
    payload["evaluations"][0]["evaluation_task_ids"] = declared
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Final policy task coverage"):
        runtime.validate_metrics(item)


def test_final_policy_completion_rejects_observed_tasks_outside_declared_set(tmp_path):
    path, payload, item = partial_rows_final_metrics(tmp_path, ["b"])
    payload["evaluations"][0]["local_epochs"][0]["tasks"] = [{"task_id": "unknown"}]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="outside evaluation_task_ids"):
        runtime.validate_metrics(item)


def test_final_policy_completion_retains_legacy_task_inference_but_requires_evidence(tmp_path):
    path, payload, item = partial_rows_final_metrics(tmp_path, ["a", "b"])
    payload["evaluations"][0].pop("evaluation_task_ids")
    path.write_text(json.dumps(payload))
    assert runtime.validate_metrics(item)
    payload["evaluations"][0]["tasks"] = []
    for epoch in payload["evaluations"][0]["local_epochs"]:
        epoch["tasks"] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Final policy task coverage"):
        runtime.validate_metrics(item)


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
@pytest.mark.parametrize("method", ("ace", "vanilla"))
def test_training_run_resolves_exact_checkpoint_and_output(tmp_path, benchmark, method):
    root = tmp_path / "training run"
    item = command(benchmark, method, RUN_DIR=str(root), STEP="215", MODEL_VARIANT="thinking")["evaluations"][0]
    checkpoint = root / "checkpoints/global_step_215"
    assert item["checkpoint_source"] == str(checkpoint)
    assert item["model"] == str(root / "checkpoints_merged/global_step_215")
    preset = "visits8" if method == "ace" else "pass8"
    assert item["run_dir"] == str(root / f"eval/{method}/{preset}/step_215")
    assert item["metrics_file"] == str(root / f"eval/{method}/{preset}/step_215/metrics_step215.json")
    assert item["benchmark"] == benchmark and item["method"] == method
    assert not root.exists()


@pytest.mark.parametrize("settings", ({}, {"MODEL_PATH": "/conflicting/model"}, {"CKPT_BASE": "/conflicting/checkpoints"}, {"STEP": "-1"}))
def test_training_run_requires_explicit_unambiguous_step(tmp_path, settings):
    values = {"RUN_DIR": str(tmp_path / "run"), "MODEL_VARIANT": "thinking", **settings}
    if settings and "STEP" not in settings:
        values["STEP"] = "215"
    with pytest.raises(subprocess.CalledProcessError):
        command("appworld", "ace", **values)


def test_checkpoint_merger_uses_exact_selected_source_and_preserves_training_state(tmp_path, monkeypatch):
    source = tmp_path / "training/checkpoints/global_step_7"
    (source / "actor/huggingface").mkdir(parents=True)
    (source / "actor/huggingface/config.json").write_text("{}")
    optimizer = source / "actor/optim_world_size_2_rank_0.pt"
    optimizer.write_text("training optimizer")
    run, item = execute_fixture(tmp_path / "output", "raise SystemExit(7)")
    target = tmp_path / "training/checkpoints_merged/global_step_7"
    item.update(checkpoint_source=str(source), model=str(target))
    run["model_source"] = "checkpoint"
    calls = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(RuntimeError, match="Evaluator failed"):
        runtime.execute(run, {**os.environ, "PYTHON_BIN": sys.executable})
    assert len(calls) == 1
    assert calls[0][calls[0].index("--source-dir") + 1] == str(source)
    assert calls[0][calls[0].index("--target-dir") + 1] == str(target)
    assert optimizer.read_text() == "training optimizer"


def test_evaluation_logging_keeps_caller_entity_and_its_output_directory(tmp_path):
    run, item = execute_fixture(tmp_path, "import json,os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({k:os.environ[k] for k in ['WANDB_ENTITY','WANDB_DIR']})); raise SystemExit(7)")
    with pytest.raises(RuntimeError, match="Evaluator failed"):
        runtime.execute(run, {**os.environ, "WANDB_PROJECT": "fixture", "WANDB_ENTITY": "personal-account"})
    logged = json.loads(Path(item["metrics_file"]).read_text())
    assert logged == {"WANDB_ENTITY": "personal-account", "WANDB_DIR": item["run_dir"]}


def test_training_run_record_supplies_model_and_thinking_mode(tmp_path):
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    (recorded / "run.json").write_text(json.dumps({"model_variant": "instruct", "training_identity": {"model": {"path": "Qwen/Qwen3-4B-Instruct-2507"}, "rllm": {"disable_thinking": True}}}))
    assert runtime.run_defaults(recorded) == {"MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507", "MODEL_VARIANT": "nonthinking", "DISABLE_THINKING": "True"}
    # Runs without a record keep the launcher's `<model>-<variant>` directory name.
    assert runtime.run_defaults(tmp_path / "qwen3-0.6b-thinking") == {"MODEL_VARIANT": "thinking", "DISABLE_THINKING": "False"}
    assert runtime.run_defaults(tmp_path / "qwen3-4b-thinking-2507")["MODEL_VARIANT"] == "thinking"
    assert runtime.run_defaults(tmp_path / "unlabelled") == {}


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
def test_run_directory_selects_thinking_mode_without_explicit_variant(tmp_path, benchmark):
    item = command(benchmark, "ace", RUN_DIR=str(tmp_path / "qwen3-4b-nonthinking"), STEP="5")["evaluations"][0]
    assert parse_evaluator(item["evaluation"]).disable_thinking is True


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp"))
def test_eight_sample_presets_have_their_own_output_and_budget(tmp_path, benchmark):
    root = tmp_path / "qwen3-4b-thinking"
    pass8 = command(benchmark, "vanilla", RUN_DIR=str(root), STEP="5", PRESET="pass8")["evaluations"][0]
    visits8 = command(benchmark, "ace", RUN_DIR=str(root), STEP="5", PRESET="visits8")["evaluations"][0]
    assert pass8["run_dir"] == str(root / "eval/vanilla/pass8/step_5")
    assert (pass8["g_act"], pass8["expected_epochs"]) == (8, 1)
    assert visits8["run_dir"] == str(root / "eval/ace/visits8/step_5")
    assert (visits8["g_act"], visits8["expected_epochs"]) == (1, 8)
