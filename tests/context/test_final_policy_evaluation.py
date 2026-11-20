"""Final playbook scoring and fail-closed reuse of completed adaptation artifacts."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_local_evaluation_concurrency import Engine, Recorder, setup_run as _setup_run

from rllm.algorithms.ace import evaluation


def setup_run(*args, **kwargs):
    run = _setup_run(*args, **kwargs)
    run["args"].local_eval_protocol = evaluation.LEGACY_LOCAL_PROTOCOL
    return run


def _parser(defaults):
    def parse(argv):
        parser = argparse.ArgumentParser()
        for name, value in defaults.items():
            if isinstance(value, bool):
                parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=value)
            else:
                parser.add_argument(f"--{name}", type=type(value) if value is not None else str, default=value)
        return parser.parse_args(argv)

    return parse


def _command(args):
    command = ["python", "-m", "examples.test.eval.ace.evaluate"]
    for key, value in vars(args).items():
        if value is None:
            continue
        if isinstance(value, bool):
            command.append(f"--{'' if value else 'no-'}{key}")
        else:
            command.extend((f"--{key}", str(value)))
    return command


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def _publish_source(source, args):
    command = _command(args)
    metrics_path = source / "metrics_step0.json"
    _write_json(source / "evaluation_complete.json", {"evaluation": command, "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest()})
    _write_json(
        source / "evaluation_config.json",
        {
            "evaluation": command,
            "model": args.model_name,
            "step": 0,
            "metrics_file": str(metrics_path),
            "expected_epochs": args.local_eval_epochs,
            "start_epoch": 0,
        },
    )


async def _source_run(root, *, visits=3):
    recorder = Recorder()
    kwargs = setup_run(root, recorder)
    kwargs["args"].local_eval_epochs = visits
    assets = root / "prompts"
    assets.mkdir()
    (assets / "base_system_prompt.txt").write_text("Advice: {{playbook}}")
    initial = root / "initial_playbook.txt"
    initial.write_text("")
    kwargs["args"].prompts_dir = kwargs["prompts_dir"] = str(assets)
    kwargs["args"].initial_playbook = kwargs["initial_playbook_path"] = str(initial)
    source = Path(kwargs["playbook_snapshot_dir"])
    kwargs["metrics_file"] = str(source / "metrics_step0.json")
    result = await evaluation.run_local_ace(**kwargs)
    _publish_source(source, kwargs["args"])
    return source, kwargs, result


def _replay_kwargs(root, source, original_args, recorder):
    kwargs = setup_run(root, recorder)
    kwargs["args"] = SimpleNamespace(**vars(original_args), final_policy_eval=True, final_policy_from_run_dir=str(source))
    # setup_run's worker hook captures its original namespace; use a general one.
    kwargs["hooks"].prepare_local_worker_args = lambda args, index, count: SimpleNamespace(**vars(args), worker_index=index)
    kwargs["hooks"].parse_evaluation_args = _parser(vars(original_args))
    return kwargs


@pytest.mark.parametrize("visits", [1, 2, 3])
def test_final_policy_is_one_extra_actor_pass_after_exact_adaptation_count(tmp_path, visits):
    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder)
        kwargs["args"].local_eval_epochs = visits
        kwargs["args"].final_policy_eval = True
        result = await evaluation.run_local_ace(**kwargs)
        entry = result.evaluations[0]
        assert [item["local_eval_epoch"] for item in entry["local_epochs"]] == list(range(1, visits + 1))
        assert entry["local_eval_epochs"] == visits
        assert entry["metrics"]["avg_reward"] == pytest.approx((visits - 1) / 10)
        assert entry["metrics"]["final_policy/avg_reward"] == pytest.approx(visits / 10)
        final = entry["final_policy_evaluation"]
        assert final["policy_visit"] == visits + 1 and final["after_local_eval_epoch"] == visits
        assert final["adaptation_applied"] is False
        assert final["work"] == {"policy_rollouts": 4, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0}
        assert len(recorder.generator_calls) == 4 * (2 * visits + 1)
        assert len(recorder.reflector_calls) == len(recorder.curator_calls) == 4 * visits
        for task_id in ("alpha", "beta", "gamma", "delta"):
            calls = [tips for task, tips, _ in recorder.generator_calls if task == task_id]
            assert calls[-1] == visits
            final_path = tmp_path / f"artifacts/task_{task_id}/chat_completions/final_policy/policy_step000000_iter_{visits}.json"
            assert final_path.is_file()
        assert len(list(tmp_path.glob("artifacts/task_*/ace_logs/eval/*.json"))) == 4 * visits
        assert not list(tmp_path.glob("artifacts/task_*/ace_logs/final_policy/*.json"))

    asyncio.run(run())


def test_completed_source_replay_only_scores_final_playbooks_and_preserves_source(tmp_path):
    async def run():
        source, kwargs, old = await _source_run(tmp_path / "source")
        original_files = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
        recorder = Recorder()
        replay = await evaluation.run_local_ace(**_replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder))
        entry = replay.evaluations[0]
        assert entry["local_epochs"] == old.evaluations[0]["local_epochs"]
        assert entry["tasks"] == old.evaluations[0]["tasks"]
        assert entry["metrics"]["avg_reward"] == old.evaluations[0]["metrics"]["avg_reward"]
        assert entry["metrics"]["final_policy/avg_reward"] == pytest.approx(0.3)
        assert len(recorder.generator_calls) == 4
        assert all(tips == 3 for _, tips, _ in recorder.generator_calls)
        assert not recorder.reflector_calls and not recorder.curator_calls
        assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == original_files
        provenance = entry["final_policy_evaluation"]["provenance"]
        assert provenance["state_restoration"] == "final_playbook_text_for_inference_only"
        assert provenance["exact_rng_continuation"] is False
        assert len(provenance["source_artifact_hashes"]) == 4

    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["marker", "hash", "model", "protocol", "taskset", "playbook", "header", "log_task", "log_reward", "already_final", "epoch", "reference"])
def test_replay_rejects_incompatible_source_before_clients_or_generation(tmp_path, corruption):
    async def run():
        source, kwargs, _ = await _source_run(tmp_path / "source", visits=2)
        source_args = kwargs["args"]
        metrics_path = source / "metrics_step0.json"
        payload = json.loads(metrics_path.read_text())
        entry = payload["evaluations"][0]
        row = entry["tasks"][0]
        snapshot = Path(row["artifacts"]["playbook_snapshot"])
        log_path = Path(row["artifacts"]["ace_log"])
        if corruption == "marker":
            (source / "evaluation_complete.json").unlink()
        elif corruption == "hash":
            metrics_path.write_text(metrics_path.read_text() + "\n")
        elif corruption == "protocol":
            source_args = SimpleNamespace(**{**vars(source_args), "max_turns": 31})
        elif corruption == "playbook":
            snapshot.unlink()
        elif corruption == "header":
            snapshot.write_text(snapshot.read_text().replace("# Task ID: alpha", "# Task ID: wrong"))
        elif corruption in {"log_task", "log_reward"}:
            log = json.loads(log_path.read_text())
            if corruption == "log_task":
                log["task_id"] = "wrong"
            else:
                log["metrics"]["avg_reward"] = 99
            _write_json(log_path, log)
        else:
            if corruption == "model":
                payload["model_name"] = "another-model"
            elif corruption == "taskset":
                entry["local_epochs"][0]["tasks"].pop()
            elif corruption == "already_final":
                entry["final_policy_evaluation"] = {"kind": "final_policy_only"}
            elif corruption == "epoch":
                entry["local_epochs"][0]["local_eval_epoch"] = 0
            elif corruption == "reference":
                entry["tasks"][0]["artifacts"]["playbook_snapshot"] = str(tmp_path / "other.txt")
            _write_json(metrics_path, payload)
            _publish_source(source, kwargs["args"])
        recorder = Recorder()
        replay_kwargs = _replay_kwargs(tmp_path / "append", source, source_args, recorder)
        # Parse defaults must still reflect the original source protocol.
        replay_kwargs["hooks"].parse_evaluation_args = _parser(vars(kwargs["args"]))
        with pytest.raises(ValueError):
            await evaluation.run_local_ace(**replay_kwargs)
        assert not recorder.engines and not recorder.rules and not recorder.generator_calls
        assert not Path(replay_kwargs["metrics_file"]).exists()

    asyncio.run(run())


@pytest.mark.parametrize("overlap", ["same", "child", "parent"])
def test_replay_rejects_output_overlap(tmp_path, overlap):
    async def run():
        source, kwargs, _ = await _source_run(tmp_path / "source", visits=1)
        recorder = Recorder()
        replay_kwargs = _replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder)
        replay_kwargs["playbook_snapshot_dir"] = str({"same": source, "child": source / "final", "parent": source.parent}[overlap])
        with pytest.raises(ValueError, match="separate from the source"):
            await evaluation.run_local_ace(**replay_kwargs)
        assert not recorder.engines

    asyncio.run(run())


def test_final_pass_scores_previously_solved_tasks_and_keeps_fresh_reward_distinct(tmp_path):
    async def run():
        source, kwargs, _ = await _source_run(tmp_path / "source", visits=1)
        path = source / "metrics_step0.json"
        payload = json.loads(path.read_text())
        entry = payload["evaluations"][0]
        # Recorded acting success is authoritative for the cumulative score;
        # it must not suppress a fresh final-playbook measurement.
        for rows in (entry["tasks"], entry["local_epochs"][0]["tasks"]):
            rows[0]["reward"] = 1.0
        for metrics in (entry["metrics"], entry["local_epochs"][0]["metrics"]):
            metrics["avg_reward"] = 0.25
            metrics["max_reward_mean"] = 0.25
        log_path = Path(entry["tasks"][0]["artifacts"]["ace_log"])
        log = json.loads(log_path.read_text())
        log["metrics"]["avg_reward"] = 1.0
        _write_json(log_path, log)
        _write_json(path, payload)
        _publish_source(source, kwargs["args"])
        recorder = Recorder()
        result = await evaluation.run_local_ace(**_replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder))
        metrics = result.evaluations[0]["final_policy_evaluation"]["metrics"]
        assert len(recorder.generator_calls) == 4
        assert metrics["avg_reward"] == pytest.approx(0.1)
        assert metrics["max_reward_mean"] == pytest.approx((1.0 + 0.1 * 3) / 4)

    asyncio.run(run())


@pytest.mark.parametrize("changed", [False, True])
def test_replay_protocol_asset_moves_require_identical_current_content(tmp_path, changed):
    async def run():
        source, kwargs, _ = await _source_run(tmp_path / "source", visits=1)
        recorder = Recorder()
        replay_kwargs = _replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder)
        new_assets = tmp_path / "moved_prompts"
        new_assets.mkdir()
        (new_assets / "base_system_prompt.txt").write_text("Different" if changed else "Advice: {{playbook}}")
        replay_kwargs["args"].prompts_dir = str(new_assets)
        if changed:
            with pytest.raises(ValueError, match="protocol differs: prompts_dir"):
                await evaluation.run_local_ace(**replay_kwargs)
            assert not recorder.engines
        else:
            result = await evaluation.run_local_ace(**replay_kwargs)
            assert result.evaluations[0]["metrics"]["final_policy/avg_reward"] == pytest.approx(0.1)

    asyncio.run(run())


def test_replay_rejects_unrecorded_default_assets(tmp_path):
    async def run():
        source, kwargs, _ = await _source_run(tmp_path / "source", visits=1)
        kwargs["args"].prompts_dir = ""
        _publish_source(source, kwargs["args"])
        recorder = Recorder()
        with pytest.raises(ValueError, match="explicit source prompt"):
            await evaluation.run_local_ace(**_replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder))
        assert not recorder.engines

    asyncio.run(run())


@pytest.mark.parametrize("leaf,filename", [("chat_completions", "policy_step000000_iter_1.json"), ("playbooks", "playbook_step0000_iter_1.txt")])
def test_replay_rejects_partial_final_outputs_before_clients(tmp_path, leaf, filename):
    async def run():
        source, kwargs, _ = await _source_run(tmp_path / "source", visits=1)
        recorder = Recorder()
        replay_kwargs = _replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder)
        path = Path(replay_kwargs["playbook_snapshot_dir"]) / "task_alpha" / leaf / "final_policy" / filename
        path.parent.mkdir(parents=True)
        path.write_text("incomplete")
        with pytest.raises(ValueError, match="overwrite an existing final acting artifact"):
            await evaluation.run_local_ace(**replay_kwargs)
        assert not recorder.engines
        assert path.read_text() == "incomplete"

    asyncio.run(run())


def test_final_pass_failure_cancels_other_workers_and_preserves_source(tmp_path):
    async def run():
        source, kwargs, _ = await _source_run(tmp_path / "source", visits=1)
        files = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
        recorder = Recorder(failure="task")
        replay_kwargs = _replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder)
        with pytest.raises(RuntimeError, match="acting failed"):
            await evaluation.run_local_ace(**replay_kwargs)
        assert not recorder.active
        assert all(engine.shutdown_calls == 1 for engine in recorder.engines)
        assert not Path(replay_kwargs["metrics_file"]).exists()
        assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == files

    asyncio.run(run())


def test_frozen_solved_tasks_still_receive_fresh_final_policy_evaluation(tmp_path):
    class FirstVisitSuccessEngine(Engine):
        async def execute_tasks(self, tasks):
            trajectories = await super().execute_tasks(tasks)
            for trajectory in trajectories:
                task_id = trajectory.task["id"]
                reward = 1.0 if sum(task == task_id for task, _, _ in self.recorder.generator_calls) == 1 else 0.0
                trajectory.reward = trajectory.steps[0].reward = reward
            return trajectories

    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder, engine_class=FirstVisitSuccessEngine)
        kwargs["args"].freeze_solved = True
        kwargs["args"].final_policy_eval = True
        result = await evaluation.run_local_ace(**kwargs)
        entry = result.evaluations[0]
        assert [local["metrics"]["n_active_tasks"] for local in entry["local_epochs"]] == [4, 0, 0]
        assert len(entry["final_policy_evaluation"]["tasks"]) == 4
        assert entry["metrics"]["final_policy/avg_reward"] == 0.0
        assert entry["metrics"]["final_policy/max_reward_mean"] == 1.0
        assert len(recorder.reflector_calls) == len(recorder.curator_calls) == 4
        assert all(tips == 0 for _, tips, _ in recorder.generator_calls[-4:])

    asyncio.run(run())


def test_empty_final_actor_batch_cannot_publish_a_zero_score(tmp_path):
    class EmptyBatchEngine(Engine):
        async def execute_tasks(self, tasks):
            await super().execute_tasks(tasks)
            return []

    async def run():
        source, kwargs, _ = await _source_run(tmp_path / "source", visits=1)
        recorder = Recorder()
        replay_kwargs = _replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder)
        replay_kwargs["hooks"].build_generator_engine = lambda args: EmptyBatchEngine(recorder, args)
        with pytest.raises(RuntimeError, match="incomplete trajectory batch"):
            await evaluation.run_local_ace(**replay_kwargs)
        assert not Path(replay_kwargs["metrics_file"]).exists()
        assert all(engine.shutdown_calls == 1 for engine in recorder.engines)

    asyncio.run(run())


@pytest.mark.parametrize("pre_receipt_artifacts", [False, True])
def test_final_policy_retry_skips_completed_final_task(tmp_path, pre_receipt_artifacts):
    class FailFinalBeta(Engine):
        async def execute_tasks(self, tasks):
            task_id = tasks[0]["id"]
            previous = sum(task == task_id for task, _, _ in self.recorder.generator_calls)
            if task_id == "beta" and previous == 3:
                raise RuntimeError("interrupted final pass")
            return await super().execute_tasks(tasks)

    tasks = [{"id": "alpha"}, {"id": "beta"}]
    first = setup_run(tmp_path / "partial", Recorder(), concurrency=1, tasks=tasks, engine_class=FailFinalBeta)
    first["args"].g_ref = first["args"].g_cur = 0
    first["args"].final_policy_eval = True
    first["args"].resume = True
    with pytest.raises(RuntimeError, match="interrupted final pass"):
        asyncio.run(evaluation.run_local_ace(**first))
    receipts = list((tmp_path / "partial/artifacts").glob("task_*/completed_final_policy/*.json"))
    assert len(receipts) == 1
    if pre_receipt_artifacts:
        receipts[0].unlink()

    recorder = Recorder()
    retry = setup_run(tmp_path / "partial", recorder, concurrency=1, tasks=tasks)
    retry["args"].g_ref = retry["args"].g_cur = 0
    retry["args"].final_policy_eval = True
    retry["args"].resume = True
    resumed = asyncio.run(evaluation.run_local_ace(**retry))
    assert [task for task, _, _ in recorder.generator_calls] == ["beta"]

    fresh = setup_run(tmp_path / "fresh", Recorder(), concurrency=1, tasks=tasks)
    fresh["args"].g_ref = fresh["args"].g_cur = 0
    fresh["args"].final_policy_eval = True
    expected = asyncio.run(evaluation.run_local_ace(**fresh))
    assert resumed.evaluations[0]["metrics"] == expected.evaluations[0]["metrics"]


def test_warm_start_final_pass_declares_tasks_missing_from_all_resumed_adaptation_rows(tmp_path):
    class SolvedSourceEngine(Engine):
        async def execute_tasks(self, tasks):
            trajectories = await super().execute_tasks(tasks)
            for trajectory in trajectories:
                trajectory.reward = trajectory.steps[0].reward = 1.0
            return trajectories

    async def run():
        source_kwargs = setup_run(tmp_path / "source", Recorder(), engine_class=SolvedSourceEngine)
        source_kwargs["args"].local_eval_epochs = 1
        await evaluation.run_local_ace(**source_kwargs)
        recorder = Recorder()
        kwargs = setup_run(tmp_path / "extension", recorder)
        kwargs["args"].local_eval_epochs = 2
        kwargs["args"].start_epoch = 1
        kwargs["args"].warm_start_from_run_dir = source_kwargs["playbook_snapshot_dir"]
        kwargs["args"].freeze_solved = True
        kwargs["args"].final_policy_eval = True
        result = await evaluation.run_local_ace(**kwargs)
        entry = result.evaluations[0]
        expected_ids = ["alpha", "beta", "gamma", "delta"]
        assert entry["evaluation_task_ids"] == expected_ids
        assert [local["local_eval_epoch"] for local in entry["local_epochs"]] == [2, 3]
        assert all(local["tasks"] == [] for local in entry["local_epochs"])
        assert entry["tasks"] == []
        assert [row["task_id"] for row in entry["final_policy_evaluation"]["tasks"]] == expected_ids
        assert entry["final_policy_evaluation"]["policy_visit"] == 4
        assert entry["metrics"]["final_policy/avg_reward"] == pytest.approx(0.1)
        assert entry["metrics"]["final_policy/max_reward_mean"] == 1.0
        assert len(recorder.generator_calls) == 4
        assert not recorder.reflector_calls and not recorder.curator_calls

    asyncio.run(run())


async def _legacy_receipt_replay_case(tmp_path, *, concurrency=1, nested=False):
    source, kwargs, _ = await _source_run(tmp_path / "source", visits=1)
    frozen = tmp_path / "frozen_source"
    prompts = frozen / "examples/appworld/train/ace_rl/prompts"
    prompts.mkdir(parents=True)
    (prompts / "base_system_prompt.txt").write_bytes((Path(kwargs["args"].prompts_dir) / "base_system_prompt.txt").read_bytes())
    (prompts / "reflector_prompt.txt").write_text("Retained reflector prompt")
    (prompts / "curator_prompt.txt").write_text("Retained curator prompt")
    initial = frozen / "examples/appworld/train/ace_rl/playbooks/initial_playbook.txt"
    initial.parent.mkdir(parents=True)
    initial.write_bytes(Path(kwargs["args"].initial_playbook).read_bytes())
    repo = Path(__file__).resolve().parents[2]
    for relative in ("examples/appworld/eval/ace/evaluate.py", "examples/appworld/eval/ace/appworld_ace_utils.py"):
        path = frozen / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo / relative).read_bytes())
    metrics_path = source / "metrics_step0.json"
    payload = json.loads(metrics_path.read_text())
    payload["experiment_name"] = "appworld_local_ace"
    _write_json(metrics_path, payload)
    omitted = {"prompts_dir", "initial_playbook", "task_concurrency"}
    recorded = SimpleNamespace(**{name: value for name, value in vars(kwargs["args"]).items() if name not in omitted})
    if concurrency != 1:
        recorded.task_concurrency = concurrency
    command = _command(recorded)
    command[2] = "examples.appworld.eval.ace.evaluate"
    config = json.loads((source / "evaluation_config.json").read_text())
    config["evaluation"] = command
    _write_json(source / "evaluation_config.json", config)
    _write_json(source / "evaluation_complete.json", {"evaluation": command, "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest()})
    deployment_path = tmp_path / "retained-deployment.json"
    _write_json(
        deployment_path,
        {
            "remote_root": str(frozen),
            "source_files": {str(path.relative_to(frozen)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in frozen.rglob("*") if path.is_file()},
        },
    )
    launch_path = tmp_path / "retained-launch.json"
    entry = {"run_dir": str(source), "model": kwargs["args"].model_name, "step": 0, "method": "ace", "env": {}}
    launch = {"common_env": {"RLLM_DIR": str(frozen)}}
    if nested:
        launch["entries"] = [{"env": {"TASK_CONCURRENCY": str(concurrency)}, "evaluations": [entry]}]
        pointer = ["entries", 0, "evaluations", 0]
    else:
        launch["common_env"]["TASK_CONCURRENCY"] = str(concurrency)
        launch["base_evaluations"] = [entry]
        pointer = ["base_evaluations", 0]
    _write_json(launch_path, launch)
    receipt_path = tmp_path / "source-receipt.json"
    receipt = {
        "version": 1,
        "source_run_dir": str(source),
        "source_root": str(frozen),
        "source_configuration_sha256": hashlib.sha256((source / "evaluation_config.json").read_bytes()).hexdigest(),
        "source_completion_sha256": hashlib.sha256((source / "evaluation_complete.json").read_bytes()).hexdigest(),
        "source_metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "deployment_manifest": {"path": str(deployment_path), "sha256": hashlib.sha256(deployment_path.read_bytes()).hexdigest()},
        "launch_manifest": {"path": str(launch_path), "sha256": hashlib.sha256(launch_path.read_bytes()).hexdigest(), "entry_path": pointer},
    }
    _write_json(receipt_path, receipt)
    recorder = Recorder()
    replay = _replay_kwargs(tmp_path / "append", source, kwargs["args"], recorder)
    replay["args"].prompts_dir = replay["prompts_dir"] = str(prompts)
    replay["args"].initial_playbook = replay["initial_playbook_path"] = str(initial)
    replay["args"].final_policy_source_manifest = str(receipt_path)
    replay["hooks"].experiment_name = "appworld_local_ace"
    replay["hooks"].parse_evaluation_args = _parser({**vars(kwargs["args"]), "prompts_dir": "", "initial_playbook": "", "task_concurrency": 1})
    return source, replay, recorder, receipt_path


@pytest.mark.parametrize("concurrency,nested", [(1, False), (4, True)])
def test_retained_receipt_resolves_historical_defaults_without_editing_old_records(tmp_path, concurrency, nested):
    async def run():
        source, kwargs, recorder, receipt_path = await _legacy_receipt_replay_case(tmp_path, concurrency=concurrency, nested=nested)
        old_files = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
        result = await evaluation.run_local_ace(**kwargs)
        final = result.evaluations[0]["final_policy_evaluation"]
        assert final["metrics"]["avg_reward"] == pytest.approx(0.1)
        assert len(recorder.generator_calls) == 4
        assert not recorder.reflector_calls and not recorder.curator_calls
        assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == old_files
        provenance = final["provenance"]["source_asset_receipt"]
        assert provenance["receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        assert provenance["recorded_task_concurrency"] == concurrency
        assert provenance["resolved_default_assets"] == {"prompts_dir": kwargs["args"].prompts_dir, "initial_playbook": kwargs["args"].initial_playbook}
        assert len(provenance["verified_source_files"]) == 6
        assert final["provenance"]["default_asset_hashes_recorded_by_legacy_source"] is False
        assert final["provenance"]["exact_rng_continuation"] is False

    asyncio.run(run())


@pytest.mark.parametrize(
    "corruption",
    [
        "version",
        "source_run_dir",
        "source_configuration_sha256",
        "source_completion_sha256",
        "source_metrics_sha256",
        "deployment_hash",
        "launch_hash",
        "source_root",
        "deployment_root",
        "launch_root",
        "launch_run",
        "launch_model",
        "launch_step",
        "launch_method",
        "launch_concurrency",
        "launch_pointer",
        "missing_asset_proof",
        "changed_asset",
        "changed_resolver",
        "unsupported_resolver",
        "current_asset_mismatch",
        "missing_receipt",
        "missing_current_assets",
        "unsupported_benchmark",
    ],
)
def test_legacy_receipt_rejects_missing_or_mismatched_proof_before_clients(tmp_path, corruption):
    async def run():
        source, kwargs, recorder, receipt_path = await _legacy_receipt_replay_case(tmp_path)
        receipt = json.loads(receipt_path.read_text())
        deployment_path = Path(receipt["deployment_manifest"]["path"])
        launch_path = Path(receipt["launch_manifest"]["path"])
        deployment, launch = json.loads(deployment_path.read_text()), json.loads(launch_path.read_text())
        asset_relative = "examples/appworld/train/ace_rl/prompts/base_system_prompt.txt"
        resolver_relative = "examples/appworld/eval/ace/evaluate.py"
        if corruption in {"version", "source_run_dir", "source_root", "source_configuration_sha256", "source_completion_sha256", "source_metrics_sha256"}:
            receipt[corruption] = 2 if corruption == "version" else str(tmp_path / "incorrect")
        elif corruption == "deployment_hash":
            receipt["deployment_manifest"]["sha256"] = "incorrect"
        elif corruption == "launch_hash":
            receipt["launch_manifest"]["sha256"] = "incorrect"
        elif corruption == "launch_pointer":
            receipt["launch_manifest"]["entry_path"] = ["base_evaluations", 99]
        elif corruption == "deployment_root":
            deployment["remote_root"] = str(tmp_path / "incorrect")
        elif corruption == "launch_root":
            launch["common_env"]["RLLM_DIR"] = str(tmp_path / "incorrect")
        elif corruption == "launch_concurrency":
            launch["common_env"]["TASK_CONCURRENCY"] = "4"
        elif corruption.startswith("launch_"):
            key = corruption.removeprefix("launch_")
            launch["base_evaluations"][0]["run_dir" if key == "run" else key] = 9 if key == "step" else str(tmp_path / "incorrect")
        elif corruption == "missing_asset_proof":
            del deployment["source_files"][asset_relative]
        elif corruption == "changed_asset":
            (Path(receipt["source_root"]) / asset_relative).write_text("Changed historical prompt")
        elif corruption in {"changed_resolver", "unsupported_resolver"}:
            path = Path(receipt["source_root"]) / resolver_relative
            path.write_text(path.read_text().replace('TRAIN_ARTIFACTS_DIR / "prompts"', 'TRAIN_ARTIFACTS_DIR / "other_prompts"'))
            if corruption == "unsupported_resolver":
                deployment["source_files"][resolver_relative]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif corruption == "current_asset_mismatch":
            initial = tmp_path / "current-initial.txt"
            initial.write_text("Different current playbook")
            kwargs["args"].initial_playbook = str(initial)
        elif corruption == "missing_receipt":
            kwargs["args"].final_policy_source_manifest = None
        elif corruption == "missing_current_assets":
            kwargs["args"].prompts_dir = ""
        elif corruption == "unsupported_benchmark":
            kwargs["hooks"].experiment_name = "another_benchmark"
        if corruption in {"deployment_root", "missing_asset_proof", "unsupported_resolver"}:
            _write_json(deployment_path, deployment)
            receipt["deployment_manifest"]["sha256"] = hashlib.sha256(deployment_path.read_bytes()).hexdigest()
        if corruption.startswith("launch_") and corruption not in {"launch_hash", "launch_pointer"}:
            _write_json(launch_path, launch)
            receipt["launch_manifest"]["sha256"] = hashlib.sha256(launch_path.read_bytes()).hexdigest()
        _write_json(receipt_path, receipt)
        old_files = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
        with pytest.raises(ValueError):
            await evaluation.run_local_ace(**kwargs)
        assert not recorder.engines and not recorder.rules and not recorder.generator_calls
        assert not Path(kwargs["metrics_file"]).exists()
        assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == old_files

    asyncio.run(run())


def test_final_policy_receipt_requires_a_completed_source_run(tmp_path):
    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder)
        kwargs["args"].final_policy_source_manifest = str(tmp_path / "receipt.json")
        with pytest.raises(ValueError, match="requires final_policy_from_run_dir"):
            await evaluation.run_local_ace(**kwargs)
        assert not recorder.engines

    asyncio.run(run())
