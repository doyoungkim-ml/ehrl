"""Launcher completion validation against files written by real evaluation flows."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_ace_evaluation_options import CandidateGeneratorEngine, CandidateRoleEngine, make_rule

from rllm.algorithms.ace import evaluation
from scripts.evaluation_runtime import validate_metrics


def produce_metrics(directory, *, mode="local", source=None, start_epoch=0, local_eval_epochs=2, protocol=evaluation.SEQUENTIAL_LOCAL_PROTOCOL):
    inputs = directory / "inputs"
    inputs.mkdir(parents=True)
    rule, _ = make_rule(inputs, [])
    roles = CandidateRoleEngine()
    rule.reflector.rollout_engine = roles
    rule.curator.rollout_engine = roles
    generator = CandidateGeneratorEngine()
    (inputs / "base_system_prompt.txt").write_text("Advice: {{playbook}}")
    hooks = evaluation.ACEExperimentHooks(
        experiment_name="producer-validator-test",
        task_id_getter=lambda task: task["id"],
        build_generator_engine=lambda args: generator,
        create_context_rule=lambda *args, **kwargs: rule,
        trajectory_reward=lambda trajectory: trajectory.reward,
        trajectory_success=lambda trajectory: trajectory.reward >= 1,
    )
    args = SimpleNamespace(
        model_name="test-model",
        local_eval_protocol=protocol,
        train_batch_size=1,
        val_batch_size=None,
        test_batch_size=None,
        epochs=1,
        local_eval_epochs=local_eval_epochs,
        g_act=1,
        g_ref=1,
        g_cur=1,
        selection_mode="best",
        test_epoch_freq=1,
        val_before_train=False,
        max_turns=2,
        wandb_step=7,
        warm_start_from_run_dir=source,
        start_epoch=start_epoch,
    )
    tasks = [{"id": "task", "instruction": "Fix the issue"}]
    run_dir = directory / "run"
    step = 7 if mode == "local" else 1
    metrics_file = run_dir / f"metrics_step{step}.json"
    common = dict(
        args=args,
        hooks=hooks,
        prompts_dir=str(inputs),
        initial_playbook_path=str(inputs / "playbook.txt"),
        playbook_snapshot_dir=str(run_dir),
        ace_log_dir=str(run_dir / "ace_logs"),
        metrics_file=str(metrics_file),
        script_path=__file__,
    )
    if mode == "local":
        result = asyncio.run(evaluation.run_local_ace(**common, train_tasks=[], eval_tasks=tasks))
        assert isinstance(result, evaluation.LocalRunMetrics)
    else:
        result = asyncio.run(evaluation.run_global_ace(**common, train_tasks=tasks, eval_tasks=tasks, chat_completion_dir=None))
        assert isinstance(result, evaluation.GlobalRunMetrics)
    item = {
        "metrics_file": str(metrics_file), "run_dir": str(run_dir), "model": args.model_name,
        "step": step, "expected_epochs": args.local_eval_epochs,
        "local_eval_protocol": protocol, "g_act": args.g_act,
    }
    return result, item, generator


def test_launcher_accepts_real_saved_local_evaluation_with_one_based_epochs(tmp_path):
    result, item, _ = produce_metrics(tmp_path)
    path = tmp_path / "run/metrics_step7.json"
    saved = json.loads(path.read_text())
    assert saved["evaluations"] == result.evaluations
    assert [epoch["local_eval_epoch"] for epoch in saved["evaluations"][0]["local_epochs"]] == [1, 2]
    assert validate_metrics(item) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_rejects_real_warm_started_extension_metrics(tmp_path):
    # Launchers run complete measured chains only; an evaluator-level extension
    # (epochs 4..6 of a saved predecessor) never completes a launcher output.
    _, source_item, _ = produce_metrics(tmp_path / "source", local_eval_epochs=3, protocol=evaluation.LEGACY_LOCAL_PROTOCOL)
    assert validate_metrics(source_item)
    _, item, _ = produce_metrics(tmp_path / "extension", source=source_item["run_dir"], start_epoch=3, local_eval_epochs=3, protocol=evaluation.LEGACY_LOCAL_PROTOCOL)
    saved = json.loads((tmp_path / "extension/run/metrics_step7.json").read_text())
    assert [epoch["local_eval_epoch"] for epoch in saved["evaluations"][0]["local_epochs"]] == [4, 5, 6]
    with pytest.raises(ValueError, match="epoch count"):
        validate_metrics(item)


def test_local_launcher_rejects_real_global_metrics_instead_of_marking_them_complete(tmp_path):
    result, item, _ = produce_metrics(tmp_path, mode="global")
    assert len(result.evaluations) == 1
    assert result.evaluations[0]["step"] == item["step"]
    assert "local_epochs" not in result.evaluations[0]
    with pytest.raises(ValueError, match="epoch count"):
        validate_metrics(item)


def test_shell_completion_uses_recorded_command_and_validates_before_writing(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from scripts.evaluation_runtime import recorded_evaluation_item, write_completion

    _, item, _ = produce_metrics(tmp_path)
    output = Path(item["run_dir"])
    command = [sys.executable, "-m", "fixture", "--model_name", item["model"], "--wandb_step", "7",
               "--g_act", "1", "--local_eval_epochs", "2", "--metrics_file", item["metrics_file"]]
    saved = {"benchmark": "mini_swe_bench", "method": "ace", "model_name": item["model"], "step": 7,
             "run_dir": str(output), "local_eval_protocol": "sequential_adaptation_v1", "evaluation": command}
    config = output / "evaluation_config.json"
    config.write_text(json.dumps(saved))
    runtime = Path(__file__).resolve().parents[2] / "scripts/evaluation_runtime.py"
    completed = subprocess.run([sys.executable, str(runtime), "mini_swe_bench", "ace", "--complete-recorded", str(config)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    marker = output / "evaluation_complete.json"
    receipt = marker.read_bytes()
    assert json.loads(receipt)["metrics_sha256"] == hashlib.sha256(Path(item["metrics_file"]).read_bytes()).hexdigest()
    write_completion(recorded_evaluation_item(config))
    assert marker.read_bytes() == receipt
    payload = json.loads(Path(item["metrics_file"]).read_text())
    payload["evaluations"][0]["local_epochs"][0]["tasks"][0]["samples"] = []
    Path(item["metrics_file"]).write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        write_completion(recorded_evaluation_item(config))
    assert marker.read_bytes() == receipt
    saved["step"] = 8
    config.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="identity"):
        recorded_evaluation_item(config)


@pytest.mark.parametrize("corruption", [None, "missing", "duplicate", "nonfinite", "summary", "step", "model"])
def test_flat_mastermind_completion_validates_real_summary(tmp_path, corruption):
    from pathlib import Path

    from examples.mastermind.eval.evaluate import summarize_task_rows
    from scripts.evaluation_runtime import write_completion

    samples = [{"task_id": "a", "attempt_id": i, "sample_index": i, "sample_id": f"a/attempt_{i}",
                "reward": float(i == 0), "success": i == 0} for i in range(8)]
    rows = [{"task_id": "a", "samples": samples, "reward": 0.12500000000000003, "success": True}]
    payload = {"model": "fixture", "step": 10, "evaluation_protocol": "independent_samples_v1", **summarize_task_rows(rows, 8)}
    if corruption == "missing":
        samples.pop()
    elif corruption == "duplicate":
        samples[-1]["sample_index"] = 0
    elif corruption == "nonfinite":
        samples[-1]["reward"] = float("nan")
    elif corruption == "summary":
        payload["pass_at_2_success_rate"] = 1
    elif corruption in {"step", "model"}:
        payload[corruption] = "wrong"
    path = tmp_path / "metrics_step10.json"
    path.write_text(json.dumps(payload))
    item = {"model": "fixture", "step": 10, "expected_epochs": 1, "g_act": 8, "local_eval_protocol": "independent_samples_v1",
            "metrics_file": str(path), "run_dir": str(tmp_path), "evaluation": ["python", "-m", "fixture"]}
    if corruption:
        with pytest.raises(ValueError):
            write_completion(item)
        assert not (tmp_path / "evaluation_complete.json").exists()
    else:
        write_completion(item)
        assert json.loads((Path(item["run_dir"]) / "evaluation_complete.json").read_text())["metrics_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
