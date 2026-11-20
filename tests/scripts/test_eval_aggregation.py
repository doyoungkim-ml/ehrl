"""Evaluation aggregation against synthetic disjoint chunks."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_script("merge_task_chunks")


def stage(task_ids, *, epoch=1, reward=0.0, padded=None, tokens=0):
    tasks = [{"task_id": str(task_id), "reward": reward} for task_id in task_ids]
    size = len(tasks)
    return {
        "local_eval_epoch": epoch,
        "tasks": tasks,
        "metrics": {
            "n_active_tasks": size,
            "num_trajectories": size,
            "success_count": size if reward >= 1 else 0,
            "success_rate": 1.0 if reward >= 1 and size else 0.0,
            "avg_reward": reward if size else 0.0,
            "max_reward_mean": reward if padded is None else padded,
            "rollout/total_completion_tokens": tokens,
            "ace/total_rollouts": size * 3,
        },
    }


def write_run(path, epochs, *, with_local_epochs=True):
    path.mkdir(parents=True)
    final = epochs[-1]
    evaluation = {"step": 7, "epoch": 0, "metrics": dict(final["metrics"]), "tasks": final["tasks"]}
    if with_local_epochs:
        evaluation.update(local_epochs=epochs, local_eval_epochs=len(epochs))
        evaluation["metrics"]["local_eval_epochs"] = len(epochs)
    payload = {"model_name": "test-model", "experiment_name": "test", "evaluations": [evaluation]}
    (path / "metrics_step7.json").write_text(json.dumps(payload))
    return path


def make_chunks(tmp_path):
    first = write_run(tmp_path / "run_step7_tasks0-0", [stage([0], reward=0, tokens=100)])
    second = write_run(tmp_path / "run_step7_tasks1-3", [stage([1, 2, 3], reward=1, tokens=900)])
    return first, second


def merged_result(tmp_path):
    return json.loads((tmp_path / "run_step7_tasksall/metrics_step7.json").read_text())["evaluations"][0]


def test_unequal_chunks_sum_counts_and_tokens_and_weight_means(tmp_path):
    chunks = make_chunks(tmp_path)
    merge.merge_step([str(path) for path in chunks], total=4)
    result = merged_result(tmp_path)
    assert result["metrics"]["success_count"] == 3
    assert result["metrics"]["num_trajectories"] == 4
    assert result["metrics"]["avg_reward"] == pytest.approx(0.75)
    assert result["metrics"]["max_reward_mean"] == pytest.approx(0.75)
    assert result["metrics"]["rollout/total_completion_tokens"] == 1000
    assert result["metrics"]["ace/total_rollouts"] == 12
    assert result["metrics"]["local_eval_epochs"] == 1
    assert result["local_epochs"][0]["metrics"]["rollout/total_completion_tokens"] == 1000
    assert [task["task_id"] for task in result["tasks"]] == ["0", "1", "2", "3"]


def test_frozen_chunk_keeps_all_task_denominator_for_padded_reward(tmp_path):
    first = write_run(tmp_path / "run_step7_tasks0-0", [stage([0], reward=1), stage([], epoch=2, padded=1)])
    second = write_run(tmp_path / "run_step7_tasks1-3", [stage([1, 2, 3], reward=0.5), stage([1, 2, 3], epoch=2, reward=0.2)])
    merge.merge_step([str(first), str(second)], total=4)
    result = merged_result(tmp_path)
    assert result["metrics"]["n_active_tasks"] == 3
    assert result["metrics"]["avg_reward"] == pytest.approx(0.2)
    assert result["metrics"]["max_reward_mean"] == pytest.approx(0.4)
    assert result["local_epochs"][0]["metrics"]["max_reward_mean"] == pytest.approx(0.625)


def test_global_evaluations_without_local_epochs_use_actual_chunk_sizes(tmp_path):
    first = write_run(tmp_path / "run_step7_tasks1", [stage([0], reward=0)], with_local_epochs=False)
    second = write_run(tmp_path / "run_step7_tasks1-3", [stage([1, 2, 3], reward=1)], with_local_epochs=False)
    for path in (first, second):
        payload = json.loads((path / "metrics_step7.json").read_text())
        del payload["evaluations"][0]["tasks"]
        (path / "metrics_step7.json").write_text(json.dumps(payload))
    merge.merge_step([str(first), str(second)], total=4)
    assert merged_result(tmp_path)["metrics"]["avg_reward"] == pytest.approx(0.75)


def test_means_use_their_observed_trajectory_population():
    stages = [stage([0], reward=0), stage([1, 2, 3], reward=1)]
    stages[0]["metrics"]["num_trajectories"] = 6
    stages[1]["metrics"]["num_trajectories"] = 3
    result = merge.merge_stage(stages, [1, 3], set())
    assert result["metrics"]["avg_reward"] == pytest.approx(1 / 3)
    assert result["metrics"]["max_reward_mean"] == pytest.approx(3 / 4)


def test_sparse_metrics_are_rejected_instead_of_treating_missing_values_as_zero():
    with pytest.raises(ValueError, match="missing from a nonempty chunk"):
        merge.merge_metric_dicts([({"avg_reward": 1}, 1), ({}, 3)])
    assert merge.merge_metric_dicts([({"avg_reward": 1}, 1), ({}, 0)]) == {"avg_reward": 1}


def test_turn_extrema_are_omitted_instead_of_averaged(tmp_path):
    chunks = make_chunks(tmp_path)
    for path in chunks:
        payload = json.loads((path / "metrics_step7.json").read_text())
        payload["evaluations"][0]["metrics"]["turn/best_avg_reward"] = 1
        (path / "metrics_step7.json").write_text(json.dumps(payload))
    merge.merge_step([str(path) for path in chunks], total=4)
    assert "turn/best_avg_reward" not in merged_result(tmp_path)["metrics"]
    provenance = json.loads((tmp_path / "run_step7_tasksall/merged_from.json").read_text())
    assert provenance["omitted_metric_keys"] == ["turn/best_avg_reward"]


@pytest.mark.parametrize("problem", ["task_id", "task_dir", "epoch", "model", "extension"])
def test_invalid_chunks_fail_before_publishing_metrics(tmp_path, problem):
    first, second = make_chunks(tmp_path)
    payload = json.loads((second / "metrics_step7.json").read_text())
    if problem == "task_id":
        payload["evaluations"][0]["tasks"][0]["task_id"] = "0"
    elif problem == "task_dir":
        (first / "task_collision").mkdir()
        (second / "task_collision").mkdir()
    elif problem == "epoch":
        payload["evaluations"][0]["local_epochs"][0]["local_eval_epoch"] = 2
    elif problem == "model":
        payload["model_name"] = "different checkpoint"
    else:
        (first / "extended_from.json").write_text(json.dumps({"extended_from": "source-a"}))
        (second / "extended_from.json").write_text(json.dumps({"extended_from": "source-b"}))
    (second / "metrics_step7.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        merge.merge_step([str(first), str(second)], total=4)
    assert not (tmp_path / "run_step7_tasksall").exists()


def test_dry_run_does_not_rename_partial_output_or_write_files(tmp_path):
    chunks = make_chunks(tmp_path)
    partial = tmp_path / "run_step7_tasksall"
    partial.mkdir()
    (partial / "saved.txt").write_text("keep")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert merge.merge_step([str(path) for path in chunks], total=4, dry_run=True).startswith("WOULD-MERGE")
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert (partial / "saved.txt").read_text() == "keep"


def test_success_preserves_partial_output_and_links_sources_without_copying(tmp_path):
    chunks = make_chunks(tmp_path)
    task = chunks[0] / "task_0"
    task.mkdir()
    (task / "saved.txt").write_text("source")
    partial = tmp_path / "run_step7_tasksall"
    partial.mkdir()
    (partial / "saved.txt").write_text("partial")
    merge.merge_step([str(path) for path in chunks], total=4)
    assert (partial / "task_0").is_symlink()
    assert (partial / "task_0/saved.txt").read_text() == "source"
    stale = list(tmp_path.glob("run_step7_tasksall_stale_*"))
    assert len(stale) == 1 and (stale[0] / "saved.txt").read_text() == "partial"
    assert merge.merge_step([str(path) for path in chunks], total=4).startswith("OK")


@pytest.mark.parametrize("ranges", [[(0, 0), (2, 3)], [(0, 2), (2, 3)]])
def test_incomplete_or_overlapping_coverage_does_not_create_output(tmp_path, ranges):
    chunks = [write_run(tmp_path / f"run_step7_tasks{first}-{last}", [stage(range(first, last + 1))]) for first, last in ranges]
    assert merge.merge_step([str(path) for path in chunks], total=4).startswith("SKIP")
    assert not (tmp_path / "run_step7_tasksall").exists()


def test_offline_script_help_runs_outside_repository_without_training_imports(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/merge_task_chunks.py"), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def write_sequential_run(path, task_ids, outcomes, *, nested_artifacts=False):
    """Two-sample measured chains with a final-visit alias and durable samples."""
    path.mkdir(parents=True)
    epochs, solved = [], set()
    count, samples_per_task = len(outcomes), 2
    applied_total = 0
    for visit, rewards_by_task in enumerate(outcomes, start=1):
        rows = []
        for task_id, rewards in zip(task_ids, rewards_by_task, strict=True):
            samples = [{"sample_index": index, "attempt_id": f"{task_id}:visit_{visit}:sample_{index}", "reward": reward, "success": reward == 1} for index, reward in enumerate(rewards)]
            if any(sample["success"] for sample in samples):
                solved.add(task_id)
            artifact_root = path / "playbooks" if nested_artifacts else path
            artifact = artifact_root / f"task_{task_id}/chat_completions/eval/policy_step000007_iter_{visit - 1}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(json.dumps({"step": 7, "num_trajectories": samples_per_task, "trajectories": samples}))
            applied = visit < count and task_id != "b"
            applied_total += applied
            rows.append({
                "task_id": task_id, "reward": sum(rewards) / samples_per_task, "samples": samples,
                "adaptation_applied": applied, "cumulative_solved": task_id in solved,
                "artifacts": {"policy_chat_completions": str(artifact)},
            })
        size = len(task_ids)
        fresh = sum(sample["success"] for row in rows for sample in row["samples"])
        any_solved = sum(any(sample["success"] for sample in row["samples"]) for row in rows)
        reflecting = visit < count
        metrics = {
            "n_active_tasks": size, "n_solved": len(solved), "num_trajectories": size * samples_per_task,
            "success_count": fresh, "success_rate": fresh / (size * samples_per_task),
            "avg_reward": sum(row["reward"] for row in rows) / size,
            "max_reward_mean": any_solved / size, "cumulative_success_rate": len(solved) / size,
            "pass_at_1_success_rate": fresh / (size * samples_per_task),
            "pass_at_2_success_rate": any_solved / size, "pass_at_k_success_rate": any_solved / size,
            "ace/policy_rollouts": size * samples_per_task,
            "ace/reflector_rollouts": size * samples_per_task if reflecting else 0,
            "ace/curator_rollouts": size if reflecting else 0, "ace/candidate_policy_rollouts": 0,
            "ace/total_rollouts": size * (2 * samples_per_task + 1 if reflecting else samples_per_task),
            "turn/best_avg_reward": any_solved / size,
        }
        epochs.append({"local_eval_epoch": visit, "tasks": rows, "metrics": metrics})
    last = epochs[-1]
    counts = {role: sum(epoch["metrics"][f"ace/{role}_rollouts"] for epoch in epochs) for role in ("policy", "reflector", "curator")}
    counts.update(candidate=0, total=sum(counts.values()))
    entry = {
        "step": 7, "epoch": 0, "local_eval_protocol": merge.SEQUENTIAL_PROTOCOL,
        "g_act": samples_per_task, "policy_visits": count, "local_eval_epochs": count,
        "adaptation_updates": count - 1, "adaptation_update_slots_per_task": count - 1,
        "actually_applied_context_updates": applied_total, "evaluation_task_ids": task_ids,
        "rollout_counts": counts, "tasks": last["tasks"], "local_epochs": epochs,
        "metrics": {**last["metrics"], "local_eval_epochs": count, **{f"final_policy/{key}": value for key, value in last["metrics"].items()}},
        "adaptation_evaluation": {"kind": "measured_adaptation_visits", "policy_visits": count - 1, "local_epochs": epochs[:-1]},
        "final_policy_evaluation": {
            "kind": "last_measured_policy_visit", "policy_visit": count, "after_local_eval_epoch": count - 1,
            "adaptation_applied": False, "tasks": last["tasks"], "metrics": last["metrics"],
            "work": {"policy_rollouts": size * samples_per_task, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0},
            "provenance": {"source": "same_run_measured_visit", "extra_policy_rollouts": 0},
        },
    }
    payload = {"model_name": "test-model", "experiment_name": "test", "local_eval_epochs": count, "local_eval_protocol": merge.SEQUENTIAL_PROTOCOL, "evaluations": [entry]}
    (path / "metrics_step7.json").write_text(json.dumps(payload))
    return path


def sequential_chunks(tmp_path):
    first = write_sequential_run(tmp_path / "run_step7_tasks0-0", ["a"], [[[1.0, 0.0]], [[0.0, 0.0]]])
    second = write_sequential_run(tmp_path / "run_step7_tasks1-2", ["b", "c"], [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]], nested_artifacts=True)
    return first, second


def test_sequential_chunks_rebuild_final_alias_and_retain_all_samples_after_regression(tmp_path):
    chunks = sequential_chunks(tmp_path)
    originals = {path: path.read_bytes() for chunk in chunks for path in chunk.rglob("*.json")}
    assert merge.merge_step(list(map(str, chunks)), total=3).startswith("MERGED")
    result = merged_result(tmp_path)
    assert result["local_eval_protocol"] == merge.SEQUENTIAL_PROTOCOL
    assert result["evaluation_task_ids"] == ["a", "b", "c"]
    assert result["rollout_counts"] == {"policy": 12, "reflector": 6, "curator": 3, "candidate": 0, "total": 21}
    assert result["actually_applied_context_updates"] == 2
    assert result["adaptation_update_slots_per_task"] == result["adaptation_updates"] == 1
    assert [epoch["tasks"][0]["reward"] for epoch in result["local_epochs"]] == [0.5, 0.0]
    assert [epoch["tasks"][0]["samples"][0]["success"] for epoch in result["local_epochs"]] == [True, False]
    assert result["metrics"]["pass_at_1_success_rate"] == pytest.approx(1 / 6)
    assert result["metrics"]["pass_at_2_success_rate"] == pytest.approx(1 / 3)
    assert result["metrics"]["avg_reward"] == pytest.approx(1 / 6)
    assert result["metrics"]["max_reward_mean"] == pytest.approx(1 / 3)
    assert result["metrics"]["cumulative_success_rate"] == pytest.approx(2 / 3)
    final = result["final_policy_evaluation"]
    assert final["tasks"] == result["tasks"] == result["local_epochs"][-1]["tasks"]
    assert final["metrics"] == result["local_epochs"][-1]["metrics"]
    assert final["work"] == {"policy_rollouts": 6, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0}
    assert final["provenance"]["extra_policy_rollouts"] == 0
    assert result["metrics"]["final_policy/num_trajectories"] == 6
    assert result["adaptation_evaluation"]["local_epochs"] == result["local_epochs"][:-1]
    attempts = [sample["attempt_id"] for epoch in result["local_epochs"] for row in epoch["tasks"] for sample in row["samples"]]
    assert len(attempts) == len(set(attempts)) == 12
    for epoch in result["local_epochs"]:
        for row in epoch["tasks"]:
            artifact = Path(row["artifacts"]["policy_chat_completions"])
            assert json.loads(artifact.read_text())["trajectories"] == row["samples"]
            linked = tmp_path / f"run_step7_tasksall/task_{row['task_id']}/chat_completions/eval/{artifact.name}"
            assert linked.read_bytes() == artifact.read_bytes()
    assert {path: path.read_bytes() for path in originals} == originals
    omitted = json.loads((tmp_path / "run_step7_tasksall/merged_from.json").read_text())["omitted_metric_keys"]
    assert omitted == ["final_policy/turn/best_avg_reward", "turn/best_avg_reward"]
    assert not any("turn/" in key for key in result["metrics"])
    runtime = load_script("evaluation_runtime")
    item = {
        "metrics_file": str(tmp_path / "run_step7_tasksall/metrics_step7.json"), "model": "test-model", "step": 7,
        "expected_epochs": 2, "g_act": 2, "local_eval_protocol": merge.SEQUENTIAL_PROTOCOL,
    }
    assert runtime.validate_metrics(item)


@pytest.mark.parametrize("problem", ["mixed_protocol", "g_act", "policy_visits", "task_ids", "missing_sample", "final_alias", "candidate_rollouts", "applied_updates"])
def test_sequential_chunk_contract_mismatches_fail_before_publishing(tmp_path, problem):
    chunks = sequential_chunks(tmp_path)
    path = chunks[-1] / "metrics_step7.json"
    payload = json.loads(path.read_text())
    entry = payload["evaluations"][0]
    if problem == "mixed_protocol":
        payload["local_eval_protocol"] = entry["local_eval_protocol"] = merge.LEGACY_PROTOCOL
    elif problem == "g_act":
        entry["g_act"] = 1
    elif problem == "policy_visits":
        entry["policy_visits"] = 3
    elif problem == "task_ids":
        entry["evaluation_task_ids"][0] = "a"
    elif problem == "missing_sample":
        entry["local_epochs"][0]["tasks"][0]["samples"].pop()
    elif problem == "final_alias":
        entry["final_policy_evaluation"]["tasks"].pop()
    elif problem == "candidate_rollouts":
        entry["rollout_counts"].update(candidate=1, total=15)
    else:
        entry["actually_applied_context_updates"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        merge.merge_step(list(map(str, chunks)), total=3)
    assert not (tmp_path / "run_step7_tasksall").exists()
