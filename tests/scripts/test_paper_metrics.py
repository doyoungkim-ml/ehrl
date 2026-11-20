"""The paper's pass@K and RSI@K scoring on synthetic metrics files."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import paper_metrics  # noqa: E402


def sequential(path, rewards_by_task):
    epochs = [{"local_eval_epoch": visit + 1, "tasks": [{"task_id": task, "samples": [{"reward": rewards[visit]}]} for task, rewards in rewards_by_task.items()]} for visit in range(8)]
    path.write_text(json.dumps({"evaluations": [{"step": 10, "local_epochs": epochs}]}))
    return path


def independent(path, rewards_by_task, flat=False):
    tasks = [{"task_id": task, "samples": [{"reward": reward} for reward in rewards]} for task, rewards in rewards_by_task.items()]
    payload = {"tasks": tasks} if flat else {"evaluations": [{"step": 10, "local_epochs": [{"local_eval_epoch": 1, "tasks": tasks}]}]}
    path.write_text(json.dumps(payload))
    return path


def test_expected_max_matches_enumeration_without_replacement():
    from itertools import combinations

    rewards = [0.0, 0.25, 0.5, 1.0, 0.0, 0.75, 0.0, 0.5]
    for k in range(1, 9):
        brute = [max(group) for group in combinations(rewards, k)]
        assert paper_metrics.expected_max(rewards, k) == pytest.approx(sum(brute) / len(brute))


def test_sequential_visits_score_the_best_of_the_first_k(tmp_path):
    path = sequential(tmp_path / "ace.json", {"a": [0, 0, 1, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 0, 0, 0, 1]})
    result = paper_metrics.score(path, [2, 3, 8], independent=False)
    assert result["metric"] == "RSI"
    assert result["RSI@2"] == (0.0, 0.0)
    assert result["RSI@3"] == pytest.approx((0.5, 0.5))
    assert result["RSI@8"] == (1.0, 0.0)


def test_discarded_edits_are_scored_as_independent_attempts(tmp_path):
    path = sequential(tmp_path / "discard.json", {"a": [0, 0, 1, 0, 0, 0, 0, 0]})
    result = paper_metrics.score(path, [3], independent=True)
    assert result["metric"] == "pass"
    assert result["pass@3"][0] == pytest.approx(3 / 8)


@pytest.mark.parametrize("flat", [False, True])
def test_vanilla_files_report_mean_and_task_sem(tmp_path, flat):
    path = independent(tmp_path / "vanilla.json", {"a": [1] + [0] * 7, "b": [0] * 8, "c": [1] * 8}, flat=flat)
    result = paper_metrics.score(path, [1], independent=False)
    values = [1 / 8, 0.0, 1.0]
    mean = sum(values) / 3
    sem = math.sqrt(sum((v - mean) ** 2 for v in values) / 2 / 3)
    assert result["pass@1"] == pytest.approx((mean, sem))


def test_incomplete_attempts_are_rejected(tmp_path):
    path = independent(tmp_path / "short.json", {"a": [1, 0, 0]})
    with pytest.raises(ValueError, match="eight attempts"):
        paper_metrics.score(path, [1], independent=False)


def test_command_line_prints_each_requested_k(tmp_path, capsys):
    path = sequential(tmp_path / "ace.json", {"a": [0, 1, 0, 0, 0, 0, 0, 0]})
    assert paper_metrics.main(["--k", "1,2", str(path)]) == 0
    assert "RSI@1=0.000±0.000  RSI@2=1.000±0.000" in capsys.readouterr().out


def test_training_compute_accumulates_actor_and_critic_updates(tmp_path):
    log = tmp_path / "metrics.jsonl"
    records = [
        {"training/global_step": 1, "perf/flops/actor_update": 2e18},
        {"training/global_step": 2, "perf/mfu/actor": 0.5, "timing_s/update_actor": 1e6, "perf/mfu/critic": 0.25, "timing_s/update_critic": 1e6},
        {"training/global_step": 2, "perf/flops/actor_update": 1e18},  # a resumed step keeps its last record
        {"training/global_step": 3, "actor/entropy": 0.1},  # no update FLOPs logged
    ]
    log.write_text("\n".join(json.dumps(record) for record in records) + "\nnot json\n")
    assert paper_metrics.training_eflops(log) == pytest.approx({1: 2.0, 2: 3.0})
