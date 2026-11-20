"""The completion marker validates real sequential evaluator outputs."""

import asyncio
import copy
import json

import pytest
from test_local_evaluation_concurrency import Engine, Recorder, setup_run

from rllm.algorithms.ace import evaluation
from scripts.evaluation_runtime import validate_metrics


def measured(tmp_path):
    class SuccessThenFailure(Engine):
        async def execute_tasks(self, tasks):
            trajectories = await super().execute_tasks(tasks)
            reward = float(len(self.recorder.generator_calls) == 1)
            for trajectory in trajectories:
                trajectory.reward = trajectory.steps[0].reward = reward
            return trajectories

    kwargs = setup_run(tmp_path, Recorder(), concurrency=1, tasks=[{"id": "alpha"}], engine_class=SuccessThenFailure)
    result = asyncio.run(evaluation.run_local_ace(**kwargs))
    item = {"model": kwargs["args"].model_name, "step": 0, "expected_epochs": 3,
            "g_act": 1, "local_eval_protocol": "sequential_adaptation_v1", "metrics_file": kwargs["metrics_file"]}
    return item, result


def test_completion_accepts_raw_success_then_failure_and_reused_final_visit(tmp_path):
    item, result = measured(tmp_path)
    assert validate_metrics(item)
    entry = result.evaluations[0]
    assert entry["final_policy_evaluation"]["metrics"]["avg_reward"] == 0
    assert entry["final_policy_evaluation"]["metrics"]["cumulative_success_rate"] == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_sample", "repeat_attempt", "padded_reward", "extra_final", "wrong_final_rows",
        "extra_rollout", "wrong_protocol", "missing_task", "candidate_scoring", "final_adaptation",
    ],
)
def test_completion_refuses_incomplete_or_reinterpreted_sequential_results(tmp_path, corruption):
    item, _ = measured(tmp_path)
    path = tmp_path / "metrics.json"
    payload = json.loads(path.read_text())
    entry = payload["evaluations"][0]
    if corruption == "missing_sample":
        entry["local_epochs"][1]["tasks"][0]["samples"] = []
    elif corruption == "repeat_attempt":
        entry["local_epochs"][1]["tasks"][0]["samples"][0]["attempt_id"] = entry["local_epochs"][0]["tasks"][0]["samples"][0]["attempt_id"]
    elif corruption == "padded_reward":
        entry["local_epochs"][1]["tasks"][0]["reward"] = 1
    elif corruption == "extra_final":
        entry["final_policy_evaluation"]["policy_visit"] = 4
    elif corruption == "wrong_final_rows":
        entry["final_policy_evaluation"]["tasks"] = copy.deepcopy(entry["local_epochs"][0]["tasks"])
    elif corruption == "extra_rollout":
        entry["rollout_counts"]["policy"] += 1
    elif corruption == "candidate_scoring":
        entry["rollout_counts"]["candidate"] = 1
    elif corruption == "final_adaptation":
        entry["local_epochs"][-1]["tasks"][0]["adaptation_applied"] = True
        entry["final_policy_evaluation"]["tasks"][0]["adaptation_applied"] = True
    elif corruption == "wrong_protocol":
        entry["local_eval_protocol"] = "legacy_search_v1"
    else:
        entry["local_epochs"][1]["tasks"] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_metrics(item)
