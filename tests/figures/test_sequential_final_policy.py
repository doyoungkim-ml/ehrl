"""New final-policy summaries reuse the last measured visit when plotting."""

import hashlib
import json

import pytest
import test_shared_comparison as fixtures


@pytest.fixture
def source():
    fixture = fixtures.SharedComparisonTest()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def test_plot_reads_last_measured_policy_and_rejects_extra_visit(source):
    config_path, metrics_path, _ = source.configuration("appworld", epochs=3)
    config = json.loads(config_path.read_text())
    config["series"] = config["series"][:1]
    config["require_completion"] = True
    config_path.write_text(json.dumps(config))
    payload = json.loads(metrics_path.read_text())
    entry = payload["evaluations"][0]
    rows = entry["local_epochs"][-1]["tasks"]
    entry["local_eval_protocol"] = "sequential_adaptation_v1"
    entry["metrics"]["final_policy/avg_reward"] = 0.2
    entry["final_policy_evaluation"] = {
        "kind": "last_measured_policy_visit", "adaptation_applied": False,
        "policy_visit": 3, "after_local_eval_epoch": 2,
        "tasks": rows, "metrics": {"avg_reward": 0.2},
        "provenance": {"extra_policy_rollouts": 0},
        "work": {"policy_rollouts": 2, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0},
    }
    record_path = metrics_path.parent / "evaluation_config.json"
    recorded = json.loads(record_path.read_text())
    recorded["evaluation"] += ["--local_eval_protocol", "sequential_adaptation_v1"]
    record_path.write_text(json.dumps(recorded))

    def save():
        metrics_path.write_text(json.dumps(payload))
        (metrics_path.parent / "evaluation_complete.json").write_text(json.dumps({
            "evaluation": recorded["evaluation"], "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest()}))

    save()
    _, output = fixtures.comparison.load_comparison(config_path, metric="final_policy/avg_reward", local_epoch=3)
    assert output[0]["value"] == 0.2
    entry["final_policy_evaluation"]["policy_visit"] = 4
    save()
    with pytest.raises(ValueError, match="final policy visit"):
        fixtures.comparison.load_comparison(config_path, metric="final_policy/avg_reward", local_epoch=3)
