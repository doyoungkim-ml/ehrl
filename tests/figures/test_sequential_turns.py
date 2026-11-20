"""RSI turn plots pad prior success across visits for every evaluation protocol."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/figures"))
import plot_turns  # noqa: E402


def fixture(root, protocol=None, *, protocol_location="entry"):
    source = root / "metrics_step10.json"
    rewards = [1.0, 0.0, 0.3]
    epochs = []
    policy_dir = root / "task_a/chat_completions/eval"
    policy_dir.mkdir(parents=True)
    sequential = protocol == "sequential_adaptation_v1"
    for visit, reward in enumerate(rewards, 1):
        epochs.append({
            "local_eval_epoch": visit,
            "tasks": [{"task_id": "a", "reward": reward}],
            "metrics": {"max_reward_mean": reward if sequential else 1.0},
        })
        trajectory = {"step": 10, "trajectories": [{"steps": [{"reward": reward}]}]}
        (policy_dir / f"policy_step000010_iter_{visit - 1}.json").write_text(json.dumps(trajectory))
    entry = {"step": 10, "local_eval_epochs": 3, "local_epochs": epochs, "evaluation_task_ids": ["a"]}
    payload = {"evaluations": [entry]}
    if protocol is not None:
        (entry if protocol_location == "entry" else payload)["local_eval_protocol"] = protocol
    source.write_text(json.dumps(payload))
    return {"source": str(source), "step": 10, "run_tag": "test", "variant": "ace/visits8", "task_label": "all"}


@pytest.mark.parametrize("protocol_location", ["entry", "payload"])
def test_sequential_turns_pad_success_by_default_and_allow_raw_inspection(tmp_path, protocol_location):
    point = fixture(tmp_path, "sequential_adaptation_v1", protocol_location=protocol_location)
    rows = plot_turns.curve_rows(point, max_turns=2)
    assert [row["value"] for row in rows] == [1.0] * 6
    assert [row["turn_global"] for row in rows] == list(range(1, 7))
    assert all(json.loads(row["trajectory_sources"]) == [] for row in rows if row["local_epoch"] > 1)
    raw = plot_turns.curve_rows(point, max_turns=2, pad_solved=False)
    assert [row["value"] for row in raw] == [1.0, 1.0, 0.0, 0.0, 0.3, 0.3]
    for row in raw:
        sources = json.loads(row["trajectory_sources"])
        assert len(sources) == 1
        assert sources[0].endswith(f"_iter_{row['local_epoch'] - 1}.json")
    iterations = plot_turns.curve_rows(point, axis="iteration")
    assert [row["value"] for row in iterations] == [1.0, 0.0, 0.3]


@pytest.mark.parametrize("protocol", [None, "legacy_search_v1"])
def test_legacy_turns_keep_historical_solved_padding(tmp_path, protocol):
    point = fixture(tmp_path, protocol)
    rows = plot_turns.curve_rows(point, max_turns=2)
    assert [row["value"] for row in rows] == [1.0] * 6
    assert all(json.loads(row["trajectory_sources"]) == [] for row in rows if row["local_epoch"] > 1)


def test_sequential_raw_inspection_requires_post_success_task_records(tmp_path):
    point = fixture(tmp_path, "sequential_adaptation_v1")
    source = Path(point["source"])
    payload = json.loads(source.read_text())
    payload["evaluations"][0]["local_epochs"][1]["tasks"] = []
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing unsolved tasks"):
        plot_turns.curve_rows(point, max_turns=2, pad_solved=False)
