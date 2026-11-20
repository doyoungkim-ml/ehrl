import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rllm.algorithms.ace.trainer import LocalACERLAgentPPOTrainer
from rllm.trainer.verl import agent_ppo_trainer as trainer_mod


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def test_train_pass_at_k_uses_explicit_step_reward_task_groups(monkeypatch):
    """Extra ACE policy rows must be grouped with their source task for pass@k."""

    captured = {}

    def fake_turn_metrics(step_rewards, **kwargs):
        captured["step_rewards"] = step_rewards
        captured["table_extra_values"] = kwargs["table_extra_values"]
        return {}

    monkeypatch.setattr(trainer_mod, "compute_turn_level_metrics", fake_turn_metrics)

    trainer = SimpleNamespace(
        _current_step_rewards_list=[[0.1], [0.9], [0.2], [0.8]],
        _current_step_reward_tasks=["task-a", "task-a", "task-b", "task-b"],
        config=SimpleNamespace(
            algorithm=AttrDict(gamma=1.0),
            rllm=AttrDict(turn_level_cutoff_mode="max", agent=AttrDict(max_steps=1)),
        ),
        global_steps=3,
    )

    metrics = trainer_mod.AgentPPOTrainer._consume_current_step_reward_metrics(
        trainer,
        logger=None,
        # This mimics policy_train_source=both: the pre-expansion rollout IDs are
        # shorter than the policy reward rows after selected-candidate rows are appended.
        rollout_uids=["task-a", "task-b"],
    )

    assert metrics["train/pass_at_k_max_reward"] == pytest.approx(0.85)
    assert captured["table_extra_values"]["pass_at_k_max_reward"] == pytest.approx(0.85)
    assert trainer._current_step_rewards_list is None
    assert trainer._current_step_reward_tasks is None


def test_local_ace_task_group_helper_preserves_selected_rows():
    """Selected-candidate rows should share the seed task key in both-mode batches."""

    trainer = object.__new__(LocalACERLAgentPPOTrainer)

    task_ids = trainer._flatten_grouped_task_ids_in_env_order(
        {
            "task-a": [object(), object(), object(), object()],
            "task-b": [object(), object()],
        },
        {"task-a": [0, 1], "task-b": [2, 3]},
    )

    assert task_ids == ["task-a", "task-a", "task-b", "task-b", "task-a", "task-a"]
