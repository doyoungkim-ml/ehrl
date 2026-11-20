"""Mastermind evaluation separates solved games from normalized partial credit."""

from types import SimpleNamespace

import pytest

from examples.mastermind.eval.evaluate import summarize_trajectories


def trajectory(uid, guesses):
    return SimpleNamespace(
        task={"uid": uid},
        steps=[SimpleNamespace(reward=reward, info={"won": won}) for reward, won in guesses],
        # A cumulative return must never be mistaken for normalized reward/success.
        reward=sum(reward for reward, _ in guesses),
        info={"termination_reason": "ENV_DONE"},
    )


def test_success_requires_won_and_reward_uses_best_guess():
    result = summarize_trajectories([trajectory("partial", [(0.5, False), (0.75, False)]), trajectory("solved", [(0.25, False), (1.0, True)])])
    assert result["success_count"] == 1
    assert result["success_rate"] == 0.5
    assert result["avg_reward"] == pytest.approx(0.875)
    assert result["tasks"][0]["success"] is False
    assert result["tasks"][0]["reward"] == 0.75
    assert result["tasks"][1]["success"] is True
    assert result["tasks"][1]["reward"] == 1.0


def test_reward_alone_is_not_proof_of_success():
    result = summarize_trajectories([trajectory("unsolved", [(1.0, False)])])
    assert result["success_count"] == 0
    assert result["avg_reward"] == 1.0


def test_empty_trajectory_is_unsuccessful_and_empty_summary_is_defined():
    result = summarize_trajectories([trajectory("empty", [])])
    assert result["num_tasks"] == 1
    assert result["success_rate"] == result["avg_reward"] == 0.0
    assert result["tasks"][0]["num_steps"] == 0
    empty = summarize_trajectories([])
    assert empty["num_tasks"] == empty["success_count"] == 0
    assert empty["avg_reward"] == empty["success_rate"] == 0.0
