import pytest

from rllm.agents.agent import Step, Trajectory
from rllm.environments.env_utils import compute_trajectory_reward


def _trajectory_with_rewards(rewards: list[float]) -> Trajectory:
    return Trajectory(steps=[Step(reward=reward) for reward in rewards])


def test_compute_trajectory_reward_defaults_to_sum() -> None:
    trajectory = _trajectory_with_rewards([0.25, 0.5, 1.0])

    compute_trajectory_reward(trajectory)

    assert trajectory.reward == pytest.approx(1.75)


def test_compute_trajectory_reward_can_use_max_step_reward() -> None:
    trajectory = _trajectory_with_rewards([0.25, 0.5, 1.0])

    compute_trajectory_reward(trajectory, aggregation="max")

    assert trajectory.reward == pytest.approx(1.0)


def test_compute_trajectory_reward_rejects_unknown_aggregation() -> None:
    trajectory = _trajectory_with_rewards([0.25])

    with pytest.raises(ValueError, match="Unsupported trajectory reward aggregation"):
        compute_trajectory_reward(trajectory, aggregation="median")
