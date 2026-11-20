from rllm.algorithms.ace.trainer_delayed_joint_base import (
    ACERLDelayedJointAgentPPOTrainer,
)


def _delayed_joint_trainer():
    return object.__new__(ACERLDelayedJointAgentPPOTrainer)


def test_best_rejection_update_filters_against_each_candidates_seed_rollout():
    trainer = _delayed_joint_trainer()
    seed_policy_groups = [
        [
            {"trajectory_reward": 0.8},
            {"trajectory_reward": 0.3},
        ]
    ]
    candidates = [
        {"task_idx": 0, "traj_idx": 0},
        {"task_idx": 0, "traj_idx": 1},
    ]

    surviving_indices = trainer._get_best_rejection_update_candidate_indices(
        candidates,
        candidate_rewards=[0.7, 0.6],
        seed_policy_groups=seed_policy_groups,
    )

    assert surviving_indices == [1]


def test_best_rejection_update_supports_local_task_seed_rollouts():
    trainer = _delayed_joint_trainer()
    seed_policy_groups = {
        "task-a": [
            {"trajectory_reward": 0.2},
            {"trajectory_reward": 0.9},
        ]
    }
    candidates = [
        {"task_id": "task-a", "traj_idx": 0},
        {"task_id": "task-a", "traj_idx": 1},
    ]

    surviving_indices = trainer._get_best_rejection_update_candidate_indices(
        candidates,
        candidate_rewards=[0.2, 0.8],
        seed_policy_groups=seed_policy_groups,
    )

    assert surviving_indices == [0]


def test_best_rejection_update_excludes_candidates_without_a_seed_rollout():
    trainer = _delayed_joint_trainer()

    surviving_indices = trainer._get_best_rejection_update_candidate_indices(
        [{"task_idx": 0, "traj_idx": 1}],
        candidate_rewards=[1.0],
        seed_policy_groups=[[{"trajectory_reward": 0.0}]],
    )

    assert surviving_indices == []
