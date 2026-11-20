"""Exercise the production ACE rejection path with real CPU reward tensors.

Reuse the fit-loop lifecycle fixture: generation and actor transport are stand-ins,
while the production UID grouping, reward gate, filtering, skip recovery, and
optimizer-step bookkeeping run unchanged. This does not validate GPU execution.
"""

import numpy as np
import pytest
from test_agent_ppo_rejected_batch_lifecycle import trainer

from rllm.trainer.verl.rejection_gate import REWARD_EQUALITY_ATOL, group_rejection_verdict


def rejection_trainer(plans, *, mode=None, **kwargs):
    instance = trainer(plans, **kwargs)
    # The lifecycle fixture extracts the production fit method without importing
    # Ray/VeRL; supply its shared production tolerance to that method's globals.
    instance.fit_agent.__func__.__globals__["REWARD_EQUALITY_ATOL"] = REWARD_EQUALITY_ATOL
    if mode is not None:
        instance.config.rllm.rejection_sample.mode = mode
    instance.retained_rewards = []
    record_update = instance.record_update

    def record(batch):
        record_update(batch)
        instance.retained_rewards.append(batch.batch["token_level_scores"].sum(-1).tolist())

    instance.record_update = record
    return instance


@pytest.mark.parametrize("mode", [None, "correct"])
def test_default_and_correct_preserve_legacy_fractional_reward_gate(mode):
    instance = rejection_trainer([{"rewards": [0, 0, 1, 1, 0.2, 0.2, 0.2, 1]}], mode=mode)
    instance.fit_agent()
    assert len(instance.retained_rewards) == 1
    assert instance.retained_rewards[0] == pytest.approx([0.2, 0.2, 0.2, 1])


@pytest.mark.parametrize("stepwise", [False, True])
def test_variance_filters_uniform_groups_and_preserves_graded_contrast(stepwise):
    instance = rejection_trainer([{"rewards": [0, 0, 1, 1, 0.2, 0.2, 0.2, 1]}], mode="variance", stepwise=stepwise)
    instance.fit_agent()
    assert len(instance.retained_rewards) == 1
    assert instance.retained_rewards[0] == pytest.approx([0.2, 1])
    assert instance.updates == [1]


@pytest.mark.parametrize(
    "rewards",
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.2, 0.2],
        [0.05, 0.2],
        [0.2, 1.0],
        [-1.0, 0.0],
        [1.0, 2.0],
        [0.0, 5e-7],
        [0.0, 2e-6],
        [0.2, np.nextafter(np.float32(0.2), np.float32(1.0))],
    ],
)
def test_variance_matches_workflow_gate_on_actual_filtered_batches(rewards):
    values = np.asarray(rewards, dtype=np.float32)
    expected_keep, _, _ = group_rejection_verdict(values, values > 0, mode="variance")
    instance = rejection_trainer([{"rewards": rewards}], mode="variance")
    instance.fit_agent()
    assert bool(instance.updates) == expected_keep
    if expected_keep:
        assert instance.retained_rewards[0] == pytest.approx(rewards)


def test_uniform_graded_rejection_restores_weights_before_next_generation():
    instance = rejection_trainer([{"rewards": [0.2, 0.2]}, {"rewards": [0.2, 1]}], mode="variance", next_step=31)
    instance.fit_agent()
    assert instance.generated_versions == [30, 30]
    assert instance.updates == [31]
    assert instance.global_steps == 32
    assert instance.state.events == [("sleep", 30), ("sync", 30), ("sleep", 30), ("sync", 31)]


def test_disabled_rejection_retains_uniform_rewards_in_variance_mode():
    instance = rejection_trainer([{"rewards": [0.2, 0.2]}], mode="variance")
    instance.config.rllm.rejection_sample.enable = False
    instance.fit_agent()
    assert instance.updates == [1]
    assert instance.retained_rewards[0] == pytest.approx([0.2, 0.2])


def test_unknown_rejection_mode_fails_before_actor_update():
    instance = rejection_trainer([{"rewards": [0, 1]}], mode="typo")
    with pytest.raises(ValueError, match="Unknown rejection_sample.mode"):
        instance.fit_agent()
    assert not instance.updates
