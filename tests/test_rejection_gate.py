"""Unit tests for the episode-level rejection gate (rejection_sample.mode).

Covers:
- graded-reward cases where "variance" fixes the "correct" gate's defect
  (all-nonzero groups with reward contrast were dropped),
- the proof obligation that "variance" ≡ "correct" for binary {0, 1} rewards,
- census buckets (solve_none/solve_all/solve_partial) being mode-independent,
- the solve_all_mixed_reward diagnostic flag.
"""

import itertools

import numpy as np
import pytest

from rllm.trainer.verl.rejection_gate import group_rejection_verdict


def verdict(rewards, mode):
    rewards = np.asarray(rewards, dtype=np.float32)
    flags = rewards > 0  # mirrors Workflow.assign_episode_correctness default
    return group_rejection_verdict(rewards=rewards, flags=flags, mode=mode)


# ---------------------------------------------------------------- graded rewards


@pytest.mark.parametrize(
    "rewards, keep_correct, keep_variance, bucket, mixed",
    [
        # all failed completely: no signal, both modes drop
        ([0.0, 0.0, 0.0, 0.0], False, False, "solve_none", False),
        # all same nonzero tier: zero variance, both modes drop
        ([0.2, 0.2, 0.2, 0.2], False, False, "solve_all", False),
        ([0.05, 0.05, 0.05, 0.05], False, False, "solve_all", False),
        ([1.0, 1.0, 1.0, 1.0], False, False, "solve_all", False),
        # THE defect case: resolve among nonzero siblings — legacy drops, variance keeps
        ([0.2, 0.2, 1.0, 0.2], False, True, "solve_all", True),
        ([0.05, 0.2, 1.0, 0.2], False, True, "solve_all", True),
        ([0.05, 0.2, 0.2, 0.05], False, True, "solve_all", True),
        # mixed flags (contains a 0): both modes keep
        ([0.0, 0.2, 0.2, 0.0], True, True, "solve_partial", False),
        ([0.0, 0.2, 1.0, 0.2], True, True, "solve_partial", False),
        ([0.0, 0.0, 0.0, 0.05], True, True, "solve_partial", False),
    ],
)
def test_graded_rewards(rewards, keep_correct, keep_variance, bucket, mixed):
    keep_c, bucket_c, mixed_c = verdict(rewards, "correct")
    keep_v, bucket_v, mixed_v = verdict(rewards, "variance")
    assert keep_c == keep_correct
    assert keep_v == keep_variance
    # census bucket and diagnostic flag are mode-independent
    assert bucket_c == bucket_v == bucket
    assert mixed_c == mixed_v == mixed


# ---------------------------------------------------------------- binary equivalence


def test_binary_rewards_modes_identical():
    """For rewards in {0, 1} (mbpp-style), variance ≡ correct exactly."""
    for n in (2, 4, 8):
        for combo in itertools.product([0.0, 1.0], repeat=n):
            keep_c, bucket_c, mixed_c = verdict(combo, "correct")
            keep_v, bucket_v, mixed_v = verdict(combo, "variance")
            assert keep_c == keep_v, f"modes disagree on binary group {combo}"
            assert bucket_c == bucket_v
            assert not mixed_c and not mixed_v  # no graded tiers → never mixed_reward


# ---------------------------------------------------------------- edge cases


def test_multi_row_groups_are_handled():
    """Rows per trajectory can exceed 1; duplicated rewards must not fake variance."""
    # two trajectories, 2 rows each, same reward: still zero variance → drop
    keep, bucket, mixed = verdict([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], "variance")
    assert not keep and bucket == "solve_all" and not mixed


def test_float32_tier_noise_within_tolerance():
    """float32 representations of one tier compare equal under the gate's atol."""
    tier = np.float32(0.05)
    keep, _, _ = verdict([tier, tier, tier, tier], "variance")
    assert not keep


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        verdict([0.0, 1.0], "typo_mode")


def test_flags_not_derived_from_rewards_still_respected():
    """In 'correct' mode the FLAGS decide, whatever the rewards say — e.g. the
    mini-swe is_correct override (resolved-only) must never run with this mode."""
    rewards = np.asarray([0.0, 0.2, 0.2, 0.0], dtype=np.float32)
    flags = np.asarray([False, False, False, False])  # resolved-only flags
    keep, bucket, _ = group_rejection_verdict(rewards=rewards, flags=flags, mode="correct")
    assert not keep  # legacy gate drops on flags, whatever the rewards say
    assert bucket == "solve_partial"  # census is reward-based, not flag-based
    keep_v, _, _ = group_rejection_verdict(rewards=rewards, flags=flags, mode="variance")
    assert keep_v  # variance gate ignores the flags for the decision


def test_census_immune_to_resolved_flag_override():
    """With mini-swe's resolved-only flags, the census and the RESCUED
    diagnostic must still report in historical reward>0 terms (a flag-based
    census made solve_all_mixed_reward structurally zero)."""
    rewards = np.asarray([0.2, 0.2, 1.0, 0.2], dtype=np.float32)
    flags = np.asarray([False, False, True, False])  # is_correct = resolved
    keep, bucket, mixed = group_rejection_verdict(rewards=rewards, flags=flags, mode="variance")
    assert keep
    assert bucket == "solve_all"  # all rewards > 0
    assert mixed  # the rescued-group diagnostic fires
