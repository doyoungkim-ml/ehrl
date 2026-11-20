"""Per-group keep/drop decision for episode-level rejection sampling.

Kept dependency-light (numpy only) so the gate logic is unit-testable without
importing the verl/ray training stack.
"""

from __future__ import annotations

import numpy as np

# Two rewards closer than this are "equal". Reward tiers across benchmarks are
# spaced far apart (e.g. mini-swe: 0 / 0.05 / 0.2 / 1.0), so this only has to
# absorb float32 representation noise, not distinguish tiers.
REWARD_EQUALITY_ATOL = 1e-6


def group_rejection_verdict(rewards: np.ndarray, flags: np.ndarray, mode: str = "correct") -> tuple[bool, str, bool]:
    """Decide keep/drop and the census bucket for one task group.

    Args:
        rewards: per-row trajectory rewards of the group's rows.
        flags: per-row is_correct flags of the same rows.
        mode: "correct" — legacy rule: keep iff flags are mixed. Exact for
            binary rewards, but for graded rewards it binarizes at reward>0
            and so drops all-nonzero groups that still have reward variance
            (e.g. [0.2, 0.2, 1.0, 0.2]).
            "variance" — keep iff the rewards are not all equal, i.e. iff the
            group has a nonzero GRPO advantage. Provably identical to
            "correct" when rewards are binary {0, 1}.

    Returns:
        (keep, bucket, mixed_reward):
        keep — whether the group enters training.
        bucket — census name: solve_none / solve_all / solve_partial. Derived
            from rewards > 0, NOT from the flags: workflows may redefine
            is_correct (e.g. mini-swe: "resolved") and the census must keep its
            historical reward-based meaning so wandb panels stay comparable.
            Identical to a flag-based census wherever flags follow the default
            reward > 0 rule (all binary benchmarks).
        mixed_reward — True for solve_all groups whose rewards are NOT all
            equal: exactly the groups "correct" mode wrongly discards. Logged
            as batch/solve_all_mixed_reward to measure that loss directly.
    """
    if mode not in ("correct", "variance"):
        raise ValueError(f"Unknown rejection_sample.mode: {mode!r} (expected 'correct' or 'variance')")

    rewards = np.asarray(rewards, dtype=np.float64)
    flags = np.asarray(flags, dtype=bool)
    has_variance = bool(rewards.max() - rewards.min() > REWARD_EQUALITY_ATOL)

    census = rewards > REWARD_EQUALITY_ATOL
    if not census.any():
        bucket = "solve_none"
    elif census.all():
        bucket = "solve_all"
    else:
        bucket = "solve_partial"

    mixed_reward = bucket == "solve_all" and has_variance
    if mode == "variance":
        keep = has_variance
    else:
        # legacy decision reads the FLAGS (not the census): keep iff mixed
        keep = bool(flags.any() and not flags.all())
    return keep, bucket, mixed_reward
