"""MiniSWEBench reward semantics shared by workflow GRPO and agent ACE-RL."""

import os


class MiniSweRewardMixin:
    REWARD_NO_PATCH = 0.0
    REWARD_PATCH_APPLY_FAILED = 0.05
    REWARD_TESTS_FAILED = 0.2
    REWARD_RESOLVED = 1.0
    # frac mode: a graded-but-unresolved trajectory can never reach 1.0, so it
    # can never cross the is_correct (>= REWARD_RESOLVED) threshold even if the
    # report's resolved flag and the per-test buckets ever disagree.
    REWARD_FRAC_CAP = 0.99
    # frac3 defaults: per-broken-PTP-test penalty and reward floor. Env/ctor
    # overridable — MINISW_FRAC3_LAMBDA / MINISW_FRAC3_FLOOR.
    REWARD_FRAC3_LAMBDA = 0.1
    REWARD_FRAC3_FLOOR = -0.3

    def configure_reward(self, reward_mode=None, frac3_lambda=None, frac3_floor=None):
        # shaped (default): the legacy 0/0.05/0.2/1.0 tiers. binary: resolve-or-
        # nothing (1.0 vs 0.0) — removes the 0.05/0.2 shaping that trained
        # "always produce an applying patch" while resolve rate collapsed
        # (2026-07-19 diagnosis). frac: like shaped, but the flat 0.2 tier for
        # applied-but-unresolved becomes 0.2 + 0.8*ftp_frac*ptp_frac — credit
        # for the fraction of FAIL_TO_PASS tests fixed, multiplicatively
        # discounted by the fraction of PASS_TO_PASS tests still passing.
        # Multiplicative (AND), never additive: PTP passes 100% for a no-op
        # patch, so any additive PTP credit pays for doing nothing.
        # frac2: pure progress — the applied-but-unresolved tier is just
        # ftp_frac*ptp_frac with NO base, so a zero-progress applied patch pays
        # exactly 0, same as no patch. Removes the activity subsidy that kept
        # 0-vs-0.2 groups alive; meant to run on a mixed-outcome task pool
        # where group variance comes from the tasks (2026-07-23 design).
        # frac3: additive credit/penalty — ftp_frac - lambda*ptp_broken,
        # clipped to [floor, 0.99]. Fixes two frac2 blind spots (2026-07-31
        # design): (1) the near-miss flat spot — under frac2, fix-all-FTP with
        # 2 broken PTP out of 300 pays 0.99 vs 1.0 for resolved, an advantage
        # gap GRPO can't see; here each broken PTP test costs a suite-size-
        # independent lambda, so the same patch pays 0.8. (2) destruction vs
        # inaction — frac2 pays 0.0 for both no-patch and a patch that breaks
        # 30 PTP tests; here breakage with no progress goes negative (floored),
        # ranking vandalism strictly below doing nothing and adding group
        # variance on all-fail tasks. Penalty is per COUNT, not fraction: one
        # broken test kills the resolved event regardless of suite size, so its
        # price must not dilute as ptp_total grows. Penalty-on-breakage has a
        # no-op baseline of 0 (unlike additive PTP *credit*, baseline 1, which
        # pays for doing nothing — the reason additive credit stays banned).
        # The floor bounds one catastrophic rollout's |advantage| so it can't
        # dominate the group std. Downstream is negative-safe: the variance
        # gate is sign-agnostic (max-min), the solve census buckets reward<=0
        # as solve_none, and is_correct here is resolved-based.
        # reward_reason keeps the shaped labels in every mode so episode-log
        # analysis can still distinguish the outcome tiers.
        self.reward_mode = (reward_mode or os.environ.get("MINISW_REWARD_MODE") or "shaped").lower()
        if self.reward_mode not in ("shaped", "binary", "frac", "frac2", "frac3"):
            raise ValueError(f"MiniSweWorkflow: unknown reward_mode {self.reward_mode!r} (expected 'shaped', 'binary', 'frac', 'frac2' or 'frac3')")
        self.frac3_lambda = float(frac3_lambda if frac3_lambda is not None else os.environ.get("MINISW_FRAC3_LAMBDA", self.REWARD_FRAC3_LAMBDA))
        self.frac3_floor = float(frac3_floor if frac3_floor is not None else os.environ.get("MINISW_FRAC3_FLOOR", self.REWARD_FRAC3_FLOOR))
        if self.frac3_lambda < 0:
            raise ValueError(f"MiniSweWorkflow: frac3_lambda must be >= 0, got {self.frac3_lambda}")
        if self.frac3_floor > 0:
            raise ValueError(f"MiniSweWorkflow: frac3_floor must be <= 0, got {self.frac3_floor}")

    def assign_episode_correctness(self, episode) -> None:
        # is_correct here means RESOLVED (tests pass), not the base-class
        # "any reward > 0" — so val pass@1 reports resolve rate, not patch rate.
        # Requires rejection_sample.mode=variance: the legacy "correct" gate
        # reads this flag and would drop every group without a resolve,
        # discarding the no-patch-vs-patch signal.
        # MINISW_CORRECT_FLAG=applied restores the base-class reward>0 rule —
        # required when resuming legacy correct-mode runs (e.g. the 12h_full
        # case-study lineage), whose gate semantics depend on the applied flag.
        total_reward = sum(trajectory.reward or 0.0 for trajectory in episode.trajectories)
        if os.environ.get("MINISW_CORRECT_FLAG", "resolved") == "applied":
            episode.is_correct = total_reward > 0
        else:
            episode.is_correct = total_reward >= self.REWARD_RESOLVED

    def compute_trajectory_reward(self, trajectory) -> None:
        last_info = {}
        if trajectory.steps:
            last_info = getattr(trajectory.steps[-1], "info", {}) or {}

        git_diff = (last_info.get("git_diff") or trajectory.info.get("git_diff") or "").strip()
        env_reward_reason = last_info.get("reward_reason", "")

        if not git_diff:
            trajectory.reward = self.REWARD_NO_PATCH
            trajectory.info["reward_reason"] = "no_valid_patch"
            trajectory.info["patch_source"] = "none"
        elif "tests_passed" in env_reward_reason:
            trajectory.reward = self.REWARD_RESOLVED
            trajectory.info["reward_reason"] = "resolved"
            trajectory.info["patch_source"] = "env_git_diff"
        elif "tests_failed" in env_reward_reason:
            trajectory.reward = self.REWARD_TESTS_FAILED
            trajectory.info["reward_reason"] = "tests_failed"
            trajectory.info["patch_source"] = "env_git_diff"
        else:
            trajectory.reward = self.REWARD_PATCH_APPLY_FAILED
            trajectory.info["reward_reason"] = "patch_no_eval"
            trajectory.info["patch_source"] = "env_git_diff"

        trajectory.info["env_reward_reason"] = env_reward_reason
        trajectory.info["reward_mode"] = self.reward_mode
        if self.reward_mode == "frac3":
            # Self-describing logs: offline scans can recompute/counterfactual
            # the reward without knowing which knobs the job ran with.
            trajectory.info["frac3_lambda"] = self.frac3_lambda
            trajectory.info["frac3_floor"] = self.frac3_floor

        # Pass/total counts from env Phase D. Logged in every mode (episode
        # scans read them); consumed as reward only in frac mode below.
        counts = last_info.get("test_counts") or {}
        ftp_frac = ptp_frac = None
        ptp_broken = None
        if counts:
            ftp_total = counts.get("ftp_total") or 0
            ptp_total = counts.get("ptp_total") or 0
            # No FTP tests => no credit; no PTP tests => nothing to break.
            ftp_frac = (counts.get("ftp_passed", 0) / ftp_total) if ftp_total else 0.0
            ptp_frac = (counts.get("ptp_passed", 0) / ptp_total) if ptp_total else 1.0
            # Ground-truth-denominator convention carries over: a PTP test that
            # didn't run (collection crash) counts as broken.
            ptp_broken = max(ptp_total - counts.get("ptp_passed", 0), 0)
            trajectory.info["test_counts"] = dict(counts)
            trajectory.info["ftp_frac"] = ftp_frac
            trajectory.info["ptp_frac"] = ptp_frac
            trajectory.info["ptp_broken"] = ptp_broken

        if self.reward_mode == "frac" and trajectory.info["reward_reason"] == "tests_failed":
            # Only the applied-and-graded tier is graded further; resolved stays
            # a literal 1.0 and no_valid_patch/patch_no_eval keep their tiers.
            # Missing counts (unexpected path, stale env) fall back to shaped 0.2.
            if ftp_frac is not None:
                graded = self.REWARD_TESTS_FAILED + (self.REWARD_RESOLVED - self.REWARD_TESTS_FAILED) * ftp_frac * ptp_frac
                trajectory.reward = min(graded, self.REWARD_FRAC_CAP)

        if self.reward_mode == "frac2" and trajectory.info["reward_reason"] == "tests_failed":
            # Pure progress: no base. Resolved stays literal 1.0, no_valid_patch
            # 0.0 and patch_no_eval (grader crash, never observed in the 07-20
            # audit) keeps 0.05 so it stays distinguishable from a measured
            # zero-progress patch. Missing counts pay 0 — frac2 never rewards
            # unmeasured progress — with a marker so log scans catch it.
            if ftp_frac is not None:
                trajectory.reward = min(ftp_frac * ptp_frac, self.REWARD_FRAC_CAP)
            else:
                trajectory.reward = 0.0
                trajectory.info["frac2_counts_missing"] = True

        if self.reward_mode == "frac3" and trajectory.info["reward_reason"] == "tests_failed":
            # Additive credit/penalty (see __init__ comment): linear FTP credit
            # minus a per-count PTP-breakage penalty, clipped to
            # [frac3_floor, REWARD_FRAC_CAP]. Resolved stays literal 1.0;
            # no_valid_patch 0.0 and patch_no_eval 0.05 keep their tiers.
            # Missing counts pay 0 with a marker, same policy as frac2.
            if ftp_frac is not None:
                graded = ftp_frac - self.frac3_lambda * ptp_broken
                trajectory.reward = min(max(graded, self.frac3_floor), self.REWARD_FRAC_CAP)
            else:
                trajectory.reward = 0.0
                trajectory.info["frac3_counts_missing"] = True

        if self.reward_mode == "binary" and trajectory.reward < self.REWARD_RESOLVED:
            trajectory.reward = 0.0
