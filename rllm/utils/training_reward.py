"""Raw actor rewards and resumable completed-data-epoch monitoring.

This module intentionally uses only the standard library so a login-node
monitor can validate the same state as the training driver.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import warnings
from pathlib import Path

VERSION = 1
METRIC_ID = "initial_actor_native_trajectory_reward_v1"
REWARD_KEYS = ("train/actor_reward_sum", "train/actor_reward_count", "train/actor_reward_invalid_count")
INVALID_TERMINATIONS = frozenset({"ERROR", "ENV_ERROR", "ENV_TIMEOUT", "ENV_EXECUTION_UNCERTAIN", "TIMEOUT", "UNKNOWN"})


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def raw_reward_metrics(rows, expected_count=None):
    """Count initial actor outcomes before rejection, expansion or padding.

    A finite native zero is a valid reward. Missing rewards and infrastructure
    failures are missing evidence, never synthetic zero-reward observations.
    """
    values = []
    invalid = 0
    for row in rows:
        reward = row.get("trajectory_reward") if isinstance(row, dict) else None
        if hasattr(reward, "item"):
            try:
                reward = reward.item()
            except (TypeError, ValueError, RuntimeError):
                reward = None
        reason = row.get("termination_reason") if isinstance(row, dict) else None
        reason = str(getattr(reason, "value", reason)).upper()
        if not _finite(reward) or reason in INVALID_TERMINATIONS:
            invalid += 1
        else:
            values.append(float(reward))
    if expected_count is not None:
        if not _count(expected_count):
            raise ValueError("Expected reward count must be a nonnegative integer")
        invalid += max(0, expected_count - len(values) - invalid)
    return dict(zip(REWARD_KEYS, (math.fsum(values), len(values), invalid), strict=True))


def _active(coverage_complete):
    return {"reward_sum": 0.0, "reward_count": 0, "invalid_count": 0, "batch_count": 0, "coverage_complete": coverage_complete}


def _validate_aggregate(value):
    if not isinstance(value, dict) or not _finite(value.get("reward_sum")):
        raise ValueError("Invalid epoch reward sum")
    if any(not _count(value.get(key)) for key in ("reward_count", "invalid_count", "batch_count")) or type(value.get("coverage_complete")) is not bool:
        raise ValueError("Invalid epoch reward counts or coverage")
    if value["reward_count"] == 0 and value["reward_sum"] != 0:
        raise ValueError("Epoch reward sum without observations")
    if value["batch_count"] == 0 and (value["reward_count"] or value["invalid_count"]):
        raise ValueError("Epoch reward observations without a batch")


def _advance(best, streak, mean):
    if mean is None:
        return best, 0
    if best is None:
        return mean, 0
    if mean >= best:
        return max(mean, best), 0
    return best, streak + 1


def validate_state(state):
    """Validate stored derived values against the complete recorded history."""
    required = {"version", "metric_id", "epoch", "active", "completed_epochs", "best", "below_best_streak"}
    if not isinstance(state, dict) or not required <= state.keys() or type(state.get("version")) is not int or state["version"] != VERSION or state.get("metric_id") != METRIC_ID:
        raise ValueError("Invalid training reward state version or metric")
    if not _count(state.get("epoch")) or not isinstance(state.get("completed_epochs"), list):
        raise ValueError("Invalid training reward epoch history")
    best = None
    streak = 0
    previous = None
    for entry in state["completed_epochs"]:
        _validate_aggregate(entry)
        if not {"epoch", "eligible", "mean", "best", "below_best_streak"} <= entry.keys():
            raise ValueError("Incomplete completed training reward epoch")
        epoch = entry.get("epoch")
        if not _count(epoch) or (previous is not None and epoch != previous + 1):
            raise ValueError("Training reward epochs must be consecutive and increasing")
        previous = epoch
        eligible = entry["coverage_complete"] and entry["invalid_count"] == 0 and entry["reward_count"] > 0 and entry["batch_count"] > 0
        mean = entry["reward_sum"] / entry["reward_count"] if eligible else None
        best, streak = _advance(best, streak, mean)
        if any(entry.get(key) is not None and not _finite(entry[key]) for key in ("mean", "best")):
            raise ValueError("Invalid completed training reward mean or best")
        if (
            type(entry.get("eligible")) is not bool or entry["eligible"] != eligible
            or entry.get("mean") != mean or entry.get("best") != best
            or not _count(entry.get("below_best_streak")) or entry["below_best_streak"] != streak
        ):
            raise ValueError("Inconsistent completed training reward epoch")
    if (state.get("best") is not None and not _finite(state["best"])) or state.get("best") != best or not _count(state.get("below_best_streak")) or state["below_best_streak"] != streak:
        raise ValueError("Inconsistent training reward best or patience")
    active = state.get("active")
    if active is not None:
        _validate_aggregate(active)
        if previous is not None and state["epoch"] != previous + 1:
            raise ValueError("Active reward epoch does not follow completed history")
    elif previous != state["epoch"]:
        raise ValueError("Completed reward state has no matching final epoch")
    return state


class TrainingRewardTracker:
    def __init__(self, epoch=0, *, coverage_complete=True, state=None):
        self.state = copy.deepcopy(validate_state(state)) if state is not None else {
            "version": VERSION, "metric_id": METRIC_ID, "epoch": epoch,
            "active": _active(coverage_complete), "completed_epochs": [],
            "best": None, "below_best_streak": 0,
        }
        validate_state(self.state)

    def begin_epoch(self, epoch):
        state = self.state
        if state["active"] is not None:
            if state["epoch"] != epoch:
                raise ValueError("Cannot discard an unfinished training reward epoch")
            return
        if epoch != state["epoch"] + 1:
            raise ValueError("Training reward epoch advanced by more than one")
        state["epoch"] = epoch
        state["active"] = _active(True)

    def observe(self, metrics, *, expected_count=None):
        active = self.state["active"]
        if active is None:
            raise ValueError("Cannot observe rewards after epoch completion")
        total, count, invalid = (metrics.get(key) for key in REWARD_KEYS)
        if not _finite(total) or not _count(count) or not _count(invalid) or (count == 0 and total != 0):
            total, count, invalid = 0.0, 0, max(1, expected_count or 0)
        if expected_count is not None and count + invalid != expected_count:
            invalid += max(1, expected_count - count - invalid)
        active["reward_sum"] += total
        active["reward_count"] += count
        active["invalid_count"] += invalid
        active["batch_count"] += 1
        _validate_aggregate(active)

    def complete_epoch(self):
        state = self.state
        if state["active"] is None:
            raise ValueError("Training reward epoch was already completed")
        entry = {"epoch": state["epoch"], **state["active"]}
        eligible = entry["coverage_complete"] and entry["invalid_count"] == 0 and entry["reward_count"] > 0 and entry["batch_count"] > 0
        mean = entry["reward_sum"] / entry["reward_count"] if eligible else None
        state["best"], state["below_best_streak"] = _advance(state["best"], state["below_best_streak"], mean)
        entry.update(eligible=eligible, mean=mean, best=state["best"], below_best_streak=state["below_best_streak"])
        state["completed_epochs"].append(entry)
        state["active"] = None
        validate_state(state)
        return entry


def read_checkpoint(path, global_step=None):
    path = Path(path)
    with path.open() as stream:
        wrapper = json.load(stream)
    if (
        not isinstance(wrapper, dict) or type(wrapper.get("version")) is not int or wrapper["version"] != VERSION
        or not _count(wrapper.get("global_step")) or (global_step is not None and wrapper["global_step"] != global_step)
    ):
        raise ValueError(f"Invalid training reward checkpoint: {path}")
    return validate_state(wrapper.get("state"))


def restore_training_reward(trainer, loaded):
    """Restore only the sidecar belonging to the checkpoint VeRL loaded."""
    trainer._training_reward_tracker = None
    trainer._training_reward_legacy_resume = bool(loaded)
    if not loaded:
        return
    config = trainer.config.trainer
    folder = Path(config.resume_from_path) if config.resume_mode == "resume_path" else Path(config.default_local_dir) / f"global_step_{trainer.global_steps}"
    path = folder / "reward_epoch_state.json"
    if not path.exists():
        warnings.warn(f"Legacy checkpoint {folder} has no training reward state; historical epoch rewards are unavailable and the resumed partial epoch is ineligible.", RuntimeWarning, stacklevel=2)
        return
    state = read_checkpoint(path, global_step=trainer.global_steps)
    if not isinstance(getattr(trainer, "_dataloader_resume_iterator_finished", None), bool):
        raise RuntimeError("Training reward resume requires the pinned VeRL full dataloader-state restore and data.pt")
    trainer._training_reward_tracker = TrainingRewardTracker(state=state)
    trainer._training_epoch = state["epoch"]
    trainer._training_reward_legacy_resume = False


def initialize_training_reward(trainer, start_epoch):
    tracker = getattr(trainer, "_training_reward_tracker", None)
    if tracker is None:
        complete = not getattr(trainer, "_training_reward_legacy_resume", False) or getattr(trainer, "_dataloader_resume_iterator_finished", None) is True
        trainer._training_reward_tracker = TrainingRewardTracker(start_epoch, coverage_complete=complete)
    else:
        if tracker.state["epoch"] != trainer._training_epoch:
            raise ValueError("Training reward checkpoint epoch disagrees with controller epoch")
        if getattr(trainer, "_dataloader_resume_iterator_finished", None) is True:
            if tracker.state["active"] is not None:
                tracker.complete_epoch()
            tracker.begin_epoch(start_epoch)
        elif tracker.state["active"] is None:
            raise ValueError("Completed training reward epoch has an unfinished dataloader")
    publish_training_reward(trainer)


def save_training_reward(trainer):
    from rllm.utils.run_state import atomic_json

    if trainer.config.trainer.get("default_hdfs_dir") is not None:
        raise NotImplementedError("Training reward checkpoints require a local/shared filesystem")
    epoch = getattr(trainer, "_training_epoch", None)
    if not _count(epoch):
        raise ValueError("Training reward checkpoint requires the actual training epoch")
    tracker = getattr(trainer, "_training_reward_tracker", None)
    if tracker is None:
        # Checkpoint API calls outside fit cannot establish prior coverage.
        tracker = trainer._training_reward_tracker = TrainingRewardTracker(epoch, coverage_complete=False)
    if tracker.state["epoch"] != epoch:
        raise ValueError("Training reward epoch disagrees with training loop")
    validate_state(tracker.state)
    path = Path(trainer.config.trainer.default_local_dir) / f"global_step_{trainer.global_steps}" / "reward_epoch_state.json"
    atomic_json(path, {"version": VERSION, "global_step": trainer.global_steps, "state": tracker.state})


def publish_training_reward(trainer):
    from rllm.utils.run_state import atomic_json

    root = os.environ.get("RLLM_RUN_DIR")
    if not root:
        checkpoint_root = getattr(trainer.config.trainer, "default_local_dir", None)
        root = str(Path(checkpoint_root).parent) if checkpoint_root else None
    if root:
        atomic_json(Path(root) / "training_reward_state.json", {
            "version": VERSION, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "written_at": time.time(),
            "global_step": trainer.global_steps, "state": validate_state(trainer._training_reward_tracker.state),
        })


def complete_training_reward_epoch(trainer):
    entry = trainer._training_reward_tracker.complete_epoch()
    publish_training_reward(trainer)
    print(f"[training reward] completed epoch={entry['epoch']} eligible={entry['eligible']} mean={entry['mean']} best={entry['best']} below_best_streak={entry['below_best_streak']}", flush=True)


def workflow_reward_metrics(batch, expected_count):
    """One native actor reward per episode; ambiguous multi-trajectory episodes fail coverage."""
    episodes = {}
    ids = batch.non_tensor_batch.get("episode_ids", [])
    trajectory_ids = batch.non_tensor_batch.get("trajectory_ids")
    reasons = batch.non_tensor_batch.get("termination_reasons")
    scores = batch.batch.get("traj_rewards")
    rewards = scores.sum(-1).detach().cpu().tolist() if scores is not None else None
    for index, episode_id in enumerate(ids):
        episodes.setdefault(episode_id, []).append(index)
    rows = []
    for indices in episodes.values():
        first = indices[0]
        single_trajectory = trajectory_ids is not None and len({trajectory_ids[index] for index in indices}) == 1
        rows.append({
            "trajectory_reward": rewards[first] if single_trajectory and rewards is not None else None,
            "termination_reason": reasons[first] if reasons is not None else "UNKNOWN",
        })
    return raw_reward_metrics(rows, expected_count=expected_count)
