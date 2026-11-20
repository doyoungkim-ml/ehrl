import argparse
import ast
import asyncio
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.context.training_inputs import concatenate_training_insights
from rllm.context import ContextAwareRunner
from rllm.utils.compute_pass_at_k import compute_pass_at_k_max_reward_from_step_rewards
from rllm.utils.example_run_naming import get_default_run_name_with_model
from rllm.utils.provider_seed import (
    TASK_SEED_KEY,
    provider_seed_config,
    provider_seed_identity,
    provider_seed_receipt,
    provider_usage_totals,
    record_provider_seeds,
    seed_schedule,
    validate_seed_schedule,
    with_provider_seed,
)

logger = logging.getLogger(__name__)

SEQUENTIAL_LOCAL_PROTOCOL = "sequential_adaptation_v1"
LEGACY_LOCAL_PROTOCOL = "legacy_search_v1"


def local_evaluation_protocol(args: argparse.Namespace) -> str:
    return getattr(args, "local_eval_protocol", SEQUENTIAL_LOCAL_PROTOCOL)


def handoff_evaluation_config(args: argparse.Namespace) -> dict:
    """Identity for opt-in controls; default runs retain their existing schema."""
    mode = getattr(args, "adaptation_mode", "apply")
    seeds = provider_seed_config(args)
    protocol = getattr(args, "handoff_protocol", "legacy")
    if mode == "apply" and seeds is None and protocol == "legacy":
        return {}
    return {"adaptation_mode": mode, "provider_seed": seeds, **({"handoff_protocol": protocol} if protocol != "legacy" else {})}


def evaluation_role_prompt_paths(prompts_dir: str, handoff_protocol: str = "legacy") -> tuple[str, str]:
    """Select only the role assets actually consumed by this protocol."""
    if handoff_protocol not in {"legacy", "feedback_v1"}:
        raise ValueError(f"Unsupported handoff_protocol: {handoff_protocol}")
    reflector = os.path.join(prompts_dir, "reflector_prompt.txt")
    repaired = os.path.join(prompts_dir, "reflector_prompt_handoff_v1.txt")
    if handoff_protocol == "feedback_v1" and os.path.isfile(repaired):
        reflector = repaired
    curator = os.path.join(prompts_dir, "curator_prompt_handoff_v1.txt" if handoff_protocol == "feedback_v1" else "curator_prompt.txt")
    return reflector, curator


def _provider_role_work(records: list[dict], role_logs: dict[str, list[dict]]) -> dict:
    work = {}
    for role, logs in role_logs.items():
        calls = []
        for sample, log in enumerate(logs):
            count = sum(item["identity"]["role"] == role and item["identity"]["sample"] == sample for item in records)
            calls.append({"sample": sample, "request_dispatches": count, "response_present": bool(log.get("response")), "predispatch_failure": bool(count == 0 and (log.get("error") or log.get("error_type")))})
        work[role] = {"scheduled_calls": len(logs), "calls": calls}
    return work


@dataclass
class ACEExperimentHooks:
    experiment_name: str
    task_id_getter: Callable[[dict], str]
    build_generator_engine: Callable[[argparse.Namespace], Any]
    create_context_rule: Callable[[argparse.Namespace, str, str, str | None], Any]
    trajectory_reward: Callable[[Any], float]
    trajectory_success: Callable[[Any], bool]
    render_prompt_in_agent: bool = False
    prepare_local_worker_args: Callable[[argparse.Namespace, int, int], argparse.Namespace] | None = None
    parse_evaluation_args: Callable[[list[str]], argparse.Namespace] | None = None


def reserve_worker_parallel_agents(args: argparse.Namespace, worker_index: int, worker_count: int, benchmark: str = "Local") -> argparse.Namespace:
    """Give each concurrent local task worker an isolated generator engine with its own g_act slots.

    Benchmarks whose environments need no per-worker resources beyond a
    separate engine (ALFWorld, MBPP, Mastermind) use this as their
    ``prepare_local_worker_args`` hook. Total in-flight trajectories
    (workers x g_act) must fit the configured parallel-agent capacity; the
    returned copy reserves exactly g_act slots for one worker and leaves the
    caller's namespace untouched.
    """
    g_act = max(1, int(getattr(args, "g_act", 1)))
    n_parallel_agents = int(getattr(args, "n_parallel_agents", 1))
    if worker_count < 1 or worker_count * g_act > n_parallel_agents:
        raise ValueError(f"{benchmark} task concurrency needs n_parallel_agents >= task_concurrency * g_act")
    if not 0 <= worker_index < worker_count:
        raise ValueError(f"{benchmark} worker index is outside the configured worker count")
    worker_args = copy.deepcopy(args)
    worker_args.n_parallel_agents = g_act
    return worker_args


@dataclass
class GlobalRunMetrics:
    run_id: str
    experiment_name: str
    model_name: str
    epochs: int
    train_batch_size: int
    total_train_tasks: int
    total_steps: int
    train_steps: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class LocalRunMetrics:
    run_id: str
    experiment_name: str
    model_name: str
    local_eval_epochs: int
    epochs: int
    train_batch_size: int
    total_train_tasks: int
    steps_per_epoch: int
    evaluations: list[dict] = field(default_factory=list)
    local_eval_protocol: str = SEQUENTIAL_LOCAL_PROTOCOL

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def empty_ace_log(error: str | None = None, skipped: bool = False) -> dict:
    payload = {"prompt": "", "response": "", "parsed": {}}
    if error:
        payload["error"] = error
    if skipped:
        payload["skipped"] = True
    return payload


def json_default(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def find_repo_root(script_dir: str) -> str:
    """Walk up from a benchmark script directory to the checkout root (pyproject.toml).

    Benchmark scripts live at examples/<bench>/ or examples/<bench>/eval/, so a fixed
    depth no longer works; fall back to two levels up if no marker is found.
    """
    current = os.path.abspath(script_dir)
    while True:
        if os.path.isfile(os.path.join(current, "pyproject.toml")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.dirname(os.path.dirname(os.path.abspath(script_dir)))
        current = parent


def resolve_path(script_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(find_repo_root(script_dir), path)


def safe_task_artifact_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(task_id))


def get_default_prompts_dir(script_dir: str) -> str:
    return os.path.join(script_dir, "prompts")


def get_default_playbook_path(script_dir: str) -> str:
    return os.path.join(script_dir, "playbooks", "initial_playbook.txt")


def normalize_eval_frequency(args: argparse.Namespace) -> None:
    test_freq = getattr(args, "test_freq", None)
    if test_freq is not None:
        args.test_epoch_freq = test_freq


def init_wandb_run(args: argparse.Namespace, script_path: str, config: dict) -> Any:
    if not getattr(args, "wandb_project", None):
        return None

    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed, skipping logging")
        return None

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or get_default_run_name_with_model(script_path, args.model_name),
        config=config,
    )


def maybe_create_playbook_tokenizer(args: argparse.Namespace):
    max_playbook_length = getattr(args, "max_playbook_length", None)
    if max_playbook_length is None:
        return None

    from transformers import AutoTokenizer

    from rllm.utils.client_factory import is_external_model

    model_name = args.model_name
    if is_external_model(model_name):
        model_name = "Qwen/Qwen2-0.5B-Instruct"
    return AutoTokenizer.from_pretrained(model_name)


def get_playbook_token_length(playbook_text: str, tokenizer) -> int:
    return len(tokenizer.encode(playbook_text, add_special_tokens=False))


def summarize_trajectories(trajectories: list, hooks: ACEExperimentHooks) -> dict:
    rewards = [hooks.trajectory_reward(traj) for traj in trajectories]
    steps = [len(getattr(traj, "steps", []) or []) for traj in trajectories]
    success_count = sum(1 for traj in trajectories if hooks.trajectory_success(traj))
    return {
        "num_trajectories": len(trajectories),
        "success_count": success_count,
        "success_rate": success_count / len(trajectories) if trajectories else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
    }


def _task_best_rewards(trajectories: list, hooks: ACEExperimentHooks) -> dict[str, float]:
    """Map each task id to the best reward achieved across its trajectories."""
    task_best_reward: dict[str, float] = {}
    for idx, trajectory in enumerate(trajectories):
        task = getattr(trajectory, "task", None)
        fallback_task_id = f"trajectory_{idx}"
        try:
            task_id = str(hooks.task_id_getter(task)) if task is not None else fallback_task_id
        except Exception:
            task_id = fallback_task_id

        reward = float(hooks.trajectory_reward(trajectory))
        previous_best = task_best_reward.get(task_id)
        if previous_best is None or reward > previous_best:
            task_best_reward[task_id] = reward
    return task_best_reward


def compute_pass_at_k_max_reward(trajectories: list, hooks: ACEExperimentHooks) -> float:
    """Average the best trajectory reward for each unique task."""
    if not trajectories:
        return 0.0

    task_best_reward = _task_best_rewards(trajectories, hooks)
    return float(np.mean(list(task_best_reward.values()))) if task_best_reward else 0.0


def compute_pass_at_k_success_rate(trajectories: list, hooks: ACEExperimentHooks) -> float:
    """Fraction of tasks solved by at least one trajectory."""
    if not trajectories:
        return 0.0

    task_best_reward = _task_best_rewards(trajectories, hooks)
    return float(np.mean([reward >= 1.0 for reward in task_best_reward.values()])) if task_best_reward else 0.0


def freeze_solved_reward(solved: set[str], task_id: str, iter_reward: float) -> float:
    """Doyoung freeze-at-1.0 rule (single primitive shared by the live freeze path
    and the offline recovery, so both compute the same `max_reward_mean`).

    Once a task reaches reward>=1.0 it is 'solved' and its recorded reward is
    padded at 1.0 for every remaining iteration. A task that has NOT solved is
    left as-is — evaluation continues and its fresh per-iteration reward is used
    (no running-max smoothing on partial-reward tasks).

    Mutates `solved` (adds task_id once it hits 1.0) and returns the padded value.
    """
    if task_id in solved or iter_reward >= 1.0:
        solved.add(task_id)
        return 1.0
    return iter_reward


def max_reward_mean_from_local_epochs(local_epochs: list[dict]) -> dict:
    """Recover the fresh vs frozen (freeze-at-1.0) per-iteration reward curves
    from saved metrics (`evaluations[i].local_epochs[*].tasks[*].reward`). Uses
    the same `freeze_solved_reward` primitive as the live path.

    Returns {"iterations", "fresh_mean", "max_reward_mean", "n_tasks"}.
    `max_reward_mean` pads solved tasks at 1.0 and leaves not-yet-solved tasks at
    their fresh reward. Only tasks present in every iteration are aggregated.
    """
    per_task: dict[str, list[float]] = {}
    for ep in local_epochs:
        for row in ep.get("tasks", []) or []:
            per_task.setdefault(str(row["task_id"]), []).append(float(row["reward"]))
    n_iter = len(local_epochs)
    tasks = [t for t in per_task if len(per_task[t]) == n_iter]
    if not tasks or n_iter == 0:
        return {"iterations": [], "fresh_mean": [], "max_reward_mean": [], "n_tasks": 0}

    fresh_mean, max_reward_mean = [], []
    solved: set[str] = set()
    for i in range(n_iter):
        fresh_mean.append(float(np.mean([per_task[t][i] for t in tasks])))
        vals = [freeze_solved_reward(solved, t, per_task[t][i]) for t in tasks]
        max_reward_mean.append(float(np.mean(vals)))
    return {
        "iterations": list(range(n_iter)),
        "fresh_mean": fresh_mean,
        "max_reward_mean": max_reward_mean,
        "n_tasks": len(tasks),
    }


def trajectory_step_rewards(trajectory: Any) -> list[float]:
    rewards = []
    for step in getattr(trajectory, "steps", []) or []:
        if getattr(step, "reward", None) is not None:
            rewards.append(float(step.reward))
    if not rewards and getattr(trajectory, "reward", None) is not None:
        rewards.append(float(trajectory.reward))
    return rewards


def build_turn_rows(
    step_rewards_list: list[list[float]],
    *,
    cutoff_mode: str = "max",
    max_steps: int | None = None,
) -> tuple[list[dict], dict]:
    if not step_rewards_list:
        return [], {}

    actual_max_turns = max(len(item) for item in step_rewards_list)
    if actual_max_turns <= 0:
        return [], {}

    max_turns = max_steps or actual_max_turns
    processed = []
    original_lengths = [len(item) for item in step_rewards_list]
    for rewards in step_rewards_list:
        if len(rewards) >= max_turns:
            processed.append(rewards[:max_turns])
            continue
        extended = list(rewards)
        if cutoff_mode == "zero":
            fill_value = 0.0
        elif cutoff_mode == "max":
            fill_value = max(rewards) if rewards else 0.0
        elif cutoff_mode == "last":
            fill_value = rewards[-1] if rewards else 0.0
        else:
            raise ValueError(f"Unknown cutoff_mode: {cutoff_mode}")
        extended.extend([fill_value] * (max_turns - len(rewards)))
        processed.append(extended)

    turn_rows = []
    turn_rewards = []
    cumulative_rewards = []
    max_rewards = []
    min_rewards = []
    alive_counts = []

    for turn_idx in range(max_turns):
        values = [item[turn_idx] for item in processed]
        turn_avg = float(np.mean(values)) if values else 0.0
        turn_rewards.append(turn_avg)
        cumulative_rewards.append(turn_avg + (cumulative_rewards[-1] if cumulative_rewards else 0.0))
        trajectory_max = [max(item[: turn_idx + 1]) for item in processed]
        trajectory_min = [min(item[: turn_idx + 1]) for item in processed]
        max_rewards.append(float(np.mean(trajectory_max)) if trajectory_max else 0.0)
        min_rewards.append(float(np.mean(trajectory_min)) if trajectory_min else 0.0)
        alive_counts.append(sum(1 for length in original_lengths if turn_idx < length))
        turn_rows.append(
            {
                "turn": turn_idx,
                "avg_reward": turn_rewards[-1],
                "cumulative_reward": cumulative_rewards[-1],
                "max_reward": max_rewards[-1],
                "min_reward": min_rewards[-1],
                "n_trajectories": alive_counts[-1],
            }
        )

    summary = {}
    if turn_rows:
        summary = {
            "final_avg_reward": turn_rewards[-1],
            "final_cumulative_reward": cumulative_rewards[-1],
            "final_max_reward": max_rewards[-1],
            "best_avg_reward": max(turn_rewards),
            "best_max_reward": max(max_rewards),
        }
    return turn_rows, summary


def log_turn_table(
    step: int,
    rows: list[dict],
    extra_values: dict | None = None,
    *,
    metric_prefix: str = "val",
) -> None:
    if not rows:
        return

    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    extra_values = extra_values or {}
    base_columns = [
        "turn",
        "avg_reward",
        "cumulative_reward",
        "max_reward",
        "min_reward",
        "n_trajectories",
    ]
    extra_row_columns = []
    seen_columns = set(base_columns)
    for row in rows:
        for key in row:
            if key not in seen_columns:
                seen_columns.add(key)
                extra_row_columns.append(key)

    columns = base_columns + extra_row_columns + list(extra_values.keys())
    table = wandb.Table(columns=columns)
    for row in rows:
        values = [row.get(key) for key in base_columns]
        values.extend(row.get(key) for key in extra_row_columns)
        values.extend(extra_values[key] for key in extra_values)
        table.add_data(*values)
    wandb.log({f"{metric_prefix}/turn_table": table}, step=step)


def _mean_metric(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _build_local_eval_table_rows(
    turn_rows: list[dict],
    *,
    local_eval_epoch: int,
    local_eval_epochs: int,
    epoch_metrics: dict,
    epoch_turn_offset: int,
) -> list[dict]:
    table_rows = []
    repeated_metric_keys = ["pass_at_k_max_reward"] + sorted(key for key in epoch_metrics if key.startswith("ace/") or key.startswith("playbook/"))

    for row in turn_rows:
        enriched = dict(row)
        local_turn = int(row["turn"])
        enriched["turn"] = epoch_turn_offset + local_turn
        enriched["local_turn"] = local_turn
        enriched["local_eval_epoch"] = int(local_eval_epoch)
        enriched["local_eval_epoch_total"] = int(local_eval_epochs)
        for key in repeated_metric_keys:
            if key in epoch_metrics:
                enriched[key] = epoch_metrics[key]
        table_rows.append(enriched)
    return table_rows


def save_policy_chat_completions(
    trajectories: list,
    chat_completion_dir: str,
    step: int,
    *,
    mode: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
    label: str | None = None,
    local_eval_epoch: int | None = None,
) -> str:
    mode_dir = os.path.join(chat_completion_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    # In local ACE validation each checkpoint step is re-run once per local_eval_epoch
    # (iteration). Without an iteration component in the filename every iteration would
    # overwrite the previous one and only the last iteration would survive on disk, so
    # suffix with _iter_<i> when local_eval_epoch is provided. Global val/train callers
    # pass None and keep the original iteration-free filename (backward compatible).
    suffix = f"_iter_{local_eval_epoch - 1}" if local_eval_epoch is not None else ""
    path = os.path.join(mode_dir, f"policy_step{step:06d}{suffix}.json")
    payload = {
        "step": step,
        "mode": mode,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "num_trajectories": len(trajectories),
        "trajectories": [trajectory.to_dict() for trajectory in trajectories],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)
    return path


def save_ace_step_log(
    ace_log_dir: str,
    step: int,
    payload: dict,
    *,
    mode: str,
    local_eval_epoch: int | None = None,
) -> str:
    mode_dir = os.path.join(ace_log_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    # Mirror save_policy_chat_completions: suffix with _iter_<i> in local ACE validation so
    # each iteration's reflector/curator log is preserved instead of overwritten. Global
    # val/train callers pass None and keep the iteration-free filename (backward compatible).
    suffix = f"_iter_{local_eval_epoch - 1}" if local_eval_epoch is not None else ""
    path = os.path.join(mode_dir, f"ace_step{step:06d}{suffix}.json")
    enriched = {
        "step": step,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        **payload,
    }
    with open(path, "w") as f:
        json.dump(enriched, f, indent=2, default=json_default)
    return path


def save_global_playbook_snapshot(playbook_text: str, snapshot_dir: str, step: int) -> str:
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, f"playbook_step{step:04d}.txt")
    with open(path, "w") as f:
        f.write(f"# Playbook Snapshot - Step {step}\n")
        f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"# Size: {len(playbook_text)} characters\n")
        f.write("#" + "=" * 60 + "\n\n")
        f.write(playbook_text)
    return path


def resolve_local_artifact_root(
    playbook_snapshot_dir: str | None,
    ace_log_dir: str | None,
) -> str | None:
    if playbook_snapshot_dir:
        return playbook_snapshot_dir
    if not ace_log_dir:
        return None
    normalized = os.path.normpath(ace_log_dir)
    if os.path.basename(normalized) == "ace_logs":
        return os.path.dirname(normalized)
    return normalized


def task_artifact_dir(task_artifact_root: str | None, task_id: str, leaf_dir: str) -> str | None:
    if not task_artifact_root:
        return None
    safe_task_id = safe_task_artifact_id(task_id)
    return os.path.join(task_artifact_root, f"task_{safe_task_id}", leaf_dir)


def _local_task_receipt_path(root: str, task_id: str, step: int, visit: int) -> Path:
    return Path(task_artifact_dir(root, task_id, "completed_visits")) / f"step{step:06d}_iter_{visit - 1}.json"


def _save_local_task_receipt(root: str, task_id: str, step: int, visit: int, result: dict, playbook_before: str, playbook_after: str) -> None:
    """Commit one whole task visit after its trajectory and ACE artifacts exist."""
    row = result["row"]
    artifacts = row["artifacts"]
    paths = {name: Path(artifacts[name]) for name in ("policy_chat_completions", "playbook_snapshot", "ace_log")}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    receipt = {
        "version": 1, "task_id": task_id, "step": step, "visit": visit,
        "playbook_before": playbook_before, "playbook_after": playbook_after,
        "row": row, "artifact_sha256": hashes,
        "reflector_rollouts": result["reflector_rollouts"],
        "curator_rollouts": result["curator_rollouts"],
        "guidebooks": result["guidebooks"],
    }
    path = _local_task_receipt_path(root, task_id, step, visit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(receipt, indent=2, default=json_default) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_local_task_receipt(root: str, task_id: str, step: int, visit: int, playbook_before: str) -> dict | None:
    path = _local_task_receipt_path(root, task_id, step, visit)
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if (receipt.get("version"), receipt.get("task_id"), receipt.get("step"), receipt.get("visit"), receipt.get("playbook_before")) != (1, task_id, step, visit, playbook_before):
        raise ValueError(f"Completed task visit does not match this evaluation: {path}")
    row = receipt["row"]
    if row.get("task_id") != task_id or row.get("local_eval_epoch") != visit:
        raise ValueError(f"Completed task visit has inconsistent task identity: {path}")
    artifacts = row["artifacts"]
    for name in ("policy_chat_completions", "playbook_snapshot", "ace_log"):
        artifact = Path(artifacts[name])
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != receipt["artifact_sha256"][name]:
            raise ValueError(f"Completed task visit artifact changed or is missing: {artifact}")
    from rllm.agents.agent import Trajectory

    saved = json.loads(Path(artifacts["policy_chat_completions"]).read_text())
    trajectories = [Trajectory.from_dict(item) for item in saved["trajectories"]]
    if saved.get("step") != step or len(trajectories) != saved.get("num_trajectories"):
        raise ValueError(f"Completed task visit has inconsistent trajectories: {path}")
    snapshot = _strip_playbook_snapshot_header(Path(artifacts["playbook_snapshot"]).read_text())
    ace_log = json.loads(Path(artifacts["ace_log"]).read_text())
    if snapshot != receipt["playbook_after"] or ace_log.get("playbook") != {"before": playbook_before, "after": snapshot}:
        raise ValueError(f"Completed task visit has inconsistent playbook state: {path}")
    if "samples" in row and [trajectory.info.get("evaluation_sample") for trajectory in trajectories] != row["samples"]:
        raise ValueError(f"Completed task visit has inconsistent policy samples: {path}")
    return {
        "row": row, "trajectories": trajectories,
        "reflector_rollouts": receipt["reflector_rollouts"],
        "curator_rollouts": receipt["curator_rollouts"],
        "guidebooks": receipt["guidebooks"],
        "playbook_after": receipt["playbook_after"],
    }


def _recover_local_task_artifacts(
    root: str, task_id: str, task_idx: int, step: int, visit: int,
    total_visits: int, playbook_before: str, args: argparse.Namespace,
    hooks: ACEExperimentHooks,
) -> dict | None:
    """Import complete task artifacts written before task receipts were added.

    Three files alone are not proof of a whole visit: verify their identities,
    playbook chain and sample budget before committing an atomic receipt.
    """
    suffix = f"_iter_{visit - 1}"
    task_dir = Path(root) / f"task_{safe_task_artifact_id(task_id)}"
    paths = {
        "policy_chat_completions": task_dir / "chat_completions" / "eval" / f"policy_step{step:06d}{suffix}.json",
        "playbook_snapshot": task_dir / "playbooks" / "eval" / f"playbook_step{step:04d}{suffix}.txt",
        "ace_log": task_dir / "ace_logs" / "eval" / f"ace_step{step:06d}{suffix}.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        recorded_ace_log = json.loads(paths["ace_log"].read_text())
    except json.JSONDecodeError:
        return None
    if recorded_ace_log.get("handoff_control", {}) != handoff_evaluation_config(args):
        raise ValueError("Task artifacts have a different handoff control configuration")
    try:
        policy = json.loads(paths["policy_chat_completions"].read_text())
        ace_log = recorded_ace_log
        from rllm.agents.agent import Trajectory

        trajectories = [Trajectory.from_dict(item) for item in policy["trajectories"]]
        after = _strip_playbook_snapshot_header(paths["playbook_snapshot"].read_text())
        values = ace_log["metrics"]
        if (
            policy.get("step") != step or policy.get("mode") != "eval"
            or policy.get("num_trajectories") != len(trajectories)
            or ace_log.get("step") != step or ace_log.get("mode") != "eval"
            or ace_log.get("task_id") != task_id or ace_log.get("task_idx") != task_idx
            or ace_log.get("local_eval_epoch") != visit or ace_log.get("local_eval_epochs") != total_visits
            or ace_log.get("local_eval_protocol") != local_evaluation_protocol(args)
            or ace_log.get("playbook") != {"before": playbook_before, "after": after}
            or ace_log.get("artifacts", {}).get("policy_chat_completions") != str(paths["policy_chat_completions"])
            or ace_log.get("artifacts", {}).get("playbook_snapshot") != str(paths["playbook_snapshot"])
            or any(hooks.task_id_getter(trajectory.task) != task_id for trajectory in trajectories)
        ):
            return None
        if local_evaluation_protocol(args) == SEQUENTIAL_LOCAL_PROTOCOL and len(trajectories) != args.g_act:
            return None
        current = summarize_trajectories(trajectories, hooks)
        if any(not math.isclose(float(values[key]), current[key], abs_tol=1e-9) for key in ("avg_reward", "success_rate", "avg_steps")):
            return None
        samples = [trajectory.info.get("evaluation_sample") for trajectory in trajectories]
        if local_evaluation_protocol(args) == SEQUENTIAL_LOCAL_PROTOCOL and any(not isinstance(sample, dict) for sample in samples):
            return None
        row = {
            **({key: ace_log[key] for key in ("handoff_control", "adaptation_generated", "adaptation_changed") if key in ace_log}),
            **({key: ace_log[key] for key in ("provider_seed", "provider_requests", "provider_usage", "provider_role_work")} if provider_seed_config(args) else {}),
            "local_eval_epoch": visit, "task_id": task_id, "reward": current["avg_reward"],
            "selected_reward": values["selected_reward"], "best_candidate_reward": values["best_candidate_reward"],
            "reject_candidate": int(values["reject_candidate"]), "num_candidates": values["num_candidates"],
            "playbook_size": values["playbook_size"], "playbook_bullets": values["playbook_bullets"],
            "playbook_growth": values["playbook_growth"],
            "artifacts": {name: str(path) for name, path in paths.items()},
        }
        if local_evaluation_protocol(args) == SEQUENTIAL_LOCAL_PROTOCOL:
            row.update(
                samples=samples, policy_playbook=playbook_before,
                adaptation_applied=ace_log["selection"]["selected_idx"] >= 0,
                adaptation_scheduled=bool(visit < total_visits and args.g_ref),
            )
        curator_rollouts = ace_log["curator"]["attempted"]
        guidebooks = curator_rollouts // args.g_cur if args.g_cur else len(ace_log["curator"].get("candidates", [])) if getattr(args, "reflector_only", False) else 0
        result = {
            "row": row, "trajectories": trajectories,
            "reflector_rollouts": len(ace_log["reflector"]["logs"]),
            "curator_rollouts": curator_rollouts,
            "guidebooks": guidebooks,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    _save_local_task_receipt(root, task_id, step, visit, result, playbook_before, after)
    return result


def save_local_playbook_snapshot(
    playbook_text: str,
    task_artifact_root: str,
    task_id: str,
    step: int,
    *,
    mode: str,
    local_eval_epoch: int | None = None,
) -> str:
    playbook_dir = task_artifact_dir(task_artifact_root, task_id, "playbooks")
    if playbook_dir is None:
        raise ValueError("task_artifact_root must be set for local playbook snapshots")
    mode_dir = os.path.join(playbook_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    suffix = f"_iter_{local_eval_epoch - 1}" if local_eval_epoch is not None else ""
    path = os.path.join(mode_dir, f"playbook_step{step:04d}{suffix}.txt")

    with open(path, "w") as f:
        f.write("# Local Playbook Snapshot\n")
        f.write(f"# Task ID: {task_id}\n")
        f.write(f"# Step: {step}\n")
        f.write(f"# Mode: {mode}\n")
        f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"# Size: {len(playbook_text)} characters\n")
        f.write("#" + "=" * 60 + "\n\n")
        f.write(playbook_text)
    return path


def _strip_playbook_snapshot_header(raw: str) -> str:
    """Inverse of save_local_playbook_snapshot's header: drop the leading
    contiguous '#' comment block and the blank separator line(s) after it.
    Only the LEADING block is stripped — playbook content itself is untouched."""
    lines = raw.splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:])


def load_warm_start_state(
    source_run_dir: str,
    task_ids: list[str],
    step: int,
    start_epoch: int,
    *,
    mode: str = "eval",
) -> tuple[dict, set, dict]:
    """Load the cross-epoch state a finished run left behind, to warm-start an
    extension run at epoch start_epoch+1.

    The epoch loop is Markov in exactly two things (see _run_single_local_validation_epoch):
      1. the per-task playbook  -> playbooks/<mode>/playbook_step%04d_iter_{start_epoch-1}.txt
      2. the solved set         -> any prior iter's ace log with avg_reward >= 1.0
                                   (same rule as freeze_solved_reward)

    Missing PLAYBOOK snapshot -> hard error (a partial warm start silently corrupts
    the curve). Missing ACE LOG -> warn + treat as not-solved (only affects the
    padded max_reward_mean metric, and only if the task never appears solved
    elsewhere).

    Returns (playbook_texts_by_task_id, solved_task_ids, provenance_dict).
    """
    if start_epoch < 1:
        raise ValueError(f"warm start requires start_epoch >= 1 (got {start_epoch})")
    playbooks: dict = {}
    solved: set = set()
    missing: list[str] = []
    missing_ace_logs = 0
    snapshot_paths: dict = {}
    for task_id in task_ids:
        pb_dir = task_artifact_dir(source_run_dir, task_id, "playbooks")
        path = os.path.join(pb_dir, mode, f"playbook_step{step:04d}_iter_{start_epoch - 1}.txt")
        if not os.path.isfile(path):
            missing.append(path)
            continue
        with open(path) as f:
            playbooks[task_id] = _strip_playbook_snapshot_header(f.read())
        snapshot_paths[task_id] = path
        ace_dir = task_artifact_dir(source_run_dir, task_id, "ace_logs")
        for i in range(start_epoch):
            log_path = os.path.join(ace_dir, mode, f"ace_step{step:06d}_iter_{i}.json")
            if not os.path.isfile(log_path):
                missing_ace_logs += 1
                continue
            try:
                with open(log_path) as f:
                    payload = json.load(f)
                    reward = float(payload.get("metrics", {}).get("avg_reward", 0.0) or 0.0)
            except Exception:
                missing_ace_logs += 1
                continue
            if payload.get("local_eval_protocol", LEGACY_LOCAL_PROTOCOL) != LEGACY_LOCAL_PROTOCOL:
                raise ValueError("Legacy warm start cannot import sequential evaluation results")
            if reward >= 1.0:
                solved.add(task_id)
                break
    if missing:
        raise ValueError(
            f"warm start from {source_run_dir}: {len(missing)}/{len(task_ids)} task playbook snapshots MISSING for iter_{start_epoch - 1} — refusing a partial warm start. First missing: {missing[0]}"
        )
    if missing_ace_logs:
        print(f"[WARM-START] warning: {missing_ace_logs} ace log(s) missing/unreadable while rebuilding the solved set — affected (task, iter) pairs treated as not-solved", flush=True)
    provenance = {
        "extended_from": os.path.abspath(source_run_dir),
        "step": step,
        "start_epoch": start_epoch,
        "mode": mode,
        "n_tasks": len(task_ids),
        "n_solved_at_start": len(solved),
        "snapshot_mtimes": {t: os.path.getmtime(p) for t, p in snapshot_paths.items()},
    }
    return playbooks, solved, provenance


# These flags describe transport, output placement, or the explicitly separate
# final pass. Everything else in the benchmark's resolved namespace must match.
_FINAL_POLICY_RUNTIME_OPTIONS = {
    "base_url",
    "ace_base_url",
    "remote_environment_url",
    "base_port",
    "n_servers",
    "n_parallel_agents",
    "task_concurrency",
    "api_key",
    "api_key_file",
    "ace_api_key",
    "ace_key_file",
    "project_id",
    "ace_project_id",
    "playbook_snapshot_dir",
    "ace_log_dir",
    "metrics_file",
    "wandb_project",
    "wandb_run_name",
    "wandb_run_id",
    "wandb_step",
    "debug",
    "final_policy_eval",
    "final_policy_from_run_dir",
    "final_policy_source_manifest",
    "warm_start_from_run_dir",
    "start_epoch",
}


def _read_final_policy_json(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid final-policy source artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in final-policy source artifact: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _source_evaluation_arguments(command: list[str]) -> list[str]:
    if not isinstance(command, list) or not all(isinstance(token, str) for token in command) or len(command) < 3 or command[1] != "-m":
        raise ValueError("Final-policy source requires a recorded module evaluation command")
    result = []
    index = 3
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            raise ValueError("Malformed final-policy source command")
        option = token[2:].split("=", 1)[0].removeprefix("no-")
        end = index + 1
        while end < len(command) and not command[end].startswith("--"):
            end += 1
        if option not in _FINAL_POLICY_RUNTIME_OPTIONS:
            result.extend(command[index:end])
        index = end
    return result


def _final_policy_protocol(args: argparse.Namespace) -> dict:
    protocol = {key: value for key, value in vars(args).items() if key not in _FINAL_POLICY_RUNTIME_OPTIONS}
    # Explicit source asset paths can move while their content stays identical.
    # Empty/default paths have no historical hashes in legacy completion records;
    # deployment provenance must establish the default asset revision separately.
    for name in ("prompts_dir", "initial_playbook"):
        value = protocol.get(name)
        if not value:
            continue
        path = Path(value)
        try:
            if name == "prompts_dir":
                files = sorted(item for item in path.rglob("*") if item.is_file())
                if not path.is_dir() or not files:
                    raise ValueError(f"Cannot verify final-policy protocol assets: {path}")
                protocol[name] = {str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest() for item in files}
            else:
                protocol[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"Cannot verify final-policy protocol assets: {path}") from exc
    return protocol


def _final_policy_manifest_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"Final-policy source receipt requires an absolute {label}")
    return Path(value).resolve()


def _verify_appworld_default_resolver(evaluator: bytes, utilities: bytes) -> None:
    """Recognize the retained AppWorld defaults without importing old source."""
    expected = {
        None: {
            "_REPO_ROOT": "Path(__file__).resolve().parents[4]",
            "TRAIN_ARTIFACTS_DIR": '_REPO_ROOT / "examples" / "appworld" / "train" / "ace_rl"',
        },
        "main": {
            "prompts_dir": 'resolve_path(script_dir, args.prompts_dir) if args.prompts_dir else str(TRAIN_ARTIFACTS_DIR / "prompts")',
            "initial_playbook": 'resolve_path(script_dir, args.initial_playbook) if args.initial_playbook else str(TRAIN_ARTIFACTS_DIR / "playbooks" / "initial_playbook.txt")',
        },
        "create_ace_context_rule": {
            "reflector_prompt_path": 'os.path.join(prompts_dir, "reflector_prompt.txt")',
            "curator_prompt_path": 'os.path.join(prompts_dir, "curator_prompt.txt")',
        },
    }
    try:
        evaluator_tree, utility_tree = ast.parse(evaluator), ast.parse(utilities)
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError("Cannot parse retained AppWorld default asset resolver") from exc
    for scope, assignments in expected.items():
        tree = utility_tree if scope == "create_ace_context_rule" else evaluator_tree
        if scope is not None:
            scopes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == scope]
            if len(scopes) != 1:
                raise ValueError("Unrecognized retained AppWorld default asset resolver")
            tree = scopes[0]
        for name, expression in assignments.items():
            values = [node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
            if len(values) != 1 or ast.dump(values[0]) != ast.dump(ast.parse(expression, mode="eval").body):
                raise ValueError(f"Unrecognized retained AppWorld default asset resolver: {name}")


def _resolve_final_policy_source_receipt(
    *,
    receipt_path: str,
    source: Path,
    source_args: argparse.Namespace,
    command: list[str],
    config: dict,
    hashes: dict[str, str],
    hooks: ACEExperimentHooks,
) -> dict:
    """Bind omitted historical defaults to retained launch/deployment evidence.

    The receipt is an explicit provenance assertion, not a signed attestation or
    a claim that legacy completion markers originally recorded asset hashes.
    """
    receipt_file = _final_policy_manifest_path(receipt_path, "receipt path")
    receipt, receipt_hash = _read_final_policy_json(receipt_file)
    if receipt.get("version") != 1 or isinstance(receipt.get("version"), bool):
        raise ValueError("Unsupported final-policy source receipt version")
    if _final_policy_manifest_path(receipt.get("source_run_dir"), "source run directory") != source:
        raise ValueError("Final-policy source receipt is bound to a different run")
    if any(receipt.get(key) != value for key, value in hashes.items()):
        raise ValueError("Final-policy source receipt hashes do not match the unchanged source run")
    if command[2] != "examples.appworld.eval.ace.evaluate" or hooks.experiment_name != "appworld_local_ace":
        raise ValueError("Default asset receipts currently support historical AppWorld ACE evaluations only")
    source_root = _final_policy_manifest_path(receipt.get("source_root"), "frozen source root")
    manifests = {}
    manifest_provenance = {}
    for name in ("deployment_manifest", "launch_manifest"):
        reference = receipt.get(name)
        if not isinstance(reference, dict):
            raise ValueError(f"Final-policy source receipt requires {name}")
        path = _final_policy_manifest_path(reference.get("path"), f"{name} path")
        payload, digest = _read_final_policy_json(path)
        if digest != reference.get("sha256"):
            raise ValueError(f"Final-policy source receipt {name} hash does not match")
        manifests[name] = payload
        manifest_provenance[name] = {"path": str(path), "sha256": digest}
    deployment, launch = manifests["deployment_manifest"], manifests["launch_manifest"]
    if _final_policy_manifest_path(deployment.get("remote_root"), "deployment remote_root") != source_root:
        raise ValueError("Final-policy source receipt deployment root differs")
    environment = launch.get("common_env", {})
    if not isinstance(environment, dict):
        raise ValueError("Final-policy retained launch common_env is malformed")
    environment = dict(environment)
    entry_path = receipt["launch_manifest"].get("entry_path")
    if not isinstance(entry_path, list) or not entry_path:
        raise ValueError("Final-policy source receipt requires a launch entry_path")
    entry = launch
    try:
        for key in entry_path:
            if isinstance(entry, dict):
                env = entry.get("env", {})
                if not isinstance(env, dict) or not isinstance(key, str):
                    raise ValueError("Malformed launch entry environment or key")
                environment.update(env)
            elif not isinstance(entry, list) or isinstance(key, bool) or not isinstance(key, int) or key < 0:
                raise ValueError("Malformed launch entry index")
            entry = entry[key]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Final-policy source receipt launch entry_path is invalid") from exc
    if not isinstance(entry, dict) or not isinstance(entry.get("env", {}), dict):
        raise ValueError("Final-policy retained launch entry is malformed")
    environment.update(entry.get("env", {}))
    if (
        _final_policy_manifest_path(entry.get("run_dir"), "launch run directory") != source
        or entry.get("model") != config.get("model")
        or entry.get("step") != config.get("step")
        or entry.get("method") != "ace"
    ):
        raise ValueError("Final-policy retained launch entry does not match the source run/model/step")
    if _final_policy_manifest_path(environment.get("RLLM_DIR"), "launch RLLM_DIR") != source_root:
        raise ValueError("Final-policy retained launch RLLM_DIR differs from the frozen source root")
    for name in ("repo", "repo_dir"):
        if config.get(name) and _final_policy_manifest_path(config[name], f"recorded {name}") != source_root:
            raise ValueError("Final-policy recorded source root disagrees with its retained launch")
    # C1 and C4 manifests may retain the same old entries. The recorded CLI is
    # independent corroboration of which source/concurrency launch applied.
    concurrency_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    concurrency_parser.add_argument("--task_concurrency", type=int, default=1)
    try:
        recorded_concurrency, _ = concurrency_parser.parse_known_args(command[3:])
        launch_concurrency = int(environment.get("TASK_CONCURRENCY", "1"))
    except (SystemExit, TypeError, ValueError) as exc:
        raise ValueError("Final-policy retained launch task concurrency is malformed") from exc
    if launch_concurrency < 1 or recorded_concurrency.task_concurrency != launch_concurrency:
        raise ValueError("Final-policy retained launch task concurrency disagrees with the recorded source command")
    source_files = deployment.get("source_files")
    if not isinstance(source_files, dict):
        raise ValueError("Final-policy retained deployment requires source_files hashes")
    verified_files = {}

    def verified_file(relative: str) -> bytes:
        path = source_root / relative
        record = source_files.get(relative)
        if source_root not in path.resolve().parents or not isinstance(record, dict):
            raise ValueError(f"Final-policy retained deployment lacks asset/resolver proof: {relative}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Cannot read final-policy frozen source file: {relative}") from exc
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record.get("sha256"):
            raise ValueError(f"Final-policy frozen source hash differs from deployment: {relative}")
        verified_files[relative] = digest
        return raw

    evaluator = verified_file("examples/appworld/eval/ace/evaluate.py")
    utilities = verified_file("examples/appworld/eval/ace/appworld_ace_utils.py")
    _verify_appworld_default_resolver(evaluator, utilities)
    assets = source_root / "examples/appworld/train/ace_rl"
    prompts = assets / "prompts"
    initial = assets / "playbooks/initial_playbook.txt"
    base_prompt = getattr(source_args, "base_system_prompt_filename", "base_system_prompt.txt")
    if not isinstance(base_prompt, str) or not base_prompt or Path(base_prompt).name != base_prompt:
        raise ValueError("Final-policy source receipt requires a simple base prompt filename")
    for path in (prompts / base_prompt, prompts / "reflector_prompt.txt", prompts / "curator_prompt.txt", initial):
        verified_file(str(path.relative_to(source_root)))
    # The protocol hashes the entire prompt directory, so its complete file set
    # must have retained deployment evidence as well.
    for path in sorted(prompts.rglob("*")):
        if path.is_file():
            verified_file(str(path.relative_to(source_root)))
    resolved_defaults = {}
    for name, path in (("prompts_dir", prompts), ("initial_playbook", initial)):
        if not getattr(source_args, name, None):
            setattr(source_args, name, str(path))
            resolved_defaults[name] = str(path)
    return {
        "receipt_path": str(receipt_file),
        "receipt_sha256": receipt_hash,
        "binding": "retained_launch_entry_and_recorded_task_concurrency",
        "source_root": str(source_root),
        **manifest_provenance,
        "launch_entry_path": entry_path,
        "recorded_task_concurrency": recorded_concurrency.task_concurrency,
        "resolved_default_assets": resolved_defaults,
        "verified_source_files": verified_files,
    }


def _validate_final_policy_outputs(task_artifact_root: str | None, task_ids: list[str], step: int, visit: int) -> None:
    for task_id in task_ids:
        for leaf, filename in (("chat_completions", f"policy_step{step:06d}_iter_{visit - 1}.json"), ("playbooks", f"playbook_step{step:04d}_iter_{visit - 1}.txt")):
            directory = task_artifact_dir(task_artifact_root, task_id, leaf)
            if directory and (Path(directory) / "final_policy" / filename).exists():
                raise ValueError("Final-policy evaluation refuses to overwrite an existing final acting artifact")


def _final_policy_receipt_path(root: str, task_id: str, step: int, visit: int) -> Path:
    return Path(task_artifact_dir(root, task_id, "completed_final_policy")) / f"step{step:06d}_iter_{visit - 1}.json"


def _load_final_policy_receipt(root: str, task_id: str, step: int, visit: int, playbook: str) -> tuple[dict, list] | None:
    path = _final_policy_receipt_path(root, task_id, step, visit)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    row = saved.get("row", {})
    if saved.get("version") != 1 or saved.get("task_id") != task_id or saved.get("step") != step or saved.get("visit") != visit or row.get("task_id") != task_id or row.get("playbook_sha256") != hashlib.sha256(playbook.encode()).hexdigest():
        raise ValueError(f"Final-policy task receipt differs from this evaluation: {path}")
    artifacts = row["artifacts"]
    for name in ("playbook_snapshot", "policy_chat_completions"):
        artifact = Path(artifacts[name])
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != saved["artifact_sha256"][name]:
            raise ValueError(f"Final-policy task artifact changed or is missing: {artifact}")
    if _strip_playbook_snapshot_header(Path(artifacts["playbook_snapshot"]).read_text()) != playbook:
        raise ValueError(f"Final-policy playbook changed: {path}")
    from rllm.agents.agent import Trajectory

    policy = json.loads(Path(artifacts["policy_chat_completions"]).read_text())
    trajectories = [Trajectory.from_dict(item) for item in policy["trajectories"]]
    if policy.get("step") != step or policy.get("num_trajectories") != len(trajectories):
        raise ValueError(f"Final-policy trajectories differ from their receipt: {path}")
    return row, trajectories


def _save_final_policy_receipt(root: str, task_id: str, step: int, visit: int, row: dict) -> None:
    path = _final_policy_receipt_path(root, task_id, step, visit)
    hashes = {name: hashlib.sha256(Path(artifact).read_bytes()).hexdigest() for name, artifact in row["artifacts"].items()}
    payload = {"version": 1, "task_id": task_id, "step": step, "visit": visit, "row": row, "artifact_sha256": hashes}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _recover_final_policy_artifacts(root: str, task_id: str, step: int, visit: int, playbook: str, args: argparse.Namespace, hooks: ACEExperimentHooks) -> tuple[dict, list] | None:
    suffix = f"_iter_{visit - 1}"
    artifacts = {
        "playbook_snapshot": str(Path(task_artifact_dir(root, task_id, "playbooks")) / "final_policy" / f"playbook_step{step:04d}{suffix}.txt"),
        "policy_chat_completions": str(Path(task_artifact_dir(root, task_id, "chat_completions")) / "final_policy" / f"policy_step{step:06d}{suffix}.json"),
    }
    if not all(Path(path).is_file() for path in artifacts.values()):
        return None
    try:
        if _strip_playbook_snapshot_header(Path(artifacts["playbook_snapshot"]).read_text()) != playbook:
            return None
        policy = json.loads(Path(artifacts["policy_chat_completions"]).read_text())
        from rllm.agents.agent import Trajectory

        trajectories = [Trajectory.from_dict(item) for item in policy["trajectories"]]
        if policy.get("step") != step or policy.get("mode") != "final_policy" or len(trajectories) != args.g_act or any(hooks.task_id_getter(trajectory.task) != task_id for trajectory in trajectories):
            return None
        row = {"task_id": task_id, "reward": summarize_trajectories(trajectories, hooks)["avg_reward"], "playbook_sha256": hashlib.sha256(playbook.encode()).hexdigest(), "artifacts": artifacts}
    except (KeyError, TypeError, ValueError):
        return None
    _save_final_policy_receipt(root, task_id, step, visit, row)
    return row, trajectories


def _load_final_policy_source(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    eval_tasks: list[dict],
    step: int,
    task_artifact_root: str | None,
    metrics_file: str | None,
) -> dict:
    """Validate completed adaptation before creating clients or modifying outputs.

    Historical evaluation snapshots serialize playbook TEXT, not controller/RNG
    state. That text is sufficient for this inference-only pass; this is not an
    exact sampled-request or training/controller continuation.
    """
    source = Path(args.final_policy_from_run_dir).resolve()
    if not task_artifact_root or not metrics_file:
        raise ValueError("Final-policy continuation requires separate artifact and metrics outputs")
    for output in (Path(task_artifact_root).resolve(), Path(metrics_file).resolve()):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Final-policy outputs must be separate from the source run (no parent/child overlap)")
    if Path(metrics_file).exists():
        raise ValueError("Final-policy continuation refuses an existing metrics output")
    if hooks.parse_evaluation_args is None:
        raise ValueError("Final-policy continuation requires the benchmark parse_evaluation_args hook")
    marker, marker_hash = _read_final_policy_json(source / "evaluation_complete.json")
    config, config_hash = _read_final_policy_json(source / "evaluation_config.json")
    command = marker.get("evaluation")
    if command != config.get("evaluation"):
        raise ValueError("Final-policy source completion and configuration commands disagree")
    source_cli = _source_evaluation_arguments(command)
    try:
        source_args = hooks.parse_evaluation_args(source_cli)
        if "--local_eval_protocol" not in source_cli:
            source_args.local_eval_protocol = LEGACY_LOCAL_PROTOCOL
        normalize_eval_frequency(source_args)
    except (SystemExit, TypeError, ValueError) as exc:
        raise ValueError("Cannot resolve final-policy source evaluation protocol") from exc
    metrics_path = source / f"metrics_step{step}.json"
    payload, metrics_hash = _read_final_policy_json(metrics_path)
    if payload.get("local_eval_protocol", LEGACY_LOCAL_PROTOCOL) != LEGACY_LOCAL_PROTOCOL:
        raise ValueError("Legacy final-policy continuation cannot import sequential evaluation results")
    if marker.get("metrics_sha256") != metrics_hash:
        raise ValueError("Final-policy source metrics do not match the completion hash")
    source_receipt = None
    if getattr(args, "final_policy_source_manifest", None):
        source_receipt = _resolve_final_policy_source_receipt(
            receipt_path=args.final_policy_source_manifest,
            source=source,
            source_args=source_args,
            command=command,
            config=config,
            hashes={"source_configuration_sha256": config_hash, "source_completion_sha256": marker_hash, "source_metrics_sha256": metrics_hash},
            hooks=hooks,
        )
    if any(hasattr(source_args, name) and not getattr(source_args, name) for name in ("prompts_dir", "initial_playbook")):
        raise ValueError("Legacy final-policy continuation requires explicit source prompt and initial playbook paths")
    old_protocol, new_protocol = _final_policy_protocol(source_args), _final_policy_protocol(args)
    differences = sorted(key for key in old_protocol.keys() | new_protocol.keys() if old_protocol.get(key) != new_protocol.get(key))
    if differences:
        raise ValueError(f"Final-policy source protocol differs: {', '.join(differences)}")
    # Source epochs must describe a complete run from the initial playbook.
    # Extending an already-extended/final-scored run needs separate provenance.
    forbidden = {"--warm_start_from_run_dir", "--start_epoch", "--final_policy_from_run_dir", "--final_policy_eval"}
    if any(token.split("=", 1)[0] in forbidden for token in command):
        raise ValueError("Final-policy source must be an original completed adaptation run")
    if payload.get("model_name") != args.model_name or config.get("model") != args.model_name:
        raise ValueError("Final-policy source model does not match")
    if payload.get("experiment_name") != hooks.experiment_name:
        raise ValueError("Final-policy source benchmark does not match")
    if config.get("step") != step or Path(config.get("metrics_file", "")).resolve() != metrics_path:
        raise ValueError("Final-policy source step or metrics path does not match")
    count = args.local_eval_epochs
    if payload.get("local_eval_epochs") != count:
        raise ValueError("Final-policy source run metadata has incompatible adaptation epochs")
    if config.get("expected_epochs") != count or config.get("start_epoch") != 0:
        raise ValueError("Final-policy source configuration has incompatible adaptation epochs")
    entries = payload.get("evaluations")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError("Final-policy source must contain exactly one completed evaluation")
    entry = entries[0]
    if entry.get("step") != step or entry.get("local_eval_epochs") != count or entry.get("final_policy_evaluation") is not None:
        raise ValueError("Final-policy source evaluation step/epochs differ or final pass already exists")
    if not isinstance(entry.get("metrics"), dict):
        raise ValueError("Final-policy source has malformed final metrics")
    if any(key.startswith("final_policy/") for key in entry["metrics"]):
        raise ValueError("Final-policy source already contains a final policy score")
    epochs = entry.get("local_epochs", [])
    if not isinstance(epochs, list) or not all(isinstance(item, dict) for item in epochs):
        raise ValueError("Final-policy source has malformed adaptation epochs")
    if [item.get("local_eval_epoch") for item in epochs] != list(range(1, count + 1)):
        raise ValueError("Final-policy source has incomplete adaptation epochs")
    task_ids = [hooks.task_id_getter(task) for task in eval_tasks]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Final-policy continuation requires nonempty unique task IDs")
    solved = set()
    for local_epoch in epochs:
        rows = local_epoch.get("tasks", [])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Final-policy source has malformed task rows")
        if [row.get("task_id") for row in rows] != task_ids:
            raise ValueError("Final-policy source task set/order is incomplete or incompatible")
        padded = []
        for row in rows:
            reward = row.get("reward")
            if isinstance(reward, bool) or not isinstance(reward, int | float) or not math.isfinite(reward):
                raise ValueError("Final-policy source contains a nonfinite acting reward")
            padded.append(freeze_solved_reward(solved, row["task_id"], reward))
        epoch_metrics = local_epoch.get("metrics", {})
        if not isinstance(epoch_metrics, dict) or epoch_metrics.get("num_trajectories") != len(task_ids) * max(1, args.g_act):
            raise ValueError("Final-policy source has incomplete acting trajectories")
        for key, expected in (("avg_reward", float(np.mean([row["reward"] for row in rows]))), ("max_reward_mean", float(np.mean(padded)))):
            actual = epoch_metrics.get(key)
            if isinstance(actual, bool) or not isinstance(actual, int | float) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Final-policy source {key} disagrees with its acting rewards")
    if entry.get("tasks") != epochs[-1].get("tasks"):
        raise ValueError("Final-policy source final task rows disagree with its last epoch")
    if any(entry.get("metrics", {}).get(key) != value for key, value in epochs[-1].get("metrics", {}).items()):
        raise ValueError("Final-policy source final metrics disagree with its last epoch")
    playbooks, hashes = {}, {}
    for row in entry["tasks"]:
        task_id = row["task_id"]
        playbook_path = Path(task_artifact_dir(str(source), task_id, "playbooks")) / "eval" / f"playbook_step{step:04d}_iter_{count - 1}.txt"
        log_path = Path(task_artifact_dir(str(source), task_id, "ace_logs")) / "eval" / f"ace_step{step:06d}_iter_{count - 1}.json"
        for key, path in (("playbook_snapshot", playbook_path), ("ace_log", log_path)):
            if Path(row.get("artifacts", {}).get(key, "")).resolve() != path.resolve() or source not in path.resolve().parents:
                raise ValueError(f"Final-policy source has invalid {key} reference for task {task_id}")
        log, log_hash = _read_final_policy_json(log_path)
        if log.get("task_id") != task_id or log.get("step") != step or log.get("local_eval_epoch") != count or log.get("mode") != "eval":
            raise ValueError(f"Final-policy source ACE log metadata differs for task {task_id}")
        if log.get("metrics", {}).get("avg_reward") != row["reward"]:
            raise ValueError(f"Final-policy source ACE log reward differs for task {task_id}")
        try:
            raw = playbook_path.read_bytes()
            header, separator, content = raw.decode().partition("#" + "=" * 60 + "\n\n")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Invalid final-policy playbook for task {task_id}") from exc
        if (
            not separator
            or not header.startswith("# Local Playbook Snapshot\n")
            or any(line not in header.splitlines() for line in (f"# Task ID: {task_id}", f"# Step: {step}", "# Mode: eval", f"# Size: {len(content)} characters"))
        ):
            raise ValueError(f"Invalid final-policy playbook header for task {task_id}")
        if log.get("playbook", {}).get("after") != content or row.get("playbook_size") != len(content):
            raise ValueError(f"Final-policy saved playbook disagrees with its adaptation log for task {task_id}")
        playbook = Playbook(content)
        if playbook.to_string() != content:
            raise ValueError(f"Final-policy playbook cannot be restored without changing rendered text for task {task_id}")
        playbooks[task_id] = playbook
        hashes[task_id] = {"playbook_sha256": hashlib.sha256(raw).hexdigest(), "ace_log_sha256": log_hash}
    return {
        "evaluation": entry,
        "playbooks": playbooks,
        "solved": solved,
        "provenance": {
            "source_run_dir": str(source),
            "source_metrics_sha256": metrics_hash,
            "source_completion_sha256": marker_hash,
            "source_configuration_sha256": config_hash,
            "source_artifact_hashes": hashes,
            "resolved_protocol": new_protocol,
            "default_asset_hashes_recorded_by_legacy_source": False,
            "source_asset_receipt": source_receipt,
            "state_restoration": "final_playbook_text_for_inference_only",
            "exact_rng_continuation": False,
        },
    }


async def _append_final_policy_evaluation(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    runner: ContextAwareRunner,
    context_rule: Any,
    eval_tasks: list[dict],
    local_playbooks: dict[str, Playbook],
    solved: set[str],
    evaluation: dict,
    step: int,
    epoch: int | None,
    task_artifact_root: str | None,
    workers: list | None,
    provenance: dict,
) -> None:
    """Score every task on its final playbook without reflection or selection."""
    visit = evaluation["local_eval_epochs"] + 1
    results = [None] * len(eval_tasks)
    if not getattr(args, "resume", False):
        _validate_final_policy_outputs(task_artifact_root, [hooks.task_id_getter(task) for task in eval_tasks], step, visit)

    async def run_task(task_idx, task, worker):
        task_id = hooks.task_id_getter(task)
        worker.context_rule.playbook = local_playbooks[task_id]
        before = worker.context_rule.get_context()
        if task_artifact_root and getattr(args, "resume", False):
            saved = _load_final_policy_receipt(task_artifact_root, task_id, step, visit, before)
            if saved is None:
                saved = _recover_final_policy_artifacts(task_artifact_root, task_id, step, visit, before, args, hooks)
            if saved is not None:
                return saved
        trajectories = await worker.runner._process_batch([task] * max(1, args.g_act), task_idx, adapt=False)
        if len(trajectories) != max(1, args.g_act):
            raise RuntimeError(f"Final-policy acting returned an incomplete trajectory batch for task {task_id}")
        if worker.context_rule.get_context() != before:
            raise RuntimeError("Final-policy acting unexpectedly changed the playbook")
        metrics = summarize_trajectories(trajectories, hooks)
        artifacts = {}
        if task_artifact_root:
            artifacts["playbook_snapshot"] = save_local_playbook_snapshot(
                before,
                task_artifact_root,
                task_id,
                step,
                mode="final_policy",
                local_eval_epoch=visit,
            )
            artifacts["policy_chat_completions"] = save_policy_chat_completions(
                trajectories,
                task_artifact_dir(task_artifact_root, task_id, "chat_completions"),
                step,
                mode="final_policy",
                epoch=epoch,
                batch_idx=task_idx + 1,
                label=f"final_policy_after_adaptation_{visit - 1}",
                local_eval_epoch=visit,
            )
        row = {"task_id": task_id, "reward": metrics["avg_reward"], "playbook_sha256": hashlib.sha256(before.encode()).hexdigest(), "artifacts": artifacts}
        if task_artifact_root:
            _save_final_policy_receipt(task_artifact_root, task_id, step, visit, row)
        return row, trajectories

    pending = iter(enumerate(eval_tasks))

    async def run_worker(worker):
        for task_idx, task in pending:
            results[task_idx] = await run_task(task_idx, task, worker)

    active = [asyncio.create_task(run_worker(worker)) for worker in (workers or [_LocalEvaluationWorker(args, runner, context_rule)])]
    try:
        await asyncio.gather(*active)
    except BaseException:
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        raise
    rows = [row for row, _ in results]
    trajectories = [trajectory for _, items in results for trajectory in items]
    final_metrics = summarize_trajectories(trajectories, hooks)
    final_metrics["pass_at_k_success_rate"] = compute_pass_at_k_success_rate(trajectories, hooks)
    final_metrics["pass_at_k_max_reward"] = compute_pass_at_k_max_reward(trajectories, hooks)
    cumulative_solved = set(solved)
    rewards = [freeze_solved_reward(cumulative_solved, row["task_id"], row["reward"]) for row in rows]
    final_metrics["max_reward_mean"] = float(np.mean(rewards)) if rewards else 0.0
    final_metrics["n_solved"] = len(cumulative_solved)
    # Resumed adaptation can omit tasks solved before its first recorded epoch.
    # Keep the full selected set explicit for final-pass completion and plotting.
    evaluation["evaluation_task_ids"] = [hooks.task_id_getter(task) for task in eval_tasks]
    evaluation["final_policy_evaluation"] = {
        "kind": "final_policy_only",
        "after_local_eval_epoch": visit - 1,
        "policy_visit": visit,
        "adaptation_applied": False,
        "metrics": final_metrics,
        "tasks": rows,
        "work": {"policy_rollouts": len(trajectories), "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0},
        "provenance": provenance,
    }
    evaluation["metrics"].update({f"final_policy/{key}": value for key, value in final_metrics.items()})


def playbook_bullet_count(playbook: Playbook) -> int:
    stats = playbook.get_stats() if hasattr(playbook, "get_stats") else {}
    return int(stats.get("total_bullets", 0))


def reflector_only_candidate(context_rule: Any, guidebook_insights: list[dict]) -> dict:
    """Build one candidate per guidebook, appending each trajectory's insight.

    The ablation bypasses Curator sampling, so g_cur does not duplicate these
    deterministic candidates. Existing candidate scoring/selection still applies.
    """
    operations_list, logs = context_rule._reflections_to_operations([item["reflection"] for item in guidebook_insights])
    result = {"operations": [operation for item in operations_list for operation in item["operations"]]}
    return {
        "result": result,
        "log": {**empty_ace_log(skipped=True), "reason": "reflector_only", "parsed": result, "reflections": logs},
    }


def concatenate_insights(insights: list[dict], include_reasoning: bool = False, training_input_parity: bool = False) -> str:
    if training_input_parity:
        return concatenate_training_insights(insights)
    merged = []
    for idx, insight in enumerate(insights):
        reflection = insight.get("reflection") or {}
        if not reflection:
            continue

        task_id = insight.get("task_id")
        trajectory_idx = insight.get("trajectory_idx")
        header_bits = []
        if task_id is not None:
            header_bits.append(f"Task {task_id}")
        if trajectory_idx is not None:
            header_bits.append(f"Trajectory {trajectory_idx}")
        header = ", ".join(header_bits) if header_bits else "Item"
        merged.append(f"--- Insight {idx + 1} ({header}) ---")
        merged.append(f"Outcome: {reflection.get('outcome', 'UNKNOWN')}")

        reasoning = reflection.get("reasoning")
        if include_reasoning and reasoning:
            merged.append(f"Reasoning: {reasoning}")

        key_insight = reflection.get("key_insight")
        if key_insight:
            merged.append(f"Key Insight: {key_insight}")
        else:
            merged.append("Key Insight: (not provided)")

        failure_reason = reflection.get("failure_reason")
        if include_reasoning and failure_reason:
            merged.append(f"Failure Reason: {failure_reason}")

        correct_approach = reflection.get("correct_approach")
        if correct_approach:
            merged.append(f"Correct Approach: {correct_approach}")

        merged.append("")
    return "\n".join(merged)


def group_trajectories_by_task(trajectories: list, batch_size: int) -> list[list]:
    groups = [[] for _ in range(batch_size)]
    for idx, trajectory in enumerate(trajectories):
        groups[idx % batch_size].append(trajectory)
    return groups


def build_runner(args: argparse.Namespace, hooks: ACEExperimentHooks, prompts_dir: str, context_rule: Any) -> ContextAwareRunner:
    return ContextAwareRunner(
        engine=hooks.build_generator_engine(args),
        context_rule=context_rule,
        render_in_agent=hooks.render_prompt_in_agent,
        empty_context_text="(No strategies learned yet)",
        base_system_prompt=os.path.join(prompts_dir, "base_system_prompt.txt"),
        batch_size=max(1, getattr(args, "train_batch_size", 1) * max(1, args.g_act)),
        verbose=True,
    )


async def run_global_validation(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    runner: ContextAwareRunner,
    eval_tasks: list[dict],
    step: int,
    epoch: int | None,
    chat_completion_dir: str | None,
    ace_log_dir: str | None,
    label: str,
    wandb_run: Any,
) -> dict:
    chunk_size = max(1, int(args.val_batch_size or args.test_batch_size or args.train_batch_size))
    all_trajectories = []

    for batch_idx, batch_start in enumerate(range(0, len(eval_tasks), chunk_size)):
        batch = eval_tasks[batch_start : batch_start + chunk_size]
        trajectories = await runner._process_batch(batch * max(1, args.g_act), batch_idx, adapt=False)
        all_trajectories.extend(trajectories)

    metrics = summarize_trajectories(all_trajectories, hooks)
    metrics["pass_at_k_success_rate"] = compute_pass_at_k_success_rate(all_trajectories, hooks)
    metrics["pass_at_k_max_reward"] = compute_pass_at_k_max_reward(all_trajectories, hooks)
    step_rewards_list = [trajectory_step_rewards(traj) for traj in all_trajectories]
    turn_rows, turn_summary = build_turn_rows(step_rewards_list)
    metrics.update({f"turn/{key}": value for key, value in turn_summary.items()})

    chat_path = None
    if chat_completion_dir:
        chat_path = save_policy_chat_completions(
            all_trajectories,
            chat_completion_dir,
            step,
            mode="eval",
            epoch=epoch,
            label=label,
        )

    ace_path = None
    if ace_log_dir:
        ace_path = save_ace_step_log(
            ace_log_dir,
            step,
            {
                "label": label,
                "epoch": epoch,
                "metrics": metrics,
                "artifacts": {
                    "policy_chat_completions": chat_path,
                },
            },
            mode="eval",
        )

    if wandb_run:
        import wandb

        payload = {
            "val/success_rate": metrics["success_rate"],
            "val/pass_at_k_success_rate": metrics["pass_at_k_success_rate"],
            "val/avg_reward": metrics["avg_reward"],
            "val/avg_steps": metrics["avg_steps"],
            "val/pass_at_k_max_reward": metrics["pass_at_k_max_reward"],
        }
        payload.update({f"val/{key}": value for key, value in turn_summary.items()})
        wandb.log(payload, step=step)
        log_turn_table(
            step,
            turn_rows,
            {
                "pass_at_k_max_reward": metrics["pass_at_k_max_reward"],
            },
        )

    return {
        "step": step,
        "epoch": epoch,
        "label": label,
        "metrics": metrics,
        "artifacts": {
            "policy_chat_completions": chat_path,
            "ace_log": ace_path,
        },
    }


async def run_global_ace(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    train_tasks: list[dict],
    eval_tasks: list[dict] | None,
    prompts_dir: str,
    initial_playbook_path: str,
    playbook_snapshot_dir: str | None,
    chat_completion_dir: str | None,
    ace_log_dir: str | None,
    metrics_file: str | None,
    script_path: str,
) -> GlobalRunMetrics:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_steps = math.ceil(len(train_tasks) / max(1, args.train_batch_size)) * args.epochs if train_tasks else 0
    metrics = GlobalRunMetrics(
        run_id=run_id,
        experiment_name=hooks.experiment_name,
        model_name=args.model_name,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        total_train_tasks=len(train_tasks),
        total_steps=total_steps,
    )

    wandb_run = init_wandb_run(
        args,
        script_path,
        {
            "experiment": hooks.experiment_name,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "val_batch_size": getattr(args, "val_batch_size", None),
            "test_batch_size": getattr(args, "test_batch_size", None),
            "g_act": args.g_act,
            "g_ref": args.g_ref,
            "g_cur": args.g_cur,
            "selection_mode": args.selection_mode,
            "test_epoch_freq": args.test_epoch_freq,
        },
    )

    context_rule = hooks.create_context_rule(args, prompts_dir, initial_playbook_path, ace_log_dir)
    runner = build_runner(args, hooks, prompts_dir, context_rule)
    reflector = context_rule.reflector
    curator = context_rule.curator
    reflector_only = getattr(getattr(context_rule, "config", None), "reflector_only", False)
    playbook_tokenizer = maybe_create_playbook_tokenizer(args)

    global_step = 0
    if args.val_before_train and eval_tasks:
        metrics.evaluations.append(
            await run_global_validation(
                args=args,
                hooks=hooks,
                runner=runner,
                eval_tasks=eval_tasks,
                step=0,
                epoch=0,
                chat_completion_dir=chat_completion_dir,
                ace_log_dir=ace_log_dir,
                label="pre_training",
                wandb_run=wandb_run,
            )
        )
        if metrics_file:
            metrics.save(metrics_file)

    for epoch in range(1, args.epochs + 1):
        epoch_tasks = list(train_tasks)
        random.shuffle(epoch_tasks)

        for batch_idx, batch_start in enumerate(range(0, len(epoch_tasks), args.train_batch_size)):
            global_step += 1
            batch_tasks = epoch_tasks[batch_start : batch_start + args.train_batch_size]
            playbook_before = context_rule.get_context()

            act_trajectories = await runner._process_batch(
                batch_tasks * max(1, args.g_act),
                global_step,
                adapt=False,
            )
            current_metrics = summarize_trajectories(act_trajectories, hooks)
            train_step_rewards = [trajectory_step_rewards(traj) for traj in act_trajectories]
            train_turn_rows, _ = build_turn_rows(train_step_rewards)
            train_turn_tasks = [batch_tasks[idx % len(batch_tasks)] for idx in range(len(act_trajectories))] if batch_tasks else None
            train_pass_at_k_max_reward = compute_pass_at_k_max_reward_from_step_rewards(
                train_step_rewards,
                tasks=train_turn_tasks,
            )
            grouped_trajectories = group_trajectories_by_task(act_trajectories, len(batch_tasks))

            policy_chat_path = None
            if chat_completion_dir:
                policy_chat_path = save_policy_chat_completions(
                    act_trajectories,
                    chat_completion_dir,
                    global_step,
                    mode="train",
                    epoch=epoch,
                    batch_idx=batch_idx + 1,
                )

            trajectory_payloads = []
            reflections_by_task = []
            reflect_tasks = []
            reflect_meta = []
            reflector_logs = []
            for task_idx, task in enumerate(batch_tasks):
                task_trajectories = grouped_trajectories[task_idx]
                task_payloads = []
                task_reflections = []
                for traj_idx, trajectory in enumerate(task_trajectories):
                    payload = {
                        "trajectory": trajectory,
                        "reward": hooks.trajectory_reward(trajectory),
                        "termination_reason": getattr(trajectory, "termination_reason", None),
                        "max_turns": getattr(args, "max_turns", None),  # include max_turns in payload for use in reflection reasoning or curator operations
                    }
                    if getattr(args, "training_input_parity", False):
                        payload["training_task"] = task
                    task_payloads.append(payload)
                    task_reflections.append([{} for _ in range(args.g_ref)])
                    for guidebook_idx in range(args.g_ref):
                        reflect_tasks.append(reflector.analyze(payload, playbook_before))
                        reflect_meta.append((task_idx, traj_idx, guidebook_idx))
                trajectory_payloads.append(task_payloads)
                reflections_by_task.append(task_reflections)

            raw_reflections = await asyncio.gather(*reflect_tasks, return_exceptions=True) if reflect_tasks else []
            for reflection_idx, reflection_result in enumerate(raw_reflections):
                if isinstance(reflection_result, Exception):
                    parsed = {}
                    reflector_logs.append(empty_ace_log(error=str(reflection_result)))
                else:
                    parsed = reflection_result.get("result", {})
                    reflector_logs.append(reflection_result.get("log", empty_ace_log()))
                task_idx, traj_idx, guidebook_idx = reflect_meta[reflection_idx]
                reflections_by_task[task_idx][traj_idx][guidebook_idx] = parsed

            candidate_playbooks = []
            candidate_records = []
            scored_candidate_records = []
            curator_tasks = []
            curator_results = []
            curator_meta = []
            for task_idx, task in enumerate(batch_tasks):
                task_id = hooks.task_id_getter(task)
                task_payloads = trajectory_payloads[task_idx]
                for guidebook_idx in range(args.g_ref):
                    guidebook_insights = []
                    for traj_idx, payload in enumerate(task_payloads):
                        guidebook_insights.append(
                            {
                                "task_id": task_id,
                                "task_idx": task_idx,
                                "trajectory_idx": traj_idx,
                                "reflection": reflections_by_task[task_idx][traj_idx][guidebook_idx],
                                "trajectory": payload["trajectory"],
                                "reward": payload["reward"],
                            }
                        )

                    merged_insights = concatenate_insights(guidebook_insights, training_input_parity=bool(getattr(args, "training_input_parity", False)))
                    if not merged_insights.strip():
                        continue

                    aggregated = {
                        "trajectory": task_payloads[0]["trajectory"] if task_payloads else None,
                        "reward": float(np.mean([payload["reward"] for payload in task_payloads])) if task_payloads else 0.0,
                        "info": {},
                    }
                    if reflector_only:
                        curator_results.append(reflector_only_candidate(context_rule, guidebook_insights))
                        curator_meta.append({"task_id": task_id, "guidebook_idx": guidebook_idx})
                    else:
                        for _ in range(args.g_cur):
                            curator_tasks.append(curator.generate_operations(aggregated, merged_insights, playbook_before))
                            curator_meta.append({"task_id": task_id, "guidebook_idx": guidebook_idx})

            if curator_tasks:
                curator_results.extend(await asyncio.gather(*curator_tasks, return_exceptions=True))
            for result_idx, curator_result in enumerate(curator_results):
                meta = curator_meta[result_idx]
                if isinstance(curator_result, Exception):
                    candidate_records.append(
                        {
                            **meta,
                            "log": empty_ace_log(error=str(curator_result)),
                            "operations": {},
                            "valid_candidate": False,
                        }
                    )
                    continue

                operations = curator_result.get("result", {})
                curator_log = curator_result.get("log", empty_ace_log())
                record = {
                    **meta,
                    "log": curator_log,
                    "operations": operations,
                    "valid_candidate": bool(operations and operations.get("operations")),
                }
                if record["valid_candidate"]:
                    candidate = Playbook(playbook_before)
                    candidate.apply_operations(operations)
                    if playbook_tokenizer is not None:
                        candidate_len = get_playbook_token_length(candidate.to_string(), playbook_tokenizer)
                        if candidate_len > args.max_playbook_length:
                            record["valid_candidate"] = False
                            record["reject_reason"] = "max_playbook_length"
                            record["candidate_playbook"] = candidate.to_string()
                            candidate_records.append(record)
                            continue
                    candidate_playbooks.append(candidate)
                    record["candidate_index"] = len(candidate_playbooks) - 1
                    record["candidate_playbook"] = candidate.to_string()
                    scored_candidate_records.append(record)
                candidate_records.append(record)

            selected_reward = current_metrics["avg_reward"]
            best_candidate_reward = current_metrics["avg_reward"]
            selected_idx = -1
            reject_candidate = False
            candidate_rewards = []
            if candidate_playbooks:
                for candidate in candidate_playbooks:
                    context_rule.playbook = candidate
                    eval_trajectories = await runner._process_batch(
                        batch_tasks * max(1, args.g_act),
                        global_step,
                        adapt=False,
                    )
                    candidate_rewards.append(summarize_trajectories(eval_trajectories, hooks)["avg_reward"])

                best_candidate_reward = float(np.max(candidate_rewards)) if candidate_rewards else current_metrics["avg_reward"]
                reject_candidate = args.selection_mode == "best_rejection" and best_candidate_reward < current_metrics["avg_reward"]
                if not reject_candidate and candidate_rewards:
                    if args.selection_mode in {"best", "best_rejection"}:
                        selected_idx = int(np.argmax(candidate_rewards))
                    else:
                        selected_idx = random.randint(0, len(candidate_playbooks) - 1)
                    selected_reward = candidate_rewards[selected_idx]
                    context_rule.playbook = candidate_playbooks[selected_idx]
                else:
                    context_rule.playbook = Playbook(playbook_before)
            else:
                context_rule.playbook = Playbook(playbook_before)

            playbook_after = context_rule.get_context()
            playbook_obj = context_rule.playbook
            playbook_size = len(playbook_after)
            playbook_bullets = playbook_bullet_count(playbook_obj)
            playbook_snapshot_path = None
            if playbook_snapshot_dir:
                playbook_snapshot_path = save_global_playbook_snapshot(playbook_after, playbook_snapshot_dir, global_step)

            for record, reward in zip(scored_candidate_records, candidate_rewards, strict=False):
                record["reward"] = reward

            step_payload = {
                "epoch": epoch,
                "batch_idx": batch_idx + 1,
                "step": global_step,
                "metrics": {
                    **current_metrics,
                    "pass_at_k_max_reward": train_pass_at_k_max_reward,
                    "selected_reward": selected_reward,
                    "best_candidate_reward": best_candidate_reward,
                    "reject_candidate": reject_candidate,
                    "num_candidates": len(candidate_playbooks),
                    "playbook_size": playbook_size,
                    "playbook_bullets": playbook_bullets,
                },
                "artifacts": {
                    "policy_chat_completions": policy_chat_path,
                    "playbook_snapshot": playbook_snapshot_path,
                },
                "playbook": {
                    "before": playbook_before,
                    "after": playbook_after,
                },
                "reflector": {
                    "valid": sum(1 for task_reflections in reflections_by_task for traj_reflections in task_reflections for item in traj_reflections if item),
                    "logs": reflector_logs,
                },
                "curator": {
                    "attempted": len(curator_tasks),
                    "candidates": candidate_records,
                },
                "selection": {
                    "mode": args.selection_mode,
                    "selected_idx": selected_idx,
                    "candidate_rewards": candidate_rewards,
                },
            }

            ace_log_path = None
            if ace_log_dir:
                ace_log_path = save_ace_step_log(ace_log_dir, global_step, step_payload, mode="train")

            metrics.train_steps.append(
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx + 1,
                    "step": global_step,
                    "metrics": step_payload["metrics"],
                    "artifacts": {
                        **step_payload["artifacts"],
                        "ace_log": ace_log_path,
                    },
                }
            )

            if wandb_run:
                import wandb

                wandb.log(
                    {
                        "train/success_rate": current_metrics["success_rate"],
                        "train/avg_reward": current_metrics["avg_reward"],
                        "train/avg_steps": current_metrics["avg_steps"],
                        "train/pass_at_k_max_reward": train_pass_at_k_max_reward,
                        "ace/current_reward": current_metrics["avg_reward"],
                        "ace/selected_reward": selected_reward,
                        "ace/best_candidate_reward": best_candidate_reward,
                        "ace/reject_candidate": int(reject_candidate),
                        "ace/candidates": len(candidate_playbooks),
                        "playbook/size": playbook_size,
                        "playbook/bullets": playbook_bullets,
                        "playbook/growth": playbook_size - len(playbook_before),
                    },
                    step=global_step,
                )
                log_turn_table(
                    global_step,
                    train_turn_rows,
                    {
                        "pass_at_k_max_reward": train_pass_at_k_max_reward,
                    },
                    metric_prefix="train",
                )

            if metrics_file:
                metrics.save(metrics_file)

        if eval_tasks and args.test_epoch_freq > 0 and epoch % args.test_epoch_freq == 0:
            metrics.evaluations.append(
                await run_global_validation(
                    args=args,
                    hooks=hooks,
                    runner=runner,
                    eval_tasks=eval_tasks,
                    step=global_step,
                    epoch=epoch,
                    chat_completion_dir=chat_completion_dir,
                    ace_log_dir=ace_log_dir,
                    label=f"epoch_{epoch}",
                    wandb_run=wandb_run,
                )
            )
            if metrics_file:
                metrics.save(metrics_file)

    save_playbook_path = getattr(args, "save_playbook", None)
    if save_playbook_path:
        os.makedirs(os.path.dirname(save_playbook_path) or ".", exist_ok=True)
        with open(save_playbook_path, "w") as f:
            f.write(context_rule.get_context())

    if wandb_run:
        import wandb

        wandb.finish()

    if metrics_file:
        metrics.save(metrics_file)
    return metrics


@dataclass
class _LocalEvaluationWorker:
    args: argparse.Namespace
    runner: ContextAwareRunner
    context_rule: Any


async def _close_local_evaluation_resources(engine: Any, context_rule: Any) -> None:
    """Drain environment threads before closing this worker's isolated resources."""

    async def finish(call, *, in_thread=False):
        try:
            result = await asyncio.to_thread(call) if in_thread else call()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("Error closing local evaluation resource: %s", exc)

    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        await finish(shutdown, in_thread=not inspect.iscoroutinefunction(shutdown))
    for env in getattr(engine, "envs", ()) or ():
        if env is not None and callable(getattr(env, "close", None)):
            await finish(env.close, in_thread=not inspect.iscoroutinefunction(env.close))
    clients = []
    role_engines = [getattr(engine, "rollout_engine", None)]
    for name in ("reflector", "curator"):
        role = getattr(context_rule, name, None)
        role_engines.append(getattr(role, "rollout_engine", None))
    for role_engine in role_engines:
        client = getattr(role_engine, "client", None)
        if client is not None and all(client is not other for other in clients):
            clients.append(client)
    for client in clients:
        close = getattr(client, "close", None)
        if close is not None:
            await finish(close)


async def _run_local_validation_task(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    runner: ContextAwareRunner,
    context_rule: Any,
    task_idx: int,
    task: dict,
    local_playbooks: dict[str, Playbook],
    local_eval_epoch: int,
    total_local_eval_epochs: int,
    step: int,
    epoch: int | None,
    task_artifact_root: str | None,
    ace_log_dir: str | None,
    playbook_tokenizer: Any,
) -> dict:
    sequential = local_evaluation_protocol(args) == SEQUENTIAL_LOCAL_PROTOCOL
    # Each visit owns one measured policy batch. Only nonfinal visits produce
    # an update, which is consumed by the next visit without rescoring/replay.
    adapt = not sequential or local_eval_epoch < total_local_eval_epochs
    g_ref = args.g_ref if adapt else 0
    reflector = context_rule.reflector
    curator = context_rule.curator
    reflector_only = getattr(getattr(context_rule, "config", None), "reflector_only", False)

    task_id = hooks.task_id_getter(task)
    context_rule.playbook = local_playbooks[task_id]
    playbook_before = local_playbooks[task_id].to_string()

    seed_config = provider_seed_config(args)
    policy_tasks = [task] * max(1, args.g_act)
    if seed_config is not None:
        policy_tasks = [{**task, TASK_SEED_KEY: provider_seed_identity(seed_config, task_id, (local_eval_epoch - 1) * args.g_act + sample)} for sample in range(args.g_act)]
    act_trajectories = await runner._process_batch(policy_tasks, task_idx, adapt=False)
    if sequential and len(act_trajectories) != args.g_act:
        raise RuntimeError(f"Incomplete trajectory batch for task {task_id}: expected {args.g_act}, received {len(act_trajectories)}")
    current_metrics = summarize_trajectories(act_trajectories, hooks)
    samples = []
    if sequential:
        for sample_index, trajectory in enumerate(act_trajectories):
            if str(hooks.task_id_getter(trajectory.task)) != str(task_id):
                raise RuntimeError(f"Trajectory task identity differs from requested task {task_id}")
            reward = float(hooks.trajectory_reward(trajectory))
            if not math.isfinite(reward):
                raise ValueError(f"Nonfinite policy reward for task {task_id}")
            sample = {
                "sample_index": sample_index,
                "attempt_id": f"{task_id}:visit_{local_eval_epoch}:sample_{sample_index}",
                "reward": reward,
                "success": bool(hooks.trajectory_success(trajectory)),
                "num_steps": len(getattr(trajectory, "steps", []) or []),
            }
            samples.append(sample)
            trajectory.info = {**(getattr(trajectory, "info", None) or {}), "evaluation_sample": dict(sample)}

    reflect_tasks = []
    reflect_meta = []
    task_payloads = []
    reflections_by_traj = []
    for traj_idx, trajectory in enumerate(act_trajectories):
        payload = {
            "trajectory": trajectory,
            "reward": hooks.trajectory_reward(trajectory),
            "termination_reason": getattr(trajectory, "termination_reason", None),
            "max_turns": getattr(args, "max_turns", None),  # include max_turns in payload for use in reflection reasoning or curator operations
        }
        if getattr(args, "training_input_parity", False):
            payload["training_task"] = task
        task_payloads.append(payload)
        reflections_by_traj.append([{} for _ in range(g_ref)])
        for guidebook_idx in range(g_ref):
            reflect_tasks.append(with_provider_seed(
                reflector.analyze(payload, playbook_before),
                provider_seed_identity(seed_config, task_id, local_eval_epoch - 1, "reflector", traj_idx * max(1, g_ref) + guidebook_idx),
            ))
            reflect_meta.append((traj_idx, guidebook_idx))

    raw_reflections = await asyncio.gather(*reflect_tasks, return_exceptions=True) if reflect_tasks else []
    reflector_logs = []
    for reflection_idx, reflection_result in enumerate(raw_reflections):
        if isinstance(reflection_result, Exception):
            parsed = {}
            reflector_logs.append(empty_ace_log(error=str(reflection_result)))
        else:
            parsed = reflection_result.get("result", {})
            reflector_logs.append(reflection_result.get("log", empty_ace_log()))
        traj_idx, guidebook_idx = reflect_meta[reflection_idx]
        reflections_by_traj[traj_idx][guidebook_idx] = parsed

    candidate_playbooks = []
    candidate_records = []
    scored_candidate_records = []
    curator_tasks = []
    curator_results = []
    task_guidebooks = 0
    for guidebook_idx in range(g_ref):
        guidebook_insights = []
        for traj_idx, payload in enumerate(task_payloads):
            guidebook_insights.append(
                {
                    "task_id": task_id,
                    "task_idx": 0,  # The local evaluator acts on one task per batch.
                    "trajectory_idx": traj_idx,
                    "reflection": reflections_by_traj[traj_idx][guidebook_idx],
                    "trajectory": payload["trajectory"],
                    "reward": payload["reward"],
                }
            )

        merged_insights = concatenate_insights(guidebook_insights, training_input_parity=bool(getattr(args, "training_input_parity", False)))
        if not merged_insights.strip():
            continue

        task_guidebooks += 1
        aggregated = {
            "trajectory": task_payloads[0]["trajectory"] if task_payloads else None,
            "reward": float(np.mean([payload["reward"] for payload in task_payloads])) if task_payloads else 0.0,
            "info": {},
        }
        if reflector_only:
            curator_results.append(reflector_only_candidate(context_rule, guidebook_insights))
        else:
            for curator_idx in range(args.g_cur):
                curator_tasks.append(with_provider_seed(
                    curator.generate_operations(aggregated, merged_insights, playbook_before),
                    provider_seed_identity(seed_config, task_id, local_eval_epoch - 1, "curator", guidebook_idx * args.g_cur + curator_idx),
                ))

    if curator_tasks:
        curator_results.extend(await asyncio.gather(*curator_tasks, return_exceptions=True))
    for curator_result in curator_results:
        if isinstance(curator_result, Exception):
            candidate_records.append(
                {
                    "log": empty_ace_log(error=str(curator_result)),
                    "operations": {},
                    "valid_candidate": False,
                }
            )
            continue

        operations = curator_result.get("result", {})
        curator_log = curator_result.get("log", empty_ace_log())
        record = {
            "log": curator_log,
            "operations": operations,
            "valid_candidate": bool(isinstance(operations, dict) and operations.get("operations")),
        }
        if record["valid_candidate"]:
            candidate = Playbook(playbook_before)
            if sequential:
                try:
                    applied = candidate.apply_operations(operations)
                except (AttributeError, TypeError, ValueError) as exc:
                    record.update(valid_candidate=False, reject_reason="invalid_delta", error=str(exc))
                    candidate_records.append(record)
                    continue
                if not applied or candidate.to_string() == playbook_before:
                    record.update(valid_candidate=False, reject_reason="no_op_delta")
                    candidate_records.append(record)
                    continue
            else:
                candidate.apply_operations(operations)
            if playbook_tokenizer is not None:
                candidate_len = get_playbook_token_length(candidate.to_string(), playbook_tokenizer)
                if candidate_len > args.max_playbook_length:
                    record["valid_candidate"] = False
                    record["reject_reason"] = "max_playbook_length"
                    record["candidate_playbook"] = candidate.to_string()
                    candidate_records.append(record)
                    continue
            candidate_playbooks.append(candidate)
            record["candidate_index"] = len(candidate_playbooks) - 1
            record["candidate_playbook"] = candidate.to_string()
            scored_candidate_records.append(record)
        candidate_records.append(record)

    selected_reward = current_metrics["avg_reward"]
    best_candidate_reward = current_metrics["avg_reward"]
    selected_idx = -1
    reject_candidate = False
    candidate_rewards = []
    if sequential and candidate_playbooks:
        # Sequential ACE applies its single valid delta independently of the
        # observed reward. The next visit is the only updated-policy rollout.
        if len(candidate_playbooks) != 1:
            raise ValueError("Sequential ACE requires a single curator delta")
        if getattr(args, "adaptation_mode", "apply") == "apply":
            selected_idx = 0
            local_playbooks[task_id] = candidate_playbooks[0]
    elif candidate_playbooks:
        for candidate in candidate_playbooks:
            context_rule.playbook = candidate
            eval_trajectories = await runner._process_batch([task] * max(1, args.g_act), task_idx, adapt=False)
            candidate_rewards.append(summarize_trajectories(eval_trajectories, hooks)["avg_reward"])

        best_candidate_reward = float(np.max(candidate_rewards)) if candidate_rewards else current_metrics["avg_reward"]
        reject_candidate = args.selection_mode == "best_rejection" and best_candidate_reward < current_metrics["avg_reward"]
        if not reject_candidate and candidate_rewards:
            if args.selection_mode in {"best", "best_rejection"}:
                selected_idx = int(np.argmax(candidate_rewards))
            else:
                selected_idx = random.randint(0, len(candidate_playbooks) - 1)
            selected_reward = candidate_rewards[selected_idx]
            local_playbooks[task_id] = candidate_playbooks[selected_idx]

    context_rule.playbook = local_playbooks[task_id]
    playbook_after = local_playbooks[task_id].to_string()
    playbook_size = len(playbook_after)
    playbook_bullet_total = playbook_bullet_count(local_playbooks[task_id])
    playbook_growth = playbook_size - len(playbook_before)

    control_config = handoff_evaluation_config(args)
    control_receipt = {}
    if control_config:
        control_receipt = {
            "handoff_control": control_config,
            **provider_seed_receipt(seed_config),
            "adaptation_generated": bool(candidate_playbooks),
            "adaptation_changed": playbook_after != playbook_before,
        }
        if seed_config is not None:
            control_receipt["provider_role_work"] = _provider_role_work(control_receipt["provider_requests"], {"reflector": reflector_logs, "curator": [record["log"] for record in candidate_records] if curator_tasks else []})
        for record in candidate_records:
            record["applied"] = selected_idx >= 0 and record.get("candidate_index") == selected_idx
            record["discarded_by_control"] = bool(record.get("valid_candidate") and getattr(args, "adaptation_mode", "apply") == "discard")

    task_chat_dir = task_artifact_dir(task_artifact_root, task_id, "chat_completions")
    policy_chat_path = None
    if task_chat_dir:
        policy_chat_path = save_policy_chat_completions(
            act_trajectories,
            task_chat_dir,
            step,
            mode="eval",
            epoch=epoch,
            batch_idx=task_idx + 1,
            label=f"local_eval_epoch_{local_eval_epoch}",
            local_eval_epoch=local_eval_epoch,
        )

    playbook_snapshot_path = None
    if task_artifact_root:
        playbook_snapshot_path = save_local_playbook_snapshot(
            playbook_after,
            task_artifact_root,
            task_id,
            step,
            local_eval_epoch=local_eval_epoch,
            mode="eval",
        )

    for record, reward in zip(scored_candidate_records, candidate_rewards, strict=False):
        record["reward"] = reward

    task_ace_dir = task_artifact_dir(task_artifact_root, task_id, "ace_logs") if task_artifact_root else ace_log_dir
    ace_log_path = None
    if task_ace_dir:
        ace_log_path = save_ace_step_log(
            task_ace_dir,
            step,
            {
                "local_eval_protocol": local_evaluation_protocol(args),
                **control_receipt,
                "epoch": epoch,
                "local_eval_epoch": local_eval_epoch,
                "local_eval_epochs": total_local_eval_epochs,
                "task_id": task_id,
                "task_idx": task_idx,
                "metrics": {
                    **current_metrics,
                    "selected_reward": selected_reward,
                    "best_candidate_reward": best_candidate_reward,
                    "reject_candidate": reject_candidate,
                    "num_candidates": len(candidate_playbooks),
                    "playbook_size": playbook_size,
                    "playbook_bullets": playbook_bullet_total,
                    "playbook_growth": playbook_growth,
                },
                "artifacts": {
                    "policy_chat_completions": policy_chat_path,
                    "playbook_snapshot": playbook_snapshot_path,
                },
                "playbook": {
                    "before": playbook_before,
                    "after": playbook_after,
                },
                "reflector": {
                    "valid": sum(1 for traj_reflections in reflections_by_traj for item in traj_reflections if item),
                    "logs": reflector_logs,
                },
                "curator": {
                    "attempted": len(curator_tasks),
                    "candidates": candidate_records,
                },
                "selection": {
                    "mode": ("discard_delta" if getattr(args, "adaptation_mode", "apply") == "discard" else "apply_delta") if sequential else args.selection_mode,
                    "selected_idx": selected_idx,
                    "candidate_rewards": candidate_rewards,
                },
            },
            mode="eval",
            local_eval_epoch=local_eval_epoch,
        )
    else:
        ace_log_path = None

    task_row = {
        **control_receipt,
        "local_eval_epoch": local_eval_epoch,
        "task_id": task_id,
        "reward": current_metrics["avg_reward"],
        "selected_reward": selected_reward,
        "best_candidate_reward": best_candidate_reward,
        "reject_candidate": int(reject_candidate),
        "num_candidates": len(candidate_playbooks),
        "playbook_size": playbook_size,
        "playbook_bullets": playbook_bullet_total,
        "playbook_growth": playbook_growth,
        "artifacts": {
            "policy_chat_completions": policy_chat_path,
            "playbook_snapshot": playbook_snapshot_path,
            "ace_log": ace_log_path,
        },
    }

    if sequential:
        task_row.update(
            samples=samples,
            policy_playbook=playbook_before,
            adaptation_applied=selected_idx >= 0,
            adaptation_scheduled=bool(adapt and g_ref),
        )
    result = {
        "row": task_row,
        "trajectories": act_trajectories,
        "reflector_rollouts": len(raw_reflections),
        "curator_rollouts": len(curator_tasks),
        "guidebooks": task_guidebooks,
    }
    if task_artifact_root and policy_chat_path and playbook_snapshot_path and ace_log_path:
        _save_local_task_receipt(task_artifact_root, task_id, step, local_eval_epoch, result, playbook_before, playbook_after)
    return result


async def _run_single_local_validation_epoch(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    runner: ContextAwareRunner,
    context_rule: Any,
    eval_tasks: list[dict],
    local_playbooks: dict[str, Playbook],
    local_eval_epoch: int,
    total_local_eval_epochs: int,
    step: int,
    epoch: int | None,
    task_artifact_root: str | None,
    ace_log_dir: str | None,
    playbook_tokenizer: Any,
    workers: list[_LocalEvaluationWorker] | None = None,
) -> dict:
    stage_rows = []
    all_trajectories = []
    playbook_sizes = []
    playbook_bullets = []
    playbook_growths = []
    selected_rewards = []
    best_candidate_rewards = []
    reject_candidates = []
    total_reflector_rollouts = 0
    total_curator_rollouts = 0
    total_guidebooks = 0
    total_candidates = 0

    async def run_task(task_idx, task, worker):
        task_id = hooks.task_id_getter(task)
        if task_artifact_root and getattr(args, "resume", False):
            saved = _load_local_task_receipt(
                task_artifact_root, task_id, step, local_eval_epoch,
                local_playbooks[task_id].to_string(),
            )
            if saved is None and _recover_local_task_artifacts(
                task_artifact_root, task_id, task_idx, step, local_eval_epoch,
                total_local_eval_epochs, local_playbooks[task_id].to_string(), args, hooks,
            ) is not None:
                saved = _load_local_task_receipt(
                    task_artifact_root, task_id, step, local_eval_epoch,
                    local_playbooks[task_id].to_string(),
                )
            if saved is not None:
                if saved["row"].get("handoff_control", {}) != handoff_evaluation_config(args):
                    raise ValueError("Completed task visit has a different handoff control configuration")
                local_playbooks[task_id] = Playbook(saved.pop("playbook_after"))
                return saved
        with record_provider_seeds():
            return await _run_local_validation_task(
                args=worker.args,
                hooks=hooks,
                runner=worker.runner,
                context_rule=worker.context_rule,
                task_idx=task_idx,
                task=task,
                local_playbooks=local_playbooks,
                local_eval_epoch=local_eval_epoch,
                total_local_eval_epochs=total_local_eval_epochs,
                step=step,
                epoch=epoch,
                task_artifact_root=task_artifact_root,
                ace_log_dir=ace_log_dir,
                playbook_tokenizer=playbook_tokenizer,
            )

    if workers:
        # Each worker owns one whole task cycle. The shared iterator is consumed
        # synchronously, and results retain input order despite completion order.
        results = [None] * len(eval_tasks)
        pending_tasks = iter(enumerate(eval_tasks))

        async def run_worker(worker):
            for task_idx, task in pending_tasks:
                results[task_idx] = await run_task(task_idx, task, worker)

        active = [asyncio.create_task(run_worker(worker)) for worker in workers]
        try:
            await asyncio.gather(*active)
        except BaseException:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            raise
    else:
        worker = _LocalEvaluationWorker(args, runner, context_rule)
        results = [await run_task(task_idx, task, worker) for task_idx, task in enumerate(eval_tasks)]

    for result in results:
        row = result["row"]
        stage_rows.append(row)
        all_trajectories.extend(result["trajectories"])
        playbook_sizes.append(row["playbook_size"])
        playbook_bullets.append(row["playbook_bullets"])
        playbook_growths.append(row["playbook_growth"])
        selected_rewards.append(row["selected_reward"])
        best_candidate_rewards.append(row["best_candidate_reward"])
        reject_candidates.append(row["reject_candidate"])
        total_reflector_rollouts += result["reflector_rollouts"]
        total_curator_rollouts += result["curator_rollouts"]
        total_guidebooks += result["guidebooks"]
        total_candidates += row["num_candidates"]

    stage_metrics = summarize_trajectories(all_trajectories, hooks)
    stage_metrics["pass_at_k_success_rate"] = compute_pass_at_k_success_rate(all_trajectories, hooks)
    stage_metrics["pass_at_k_max_reward"] = compute_pass_at_k_max_reward(all_trajectories, hooks)
    turn_rows, turn_summary = build_turn_rows(
        [trajectory_step_rewards(traj) for traj in all_trajectories],
        max_steps=getattr(args, "max_turns", None),
    )
    stage_metrics.update({f"turn/{key}": value for key, value in turn_summary.items()})
    stage_metrics["avg_playbook_size"] = _mean_metric(playbook_sizes)
    stage_metrics["avg_playbook_bullets"] = _mean_metric(playbook_bullets)
    stage_metrics["ace/policy_rollouts"] = len(all_trajectories)
    stage_metrics["ace/reflector_rollouts"] = total_reflector_rollouts
    stage_metrics["ace/curator_rollouts"] = total_curator_rollouts
    stage_metrics["ace/total_rollouts"] = len(all_trajectories) + total_reflector_rollouts + total_curator_rollouts
    stage_metrics["ace/candidates"] = total_candidates
    stage_metrics["ace/guidebooks"] = total_guidebooks
    stage_metrics["ace/current_reward"] = stage_metrics["avg_reward"]
    stage_metrics["ace/selected_reward"] = _mean_metric(selected_rewards)
    stage_metrics["ace/best_candidate_reward"] = _mean_metric(best_candidate_rewards)
    stage_metrics["ace/reject_candidate"] = _mean_metric(reject_candidates)
    stage_metrics["playbook/size"] = stage_metrics["avg_playbook_size"]
    stage_metrics["playbook/bullets"] = stage_metrics["avg_playbook_bullets"]
    stage_metrics["playbook/growth"] = _mean_metric(playbook_growths)

    table_rows = _build_local_eval_table_rows(
        turn_rows,
        local_eval_epoch=local_eval_epoch,
        local_eval_epochs=total_local_eval_epochs,
        epoch_metrics=stage_metrics,
        epoch_turn_offset=(local_eval_epoch - 1) * len(turn_rows),
    )

    return {
        "local_eval_epoch": local_eval_epoch,
        "metrics": stage_metrics,
        "tasks": stage_rows,
        "turn_rows": turn_rows,
        "turn_summary": turn_summary,
        "table_rows": table_rows,
    }


def _validate_sequential_local_args(args: argparse.Namespace) -> None:
    for name in ("local_eval_epochs", "g_act"):
        value = getattr(args, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("g_ref", "g_cur"):
        value = getattr(args, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise ValueError(f"Sequential ACE requires {name} to be 0 or 1; branching search requires legacy_search_v1")
    capacity = getattr(args, "n_parallel_agents", None)
    if capacity is not None and capacity < args.g_act:
        raise ValueError("Parallel pass@k requires n_parallel_agents >= g_act")
    reflector_only = bool(getattr(args, "reflector_only", False))
    if args.g_ref == 0 and args.g_cur != 0:
        raise ValueError("Vanilla sequential evaluation requires g_ref=g_cur=0")
    if args.g_ref and not args.g_cur and not reflector_only:
        raise ValueError("Sequential ACE requires g_ref=g_cur=1, or the explicit reflector_only ablation")
    if getattr(args, "freeze_solved", False):
        raise ValueError("Sequential evaluation requires freeze_solved=False so every measured visit is retained")
    if any(getattr(args, name, None) for name in ("warm_start_from_run_dir", "start_epoch", "final_policy_from_run_dir", "final_policy_source_manifest")):
        raise ValueError("Sequential evaluation cannot import legacy warm-start/final-policy state; start a fresh output or explicitly select legacy_search_v1")


def _sequential_sample_metrics(rows: list[dict], g_act: int) -> dict:
    """Unbiased pass@r estimates from all k retained samples, for every r<=k."""
    metrics = {}
    for r in range(1, g_act + 1):
        estimates = []
        for row in rows:
            successes = sum(sample["success"] for sample in row["samples"])
            failures = g_act - successes
            estimates.append(1.0 - (math.comb(failures, r) / math.comb(g_act, r) if failures >= r else 0.0))
        metrics[f"pass_at_{r}_success_rate"] = _mean_metric(estimates)
    metrics["pass_at_k_success_rate"] = metrics[f"pass_at_{g_act}_success_rate"]
    return metrics


async def _run_sequential_local_validation_stage(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    runner: ContextAwareRunner,
    context_rule: Any,
    eval_tasks: list[dict],
    step: int,
    epoch: int | None,
    task_artifact_root: str | None,
    ace_log_dir: str | None,
    metrics: LocalRunMetrics,
    wandb_run: Any,
    workers: list[_LocalEvaluationWorker] | None,
) -> dict:
    _validate_sequential_local_args(args)
    task_ids = [hooks.task_id_getter(task) for task in eval_tasks]
    local_playbooks = {task_id: Playbook(context_rule.get_context()) for task_id in task_ids}
    tokenizer = maybe_create_playbook_tokenizer(args)
    solved = set()
    local_epochs, table_rows = [], []
    counts = {"policy": 0, "reflector": 0, "curator": 0, "candidate": 0}
    for visit in range(1, args.local_eval_epochs + 1):
        result = await _run_single_local_validation_epoch(
            args=args, hooks=hooks, runner=runner, context_rule=context_rule,
            eval_tasks=eval_tasks, local_playbooks=local_playbooks,
            local_eval_epoch=visit, total_local_eval_epochs=args.local_eval_epochs,
            step=step, epoch=epoch, task_artifact_root=task_artifact_root,
            ace_log_dir=ace_log_dir, playbook_tokenizer=tokenizer, workers=workers,
        )
        visit_metrics = result["metrics"]
        visit_metrics.update(_sequential_sample_metrics(result["tasks"], args.g_act))
        for row in result["tasks"]:
            if any(sample["success"] for sample in row["samples"]):
                solved.add(row["task_id"])
            row["cumulative_solved"] = row["task_id"] in solved
        visit_metrics["n_active_tasks"] = len(eval_tasks)
        visit_metrics["n_solved"] = len(solved)
        visit_metrics["cumulative_success_rate"] = len(solved) / len(eval_tasks) if eval_tasks else 0.0
        # The sequential protocol reports current measured quality here.
        # Success across preceding visits is only cumulative_success_rate.
        visit_metrics["max_reward_mean"] = visit_metrics["pass_at_k_max_reward"]
        visit_metrics["ace/candidate_policy_rollouts"] = 0
        for role in ("policy", "reflector", "curator"):
            counts[role] += visit_metrics[f"ace/{role}_rollouts"]
        local_epochs.append({"local_eval_epoch": visit, "metrics": visit_metrics, "tasks": result["tasks"]})
        if provider_seed_config(args) is not None:
            local_epochs[-1]["provider_usage"] = provider_usage_totals([request for row in result["tasks"] for request in row["provider_requests"]])
        table_rows.extend(result["table_rows"])
    counts["total"] = sum(counts.values())
    last = local_epochs[-1]
    final_metrics = dict(last["metrics"])
    evaluation = {
        "step": step,
        "epoch": epoch,
        "local_eval_protocol": SEQUENTIAL_LOCAL_PROTOCOL,
        **({"handoff_control": handoff_evaluation_config(args)} if handoff_evaluation_config(args) else {}),
        "g_act": args.g_act,
        "policy_visits": args.local_eval_epochs,
        "adaptation_updates": args.local_eval_epochs - 1 if args.g_ref else 0,
        "adaptation_update_slots_per_task": args.local_eval_epochs - 1 if args.g_ref else 0,
        "actually_applied_context_updates": sum(row["adaptation_applied"] for visit in local_epochs for row in visit["tasks"]),
        "evaluation_task_ids": task_ids,
        "rollout_counts": counts,
        "metrics": {**final_metrics, "local_eval_epochs": args.local_eval_epochs},
        "tasks": last["tasks"],
        "local_eval_epochs": args.local_eval_epochs,
        "local_epochs": local_epochs,
        "adaptation_evaluation": {
            "kind": "measured_adaptation_visits",
            "policy_visits": args.local_eval_epochs - 1,
            "local_epochs": local_epochs[:-1],
        },
        "final_policy_evaluation": {
            "kind": "last_measured_policy_visit",
            "policy_visit": args.local_eval_epochs,
            "after_local_eval_epoch": args.local_eval_epochs - 1,
            "adaptation_applied": False,
            "metrics": final_metrics,
            "tasks": last["tasks"],
            "work": {"policy_rollouts": len(eval_tasks) * args.g_act, "reflector_rollouts": 0, "curator_rollouts": 0, "candidate_rollouts": 0},
            "provenance": {"source": "same_run_measured_visit", "extra_policy_rollouts": 0},
        },
    }
    evaluation["metrics"].update({f"final_policy/{key}": value for key, value in final_metrics.items()})
    if provider_seed_config(args) is not None:
        evaluation["provider_usage"] = provider_usage_totals([request for visit in local_epochs for row in visit["tasks"] for request in row["provider_requests"]])
    if wandb_run:
        import wandb

        wandb.log({f"val/{key}": value for key, value in evaluation["metrics"].items()}, step=step)
        log_turn_table(step, table_rows)
    metrics.evaluations.append(evaluation)
    return evaluation


async def run_local_validation_stage(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    runner: ContextAwareRunner,
    context_rule: Any,
    eval_tasks: list[dict],
    step: int,
    epoch: int | None,
    task_artifact_root: str | None,
    ace_log_dir: str | None,
    metrics: LocalRunMetrics,
    wandb_run: Any,
    workers: list[_LocalEvaluationWorker] | None = None,
    final_policy_source: dict | None = None,
) -> dict:
    if local_evaluation_protocol(args) == SEQUENTIAL_LOCAL_PROTOCOL:
        if final_policy_source is not None:
            raise ValueError("Sequential evaluation cannot import legacy final-policy state")
        return await _run_sequential_local_validation_stage(
            args=args, hooks=hooks, runner=runner, context_rule=context_rule,
            eval_tasks=eval_tasks, step=step, epoch=epoch,
            task_artifact_root=task_artifact_root, ace_log_dir=ace_log_dir,
            metrics=metrics, wandb_run=wandb_run, workers=workers,
        )
    if final_policy_source is not None:
        evaluation = final_policy_source["evaluation"]
        await _append_final_policy_evaluation(
            args=args,
            hooks=hooks,
            runner=runner,
            context_rule=context_rule,
            eval_tasks=eval_tasks,
            local_playbooks=final_policy_source["playbooks"],
            solved=final_policy_source["solved"],
            evaluation=evaluation,
            step=step,
            epoch=epoch,
            task_artifact_root=task_artifact_root,
            workers=workers,
            provenance=final_policy_source["provenance"],
        )
        metrics.evaluations.append(evaluation)
        return evaluation
    playbook_tokenizer = maybe_create_playbook_tokenizer(args)
    initial_text = context_rule.get_context()
    local_playbooks = {hooks.task_id_getter(task): Playbook(initial_text) for task in eval_tasks}
    # Restore per-task playbooks and solved-task state for epoch extension.
    # args.warm_start_from_run_dir + args.start_epoch resume a finished run's
    # state: per-task playbooks from its iter_{start_epoch-1} snapshots and the
    # solved set from its ace logs. The loop below then runs GLOBAL epoch
    # numbers start_epoch+1 .. start_epoch+local_eval_epochs, so artifacts land
    # as _iter_{start_epoch}.. (no renumbering for the splicer). Both args unset
    # -> byte-identical behavior to before.
    warm_start_dir = getattr(args, "warm_start_from_run_dir", None)
    epoch_offset = max(0, int(getattr(args, "start_epoch", 0) or 0))
    warm_solved: set = set()
    if bool(warm_start_dir) != bool(epoch_offset):
        raise ValueError("--warm_start_from_run_dir and --start_epoch must be set together")
    if warm_start_dir:
        task_ids_all = [hooks.task_id_getter(task) for task in eval_tasks]
        loaded_texts, warm_solved, provenance = load_warm_start_state(
            warm_start_dir,
            task_ids_all,
            step,
            epoch_offset,
            mode="eval",
        )
        local_playbooks = {tid: Playbook(text) for tid, text in loaded_texts.items()}
        if task_artifact_root:
            os.makedirs(task_artifact_root, exist_ok=True)
            with open(os.path.join(task_artifact_root, "extended_from.json"), "w") as f:
                json.dump(provenance, f, indent=2)
        print(f"[WARM-START] {len(local_playbooks)} playbooks from {warm_start_dir} (start_epoch={epoch_offset}, solved_at_start={len(warm_solved)})", flush=True)
    total_local_eval_epochs = epoch_offset + max(1, int(getattr(args, "local_eval_epochs", 1) or 1))

    local_epoch_evaluations = []
    combined_table_rows = []
    final_epoch_result = None

    # Doyoung freeze-at-1.0 fix. `max_reward_mean` (solved tasks padded at 1.0,
    # not-yet-solved tasks left at their fresh per-iteration reward) is ALWAYS
    # computed — a free re-aggregation that matches the offline recovery. Solved
    # tasks lock at 1.0; unsolved tasks continue the evaluation with fresh rewards
    # (no running-max smoothing on partials). --freeze_solved additionally SKIPS
    # generating already-solved tasks (compute saving); the metric is identical
    # either way, since a solved task contributes 1.0 whether re-run or frozen.
    freeze_solved = bool(getattr(args, "freeze_solved", False))
    all_task_ids = [hooks.task_id_getter(task) for task in eval_tasks]
    solved: set[str] = set(warm_solved)

    for local_eval_epoch in range(epoch_offset + 1, total_local_eval_epochs + 1):
        if freeze_solved:
            active_tasks = [task for task in eval_tasks if hooks.task_id_getter(task) not in solved]
        else:
            active_tasks = eval_tasks

        epoch_result = await _run_single_local_validation_epoch(
            args=args,
            hooks=hooks,
            runner=runner,
            context_rule=context_rule,
            eval_tasks=active_tasks,
            local_playbooks=local_playbooks,
            local_eval_epoch=local_eval_epoch,
            total_local_eval_epochs=total_local_eval_epochs,
            step=step,
            epoch=epoch,
            task_artifact_root=task_artifact_root,
            ace_log_dir=ace_log_dir,
            playbook_tokenizer=playbook_tokenizer,
            workers=workers,
        )
        # This iteration's fresh acting rewards (only for tasks that ran).
        fresh_this_iter = {str(row["task_id"]): float(row["reward"]) for row in epoch_result["tasks"]}
        # max_reward_mean over ALL tasks: solved -> 1.0 (padded, incl. tasks frozen
        # this iter and prior); not-yet-solved -> fresh reward this iteration.
        vals = [freeze_solved_reward(solved, tid, fresh_this_iter.get(tid, 0.0)) for tid in all_task_ids]
        max_reward_mean = float(np.mean(vals)) if vals else 0.0
        epoch_result["metrics"]["max_reward_mean"] = max_reward_mean
        epoch_result["metrics"]["n_active_tasks"] = len(active_tasks)
        epoch_result["metrics"]["n_solved"] = len(solved)
        local_epoch_evaluations.append(epoch_result)
        combined_table_rows.extend(epoch_result["table_rows"])
        final_epoch_result = epoch_result

    if final_epoch_result is None:
        final_epoch_result = {
            "metrics": {},
            "tasks": [],
            "turn_summary": {},
        }

    final_stage_metrics = dict(final_epoch_result["metrics"])
    final_stage_metrics["local_eval_epochs"] = total_local_eval_epochs

    if wandb_run:
        import wandb

        wandb.log(
            {
                "val/success_rate": final_stage_metrics.get("success_rate", 0.0),
                "val/pass_at_k_success_rate": final_stage_metrics.get("pass_at_k_success_rate", 0.0),
                "val/avg_reward": final_stage_metrics.get("avg_reward", 0.0),
                "val/max_reward_mean": final_stage_metrics.get("max_reward_mean", 0.0),
                "val/avg_steps": final_stage_metrics.get("avg_steps", 0.0),
                "val/pass_at_k_max_reward": final_stage_metrics.get("pass_at_k_max_reward", 0.0),
                "val/local_ace_epoch": total_local_eval_epochs,
                "val/local_ace_eval_epochs": total_local_eval_epochs,
                **{f"val/{key}": value for key, value in final_epoch_result.get("turn_summary", {}).items()},
            },
            step=step,
        )
        log_turn_table(
            step,
            combined_table_rows,
        )

    evaluation = {
        "step": step,
        "epoch": epoch,
        "metrics": final_stage_metrics,
        "tasks": final_epoch_result["tasks"],
        "local_eval_epochs": total_local_eval_epochs,
        "local_epochs": [
            {
                "local_eval_epoch": result["local_eval_epoch"],
                "metrics": result["metrics"],
                "tasks": result["tasks"],
            }
            for result in local_epoch_evaluations
        ],
    }
    if getattr(args, "final_policy_eval", False):
        await _append_final_policy_evaluation(
            args=args,
            hooks=hooks,
            runner=runner,
            context_rule=context_rule,
            eval_tasks=eval_tasks,
            local_playbooks=local_playbooks,
            solved=solved,
            evaluation=evaluation,
            step=step,
            epoch=epoch,
            task_artifact_root=task_artifact_root,
            workers=workers,
            provenance={"source": "same_run_final_adaptation", "state_restoration": "in_memory_final_playbook", "exact_rng_continuation": False, "resolved_protocol": _final_policy_protocol(args)},
        )
    metrics.evaluations.append(evaluation)
    return evaluation


async def run_local_ace(
    *,
    args: argparse.Namespace,
    hooks: ACEExperimentHooks,
    train_tasks: list[dict],
    eval_tasks: list[dict],
    prompts_dir: str,
    initial_playbook_path: str,
    playbook_snapshot_dir: str | None,
    ace_log_dir: str | None,
    metrics_file: str | None,
    script_path: str,
    base_system_prompt_filename: str = "base_system_prompt.txt",
    base_system_prompt_template: str | None = None,
) -> LocalRunMetrics:
    protocol = local_evaluation_protocol(args)
    if protocol not in (SEQUENTIAL_LOCAL_PROTOCOL, LEGACY_LOCAL_PROTOCOL):
        raise ValueError(f"Unknown local evaluation protocol: {protocol}")
    sequential = protocol == SEQUENTIAL_LOCAL_PROTOCOL
    mode = getattr(args, "adaptation_mode", "apply")
    seed_config = provider_seed_config(args)
    handoff_protocol = getattr(args, "handoff_protocol", "legacy")
    if handoff_protocol not in {"legacy", "feedback_v1"}:
        raise ValueError("handoff_protocol must be legacy or feedback_v1")
    if mode not in ("apply", "discard"):
        raise ValueError("adaptation_mode must be apply or discard")
    if (mode != "apply" or seed_config is not None or handoff_protocol != "legacy") and not sequential:
        raise ValueError("Handoff controls and provider seeds require sequential_adaptation_v1")
    if mode == "discard" and (args.g_ref != 1 or args.g_cur != 1):
        raise ValueError("Discard control requires normal ACE reflection and curation")
    if seed_config is not None:
        validate_seed_schedule(seed_schedule(seed_config, [hooks.task_id_getter(task) for task in eval_tasks], args.local_eval_epochs * args.g_act, args.max_turns, auxiliary=bool(args.g_ref), group_size=args.g_act))
    if sequential:
        _validate_sequential_local_args(args)
        if not eval_tasks:
            raise ValueError("Sequential evaluation requires at least one task")
        task_ids = [hooks.task_id_getter(task) for task in eval_tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Sequential evaluation requires unique task IDs")
        if len({safe_task_artifact_id(task_id) for task_id in task_ids}) != len(task_ids):
            raise ValueError("Sequential evaluation requires unique task artifact paths")
        if metrics_file and Path(metrics_file).exists() and not getattr(args, "resume", False):
            raise ValueError("Sequential evaluation refuses an existing metrics output")
    final_policy = not sequential and getattr(args, "final_policy_eval", False)
    final_policy_from = getattr(args, "final_policy_from_run_dir", None)
    if getattr(args, "final_policy_source_manifest", None) and not final_policy_from:
        raise ValueError("final_policy_source_manifest requires final_policy_from_run_dir")
    if final_policy_from and not final_policy:
        raise ValueError("final_policy_from_run_dir requires final_policy_eval")
    if final_policy_from and (getattr(args, "warm_start_from_run_dir", None) or getattr(args, "start_epoch", 0)):
        raise ValueError("Final-policy continuation cannot be combined with adaptation warm start")
    if final_policy and (isinstance(args.local_eval_epochs, bool) or not isinstance(args.local_eval_epochs, int) or args.local_eval_epochs < 1):
        raise ValueError("Final-policy evaluation requires a positive integer adaptation visit count")
    if final_policy and not eval_tasks:
        raise ValueError("Final-policy evaluation requires at least one task")
    if final_policy and metrics_file and Path(metrics_file).exists() and not getattr(args, "resume", False):
        raise ValueError("Final-policy evaluation refuses an existing metrics output")
    task_concurrency = getattr(args, "task_concurrency", 1)
    if isinstance(task_concurrency, bool) or not isinstance(task_concurrency, int) or task_concurrency < 1:
        raise ValueError("task_concurrency must be a positive integer")
    if task_concurrency > 1:
        if hooks.prepare_local_worker_args is None:
            raise ValueError("Concurrent local evaluation requires prepare_local_worker_args")
        task_ids = [hooks.task_id_getter(task) for task in eval_tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Concurrent local evaluation requires unique task IDs")
        artifact_ids = [safe_task_artifact_id(task_id) for task_id in task_ids]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Concurrent local evaluation requires unique task artifact paths")

    task_artifact_root = resolve_local_artifact_root(playbook_snapshot_dir, ace_log_dir)
    step = getattr(args, "wandb_step", None) or 0
    final_policy_source = None
    if final_policy:
        task_ids = [hooks.task_id_getter(task) for task in eval_tasks]
        if len(set(task_ids)) != len(task_ids) or len({safe_task_artifact_id(task_id) for task_id in task_ids}) != len(task_ids):
            raise ValueError("Final-policy evaluation requires unique task IDs and artifact paths")
        final_visit = int(getattr(args, "start_epoch", 0) or 0) + args.local_eval_epochs + 1
        if not getattr(args, "resume", False):
            _validate_final_policy_outputs(task_artifact_root, task_ids, step, final_visit)
    if final_policy_from:
        final_policy_source = _load_final_policy_source(
            args=args,
            hooks=hooks,
            eval_tasks=eval_tasks,
            step=step,
            task_artifact_root=task_artifact_root,
            metrics_file=metrics_file,
        )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = LocalRunMetrics(
        run_id=run_id,
        experiment_name=hooks.experiment_name,
        model_name=args.model_name,
        local_eval_epochs=max(1, int(getattr(args, "local_eval_epochs", 1) or 1)),
        epochs=0,
        train_batch_size=0,
        total_train_tasks=0,
        steps_per_epoch=0,
        local_eval_protocol=protocol,
    )

    wandb_run = init_wandb_run(
        args,
        script_path,
        {
            "experiment": hooks.experiment_name,
            "local_eval_protocol": protocol,
            "model_name": args.model_name,
            "eval_tasks": len(eval_tasks),
            "local_eval_epochs": max(1, int(getattr(args, "local_eval_epochs", 1) or 1)),
            "g_act": args.g_act,
            "g_ref": args.g_ref,
            "g_cur": args.g_cur,
            "selection_mode": args.selection_mode,
        },
    )

    if base_system_prompt_template is not None:
        # Raw template string (e.g. the agent's built-in prompt for vanilla
        # evals); ContextAwareRunner accepts a file path or a raw template.
        base_system_prompt = base_system_prompt_template
        print("Using base system prompt: <raw template passed by the benchmark script>")
    else:
        base_system_prompt = os.path.join(prompts_dir, base_system_prompt_filename)
        print(f"Using base system prompt: {base_system_prompt_filename}")
    workers = []
    try:
        for worker_index in range(task_concurrency):
            worker_args = args if task_concurrency == 1 else hooks.prepare_local_worker_args(args, worker_index, task_concurrency)
            context_rule, engine = None, None
            try:
                context_rule = hooks.create_context_rule(worker_args, prompts_dir, initial_playbook_path, ace_log_dir=None)
                engine = hooks.build_generator_engine(worker_args)
                runner = ContextAwareRunner(
                    engine=engine,
                    context_rule=context_rule,
                    render_in_agent=hooks.render_prompt_in_agent,
                    empty_context_text="(No strategies learned yet)",
                    base_system_prompt=base_system_prompt,
                    batch_size=max(1, args.g_act),
                    verbose=True,
                )
            except BaseException:
                if task_concurrency > 1:
                    await _close_local_evaluation_resources(engine, context_rule)
                raise
            workers.append(_LocalEvaluationWorker(worker_args, runner, context_rule))
        runner, context_rule = workers[0].runner, workers[0].context_rule
        await run_local_validation_stage(
            args=args,
            hooks=hooks,
            runner=runner,
            context_rule=context_rule,
            eval_tasks=eval_tasks,
            step=step,
            epoch=0,
            task_artifact_root=task_artifact_root,
            ace_log_dir=ace_log_dir,
            metrics=metrics,
            wandb_run=wandb_run,
            workers=workers if task_concurrency > 1 else None,
            final_policy_source=final_policy_source,
        )
    finally:
        if task_concurrency > 1:
            cleanup = await asyncio.gather(*(_close_local_evaluation_resources(worker.runner.engine, worker.context_rule) for worker in workers), return_exceptions=True)
            for error in cleanup:
                if isinstance(error, BaseException):
                    logger.warning("Error closing local evaluation worker: %s", error)

    if metrics_file:
        metrics.save(metrics_file)

    if wandb_run:
        import wandb

        wandb.finish()

    if metrics_file:
        metrics.save(metrics_file)
    return metrics
