import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import torch


def normalize_validation_data_source(data_source: str | None) -> str:
    """Normalize validation data sources for metric grouping and logging."""
    if data_source is None:
        return ""

    normalized = str(data_source).strip().strip("/")
    if not normalized or normalized.lower() == "unknown":
        return ""
    return normalized


def get_validation_metric_prefix(data_source: str | None) -> str:
    """Return the metric prefix for validation metrics."""
    normalized_source = normalize_validation_data_source(data_source)
    return f"val/{normalized_source}" if normalized_source else "val"


def _get_task_key(task) -> str:
    """Return a stable grouping key for pass@k-style aggregation."""
    if isinstance(task, dict):
        task_str = json.dumps(task, sort_keys=True)
    else:
        task_str = str(task)
    return hashlib.md5(task_str.encode()).hexdigest()


def compute_pass_at_k_max_reward_from_step_rewards(
    step_rewards_list: list[list[float]],
    tasks: list | None = None,
) -> float:
    """Average the best trajectory max reward for each unique task."""
    if not step_rewards_list:
        return 0.0

    per_trajectory_max = [float(max(step_rewards)) if step_rewards else 0.0 for step_rewards in step_rewards_list]

    if tasks is not None and len(tasks) == len(per_trajectory_max):
        task_to_max_rewards: defaultdict[str, list[float]] = defaultdict(list)
        for task, traj_max in zip(tasks, per_trajectory_max, strict=False):
            task_to_max_rewards[_get_task_key(task)].append(traj_max)

        per_task_max = [max(rewards) for rewards in task_to_max_rewards.values()]
        return float(np.mean(per_task_max)) if per_task_max else 0.0

    return float(np.mean(per_trajectory_max)) if per_trajectory_max else 0.0


def compute_pass_at_k(results):
    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        task = trajectory.task

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = 1 if trajectory.reward > 0 else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    # Calculate pass@1 and pass@k
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = sum(1 for _, correct in problem_correct_map.items() if correct > 0) / total_problems

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)


def compute_average_reward(results):
    """Aggregate average and pass@k-style reward statistics over trajectories."""
    problem_reward_sum_map: defaultdict[str, float] = defaultdict(float)
    problem_max_reward_map: defaultdict[str, float] = defaultdict(lambda: float("-inf"))
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    for trajectory in results:
        task = trajectory.task

        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        problem_reward_sum_map[problem_hash] += trajectory.reward
        problem_max_reward_map[problem_hash] = max(problem_max_reward_map[problem_hash], trajectory.reward)
        problem_total_map[problem_hash] += 1

    total_problems = len(problem_reward_sum_map)
    total_reward = sum(problem_reward_sum_map.values())
    total_trajectories = sum(problem_total_map.values())
    average_reward = total_reward / total_trajectories if total_trajectories > 0 else 0.0
    average_reward_per_problem = sum(problem_reward_sum_map.values()) / total_problems if total_problems > 0 else 0.0
    pass_at_k_reward = sum(problem_max_reward_map.values()) / total_problems if total_problems > 0 else 0.0

    print("Total unique problems:", total_problems)
    print("Total trajectories:", total_trajectories)
    print("Average Reward (per trajectory):", average_reward)
    print("Average Reward (per problem):", average_reward_per_problem)
    print("Pass@k Reward (max reward per problem, averaged):", pass_at_k_reward)

    return {
        "total_unique_problems": total_problems,
        "total_trajectories": total_trajectories,
        "average_reward_per_trajectory": average_reward,
        "average_reward_per_problem": average_reward_per_problem,
        "pass_at_k_reward": pass_at_k_reward,
    }


def compute_turn_level_metrics(
    step_rewards_list: list[list[float]],
    logger=None,
    gamma: float = 1.0,
    step: int = 0,
    cutoff_mode: str = "zero",
    max_steps: int | None = None,
    prefix: str = "turn",
    include_per_turn_metrics: bool = False,
    table_extra_values: dict | None = None,
    table_prefix: str | None = None,
) -> dict:
    """
    Compute turn-level reward summaries across trajectories.

    Missing turns are extended according to `cutoff_mode` so different-length
    trajectories can be compared on the same turn axis.
    """
    if not step_rewards_list:
        return {}

    actual_max_turns = max(len(sr) for sr in step_rewards_list) if step_rewards_list else 0
    if actual_max_turns == 0:
        return {}

    max_turns = max_steps if max_steps is not None else actual_max_turns
    original_lengths = [len(sr) for sr in step_rewards_list]

    processed_rewards_list = []
    for step_rewards in step_rewards_list:
        if len(step_rewards) >= max_turns:
            processed_rewards_list.append(step_rewards[:max_turns])
            continue

        extended = list(step_rewards)
        if cutoff_mode == "zero":
            extended.extend([0.0] * (max_turns - len(step_rewards)))
        elif cutoff_mode == "max":
            fill_value = max(step_rewards) if step_rewards else 0.0
            extended.extend([fill_value] * (max_turns - len(step_rewards)))
        elif cutoff_mode == "last":
            fill_value = step_rewards[-1] if step_rewards else 0.0
            extended.extend([fill_value] * (max_turns - len(step_rewards)))
        else:
            raise ValueError(f"Unknown cutoff_mode: {cutoff_mode}. Must be 'zero', 'max', or 'last'.")
        processed_rewards_list.append(extended)

    turn_rewards = []
    for turn_idx in range(max_turns):
        turn_reward_values = [sr[turn_idx] for sr in processed_rewards_list]
        turn_rewards.append(np.mean(turn_reward_values))

    turn_counts = []
    for turn_idx in range(max_turns):
        alive_count = sum(1 for length in original_lengths if turn_idx < length)
        turn_counts.append(alive_count)

    cumulative_rewards = []
    for turn_idx in range(max_turns):
        cum_reward = 0.0
        discount = 1.0
        for past_turn in range(turn_idx + 1):
            cum_reward += turn_rewards[past_turn] * discount
            discount *= gamma
        cumulative_rewards.append(cum_reward)

    max_rewards = []
    for turn_idx in range(max_turns):
        trajectory_max_rewards = [max(step_rewards[: turn_idx + 1]) for step_rewards in processed_rewards_list]
        max_rewards.append(np.mean(trajectory_max_rewards))

    min_rewards = []
    for turn_idx in range(max_turns):
        trajectory_min_rewards = [min(step_rewards[: turn_idx + 1]) for step_rewards in processed_rewards_list]
        min_rewards.append(np.mean(trajectory_min_rewards))

    metrics_dict = {}
    if include_per_turn_metrics:
        for turn_idx in range(max_turns):
            metrics_dict[f"{prefix}/turn_{turn_idx}_avg_reward"] = turn_rewards[turn_idx]
            metrics_dict[f"{prefix}/turn_{turn_idx}_cumulative_reward"] = cumulative_rewards[turn_idx]
            metrics_dict[f"{prefix}/turn_{turn_idx}_max_reward"] = max_rewards[turn_idx]
            metrics_dict[f"{prefix}/turn_{turn_idx}_min_reward"] = min_rewards[turn_idx]

    if max_turns > 0:
        metrics_dict[f"{prefix}/final_avg_reward"] = turn_rewards[-1]
        metrics_dict[f"{prefix}/final_cumulative_reward"] = cumulative_rewards[-1]
        metrics_dict[f"{prefix}/final_max_reward"] = max_rewards[-1]
        metrics_dict[f"{prefix}/best_avg_reward"] = max(turn_rewards)
        metrics_dict[f"{prefix}/best_max_reward"] = max(max_rewards)

        per_trajectory_max = [max(sr) for sr in processed_rewards_list]
        metrics_dict[f"{prefix}/trajectory_max_reward_mean"] = np.mean(per_trajectory_max)
        metrics_dict[f"{prefix}/trajectory_max_reward_max"] = np.max(per_trajectory_max)
        metrics_dict[f"{prefix}/n_trajectories"] = len(processed_rewards_list)
        metrics_dict[f"{prefix}/max_turns"] = max_turns
        metrics_dict[f"{prefix}/cutoff_mode"] = cutoff_mode

    if logger is not None:
        wandb_logger = None
        try:
            if hasattr(logger, "logger") and isinstance(logger.logger, dict):
                wandb_logger = logger.logger.get("wandb")
        except Exception:
            pass

        if wandb_logger is None:
            try:
                import wandb

                if wandb.run is not None:
                    wandb_logger = wandb
            except Exception:
                pass

        if wandb_logger is not None:
            table_extra_values = table_extra_values or {}
            table_log_prefix = table_prefix or prefix
            table_data = []
            for turn_idx in range(max_turns):
                table_data.append(
                    [
                        turn_idx,
                        turn_rewards[turn_idx],
                        cumulative_rewards[turn_idx],
                        max_rewards[turn_idx],
                        min_rewards[turn_idx],
                        turn_counts[turn_idx],
                        *[table_extra_values[key] for key in table_extra_values],
                    ]
                )

            if table_data:
                import wandb

                turn_table = wandb.Table(
                    columns=[
                        "turn",
                        "avg_reward",
                        "cumulative_reward",
                        "max_reward",
                        "min_reward",
                        "n_trajectories",
                        *list(table_extra_values.keys()),
                    ],
                    data=table_data,
                )
                wandb_logger.log({f"{table_log_prefix}/turn_table": turn_table}, step=step)

    return metrics_dict


def compute_validation_turn_metrics(
    step_rewards_list: list[list[float]],
    tasks: list = None,
    cutoff_mode: str = "max",
    max_steps: int | None = None,
    data_source: str = "unknown",
) -> dict:
    """Compute validation turn-level metrics with pass@k-style aggregation."""
    if not step_rewards_list:
        return {}

    actual_max_turns = max(len(sr) for sr in step_rewards_list) if step_rewards_list else 0
    if actual_max_turns == 0:
        return {}

    max_turns = max_steps if max_steps is not None else actual_max_turns

    processed_rewards_list = []
    for step_rewards in step_rewards_list:
        if len(step_rewards) >= max_turns:
            processed_rewards_list.append(step_rewards[:max_turns])
            continue

        extended = list(step_rewards)
        if cutoff_mode == "zero":
            extended.extend([0.0] * (max_turns - len(step_rewards)))
        elif cutoff_mode == "max":
            fill_value = max(step_rewards) if step_rewards else 0.0
            extended.extend([fill_value] * (max_turns - len(step_rewards)))
        elif cutoff_mode == "last":
            fill_value = step_rewards[-1] if step_rewards else 0.0
            extended.extend([fill_value] * (max_turns - len(step_rewards)))
        else:
            raise ValueError(f"Unknown cutoff_mode: {cutoff_mode}")
        processed_rewards_list.append(extended)

    turn_rewards = []
    cumulative_rewards = []
    max_rewards = []

    for turn_idx in range(max_turns):
        turn_reward_values = [sr[turn_idx] for sr in processed_rewards_list]
        turn_rewards.append(np.mean(turn_reward_values))
        cumulative_rewards.append(sum(turn_rewards[: turn_idx + 1]))
        trajectory_max_rewards = [max(sr[: turn_idx + 1]) for sr in processed_rewards_list]
        max_rewards.append(np.mean(trajectory_max_rewards))

    per_trajectory_max = [max(sr) for sr in processed_rewards_list]
    success_rate = np.mean([traj_max >= 1.0 for traj_max in per_trajectory_max]) if per_trajectory_max else 0.0

    if tasks is not None and len(tasks) == len(processed_rewards_list):
        task_to_max_rewards: defaultdict[str, list[float]] = defaultdict(list)
        for task, traj_max in zip(tasks, per_trajectory_max, strict=False):
            task_to_max_rewards[_get_task_key(task)].append(traj_max)

        per_task_max = [max(rewards) for rewards in task_to_max_rewards.values()]
        pass_at_k_max_reward = np.mean(per_task_max) if per_task_max else 0.0
        pass_at_k_success_rate = np.mean([task_max >= 1.0 for task_max in per_task_max]) if per_task_max else 0.0
    else:
        pass_at_k_max_reward = np.mean(per_trajectory_max)
        pass_at_k_success_rate = success_rate

    prefix = get_validation_metric_prefix(data_source)
    metrics_dict = {
        f"{prefix}/max_reward": np.mean(per_trajectory_max),
        f"{prefix}/success_rate": success_rate,
        f"{prefix}/pass_at_k_success_rate": pass_at_k_success_rate,
        f"{prefix}/pass_at_k_max_reward": pass_at_k_max_reward,
        f"{prefix}/n_trajectories": len(processed_rewards_list),
        f"{prefix}/max_turns": max_turns,
    }

    if max_turns > 0:
        metrics_dict[f"{prefix}/final_avg_reward"] = turn_rewards[-1]
        metrics_dict[f"{prefix}/final_cumulative_reward"] = cumulative_rewards[-1]
        metrics_dict[f"{prefix}/final_max_reward"] = max_rewards[-1]

    return metrics_dict


def save_trajectories(results, save_dir="./trajectories", filename="trajectories.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(results, save_path)
    print(f"Trajectories saved to {save_path}")
    return save_path
