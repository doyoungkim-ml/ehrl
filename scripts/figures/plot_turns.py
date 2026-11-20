"""Plot measured evaluation rewards by turn or local ACE iteration.

Terminated trajectories are padded with their maximum observed reward. Across
iterations, solved tasks stay at 1; partial rewards retain their fresh value.
Every curve names its checkpoint and evaluation preset. A view through visit K
requires K measured visits; nothing is extrapolated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from plot_comparison import load_comparison
from plot_eval import evaluation_entries
from plot_eval import load_points as load_evaluations


def finite(value, source):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{source}: reward must be a finite number")
    return float(value)


def trajectory_rewards(trajectory, source):
    # This is the same precedence as the evaluator's trajectory_step_rewards.
    rewards = [row["reward"] for row in trajectory.get("steps", []) if row.get("reward") is not None]
    if not rewards:
        rewards = trajectory.get("info", {}).get("step_rewards") or trajectory.get("step_rewards") or []
    if not rewards and trajectory.get("reward") is not None:
        rewards = [trajectory["reward"]]
    if not rewards:
        raise ValueError(f"{source}: trajectory has no recorded rewards")
    return [finite(value, source) for value in rewards]


_FLAT_GAMES = {}


def flat_games(path):
    """Per-game records written by the Mastermind vanilla evaluator, read once."""
    key = str(path.resolve())
    if key not in _FLAT_GAMES:
        games = json.loads(path.read_text())
        if not isinstance(games, list) or not games:
            raise ValueError(f"{path}: no recorded games")
        _FLAT_GAMES[key] = games
    return _FLAT_GAMES[key]


def command_turn_budget(command):
    """Shared evaluators spell the option --max_turns; Mastermind vanilla uses --max-turns."""
    for flag in ("--max_turns", "--max-turns"):
        if flag in command[:-1]:
            return int(command[command.index(flag) + 1])
    return None


def flat_turn_budget(root):
    path = root / "trajectories.json"
    if not path.is_file():
        return None
    budgets = [game.get("task", {}).get("max_turns") for game in flat_games(path)]
    if any(isinstance(budget, bool) or not isinstance(budget, int) or budget < 1 for budget in budgets):
        raise ValueError(f"{path}: every game needs a recorded max_turns budget")
    return max(budgets)


def recorded_artifact_root(root):
    """Per-task artifacts live under the recorded playbook snapshot directory.

    Launchers record their exact evaluator command; most snapshot into the run
    directory itself, while Mastermind ACE nests everything under playbooks/.
    """
    config_path = root / "evaluation_config.json"
    if not config_path.is_file():
        return root
    config = json.loads(config_path.read_text())
    command = config.get("evaluation", [])
    if "--playbook_snapshot_dir" not in command[:-1] or not config.get("run_dir"):
        return root
    snapshot = Path(command[command.index("--playbook_snapshot_dir") + 1])
    try:
        relative = snapshot.relative_to(Path(config["run_dir"]))
    except ValueError as error:
        raise ValueError(f"{config_path}: playbook snapshot directory {snapshot} is outside the recorded run directory") from error
    return (root / relative).resolve()


def task_trajectories(root, task_id, step, epoch, total_epochs):
    safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in task_id)
    folder = root / f"task_{safe_id}" / "chat_completions" / "eval"
    path = folder / f"policy_step{step:06d}_iter_{epoch - 1}.json"
    # An unsuffixed file cannot identify a visit in a multi-epoch evaluation.
    if not path.is_file() and total_epochs == 1:
        path = folder / f"policy_step{step:06d}.json"
    flat = root / "trajectories.json"
    if not path.is_file() and total_epochs == 1 and flat.is_file():
        games = [game for game in flat_games(flat) if str(game.get("task_id")) == task_id]
        if not games:
            raise ValueError(f"{flat}: no recorded game for task {task_id}")
        return [trajectory_rewards(game, flat) for game in games], str(flat.resolve())
    payload = json.loads(path.read_text())
    if payload.get("step") != step:
        raise ValueError(f"{path}: trajectory checkpoint step disagrees with metrics")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(f"{path}: no trajectories")
    return [trajectory_rewards(trajectory, path) for trajectory in trajectories], str(path.resolve())


def measured_epochs(entry, source, local_epoch):
    epochs = entry.get("local_epochs", [])
    if not isinstance(epochs, list) or not epochs:
        raise ValueError(f"{source}: no measured local epochs")
    numbers = [epoch.get("local_eval_epoch") for epoch in epochs]
    if any(isinstance(number, bool) or not isinstance(number, int) or number < 1 for number in numbers) or numbers != list(range(numbers[0], numbers[-1] + 1)):
        raise ValueError(f"{source}: local epochs must be ordered and contiguous")
    if local_epoch is not None:
        if local_epoch not in numbers:
            raise ValueError(f"{source}: measured local epoch {local_epoch} unavailable; evaluate it first")
        epochs = [epoch for epoch in epochs if epoch["local_eval_epoch"] <= local_epoch]
    return epochs


def curve_rows(point, *, axis="turn", local_epoch=None, max_turns=None, pad_solved=True):
    source = Path(point["source"])
    payload = json.loads(source.read_text())
    entry = evaluation_entries(payload, point["step"], source)[-1]
    epochs = measured_epochs(entry, source, local_epoch)
    base = {key: point[key] for key in ("run_tag", "variant", "step", "task_label", "source")}
    if axis == "iteration":
        return [
            {
                **base,
                "local_epoch": epoch["local_eval_epoch"],
                "turn": "",
                "turn_global": "",
                "metric": "max_reward_mean",
                "value": finite(epoch.get("metrics", {}).get("max_reward_mean"), source),
                "trajectory_sources": "",
            }
            for epoch in epochs
        ]
    if epochs[0]["local_eval_epoch"] != 1:
        raise ValueError(f"{source}: turns need the full measured visit chain starting at 1")
    first_ids = [row.get("task_id") for row in epochs[0].get("tasks", [])]
    if not first_ids or any(not isinstance(task_id, str) or not task_id for task_id in first_ids) or len(set(first_ids)) != len(first_ids):
        raise ValueError(f"{source}: first epoch needs unique task identities")
    task_ids = set(first_ids)
    declared = entry.get("evaluation_task_ids")
    if declared is not None and (len(declared) != len(task_ids) or set(declared) != task_ids):
        raise ValueError(f"{source}: first epoch does not cover the declared evaluation tasks")
    safe_ids = {"".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in task_id) for task_id in task_ids}
    if len(safe_ids) != len(task_ids):
        raise ValueError(f"{source}: colliding task artifact identifiers")
    solved = set()
    observations = []
    inferred_turns = 0
    for epoch in epochs:
        number = epoch["local_eval_epoch"]
        records = epoch.get("tasks", [])
        fresh = {row["task_id"]: finite(row.get("reward"), source) for row in records}
        if len(fresh) != len(records) or fresh.keys() - task_ids or task_ids - fresh.keys() - solved:
            raise ValueError(f"{source}: duplicate, new, or missing unsolved tasks in epoch {number}")
        if epoch.get("source_run_dir"):
            raise ValueError(f"{source}: epochs drawn from other evaluation directories are not plot inputs")
        root = source.parent
        if any(part in ("archive", "historical") for part in root.parts):
            raise ValueError(f"{root}: archived sources are not maintained plot inputs")
        root = recorded_artifact_root(root)
        trajectories, paths = {}, []
        for task_id in sorted(task_ids):
            if task_id in solved:
                trajectories[task_id] = [[1.0]]
            else:
                values, path = task_trajectories(root, task_id, point["step"], number, entry["local_eval_epochs"])
                trajectories[task_id] = values
                if path not in paths:
                    paths.append(path)
                inferred_turns = max(inferred_turns, max(map(len, values)))
        observations.append((number, trajectories, paths))
        # RSI carries a successful task through later visits, regardless of
        # the evaluation protocol used to record its fresh visit rewards.
        if pad_solved:
            solved.update(task_id for task_id, reward in fresh.items() if reward >= 1)
    if max_turns is None:
        config_path = source.parent / "evaluation_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        max_turns = config.get("max_turns")
        if max_turns is None:
            max_turns = command_turn_budget(config.get("evaluation", []))
        # Completion commands retain the evaluator's actual turn budget.
        marker = source.parent / "evaluation_complete.json"
        if max_turns is None and marker.is_file():
            max_turns = command_turn_budget(json.loads(marker.read_text()).get("evaluation", []))
        # The flat Mastermind vanilla record keeps each game's effective turn budget.
        if max_turns is None:
            max_turns = flat_turn_budget(source.parent)
        if max_turns is None:
            raise ValueError(f"{source}: no recorded max_turns; pass --max-turns explicitly")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < inferred_turns or max_turns < 1:
        raise ValueError(f"{source}: max-turns must cover every recorded trajectory (at least {inferred_turns})")
    rows = []
    for number, trajectories, paths in observations:
        for turn in range(max_turns):
            task_rewards = []
            for values in trajectories.values():
                # Equal weight per task, average every acting rollout for that task.
                task_rewards.append(sum(rewards[turn] if turn < len(rewards) else max(rewards) for rewards in values) / len(values))
            rows.append(
                {
                    **base,
                    "local_epoch": number,
                    "turn": turn + 1,
                    "turn_global": (number - 1) * max_turns + turn + 1,
                    "metric": "padded_avg_reward",
                    "value": sum(task_rewards) / len(task_ids),
                    "trajectory_sources": json.dumps(paths),
                }
            )
    return rows


def load_points(runs_dir=None, *, config=None, axis="turn", task_label="all", run_tags=None, steps=None, local_epoch=None, eval_preset=None, max_turns=None, benchmark=None):
    if config is not None:
        if eval_preset or run_tags:
            raise ValueError("Comparison inputs are explicitly selected by --config; do not add --eval-preset or --run-tag")
        _, selected = load_comparison(config, benchmark=benchmark)
        points = [{"run_tag": row["label"], "variant": row["series"], **{key: row[key] for key in ("step", "task_label", "source")}} for row in selected if row["kind"] == "evaluation"]
    else:
        points = load_evaluations(runs_dir, "max_reward_mean", task_label, run_tags, eval_preset=eval_preset)
    if steps:
        points = [point for point in points if point["step"] in steps]
        missing = set(steps) - {point["step"] for point in points}
        if missing:
            raise ValueError(f"Missing requested checkpoint steps: {sorted(missing)}")
    rows = []
    for point in points:
        rows.extend(curve_rows(point, axis=axis, local_epoch=None if point["variant"] == "grpo_vanilla" else local_epoch, max_turns=max_turns))
    return rows


def render(rows, out, benchmark, axis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(11, 5.5))
    group_keys = list(dict.fromkeys((row["run_tag"], row["variant"], row["step"]) for row in rows))
    x_key = "local_epoch" if axis == "iteration" else "turn_global"
    for tag, variant, step in group_keys:
        points = [row for row in rows if (row["run_tag"], row["variant"], row["step"]) == (tag, variant, step)]
        ax.plot([row[x_key] for row in points], [row["value"] for row in points], linewidth=1.4, marker="o" if axis == "iteration" else None, label=f"{Path(tag).name} [{variant}] step {step}")
    if axis == "turn":
        boundaries = sorted({row["turn_global"] - 0.5 for row in rows if row["turn"] == 1 and row["local_epoch"] > 1})
        for boundary in boundaries:
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set(
        xlabel="Local evaluation epoch" if axis == "iteration" else "Turn across local evaluation epochs",
        ylabel="Average reward (max padded)",
        title=f"{benchmark}: measured {'iteration' if axis == 'iteration' else 'turn'} rewards",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", dpi=180)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--runs-dir", type=Path, help="Evaluation artifact directory")
    inputs.add_argument("--config", type=Path, help="Explicit comparison JSON (same inputs as comparison; all benchmarks)")
    parser.add_argument("--axis", choices=("turn", "iteration"), default="turn")
    parser.add_argument("--task-label", default="all")
    parser.add_argument("--run-tag", action="append", help="Exact run tag to include")
    parser.add_argument("--step", type=int, action="append", help="Checkpoint step to include; repeat for a ladder")
    parser.add_argument("--local-epoch", type=int, help="Include measured visits through this one (e.g. 3 of visits8)")
    parser.add_argument("--eval-preset", help="Exact evaluation preset: visits8 (ACE) or pass8 (vanilla)")
    parser.add_argument("--max-turns", type=int, help="Padding width per epoch; otherwise use the recorded evaluation budget")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate all selected metrics and trajectories without rendering")
    args = parser.parse_args(argv)
    try:
        rows = load_points(
            args.runs_dir,
            config=args.config,
            axis=args.axis,
            task_label=args.task_label,
            run_tags=args.run_tag,
            steps=args.step,
            local_epoch=args.local_epoch,
            eval_preset=args.eval_preset,
            max_turns=args.max_turns,
            benchmark=args.benchmark,
        )
        if args.check:
            print(f"Validated {len(rows)} measured {args.axis} points")
        else:
            render(rows, args.out, args.benchmark, args.axis)
            print(f"Wrote {args.out.with_suffix('.pdf')}, PNG and CSV")
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
