"""Names shared by gallery extraction and rendering, and the CSV table layout."""

import csv
import json
from pathlib import Path

BENCHMARKS = {"appworld", "mbpp", "mini_swe_bench", "mastermind"}
PLOTS = ("learning", "comparison", "comparison_extended", "variants", "rollout", "turns")
STEP_PLOTS = set(PLOTS) - {"turns"}
TOKEN_AXES = ("total_gradient_update_tokens", "total_completion_tokens")
METHODS = ("base", "grpo", "ppo", "joint_ppo", "erl", "ace_rl", "delayed_joint", "alternating", "ace_n_evals")
PLOT_METHODS = {
    "learning": ("grpo", "ppo", "joint_ppo", "erl", "ace_rl"),
    "comparison": METHODS,
    "comparison_extended": METHODS,
    "turns": METHODS,
    "variants": ("ace_rl", "delayed_joint", "alternating", "ace_n_evals"),
    "rollout": tuple(method for method in METHODS if method != "base"),
}
TURN_KEYS = ("turns", "turns_nopad", "turns_extended", "turns_extended_nopad")

# One row per plotted value. Cumulative token columns are the x positions of
# the token-axis figures; an empty cell means no exact token position exists.
TOKEN_COLUMNS = tuple(f"cumulative_{axis}" for axis in TOKEN_AXES)
TABLES = {
    # Training scalars in journal order: critic/score/mean and every rollout/* key.
    "training": ("method", "metric", "segment", "record", "step", *TOKEN_COLUMNS, "value"),
    # Checkpoint evaluations: measure is pass, rsi (success by visit k) or
    # reward_carried (mean reward through visit k with full successes carried).
    "evaluation": ("method", "measure", "k", "step", *TOKEN_COLUMNS, "value"),
    # Average reward per turn across adaptation visits for each checkpoint.
    "turns": ("method", "curve", "step", "local_epoch", "turn", "turn_global", "value"),
}
INTEGER_COLUMNS = {"segment", "record", "step", "k", "local_epoch", "turn", "turn_global"}
FLOAT_COLUMNS = {"value", *TOKEN_COLUMNS}
# Columns identifying one plotted value; --append overwrites rows with equal keys.
ROW_KEYS = {
    "training": ("method", "metric", "record"),
    "evaluation": ("method", "measure", "k", "step"),
    "turns": ("method", "curve", "step", "local_epoch", "turn"),
}


def read_tables(directory: Path) -> tuple[dict[str, list[dict]], dict]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"No extracted gallery data at {directory}; run `python plot.py <benchmark>` first")
    tables = {}
    for name, fields in TABLES.items():
        with (directory / f"{name}.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError(f"{directory / name}.csv: expected columns {', '.join(fields)}")
            tables[name] = [
                {key: (None if value == "" else int(value) if key in INTEGER_COLUMNS
                       else float(value) if key in FLOAT_COLUMNS else value)
                 for key, value in row.items()}
                for row in reader
            ]
    return tables, json.loads(manifest_path.read_text())
