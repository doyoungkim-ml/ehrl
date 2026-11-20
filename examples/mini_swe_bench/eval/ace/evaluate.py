"""Mini-SWE validation runner with optional CSV filtering before offset/limit.

The shared ACE loop also handles vanilla evaluation with g_ref=g_cur=0.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ACE evaluation shares policy, reflector, and curator assets with training.
TRAIN_ARTIFACTS_DIR = _REPO_ROOT / "examples" / "mini_swe_bench" / "train" / "ace_rl"

# Direct script execution adds the checkout root before project imports.


def set_seed(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from rllm.utils.provider_seed import add_provider_seed_arguments

    parser = argparse.ArgumentParser(description="Local ACE validation-only runner for mini SWE-bench")
    add_provider_seed_arguments(parser)
    parser.add_argument("--handoff_protocol", choices=("legacy", "feedback_v1"), default="legacy", help="Select legacy reflection inputs or the repaired feedback adapter and role prompts.")
    parser.add_argument("--adaptation_mode", choices=("apply", "discard"), default="apply", help="Apply valid ACE deltas, or generate them and retain the initial book as a matched handoff control.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1")
    parser.add_argument("--n_parallel_agents", type=int, default=8)
    parser.add_argument(
        "--task_concurrency",
        type=int,
        default=1,
        help="Independent local task workers with separate generator engines and sandboxes; needs n_parallel_agents >= task_concurrency * g_act. Concurrent requests change server RNG draw order.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--prompts_dir", type=str, default="")
    parser.add_argument("--base_system_prompt_filename", type=str, default="base_system_prompt_tools.txt")
    parser.add_argument("--initial_playbook", type=str, default="")
    parser.add_argument("--save_playbook_dir", type=str, default=None)
    parser.add_argument("--playbook_snapshot_dir", type=str, default=None)
    parser.add_argument("--ace_log_dir", type=str, default=None)
    parser.add_argument("--metrics_file", type=str, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument(
        "--task_offset",
        type=int,
        default=0,
        help="0-indexed start offset into the eval task list (applied before --max_tasks). Chunk N covers tasks[task_offset : task_offset + max_tasks].",
    )
    parser.add_argument(
        "--task_list",
        type=str,
        default=None,
        help="Optional CSV (with 'instance_id' column) restricting eval tasks to the listed instance_ids. Applied BEFORE --task_offset / --max_tasks.",
    )
    parser.add_argument("--max_train_tasks", type=int, default=None)
    parser.add_argument(
        "--accumulate_thinking",
        action="store_true",
        help="Keep prior-turn <think>...</think> in the message history (default off: stripped from history, kept only for the turn it was generated). Qwen-only effect; no-op for Gemini.",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Close the thinking block in the generation prompt so the model answers without reasoning (Qwen chat-template effect; no-op for Gemini chat-completions path).",
    )
    parser.add_argument("--max_turns", type=int, default=10, help="Per-trajectory step cap (also passed to AgentExecutionEngine as max_steps).")
    parser.add_argument("--max_prompt_length", type=int, default=8192, help="Per-trajectory initial-prompt gate (checked once at trajectory start). Matches mbpp default (16384).")
    parser.add_argument(
        "--max_response_length",
        type=int,
        default=32768,
        help="Trajectory-level total output budget (sum of assistant + env tokens across all turns). When exceeded, trajectory truncates. Matches mbpp default (16384).",
    )
    parser.add_argument("--local_eval_epochs", type=int, default=8, help="Total policy visits per task; sequential adaptation performs one fewer context updates.")
    parser.add_argument("--local_eval_protocol", choices=["sequential_adaptation_v1", "legacy_search_v1"], default="sequential_adaptation_v1", help="Sequential measured policy visits, or explicit legacy candidate-search reproduction.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_model_length", type=int, default=32768)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_step", type=int, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="validation = SWE-bench Verified (eval default); train = SWE-Gym-Subset (in-distribution sanity)",
    )
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--val_split", type=str, default=None, choices=["train", "validation"])
    parser.add_argument("--max_val_tasks", type=int, default=None)
    parser.add_argument("--g_act", type=int, default=1)
    parser.add_argument("--g_ref", type=int, default=1)
    parser.add_argument("--g_cur", type=int, default=1)
    parser.add_argument("--selection_mode", type=str, default="best_rejection", choices=["random", "best", "best_rejection"])
    parser.add_argument("--ace_temperature", type=float, default=0.7)
    parser.add_argument("--max_playbook_length", type=int, default=None)
    parser.add_argument(
        "--reflector_obs_max_chars",
        type=int,
        default=300,
        help="Per-step observation char cap in the Reflector's conversation history. 0 disables the cap (keep env-truncated observations in full). Default 300 preserves the prior hardcoded behavior.",
    )
    parser.add_argument(
        "--reflector_test_sample_n",
        type=int,
        default=0,
        help="Max per-test case names sampled into the Reflector's Trajectory "
        "Summary, per category (FAIL_TO_PASS, PASS_TO_PASS), split 2:3 "
        "success:failure, random sampling, no backfill. 0 = names off "
        "(counts only; no FAIL_TO_PASS test names reach the playbook). "
        "Only failed graded trajectories carry test names.",
    )
    parser.add_argument(
        "--reflector_simplified_history",
        action="store_true",
        help="Use simplified Option-A conversation history for the Reflector: "
        "strip Generator reasoning (<think> blocks), keep only per-turn "
        "actions and observations under tagged headers. Default off "
        "(keeps the legacy format that includes AGENT REASONING).",
    )
    parser.add_argument(
        "--reflector_only",
        action="store_true",
        help="Skip the Curator LLM call entirely. Each reflection's "
        "key_insight is ADDed to the playbook verbatim (routed to the "
        "'others' section, which works in both flat and section playbook "
        "modes). Builds one candidate per reflector group; g_cur is "
        "ignored. Candidate evaluation and selection still apply.",
    )
    parser.add_argument("--val_before_train", action="store_true")
    parser.add_argument("--test_epoch_freq", type=int, default=1)
    parser.add_argument("--test_freq", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_file", type=str, default=None)
    parser.add_argument("--project_id", type=str, default=None)
    parser.add_argument("--ace_model", type=str, default=None)
    parser.add_argument("--ace_base_url", type=str, default=None)
    parser.add_argument("--ace_api_key", type=str, default=None)
    parser.add_argument("--ace_key_file", type=str, default=None)
    parser.add_argument("--ace_project_id", type=str, default=None)
    parser.add_argument("--include_thoughts", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse verified completed task visits in this output directory.",
    )
    return parser.parse_args(argv)


def _load_task_list(csv_path: str) -> set[str]:
    """Read a CSV with at minimum an 'instance_id' column. Returns the set
    of instance_ids found. Fails loudly on missing file or missing column."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"--task_list CSV not found: {csv_path}")
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "instance_id" not in reader.fieldnames:
            raise ValueError(f"--task_list CSV must have an 'instance_id' column; got columns: {reader.fieldnames}")
        ids = {row["instance_id"] for row in reader if row.get("instance_id")}
    if not ids:
        raise ValueError(f"--task_list CSV had no instance_id rows: {csv_path}")
    return ids


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = parse_args()

    from examples.mini_swe_bench.eval.ace.mini_swe_bench_ace_utils import (
        build_generator_engine,
        create_ace_context_rule,
        load_swe_tasks,
        prepare_local_worker_args,
        task_id_getter,
        trajectory_reward,
        trajectory_success,
    )
    from rllm.algorithms.ace.evaluation import (
        ACEExperimentHooks,
        normalize_eval_frequency,
        resolve_path,
        run_local_ace,
    )

    normalize_eval_frequency(args)
    set_seed(args.seed)

    if args.val_split is not None:
        args.split = args.val_split
    if args.max_val_tasks is not None:
        args.max_tasks = args.max_val_tasks

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = resolve_path(script_dir, args.prompts_dir) if args.prompts_dir else str(TRAIN_ARTIFACTS_DIR / "prompts")
    initial_playbook = resolve_path(script_dir, args.initial_playbook) if args.initial_playbook else str(TRAIN_ARTIFACTS_DIR / "playbooks" / "initial_playbook.txt")
    playbook_snapshot_dir = resolve_path(script_dir, args.playbook_snapshot_dir) if args.playbook_snapshot_dir else None
    ace_log_dir = resolve_path(script_dir, args.ace_log_dir) if args.ace_log_dir else None
    metrics_file = resolve_path(script_dir, args.metrics_file) if args.metrics_file else None
    data_dir = resolve_path(script_dir, args.data_dir) if args.data_dir else None
    del args.save_playbook_dir

    eval_tasks = load_swe_tasks(args.split, data_dir=data_dir)
    total_tasks = len(eval_tasks)

    # Filter by --task_list (if provided) BEFORE offset/max_tasks slicing,
    # so chunking still works against the filtered set.
    if args.task_list:
        target_ids = _load_task_list(args.task_list)
        before_count = len(eval_tasks)
        eval_tasks = [t for t in eval_tasks if task_id_getter(t) in target_ids]
        missing = target_ids - {task_id_getter(t) for t in eval_tasks}
        print(f"[task_list] {args.task_list}: filtered to {len(eval_tasks)}/{before_count} tasks (target_ids={len(target_ids)}, missing_from_dataset={len(missing)})")
        if missing:
            sample = sorted(missing)[:5]
            print(f"[task_list] WARNING: {len(missing)} ids not in dataset, first 5: {sample}")

    if args.task_offset:
        eval_tasks = eval_tasks[args.task_offset :]
    if args.max_tasks is not None:
        eval_tasks = eval_tasks[: args.max_tasks]

    end_idx = args.task_offset + len(eval_tasks) - 1
    print(f"Loaded {len(eval_tasks)}/{total_tasks} evaluation tasks from split '{args.split}' (offset={args.task_offset}, range=[{args.task_offset}..{end_idx}])")

    hooks = ACEExperimentHooks(
        experiment_name="mini_swe_bench_local_ace",
        task_id_getter=task_id_getter,
        build_generator_engine=build_generator_engine,
        prepare_local_worker_args=prepare_local_worker_args,
        create_context_rule=create_ace_context_rule,
        trajectory_reward=trajectory_reward,
        trajectory_success=trajectory_success,
    )

    asyncio.run(
        run_local_ace(
            args=args,
            hooks=hooks,
            train_tasks=[],
            eval_tasks=eval_tasks,
            prompts_dir=prompts_dir,
            initial_playbook_path=initial_playbook,
            playbook_snapshot_dir=playbook_snapshot_dir,
            ace_log_dir=ace_log_dir,
            metrics_file=metrics_file,
            script_path=__file__,
            base_system_prompt_filename=args.base_system_prompt_filename,
        )
    )


if __name__ == "__main__":
    main()
