from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Direct script execution adds the checkout root before project imports.
from examples.mbpp.eval.ace.mbpp_ace_utils import (  # noqa: E402
    build_generator_engine,
    create_ace_context_rule,
    load_test_data,
    load_train_data,
    prepare_local_worker_args,
    task_id_getter,
    trajectory_reward,
    trajectory_success,
)
from rllm.algorithms.ace.evaluation import (  # noqa: E402
    ACEExperimentHooks,
    normalize_eval_frequency,
    resolve_path,
    run_local_ace,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    from rllm.utils.provider_seed import add_provider_seed_arguments

    parser = argparse.ArgumentParser(description="Local ACE validation-only runner for MBPP")
    add_provider_seed_arguments(parser)
    parser.add_argument("--handoff_protocol", choices=("legacy", "feedback_v1"), default="legacy", help="Select legacy reflection inputs or the repaired feedback adapter and role prompts.")
    parser.add_argument("--adaptation_mode", choices=("apply", "discard"), default="apply", help="Apply valid ACE deltas, or generate them and retain the initial book as a matched handoff control.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1")
    parser.add_argument("--n_parallel_agents", type=int, default=32)
    parser.add_argument(
        "--task_concurrency",
        type=int,
        default=1,
        help=("Independent local task workers with separate generator engines; needs n_parallel_agents >= task_concurrency * g_act. Concurrent requests change server RNG draw order."),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--prompts_dir", type=str, default="")
    parser.add_argument("--initial_playbook", type=str, default="")
    parser.add_argument("--save_playbook_dir", type=str, default=None)
    parser.add_argument("--playbook_snapshot_dir", type=str, default=None)
    parser.add_argument("--ace_log_dir", type=str, default=None)
    parser.add_argument("--metrics_file", type=str, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Reuse verified completed task visits in this output directory.")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument(
        "--task_offset",
        type=int,
        default=0,
        help="0-indexed start offset into the eval task list (applied before --max_tasks). Chunk N covers tasks[task_offset : task_offset + max_tasks].",
    )
    parser.add_argument("--max_train_tasks", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--feedback", choices=("full", "hidden"), default="full", help="MBPP test feedback: full shows failed assertions and errors; hidden shows only per-test outcomes.")
    parser.add_argument("--local_eval_epochs", type=int, default=8, help="Total policy visits per task; sequential adaptation performs one fewer context updates.")
    parser.add_argument("--local_eval_protocol", choices=["sequential_adaptation_v1", "legacy_search_v1"], default="sequential_adaptation_v1", help="Sequential measured policy visits, or explicit legacy candidate-search reproduction.")
    parser.add_argument("--warm_start_from_run_dir", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    # Doyoung cross-iteration freeze/max fix. max_reward_mean (per-task running-max
    # across iterations) is always logged; this flag additionally SKIPS generating
    # tasks already at reward>=1.0 (compute saving, ~40%). Off = current behavior.
    parser.add_argument("--freeze_solved", action="store_true", help="Skip re-generating tasks that already reached reward=1.0 in a prior local-eval iteration.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_model_length", type=int, default=16384)
    # Generator-engine knobs (defaults preserve prior hardcoded behavior; eval overrides them to match training).
    parser.add_argument("--gen_max_response_length", type=int, default=16384)
    parser.add_argument("--gen_max_prompt_length", type=int, default=16384)
    parser.add_argument("--gen_temperature", type=float, default=0.6)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_step", type=int, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--val_split", type=str, default=None, choices=["train", "test"])
    parser.add_argument("--max_val_tasks", type=int, default=None)
    parser.add_argument("--g_act", type=int, default=1)
    parser.add_argument("--g_ref", type=int, default=1)
    parser.add_argument("--g_cur", type=int, default=1)
    parser.add_argument("--selection_mode", type=str, default="best_rejection", choices=["random", "best", "best_rejection"])
    parser.add_argument("--ace_temperature", type=float, default=0.7)
    parser.add_argument("--max_playbook_length", type=int, default=None)
    # Per-role ACE budgets. Defaults = legacy eval convention (16384/16384 both roles);
    # the slurm harness passes the ACE-RL TRAINING values explicitly (wandb rllm.ace:
    # reflector 20480/2048, curator 8192/4096 — aligned with Doyoung 2026-07-07).
    parser.add_argument("--reflector_max_prompt_length", type=int, default=16384)
    parser.add_argument("--reflector_max_response_length", type=int, default=16384)
    parser.add_argument("--curator_max_prompt_length", type=int, default=16384)
    parser.add_argument("--curator_max_response_length", type=int, default=16384)
    # Role system prompts — pass the ACE-RL TRAINING strings for train-faithful evals
    # (mbpp training used the trainer code defaults; see the slurm harness). Default ""
    # = legacy user-only role messages (pre-2026-07-08 convention).
    parser.add_argument("--reflector_system_prompt", type=str, default="")
    parser.add_argument("--curator_system_prompt", type=str, default="")
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
    # Single thinking knob shared by Generator + Reflector + Curator (build_generator_engine and
    # create_ace_context_rule both read args.disable_thinking). Default False = thinking ON, matching
    # mbpp training. Use --no-disable_thinking to force-enable if ever toggled.
    parser.add_argument("--disable_thinking", action=argparse.BooleanOptionalAction, default=False, help="Disable Qwen3 thinking for Generator/Reflector/Curator (mbpp training used False).")
    # Vanilla eval uses base_system_prompt_v1.txt (no playbook scaffolding); full-ACE uses the default.
    parser.add_argument("--base_system_prompt_filename", type=str, default="base_system_prompt.txt")
    return parser.parse_args()


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = parse_args()
    normalize_eval_frequency(args)
    set_seed(args.seed)

    if args.val_split is not None:
        args.split = args.val_split
    if args.max_val_tasks is not None:
        args.max_tasks = args.max_val_tasks

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = resolve_path(script_dir, args.prompts_dir) if args.prompts_dir else str(_REPO_ROOT / "examples/mbpp/train/ace_rl/prompts")
    initial_playbook = resolve_path(script_dir, args.initial_playbook) if args.initial_playbook else str(_REPO_ROOT / "examples/mbpp/train/ace_rl/playbooks/initial_playbook.txt")
    playbook_snapshot_dir = resolve_path(script_dir, args.playbook_snapshot_dir) if args.playbook_snapshot_dir else None
    ace_log_dir = resolve_path(script_dir, args.ace_log_dir) if args.ace_log_dir else None
    metrics_file = resolve_path(script_dir, args.metrics_file) if args.metrics_file else None
    data_dir = resolve_path(script_dir, args.data_dir) if args.data_dir else None
    del args.save_playbook_dir

    eval_tasks = load_train_data(script_dir, data_dir=data_dir) if args.split == "train" else load_test_data(script_dir, data_dir=data_dir)
    total_tasks = len(eval_tasks)
    if args.task_offset < 0 or args.task_offset >= total_tasks:
        raise ValueError(f"task_offset={args.task_offset} is outside the {total_tasks}-task split")
    if args.max_tasks is not None and (args.max_tasks < 1 or args.task_offset + args.max_tasks > total_tasks):
        raise ValueError("The requested task slice must be positive and fit within the selected split")
    if args.task_offset:
        eval_tasks = eval_tasks[args.task_offset :]
    if args.max_tasks is not None:
        eval_tasks = eval_tasks[: args.max_tasks]

    end_idx = args.task_offset + len(eval_tasks) - 1
    print(f"Loaded {len(eval_tasks)}/{total_tasks} evaluation tasks from split '{args.split}' (offset={args.task_offset}, range=[{args.task_offset}..{end_idx}])")

    hooks = ACEExperimentHooks(
        experiment_name="mbpp_local_ace",
        task_id_getter=task_id_getter,
        build_generator_engine=build_generator_engine,
        create_context_rule=create_ace_context_rule,
        trajectory_reward=trajectory_reward,
        trajectory_success=trajectory_success,
        prepare_local_worker_args=prepare_local_worker_args,
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
