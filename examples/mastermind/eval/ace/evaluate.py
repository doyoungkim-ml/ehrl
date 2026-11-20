"""Local ACE validation-only runner for Mastermind.

Same shape as examples/mbpp/eval/ace/evaluate.py: the generic loop lives in
rllm.algorithms.ace.evaluation.run_local_ace, and this file only supplies the CLI and
the Mastermind hooks. Defaults follow the values recorded in the "mastermind" tab of the
ACE Eval Status sheet (training values from wandb oyt135wd / 9wcyzbbv / hydybnem).

Qwen generator + Qwen ACE roles is the default. For Qwen generator + Gemini ACE roles,
pass --ace_model gemini-2.5-pro --ace_key_file <path> (launcher: ACE_MODEL, ACE_KEY_FILE).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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

    parser = argparse.ArgumentParser(description="Local ACE validation-only runner for Mastermind")
    add_provider_seed_arguments(parser)
    parser.add_argument("--handoff_protocol", choices=("legacy", "feedback_v1"), default="legacy", help="Select legacy reflection inputs or the repaired feedback adapter and role prompts.")
    parser.add_argument("--adaptation_mode", choices=("apply", "discard"), default="apply", help="Apply valid ACE deltas, or generate them and retain the initial book as a matched handoff control.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer for a served checkpoint alias; defaults to model_name")
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1")
    parser.add_argument("--n_parallel_agents", type=int, default=8)
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
    parser.add_argument("--task_offset", type=int, default=0)
    parser.add_argument("--max_train_tasks", type=int, default=None)
    # Mastermind games run 20 turns (dataset max_turns); rows carry their own value too.
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--local_eval_epochs", type=int, default=8, help="Total policy visits per task; sequential adaptation performs one fewer context updates.")
    parser.add_argument("--local_eval_protocol", choices=["sequential_adaptation_v1", "legacy_search_v1"], default="sequential_adaptation_v1", help="Sequential measured policy visits, or explicit legacy candidate-search reproduction.")
    parser.add_argument("--freeze_solved", action="store_true", help="Skip re-generating tasks that already reached reward=1.0 in a prior local-eval iteration.")
    parser.add_argument("--debug", action="store_true")
    # Qwen3-0.6B max_position_embeddings = 40960.
    parser.add_argument("--max_model_length", type=int, default=40960)
    parser.add_argument("--gen_max_response_length", type=int, default=32768)
    parser.add_argument("--gen_max_prompt_length", type=int, default=8192)
    # ROLLOUT sampling params matching training: 0.7 / 1.0.
    parser.add_argument("--gen_temperature", type=float, default=0.7)
    parser.add_argument("--gen_top_p", type=float, default=1.0)
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
    # Evaluation convention: 1/1, against 4/4 in training.
    parser.add_argument("--g_ref", type=int, default=1)
    parser.add_argument("--g_cur", type=int, default=1)
    parser.add_argument("--selection_mode", type=str, default="best_rejection", choices=["random", "best", "best_rejection"])
    parser.add_argument("--ace_temperature", type=float, default=0.7)
    parser.add_argument("--max_playbook_length", type=int, default=2048)
    # Training role budgets (wandb hydybnem).
    parser.add_argument("--reflector_max_prompt_length", type=int, default=30720)
    parser.add_argument("--reflector_max_response_length", type=int, default=2048)
    parser.add_argument("--curator_max_prompt_length", type=int, default=8192)
    parser.add_argument("--curator_max_response_length", type=int, default=4096)
    parser.add_argument("--reflector_system_prompt", type=str, default="")
    parser.add_argument("--curator_system_prompt", type=str, default="")
    parser.add_argument("--test_epoch_freq", type=int, default=1)
    parser.add_argument("--test_freq", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_file", type=str, default=None)
    parser.add_argument("--project_id", type=str, default=None)
    parser.add_argument("--ace_model", type=str, default=None, help="Model for Reflector/Curator; default = the generator model")
    parser.add_argument("--ace_base_url", type=str, default=None)
    parser.add_argument("--ace_api_key", type=str, default=None)
    parser.add_argument("--ace_key_file", type=str, default=None)
    parser.add_argument("--ace_project_id", type=str, default=None)
    # Gemini 2.5 spends max_tokens on thought tokens; 2048 is the value the alfworld
    # Gemini wave settled on (804/804 reflections valid).
    parser.add_argument("--ace_thinking_budget", type=int, default=2048)
    parser.add_argument("--include_thoughts", action="store_true")
    # Single thinking knob for Generator + Reflector + Curator. Default False = thinking ON:
    # the 2026-09-11 vanilla baseline scored 0.334 thinking vs 0.105 nonthinking.
    parser.add_argument("--disable_thinking", action=argparse.BooleanOptionalAction, default=False, help="Disable Qwen3 thinking for Generator/Reflector/Curator")
    parser.add_argument("--base_system_prompt_filename", type=str, default="base_system_prompt.txt")
    return parser.parse_args(argv)


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = parse_args()

    from examples.mastermind.eval.ace.mastermind_ace_utils import (
        build_generator_engine,
        create_ace_context_rule,
        load_test_data,
        load_train_data,
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
    prompts_dir = resolve_path(script_dir, args.prompts_dir) if args.prompts_dir else str(_REPO_ROOT / "examples/mastermind/train/ace_rl/prompts")
    initial_playbook = resolve_path(script_dir, args.initial_playbook) if args.initial_playbook else str(_REPO_ROOT / "examples/mastermind/train/ace_rl/playbooks/initial_playbook.txt")
    playbook_snapshot_dir = resolve_path(script_dir, args.playbook_snapshot_dir) if args.playbook_snapshot_dir else None
    ace_log_dir = resolve_path(script_dir, args.ace_log_dir) if args.ace_log_dir else None
    metrics_file = resolve_path(script_dir, args.metrics_file) if args.metrics_file else None
    data_dir = resolve_path(script_dir, args.data_dir) if args.data_dir else None
    del args.save_playbook_dir

    eval_tasks = load_train_data(script_dir, data_dir=data_dir) if args.split == "train" else load_test_data(script_dir, data_dir=data_dir)
    total_tasks = len(eval_tasks)
    if args.task_offset:
        eval_tasks = eval_tasks[args.task_offset :]
    if args.max_tasks is not None:
        eval_tasks = eval_tasks[: args.max_tasks]

    if not eval_tasks:
        # Without this the run writes a metrics file covering zero tasks, which a
        # campaign merge would silently count as a real chunk. evaluate.py raises the
        # same way for an empty slice.
        raise SystemExit(f"The selected Mastermind {args.split} slice is empty: offset={args.task_offset} max_tasks={args.max_tasks} against {total_tasks} tasks.")

    end_idx = args.task_offset + len(eval_tasks) - 1
    print(f"Loaded {len(eval_tasks)}/{total_tasks} evaluation tasks from split '{args.split}' (offset={args.task_offset}, range=[{args.task_offset}..{end_idx}])")

    hooks = ACEExperimentHooks(
        experiment_name="mastermind_local_ace",
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
