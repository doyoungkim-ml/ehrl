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

# ACE evaluation shares policy, reflector, and curator assets with training.
TRAIN_ARTIFACTS_DIR = _REPO_ROOT / "examples" / "appworld" / "train" / "ace_rl"

# Direct script execution adds the checkout root before project imports.
from examples.appworld.eval.ace.appworld_ace_utils import (  # noqa: E402
    build_generator_engine,
    create_ace_context_rule,
    load_appworld_tasks,
    load_specific_tasks,
    prepare_local_worker_args,
    task_id_getter,
    trajectory_reward,
    trajectory_success,
)
from rllm.agents.appworld_react_agents import AppWorldReactAgent  # noqa: E402
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from rllm.utils.provider_seed import add_provider_seed_arguments

    parser = argparse.ArgumentParser(description="Local ACE validation-only runner for AppWorld")
    add_provider_seed_arguments(parser)
    parser.add_argument("--handoff_protocol", choices=("legacy", "feedback_v1"), default="legacy", help="Select legacy reflection inputs or the repaired feedback adapter and role prompts.")
    parser.add_argument("--adaptation_mode", choices=("apply", "discard"), default="apply", help="Apply valid ACE deltas, or generate them and retain the initial book as a matched handoff control.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1")
    parser.add_argument("--n_parallel_agents", type=int, default=4)
    parser.add_argument(
        "--task_concurrency",
        type=int,
        default=1,
        help="Independent local task workers; requires one agent/server per worker and g_act=1. Concurrent request ordering changes sampled RNG draws.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--trajectory_hard_timeout", type=float, default=7200, help="Hard timeout in seconds for each generator trajectory.")
    parser.add_argument(
        "--training_input_parity",
        action="store_true",
        help="Use ACE training chat reconstruction, reflector prompt fitting/sampling, and complete curator insights; retain the configured evaluation search budget.",
    )
    parser.add_argument("--max_response_length", type=int, default=16384, help="Cumulative generator trajectory token budget, including tokenized environment additions.")
    parser.add_argument("--max_prompt_length", type=int, default=16384, help="Max prompt tokens enforced by the agent execution engine.")
    parser.add_argument("--gen_temperature", type=float, default=0.6, help="Generator sampling temperature (pin to the training rollout value for faithful evals).")
    parser.add_argument("--gen_top_p", type=float, default=0.95, help="Generator sampling top_p (pin to the training rollout value for faithful evals).")
    # Thinking knobs mirror the training env knobs in examples/appworld/train/grpo/train.slurm
    # (ACCUMULATE_THINKING default True, DISABLE_THINKING default False). disable_thinking is the
    # single knob shared with the Reflector/Curator (create_ace_context_rule) — one flag controls
    # all three roles, mirroring the mbpp/alfworld June harnesses.
    parser.add_argument(
        "--accumulate_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep <think>...</think> blocks in assistant message history across turns (training default True; pass --no-accumulate_thinking for atFalse runs).",
    )
    parser.add_argument(
        "--disable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Close <think></think> in the generation prompt so the model produces content-only output (Qwen tokenizer only; no-op for external models).",
    )
    parser.add_argument(
        "--accumulate_reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Engine-level knob (config.rllm.accumulate_reasoning, consumed in "
        "rllm/engine/agent_execution_engine.py) — keep reasoning content in the "
        "reconstructed generation stream (training default True; pass "
        "--no-accumulate_reasoning for arFalse runs).",
    )
    parser.add_argument("--remote_environment_url", type=str, default=None)
    parser.add_argument("--n_servers", type=int, default=1)
    parser.add_argument("--base_port", type=int, default=9000)
    parser.add_argument("--prompts_dir", type=str, default="")
    parser.add_argument("--initial_playbook", type=str, default="")
    parser.add_argument("--playbook_snapshot_dir", type=str, default=None)
    parser.add_argument("--ace_log_dir", type=str, default=None)
    parser.add_argument("--metrics_file", type=str, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Reuse verified completed task visits in this output directory.")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--task_offset", type=int, default=0, help="0-indexed start offset into the eval task list (applied before --max_tasks).")
    parser.add_argument("--max_train_tasks", type=int, default=None)
    parser.add_argument("--task_id", type=str, default=None)
    parser.add_argument("--task_ids", type=str, nargs="+", default=None)
    parser.add_argument("--test_split", type=str, default="dev", choices=["train", "dev", "test_normal", "test_challenge"])
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "dev", "test_normal", "test_challenge"])
    parser.add_argument("--local_eval_epochs", type=int, default=8, help="Total policy visits per task; sequential adaptation performs one fewer context updates.")
    parser.add_argument("--local_eval_protocol", choices=["sequential_adaptation_v1", "legacy_search_v1"], default="sequential_adaptation_v1", help="Sequential measured policy visits, or explicit legacy candidate-search reproduction.")
    parser.add_argument(
        "--final_policy_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy search only: append a policy-only pass. Sequential adaptation already reports its last measured visit.",
    )
    parser.add_argument(
        "--final_policy_from_run_dir",
        type=str,
        default=None,
        help="Reuse a verified completed run's final playbooks and run only the final policy pass in a separate output directory.",
    )
    parser.add_argument(
        "--final_policy_source_manifest",
        type=str,
        default=None,
        help="Provenance receipt binding a legacy source run to retained launch/deployment manifests and frozen default assets.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_model_length", type=int, default=16384)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_step", type=int, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--g_act", type=int, default=1)
    parser.add_argument("--g_ref", type=int, default=1)
    parser.add_argument("--g_cur", type=int, default=1)
    # Per-role token budgets (consumed in appworld_ace_utils.create_ace_context_rule).
    # None = legacy convention (prompt=max_model_length, response=16384). The standard
    # eval slurm passes the ACE-RL TRAINING values: reflector 20480/2048, curator 8192/4096.
    parser.add_argument("--reflector_max_prompt_length", type=int, default=None)
    parser.add_argument("--reflector_max_response_length", type=int, default=None)
    parser.add_argument("--curator_max_prompt_length", type=int, default=None)
    parser.add_argument("--curator_max_response_length", type=int, default=None)
    parser.add_argument("--reflector_max_history_tokens", type=int, default=20000, help="Token cap for the reflector trajectory history (Doyoung ad330ec2 adapter fix; his slurm default 20000).")
    # Role system prompts ("" = legacy user-only calls). Set to the ACE-RL training
    # strings for train-faithful evals (wandb rllm.ace.{reflector,curator}_system_prompt).
    parser.add_argument("--reflector_system_prompt", type=str, default="")
    parser.add_argument("--curator_system_prompt", type=str, default="")
    parser.add_argument("--max_playbook_length", type=int, default=None, help="Maximum tokenizer tokens in a curated playbook; longer candidates are rejected. None = no cap.")
    parser.add_argument(
        "--freeze_solved", action="store_true", help="Skip re-running tasks that already hit reward 1.0 in a prior iteration (Fig5 freeze-at-1.0 mechanism; consumed in ace_shared.py)."
    )
    parser.add_argument("--selection_mode", type=str, default="best_rejection", choices=["random", "best", "best_rejection"])
    parser.add_argument("--ace_temperature", type=float, default=0.7)
    parser.add_argument("--val_before_train", action="store_true")
    parser.add_argument("--test_epoch_freq", type=int, default=1)
    parser.add_argument("--test_freq", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ace_model", type=str, default=None)
    parser.add_argument("--ace_base_url", type=str, default=None)
    parser.add_argument("--ace_api_key", type=str, default=None)
    parser.add_argument("--ace_key_file", type=str, default=None)
    parser.add_argument("--ace_project_id", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_file", type=str, default=None)
    parser.add_argument("--project_id", type=str, default=None)
    parser.add_argument("--include_thoughts", action="store_true")
    parser.add_argument("--warm_start_from_run_dir", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--base_system_prompt_filename", type=str, default="base_system_prompt.txt")
    return parser.parse_args(argv)


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = parse_args()
    normalize_eval_frequency(args)
    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = resolve_path(script_dir, args.prompts_dir) if args.prompts_dir else str(TRAIN_ARTIFACTS_DIR / "prompts")
    initial_playbook = resolve_path(script_dir, args.initial_playbook) if args.initial_playbook else str(TRAIN_ARTIFACTS_DIR / "playbooks" / "initial_playbook.txt")
    playbook_snapshot_dir = resolve_path(script_dir, args.playbook_snapshot_dir) if args.playbook_snapshot_dir else None
    ace_log_dir = resolve_path(script_dir, args.ace_log_dir) if args.ace_log_dir else None
    metrics_file = resolve_path(script_dir, args.metrics_file) if args.metrics_file else None

    task_ids = [args.task_id] if args.task_id else list(args.task_ids or [])
    if task_ids:
        eval_tasks = load_specific_tasks(task_ids)
    else:
        eval_tasks = load_appworld_tasks(
            split=args.test_split,
            num_tasks=args.max_tasks,
            task_offset=args.task_offset,
        )

    end_idx = args.task_offset + len(eval_tasks) - 1
    print(f"Loaded {len(eval_tasks)} evaluation tasks from split '{args.test_split}' (offset={args.task_offset}, range=[{args.task_offset}..{end_idx}])")

    hooks = ACEExperimentHooks(
        render_prompt_in_agent=True,
        experiment_name="appworld_local_ace",
        task_id_getter=task_id_getter,
        build_generator_engine=build_generator_engine,
        prepare_local_worker_args=prepare_local_worker_args,
        create_context_rule=create_ace_context_rule,
        trajectory_reward=trajectory_reward,
        trajectory_success=trajectory_success,
        parse_evaluation_args=parse_args,
    )

    # Vanilla = both ACE roles off (g_ref=0 and g_cur=0; the launcher implies it
    # this way, no separate flag). Vanilla renders the agent's own built-in
    # REACT_PROMPT -- the exact GRPO-training prompt path -- instead of a copied
    # prompt file, so train/eval prompt parity does not depend on a hand-kept copy.
    vanilla = int(args.g_ref) == 0 and int(args.g_cur) == 0
    base_system_prompt_template = AppWorldReactAgent.REACT_PROMPT if vanilla else None
    if vanilla:
        print(
            "Vanilla eval (g_ref=0, g_cur=0): base system prompt = AppWorldReactAgent.REACT_PROMPT "
            f"(built-in, same as GRPO training); --base_system_prompt_filename={args.base_system_prompt_filename!r} ignored"
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
            base_system_prompt_template=base_system_prompt_template,
        )
    )


if __name__ == "__main__":
    main()
