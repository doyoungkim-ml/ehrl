"""Evaluate registered Mastermind test tasks through an existing serving endpoint."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

_ASSETS = Path(__file__).resolve().parent / "ace"


RAW_OUTPUT_KEY = "evaluation_raw_output"


def _load_mastermind_agent():
    # Deferred so --help and result summarization stay independent of the execution stack.
    from rllm.agents.mastermind_agent import MastermindAgent

    return MastermindAgent


def build_evaluation_agent_class():
    """Return a MastermindAgent subclass that retains raw model outputs per step.

    The shared agent replaces each step's ``info`` with the environment's info on
    ``update_from_env``; the raw output captured on ``update_from_model`` is
    re-attached afterwards so ``trajectories.json`` keeps every sample's raw text.
    """
    MastermindAgent = _load_mastermind_agent()

    class EvaluationAgent(MastermindAgent):
        """Capture evaluation outputs without changing the shared agent class."""

        def __init__(self, *agent_args, save_raw_outputs: bool = False, **agent_kwargs):
            self.save_raw_outputs = bool(save_raw_outputs)
            super().__init__(*agent_args, **agent_kwargs)

        def update_from_model(self, response, **kwargs):
            action = super().update_from_model(response, **kwargs)
            if self.save_raw_outputs and self.trajectory.steps:
                model_output = kwargs.get("model_output")
                self.trajectory.steps[-1].info[RAW_OUTPUT_KEY] = {
                    "raw_response": response,
                    "engine_text": getattr(model_output, "text", None),
                    "engine_content": getattr(model_output, "content", None),
                    "engine_reasoning": getattr(model_output, "reasoning", None),
                    "finish_reason": getattr(model_output, "finish_reason", None),
                    "completion_length": getattr(model_output, "completion_length", None),
                }
            return action

        def update_from_env(self, observation, reward, done, info, **kwargs):
            step = self.trajectory.steps[-1] if self.trajectory.steps else None
            raw_output = (step.info or {}).get(RAW_OUTPUT_KEY) if step is not None else None
            result = super().update_from_env(observation, reward, done, info, **kwargs)
            if raw_output is not None and self.trajectory.steps and self.trajectory.steps[-1] is step:
                step.info = {**(step.info or {}), RAW_OUTPUT_KEY: raw_output}
            return result

    return EvaluationAgent


def replicate_tasks(tasks: list[dict], samples_per_task: int, seed: int, *, offset: int = 0) -> list[dict]:
    """Keep game seeds fixed while assigning independent policy samples stable IDs."""
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be positive")
    samples = []
    seen = set()
    for index, task in enumerate(tasks, start=offset):
        task_id = str(task.get("uid", task.get("index", index)))
        if task_id in seen:
            raise ValueError(f"Duplicate Mastermind task ID: {task_id}")
        seen.add(task_id)
        for attempt_id in range(samples_per_task):
            sample = copy.deepcopy(task)
            sample.update(_evaluation_task_id=task_id, _evaluation_attempt_id=attempt_id, _evaluation_sampling_seed=seed + attempt_id)
            samples.append(sample)
    return samples


def sample_identity(task: dict, index: int) -> dict:
    task_id = str(task.get("_evaluation_task_id", task.get("uid", task.get("index", index))))
    attempt_id = int(task.get("_evaluation_attempt_id", 0))
    return {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "sample_index": attempt_id,
        "sample_id": f"{task_id}/attempt_{attempt_id}",
        "sampling_seed": task.get("_evaluation_sampling_seed"),
    }


def summarize_trajectories(trajectories: list[Any], samples_per_task: int = 1) -> dict:
    """Group every attempt; report solved tasks separately from partial reward."""
    grouped: dict[str, list[dict]] = {}
    for index, trajectory in enumerate(trajectories):
        steps = trajectory.steps
        identity = sample_identity(trajectory.task or {}, index)
        grouped.setdefault(identity["task_id"], []).append(
            {
                **identity,
                "success": any((step.info or {}).get("won", False) for step in steps),
                "reward": max((float(step.reward) for step in steps), default=0.0),
                "num_steps": len(steps),
                "termination_reason": trajectory.info.get("termination_reason"),
            }
        )
    tasks = []
    for task_id, samples in grouped.items():
        samples.sort(key=lambda sample: sample["attempt_id"])
        if [sample["attempt_id"] for sample in samples] != list(range(samples_per_task)):
            raise ValueError(f"Incomplete or duplicate attempts for task {task_id}; expected {samples_per_task}")
        tasks.append(
            {
                "task_id": task_id,
                "success": any(sample["success"] for sample in samples),
                "reward": sum(sample["reward"] for sample in samples) / samples_per_task,
                "num_steps": sum(sample["num_steps"] for sample in samples),
                "termination_reason": samples[0]["termination_reason"] if samples_per_task == 1 else None,
                "samples": samples,
            }
        )
    return summarize_task_rows(tasks, samples_per_task)


def summarize_task_rows(tasks: list[dict], samples_per_task: int) -> dict:
    count = len(tasks)
    successes = sum(task["success"] for task in tasks)
    pass_rates = {}
    prefix_rates = {}
    for k in range(1, samples_per_task + 1):
        estimates = []
        for task in tasks:
            failures = samples_per_task - sum(sample["success"] for sample in task["samples"])
            estimates.append(1.0 - (math.comb(failures, k) / math.comb(samples_per_task, k) if failures >= k else 0.0))
        pass_rates[str(k)] = sum(estimates) / count if count else 0.0
        prefix_rates[str(k)] = sum(any(sample["success"] for sample in task["samples"][:k]) for task in tasks) / count if count else 0.0
    return {
        "num_tasks": count,
        "num_samples": count * samples_per_task,
        "samples_per_task": samples_per_task,
        "success_count": successes,
        "success_rate": successes / count if count else 0.0,
        "sample_success_rate": sum(sample["success"] for task in tasks for sample in task["samples"]) / (count * samples_per_task) if count else 0.0,
        "pass_at_k_success_rate": successes / count if count else 0.0,
        "pass_at_k_success_rates": pass_rates,
        **{f"pass_at_{k}_success_rate": rate for k, rate in pass_rates.items()},
        "prefix_success_rates": prefix_rates,
        "avg_reward": sum(task["reward"] for task in tasks) / count if count else 0.0,
        "reward_aggregation": "max",
        "sample_reward_aggregation": "mean",
        "tasks": tasks,
    }


def task_slice(args, n_tasks: int | None = None) -> str:
    """Name the chunk by the tasks actually evaluated.

    Labelling straight from --limit overstates a clipped slice: with 100 test tasks,
    offset 90 limit 33 would read tasks90-122 for 10 real games, and a campaign merge
    reading run dirs would trust that name. A full run keeps the "-all" label.
    """
    offset = args.offset
    if args.limit is None:
        return f"{offset}-all"
    return f"{offset}-{offset + (n_tasks or args.limit) - 1}"


def trajectory_payload(trajectory, index: int) -> dict:
    task = trajectory.task or {}
    steps = []
    for step in trajectory.steps:
        info = dict(step.info or {})
        info.pop("secret_code", None)  # never persist the answer next to the game
        entry = {
            "observation": step.observation,
            "action": step.action,
            "model_response": step.model_response,  # filtered: thinking stripped, engine content preferred
            "reward": step.reward,
            "info": info,
        }
        entry.update(info.pop(RAW_OUTPUT_KEY, {}))
        steps.append(entry)
    final = trajectory.steps[-1] if trajectory.steps else None
    return {
        **sample_identity(task, index),
        "task": {k: task.get(k) for k in ("uid", "seed", "code_length", "num_digits", "max_turns")},
        "success": any((step.info or {}).get("won", False) for step in trajectory.steps),
        "reward_max": max((float(step.reward) for step in trajectory.steps), default=0.0),
        "num_steps": len(trajectory.steps),
        "steps": steps,
        "chat": list(getattr(final, "chat_completions", []) or []),
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Existing OpenAI-compatible serving URL, including /v1; requires /completions")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name served by that endpoint")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B", help="Matching tokenizer name or local directory")
    parser.add_argument("--model-variant", choices=("nonthinking", "thinking"), default="nonthinking")
    parser.add_argument("--limit", type=positive_int, help="Number of registered test tasks; default: all")
    parser.add_argument("--offset", type=int, default=0, help="Start at this position in the registered test split")
    parser.add_argument("--parallel", type=positive_int, default=4, help="Maximum concurrent games")
    parser.add_argument("--g-act", "--samples-per-task", dest="g_act", type=positive_int, default=1, help="Independent parallel attempts per task; retain all attempts for pass@k")
    parser.add_argument("--playbook", type=Path, help="Fixed UTF-8 playbook; renders train/ace_rl/prompts/base_system_prompt.txt")
    parser.add_argument("--max-prompt-length", type=positive_int, default=8192)
    parser.add_argument("--max-response-length", type=positive_int, default=10240, help="Total trajectory response/history token budget")
    parser.add_argument("--max-model-length", type=positive_int, default=32768, help="Context limit supported by the serving endpoint")
    parser.add_argument("--max-turns", type=positive_int, default=5, help="Guess budget per game; overrides each dataset row's max_turns")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed; registered task seeds are unchanged")
    parser.add_argument("--output", type=Path, help="Also write the JSON result to this file")
    parser.add_argument("--run-tag", help="Run-directory label; optional with --output")
    parser.add_argument("--step", type=int, default=0, help="Checkpoint step being evaluated")
    parser.add_argument("--run-dir", type=Path, help="Exact evaluation output directory")
    parser.add_argument("--runs-base", type=Path, default=Path(__file__).resolve().parent / "vanilla" / "results")
    parser.add_argument("--save-trajectories", action="store_true", help="Write per-game chats and raw model outputs")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Reuse completed task batches in the same output directory")
    args = parser.parse_args(argv)
    if args.offset < 0:
        parser.error("--offset must be nonnegative")
    if args.parallel < args.g_act:
        parser.error("--parallel must be at least --g-act so each task's attempts can run concurrently")
    if args.g_act > 1:
        args.save_trajectories = True
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error("--temperature must be in [0, 2] and --top-p in (0, 1]")
    if args.max_prompt_length >= args.max_model_length:
        parser.error("--max-prompt-length must be smaller than --max-model-length")
    if args.playbook is not None and not args.playbook.is_file():
        parser.error(f"Playbook does not exist: {args.playbook}")
    return args


def _batch_receipt_path(run_dir: Path, start: int) -> Path:
    return run_dir / "completed_batches" / f"batch_{start:06d}.json"


def _batch_identity(args: argparse.Namespace, task_ids: list[str], playbook_text: str) -> dict:
    return {
        "model": args.model, "tokenizer": args.tokenizer, "model_variant": args.model_variant,
        "step": args.step, "g_act": args.g_act, "seed": args.seed,
        "max_prompt_length": args.max_prompt_length, "max_response_length": args.max_response_length,
        "max_model_length": args.max_model_length, "max_turns": args.max_turns, "temperature": args.temperature, "top_p": args.top_p,
        "playbook_sha256": hashlib.sha256(playbook_text.encode()).hexdigest(), "task_ids": task_ids,
    }


def _load_batch_receipt(path: Path, identity: dict) -> dict | None:
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved.get("version") != 1 or saved.get("identity") != identity:
        raise ValueError(f"Completed Mastermind task batch belongs to a different evaluation: {path}")
    rows = saved.get("tasks")
    trajectories = saved.get("trajectories")
    if not isinstance(rows, list) or [row.get("task_id") for row in rows] != identity["task_ids"] or not isinstance(trajectories, list) or len(trajectories) != len(rows) * identity["g_act"]:
        raise ValueError(f"Incomplete Mastermind task batch receipt: {path}")
    for row in rows:
        samples = row.get("samples", [])
        if [sample.get("attempt_id") for sample in samples] != list(range(identity["g_act"])) or any(sample.get("task_id") != row["task_id"] for sample in samples):
            raise ValueError(f"Incomplete Mastermind task samples: {path}")
    expected_samples = [sample["sample_id"] for row in rows for sample in row["samples"]]
    if [trajectory.get("sample_id") for trajectory in trajectories] != expected_samples:
        raise ValueError(f"Mastermind task batch trajectories differ from its samples: {path}")
    return saved


def _save_batch_receipt(path: Path, identity: dict, task_rows: list[dict], trajectories: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps({"version": 1, "identity": identity, "tasks": task_rows, "trajectories": trajectories}, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def evaluate(args: argparse.Namespace) -> dict:
    # Keep --help and result summarization independent of the execution stack.
    from transformers import AutoTokenizer

    from rllm.context import ContextAwareRunner, NoContextRule
    from rllm.data import DatasetRegistry
    MastermindAgent = _load_mastermind_agent()
    from rllm.engine import AgentExecutionEngine
    from rllm.environments.mastermind.mastermind import MastermindEnv

    dataset = DatasetRegistry.load_dataset("mastermind", "test")
    if dataset is None:
        # Same deterministic tasks as ACE evaluation and training.
        from examples.mastermind.prepare_mastermind_data import prepare_mastermind_data

        prepare_mastermind_data()
        dataset = DatasetRegistry.load_dataset("mastermind", "test")
    if dataset is None:
        raise ValueError("Mastermind test tasks could not be registered.")
    stop = None if args.limit is None else args.offset + args.limit
    # The effective budget replaces each row's max_turns, so trajectory task metadata records it.
    tasks = [{**task, "max_turns": args.max_turns} for task in dataset.get_data()[args.offset : stop]]
    if not tasks:
        raise ValueError("The selected Mastermind test slice is empty.")
    max_steps = args.max_turns
    thinking = args.model_variant == "thinking"
    playbook_text = args.playbook.read_text(encoding="utf-8") if args.playbook is not None else ""
    prompt = _ASSETS / "prompts" / "base_system_prompt.txt" if args.playbook is not None else MastermindAgent.SYSTEM_PROMPT

    class FixedPlaybook(NoContextRule):
        def get_context(self) -> str:
            return playbook_text

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    class EvaluationEngine(AgentExecutionEngine):
        """Carry each sample's seed through the shared engine without global mutation."""

        async def run_agent_trajectory_async(self, idx, application_id, **kwargs):
            task = self.agents[idx].trajectory.task
            return await super().run_agent_trajectory_async(idx, application_id, _evaluation_sampling_seed=task["_evaluation_sampling_seed"], **kwargs)

        async def get_model_response(self, prompt, application_id, **kwargs):
            kwargs["seed"] = kwargs.pop("_evaluation_sampling_seed")
            return await super().get_model_response(prompt, application_id, **kwargs)

    engine = EvaluationEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        agent_class=build_evaluation_agent_class(),
        env_class=MastermindEnv,
        env_args={"max_turns": args.max_turns},
        agent_args={"max_steps": max_steps, "use_accumulate_history": True, "use_accumulate_thinking": thinking, "save_raw_outputs": args.save_trajectories},
        n_parallel_agents=args.parallel,
        max_workers=args.parallel,
        max_steps=max_steps,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        config={"rllm": {"disable_thinking": not thinking, "accumulate_reasoning": thinking, "agent": {"trajectory_reward_aggregation": "max"}}},
        rollout_engine_args={
            "model": args.model,
            "base_url": args.base_url,
            "api_key": os.environ.get("OPENAI_API_KEY") or "EMPTY",
            "max_model_length": args.max_model_length,
            "accumulate_reasoning": thinking,
        },
        sampling_params={"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed},
    )
    try:
        # Keep each task's full sample group in one concurrent runner batch.
        batch_size = (args.parallel // args.g_act) * args.g_act
        runner = ContextAwareRunner(engine, FixedPlaybook(), prompt, batch_size=batch_size, verbose=False)
        run_tag = args.run_tag or f"mastermind_vanilla_{args.model_variant}"
        run_dir = (args.run_dir or args.runs_base / f"{run_tag}_step{args.step}_tasks{task_slice(args, len(tasks))}") if (args.run_dir or args.run_tag or args.save_trajectories) else None
        task_rows, trajectory_rows = [], []
        tasks_per_batch = batch_size // args.g_act
        for task_start in range(0, len(tasks), tasks_per_batch):
            chunk = tasks[task_start : task_start + tasks_per_batch]
            samples = replicate_tasks(chunk, args.g_act, args.seed, offset=args.offset + task_start)
            expected_ids = [str(task.get("uid", task.get("index", args.offset + task_start + index))) for index, task in enumerate(chunk)]
            identity = _batch_identity(args, expected_ids, playbook_text)
            receipt_path = _batch_receipt_path(run_dir, task_start) if run_dir is not None else None
            saved = _load_batch_receipt(receipt_path, identity) if receipt_path is not None and args.resume else None
            if saved is None:
                trajectories = await runner.run(samples, adapt=False)
                if len(trajectories) != len(samples):
                    raise ValueError(f"Expected {len(samples)} Mastermind attempts; received {len(trajectories)}")
                rows = summarize_trajectories(trajectories, args.g_act)["tasks"]
                if [row["task_id"] for row in rows] != expected_ids:
                    raise ValueError("Mastermind task batch returned different task identities")
                payloads = [trajectory_payload(trajectory, index) for index, trajectory in enumerate(trajectories)]
                if receipt_path is not None:
                    _save_batch_receipt(receipt_path, identity, rows, payloads)
            else:
                rows, payloads = saved["tasks"], saved["trajectories"]
            task_rows.extend(rows)
            if args.save_trajectories:
                trajectory_rows.extend(payloads)
    finally:
        try:
            await engine.rollout_engine.client.close()
        finally:
            engine.shutdown()
    return {
        "dataset": "mastermind",
        "split": "test",
        "offset": args.offset,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "model_variant": args.model_variant,
        "playbook": str(args.playbook.resolve()) if args.playbook is not None else None,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed},
        "max_turns": args.max_turns,
        "evaluation_protocol": "independent_samples_v1",
        "sampling_seed_rule": "seed + attempt_id (game seeds unchanged)",
        **summarize_task_rows(task_rows, args.g_act),
        **({"trajectories": trajectory_rows} if args.save_trajectories else {}),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Shared execution utilities print progress; reserve stdout for the JSON result.
    with redirect_stdout(sys.stderr):
        result = asyncio.run(evaluate(args))
    trajectories = result.pop("trajectories", None)
    if args.run_dir or args.run_tag or args.save_trajectories:
        run_tag = args.run_tag or f"mastermind_vanilla_{args.model_variant}"
        run_dir = args.run_dir or args.runs_base / f"{run_tag}_step{args.step}_tasks{task_slice(args, result['num_tasks'])}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / f"metrics_step{args.step}.json"
        if (metrics_path.exists() or (run_dir / "trajectories.json").exists()) and not args.resume:
            raise FileExistsError(f"Evaluation output already exists in {run_dir}; choose a new --run-tag")
        result.update(run_tag=run_tag, step=args.step, run_dir=str(run_dir))
        if trajectories is not None:
            (run_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        metrics_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
