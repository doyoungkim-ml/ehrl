"""Mini-SWE evaluation glue for generator engines and per-task ACE context.

Generator budgets and thinking behavior are configurable independently of
reflection and curation. Prompt variants are selected explicitly by the CLI.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from transformers import AutoTokenizer

from examples.mini_swe_bench.env import MiniSWEnv
from examples.mini_swe_bench.preprocess_swegym import get_default_data_dir, preprocess_swegym
from examples.mini_swe_bench.train.ace_rl.adapter import MiniSweReflectorAdapter
from rllm.agents.swe_agent import SWEAgent
from rllm.algorithms.ace.context import ACEConfig, ACEContextRule
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout import OpenAIEngine
from rllm.utils.client_factory import create_client, is_external_model

SPLIT_TO_PARQUET = {
    "validation": "validation.parquet",
    "train": "train.parquet",
}

# 8192 (2026-06-02): measured SWE-bench_Verified initial prompts (system +
# problem_statement, Qwen chat template) — max=6285 over the 100-task subset, so
# 8192 rejects 0 tasks. Aligns with AppWorld's max_prompt_length. The gate is the
# initial prompt only; the growing per-turn prompt is bounded by max_model_length.
DEFAULT_MAX_PROMPT_LENGTH = 8192
DEFAULT_MAX_RESPONSE_LENGTH = 32768  # bumped 2026-05-19 — 16384 caused universal TRUNCATION on mini_swe_bench's 30-turn agent rollouts


def _resolve_data_dir(data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir)
    return get_default_data_dir()


def _ensure_parquet(data_dir: Path) -> None:
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    if train_path.exists() and val_path.exists():
        return
    print(f"[load_swe_tasks] parquet missing under {data_dir}; running preprocess_swegym()...")
    preprocess_swegym(output_dir=data_dir)


def load_swe_tasks(split: str, data_dir: str | None = None) -> list[dict]:
    if split not in SPLIT_TO_PARQUET:
        raise ValueError(f"Unknown split {split!r}; valid: {sorted(SPLIT_TO_PARQUET)}")
    resolved = _resolve_data_dir(data_dir)
    _ensure_parquet(resolved)
    parquet_path = resolved / SPLIT_TO_PARQUET[split]

    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path))
    return table.to_pylist()


def task_id_getter(task: dict) -> str:
    instance = task.get("instance", task)
    return str(instance.get("instance_id", ""))


def trajectory_reward(trajectory) -> float:
    reward = getattr(trajectory, "reward", None)
    return float(reward) if reward is not None else 0.0


def trajectory_success(trajectory) -> bool:
    return trajectory_reward(trajectory) == 1.0


def _resolve_scratch_root() -> str:
    explicit = os.environ.get("MINISW_SCRATCH_ROOT")
    if explicit:
        return explicit
    from examples.mini_swe_bench.shared.apptainer_utils import get_scratch_root

    return str(get_scratch_root())


def prepare_local_worker_args(args, worker_index: int, worker_count: int):
    """Give each concurrent local task worker an isolated single-task generator engine.

    Every MiniSWEnv instance owns its own apptainer testbed, overlay and grading
    directory, so workers only need separate engines. Total in-flight
    trajectories (workers x g_act) must fit the configured parallel-agent
    capacity; each worker engine reserves exactly its own g_act slots.
    """
    g_act = max(1, int(getattr(args, "g_act", 1)))
    n_parallel_agents = int(getattr(args, "n_parallel_agents", 1))
    if worker_count < 1 or worker_count * g_act > n_parallel_agents:
        raise ValueError("MiniSWE task concurrency needs n_parallel_agents >= task_concurrency * g_act")
    if not 0 <= worker_index < worker_count:
        raise ValueError("MiniSWE worker index is outside the configured worker count")
    worker_args = copy.deepcopy(args)
    worker_args.n_parallel_agents = g_act
    return worker_args


def build_generator_engine(args) -> AgentExecutionEngine:
    if is_external_model(args.model_name):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        print(f"[Generator] External model: {args.model_name} (proxy tokenizer)")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    scratch_root = _resolve_scratch_root()
    step_timeout = int(os.environ.get("MINISW_STEP_TIMEOUT", "120"))
    reward_timeout = int(os.environ.get("MINISW_REWARD_TIMEOUT", "600"))

    max_prompt_length = getattr(args, "max_prompt_length", DEFAULT_MAX_PROMPT_LENGTH)
    max_response_length = getattr(args, "max_response_length", DEFAULT_MAX_RESPONSE_LENGTH)
    accumulate_thinking = getattr(args, "accumulate_thinking", False)
    disable_thinking = getattr(args, "disable_thinking", False)
    print(
        f"[Generator] agent engine: max_prompt_length={max_prompt_length} "
        f"max_response_length={max_response_length} max_steps={args.max_turns} "
        f"accumulate_thinking={accumulate_thinking} disable_thinking={disable_thinking}",
        flush=True,
    )

    engine = AgentExecutionEngine(
        agent_class=SWEAgent,
        env_class=MiniSWEnv,
        agent_args={
            "scaffold": "sweagent",
            "use_fn_calling": False,
            "format_model_response": False,
            "accumulate_thinking": accumulate_thinking,
        },
        # AgentExecutionEngine reads disable_thinking only from config.rllm.disable_thinking
        # (agent_execution_engine.py:89). Build a minimal config dict so the flag reaches
        # the Qwen chat parser; no-op for Gemini (chat-completions path skips the parser).
        config={"rllm": {"disable_thinking": disable_thinking}},
        env_args={
            "max_steps": args.max_turns,
            "step_timeout": step_timeout,
            "reward_timeout": reward_timeout,
            "scratch_root": scratch_root,
        },
        max_steps=args.max_turns,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={"temperature": 0.6, "top_p": 0.95, "model": args.model_name},
        rollout_engine_args={
            "base_url": args.base_url,
            "api_key": args.api_key or "None",
            "debug": args.debug,
            "max_model_length": args.max_model_length,
        },
        max_response_length=max_response_length,
        max_prompt_length=max_prompt_length,
        n_parallel_agents=args.n_parallel_agents,
    )

    if is_external_model(args.model_name):
        engine.rollout_engine.client = create_client(
            model=args.model_name,
            base_url=args.base_url,
            api_key=args.api_key,
            key_file=args.api_key_file,
            project_id=args.project_id,
        )
        engine.rollout_engine._use_chat_completions = True
        engine.rollout_engine.model = args.model_name
        print(f"[Generator] Swapped to external client for: {args.model_name}")

    return engine


def create_ace_context_rule(args, prompts_dir: str, initial_playbook_path: str, ace_log_dir: str | None) -> ACEContextRule:
    from rllm.algorithms.ace.evaluation import evaluation_role_prompt_paths

    handoff_protocol = getattr(args, "handoff_protocol", "legacy")
    reflector_prompt_path, curator_prompt_path = evaluation_role_prompt_paths(prompts_dir, handoff_protocol)
    for path, name in [
        (reflector_prompt_path, "reflector prompt"),
        (curator_prompt_path, "curator prompt"),
        (initial_playbook_path, "initial playbook"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ACE {name} file not found: {path}")

    effective_ace_model = args.ace_model or args.model_name
    if is_external_model(effective_ace_model):
        ace_rollout_engine = OpenAIEngine(
            model=effective_ace_model,
            tokenizer=None,
            base_url="placeholder",
            api_key="placeholder",
            max_prompt_length=100000,
            max_response_length=32768,
            max_model_length=100000,
            sampling_params={"temperature": args.ace_temperature},
            debug=args.debug,
        )
        ace_rollout_engine.client = create_client(
            model=effective_ace_model,
            base_url=args.ace_base_url or args.base_url,
            api_key=args.ace_api_key,
            key_file=args.ace_key_file,
            project_id=args.ace_project_id,
            include_thoughts=args.include_thoughts,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(effective_ace_model)
        ace_rollout_engine = OpenAIEngine(
            model=effective_ace_model,
            tokenizer=tokenizer,
            base_url=args.ace_base_url or args.base_url,
            api_key="None",
            max_prompt_length=args.max_model_length,
            max_response_length=32768,
            max_model_length=args.max_model_length,
            sampling_params={"temperature": args.ace_temperature},
            debug=args.debug,
        )

    config = ACEConfig(
        reflector_prompt_path=reflector_prompt_path,
        curator_prompt_path=curator_prompt_path,
        initial_playbook_path=initial_playbook_path,
        save_playbook_path=getattr(args, "save_playbook", "") or "",
        model_name=effective_ace_model,
        temperature=args.ace_temperature,
        ace_log_dir=ace_log_dir,
        reflector_only=getattr(args, "reflector_only", False),
    )
    # Same role system prompts as ACE-RL training, so both read the prompts identically.
    from examples.mini_swe_bench.train.ace_rl.train import ACE_DEFAULTS

    return ACEContextRule(
        config,
        rollout_engine=ace_rollout_engine,
        reflector_system_prompt=ACE_DEFAULTS["reflector_system_prompt"],
        curator_system_prompt=ACE_DEFAULTS["curator_system_prompt"],
        reflector_adapter=MiniSweReflectorAdapter(
            max_turns=args.max_turns,
            obs_max_chars=getattr(args, "reflector_obs_max_chars", 300),
            test_sample_n=getattr(args, "reflector_test_sample_n", 0),
            simplified_history=getattr(args, "reflector_simplified_history", True),
            handoff_protocol=handoff_protocol,
        ),
    )
