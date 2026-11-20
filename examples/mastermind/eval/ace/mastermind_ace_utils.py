"""Mastermind hooks for the shared Local-ACE evaluation runner.

Mirrors examples/mbpp/eval/ace/mbpp_ace_utils.py. Differences are all Mastermind's:
agent kwargs are use_accumulate_* (not accumulate_*), env args carry the per-task game
shape, a game's reward is its BEST guess (aggregation=max) and only a fully solved code
reaches 1.0, and the task id is the dataset `uid`.

Suffix-free name because this file is the benchmark's ACE-eval glue, same role as the
mbpp/appworld/alfworld *_ace_utils.py files.
"""

from __future__ import annotations

from transformers import AutoTokenizer

from examples.mastermind.train.ace_rl.mastermind_adapter import MastermindReflectorAdapter
from rllm.agents.mastermind_agent import MastermindAgent
from rllm.algorithms.ace.context import ACEConfig, ACEContextRule
from rllm.algorithms.ace.evaluation import reserve_worker_parallel_agents
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout import OpenAIEngine
from rllm.environments.mastermind.mastermind import MastermindEnv
from rllm.utils.client_factory import create_client, is_external_model

# Proxy tokenizer for external generators (Gemini): only used for length bookkeeping.
_EXTERNAL_PROXY_TOKENIZER = "Qwen/Qwen3-0.6B"


def _gemini_thinking_extra(model_name: str, thinking_budget: int) -> dict:
    """extra_body for Gemini's OpenAI-compat layer (empty for non-Gemini models).

    Same as the alfworld eval helper: on Gemini 2.5 thinking is on by default and thought
    tokens spend max_tokens, so an uncapped budget can starve the visible answer.
    include_thoughts stays False because the Reflector/Curator parse ModelOutput.text.
    """
    if not (model_name or "").lower().startswith("gemini"):
        return {}
    return {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "thinking_budget": thinking_budget,
                    "include_thoughts": False,
                }
            }
        }
    }


def _load_split(split: str):
    if not DatasetRegistry.dataset_exists("mastermind", split):
        from examples.mastermind.prepare_mastermind_data import prepare_mastermind_data

        prepare_mastermind_data()
    dataset = DatasetRegistry.load_dataset("mastermind", split)
    if dataset is None:
        raise RuntimeError(f"Mastermind {split} split is not registered; run python -m examples.mastermind.prepare_mastermind_data")
    return dataset.get_data()


def load_train_data(script_dir: str, data_dir: str | None = None):
    return _load_split("train")


def load_test_data(script_dir: str, data_dir: str | None = None):
    return _load_split("test")


def prepare_local_worker_args(args, worker_index: int, worker_count: int):
    """Each concurrent local task worker gets its own generator engine; Mastermind games are in-process per trajectory."""
    return reserve_worker_parallel_agents(args, worker_index, worker_count, benchmark="Mastermind")


def build_generator_engine(args) -> AgentExecutionEngine:
    if is_external_model(args.model_name):
        tokenizer = AutoTokenizer.from_pretrained(_EXTERNAL_PROXY_TOKENIZER)
        print(f"[Generator] External model: {args.model_name} (proxy tokenizer)")
    else:
        tokenizer = AutoTokenizer.from_pretrained(getattr(args, "tokenizer", None) or args.model_name)

    # Thinking: Mastermind training ran both arms (wandb oyt135wd nonthinking,
    # 9wcyzbbv thinking). disable_thinking is the single knob shared with the
    # Reflector/Curator, same as mbpp. Default False = thinking ON: the vanilla
    # baseline scored 0.334 with thinking vs 0.105 without (2026-09-11).
    disable_thinking = getattr(args, "disable_thinking", False)
    accumulate_thinking = not disable_thinking

    engine = AgentExecutionEngine(
        agent_class=MastermindAgent,
        env_class=MastermindEnv,
        config={
            "rllm": {
                # A game's reward is its best guess, matching training (common.sh:62).
                "agent": {"trajectory_reward_aggregation": "max"},
                "disable_thinking": disable_thinking,
                "accumulate_reasoning": accumulate_thinking,
            }
        },
        # MastermindAgent takes use_accumulate_* (mbpp's accumulate_* would raise TypeError).
        agent_args={
            "max_steps": args.max_turns,
            "use_accumulate_history": True,
            "use_accumulate_thinking": accumulate_thinking,
        },
        # code_length / num_digits / seed come from each task row. env_args override row
        # fields, so this max_turns replaces every row's budget (the engine merges {**row, **env_args}).
        env_args={"max_turns": args.max_turns},
        max_steps=args.max_turns,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={
            "temperature": getattr(args, "gen_temperature", 0.7),
            "top_p": getattr(args, "gen_top_p", 1.0),
            "model": args.model_name,
        },
        rollout_engine_args={
            "base_url": args.base_url,
            "api_key": args.api_key or "None",
            "debug": args.debug,
            "max_model_length": args.max_model_length,
        },
        max_response_length=getattr(args, "gen_max_response_length", 32768),
        max_prompt_length=getattr(args, "gen_max_prompt_length", 8192),
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
    import os

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

    reflector_system_prompt = getattr(args, "reflector_system_prompt", "") or ""
    curator_system_prompt = getattr(args, "curator_system_prompt", "") or ""
    effective_ace_model = args.ace_model or args.model_name
    ace_disable_thinking = getattr(args, "disable_thinking", False)

    if is_external_model(effective_ace_model):
        # Qwen generator + Gemini ACE roles: one shared engine, external client.
        ace_rollout_engine = OpenAIEngine(
            model=effective_ace_model,
            tokenizer=None,
            base_url="placeholder",
            api_key="placeholder",
            max_prompt_length=100000,
            max_response_length=16384,
            max_model_length=100000,
            sampling_params={
                "temperature": args.ace_temperature,
                **_gemini_thinking_extra(effective_ace_model, getattr(args, "ace_thinking_budget", 2048)),
            },
            debug=args.debug,
            disable_thinking=ace_disable_thinking,
        )
        print(
            f"[ACE] external roles: model={effective_ace_model} thinking_budget={getattr(args, 'ace_thinking_budget', 2048)} temperature={args.ace_temperature} (single shared engine for both roles)",
            flush=True,
        )
        ace_rollout_engine.client = create_client(
            model=effective_ace_model,
            base_url=args.ace_base_url or args.base_url,
            api_key=args.ace_api_key,
            key_file=args.ace_key_file,
            project_id=args.ace_project_id,
            include_thoughts=args.include_thoughts,
        )
        curator_rollout_engine = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(getattr(args, "tokenizer", None) or effective_ace_model)
        # Per-role budgets, passed by the launcher from the ACE-RL TRAINING values
        # (wandb hydybnem: reflector 30720/2048, curator 8192/4096).
        reflector_max_prompt = getattr(args, "reflector_max_prompt_length", 30720)
        reflector_max_response = getattr(args, "reflector_max_response_length", 2048)
        curator_max_prompt = getattr(args, "curator_max_prompt_length", 8192)
        curator_max_response = getattr(args, "curator_max_response_length", 4096)
        print(
            f"[ACE] role budgets: reflector {reflector_max_prompt}/{reflector_max_response} curator {curator_max_prompt}/{curator_max_response} (prompt/response)",
            flush=True,
        )
        print(
            f"[ACE] role system prompts: reflector={'SET' if reflector_system_prompt else 'none'} curator={'SET' if curator_system_prompt else 'none'}",
            flush=True,
        )

        def _role_engine(max_prompt_length: int, max_response_length: int) -> OpenAIEngine:
            return OpenAIEngine(
                model=effective_ace_model,
                tokenizer=tokenizer,
                base_url=args.ace_base_url or args.base_url,
                api_key="None",
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                max_model_length=args.max_model_length,
                sampling_params={"temperature": args.ace_temperature},
                debug=args.debug,
                disable_thinking=ace_disable_thinking,
            )

        ace_rollout_engine = _role_engine(reflector_max_prompt, reflector_max_response)
        curator_rollout_engine = _role_engine(curator_max_prompt, curator_max_response)

    config = ACEConfig(
        reflector_prompt_path=reflector_prompt_path,
        curator_prompt_path=curator_prompt_path,
        initial_playbook_path=initial_playbook_path,
        save_playbook_path=getattr(args, "save_playbook", "") or "",
        model_name=effective_ace_model,
        temperature=args.ace_temperature,
        ace_log_dir=ace_log_dir,
    )
    rule = ACEContextRule(
        config,
        rollout_engine=ace_rollout_engine,
        reflector_adapter=MastermindReflectorAdapter(max_turns=args.max_turns, handoff_protocol=handoff_protocol),
        reflector_system_prompt=reflector_system_prompt,
        curator_system_prompt=curator_system_prompt,
    )
    if curator_rollout_engine is not None:
        rule.curator.rollout_engine = curator_rollout_engine
    return rule


def trajectory_reward(trajectory) -> float:
    if getattr(trajectory, "reward", None) is not None:
        return float(trajectory.reward)
    return 0.0


def trajectory_success(trajectory) -> bool:
    # Only a fully solved code scores 1.0; partial digit matches never reach it.
    return trajectory_reward(trajectory) >= 1.0


def task_id_getter(task: dict) -> str:
    return str(task.get("uid") or task.get("index"))
