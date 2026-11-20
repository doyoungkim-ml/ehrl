from __future__ import annotations

import os

from transformers import AutoTokenizer

from examples.mbpp.train.ace_rl.mbpp_adapter import MBPPReflectorAdapter
from rllm.agents.mbpp_agent import MBPPAgent
from rllm.algorithms.ace.context import ACEConfig, ACEContextRule
from rllm.algorithms.ace.evaluation import evaluation_role_prompt_paths, reserve_worker_parallel_agents
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout import OpenAIEngine
from rllm.environments.mbpp.mbpp import MBPPEnv
from rllm.utils.client_factory import create_client, is_external_model


def load_train_data(script_dir: str, data_dir: str | None = None):
    if DatasetRegistry.dataset_exists("mbpp", "train"):
        train_dataset = DatasetRegistry.load_dataset("mbpp", "train")
        return train_dataset.get_data()

    from examples.mbpp.prepare_mbpp_data import prepare_mbpp_data

    if data_dir and os.path.exists(os.path.join(data_dir, "train.parquet")):
        train_dataset, _ = prepare_mbpp_data(local_dir=data_dir, load_from_local=True)
        return train_dataset.get_data()

    train_dataset, _ = prepare_mbpp_data(local_dir=data_dir)
    return train_dataset.get_data()


def load_test_data(script_dir: str, data_dir: str | None = None):
    if DatasetRegistry.dataset_exists("mbpp", "test"):
        test_dataset = DatasetRegistry.load_dataset("mbpp", "test")
        return test_dataset.get_data()

    from examples.mbpp.prepare_mbpp_data import prepare_mbpp_data

    if data_dir and os.path.exists(os.path.join(data_dir, "test.parquet")):
        _, test_dataset = prepare_mbpp_data(local_dir=data_dir, load_from_local=True)
        return test_dataset.get_data()

    _, test_dataset = prepare_mbpp_data(local_dir=data_dir)
    return test_dataset.get_data()


def prepare_local_worker_args(args, worker_index: int, worker_count: int):
    """Each concurrent local task worker gets its own generator engine; MBPP test execution shares a locked process pool."""
    return reserve_worker_parallel_agents(args, worker_index, worker_count, benchmark="MBPP")


def build_generator_engine(args) -> AgentExecutionEngine:
    if is_external_model(args.model_name):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        print(f"[Generator] External model: {args.model_name} (proxy tokenizer)")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    engine = AgentExecutionEngine(
        agent_class=MBPPAgent,
        env_class=MBPPEnv,
        # Thinking flags aligned to mbpp_local_ace_rl training (wandb ny9pmtnv):
        # disable_thinking=False, accumulate_thinking=True, accumulate_reasoning=True.
        # disable_thinking is the single knob shared with the Reflector/Curator (create_ace_context_rule);
        # default False keeps mbpp thinking-ON, matching training.
        config={"rllm": {"agent": {"trajectory_reward_aggregation": "max"}, "disable_thinking": getattr(args, "disable_thinking", False), "accumulate_reasoning": True}},
        agent_args={"max_steps": args.max_turns, "accumulate_history": True, "accumulate_thinking": True},
        env_args={"max_turns": args.max_turns, "timeout": 10, "feedback": getattr(args, "feedback", "full")},
        max_steps=args.max_turns,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={
            "temperature": getattr(args, "gen_temperature", 0.6),
            "top_p": getattr(args, "gen_top_p", 0.95),
            "model": args.model_name,
        },
        rollout_engine_args={
            "base_url": args.base_url,
            "api_key": args.api_key or "None",
            "debug": args.debug,
            "max_model_length": args.max_model_length,
        },
        max_response_length=getattr(args, "gen_max_response_length", 16384),
        max_prompt_length=getattr(args, "gen_max_prompt_length", 16384),
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
    handoff_protocol = getattr(args, "handoff_protocol", "legacy")
    reflector_prompt_path, curator_prompt_path = evaluation_role_prompt_paths(prompts_dir, handoff_protocol)
    for path, name in [
        (reflector_prompt_path, "reflector prompt"),
        (curator_prompt_path, "curator prompt"),
        (initial_playbook_path, "initial playbook"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ACE {name} file not found: {path}")

    # Role system prompts — ACE-RL TRAINING always sets these (mbpp: trainer code
    # defaults in ace_rl_trainer_base.py:323-331, NOT visible in wandb; alfworld: slurm
    # envs). "" = legacy user-only messages (all evals before 2026-07-08 ran that way).
    reflector_system_prompt = getattr(args, "reflector_system_prompt", "") or ""
    curator_system_prompt = getattr(args, "curator_system_prompt", "") or ""
    effective_ace_model = args.ace_model or args.model_name
    # Single knob: Reflector/Curator inherit the SAME disable_thinking as the Generator/policy
    # (build_generator_engine), so one flag controls all three — mirroring the training path where
    # all three share one rollout engine. Local-tokenizer path only; no-op for external ace_model.
    ace_disable_thinking = getattr(args, "disable_thinking", False)
    if is_external_model(effective_ace_model):
        ace_rollout_engine = OpenAIEngine(
            model=effective_ace_model,
            tokenizer=None,
            base_url="placeholder",
            api_key="placeholder",
            max_prompt_length=100000,
            max_response_length=16384,
            max_model_length=100000,
            sampling_params={"temperature": args.ace_temperature},
            debug=args.debug,
            disable_thinking=ace_disable_thinking,
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
        # Per-role budgets. The slurm harness passes the ACE-RL TRAINING values (wandb
        # rllm.ace, all June mbpp runs: reflector 20480/2048, curator 8192/4096 — aligned
        # with Doyoung 2026-07-07). Training swaps these limits on ONE shared engine per
        # phase (ace_rl_trainer_base.py:_set_component_token_limits); eval mirrors it with
        # two engine handles on the same vLLM server — identical effect, race-free.
        # Fallbacks = legacy convention (16384/16384, what all pre-2026-07-07 evals used).
        reflector_max_prompt = getattr(args, "reflector_max_prompt_length", 16384)
        reflector_max_response = getattr(args, "reflector_max_response_length", 16384)
        curator_max_prompt = getattr(args, "curator_max_prompt_length", 16384)
        curator_max_response = getattr(args, "curator_max_response_length", 16384)
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
        reflector_adapter=MBPPReflectorAdapter(max_turns=args.max_turns, evaluation_summary=True, handoff_protocol=handoff_protocol),
        reflector_system_prompt=reflector_system_prompt,
        curator_system_prompt=curator_system_prompt,
    )
    # Curator gets its own budgeted engine; Reflector/Curator read self.rollout_engine
    # at call time so this reassignment is the full wiring. External ace_model keeps
    # one shared engine (training never used external ACE roles).
    if not is_external_model(effective_ace_model):
        rule.curator.rollout_engine = curator_rollout_engine
    return rule


def trajectory_reward(trajectory) -> float:
    if getattr(trajectory, "reward", None) is not None:
        return float(trajectory.reward)
    return 0.0


def trajectory_success(trajectory) -> bool:
    return trajectory_reward(trajectory) == 1.0


def task_id_getter(task: dict) -> str:
    return str(task["task_id"])
