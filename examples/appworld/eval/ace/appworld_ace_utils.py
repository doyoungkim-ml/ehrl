from __future__ import annotations

import copy
import os
from urllib.parse import urlsplit

from transformers import AutoTokenizer

from examples.appworld.eval.ace.training_inputs import AppWorldTrainingInputReflector
from examples.appworld.eval.task_loading import load_appworld_tasks as load_appworld_tasks
from examples.appworld.eval.task_loading import load_specific_tasks as load_specific_tasks
from examples.appworld.train.ace_rl.appworld_adapter import AppWorldReflectorAdapter
from rllm.agents.appworld_react_agents_ace import AppWorldReactAgentACE
from rllm.algorithms.ace.context import ACEConfig, ACEContextRule
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout import OpenAIEngine
from rllm.environments.appworld.appworld_env import AppWorldEnv
from rllm.utils.client_factory import create_client, is_external_model


def prepare_local_worker_args(args, worker_index: int, worker_count: int):
    """Reserve enough stateful AppWorld servers for every active trajectory."""
    n_servers = int(getattr(args, "n_servers", 1))
    n_parallel_agents = int(getattr(args, "n_parallel_agents", 1))
    base_port = int(getattr(args, "base_port", 9000))
    g_act = int(getattr(args, "g_act", 1))
    if g_act < 1 or worker_count < 1 or worker_count * g_act > min(n_servers, n_parallel_agents):
        raise ValueError("AppWorld task concurrency needs one agent and one stateful server per trajectory")
    if not 0 <= worker_index < worker_count:
        raise ValueError("AppWorld worker index is outside the configured worker count")
    if not 1 <= base_port <= 65535 or base_port + n_servers - 1 > 65535:
        raise ValueError("AppWorld server ports must fit in 1..65535")
    url = urlsplit(getattr(args, "remote_environment_url", None) or f"http://127.0.0.1:{base_port}")
    if url.scheme not in {"http", "https"} or not url.hostname or url.port != base_port:
        raise ValueError("Concurrent AppWorld evaluation needs a server URL whose port matches base_port")
    host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
    userinfo = url.netloc.rsplit("@", 1)[0] + "@" if "@" in url.netloc else ""
    worker_args = copy.deepcopy(args)
    if g_act == 1:
        # A single server bypasses AppWorldEnv's process-global port pool.
        worker_args.base_port = base_port + worker_index
        worker_args.n_servers = 1
    else:
        # All workers share one port pool. AppWorldEnv checks out a unique port
        # for each trajectory and returns it only after closing that task world.
        worker_args.base_port = base_port
        worker_args.n_servers = n_servers
    worker_args.remote_environment_url = url._replace(netloc=f"{userinfo}{host}:{worker_args.base_port}").geturl()
    worker_args.n_parallel_agents = g_act
    return worker_args


def build_generator_engine(args) -> AgentExecutionEngine:
    n_servers = getattr(args, "n_servers", 1)
    base_port = getattr(args, "base_port", 9000)
    if n_servers < 1 or not 1 <= args.n_parallel_agents <= n_servers:
        raise ValueError("AppWorld evaluation requires 1 <= n_parallel_agents <= n_servers; each stateful server owns one concurrent task")
    if not 1 <= base_port <= 65535 or base_port + n_servers - 1 > 65535:
        raise ValueError("AppWorld server ports must fit in 1..65535")
    if getattr(args, "training_input_parity", False) and is_external_model(args.model_name):
        raise ValueError("AppWorld training input parity requires a local generator tokenizer and completion engine")
    if is_external_model(args.model_name):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    max_response_length = getattr(args, "max_response_length", 16384)
    max_prompt_length = getattr(args, "max_prompt_length", 16384)
    accumulate_thinking = getattr(args, "accumulate_thinking", True)
    disable_thinking = getattr(args, "disable_thinking", False)
    accumulate_reasoning = getattr(args, "accumulate_reasoning", True)
    gen_temperature = getattr(args, "gen_temperature", 0.6)
    gen_top_p = getattr(args, "gen_top_p", 0.95)
    print(
        f"[Generator] agent engine: max_prompt_length={max_prompt_length} "
        f"max_response_length={max_response_length} max_steps={args.max_turns} "
        f"temperature={gen_temperature} top_p={gen_top_p} "
        f"accumulate_thinking={accumulate_thinking} disable_thinking={disable_thinking} "
        f"accumulate_reasoning={accumulate_reasoning}",
        flush=True,
    )

    engine = AgentExecutionEngine(
        agent_class=AppWorldReactAgentACE,
        env_class=AppWorldEnv,
        agent_args={
            "accumulate_thinking": accumulate_thinking,
        },
        env_args={
            "max_turns": args.max_turns,
            "remote_environment_url": args.remote_environment_url or f"http://127.0.0.1:{base_port}",
            "n_servers": n_servers,
            "base_port": base_port,
        },
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={"temperature": gen_temperature, "top_p": gen_top_p, "model": args.model_name},
        rollout_engine_args={
            "base_url": args.base_url,
            "api_key": args.api_key or "None",
            "debug": args.debug,
            "max_model_length": args.max_model_length,
            # Generation-prompt rendering knob: OpenAIEngine.get_model_response falls back
            # to ITS OWN accumulate_reasoning (default False) — AgentExecutionEngine does
            # not forward the config.rllm value here, so it must be passed explicitly or
            # arTrue runs would render prompts without accumulated reasoning (training's
            # VerlEngine reads it from config.rllm — see verl_engine.py:40,80).
            # External-generator (chat-completions) path rejects True; pass
            # --no-accumulate_reasoning when evaluating Gemini/GPT generators.
            "accumulate_reasoning": accumulate_reasoning,
        },
        # AgentExecutionEngine reads disable_thinking/accumulate_reasoning only from
        # config.rllm.* (see rllm/engine/agent_execution_engine.py:91-92) — a minimal config
        # dict is required for the flags to reach the Qwen chat parser / generation stream;
        # no-op for external models since the OpenAI engine swaps to
        # _use_chat_completions=True and skips the parser.
        config={"rllm": {"disable_thinking": disable_thinking, "accumulate_reasoning": accumulate_reasoning}},
        max_response_length=max_response_length,
        max_prompt_length=max_prompt_length,
        n_parallel_agents=args.n_parallel_agents,
        max_steps=args.max_turns,
        trajectory_hard_timeout=getattr(args, "trajectory_hard_timeout", 7200),
    )

    if getattr(args, "training_input_parity", False):
        # VeRL uses the configured capacity without the legacy extra token reserve.
        engine.rollout_engine.max_model_length = args.max_model_length
        engine.sampling_params["extra_body"] = {"top_k": -1}

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
    return engine


def create_ace_context_rule(args, prompts_dir: str, initial_playbook_path: str, ace_log_dir: str | None) -> ACEContextRule:
    from rllm.algorithms.ace.evaluation import evaluation_role_prompt_paths

    training_input_parity = bool(getattr(args, "training_input_parity", False))
    handoff_protocol = getattr(args, "handoff_protocol", "legacy")
    # Preserve the recognized legacy resolver for verified historical replay.
    reflector_prompt_path = os.path.join(prompts_dir, "reflector_prompt.txt")
    curator_prompt_path = os.path.join(prompts_dir, "curator_prompt.txt")
    if handoff_protocol != "legacy":
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
    if training_input_parity and is_external_model(effective_ace_model):
        raise ValueError("AppWorld training input parity requires a local model tokenizer and completion engine")
    # Single knob: Reflector/Curator inherit the SAME disable_thinking as the Generator/policy
    # (build_generator_engine), so one flag controls all three roles — mirroring the training
    # path where all three share one rollout engine (same fix as mbpp/alfworld June harnesses).
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
        # Training swaps role limits on one engine. Evaluation uses separate handles
        # on the same server. Unset role limits preserve the legacy evaluation budget.
        reflector_max_prompt = getattr(args, "reflector_max_prompt_length", None) or args.max_model_length
        reflector_max_response = getattr(args, "reflector_max_response_length", None) or 16384
        curator_max_prompt = getattr(args, "curator_max_prompt_length", None) or args.max_model_length
        curator_max_response = getattr(args, "curator_max_response_length", None) or 16384
        print(
            f"[ACE] role budgets: reflector {reflector_max_prompt}/{reflector_max_response} curator {curator_max_prompt}/{curator_max_response} (prompt/response)",
            flush=True,
        )
        print(
            f"[ACE] role system prompts: reflector={'SET' if reflector_system_prompt else 'none'} curator={'SET' if curator_system_prompt else 'none'}",
            flush=True,
        )

        def _role_engine(max_prompt_length: int, max_response_length: int) -> OpenAIEngine:
            engine = OpenAIEngine(
                model=effective_ace_model,
                tokenizer=tokenizer,
                base_url=args.ace_base_url or args.base_url,
                api_key="None",
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                max_model_length=args.max_model_length,
                sampling_params={"temperature": args.ace_temperature, **({"top_p": 1.0, "extra_body": {"top_k": -1}} if training_input_parity else {})},
                debug=args.debug,
                disable_thinking=ace_disable_thinking,
            )
            if training_input_parity:
                engine.max_model_length = args.max_model_length
            return engine

        ace_rollout_engine = _role_engine(reflector_max_prompt, reflector_max_response)
        curator_rollout_engine = _role_engine(curator_max_prompt, curator_max_response)

    config = ACEConfig(
        reflector_prompt_path=reflector_prompt_path,
        curator_prompt_path=curator_prompt_path,
        initial_playbook_path=initial_playbook_path,
        model_name=effective_ace_model,
        temperature=args.ace_temperature,
        ace_log_dir=ace_log_dir,
    )
    # The adapter history cap applies to both evaluation protocols. Training input
    # parity also fits the fully chat-templated reflector prompt below.
    role_tokenizer = None
    if not is_external_model(effective_ace_model):
        role_tokenizer = AutoTokenizer.from_pretrained(effective_ace_model)
    rule = ACEContextRule(
        config,
        rollout_engine=ace_rollout_engine,
        reflector_adapter=AppWorldReflectorAdapter(
            max_turns=args.max_turns,
            tokenizer=role_tokenizer,
            max_history_tokens=getattr(args, "reflector_max_history_tokens", 20000),
            handoff_protocol=handoff_protocol,
        ),
        reflector_system_prompt=reflector_system_prompt,
        curator_system_prompt=curator_system_prompt,
    )
    if training_input_parity:
        rule.reflector = AppWorldTrainingInputReflector(
            prompt_template=rule.reflector.prompt_template,
            adapter=rule.reflector.adapter,
            rollout_engine=ace_rollout_engine,
            model_name=effective_ace_model,
            temperature=args.ace_temperature,
            max_retries=config.max_retries,
            system_prompt=reflector_system_prompt,
        )
    if role_tokenizer is not None:
        rule.curator.tokenizer = role_tokenizer
    # Curator gets its own budgeted engine; Reflector/Curator read self.rollout_engine
    # at call time so this reassignment is the full wiring. External ace_model keeps
    # one shared engine (training never used external ACE roles).
    if not is_external_model(effective_ace_model):
        rule.curator.rollout_engine = curator_rollout_engine
    return rule


def trajectory_reward(trajectory) -> float:
    rewards = []
    if getattr(trajectory, "reward", None) is not None:
        rewards.append(float(trajectory.reward))
    for step in getattr(trajectory, "steps", []) or []:
        if getattr(step, "reward", None) is not None:
            rewards.append(float(step.reward))
    return max(rewards) if rewards else 0.0


def trajectory_success(trajectory) -> bool:
    return trajectory_reward(trajectory) > 0.0


def task_id_getter(task: dict) -> str:
    return str(task["task_id"])
