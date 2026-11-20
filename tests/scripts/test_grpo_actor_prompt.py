"""GRPO_ACTOR_PROMPT gives the GRPO actor ACE-RL's step-0 prompt and nothing else."""

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rllm.agents.agent import BaseAgent
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.utils.run_state import training_identity

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("mastermind", "mbpp", "appworld", "mini_swe_bench")
# Modules that define each benchmark's ACE_DEFAULTS.
ACE_MODULES = {
    "mastermind": "examples/mastermind/train/ace_rl/train.py",
    "mbpp": "examples/mbpp/train/ace_rl/training_entrypoints.py",
    "appworld": "examples/appworld/train/ace_rl/training_entrypoints.py",
    "mini_swe_bench": "examples/mini_swe_bench/train/ace_rl/train.py",
}
ACE_AGENTS = {
    "mastermind": "rllm.agents.mastermind_agent.MastermindAgent",
    "mbpp": "rllm.agents.mbpp_agent.MBPPAgent",
    "appworld": "rllm.agents.appworld_react_agents_ace.AppWorldReactAgentACE",
    "mini_swe_bench": "examples.mini_swe_bench.train.ace_rl.agent.MiniSweACEAgent",
}
# ACE-RL launchers default to a different turn budget; compare at a common one.
TURNS = {"mastermind": ("MASTERMIND_MAX_TURNS", "5"), "mbpp": ("MBPP_MAX_TURNS", "6"), "appworld": ("APPWORLD_MAX_TURNS", "10"), "mini_swe_bench": ("MINISW_MAX_STEPS", "10")}
APPWORLD_TASK = {
    "instruction": "Like every song in my Spotify queue.",
    "user_info": {"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com", "phone_number": "+15550100"},
    "app_descriptions": "spotify: music app",
}
SWE_PROBLEM = "Fix the off-by-one error in utils.window()."


def import_object(path):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def run_launcher(benchmark, method, **env):
    launcher = ROOT / "examples" / benchmark / "train" / method / "train.slurm"
    clean_env = {key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ}
    clean_env.update(RLLM_DIR=str(ROOT), RLLM_DRY_RUN="1", MODEL_VARIANT="nonthinking", PYTHON_BIN=sys.executable, **env)
    return subprocess.run(["bash", str(launcher)], cwd=ROOT, env=clean_env, text=True, capture_output=True)


def launcher_config(benchmark, method, **env):
    """Return the dry-run argv and its agent, environment and workflow overrides."""
    result = run_launcher(benchmark, method, **env)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)
    overrides = [arg.lstrip("+") for arg in argv[argv.index("-m") + 2 :]]
    keep = ("rllm.agent.", "rllm.env.", "rllm.workflow.", "rllm.actor_prompt=")
    config = OmegaConf.from_dotlist([arg for arg in overrides if arg.startswith(keep)])
    OmegaConf.set_struct(config, True)  # Hydra configs reject unknown keys.
    return argv, config


def agent_args(config):
    args = config.rllm.agent.get("agent_args")
    return OmegaConf.to_container(args) if args is not None else {}


def ace_defaults(benchmark):
    """Evaluate the benchmark's ACE_DEFAULTS without importing GPU dependencies."""
    path = ROOT / ACE_MODULES[benchmark]
    tree = ast.parse(path.read_text())
    names = [node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else "" for node in tree.body]
    nodes = [node for node, name in zip(tree.body, names, strict=True) if name.endswith("_DIR") or name in {"_ASSETS", "ACE_DEFAULTS"}]
    namespace = {"Path": Path, "__file__": str(path)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ACE_DEFAULTS"]


def ace_trainer():
    """The production ACE prompt-injection methods, with only GPU imports excluded."""
    path = ROOT / "rllm/algorithms/ace/trainer_base.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_render_template"]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ACERLAgentPPOTrainer")
    cls.bases = []
    methods = {"_render_base_system_prompt", "_apply_playbook_prompt_to_agent", "_inject_playbook_into_agents"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {}
    exec(compile(ast.Module(body=[*nodes, cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ACERLAgentPPOTrainer"]()


def first_turn(benchmark, agent, config, *, engine):
    """Reset the agent and give it the first observation as its trainer would."""
    if benchmark == "mastermind":
        from rllm.environments.mastermind.mastermind import MastermindEnv

        observation, info = MastermindEnv.from_dict({"seed": 7, **config.rllm.env.env_args}).reset()
    elif benchmark == "mbpp":
        from rllm.environments.mbpp.mbpp import MBPPEnv

        task = {"text": "Write a function add(a, b).", "test_list": ["assert add(1, 2) == 3"], "task_id": 1}
        observation, info = MBPPEnv.from_dict({**task, **config.rllm.env.env_args}).reset()
    elif benchmark == "appworld":
        observation, info = dict(APPWORLD_TASK), {}
    else:
        # MiniSWEnv.reset reports its own step limit; the agent engine replaces it.
        env_steps = engine.env.max_steps if hasattr(engine, "env") else None
        observation, info = SWE_PROBLEM, {"instance_id": "repo__1", "max_steps": env_steps}
    if engine == "agent":
        info["max_steps"] = config.rllm.agent.max_steps  # AgentExecutionEngine.run_agent_trajectory_async
    agent.reset()
    agent.update_from_env(observation, 0.0, False, info)
    return agent


def ace_initial_messages(benchmark):
    """ACE-RL's actor messages at step 0, with its initial playbook and a common turn budget."""
    variable, turns = TURNS[benchmark]
    argv, config = launcher_config(benchmark, "ace_rl", **{variable: turns})
    defaults = ace_defaults(benchmark)
    settings = dict(arg.lstrip("+").split("=", 1) for arg in argv if "=" in arg)
    prompts_dir = settings.get("rllm.ace.prompts_dir", defaults["prompts_dir"])
    initial_playbook = settings.get("rllm.ace.initial_playbook", defaults["initial_playbook"])
    trainer = ace_trainer()
    with open(os.path.join(os.path.abspath(prompts_dir), "base_system_prompt.txt")) as f:  # ACE trainer initialization
        trainer.base_system_prompt_template = f.read()
    with open(os.path.abspath(initial_playbook)) as f:
        playbook_text = Playbook(f.read()).to_string()  # _get_task_playbook(...).to_string()
    agent = import_object(ACE_AGENTS[benchmark])(**agent_args(config))
    trainer._inject_playbook_into_agents([agent], playbook_text)
    return first_turn(benchmark, agent, config, engine="agent").chat_completions


def grpo_actor(benchmark, tmp_path, **env):
    """Build the GRPO actor exactly as the selected GRPO entrypoint configures it."""
    argv, config = launcher_config(benchmark, "grpo", RUN_DIR=str(tmp_path / "run"), **env)
    train = importlib.import_module(f"examples.{benchmark}.train.grpo.train")
    if benchmark in {"mastermind", "mbpp"}:
        train.configure_actor_prompt(config)
        agent_class = import_object(ACE_AGENTS[benchmark])
    elif benchmark == "appworld":
        agent_class = train.select_agent_class(config)
    else:
        from examples.mini_swe_bench.workflow import MiniSweWorkflow

        workflow_args = {**train.select_workflow_args(config), **OmegaConf.to_container(config.rllm.workflow.workflow_args)}
        workflow = MiniSweWorkflow(rollout_engine=SimpleNamespace(), executor=None, **workflow_args)
        return argv, config, first_turn(benchmark, workflow.agent, config, engine=workflow)
    agent = agent_class(**agent_args(config))
    return argv, config, first_turn(benchmark, agent, config, engine="agent")


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_ace_rl_step_zero_uses_an_empty_initial_playbook(benchmark):
    playbook = Path(ace_defaults(benchmark)["initial_playbook"])
    assert playbook.read_text() == ""
    assert "{{ playbook }}" in (Path(ace_defaults(benchmark)["prompts_dir"]) / "base_system_prompt.txt").read_text()


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_option_gives_grpo_the_ace_rl_step_zero_actor_messages(benchmark, tmp_path):
    _, _, agent = grpo_actor(benchmark, tmp_path, GRPO_ACTOR_PROMPT="ace_empty_playbook")
    expected = ace_initial_messages(benchmark)
    assert json.dumps(agent.chat_completions) == json.dumps(expected)
    roles = [message["role"] for message in expected]
    assert roles == (["user"] if benchmark == "appworld" else ["system", "user"])


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_default_grpo_keeps_the_agent_prompt(benchmark, tmp_path):
    argv, config, agent = grpo_actor(benchmark, tmp_path)
    assert not any("actor_prompt" in arg or "ace_prompt_template" in arg or "system_prompt" in arg for arg in argv)
    assert "actor_prompt" not in config.rllm
    first = agent.chat_completions[0]["content"]
    if benchmark == "mastermind":
        from rllm.agents.mastermind_agent import MastermindAgent

        assert type(agent) is MastermindAgent and first == MastermindAgent.SYSTEM_PROMPT
    elif benchmark == "mbpp":
        from rllm.agents.mbpp_agent import MBPPAgent

        assert type(agent) is MBPPAgent and first == MBPPAgent.SYSTEM_PROMPT
    elif benchmark == "appworld":
        from rllm.agents.appworld_react_agents import AppWorldReactAgent

        assert type(agent) is AppWorldReactAgent and first.startswith("I am your supervisor")
    else:
        from rllm.agents.swe_agent import SWEAgent
        from rllm.agents.system_prompts import SWEAGENT_SYSTEM_PROMPT

        assert type(agent) is SWEAgent and first == SWEAGENT_SYSTEM_PROMPT
    assert agent.chat_completions != ace_initial_messages(benchmark)


@pytest.mark.parametrize("benchmark", ("appworld", "mini_swe_bench"))
def test_ace_agent_classes_keep_grpo_turn_handling(benchmark, tmp_path):
    _, _, default = grpo_actor(benchmark, tmp_path)
    _, _, selected = grpo_actor(benchmark, tmp_path, GRPO_ACTOR_PROMPT="ace_empty_playbook")
    assert isinstance(selected, type(default)) and isinstance(selected, BaseAgent)
    if benchmark == "appworld":
        response, feedback = "Thought.\n```python\nprint(apis.spotify.show_queue())\n```", {"success": True, "output": "[1, 2]"}
    else:
        response, feedback = "Look first.\n<function=execute_bash>\n<parameter=cmd>ls</parameter>\n</function>", "README.md"
    start = {agent: len(agent.chat_completions) for agent in (default, selected)}
    actions = [agent.update_from_model(response).action for agent in (default, selected)]
    for agent in (default, selected):
        agent.update_from_env(feedback, 0.0, False, {"max_steps": 10})
    assert actions[0] == actions[1]
    assert default.chat_completions[start[default] :] == selected.chat_completions[start[selected] :]


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_option_changes_only_the_selector_name_and_run_identity(benchmark, tmp_path):
    default_argv, default_config, _ = grpo_actor(benchmark, tmp_path)
    argv, config, _ = grpo_actor(benchmark, tmp_path, GRPO_ACTOR_PROMPT="ace_empty_playbook")
    names = {"trainer.experiment_name", "rllm.env.env_args.experiment_name"}

    def stripped(values):
        return [arg for arg in values[1:] if arg.lstrip("+").split("=", 1)[0] not in names]

    assert stripped(argv) == [*stripped(default_argv), "+rllm.actor_prompt=ace_empty_playbook"]
    experiment = next(arg for arg in argv if arg.startswith("trainer.experiment_name="))
    assert "_ace_empty_playbook" in experiment
    identity = training_identity(OmegaConf.to_container(config))
    assert identity != training_identity(OmegaConf.to_container(default_config))
    assert identity["rllm"]["actor_prompt"] == "ace_empty_playbook"


def test_option_requires_its_own_run_dir_and_a_known_value():
    missing = run_launcher("mastermind", "grpo", GRPO_ACTOR_PROMPT="ace_empty_playbook")
    assert missing.returncode == 2 and "set its own RUN_DIR" in missing.stderr
    unknown = run_launcher("mbpp", "grpo", GRPO_ACTOR_PROMPT="ace", RUN_DIR="/nonexistent/run")
    assert unknown.returncode == 2 and "must be agent or ace_empty_playbook" in unknown.stderr
    with pytest.raises(ValueError, match="rllm.actor_prompt"):
        from rllm.algorithms.ace.agents import ace_empty_playbook_prompt

        ace_empty_playbook_prompt(OmegaConf.create({"rllm": {"actor_prompt": "ace"}}), ROOT)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
@pytest.mark.parametrize("method", ("ppo", "erl"))
def test_methods_reusing_grpo_presets_reject_the_option(benchmark, method):
    result = run_launcher(benchmark, method, GRPO_ACTOR_PROMPT="ace_empty_playbook", RUN_DIR="/nonexistent/run")
    assert result.returncode == 2 and "applies only to the GRPO launcher" in result.stderr


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_option_composes_and_configures_the_pinned_hydra_schema(benchmark, tmp_path):
    from hydra import compose, initialize_config_module

    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE to validate the actual pinned Hydra schema")
    argv, _ = launcher_config(benchmark, "grpo", RUN_DIR=str(tmp_path / "run"), GRPO_ACTOR_PROMPT="ace_empty_playbook")
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
    with initialize_config_module(config_module="rllm.trainer.config", version_base=None):
        config = compose(config_name="agent_ppo_trainer", overrides=argv[argv.index("-m") + 2 :] + [f"hydra.searchpath=[file://{source}/verl/trainer/config]"])
    train = importlib.import_module(f"examples.{benchmark}.train.grpo.train")
    template = (ROOT / "examples" / benchmark / "train/ace_rl/prompts/base_system_prompt.txt").read_text()
    if benchmark in {"mastermind", "mbpp"}:
        train.configure_actor_prompt(config)
        assert config.rllm.agent.agent_args.system_prompt == template.replace("{{ playbook }}", "")
    elif benchmark == "appworld":
        assert train.select_agent_class(config).__name__ == "AppWorldReactAgentACE"
        assert config.rllm.agent.agent_args.ace_prompt_template == template
    else:
        assert train.select_workflow_args(config)["agent_cls"].__name__ == "MiniSweACEAgent"
        assert config.rllm.workflow.workflow_args.agent_args.ace_prompt_template == template
    resolved = OmegaConf.to_container(config, resolve=True)  # run_state.prepare_training_run
    assert training_identity(resolved)["rllm"]["actor_prompt"] == "ace_empty_playbook"
