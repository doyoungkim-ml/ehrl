"""AppWorld opt-in inputs match real trainer methods without importing GPU workers."""

import ast
import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.appworld.eval.ace.training_inputs import AppWorldTrainingInputReflector
from examples.appworld.train.ace_rl.appworld_adapter import AppWorldReflectorAdapter
from rllm.agents.appworld_react_agents_ace import AppWorldReactAgentACE
from rllm.algorithms.ace.context import ACEConfig, ACEContextRule
from rllm.algorithms.ace.context.reflector import Reflector
from rllm.algorithms.ace.context.training_inputs import concatenate_training_insights

ROOT = Path(__file__).resolve().parents[2]


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


class RecordingChatParser:
    def __init__(self):
        self.calls = []

    def parse(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages) + "<assistant>"


def trainer_reference(*, adapter=None, template="", limit=30720, system_prompt="reflector system"):
    """Execute the production training methods, with only GPU imports excluded."""
    path = ROOT / "rllm/algorithms/ace/trainer_base.py"
    tree = ast.parse(path.read_text())
    helper_names = {"_render_template", "_parse_json_response", "_coerce_reflection_payload", "_truncate_response_text", "_concatenate_insights"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ACERLAgentPPOTrainer")
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"_trajectory_steps_from_chat", "_build_reflector_prompt_from_dict"}]
    nodes.append(cls)
    namespace = {"json": json, "SimpleNamespace": SimpleNamespace, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    trainer = namespace["ACERLAgentPPOTrainer"]()
    trainer.tokenizer = CharacterTokenizer()
    trainer.reflector_adapter = adapter or AppWorldReflectorAdapter(max_turns=30, tokenizer=trainer.tokenizer, max_history_tokens=20000)
    trainer.reflector_prompt_template = template
    trainer.reflector_system_prompt = system_prompt
    trainer.ace_config = {"max_turns": 30}
    trainer.component_token_limits = {"reflector": {"max_prompt_length": limit}}
    trainer.agent_execution_engine = SimpleNamespace(rollout_engine=SimpleNamespace(chat_parser=RecordingChatParser()))
    trainer._get_policy_task_for_trajectory = lambda data, task_idx: data["training_task"]
    return trainer, namespace


def make_agent(*, long_response=False):
    agent = AppWorldReactAgentACE(accumulate_thinking=True)
    agent.set_ace_playbook_prompt(template="USER:\nTask: {{input_str}}\nAdvice: {{playbook}}", playbook="")
    agent.update_from_env({"instruction": "REAL TASK " + "x" * 350}, 0, False, {})
    for index in range(2):
        content = ("assistant discussion " + "z" * (23000 if long_response else 3)) + f"\nCode:\n```python\nprint({index})\n```"
        reasoning = f"PRIVATE REASONING {index}"
        raw = f"<think>{reasoning}</think>\n{content}"
        output = SimpleNamespace(content=content, reasoning=reasoning, text=raw)
        agent.update_from_model(raw, model_output=output)
        agent.update_from_env("FIRST REPLY " + "y" * 350 if index == 0 else "TERMINAL REPLY", 0.0, index == 1, {})
    agent.trajectory.info.update({"step_rewards": [0.0, 0.0], "termination_reason": "PROMPT_TRUNCATION", "all_passed": False})
    return agent


def reflector_fixture(trajectory, *, template=None, limit=30720, playbook="", history_limit=20000):
    template = template or (ROOT / "examples/appworld/train/ace_rl/prompts/reflector_prompt.txt").read_text()
    tokenizer = CharacterTokenizer()
    parser = RecordingChatParser()
    adapter = AppWorldReflectorAdapter(max_turns=30, tokenizer=tokenizer, max_history_tokens=history_limit)
    engine = SimpleNamespace(tokenizer=tokenizer, chat_parser=parser, max_prompt_length=limit)
    reflector = AppWorldTrainingInputReflector(template, adapter=adapter, rollout_engine=engine, system_prompt="reflector system", temperature=0.7)
    task = {"task_id": "task", "split": "dev", "max_turns": 30}
    payload = {"trajectory": trajectory, "reward": 0.0, "max_turns": 30, "training_task": task}
    trainer, helpers = trainer_reference(adapter=adapter, template=template, limit=limit)
    train_payload = {
        "chat_completions": trajectory.steps[-1].chat_completions if trajectory.steps else [],
        "step_rewards": trajectory.info.get("step_rewards", []),
        "trajectory_reward": 0.0,
        "all_passed": False,
        "termination_reason": trajectory.info.get("termination_reason"),
        "training_task": task,
    }
    return reflector, payload, trainer, train_payload, helpers


@pytest.mark.parametrize("long_response", [False, True])
def test_actual_agent_chat_builds_identical_training_reflector_prompt(long_response):
    agent = make_agent(long_response=long_response)
    reflector, payload, trainer, train_payload, _ = reflector_fixture(agent.trajectory)
    actual = reflector._build_prompt(payload, "[1] Existing playbook")
    expected = trainer._build_reflector_prompt_from_dict(train_payload, "[1] Existing playbook", 0)
    assert actual == expected
    assert "PRIVATE REASONING" not in actual and "<think>" not in actual
    assert "TERMINAL REPLY" not in actual
    assert "PROMPT_TRUNCATION" in actual
    assert "x" * 301 not in actual and "y" * 301 not in actual
    assert "assistant discussion" in actual
    if not long_response:
        assert actual.count("print(0)") == 2  # Training repeats assistant content as action.
    assert trainer._get_policy_task_for_trajectory(train_payload, 0).get("instruction", "") == ""
    assert payload["trajectory"].task["instruction"].startswith("REAL TASK")
    messages = [{"role": "system", "content": reflector.system_prompt}, {"role": "user", "content": actual}]
    assert len(reflector.rollout_engine.chat_parser.parse(messages)) <= 30720
    assert all(call[1]["accumulate_reasoning"] is False for call in trainer.agent_execution_engine.rollout_engine.chat_parser.calls)


def test_whole_prompt_guard_shrinks_only_history_and_matches_training():
    agent = make_agent(long_response=True)
    reflector, payload, trainer, train_payload, _ = reflector_fixture(
        agent.trajectory, template="TASK={{task_instr}}|SUMMARY={{trajectory_summary}}|BOOK={{playbook}}|HISTORY={{conversation_history}}", limit=1500
    )
    playbook = "[1] " + "fixed advice " * 40
    actual = reflector._build_prompt(payload, playbook)
    assert actual == trainer._build_reflector_prompt_from_dict(train_payload, playbook, 0)
    assert playbook in actual and actual.startswith("TASK=|SUMMARY=")
    assert "PROMPT_TRUNCATION" in actual
    rendered = reflector.rollout_engine.chat_parser.parse([{"role": "system", "content": reflector.system_prompt}, {"role": "user", "content": actual}])
    assert len(rendered) <= 1500
    assert len(reflector.rollout_engine.chat_parser.calls) >= 3


def test_nonhistory_overflow_matches_training_failure():
    reflector, payload, trainer, train_payload, _ = reflector_fixture(make_agent().trajectory, limit=20)
    for call in (lambda: reflector._build_prompt(payload, "large fixed playbook"), lambda: trainer._build_reflector_prompt_from_dict(train_payload, "large fixed playbook", 0)):
        with pytest.raises(ValueError, match="non-history fields exceed"):
            call()


def test_parity_requires_original_environment_payload_and_cumulative_chat():
    reflector, payload, _, _, _ = reflector_fixture(make_agent().trajectory)
    del payload["training_task"]
    with pytest.raises(ValueError, match="original environment task"):
        reflector._build_prompt(payload, "")
    payload["training_task"] = {}
    payload["trajectory"].steps[-1].chat_completions = []
    with pytest.raises(ValueError, match="cumulative Step.chat_completions"):
        reflector._build_prompt(payload, "")


@pytest.mark.parametrize(
    "response",
    [
        '{"reasoning": "why", "key_insight": "advice", "failure_reason": "cause", "correct_approach": "repair"}',
        '{"reflection": {"key_insight": 42}}',
        '[null, 3, {"reflection": {"reasoning": "reason"}}, {"key_insight": "unused"}]',
        '```json\n{"key_insight": "fenced"}\n```',
        'prefix {"key_insight": "embedded"} suffix',
        "[]",
        "null",
        "42",
        "invalid",
        "",
    ],
)
def test_parity_reflection_and_curator_text_match_real_training_helpers(response):
    reflector, _, _, _, helpers = reflector_fixture(make_agent().trajectory)
    expected = helpers["_coerce_reflection_payload"](helpers["_parse_json_response"](response), response_text=response)
    actual = reflector._parse_response(response)
    assert actual == expected
    inputs = [{"task_idx": 0, "reflection": actual}, {"task_idx": 7, "reflection": {"reasoning": "other", "outcome": "SUCCESS"}}, {"reflection": None}]
    assert concatenate_training_insights(inputs) == helpers["_concatenate_insights"](inputs)


def test_failed_parity_reflection_produces_no_insight():
    reflector, payload, _, _, _ = reflector_fixture(make_agent().trajectory)

    async def empty_call(prompt):
        return ""

    reflector._llm_call = empty_call
    assert asyncio.run(reflector.analyze(payload, ""))["result"] == {}


class FakeOpenAIEngine:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.max_model_length = kwargs["max_model_length"] - 1
        self.chat_parser = RecordingChatParser()


class FakeAgentExecutionEngine:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.rollout_engine = SimpleNamespace(max_model_length=kwargs["rollout_engine_args"]["max_model_length"] - 1)


def factories():
    path = ROOT / "examples/appworld/eval/ace/appworld_ace_utils.py"
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in {"create_ace_context_rule", "build_generator_engine"}]
    namespace = {
        "os": os,
        "AutoTokenizer": SimpleNamespace(from_pretrained=lambda name: CharacterTokenizer()),
        "ACEConfig": ACEConfig,
        "ACEContextRule": ACEContextRule,
        "AppWorldReflectorAdapter": AppWorldReflectorAdapter,
        "AppWorldTrainingInputReflector": AppWorldTrainingInputReflector,
        "OpenAIEngine": FakeOpenAIEngine,
        "AgentExecutionEngine": FakeAgentExecutionEngine,
        "AppWorldReactAgentACE": AppWorldReactAgentACE,
        "AppWorldEnv": object,
        "is_external_model": lambda model: model.startswith("external"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def factory_args(parity):
    return SimpleNamespace(
        training_input_parity=parity,
        model_name="test-model",
        ace_model=None,
        ace_temperature=0.7,
        max_model_length=32768,
        max_prompt_length=8192,
        max_response_length=32768,
        reflector_max_prompt_length=30720,
        reflector_max_response_length=2048,
        reflector_max_history_tokens=20000,
        curator_max_prompt_length=8192,
        curator_max_response_length=4096,
        reflector_system_prompt="reflector system",
        curator_system_prompt="curator system",
        ace_base_url=None,
        base_url="http://127.0.0.1:8000/v1",
        api_key=None,
        debug=False,
        max_turns=30,
        n_servers=4,
        n_parallel_agents=4,
        base_port=9000,
        remote_environment_url=None,
        gen_temperature=0.7,
        gen_top_p=1.0,
        disable_thinking=False,
        accumulate_thinking=True,
        accumulate_reasoning=True,
        trajectory_hard_timeout=7200,
    )


@pytest.mark.parametrize("parity", [False, True])
def test_opt_in_factories_keep_defaults_and_apply_role_sampling_and_capacity(tmp_path, parity):
    functions = factories()
    args = factory_args(parity)
    (tmp_path / "reflector_prompt.txt").write_text("{{conversation_history}}")
    (tmp_path / "curator_prompt.txt").write_text("{{current_playbook}}\n{{guidebook}}")
    (tmp_path / "playbook.txt").write_text("")
    rule = functions["create_ace_context_rule"](args, str(tmp_path), str(tmp_path / "playbook.txt"), None)
    generator = functions["build_generator_engine"](args)
    assert type(rule.reflector) is (AppWorldTrainingInputReflector if parity else Reflector)
    assert rule.reflector.temperature == (0.7 if parity else 0.0)
    assert rule.curator.temperature == 0.7
    assert rule.reflector.rollout_engine.max_prompt_length == 30720
    assert rule.reflector.rollout_engine.max_response_length == 2048
    assert rule.curator.rollout_engine.max_prompt_length == 8192
    assert rule.curator.rollout_engine.max_response_length == 4096
    for engine in (rule.reflector.rollout_engine, rule.curator.rollout_engine, generator.rollout_engine):
        assert engine.max_model_length == (32768 if parity else 32767)
    expected_roles = {"temperature": 0.7, **({"top_p": 1.0, "extra_body": {"top_k": -1}} if parity else {})}
    assert rule.reflector.rollout_engine.sampling_params == rule.curator.rollout_engine.sampling_params == expected_roles
    assert generator.sampling_params == {"temperature": 0.7, "top_p": 1.0, "model": "test-model", **({"extra_body": {"top_k": -1}} if parity else {})}
    assert generator.trajectory_hard_timeout == 7200
    assert generator.max_prompt_length == 8192 and generator.max_response_length == 32768
    assert generator.config["rllm"] == {"disable_thinking": False, "accumulate_reasoning": True}


def test_full_model_capacity_preserves_training_reflector_response_budget():
    path = ROOT / "rllm/engine/rollout/openai_engine.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "OpenAIEngine")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_max_tokens_param")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    prepare = namespace["_prepare_max_tokens_param"]
    assert prepare(SimpleNamespace(max_model_length=32768, max_response_length=2048), {}, 30720) == {"max_tokens": 2048}
    assert prepare(SimpleNamespace(max_model_length=32767, max_response_length=2048), {}, 30720) == {"max_tokens": 2047}


class TrialRoleEngine:
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.chat_parser = RecordingChatParser()
        self.max_prompt_length = 30720
        self.calls = []

    async def get_model_response(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if messages[-1]["content"].startswith("REFLECTOR|"):
            result = {"reflection": {"reasoning": "Inspect the schema", "failure_reason": "Missing API details", "key_insight": "Read specifications first"}}
        else:
            result = {"operations": [{"type": "ADD", "section": "others", "content": "Read specifications before each API call"}]}
        return SimpleNamespace(text=json.dumps(result))


class TrialGeneratorEngine:
    agent_class = AppWorldReactAgentACE

    def __init__(self):
        self.prompts = []

    async def execute_tasks(self, tasks):
        trajectories = []
        for task in tasks:
            agent = self.agent_class()
            agent.reset()
            agent.update_from_env({"instruction": "Solve " + task["task_id"]}, 0, False, {})
            prompt = agent.chat_completions[0]["content"]
            self.prompts.append(prompt)
            content = "Code:\n```python\nprint(1)\n```"
            output = SimpleNamespace(content=content, reasoning="private", text="<think>private</think>" + content)
            agent.update_from_model(output.text, model_output=output)
            reward = 1.0 if "Read specifications before each API call" in prompt else 0.0
            agent.update_from_env("Terminal reply", reward, True, {})
            agent.trajectory.reward = reward
            agent.trajectory.info.update({"step_rewards": [reward], "all_passed": reward == 1, "termination_reason": "ENV_DONE"})
            # AgentExecutionEngine.execute_tasks restores the input task after
            # the agent builds its instruction-only environment payload.
            agent.trajectory.task = task
            trajectories.append(agent.trajectory)
        return trajectories


@pytest.mark.parametrize("mode", ["local", "global"])
def test_complete_evaluation_uses_training_inputs_with_common_one_one_one_budget(tmp_path, mode):
    from rllm.algorithms.ace import evaluation

    roles = TrialRoleEngine()
    generator = TrialGeneratorEngine()
    (tmp_path / "reflector_prompt.txt").write_text("REFLECTOR|TASK={{task_instr}}|{{conversation_history}}|{{trajectory_summary}}|{{playbook}}")
    (tmp_path / "curator_prompt.txt").write_text("CURATOR|{{current_playbook}}|{{guidebook}}")
    (tmp_path / "playbook.txt").write_text("")
    (tmp_path / "base_system_prompt.txt").write_text("USER:\nTask: {{input_str}}\nAdvice: {{playbook}}")
    rule = ACEContextRule(
        ACEConfig(
            reflector_prompt_path=str(tmp_path / "reflector_prompt.txt"),
            curator_prompt_path=str(tmp_path / "curator_prompt.txt"),
            initial_playbook_path=str(tmp_path / "playbook.txt"),
            max_retries=1,
        ),
        rollout_engine=roles,
        reflector_adapter=AppWorldReflectorAdapter(max_turns=30, tokenizer=roles.tokenizer, max_history_tokens=20000),
        reflector_system_prompt="reflector system",
        curator_system_prompt="curator system",
    )
    rule.reflector = AppWorldTrainingInputReflector(
        rule.reflector.prompt_template,
        adapter=rule.reflector.adapter,
        rollout_engine=roles,
        system_prompt="reflector system",
        temperature=0.7,
        max_retries=1,
    )
    hooks = evaluation.ACEExperimentHooks(
        experiment_name="appworld-parity-test",
        task_id_getter=lambda task: task["task_id"],
        build_generator_engine=lambda args: generator,
        create_context_rule=lambda *args, **kwargs: rule,
        trajectory_reward=lambda trajectory: trajectory.reward,
        trajectory_success=lambda trajectory: trajectory.reward == 1.0,
        render_prompt_in_agent=True,
    )
    args = SimpleNamespace(
        model_name="test-model",
        train_batch_size=1,
        epochs=1,
        local_eval_epochs=3,
        g_act=1,
        g_ref=1,
        g_cur=1,
        selection_mode="best_rejection",
        test_epoch_freq=1,
        val_before_train=False,
        max_turns=30,
        training_input_parity=True,
        freeze_solved=False,
    )
    tasks = [{"task_id": "task", "split": "dev", "max_turns": 30}]
    common = dict(
        args=args, hooks=hooks, prompts_dir=str(tmp_path), initial_playbook_path=str(tmp_path / "playbook.txt"), playbook_snapshot_dir=None, ace_log_dir=None, metrics_file=None, script_path=__file__
    )
    if mode == "local":
        result = asyncio.run(evaluation.run_local_ace(**common, train_tasks=[], eval_tasks=tasks))
        assert result.local_eval_epochs == 3
        assert result.evaluations[0]["tasks"][0]["local_eval_epoch"] == 3
        entry = result.evaluations[0]
        assert entry["local_eval_protocol"] == evaluation.SEQUENTIAL_LOCAL_PROTOCOL
        assert entry["final_policy_evaluation"]["tasks"] == entry["local_epochs"][-1]["tasks"]
        assert [visit["tasks"][0]["reward"] for visit in entry["local_epochs"]] == [0.0, 1.0, 1.0]
        updates, policy_visits = 2, 3
    else:
        result = asyncio.run(evaluation.run_global_ace(**common, train_tasks=tasks, eval_tasks=[], chat_completion_dir=None))
        assert len(result.train_steps) == 1
        updates, policy_visits = 1, 2  # Global training retains candidate search.
    reflection_calls = [call for call in roles.calls if call[0][-1]["content"].startswith("REFLECTOR|")]
    curator_calls = [call for call in roles.calls if call[0][-1]["content"].startswith("CURATOR|")]
    assert len(reflection_calls) == len(curator_calls) == updates
    assert len(generator.prompts) == policy_visits
    assert all(call[0][-1]["content"].startswith("REFLECTOR|TASK=|") for call in reflection_calls)
    assert all("PRIVATE" not in call[0][-1]["content"] and "<think>" not in call[0][-1]["content"] for call in reflection_calls)
    assert all(call[1]["temperature"] == 0.7 for call in reflection_calls)
    for messages, _ in curator_calls:
        prompt = messages[-1]["content"]
        assert "--- Insight 1 (Task 0) ---" in prompt
        assert "Outcome: UNKNOWN" in prompt
        assert "Reasoning: Inspect the schema" in prompt
        assert "Failure Reason: Missing API details" in prompt
    assert "Read specifications before each API call" in generator.prompts[-1]
