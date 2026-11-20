"""Context runner behavior with an in-process engine and real rendering."""

import asyncio
import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Template, UndefinedError

from rllm.context import ContextAwareRunner, ContextRule, ContextUpdateLevel, NoContextRule


class RecordingRule(ContextRule):
    def __init__(self, context="seed", *, level=ContextUpdateLevel.BATCH, failure=None):
        self.context = context
        self.level = level
        self.failure = failure
        self.events = []

    @property
    def update_level(self):
        return self.level

    def get_context(self):
        return self.context

    def reset_batch(self):
        self.events.append(("reset",))

    def on_trajectory_complete(self, trajectory):
        self.events.append(("trajectory", trajectory.task["id"]))
        if self.failure == "trajectory":
            raise RuntimeError("trajectory failed")

    async def on_batch_complete(self, trajectories):
        self.events.append(("batch", tuple(trajectory.task["id"] for trajectory in trajectories)))
        if self.failure == "batch":
            raise RuntimeError("batch failed")
        self.context += f":{len(trajectories)}"

    def save(self, path):
        Path(path).write_text(self.context)

    def load(self, path):
        self.context = Path(path).read_text()


class RecordingAgent:
    def reset(self, *, initial_messages=None, fail=False):
        if fail:
            raise RuntimeError("reset failed")
        self.messages = copy.deepcopy(initial_messages or [])
        return "reset-result"

    def initialize(self, task):
        pass


class TaskRenderingAgent(RecordingAgent):
    def initialize(self, task):
        prompt = Template(self._injected_template).render(
            context=self._injected_context,
            playbook=self._injected_playbook,
            instruction=task["instruction"],
        )
        self.messages = [{"role": "system", "content": prompt}]


class RecordingEngine:
    def __init__(self, agent_class=RecordingAgent, *, initial_messages=None, failure=None):
        self.agent_class = agent_class
        self.initial_messages = initial_messages
        self.failure = failure
        self.agents = []
        self.calls = []

    async def execute_tasks(self, tasks):
        self.calls.append(tasks)
        results = []
        for task in tasks:
            agent = self.agent_class()
            result = agent.reset(initial_messages=self.initial_messages, fail=self.failure == "reset")
            assert result == "reset-result"
            agent.initialize(task)
            self.agents.append(agent)
            if self.failure == "execute":
                raise RuntimeError("execute failed")
            results.append(SimpleNamespace(task=task, reward=task.get("reward", 0), steps=[None] * task.get("steps", 1)))
        return results


def make_runner(*, engine=None, rule=None, prompt="Advice: {{ context }}", **kwargs):
    return ContextAwareRunner(
        engine=engine if engine is not None else RecordingEngine(),
        context_rule=rule if rule is not None else RecordingRule(),
        base_system_prompt=prompt,
        verbose=False,
        **kwargs,
    )


def test_context_import_does_not_load_algorithms_or_training_dependencies():
    root = Path(__file__).resolve().parents[2]
    code = """
import sys
import rllm.context
for name in ('torch', 'ray', 'verl', 'rllm.algorithms.ace', 'rllm.engine.agent_execution_engine'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


def test_batch_hooks_complete_before_next_batch_uses_updated_context():
    engine = RecordingEngine()
    rule = RecordingRule()
    runner = make_runner(engine=engine, rule=rule, batch_size=2)
    tasks = [{"id": "a", "reward": 1, "steps": 2}, {"id": "b", "reward": 0}, {"id": "c", "reward": -1}]
    results = asyncio.run(runner.run(tasks))

    assert [result.task["id"] for result in results] == ["a", "b", "c"]
    assert [agent.messages[0]["content"] for agent in engine.agents] == ["Advice: seed", "Advice: seed", "Advice: seed:2"]
    assert rule.events == [
        ("reset",),
        ("trajectory", "a"),
        ("trajectory", "b"),
        ("batch", ("a", "b")),
        ("reset",),
        ("trajectory", "c"),
        ("batch", ("c",)),
    ]
    assert engine.agent_class is RecordingAgent
    assert rule.context == "seed:2:1"
    assert runner.get_batch_metrics() == [
        {"batch_idx": 0, "num_trajectories": 2, "success_count": 1, "success_rate": 0.5, "avg_steps": 1.5, "avg_reward": 0.5},
        {"batch_idx": 1, "num_trajectories": 1, "success_count": 0, "success_rate": 0.0, "avg_steps": 1.0, "avg_reward": -1.0},
    ]
    exposed = runner.get_batch_metrics()
    exposed[0]["success_count"] = 99
    assert runner.get_batch_metrics()[0]["success_count"] == 1


@pytest.mark.parametrize("private_batch", [False, True])
def test_evaluation_does_not_reset_or_adapt_rule(private_batch):
    rule = RecordingRule(failure="batch")
    runner = make_runner(rule=rule, batch_size=1)
    tasks = [{"id": "a"}, {"id": "b"}]
    before = copy.deepcopy(vars(rule))
    if private_batch:
        results = asyncio.run(runner._process_batch(tasks, 8, adapt=False))
    else:
        results = asyncio.run(runner.run(tasks, adapt=False))
    assert len(results) == 2
    assert vars(rule) == before
    assert all(agent.messages[0]["content"] == "Advice: seed" for agent in runner.engine.agents)


@pytest.mark.parametrize("failure", ["reset", "execute", "trajectory", "batch"])
def test_agent_class_restored_on_execution_and_hook_errors(failure):
    engine = RecordingEngine(failure=failure)
    rule = RecordingRule(failure=failure)
    runner = make_runner(engine=engine, rule=rule)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        asyncio.run(runner.run([{"id": "a"}]))
    assert engine.agent_class is RecordingAgent
    assert runner.get_batch_metrics() == []
    if failure in {"reset", "execute"}:
        assert rule.events == [("reset",)]


@pytest.mark.parametrize(
    "initial_messages, expected_tail",
    [
        ([], []),
        ([{"role": "system", "content": "old"}, {"role": "user", "content": "task"}], [{"role": "user", "content": "task"}]),
        ([{"role": "user", "content": "task"}], [{"role": "user", "content": "task"}]),
    ],
)
def test_reset_injection_preserves_non_system_messages(initial_messages, expected_tail):
    engine = RecordingEngine(initial_messages=initial_messages)
    runner = make_runner(engine=engine)
    asyncio.run(runner.run([{"id": "a"}], adapt=False))
    agent = engine.agents[0]
    assert agent.messages == [{"role": "system", "content": "Advice: seed"}, *expected_tail]
    assert agent._injected_template == "Advice: {{ context }}"
    assert agent._injected_context == agent._injected_playbook == "seed"
    assert not hasattr(RecordingAgent, "_injected_context")


def test_template_file_preserves_context_whitespace_and_legacy_alias(tmp_path):
    template = tmp_path / "prompt.txt"
    template.write_text("{{ context }}|{{ playbook }}\n")
    runner = make_runner(prompt=template, rule=RecordingRule("  learned\n"))
    assert runner.get_rendered_prompt() == "  learned\n|  learned\n"


@pytest.mark.parametrize("benchmark", ["mbpp"])
@pytest.mark.parametrize("trailing_newline", [False, True])
def test_benchmark_prompt_bytes_match_existing_jinja_rendering(benchmark, trailing_newline):
    root = Path(__file__).resolve().parents[2]
    source = (root / f"examples/{benchmark}/train/ace_rl/prompts/base_system_prompt.txt").read_text()
    if trailing_newline:
        source += "\n"
    context = "  learned context\n"
    expected = Template(source).render(playbook=context)
    runner = make_runner(prompt=source, rule=RecordingRule(context))
    assert runner.get_rendered_prompt().encode("utf-8") == expected.encode("utf-8")


def test_task_variable_names_in_prose_do_not_change_rendering():
    runner = make_runner(prompt="input_str is a field name. Advice: {{ context }}")
    assert runner.get_rendered_prompt() == "input_str is a field name. Advice: seed"


def test_unresolved_template_fields_require_explicit_rendering_strategy():
    runner = make_runner(prompt="{{ instruction }} {{ context }}")
    with pytest.raises(UndefinedError, match="instruction"):
        asyncio.run(runner.run([{"id": "a"}], adapt=False))
    assert runner.engine.calls == []
    assert runner.engine.agent_class is RecordingAgent


def test_agent_side_rendering_receives_raw_context_and_task_fields():
    engine = RecordingEngine(agent_class=TaskRenderingAgent)
    template = "{{ instruction }}: {{ context }} / {{ playbook }}"
    runner = make_runner(engine=engine, prompt=template, rule=RecordingRule(" advice "), render_in_agent=True)
    assert runner.get_rendered_prompt() == template
    asyncio.run(runner.run([{"id": "a", "instruction": "First"}, {"id": "b", "instruction": "Second"}], adapt=False))
    assert [agent.messages[0]["content"] for agent in engine.agents] == ["First:  advice  /  advice ", "Second:  advice  /  advice "]
    assert engine.agent_class is TaskRenderingAgent


def test_custom_prompt_renderer_replaces_default_template_language():
    calls = []

    def render(template, context):
        calls.append((template, context))
        return template.replace("[context]", context)

    runner = make_runner(prompt="Advice: [context]", prompt_renderer=render)
    asyncio.run(runner.run([{"id": "a"}], adapt=False))
    assert calls == [("Advice: [context]", "seed")]
    assert runner.engine.agents[0].messages[0]["content"] == "Advice: seed"


@pytest.mark.parametrize("empty_text_option", ["empty_context_text", "empty_playbook_text"])
def test_empty_context_placeholder_accepts_generic_and_legacy_option(empty_text_option):
    runner = make_runner(rule=NoContextRule(), **{empty_text_option: "empty"})
    assert runner.get_rendered_prompt() == "Advice: empty"


@pytest.mark.parametrize("context, expected", [("learned", "Instructions\n\nlearned"), ("", "Instructions"), ("  ", "Instructions")])
def test_plain_prompt_appends_only_nonempty_context(context, expected):
    assert make_runner(prompt="Instructions", rule=RecordingRule(context)).get_rendered_prompt() == expected


@pytest.mark.parametrize("level", [ContextUpdateLevel.STEP, ContextUpdateLevel.TRAJECTORY])
def test_unsupported_update_levels_fail_before_execution_or_mutation(level):
    engine = RecordingEngine()
    rule = RecordingRule(level=level)
    with pytest.raises(NotImplementedError, match="only BATCH"):
        make_runner(engine=engine, rule=rule)
    assert engine.calls == []
    assert rule.events == []


def test_changed_update_level_is_rejected_before_reset():
    rule = RecordingRule()
    runner = make_runner(rule=rule)
    rule.level = ContextUpdateLevel.STEP
    with pytest.raises(NotImplementedError, match="only BATCH"):
        asyncio.run(runner.run([{"id": "a"}]))
    assert rule.events == []
    assert runner.engine.calls == []


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        make_runner(batch_size=batch_size)


def test_conflicting_rendering_strategies_are_rejected():
    with pytest.raises(ValueError, match="either"):
        make_runner(render_in_agent=True, prompt_renderer=lambda template, context: context)


def test_empty_task_list_does_not_mutate_rule():
    rule = RecordingRule()
    runner = make_runner(rule=rule)
    before = copy.deepcopy(vars(rule))
    assert asyncio.run(runner.run([])) == []
    assert vars(rule) == before
    assert runner.get_batch_metrics() == []


def test_context_save_and_load_are_delegated(tmp_path):
    rule = RecordingRule()
    runner = make_runner(rule=rule)
    checkpoint = tmp_path / "context.txt"
    runner.save_context(str(checkpoint))
    rule.context = "changed"
    runner.load_context(str(checkpoint))
    assert runner.get_context() == "seed"
