"""Real agent updates retain the reply used by opt-in ACE feedback history."""

import json

import pytest

from rllm.agents.agent import Trajectory
from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.agents.appworld_react_agents_ace import AppWorldReactAgentACE
from rllm.agents.mbpp_agent import MBPPAgent
from rllm.agents.swe_agent import SWEAgent
from rllm.algorithms.ace.context.utils import format_trajectory_as_conversation, format_trajectory_with_feedback


@pytest.mark.parametrize("agent_type", [AppWorldReactAgent, AppWorldReactAgentACE])
@pytest.mark.parametrize("reply", ["terminal execution failed", {"success": False, "error": "terminal API error", "stderr": "detail"}])
def test_appworld_real_terminal_update_reaches_feedback_history(agent_type, reply):
    agent = agent_type(accumulate_thinking=False)
    agent.update_from_env({"instruction": "Complete this test task"}, 0.0, False, {})
    agent.update_from_model("```python\nprint('probe')\n```")
    legacy_before = format_trajectory_as_conversation(agent.trajectory)
    step = agent.trajectory.steps[-1]
    assert step.next_observation is None

    agent.update_from_env(reply, 0.0, True, {"step": 1})

    rendered = agent.chat_completions[-1]["content"]
    assert step.next_observation == rendered
    assert step.done is True
    assert step.reward == 0.0
    assert step.info == {"step": 1}
    history = format_trajectory_with_feedback(agent.trajectory)
    assert f"ENVIRONMENT REPLY:\n{rendered}" in history
    assert history.index("ACTION:") < history.index("ENVIRONMENT REPLY:")
    restored = Trajectory.from_dict(json.loads(json.dumps(agent.trajectory.to_dict())))
    assert restored.steps[-1].next_observation == rendered
    assert format_trajectory_with_feedback(restored) == history
    old_payload = agent.trajectory.to_dict()
    del old_payload["steps"][-1]["next_observation"]
    old_restored = Trajectory.from_dict(old_payload)
    assert old_restored.steps[-1].next_observation is None
    assert "(no environment reply recorded)" in format_trajectory_with_feedback(old_restored)
    assert format_trajectory_as_conversation(agent.trajectory) == legacy_before


@pytest.mark.parametrize("agent_type", [AppWorldReactAgent, AppWorldReactAgentACE])
def test_appworld_nonterminal_reply_stays_with_its_action(agent_type):
    agent = agent_type(accumulate_thinking=False)
    agent.update_from_env({"instruction": "Complete this test task"}, 0.0, False, {})
    agent.update_from_model("```python\nprint('first')\n```")
    agent.update_from_env({"output": "first execution reply"}, 0.5, False, {"step": 1})
    first = agent.trajectory.steps[0]
    agent.update_from_model("```python\nprint('second')\n```")
    agent.update_from_env({"output": "final execution reply"}, 1.0, True, {"step": 2})

    assert first.next_observation == "Output: first execution reply"
    assert agent.trajectory.steps[-1].next_observation == "Output: final execution reply"
    history = format_trajectory_with_feedback(agent.trajectory)
    assert history.count("first execution reply") == 1
    assert history.count("final execution reply") == 1
    # Token-mode training already consumes this complete chat history.
    assert agent.chat_completions[-1] == {"role": "user", "content": "Output: final execution reply"}


def test_mbpp_real_terminal_update_reaches_feedback_history():
    agent = MBPPAgent(max_steps=2)
    agent.update_from_env({"task_description": "Write add(a, b)."}, 0.0, False, {})
    assert agent.trajectory.steps == []
    agent.update_from_model("```python\ndef add(a, b):\n    return a - b\n```")
    agent.update_from_env({"result": "first test reply"}, 0.0, False, {"all_passed": False})
    first = agent.trajectory.steps[0]
    assert first.next_observation == agent.chat_completions[-1]["content"]
    agent.update_from_model("```python\ndef add(a, b):\n    return a * b\n```")
    legacy_before = format_trajectory_as_conversation(agent.trajectory)
    agent.update_from_env({"result": "terminal test failure"}, 0.0, True, {"all_passed": False})

    step = agent.trajectory.steps[-1]
    # The same text that Token-mode training reconstructs from the chat.
    assert step.next_observation == agent.chat_completions[-1]["content"]
    history = format_trajectory_with_feedback(agent.trajectory)
    assert history.count("first test reply") == 1
    assert "terminal test failure" in history
    restored = Trajectory.from_dict(json.loads(json.dumps(agent.trajectory.to_dict())))
    assert format_trajectory_with_feedback(restored) == history
    assert format_trajectory_as_conversation(agent.trajectory) == legacy_before


def test_miniswe_existing_real_agent_reply_survives_serialization():
    agent = SWEAgent()
    agent.update_from_env("initial repository task", 0.0, False, {})
    agent.update_from_model("<function=submit></function>")
    agent.update_from_env("terminal grader reply", 0.0, True, {})
    reply = agent.trajectory.steps[-1].next_observation
    assert reply == "terminal grader reply"
    restored = Trajectory.from_dict(json.loads(json.dumps(agent.trajectory.to_dict())))
    assert restored.steps[-1].next_observation == reply
    assert "terminal grader reply" in format_trajectory_with_feedback(restored)


@pytest.mark.parametrize("agent_type", [AppWorldReactAgent, AppWorldReactAgentACE])
def test_initial_observation_and_reset_create_no_synthetic_reply(agent_type):
    agent = agent_type(accumulate_thinking=False)
    agent.update_from_env({"instruction": "test task"}, 0.0, False, {})
    assert agent.trajectory.steps == []
    agent.update_from_model("```python\nprint('probe')\n```")
    assert agent.trajectory.steps[0].next_observation is None
    agent.reset()
    assert agent.trajectory.steps == []
