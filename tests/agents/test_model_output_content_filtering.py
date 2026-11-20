from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.agents.mastermind_agent import MastermindAgent
from rllm.engine.rollout import ModelOutput


def qwen_output(raw: str, content: str) -> ModelOutput:
    return ModelOutput(text=raw, content=content, reasoning="")


def test_code_agent_uses_filtered_model_output_content_for_action_and_history():
    agent = CompetitionCodingAgent(accumulate_thinking=False)
    agent.update_from_env({"question": "Print ok"}, 0.0, False, {})

    raw = "print('ok')<|im_end|>"
    filtered = "print('ok')"
    action = agent.update_from_model(raw, model_output=qwen_output(raw, filtered))

    assert action.action == filtered
    assert agent.messages[-1]["content"] == filtered
    assert agent.trajectory.steps[-1].model_response == filtered


def test_mastermind_agent_records_filtered_model_output_content_in_step():
    agent = MastermindAgent(use_accumulate_thinking=False)
    agent.update_from_env({"instructions": "Guess", "code_length": 4, "num_digits": 6}, 0.0, False, {})

    raw = "I should guess. ```1234```<|im_end|>"
    filtered = "I should guess. ```1234```"
    action = agent.update_from_model(raw, model_output=qwen_output(raw, filtered))

    assert action.action == "1234"
    assert agent.messages[-1]["content"] == filtered
    assert agent.messages[-1]["raw_completion"] == raw
    assert agent.trajectory.steps[-1].model_response == filtered


def test_appworld_agent_uses_filtered_model_output_content_for_checkpoint_history():
    agent = AppWorldReactAgent(accumulate_thinking=False)
    agent._initialize_from_task(
        {
            "instruction": "Inspect APIs",
            "user_info": {
                "first_name": "Test",
                "last_name": "User",
                "email": "test@example.com",
                "phone_number": "+123****7890",
            },
        }
    )
    agent.current_observation = "observation"

    raw = "```python\nprint('ok')\n```\n<|im_end|>"
    filtered = "```python\nprint('ok')\n```"
    action = agent.update_from_model(raw, model_output=qwen_output(raw, filtered))

    assert action.action == "print('ok')"
    assert agent.messages[-1]["content"] == filtered
    assert agent.messages[-1]["raw_completion"] == raw
    assert agent.trajectory.steps[-1].model_response == filtered
