"""MBPP feedback modes: full is the historical view, hidden never shows a test."""

import json
import re

import pytest

from rllm.agents.mbpp_agent import MBPPAgent
from rllm.environments.mbpp.mbpp import MBPPEnv, run_code_with_tests

TESTS = ["assert scale(17) == 4242", "assert scale(3) == 9191", "assert scale(5) == 7373"]
# Assertion source and expected values must never reach a hidden observation.
SECRETS = ("assert scale", "scale(17)", "scale(3)", "scale(5)", "4242", "9191", "7373")
TASK = {"task_id": 7, "text": "Scale a number.", "test_list": TESTS, "test_setup_code": ""}


def submission(code):
    return f"```python\n{code}\n```"


def run_episode(code, feedback=None, timeout=10):
    task = dict(TASK, timeout=timeout)
    if feedback is not None:
        task["feedback"] = feedback
    env = MBPPEnv.from_dict(task)
    agent = MBPPAgent(max_steps=6)
    observation, info = env.reset()
    agent.update_from_env(observation, 0.0, False, info)
    agent.update_from_model(submission(code))
    observation, reward, done, info = env.step(submission(code))
    agent.update_from_env(observation, reward, done, info)
    return env, observation, reward, info, agent.messages[-1]["content"]


def visible_text(env, observation, info, agent_text):
    return "\n".join((json.dumps(observation), json.dumps(info), json.dumps(env.last_test_results), agent_text))


# One correct test (scale(3)); the other two fail in each way listed.
FAILURES = {
    "wrong_answer": ("def scale(x):\n    return 9191 if x == 3 else 0", "wrong answer"),
    "exception_in_function": ("def scale(x):\n    if x == 3:\n        return 9191\n    return x + 'a'", "TypeError: unsupported operand type(s) for +: 'int' and 'str'"),
    "candidate_assert": ("def scale(x):\n    assert x == 3\n    return 9191", "AssertionError"),
    "eq_raises_expected_value": (
        "class Leak:\n    def __eq__(self, other):\n        raise ValueError(other)\n\ndef scale(x):\n    return 9191 if x == 3 else Leak()",
        "ValueError",
    ),
    "input_in_message": ("def scale(x):\n    return {3: 9191}[x]", "KeyError"),
    "reads_test_source": (
        "import sys\n\ndef scale(x):\n    if x == 3:\n        return 9191\n    raise RuntimeError(open(sys.argv[0]).read().splitlines()[-2])",
        "RuntimeError",
    ),
}


@pytest.mark.parametrize("name", FAILURES)
def test_hidden_failures_show_outcomes_but_no_test_source_or_values(name):
    code, summary = FAILURES[name]
    env, observation, reward, info, agent_text = run_episode(code, feedback="hidden")
    assert reward == pytest.approx(1 / 3)
    results = observation["test_results"]
    assert [result["test"] for result in results] == ["hidden test 1", "hidden test 2", "hidden test 3"]
    assert [result["passed"] for result in results] == [False, True, False]
    assert results[0]["error"] == results[2]["error"] == summary
    assert "Test 1: hidden test 1" in agent_text and f"Error: {summary}" in agent_text
    text = visible_text(env, observation, info, agent_text)
    for secret in SECRETS:
        assert secret not in text


@pytest.mark.parametrize(
    ("code", "summary"),
    [
        ("def scale(x):\n    return (x", "SyntaxError: '(' was never closed"),
        ("def scale(x)\n    return x", "SyntaxError: expected ':'"),
        ("def scale(x):\nreturn x", "IndentationError: expected an indented block after function definition on line 1"),
        # An unclosed bracket makes Python print the assertion line itself.
        ("def scale(x):\n    return [x,", "SyntaxError: '[' was never closed"),
    ],
)
def test_hidden_syntax_errors_keep_only_the_final_line(code, summary):
    env, observation, reward, info, agent_text = run_episode(code, feedback="hidden")
    assert reward == 0.0
    assert [result["error"] for result in observation["test_results"]] == [summary] * 3
    text = visible_text(env, observation, info, agent_text)
    for secret in SECRETS:
        assert secret not in text


def test_hidden_timeout_and_early_exit_keep_their_messages():
    env, observation, reward, info, agent_text = run_episode("def scale(x):\n    while True:\n        pass", feedback="hidden", timeout=1)
    assert reward == 0.0
    assert {result["error"] for result in observation["test_results"]} == {"Timeout: execution exceeded 1 seconds"}
    env, observation, reward, info, agent_text = run_episode("import sys\n\ndef scale(x):\n    return 0\n\nsys.exit(0)", feedback="hidden")
    assert reward == 0.0
    assert {result["error"] for result in observation["test_results"]} == {"Test did not run: the program exited before the test assertion finished"}
    for secret in SECRETS:
        assert secret not in visible_text(env, observation, info, agent_text)


def test_full_feedback_is_the_default_and_shows_assertions():
    code = FAILURES["wrong_answer"][0]
    default = run_episode(code)
    explicit = run_episode(code, feedback="full")
    for env, observation, reward, info, agent_text in (default, explicit):
        assert env.feedback == "full"
        assert [result["test"] for result in observation["test_results"]] == TESTS
        error = observation["test_results"][0]["error"]
        assert error.startswith("Traceback") and "assert scale(17) == 4242" in error and error.endswith("AssertionError")
        assert "Test 1: assert scale(17) == 4242" in agent_text
    # Each run uses a fresh temporary directory, which tracebacks name.
    def normalized(text):
        return re.sub(r"rllm-mbpp-test-[^/]+", "TMP", text)

    assert normalized(json.dumps(default[1])) == normalized(json.dumps(explicit[1]))
    assert normalized(default[4]) == normalized(explicit[4])


@pytest.mark.parametrize("code", [code for code, _ in FAILURES.values()] + ["def scale(x):\n    return {17: 4242, 3: 9191, 5: 7373}[x]", "def scale(x):\n    return (x"])
def test_reward_and_pass_flags_do_not_depend_on_feedback(code):
    full = run_episode(code, feedback="full")
    hidden = run_episode(code, feedback="hidden")
    assert full[2] == hidden[2]
    assert full[3] == hidden[3]
    for key in ("passed_tests", "total_tests", "all_passed"):
        assert full[1][key] == hidden[1][key]
    assert [result["passed"] for result in full[1]["test_results"]] == [result["passed"] for result in hidden[1]["test_results"]]


def test_parallel_hidden_results_keep_test_order_and_distinct_labels():
    tests = [TESTS[0], TESTS[1], TESTS[0]]
    result = run_code_with_tests(FAILURES["wrong_answer"][0], tests, timeout=10, parallel=True, feedback="hidden")
    assert [item["test"] for item in result["test_results"]] == ["hidden test 1", "hidden test 2", "hidden test 3"]
    assert [item["passed"] for item in result["test_results"]] == [False, True, False]
    assert result["passed_tests"] == 1


def test_from_dict_reads_feedback_and_rejects_unknown_modes():
    assert MBPPEnv.from_dict(dict(TASK)).feedback == "full"
    assert MBPPEnv.from_dict(dict(TASK, feedback="hidden")).feedback == "hidden"
    for value in ("partial", "Hidden", "", None):
        with pytest.raises(ValueError, match="MBPP feedback"):
            MBPPEnv.from_dict(dict(TASK, feedback=value))
    with pytest.raises(ValueError, match="MBPP feedback"):
        run_code_with_tests("x = 1", TESTS, feedback="none")
