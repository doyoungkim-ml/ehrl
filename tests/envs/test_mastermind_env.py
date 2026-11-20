"""Mastermind keeps partial credit while normalizing every game to reward one."""

import itertools
from types import SimpleNamespace

import pytest

from examples.mastermind.train.ace_rl.mastermind_adapter import MastermindReflectorAdapter
from rllm.agents.mastermind_agent import MastermindAgent
from rllm.environments.mastermind.mastermind import MastermindEnv


@pytest.mark.parametrize("length", [3, 4, 5])
def test_solved_games_score_one_at_every_dataset_code_length(length):
    env = MastermindEnv(code_length=length, seed=42)
    env.reset()
    _, reward, done, info = env.step(env.secret_code)
    assert reward == 1.0
    assert done and info["won"]


def test_partial_credit_preserves_exact_to_value_ratio():
    env = MastermindEnv(code_length=4, num_digits=6)
    env.reset()
    env.secret_code = [0, 1, 2, 3]
    assert env.step([0, 1, 5, 5])[1] == 0.5
    assert env.step([1, 0, 5, 5])[1] == 0.25
    assert env.step([5, 5, 5, 5])[1] == 0.0


def test_all_guesses_stay_in_unit_interval_including_duplicate_digits():
    env = MastermindEnv(code_length=3, num_digits=3, max_turns=100)
    for secret in itertools.product(range(3), repeat=3):
        for guess in itertools.product(range(3), repeat=3):
            env.reset()
            env.secret_code = list(secret)
            _, reward, _, info = env.step(list(guess))
            assert 0 <= reward <= 1
            assert (reward == 1) == (guess == secret)
            assert info["won"] == (guess == secret)


def test_custom_reward_weights_preserve_scale_invariance():
    env = MastermindEnv(code_length=4, reward_exact_match=2, reward_value_match=1)
    env.reset()
    env.secret_code = [0, 1, 2, 3]
    assert env.step([0, 1, 5, 5])[1] == 0.5
    assert env.step([0, 1, 2, 3])[1] == 1


def test_invalid_guess_has_zero_reward():
    env = MastermindEnv(seed=42)
    env.reset()
    obs, reward, done, info = env.step("not a guess")
    assert reward == 0 and not done and not info["won"]
    assert "error" in obs


@pytest.mark.parametrize("won,reward", [(False, 0.5), (True, 1.0)])
def test_reflector_outcome_uses_solved_flag(won, reward):
    adapter = MastermindReflectorAdapter(max_turns=20)
    adapter.format_conversation_history = lambda trajectory: "guess and feedback"
    trajectory = SimpleNamespace(steps=[SimpleNamespace(info={"won": won})])
    variables = adapter.get_template_variables(trajectory, reward, "hint")
    assert variables["outcome"] == ("SUCCESS" if won else "FAILED")
    assert variables["playbook"] == "hint"


def test_step_info_carries_each_guess_feedback_including_the_last():
    env = MastermindEnv(code_length=4, num_digits=6, max_turns=2)
    env.reset()
    env.secret_code = [0, 1, 2, 3]
    info = env.step("0 1 5 5")[3]
    assert (info["exact_matches"], info["value_matches"]) == (2, 0)
    _, _, done, info = env.step("5 5 5")
    assert done and "exact_matches" not in info and "4 digits" in info["guess_error"]


@pytest.mark.parametrize(
    ("reply", "guess"),
    [
        ("```python\n```0 1 2```", "0 1 2"),
        ("reasoning\n```0 1 2 3```", "0 1 2 3"),
        ("```text\n3 3 1\n```", "3 3 1"),
        ("first ```0 0 0``` then ```1 2 3```", "1 2 3"),
    ],
)
def test_guess_is_read_from_the_last_fenced_digits(reply, guess):
    assert MastermindAgent()._parse_model_response(reply)[1] == guess


def test_copied_game_state_is_still_rejected():
    action = MastermindAgent()._parse_model_response("```plaintext\nTurn 2:\nCode length: 4\n```")[1]
    env = MastermindEnv(seed=1)
    env.reset()
    assert "guess_error" in env.step(action)[3]


def test_guess_table_marks_rejections_repeats_and_contradictions():
    env = MastermindEnv(code_length=4, num_digits=6, max_turns=10)
    env.reset()
    env.secret_code = [2, 1, 3, 5]
    steps = []
    for action in ["0 1 2 3", "python", "0 1 2 3", "4 5 4 5"]:
        _, reward, _, info = env.step(action)
        steps.append(SimpleNamespace(action=action, reward=reward, info=info, model_response=f"why {action}"))
    task = {"code_length": 4, "num_digits": 6}
    table = MastermindReflectorAdapter().format_conversation_history(SimpleNamespace(steps=steps, task=task))
    assert "Accepted guesses: 3 (2 distinct), rejected: 1, repeats of an earlier guess: 1, guesses contradicting earlier feedback: 1" in table
    assert "Turn 2: python -> REJECTED" in table
    assert "Turn 3: 0 1 2 3 -> exact=1, value=2 reward=0.5 [REPEATS turn 1]" in table
    assert "Turn 4: 4 5 4 5 -> exact=1, value=0 reward=0.25 [contradicts feedback of turn 1]" in table
    assert "2 1 3 5" not in table


def test_reflection_outcome_follows_the_environment_not_the_model():
    adapter = MastermindReflectorAdapter()
    lost = {"trajectory": SimpleNamespace(steps=[SimpleNamespace(info={"won": False})])}
    won = {"trajectory": SimpleNamespace(steps=[SimpleNamespace(info={"won": True})])}
    assert adapter.finalize_reflection({"outcome": "SUCCESS", "key_insight": "k"}, lost) == {"outcome": "FAILED", "key_insight": "k"}
    assert adapter.finalize_reflection({"outcome": "FAILED"}, won)["outcome"] == "SUCCESS"


def test_training_reconstruction_gives_the_same_guess_table():
    from rllm.algorithms.ace.context.training_inputs import steps_from_training_chat

    agent = MastermindAgent(max_steps=3)
    env = MastermindEnv(code_length=3, num_digits=4, max_turns=3)
    observation, _ = env.reset()
    env.secret_code = [1, 2, 3]
    agent.update_from_env(observation, 0, False, {})
    for reply in ["```python\n```0 1 2```", "```0 1 2```", "```1 2 3```"]:
        action = agent.update_from_model(reply).action
        observation, reward, done, info = env.step(action)
        agent.update_from_env(observation, reward, done, info)
    steps = steps_from_training_chat(agent.chat_completions, [s.reward for s in agent.trajectory.steps], all_passed=True)
    task = {"code_length": 3, "num_digits": 4}
    table = MastermindReflectorAdapter().format_conversation_history(SimpleNamespace(steps=steps, task=task))
    assert "Turn 1: 0 1 2 -> exact=0, value=2" in table
    assert "Turn 2: 0 1 2 -> exact=0, value=2 reward=0.333 [REPEATS turn 1]" in table
    assert "Turn 3: 1 2 3 -> exact=3, value=0 reward=1" in table
