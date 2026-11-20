"""ACE evaluation ablations and trajectory formatting with real context objects."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from rllm.agents.agent import Action, Step, Trajectory
from rllm.agents.swe_agent import SWEAgent
from rllm.algorithms.ace.context.rule import ACEConfig, ACEContextRule
from rllm.algorithms.ace.context.utils import format_trajectory_actions_only, format_trajectory_as_conversation


class RecordingEngine:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def get_model_response(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=json.dumps(response))


def make_rule(tmp_path, responses, *, reflector_only=False, initial_playbook="[1] Seed instruction", create_curator_prompt=True):
    reflector_path = tmp_path / "reflector.txt"
    reflector_path.write_text("REFLECTOR|{{playbook}}|{{conversation_history}}")
    curator_path = tmp_path / "curator.txt"
    if create_curator_prompt:
        curator_path.write_text("CURATOR|{{current_playbook}}|{{guidebook}}")
    playbook_path = tmp_path / "playbook.txt"
    playbook_path.write_text(initial_playbook)
    config = ACEConfig(
        reflector_prompt_path=str(reflector_path),
        curator_prompt_path=str(curator_path),
        initial_playbook_path=str(playbook_path),
        ace_log_dir=str(tmp_path / "logs"),
        max_retries=1,
        reflector_only=reflector_only,
    )
    engine = RecordingEngine(responses)
    rule = ACEContextRule(config, rollout_engine=engine, reflector_system_prompt="reflection role", curator_system_prompt="curation role")
    return rule, engine


def test_full_ace_default_reviews_reflections_and_uses_one_playbook_snapshot(tmp_path):
    initial_playbook = "[1] Seed instruction"
    rule, engine = make_rule(
        tmp_path,
        [
            {"key_insight": "first insight"},
            {"key_insight": "second insight"},
            {"operations": [{"type": "UPDATE", "bullet_id": "1", "content": "Reviewed instruction"}]},
            {"operations": [{"type": "ADD", "section": "others", "content": "Reviewed addition"}]},
        ],
    )
    trajectories = [Trajectory(steps=[Step(action="first")], reward=0), Trajectory(steps=[Step(action="second")], reward=0)]
    asyncio.run(rule.on_batch_complete(trajectories))

    assert rule.config.reflector_only is False
    assert rule.curator is not None
    assert len(engine.calls) == 4
    assert [call[0][0]["content"] for call in engine.calls] == ["reflection role", "reflection role", "curation role", "curation role"]
    assert all(initial_playbook in call[0][-1]["content"] for call in engine.calls)
    assert [call[0][-1]["content"] for call in engine.calls[2:]] == [
        "CURATOR|[1] Seed instruction|first insight",
        "CURATOR|[1] Seed instruction|second insight",
    ]
    assert rule.get_context() == "[1] Reviewed instruction\n[2] Reviewed addition"
    assert rule.get_stats()["operations_applied"] == 2
    assert json.loads((tmp_path / "logs/batch_0000/batch_summary.json").read_text())["curator_skipped"] == 0


@pytest.mark.parametrize("initial_playbook", ["", "## STRATEGIES AND HARD RULES\n[shr-00001] Seed instruction"])
def test_reflector_only_skips_curator_assets_and_calls_and_preserves_log_alignment(tmp_path, initial_playbook):
    rule, engine = make_rule(
        tmp_path,
        [{"key_insight": "  Useful insight  "}, RuntimeError("model unavailable"), {"key_insight": "  "}, {"key_insight": 42}, {"key_insight": "Final insight"}],
        reflector_only=True,
        initial_playbook=initial_playbook,
        create_curator_prompt=False,
    )
    trajectories = [Trajectory(steps=[Step(action=str(index))], reward=0) for index in range(5)]
    asyncio.run(rule.on_batch_complete(trajectories))

    assert rule.curator is None
    assert len(engine.calls) == 5
    assert all(call[0][0]["content"] == "reflection role" for call in engine.calls)
    assert "Useful insight" in rule.get_context()
    assert "Final insight" in rule.get_context()
    assert "42" not in rule.get_context()
    assert rule.get_stats()["operations_applied"] == 2
    if initial_playbook:
        assert "## OTHERS" in rule.get_context()
        assert "[misc-" in rule.get_context()
    else:
        assert rule.get_context() == "[1] Useful insight\n[2] Final insight"
    for index in range(5):
        logs = json.loads((tmp_path / f"logs/batch_0000/task_{index:04d}.json").read_text())
        curator_log = logs[1]
        assert curator_log["skipped"] is True
        assert curator_log["reason"] == "reflector_only"
        if index in (1, 2, 3):
            assert curator_log["empty_key_insight"] is True
            assert curator_log["parsed"] == {}
        else:
            assert curator_log["parsed"]["operations"][0]["type"] == "ADD"
    assert rule.get_stats()["trajectories_processed"] == 5
    assert rule.pending_trajectories == []


def test_default_formatter_preserves_observation_cap_and_reasoning():
    trajectory = Trajectory(
        steps=[
            Step(observation="x" * 301, thought="first thought", model_response="unused", action=Action(action="inspect")),
            Step(observation="ready", model_response="<think>second thought</think>", action="submit", next_observation="final reply"),
        ]
    )
    expected = (
        "\n=== EPISODE TRAJECTORY ===\n\n\n--- Step 1 ---\nOBSERVATION:\n" + "x" * 300 + "...\nAGENT REASONING: first thought\nACTION: inspect\n\n--- Step 2 ---\nOBSERVATION:\nready\n"
        "AGENT REASONING: <think>second thought</think>\nACTION: submit"
    )
    assert format_trajectory_as_conversation(trajectory) == expected
    assert format_trajectory_as_conversation(trajectory, obs_max_chars=300) == expected
    assert "x" * 301 in format_trajectory_as_conversation(trajectory, obs_max_chars=0)


def test_actions_only_pairs_real_swe_actions_with_terminal_environment_reply():
    agent = SWEAgent()
    agent.update_from_env("Fix the issue", 0, False, {})
    agent.update_from_model("private reasoning\n<function=bash><parameter=command>inspect</parameter></function>")
    agent.update_from_env("file contents", 0, False, {})
    agent.update_from_model("more private reasoning\n<function=submit></function>")
    agent.update_from_env("terminal grading reply", 1, True, {})

    history = format_trajectory_actions_only(agent.trajectory, include_initial_user_message=False)
    assert history == (
        "=== Turn 1 (Generator Action) ===\n\n<function=bash><parameter=command>inspect</parameter></function>\n\n"
        "=== Turn 1 (Observation) ===\n\nfile contents\n\n"
        "=== Turn 2 (Generator Action) ===\n\n<function=submit></function>\n\n"
        "=== Turn 2 (Observation) ===\n\nterminal grading reply"
    )
    with_initial = format_trajectory_actions_only(agent.trajectory)
    assert with_initial.startswith("=== Initial User Message ===\n\n")
    assert "Fix the issue" in with_initial
    assert "private reasoning" not in with_initial


@pytest.mark.parametrize("observation", [False, 0, "reply"])
@pytest.mark.parametrize("storage", ["next_observation", "info"])
def test_actions_only_preserves_present_false_values_and_supports_info_fallback(observation, storage):
    step = SimpleNamespace(action=0)
    if storage == "next_observation":
        step.next_observation = observation
        step.info = {"next_observation": "stale fallback"}
    else:
        step.next_observation = None
        step.info = {"next_observation": observation}
    history = format_trajectory_actions_only(SimpleNamespace(steps=[step]))
    assert history == f"=== Turn 1 (Generator Action) ===\n\n0\n\n=== Turn 1 (Observation) ===\n\n{observation}"


@pytest.mark.parametrize("action", [None, " ", Action(action=None), Action(action=" ")])
def test_actions_only_normalizes_missing_actions(action):
    history = format_trajectory_actions_only(Trajectory(steps=[Step(action=action)]))
    assert history == "=== Turn 1 (Generator Action) ===\n\n(no action)\n\n=== Turn 1 (Observation) ===\n\n(no observation)"


def test_actions_only_caps_observations_but_keeps_complete_actions():
    trajectory = Trajectory(steps=[Step(observation="initial task", action="complete command", next_observation="123456")])
    capped = format_trajectory_actions_only(trajectory, obs_max_chars=3)
    assert "ini\n... (truncated, 12 total chars)" in capped
    assert "123\n... (truncated, 6 total chars)" in capped
    assert "complete command" in capped
    uncapped = format_trajectory_actions_only(trajectory, obs_max_chars=0)
    assert "initial task" in uncapped and "123456" in uncapped
    assert "truncated" not in uncapped


@pytest.mark.parametrize("formatter", [format_trajectory_as_conversation, format_trajectory_actions_only])
@pytest.mark.parametrize("cap", [-1, 1.5, True, None])
def test_invalid_observation_caps_fail_instead_of_silently_truncating(formatter, cap):
    with pytest.raises(ValueError, match="nonnegative integer"):
        formatter(Trajectory(), obs_max_chars=cap)


@pytest.mark.parametrize("formatter", [format_trajectory_as_conversation, format_trajectory_actions_only])
def test_empty_trajectory_is_explicit(formatter):
    assert formatter(Trajectory()) == "(No steps recorded in trajectory)"


class CandidateRoleEngine:
    def __init__(self, *, empty_insights=False):
        self.empty_insights = empty_insights
        self.reflections = []
        self.curations = []

    async def get_model_response(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if prompt.startswith("REFLECTOR|"):
            index = len(self.reflections)
            self.reflections.append(prompt)
            result = {"key_insight": "" if self.empty_insights else f"insight-{index}"}
        else:
            index = len(self.curations)
            self.curations.append(prompt)
            result = {"operations": [{"type": "ADD", "section": "others", "content": f"curated-{index} instruction"}]}
        return SimpleNamespace(text=json.dumps(result))


class CandidateGeneratorAgent:
    def reset(self):
        self.messages = []


class CandidateGeneratorEngine:
    agent_class = CandidateGeneratorAgent

    def __init__(self, *, reject_candidates=False):
        self.reject_candidates = reject_candidates
        self.prompts = []

    async def execute_tasks(self, tasks):
        import re

        trajectories = []
        for task in tasks:
            agent = self.agent_class()
            agent.reset()
            prompt = agent.messages[0]["content"]
            self.prompts.append(prompt)
            indices = [int(value) for value in re.findall(r"(?:insight|curated)-(\d+)", prompt)]
            reward = (0.1 if self.reject_candidates else 0.5 + 0.05 * max(indices)) if indices else 0.4
            trajectories.append(Trajectory(task=task, reward=reward, steps=[Step(observation=task["instruction"], action="inspect", reward=reward, done=True)]))
        return trajectories


def run_candidate_evaluation(tmp_path, *, mode, reflector_only, g_ref=2, g_cur=3, selection_mode="best", empty_insights=False, reject_candidates=False):
    from rllm.algorithms.ace import evaluation

    role_engine = CandidateRoleEngine(empty_insights=empty_insights)
    rule, _ = make_rule(tmp_path, [], reflector_only=reflector_only, create_curator_prompt=not reflector_only)
    rule.reflector.rollout_engine = role_engine
    if rule.curator is not None:
        rule.curator.rollout_engine = role_engine
    generator_engine = CandidateGeneratorEngine(reject_candidates=reject_candidates)
    (tmp_path / "base_system_prompt.txt").write_text("Advice: {{playbook}}")
    hooks = evaluation.ACEExperimentHooks(
        experiment_name="test",
        task_id_getter=lambda task: task["id"],
        build_generator_engine=lambda args: generator_engine,
        create_context_rule=lambda *args, **kwargs: rule,
        trajectory_reward=lambda trajectory: trajectory.reward,
        trajectory_success=lambda trajectory: trajectory.reward >= 1,
    )
    args = SimpleNamespace(
        model_name="test-model",
        local_eval_protocol=evaluation.LEGACY_LOCAL_PROTOCOL,
        train_batch_size=1,
        epochs=1,
        local_eval_epochs=1,
        g_act=2,
        g_ref=g_ref,
        g_cur=g_cur,
        selection_mode=selection_mode,
        test_epoch_freq=1,
        val_before_train=False,
        max_turns=2,
    )
    tasks = [{"id": "task", "instruction": "Fix the issue"}]
    common = dict(
        args=args,
        hooks=hooks,
        prompts_dir=str(tmp_path),
        initial_playbook_path=str(tmp_path / "playbook.txt"),
        playbook_snapshot_dir=None,
        ace_log_dir=None,
        metrics_file=None,
        script_path=__file__,
    )
    if mode == "global":
        result = asyncio.run(evaluation.run_global_ace(**common, train_tasks=tasks, eval_tasks=[], chat_completion_dir=None))
        metrics = result.train_steps[0]["metrics"]
        candidate_count = metrics["num_candidates"]
        rejected = metrics["reject_candidate"]
    else:
        result = asyncio.run(evaluation.run_local_ace(**common, train_tasks=[], eval_tasks=tasks))
        metrics = result.evaluations[0]["metrics"]
        candidate_count = metrics["ace/candidates"]
        rejected = bool(metrics["ace/reject_candidate"])
    return rule, role_engine, generator_engine, metrics, candidate_count, rejected


@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("reflector_only", [False, True])
@pytest.mark.parametrize("g_cur", [0, 1, 3])
def test_complete_candidate_paths_honor_curator_mode_and_sampling(tmp_path, mode, reflector_only, g_cur):
    rule, roles, generator, metrics, candidate_count, _ = run_candidate_evaluation(tmp_path, mode=mode, reflector_only=reflector_only, g_cur=g_cur)
    expected_candidates = 2 if reflector_only else 2 * g_cur
    assert len(roles.reflections) == 4
    assert len(roles.curations) == (0 if reflector_only else 2 * g_cur)
    assert candidate_count == expected_candidates
    assert len(generator.prompts) == 2 * (1 + expected_candidates)
    assert all("Seed instruction" in prompt for prompt in roles.reflections)
    assert all("[2]" not in prompt for prompt in roles.reflections + roles.curations)
    if reflector_only:
        assert "insight-1" in rule.get_context() and "insight-3" in rule.get_context()
        assert "insight-0" not in rule.get_context() and "insight-2" not in rule.get_context()
    elif g_cur:
        assert f"curated-{2 * g_cur - 1}" in rule.get_context()
        assert "insight-" not in rule.get_context()
    else:
        assert rule.get_context() == "[1] Seed instruction"
    if mode == "local":
        assert metrics["ace/curator_rollouts"] == len(roles.curations)
        assert metrics["ace/reflector_rollouts"] == 4


@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("reflector_only", [False, True])
def test_candidate_rejection_preserves_seed_context_in_both_modes(tmp_path, mode, reflector_only):
    rule, _, _, _, count, rejected = run_candidate_evaluation(
        tmp_path,
        mode=mode,
        reflector_only=reflector_only,
        selection_mode="best_rejection",
        reject_candidates=True,
    )
    assert count > 0
    assert rejected
    assert rule.get_context() == "[1] Seed instruction"


@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("reflector_only", [False, True])
def test_random_selection_uses_same_candidate_indices_in_both_modes(tmp_path, monkeypatch, mode, reflector_only):
    from rllm.algorithms.ace import evaluation

    monkeypatch.setattr(evaluation.random, "randint", lambda start, end: 0)
    rule, _, _, _, _, _ = run_candidate_evaluation(tmp_path, mode=mode, reflector_only=reflector_only, selection_mode="random")
    if reflector_only:
        assert "insight-0" in rule.get_context() and "insight-2" in rule.get_context()
        assert "insight-1" not in rule.get_context()
    else:
        assert "curated-0" in rule.get_context()
        assert "curated-1" not in rule.get_context()


@pytest.mark.parametrize("mode", ["local", "global"])
def test_empty_reflector_insights_do_not_create_candidates_or_call_curator(tmp_path, mode):
    rule, roles, generator, _, count, _ = run_candidate_evaluation(tmp_path, mode=mode, reflector_only=True, empty_insights=True)
    assert len(roles.reflections) == 4
    assert not roles.curations
    assert count == 0
    assert len(generator.prompts) == 2
    assert rule.get_context() == "[1] Seed instruction"


@pytest.mark.parametrize("mode", ["local", "global"])
def test_zero_reflections_runs_vanilla_without_role_calls(tmp_path, mode):
    rule, roles, generator, _, count, _ = run_candidate_evaluation(tmp_path, mode=mode, reflector_only=False, g_ref=0, g_cur=0)
    assert not roles.reflections and not roles.curations
    assert count == 0
    assert len(generator.prompts) == 2
    assert rule.get_context() == "[1] Seed instruction"
