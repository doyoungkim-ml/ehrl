"""Opt-in mutation and feedback contracts, with unchanged legacy behavior."""

from types import SimpleNamespace

import pytest

from examples.appworld.train.ace_rl.appworld_adapter import AppWorldReflectorAdapter
from examples.mastermind.train.ace_rl.mastermind_adapter import MastermindReflectorAdapter
from examples.mbpp.train.ace_rl.mbpp_adapter import MBPPReflectorAdapter
from rllm.agents.agent import Action, Step, Trajectory
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.algorithms.ace.context.utils import format_trajectory_as_conversation, format_trajectory_with_feedback


def test_legacy_mutation_semantics_remain_explicitly_separate():
    book = Playbook("[1] original")
    assert book.apply_operations({"operations": [{"type": "UPDATE", "tip_id": "1", "content": "new"}]}) == 0
    assert book.to_string() == "[1] original"
    assert book.apply_operations({"operations": [{"type": "UPDATE", "bullet_id": "1", "content": "original"}]}) == 1


def test_report_normalizes_alias_and_keeps_batch_ids_until_final_renumbering():
    book = Playbook("[1] first\n[2] second")
    report = book.apply_operations_with_report(
        {"operations": [{"type": "REMOVE", "tip_id": 1}, {"type": "UPDATE", "tip_id": "[2]", "content": "corrected"}]}
    )
    assert book.to_string() == "[1] corrected"
    assert report["contract"] == "canonical_operations_v1"
    assert report["requested_count"] == report["applied_count"] == report["normalized_tip_id_count"] == 2
    assert report["changed"] is True
    assert [op["canonical_operation"]["bullet_id"] for op in report["operations"]] == ["1", "2"]
    assert all(op["status"] == "applied" for op in report["operations"])


def test_report_exposes_missing_targets_conflicting_aliases_and_unchanged_updates():
    book = Playbook("[1] original")
    report = book.apply_operations_with_report(
        {"operations": [
            {"type": "UPDATE", "bullet_id": "1", "tip_id": "2", "content": "bad target"},
            {"type": "UPDATE", "bullet_id": "1", "content": "original"},
            {"type": "REMOVE", "bullet_id": "missing"},
            {"type": "UPDATE", "tip_id": False, "content": "invalid"},
            {"type": "ADD", "content": None},
        ]}
    )
    assert book.to_string() == "[1] original"
    assert report["requested_count"] == 5
    assert report["applied_count"] == 0
    assert report["changed"] is False
    assert [(op["status"], op["reason"]) for op in report["operations"]] == [
        ("invalid", "conflicting_bullet_id_and_tip_id"),
        ("unchanged", "content_unchanged"),
        ("not_applied", "bullet_not_found"),
        ("invalid", "invalid_tip_id"),
        ("invalid", "missing_or_invalid_content"),
    ]


def test_report_distinguishes_applied_operations_from_net_book_change():
    book = Playbook("[1] original")
    report = book.apply_operations_with_report(
        [{"type": "UPDATE", "bullet_id": "1", "content": "temporary"}, {"type": "UPDATE", "bullet_id": "1", "content": "original"}]
    )
    assert report["applied_count"] == 2
    assert report["changed"] is False


@pytest.mark.parametrize("payload", [None, {}, {"operations": "wrong"}, "wrong"])
def test_report_rejects_invalid_envelopes_without_mutating(payload):
    book = Playbook("[1] original")
    report = book.apply_operations_with_report(payload)
    assert report["error"] == "operations_must_be_a_list"
    assert report["applied_count"] == 0
    assert book.to_string() == "[1] original"


def test_feedback_history_pairs_actions_with_replies_including_terminal():
    trajectory = Trajectory(steps=[
        Step(observation="initial state", model_response="private duplicated thought", action=Action(action="inspect"), next_observation="inspection reply"),
        Step(observation="inspection reply", action="submit", next_observation="terminal failure detail"),
    ])
    history = format_trajectory_with_feedback(trajectory)
    assert history.index("ACTION:\ninspect") < history.index("ENVIRONMENT REPLY:\ninspection reply")
    assert history.index("ACTION:\nsubmit") < history.index("ENVIRONMENT REPLY:\nterminal failure detail")
    assert history.count("inspection reply") == 1
    assert "private duplicated thought" not in history
    assert "terminal failure detail" not in format_trajectory_as_conversation(trajectory)


@pytest.mark.parametrize("reply", [0, False, "terminal info reply"])
def test_feedback_preserves_present_values_and_info_fallback(reply):
    trajectory = SimpleNamespace(steps=[SimpleNamespace(action="submit", next_observation=None, info={"next_observation": reply})])
    assert format_trajectory_with_feedback(trajectory).endswith(f"ENVIRONMENT REPLY:\n{reply}")


def test_feedback_uses_next_step_observation_without_inventing_terminal_reply():
    trajectory = SimpleNamespace(steps=[SimpleNamespace(action="inspect"), SimpleNamespace(observation="actual reply", action="submit")])
    history = format_trajectory_with_feedback(trajectory)
    assert "ACTION:\ninspect\n\nENVIRONMENT REPLY:\nactual reply" in history
    assert history.endswith("ACTION:\nsubmit\n\nENVIRONMENT REPLY:\n(no environment reply recorded)")


def test_feedback_observation_excerpt_preserves_error_at_end_with_a_bound():
    reply = "RESPONSE HEADER" + "x" * 5000 + "TERMINAL ERROR DETAIL"
    history = format_trajectory_with_feedback(Trajectory(steps=[Step(action="submit", next_observation=reply)]))
    rendered_reply = history.split("ENVIRONMENT REPLY:\n", 1)[1]
    assert rendered_reply.startswith("RESPONSE HEADER")
    assert rendered_reply.endswith("TERMINAL ERROR DETAIL")
    assert "observation truncated" in rendered_reply
    assert len(rendered_reply) == 1500


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8")


@pytest.mark.parametrize("step_count", [1, 2])
def test_feedback_token_budget_retains_terminal_reply_and_enforces_actual_limit(step_count):
    steps = [
        Step(observation="initial " * 100, action="inspect " * 100, next_observation="middle " * 100),
        Step(action="submit " * 100, next_observation="terminal failure detail"),
    ]
    trajectory = Trajectory(steps=steps[-step_count:])
    tokenizer = CharacterTokenizer()
    history = format_trajectory_with_feedback(trajectory, tokenizer=tokenizer, max_tokens=160)
    assert len(tokenizer.encode(history)) <= 160
    assert history.endswith("terminal failure detail")


def test_feedback_tiny_token_budget_prioritizes_terminal_reply():
    trajectory = Trajectory(steps=[Step(action="inspect " * 100, next_observation="terminal failure detail")])
    tokenizer = CharacterTokenizer()
    history = format_trajectory_with_feedback(trajectory, tokenizer=tokenizer, max_tokens=30)
    assert len(tokenizer.encode(history)) <= 30
    assert history.endswith("terminal failure detail")


def test_feedback_empty_history_still_obeys_token_budget():
    tokenizer = CharacterTokenizer()
    history = format_trajectory_with_feedback(Trajectory(), tokenizer=tokenizer, max_tokens=1)
    assert len(tokenizer.encode(history)) <= 1


@pytest.mark.parametrize("adapter_type", [AppWorldReflectorAdapter, MBPPReflectorAdapter])
def test_adapters_preserve_legacy_default_and_select_new_format_explicitly(adapter_type):
    trajectory = Trajectory(steps=[Step(observation="state", action="submit", next_observation="terminal reply")])
    assert adapter_type().format_conversation_history(trajectory) == format_trajectory_as_conversation(trajectory)
    assert "terminal reply" in adapter_type(handoff_protocol="feedback_v1").format_conversation_history(trajectory)
    with pytest.raises(ValueError, match="Unsupported handoff_protocol"):
        adapter_type(handoff_protocol="unknown")


def test_mbpp_feedback_keeps_summary_and_applies_the_optional_history_budget():
    steps = [Step(observation="problem", action="def f(): pass", next_observation="Turn 1:\n" + "x" * 3000 + "\nterminal assertion failed", info={"all_passed": False})]
    trajectory = Trajectory(steps=steps)
    for summary in (False, True):
        legacy, repaired = MBPPReflectorAdapter(evaluation_summary=summary), MBPPReflectorAdapter(evaluation_summary=summary, handoff_protocol="feedback_v1")
        legacy_vars = legacy.get_template_variables(trajectory, 0.0, "[1] tip", max_turns=6)
        repaired_vars = repaired.get_template_variables(trajectory, 0.0, "[1] tip", max_turns=6)
        assert legacy_vars["trajectory_summary"] == repaired_vars["trajectory_summary"]
        assert legacy_vars["conversation_history"] == format_trajectory_as_conversation(trajectory)
        assert "terminal assertion failed" in repaired_vars["conversation_history"]
    tokenizer = CharacterTokenizer()
    bounded = MBPPReflectorAdapter(handoff_protocol="feedback_v1", tokenizer=tokenizer, max_history_tokens=400).format_conversation_history(trajectory)
    assert len(tokenizer.encode(bounded)) <= 400
    assert bounded.endswith("terminal assertion failed")


def test_mastermind_feedback_keeps_the_guess_table_without_a_token_budget():
    # Evaluation steps carry the parsed guess and its feedback in step.info.
    steps = [
        SimpleNamespace(action=Action(action="0 1 2 3"), info={"exact_matches": 1, "value_matches": 2}, next_observation=None),
        SimpleNamespace(action=Action(action="4 3 2 5"), info={"exact_matches": 4, "value_matches": 0, "won": True}, next_observation="Game over. Secret code was: 4 3 2 5"),
    ]
    trajectory = SimpleNamespace(task={"code_length": 4, "num_digits": 6}, steps=steps)
    legacy = MastermindReflectorAdapter().get_template_variables(trajectory, 1.0, "[1] tip")
    repaired = MastermindReflectorAdapter(handoff_protocol="feedback_v1").get_template_variables(trajectory, 1.0, "[1] tip")
    assert repaired == legacy
    assert "Turn 2: 4 3 2 5 -> exact=4, value=0" in repaired["conversation_history"]
    assert "Secret code" not in repaired["conversation_history"]
    with pytest.raises(ValueError, match="Unsupported handoff_protocol"):
        MastermindReflectorAdapter(handoff_protocol="unknown")
    with pytest.raises(ValueError, match="max_history_tokens"):
        MastermindReflectorAdapter(handoff_protocol="feedback_v1", tokenizer=CharacterTokenizer(), max_history_tokens=100)


@pytest.mark.parametrize("kwargs", [{"obs_max_chars": -1}, {"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": 100}])
def test_feedback_rejects_invalid_limits_and_missing_tokenizer(kwargs):
    with pytest.raises(ValueError):
        format_trajectory_with_feedback(Trajectory(), **kwargs)
