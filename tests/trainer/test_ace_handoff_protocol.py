"""Execute actual ACE protocol setup, mutation blocks, and checkpoint guards.

Only the distributed trainer imports are isolated. These CPU tests use real
benchmark adapters, Playbooks, and controller checkpoint code.
"""

import ast
import json
import logging
from types import SimpleNamespace

import pytest
from test_ace_package_boundaries import ROOT, _benchmark_defaults, _initialize
from test_local_ace_checkpoint_resume import Playbook, helper, trainer
from test_standard_rejection_update import execute, generation


@pytest.mark.parametrize("benchmark", ["appworld", "mini_swe_bench", "mastermind", "mbpp"])
@pytest.mark.parametrize("protocol", ["legacy", "feedback_v1"])
def test_protocol_initializes_matching_curator_asset_and_adapter(benchmark, protocol):
    defaults = _benchmark_defaults(benchmark)
    defaults["handoff_protocol"] = protocol
    instance = _initialize(defaults, object(), preserve_terminal_feedback=protocol == "feedback_v1")
    expected = "curator_prompt_handoff_v1.txt" if protocol == "feedback_v1" else "curator_prompt.txt"
    assert instance.handoff_protocol == protocol
    assert instance.prompt_paths["curator_prompt"].endswith("/" + expected)
    assert instance.prompt_paths["reflector_prompt"].endswith("/reflector_prompt_handoff_v1.txt" if protocol == "feedback_v1" else "/reflector_prompt.txt")
    assert instance.reflector_adapter.handoff_protocol == protocol


class CharacterTokenizer:
    def encode(self, text, **_kwargs):
        return list(map(ord, text))

    def decode(self, values, **_kwargs):
        return "".join(map(chr, values))


def _trainer_prompt_builders(instance):
    """Bind the production prompt builders to an initialized trainer's real assets."""
    path = ROOT / "rllm/algorithms/ace/trainer_base.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_render_template"]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ACERLAgentPPOTrainer")
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"_trajectory_steps_from_chat", "_build_reflector_prompt_from_dict", "_build_curator_prompt"}]
    nodes.append(cls)
    namespace = {"SimpleNamespace": SimpleNamespace, "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    builder = namespace["ACERLAgentPPOTrainer"]()
    builder.__dict__.update(
        handoff_protocol=instance.handoff_protocol,
        reflector_adapter=instance.reflector_adapter,
        reflector_prompt_template=instance.reflector_prompt_template,
        curator_prompt_template=instance.curator_prompt_template,
        reflector_system_prompt=instance.reflector_system_prompt,
        ace_config=instance.ace_config,
        tokenizer=CharacterTokenizer(),
        component_token_limits={"reflector": {"max_prompt_length": 30720}},
        agent_execution_engine=SimpleNamespace(rollout_engine=SimpleNamespace(chat_parser=None)),
        _get_policy_task_for_trajectory=lambda data, task_idx: data["task"],
    )
    return builder


# Token-mode training rows: the chat ends with the environment's terminal reply.
SMALL_MODEL_TRAINING_ROWS = {
    "mastermind": {
        "chat_completions": [
            {"role": "user", "content": "Turn 1:\nInstructions: guess the code.\nCode length: 4\nValid digits: 0-5"},
            {"role": "assistant", "content": "Start broad.\n```0 1 2 3```"},
            {"role": "user", "content": "Turn 2:\nLast guess: 0 1 2 3\n  Exact matches: 1\n  Value matches: 2\n  Reward: 0.5\n\nGame over. The secret code was: 4 3 2 5\nSecret code was: 4 3 2 5"},
        ],
        "task": {"code_length": 4, "num_digits": 6},
        "terminal": "exact=1, value=2",
    },
    "mbpp": {
        "chat_completions": [
            {"role": "user", "content": "Write add(a, b).\n\nRemaining attempts: 1"},
            {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a - b\n```"},
            {"role": "user", "content": "Turn 1:\nTest Results:\nassert add(1, 2) == 3 FAILED\n\n1 test(s) failed. Please review the errors and try again."},
        ],
        "task": {"task_id": 1, "text": "Write add(a, b)."},
        "terminal": "ENVIRONMENT REPLY:\nTurn 1:\nTest Results:\nassert add(1, 2) == 3 FAILED",
    },
}


@pytest.mark.parametrize("benchmark", ["mastermind", "mbpp"])
@pytest.mark.parametrize("protocol", ["legacy", "feedback_v1"])
def test_small_model_role_prompts_render_every_placeholder_from_training_rows(benchmark, protocol):
    defaults = _benchmark_defaults(benchmark)
    defaults.update(handoff_protocol=protocol, max_turns=6)
    builder = _trainer_prompt_builders(_initialize(defaults, object(), preserve_terminal_feedback=protocol == "feedback_v1"))
    row = SMALL_MODEL_TRAINING_ROWS[benchmark]
    payload = {"chat_completions": row["chat_completions"], "step_rewards": [0.5], "trajectory_reward": 0.5, "all_passed": False, "termination_reason": "ENV_DONE", "task": row["task"]}
    reflected = builder._build_reflector_prompt_from_dict(payload, "[1] Existing tip.", 0)
    curated = builder._build_curator_prompt("Insight 1: keep the confirmed digits.", "[1] Existing tip.", task_instr="")
    for rendered in (reflected, curated):
        assert "{{" not in rendered and "}}" not in rendered
    assert "[1] Existing tip." in curated and "Insight 1: keep the confirmed digits." in curated
    if benchmark == "mastermind":
        # The guess table carries terminal feedback in both protocols; the code never reaches a role.
        assert row["terminal"] in reflected
        assert "Secret code" not in reflected and "4 3 2 5" not in reflected
    else:
        assert (row["terminal"] in reflected) is (protocol == "feedback_v1")
    if protocol == "feedback_v1":
        assert "Success does not update the Generator weights" in reflected
        assert "A request to remove is not evidence by itself." in curated


def test_unknown_protocol_fails_before_reading_assets():
    with pytest.raises(ValueError, match="Unsupported ace.handoff_protocol"):
        _initialize({"handoff_protocol": "invented"}, object())


@pytest.mark.parametrize("protocol,adapter_protocol", [("legacy", "feedback_v1"), ("feedback_v1", "legacy")])
def test_adapter_kwargs_cannot_select_a_conflicting_protocol(protocol, adapter_protocol):
    defaults = _benchmark_defaults("mbpp")
    defaults.update(handoff_protocol=protocol, reflector_adapter_kwargs={"handoff_protocol": adapter_protocol})
    with pytest.raises(ValueError, match="handoff_protocol"):
        _initialize(defaults, object(), preserve_terminal_feedback=protocol == "feedback_v1")


def test_feedback_protocol_requires_terminal_feedback_preservation():
    defaults = _benchmark_defaults("mbpp")
    defaults["handoff_protocol"] = "feedback_v1"
    with pytest.raises(ValueError, match="requires rllm.agent.preserve_terminal_feedback=true"):
        _initialize(defaults, object())


@pytest.mark.parametrize("variant", [False, True])
def test_feedback_protocol_selects_optional_reflector_variant_when_present(tmp_path, variant):
    defaults = _benchmark_defaults("mbpp")
    for name in ("base_system_prompt.txt", "reflector_prompt.txt", "curator_prompt_handoff_v1.txt"):
        (tmp_path / name).write_text(name)
    if variant:
        (tmp_path / "reflector_prompt_handoff_v1.txt").write_text("repaired reflector")
    defaults.update(handoff_protocol="feedback_v1", prompts_dir=str(tmp_path))
    instance = _initialize(defaults, object(), preserve_terminal_feedback=True)
    assert instance.reflector_prompt_template == ("repaired reflector" if variant else "reflector_prompt.txt")


def mutation_result(kind, protocol, operation):
    function = generation(kind)
    loop = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.For)
        and ast.unparse(node.iter) == "enumerate(curator_trajectories)"
        and "apply_operations_with_report" in ast.unparse(node)
    )
    prompt = {"task_id": "task-a", "task_idx": 0, "guidebook_uid": "guide-a", "prompt_text": "curator prompt"}
    namespace = {
        "self": SimpleNamespace(handoff_protocol=protocol, max_playbook_length=None, _decode_trajectory_response=lambda row: row["response"]),
        "Playbook": Playbook,
        "_parse_json_response": json.loads,
        "logger": logging.getLogger(__name__),
        "curator_trajectories": [{"response": json.dumps({"operations": [operation]})}],
        "curator_prompts": [prompt],
        "curator_to_guidebook": [],
        "curator_records": [],
        "candidate_playbooks": [],
        "playbook_before_text_by_task": {"task-a": "[1] original"},
        "playbook_text": "[1] original",
    }
    execute([loop], namespace)
    assert len(namespace["curator_records"]) == 1
    return namespace["curator_records"][0], namespace["candidate_playbooks"]


@pytest.mark.parametrize("kind", ["base", "standard"])
@pytest.mark.parametrize("protocol", ["legacy", "feedback_v1"])
@pytest.mark.parametrize("operation,changed", [
    ({"type": "UPDATE", "tip_id": "1", "content": "corrected"}, True),
    ({"type": "UPDATE", "bullet_id": "1", "content": "corrected"}, True),
    ({"type": "UPDATE", "bullet_id": "1", "content": "original"}, False),
    ({"type": "REMOVE", "bullet_id": "missing"}, False),
])
def test_training_candidate_mutations_are_reported_and_noops_rejected_only_with_opt_in(kind, protocol, operation, changed):
    record, candidates = mutation_result(kind, protocol, operation)
    if protocol == "feedback_v1":
        assert record["mutation"]["changed"] is changed
        assert record["accepted"] is changed
        assert len(candidates) == int(changed)
        if not changed:
            assert record["reject_reason"] == "unchanged_playbook"
    else:
        assert "mutation" not in record
        assert record["accepted"] is True
        assert len(candidates) == 1
    if operation.get("tip_id") and protocol == "legacy":
        assert record["candidate_playbook"] == "[1] original"
    elif changed:
        assert record["candidate_playbook"] == "[1] corrected"


@pytest.mark.parametrize("saved_protocol", [None, "legacy", "feedback_v1"])
@pytest.mark.parametrize("current_protocol", ["legacy", "feedback_v1"])
def test_controller_protocol_must_match_before_state_is_restored(tmp_path, saved_protocol, current_protocol):
    state = helper()
    original = trainer()
    original.handoff_protocol = saved_protocol or "legacy"
    state.save_ace_state(original, tmp_path)
    path = tmp_path / "ace_state.json"
    document = json.loads(path.read_text())
    assert document["handoff_protocol"] == (saved_protocol or "legacy")
    if saved_protocol is None:
        document.pop("handoff_protocol")
        path.write_text(json.dumps(document))
    resumed = trainer()
    resumed.handoff_protocol = current_protocol
    resumed.local_train_task_seen_counts = {"untouched": 9}
    if (saved_protocol or "legacy") == current_protocol:
        assert state.load_ace_state(resumed, tmp_path)
        assert resumed.local_train_task_seen_counts == original.local_train_task_seen_counts
    else:
        with pytest.raises(ValueError, match="incompatible handoff_protocol"):
            state.load_ace_state(resumed, tmp_path)
        assert resumed.local_train_task_seen_counts == {"untouched": 9}


def test_feedback_protocol_cannot_cold_resume_without_controller_state(tmp_path):
    resumed = trainer()
    resumed.handoff_protocol = "feedback_v1"
    resumed.local_train_task_seen_counts = {"untouched": 9}
    with pytest.raises(ValueError, match="missing ACE state.*handoff"):
        helper().load_ace_state(resumed, tmp_path)
    assert resumed.local_train_task_seen_counts == {"untouched": 9}
