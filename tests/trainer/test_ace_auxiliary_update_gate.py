"""CPU regressions for the production ACE gradient-admission blocks.

Selection, reward assignment, row admission, UID assembly and token reporting
are executable trainer statements. Only distributed transformation and parent
batch expansion are adapters. No model or rollout engine is started.
"""

import ast
import copy
from types import SimpleNamespace

import pytest
from test_standard_rejection_update import assignment, auxiliary_rewards, execute, generation, local_selection, run_selection, trainer_namespace, tree

from rllm.utils.training_reward import REWARD_KEYS, raw_reward_metrics


def capture_actor_rewards(function, ns, expected_count):
    capture = next(node for node in ast.walk(function) if assignment(node, "actor_reward_metrics"))
    ns["raw_reward_metrics"] = raw_reward_metrics
    ns["original_envs"] = [None] * expected_count
    ns["self"].agent_execution_engine = SimpleNamespace(envs=ns["original_envs"])
    execute([capture], ns)
    return capture


def token_metrics():
    return {"total_prompt_tokens": 7, "total_completion_tokens": 3, "total_gradient_update_tokens": 3}


def combine_block(kind):
    function = generation(kind)
    for node in ast.walk(function):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        starts = [i for i, statement in enumerate(body) if isinstance(statement, ast.For) and ast.unparse(statement.iter) == "policy_trajectories_for_training"]
        if not starts:
            continue
        start = starts[0]
        end = next(i for i in range(start, len(body)) if isinstance(body[i], ast.Assign) and ast.unparse(body[i].targets[0]) == "metrics['ace/policy_rollouts']")
        return body[start:end]
    raise AssertionError("No production admission/metadata/metrics block")


def prepare_rows(ns, *, invalid_curator=False):
    candidates = ns["candidate_playbooks"]
    ns["reflector_trajectories"] = [{"row": f"reflector-{i}", "metrics": token_metrics()} for i in range(len(candidates))]
    ns["curator_trajectories"] = [{"row": f"curator-{i}", "metrics": token_metrics()} for i in range(len(candidates))]
    ns["all_reflections"] = [
        {"task_id": record["task_id"], "guidebook_uid": candidate["guidebook_uid"], "reflector_uid": f"ref-{record['task_id']}"}
        for candidate, record in zip(candidates, ns["curator_records"], strict=True)
    ]
    ns["curator_to_guidebook"] = [candidate["guidebook_uid"] for candidate in candidates]
    if invalid_curator:
        ns["curator_trajectories"].append({"row": "invalid-curator", "metrics": token_metrics()})
        ns["curator_records"].append(dict(ns["curator_records"][-1]))
        ns["curator_to_guidebook"].append(ns["curator_to_guidebook"][-1])
    for row in ns["policy_trajectories"]:
        row["metrics"] = token_metrics()
    # Selection creates copies before the metric fixtures are attached.
    for rows in [ns.get("policy_trajectories_for_training", []), *ns.get("grouped_policy_for_training_by_task", {}).values()]:
        for row in rows:
            row["metrics"] = token_metrics()


def combine(kind, ns, *, gate="strict_improvement", active_component="all", validation=False):
    trainer = ns["self"]
    capture_actor_rewards(generation(kind), ns, len(ns["policy_trajectories"]))
    trainer.auxiliary_update_gate = gate
    trainer.training_mode = "standard"
    trainer._advance_component_schedule = lambda: None
    trainer._current_batch = SimpleNamespace(non_tensor_batch={})
    trainer._last_candidate_eval_token_metrics = [token_metrics() for _ in ns["candidate_playbooks"]]
    trainer._transform_agent_trajectories = lambda rows: (copy.deepcopy(rows), {})
    trainer._append_policy_and_component_rows_to_batch = lambda batch, **kwargs: setattr(trainer, "expanded_metadata", kwargs)
    ns.update(
        meta_info={"validate": validation},
        is_validation=validation,
        active_component=active_component,
        schedule_state=None,
        parent_batch=trainer._current_batch,
        policy_step_reward_tasks=["task-a"] * len(ns["policy_trajectories"]),
        candidate_eval_token_metrics=trainer._last_candidate_eval_token_metrics,
    )
    if kind == "standard":
        ns["policy_trajectories_for_training"] = [row for rows in ns["grouped_policy_for_training_by_task"].values() for row in rows]
    execute(auxiliary_rewards(kind), ns)
    execute(combine_block(kind), ns)
    return ns


@pytest.mark.parametrize("kind", ["base", "standard"])
@pytest.mark.parametrize("reward,expected_aux", [(0.0, False), (0.5, False), (1.0, True)])
def test_strict_gate_keeps_actor_and_does_not_change_selected_playbook(kind, reward, expected_aux):
    ns = run_selection(kind, [reward], [0.5, 1.0], mode="best", policy_source="seed")
    prepare_rows(ns)
    before = ns["playbook_text"]
    combine(kind, ns)
    assert [row["trajectory_reward"] for row in ns["all_trajectories"][:2]] == [0.5, 1.0]
    assert len(ns["all_trajectories"]) == (4 if expected_aux else 2)
    assert ns["playbook_text"] == before == "candidate-0"
    assert ns["metrics"]["ace/auxiliary_tasks_admitted"] == int(expected_aux)
    assert ns["metrics"]["ace/auxiliary_tasks_dropped"] == int(not expected_aux)


@pytest.mark.parametrize("kind", ["base", "standard"])
def test_source_seed_baseline_is_used_for_rejection_update(kind):
    ns = run_selection(kind, [0.4], [0.2, 1.0], policy_source="seed")
    prepare_rows(ns)
    combine(kind, ns)
    # Candidate improves its own seed despite falling below task mean 0.6.
    assert len(ns["all_trajectories"]) == 4
    assert ns["metrics"]["ace/auxiliary_tasks_admitted"] == 1


@pytest.mark.parametrize("kind", ["base", "standard"])
def test_exact_tie_to_source_drops_auxiliaries_even_above_task_mean(kind):
    ns = run_selection(kind, [1.0], [1.0, 0.0], policy_source="seed")
    prepare_rows(ns)
    combine(kind, ns)
    assert len(ns["all_trajectories"]) == 2
    assert ns["playbook_text"] == "candidate-0"


@pytest.mark.parametrize("kind", ["base", "standard"])
def test_admitted_groups_keep_failures_invalid_candidates_and_original_uids(kind):
    ns = run_selection(kind, [0.0, 1.0], [0.0, 0.0], policy_source="seed")
    # Two candidates share the curator GRPO prompt; the failed parse is the
    # third row in that group and must remain with zero reward.
    ns["candidate_playbooks"][1]["guidebook_uid"] = ns["candidate_playbooks"][0]["guidebook_uid"]
    prepare_rows(ns, invalid_curator=True)
    combine(kind, ns)
    assert len(ns["training_reflector_trajectories"]) == 2
    assert [row["trajectory_reward"] for row in ns["training_curator_trajectories"]] == [0.0, 1.0, 0.0]
    assert ns["self"].expanded_metadata["component_extra_uids"] == ["ref-task-a", "ref-task-a", "cur_gb-0", "cur_gb-0", "cur_gb-0"]
    assert ns["metrics"]["ace/reflector_training_groups"] == 1
    assert ns["metrics"]["ace/curator_training_groups"] == 1


def test_batched_tasks_filter_rows_and_uids_together():
    ns = run_selection("standard", [0.0, 1.0], [0.0, 0.0], policy_source="seed")
    # Task A ties and is removed; B improves and keeps both rows in its group.
    for candidate in ns["candidate_playbooks"]:
        candidate["task_id"] = "task-b"
    for record in ns["curator_records"]:
        record["task_id"] = "task-b"
    ns["candidate_playbooks"] = [dict(candidate, task_id="task-a", curator_idx=i, guidebook_uid="gb-a") for i, candidate in enumerate(ns["candidate_playbooks"])] + [
        dict(candidate, curator_idx=i + 2, guidebook_uid="gb-b") for i, candidate in enumerate(ns["candidate_playbooks"])
    ]
    ns["curator_records"] = [dict(record, task_id="task-a") for record in ns["curator_records"]] + ns["curator_records"]
    ns["candidate_rewards"] = [1.0, 1.0, 0.0, 1.0]
    ns["task_ids"] = ["task-a", "task-b"]
    ns["grouped_policy_trajectories"] = {
        "task-a": [{"row": "a0", "trajectory_reward": 1.0}, {"row": "a1", "trajectory_reward": 1.0}],
        "task-b": [{"row": "b0", "trajectory_reward": 0.0}, {"row": "b1", "trajectory_reward": 0.0}],
    }
    ns["current_rewards_by_task"] = {"task-a": 1.0, "task-b": 0.0}
    ns["playbook_before_text_by_task"] = {"task-a": "a-before", "task-b": "b-before"}
    ns["policy_trajectories"] = [row for group in ns["grouped_policy_trajectories"].values() for row in group]
    # The maximum ties are deliberate; deterministic tie-breaking is irrelevant
    # to admission here because their source rewards agree.
    ns["random"].choice = lambda values: values[0]
    execute([local_selection()], ns)
    prepare_rows(ns)
    combine("standard", ns)
    assert [row["row"] for row in ns["all_trajectories"]] == ["a0", "a1", "b0", "b1", "reflector-2", "reflector-3", "curator-2", "curator-3"]
    assert ns["self"].expanded_metadata["component_extra_uids"] == ["ref-task-b", "ref-task-b", "cur_gb-b", "cur_gb-b"]
    assert ns["metrics"]["ace/auxiliary_tasks_admitted"] == 1
    assert ns["metrics"]["ace/auxiliary_tasks_dropped"] == 1


@pytest.mark.parametrize("kind", ["base", "standard"])
def test_actor_is_eligible_during_auxiliary_component_schedule(kind):
    ns = run_selection(kind, [0.0], [0.0, 1.0], policy_source="seed")
    prepare_rows(ns)
    combine(kind, ns, active_component="reflector")
    assert [row["trajectory_reward"] for row in ns["policy_trajectories_for_training"]] == [0.0, 1.0]


@pytest.mark.parametrize("kind", ["base", "standard"])
def test_none_preserves_legacy_admission_and_role_masking(kind):
    ns = run_selection(kind, [0.0], [0.0, 1.0], policy_source="seed")
    prepare_rows(ns)
    combine(kind, ns, gate="none", active_component="reflector")
    assert len(ns["all_trajectories"]) == 4
    assert [row["trajectory_reward"] for row in ns["policy_trajectories_for_training"]] == [0.0, 0.0]
    assert {key: ns["metrics"][key] for key in REWARD_KEYS} == dict(zip(REWARD_KEYS, [1.0, 2, 0], strict=True))


@pytest.mark.parametrize("kind", ["base", "standard"])
def test_dropped_generation_still_counts_and_candidate_tokens_are_not_duplicated(kind):
    ns = run_selection(kind, [0.0], [0.0, 1.0], policy_source="seed")
    prepare_rows(ns)
    combine(kind, ns)
    metrics = ns["metrics"]
    assert metrics["rollout/total_prompt_tokens"] == 5 * 7
    assert metrics["rollout/total_completion_tokens"] == 5 * 3
    assert metrics["rollout/total_trajectories"] == 5
    assert metrics["rollout/total_gradient_update_tokens"] == 2 * 3


@pytest.mark.parametrize("kind", ["base", "standard"])
def test_validation_rows_rewards_and_metrics_ignore_gate(kind):
    outputs = []
    for gate in ["none", "strict_improvement"]:
        ns = run_selection(kind, [0.0], [0.0, 1.0], policy_source="seed")
        prepare_rows(ns)
        combine(kind, ns, gate=gate, active_component="reflector", validation=True)
        outputs.append((ns["all_trajectories"], ns["metrics"], ns["self"].expanded_metadata))
    assert outputs[0] == outputs[1]
    assert not set(REWARD_KEYS).intersection(outputs[0][1])


@pytest.mark.parametrize("mode", ["delayed_joint", "joint_ppo"])
def test_delayed_paths_do_not_apply_gate_and_initialize_effective_none(mode):
    trainer = trainer_namespace("best_rejection_update")["self"]
    trainer.auxiliary_update_gate = "strict_improvement"
    trainer.training_mode = mode
    assert not trainer._uses_auxiliary_update_gate()
    for kind in ["delayed_joint", "delayed_joint_base"]:
        assert "_prepare_auxiliary_training_rows" not in ast.unparse(generation(kind))
    init = next(node for node in ast.walk(tree("delayed_joint_base")) if isinstance(node, ast.FunctionDef) and node.name == "_init_ace_components")
    assert "self.auxiliary_update_gate = 'none'" in ast.unparse(init)


def test_missing_selected_source_cannot_admit_auxiliary_rows():
    trainer = trainer_namespace("best_rejection_update")["self"]
    assert trainer._selected_playbook_improvement(0, 1.0, 0.0, [{"curator_idx": 4}], [], {}) == (False, None)
    assert trainer._selected_playbook_improvement(-1, 1.0, 0.0, [], [], {}) == (False, None)


def test_task_aggregation_sums_admission_counts():
    path = tree("standard")
    method = next(node for node in ast.walk(path) if isinstance(node, ast.FunctionDef) and node.name == "_aggregate_task_metrics")
    ns = trainer_namespace("best")
    execute([method], ns)
    result = ns["_aggregate_task_metrics"](ns["self"], [{"ace/reflector_training_rows": 4, "ace/auxiliary_tasks_admitted": 1}, {"ace/reflector_training_rows": 0, "ace/auxiliary_tasks_admitted": 0}])
    assert result == {"ace/reflector_training_rows": 4.0, "ace/auxiliary_tasks_admitted": 1.0}


@pytest.mark.parametrize("kind", ["base", "standard", "delayed_joint", "delayed_joint_base"])
def test_raw_actor_rewards_survive_candidate_rejection_and_auxiliary_credit(kind):
    # Each candidate is worse than its source actor, so neither is installed.
    ns = run_selection(kind, [0.2, 0.0], [0.75, 0.25], mode="best_rejection_update")
    capture_actor_rewards(generation(kind), ns, 2)
    assert ns["reject_candidate"] is True
    from test_standard_rejection_update import run_auxiliary

    assert run_auxiliary(kind, ns) == ([0.2, 0.0], [0.2, 0.0])
    assert ns["actor_reward_metrics"] == dict(zip(REWARD_KEYS, [1.0, 2, 0], strict=True))


@pytest.mark.parametrize(("kind", "method"), [
    ("base", "generate_agent_trajectory"),
    ("standard", "_generate_multi_task_batched_full_ace"),
    ("delayed_joint_base", "generate_agent_trajectory"),
    ("delayed_joint", "_generate_multi_task_batched_delayed_joint"),
    ("base", "_generate_policy_only_trajectory"),
    ("standard", "_maybe_generate_batched_policy_only_training_step"),
])
def test_all_ace_collectors_snapshot_native_actor_rewards_before_transform(kind, method):
    function = next(node for node in ast.walk(tree(kind)) if isinstance(node, ast.FunctionDef) and node.name == method)
    raw_rows = [
        {"trajectory_reward": -0.25},
        {"trajectory_reward": 0.0},
        {"trajectory_reward": 0.75},
        {"trajectory_reward": float("nan")},
        {},
        {"trajectory_reward": 0.0, "termination_reason": "ENV_ERROR"},
    ]
    ns = {"self": SimpleNamespace(), "policy_trajectories": raw_rows, "seed_policy_trajectories": raw_rows}
    capture = capture_actor_rewards(function, ns, 7)  # One missing collection result.
    transform = next(node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_transform_agent_trajectories")
    assert capture.lineno < transform.lineno
    # Downstream rewriting and auxiliary inclusion must not alter the snapshot.
    for row in raw_rows:
        row["trajectory_reward"] = 1.0
    ns["all_trajectories"] = raw_rows + [{"trajectory_reward": 100.0}] * 4
    ns["metrics"] = {}
    publication = next(node for node in ast.walk(function) if isinstance(node, ast.Expr) and ast.unparse(node) == "metrics.update(actor_reward_metrics)")
    execute([publication], ns)
    assert ns["metrics"] == dict(zip(REWARD_KEYS, [0.5, 3, 4], strict=True))


def test_task_aggregation_sums_raw_actor_rewards_and_counts_for_weighted_mean():
    method = next(node for node in ast.walk(tree("standard")) if isinstance(node, ast.FunctionDef) and node.name == "_aggregate_task_metrics")
    ns = trainer_namespace("best")
    execute([method], ns)
    task_metrics = [dict(zip(REWARD_KEYS, values, strict=True)) for values in [(0.25, 1, 0), (1.25, 3, 2)]]
    result = ns["_aggregate_task_metrics"](ns["self"], task_metrics)
    assert result == dict(zip(REWARD_KEYS, [1.5, 4, 2], strict=True))
    assert type(result["train/actor_reward_count"]) is int
    assert type(result["train/actor_reward_invalid_count"]) is int
