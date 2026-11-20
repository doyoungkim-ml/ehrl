"""CPU-only regressions executing the trainers' real selection/credit AST blocks.

The full trainer imports VeRL/Ray/GPU workers. Extract executable statements, not
copies of their logic, so these tests also run with ``python -m unittest`` on a
checkout without the training stack. Only timers, snapshots and playbooks are
stand-ins; policy-row selection and seed threshold helpers are real methods.
Standard candidate metadata also comes from the real producer AST, so tests
cannot accidentally supply source indices that production only keeps in records.
"""

import ast
import copy
import logging
import random
import unittest
from contextlib import nullcontext
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAINERS = ROOT / "rllm/algorithms/ace"
KINDS = ("standard", "base", "delayed_joint", "delayed_joint_base")
LOCAL_KINDS = ("standard", "delayed_joint")


@cache
def tree(kind):
    return ast.parse((TRAINERS / ("trainer.py" if kind == "standard" else f"trainer_{kind}.py")).read_text())


def execute(nodes, namespace):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *copy.deepcopy(nodes)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<real-trainer-block>", "exec"), namespace)


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def generation(kind):
    name = {"standard": "_generate_multi_task_batched_full_ace", "delayed_joint": "_generate_multi_task_batched_delayed_joint"}.get(kind, "generate_agent_trajectory")
    return next(n for n in ast.walk(tree(kind)) if isinstance(n, ast.FunctionDef) and n.name == name)


def local_selection(kind="standard"):
    return next(n for n in ast.walk(generation(kind)) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "task_id" and any(assignment(c, "selected_idx") for c in n.body))


def base_selection(kind="base"):
    body = generation(kind).body
    start = next(i for i, n in enumerate(body) if assignment(n, "policy_trajectories_for_training"))
    end = next(i for i, node in enumerate(body[start:], start) if assignment(node, "snapshot_path") or (isinstance(node, ast.If) and ast.unparse(node.test) == "is_validation"))
    return body[start:end]


class Playbook:
    def __init__(self, text):
        self.text = text

    def to_string(self):
        return self.text

    def get_stats(self):
        return {"total_bullets": 1}


def trainer_namespace(mode, policy_source="selected_candidate", *, kind="base", random_source=None):
    forbidden_random = Mock(side_effect=AssertionError("selection must not fall through to random"))
    namespace = {
        "copy": copy,
        "np": np,
        "logger": logging.getLogger(__name__),
        "Playbook": Playbook,
        "random": random_source if random_source is not None else SimpleNamespace(choice=forbidden_random, randint=forbidden_random),
        "marked_timer": lambda *args: nullcontext(),
    }
    names = {
        "_get_candidate_seed_reward",
        "_get_best_rejection_update_candidate_indices",
        "_select_best_rejection_update_candidate_index",
        "_select_policy_trajectories_for_training",
        "_assign_rollout_aligned_aux_rewards",
        "_uses_auxiliary_update_gate",
        "_selected_playbook_improvement",
        "_prepare_auxiliary_training_rows",
        "_auxiliary_training_metrics",
        "_compute_rollout_token_totals",
    }
    class_names = {}
    for source_kind in ("base", "standard", "delayed_joint_base", "delayed_joint"):
        cls = copy.deepcopy(next(n for n in tree(source_kind).body if isinstance(n, ast.ClassDef)))
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names] or [ast.Pass()]
        if source_kind == "base":
            cls.bases = []
        execute([cls], namespace)
        class_names[source_kind] = cls.name
    trainer = namespace[class_names[kind]]()
    trainer.selection_mode = mode
    trainer.policy_train_source = policy_source
    trainer.playbook = Playbook("baseline")
    trainer._save_local_snapshot = lambda **kwargs: None
    namespace["self"] = trainer
    return namespace


def standard_candidate_metadata(kind, candidates):
    """Execute the real curator-record and candidate dictionary construction."""
    function = generation(kind)
    record_node = next(n for n in ast.walk(function) if isinstance(n, ast.Assign) and assignment(n, "record") and isinstance(n.value, ast.Dict) and "curator_prompt_idx" in ast.unparse(n))
    append_node = next(n for n in ast.walk(function) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "candidate_playbooks.append")
    ns: dict[str, Any] = {"candidate_playbooks": []}
    records = []
    for i, candidate in enumerate(candidates):
        prompt = {key: candidate[key] for key in ("task_id", "task_idx", "traj_idx", "guidebook_uid")}
        prompt["prompt_text"] = "test prompt"
        ns.update(i=i, cp=prompt, prompt=prompt, candidate=candidate["playbook"], task_id=candidate["task_id"], ops={}, response_text="{}")
        execute([record_node, append_node], ns)
        records.append(ns["record"])
    return ns["candidate_playbooks"], records


def run_selection(kind, rewards, seeds, *, mode="best_rejection_update", source_indices=None, policy_source="selected_candidate", random_source=None, is_validation=True):
    ns = trainer_namespace(mode, policy_source, kind=kind, random_source=random_source)
    rows = [{"trajectory_reward": r, "row": f"seed-{i}"} for i, r in enumerate(seeds)]
    if source_indices is None:
        source_indices = list(range(len(rewards)))
    candidates = [
        {"task_id": "task-a", "task_idx": 0, "traj_idx": source, "playbook": Playbook(f"candidate-{i}"), "guidebook_uid": f"gb-{i}", "reflector_idx": i, "curator_idx": i}
        for i, source in enumerate(source_indices)
    ]
    curator_records = [dict(candidate) for candidate in candidates]
    if kind in ("standard", "base"):
        candidates, curator_records = standard_candidate_metadata(kind, candidates)
    eval_rows = {i: [{"row": f"eval-{i}-{j}", "trajectory_reward": reward} for j in range(len(rows))] for i, reward in enumerate(rewards)}
    ns.update(
        candidate_playbooks=candidates,
        curator_records=curator_records,
        candidate_rewards=rewards,
        timing_raw={},
        policy_trajectories=rows,
        seed_policy_trajectories=rows,
        candidate_policy_trajectories=list(eval_rows.values()),
        task_groups=[rows],
        current_avg_reward=float(np.mean(seeds)),
    )
    ns["self"]._last_candidate_eval_trajectories = eval_rows
    if kind in LOCAL_KINDS:
        ns.update(
            task_ids=["task-a"],
            current_rewards_by_task={"task-a": ns["current_avg_reward"]},
            playbook_before_text_by_task={"task-a": "baseline"},
            grouped_policy_trajectories={"task-a": rows},
            grouped_seed_policy={"task-a": rows},
            candidate_eval_trajectories=eval_rows,
            is_validation=is_validation,
            store={},
        )
        for name in (
            "selected_idx_by_task",
            "selected_reward_by_task",
            "best_candidate_reward_by_task",
            "reject_candidate_by_task",
            "playbook_after_text_by_task",
            "snapshot_paths",
            "next_validation_trajectories",
            "grouped_policy_for_training_by_task",
            "policy_train_source_by_task",
        ):
            ns[name] = {}
        for name in ("task_playbook_sizes", "task_playbook_bullets", "task_metric_pairs"):
            ns[name] = []
        execute([local_selection(kind)], ns)
        ns["playbook_text"] = ns["playbook_after_text_by_task"]["task-a"]
        ns["policy_train_source"] = ns["policy_train_source_by_task"]["task-a"]
    else:
        execute(base_selection(kind), ns)
        ns["playbook_text"] = ns["self"].playbook.to_string()
        if kind == "delayed_joint_base":
            ns["selected_idx"] = ns["selected_candidate_idx"]
    return ns


def auxiliary_rewards(kind):
    function = generation(kind)
    if kind == "base":
        return [next(n for n in function.body if isinstance(n, ast.If) and "reject_candidate" in ast.unparse(n.test) and any(assignment(c, "guidebook_rewards") for c in n.orelse))]
    if kind == "delayed_joint_base":
        return [next(n for n in function.body if isinstance(n, ast.If) and "reject_candidate" in ast.unparse(n.test) and "_assign_rollout_aligned_aux_rewards" in ast.unparse(n))]
    for node in ast.walk(function):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for start, statement in enumerate(body):
            if assignment(statement, "guidebook_rewards") or assignment(statement, "reflector_idx_to_reward"):
                end = next(i for i, s in enumerate(body[start:], start) if isinstance(s, ast.For) and ast.unparse(s.iter) == "policy_trajectories_for_training")
                return body[start:end]
    raise AssertionError("Auxiliary reward block not found")


def run_auxiliary(kind, ns):
    count = len(ns["candidate_playbooks"])
    ns.update(
        reflector_trajectories=[{} for _ in range(count)],
        curator_trajectories=[{} for _ in range(count)],
        all_reflections=[{"task_id": "task-a", "guidebook_uid": f"gb-{i}"} for i in range(count)],
        reflector_records=[{"task_id": "task-a"} for _ in range(count)],
        curator_records=[{"task_id": "task-a"} for _ in range(count)],
    )
    execute(auxiliary_rewards(kind), ns)
    return ([t["trajectory_reward"] for t in ns["reflector_trajectories"]], [t["trajectory_reward"] for t in ns["curator_trajectories"]])


class RejectionUpdateSelectionTests(unittest.TestCase):
    def test_standard_candidates_resolve_seed_from_their_curator_record(self):
        for kind in ("standard", "base"):
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.8, 0.4], [1.0, 0.2])
                self.assertEqual(ns["selected_idx"], 1)
                self.assertEqual(ns["playbook_text"], "candidate-1")

    def test_skipped_curator_does_not_shift_candidate_seed_alignment(self):
        for kind in ("standard", "base"):
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.8, 0.4], [1.0, 0.2])
                # A curator response with no valid operation produces a record,
                # but no candidate. Candidate indices need not equal curator indices.
                ns["curator_records"].insert(0, {"task_id": "task-a", "task_idx": 0, "traj_idx": 0})
                for candidate in ns["candidate_playbooks"]:
                    candidate["curator_idx"] += 1
                execute([local_selection(kind)] if kind in LOCAL_KINDS else base_selection(kind), ns)
                self.assertEqual(ns["selected_idx"], 1)
                self.assertFalse(ns["reject_candidate"])

    def test_tied_maxima_randomness_is_eligible_only_and_replayable(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                rng = random.Random(37)
                before = rng.getstate()

                def draw(kind=kind, rng=rng):
                    # Candidate 0 has a higher reward but fails its seed gate;
                    # candidate 2 survives but is not maximal.
                    return run_selection(kind, [0.9, 0.8, 0.6, 0.8], [1.0, 0.1, 0.1, 0.1], random_source=rng)["selected_idx"]

                selected = [draw() for _ in range(8)]
                self.assertEqual(set(selected), {1, 3})
                self.assertNotEqual(rng.getstate(), before)
                rng.setstate(before)
                self.assertEqual([draw() for _ in range(8)], selected)

    def test_rejected_candidates_keep_all_auxiliary_rewards(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.8, 0.4], [1.0, 0.9])
                self.assertTrue(ns["reject_candidate"])
                self.assertEqual(ns["selected_idx"], -1)
                self.assertEqual(ns["playbook_text"], "baseline")
                self.assertEqual(ns["policy_trajectories_for_training"], ns["policy_trajectories"])
                self.assertEqual(run_auxiliary(kind, ns), ([0.8, 0.4], [0.8, 0.4]))

    def test_chooses_max_eligible_not_random_or_task_mean(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                # Highest score is worse than its own seed; winner is below the
                # task mean but above its own seed. Selection is per source.
                ns = run_selection(kind, [0.8, 0.4, 0.3], [1.0, 0.2, 0.2])
                self.assertEqual(ns["selected_idx"], 1)
                self.assertEqual(ns["selected_reward"], 0.4)
                self.assertEqual(ns["playbook_text"], "candidate-1")
                self.assertEqual(ns["policy_train_source"], "selected_candidate")
                self.assertEqual(ns["policy_trajectories_for_training"][0]["row"], "eval-1-0")

    def test_seed_reward_equality_is_eligible(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.2, 0.8], [0.2, 0.9])
                self.assertFalse(ns["reject_candidate"])
                self.assertEqual(ns["selected_idx"], 0)
                self.assertEqual(ns["selected_reward"], 0.2)

    def test_invalid_seed_sources_keep_baseline(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [1.0, 1.0], [0.2], source_indices=[None, 4])
                self.assertTrue(ns["reject_candidate"])
                self.assertEqual(ns["selected_idx"], -1)
                self.assertEqual(ns["selected_reward"], 0.2)
                self.assertEqual(ns["playbook_text"], "baseline")
                self.assertEqual(ns["policy_train_source"], "seed")

    def test_no_candidates_or_no_evaluations_keep_baseline_without_rng(self):
        for kind in KINDS:
            for source_indices in ([], [0]):
                with self.subTest(kind=kind, sources=source_indices):
                    ns = run_selection(kind, [], [0.2], source_indices=source_indices)
                    self.assertFalse(ns["reject_candidate"])
                    self.assertEqual(ns["selected_idx"], -1)
                    self.assertEqual(ns["selected_reward"], 0.2)
                    self.assertEqual(ns["playbook_text"], "baseline")
                    self.assertEqual(ns["policy_train_source"], "seed")
                    self.assertEqual(ns["policy_trajectories_for_training"], ns["policy_trajectories"])
                    zeros = [0.0] * len(source_indices)
                    self.assertEqual(run_auxiliary(kind, ns), (zeros, zeros))

    def test_unevaluated_candidates_cannot_win(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.4], [0.2, 0.0], source_indices=[0, 1])
                self.assertEqual(ns["selected_idx"], 0)
                self.assertEqual(run_auxiliary(kind, ns), ([0.4, 0.0], [0.4, 0.0]))

    def test_local_task_selection_cannot_take_another_tasks_candidate(self):
        for kind in LOCAL_KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.9, 0.8], [0.0, 0.0])
                ns["candidate_playbooks"][0]["task_id"] = "task-b"
                execute([local_selection(kind)], ns)
                self.assertEqual(ns["selected_idx_by_task"], {"task-a": 1})
                self.assertEqual(ns["playbook_after_text_by_task"], {"task-a": "candidate-1"})

    def test_seed_lookup_respects_task_and_requested_subset(self):
        for kind in KINDS:
            for local in (False, True):
                with self.subTest(kind=kind, local=local):
                    trainer = trainer_namespace("best_rejection_update", kind=kind)["self"]
                    candidates = [
                        {"task_id": "task-a", "task_idx": 0, "traj_idx": 0},
                        {"task_id": "task-b", "task_idx": 1, "traj_idx": 0},
                        {"task_id": "task-b", "task_idx": 1, "traj_idx": 1},
                        {"task_id": "missing", "task_idx": 5, "traj_idx": 0},
                        {"task_id": "task-b", "task_idx": 1, "traj_idx": 0},
                    ]
                    groups = [[{"trajectory_reward": 0.9}], [{"trajectory_reward": 0.2}]]
                    if local:
                        groups = dict(zip(("task-a", "task-b"), groups, strict=False))
                    survivors = trainer._get_best_rejection_update_candidate_indices(candidates, [1.0, 0.2, 1.0, 1.0], groups, [1, 2, 3, 4])
                    self.assertEqual(survivors, [1])

    def test_accepted_task_preserves_all_candidate_auxiliary_credits(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.8, 0.4], [1.0, 0.2])
                self.assertEqual(ns["selected_idx"], 1)
                self.assertEqual(run_auxiliary(kind, ns), ([0.8, 0.4], [0.8, 0.4]))

    def test_rejected_standard_guidebook_keeps_mean_of_all_curator_rewards(self):
        for kind in ("standard", "base"):
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.8, 0.4], [1.0, 0.9])
                ns["candidate_playbooks"][1]["guidebook_uid"] = "gb-0"
                reflector_rewards, curator_rewards = run_auxiliary(kind, ns)
                self.assertEqual(reflector_rewards, [float(np.mean([0.8, 0.4])), 0.0])
                self.assertEqual(curator_rewards, [0.8, 0.4])

    def test_policy_source_modes_are_preserved_on_acceptance(self):
        expected = {
            "seed": ["seed-0", "seed-1"],
            "selected_candidate": ["eval-1-0", "eval-1-1"],
            "both": ["seed-0", "seed-1", "eval-1-0", "eval-1-1"],
        }
        for kind in KINDS:
            for source, rows in expected.items():
                with self.subTest(kind=kind, source=source):
                    ns = run_selection(kind, [0.8, 0.4], [1.0, 0.2], policy_source=source)
                    self.assertEqual(ns["policy_train_source"], source)
                    self.assertEqual([row["row"] for row in ns["policy_trajectories_for_training"]], rows)

    def test_rejection_keeps_seed_policy_for_every_policy_source(self):
        for kind in KINDS:
            for source in ("seed", "selected_candidate", "both"):
                with self.subTest(kind=kind, source=source):
                    ns = run_selection(kind, [0.8, 0.4], [1.0, 0.9], policy_source=source)
                    self.assertEqual(ns["policy_train_source"], "seed")
                    self.assertEqual(ns["policy_trajectories_for_training"], ns["policy_trajectories"])
                    self.assertIsNot(ns["policy_trajectories_for_training"][0], ns["policy_trajectories"][0])

    def test_seed_and_candidates_trains_every_candidate_rollout_with_seed_rows(self):
        candidate_rows = ["eval-0-0", "eval-0-1", "eval-1-0", "eval-1-1"]
        for seeds in ([1.0, 0.2], [1.0, 0.9]):  # accepted and rejected candidates
            with self.subTest(seeds=seeds):
                ns = run_selection("standard", [0.8, 0.4], seeds, policy_source="seed_and_candidates", is_validation=False)
                self.assertEqual(ns["policy_train_source"], "seed_and_candidates")
                self.assertEqual([row["row"] for row in ns["policy_trajectories_for_training"]], ["seed-0", "seed-1", *candidate_rows])

    def test_seed_and_candidates_validation_keeps_seed_rows(self):
        ns = run_selection("standard", [0.8, 0.4], [1.0, 0.2], policy_source="seed_and_candidates")
        self.assertEqual(ns["policy_train_source"], "seed")
        self.assertEqual([row["row"] for row in ns["policy_trajectories_for_training"]], ["seed-0", "seed-1"])
        self.assertEqual(ns["playbook_text"], "candidate-1")

    def test_best_still_accepts_maximum_below_baseline(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.4, 0.8], [1.0, 0.9], mode="best")
                self.assertEqual(ns["selected_idx"], 1)
                self.assertFalse(ns["reject_candidate"])
                self.assertEqual(run_auxiliary(kind, ns), ([0.4, 0.8], [0.4, 0.8]))

    def test_best_rejection_still_compares_task_mean_not_candidate_seed(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.8, 0.4], [1.0, 0.2], mode="best_rejection")
                self.assertEqual(ns["selected_idx"], 0)
                self.assertFalse(ns["reject_candidate"])

    def test_best_rejection_still_accepts_task_mean_equality(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.5, 0.4], [0.9, 0.1], mode="best_rejection")
                self.assertEqual(ns["selected_idx"], 0)
                self.assertFalse(ns["reject_candidate"])

    def test_best_rejection_still_zeroes_rejected_auxiliary_credit(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                ns = run_selection(kind, [0.8, 0.4], [1.0, 0.9], mode="best_rejection")
                self.assertTrue(ns["reject_candidate"])
                self.assertEqual(ns["playbook_text"], "baseline")
                self.assertEqual(run_auxiliary(kind, ns), ([0.0, 0.0], [0.0, 0.0]))

    def test_other_best_strategies_keep_first_tie_without_rng(self):
        for kind in KINDS:
            for mode in ("best", "best_rejection"):
                with self.subTest(kind=kind, mode=mode):
                    ns = run_selection(kind, [0.8, 0.8], [0.0, 0.0], mode=mode)
                    self.assertEqual(ns["selected_idx"], 0)

    def test_random_still_samples_all_candidates_without_seed_filtering(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                rng = random.Random(37)
                selected = {run_selection(kind, [0.8, 0.4], [1.0, 0.9], mode="random", random_source=rng)["selected_idx"] for _ in range(8)}
                self.assertEqual(selected, {0, 1})


if __name__ == "__main__":
    unittest.main()
