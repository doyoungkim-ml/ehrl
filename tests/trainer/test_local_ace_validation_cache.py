"""Validation cache regressions executing production ACT and selection blocks.

The model/environment sampler is a CPU adapter. Cache consumption, task grouping,
selection, and metadata expansion are production methods/statements. Set
ACE_VALIDATION_REAL_DATAPROTO=1 in an existing VeRL environment to also check real
DataProto/TensorDict unions; no Ray workers or GPUs are started. The file runs
with unittest, including environments without pytest.
"""

from __future__ import annotations

import ast
import copy
import logging
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from test_standard_rejection_update import (
    Playbook,
    assignment,
    execute,
    local_selection,
    standard_candidate_metadata,
    trainer_namespace,
)

from rllm.utils.training_reward import raw_reward_metrics

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("ACE_VALIDATION_TRAINER_SOURCE", ROOT / "rllm/algorithms/ace/trainer.py"))
TASK_INDICES = {"task-a": list(range(8)), "task-b": list(range(8, 16))}


def production_methods(trainer):
    names = {
        "_pop_local_eval_policy_trajectory_cache_by_task",
        "_pop_validation_policy_trajectory_cache",
        "_store_local_eval_policy_trajectory_cache_by_task",
        "_group_trajectories_by_task",
        "_flatten_grouped_trajectories_in_env_order",
        "_ordered_grouped_entries",
        "_flatten_grouped_base_indices_in_env_order",
    }
    namespace = {"copy": copy, "np": np, "logger": logging.getLogger(__name__)}
    for filename, selected in (
        (ROOT / "rllm/algorithms/ace/trainer.py", names),
        (ROOT / "rllm/algorithms/ace/trainer_base.py", {"_append_policy_and_component_rows_to_batch"}),
    ):
        tree = ast.parse(filename.read_text())
        methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in selected]
        assert {node.name for node in methods} == selected
        for method in methods:
            execute([method], namespace)
            setattr(trainer, method.name, MethodType(namespace[method.name], trainer))
    return trainer


def rows(task, count, *, offset=0, reward=0.0):
    return [{"idx": offset + i, "source_task": task, "sample": i, "trajectory_reward": reward, "step_rewards": [reward]} for i in range(count)]


def sampler_trainer():
    trainer = production_methods(trainer_namespace("best_rejection_update", "seed", kind="standard")["self"])
    trainer._local_eval_policy_trajectory_cache = {}
    trainer._set_component_token_limits = lambda component: None
    trainer._inject_task_playbooks_into_agents = lambda *args, **kwargs: None
    trainer._in_local_eval = True
    trainer.agent_execution_engine = SimpleNamespace(envs=[object() for _ in range(16)])
    trainer.generation_calls = 0

    def sample(**kwargs):
        trainer.generation_calls += 1
        # Different task rewards make accidental reassignment observable.
        yield from rows("task-a", 8, reward=0.0)
        yield from rows("task-b", 8, offset=8, reward=1.0)

    trainer.generate_agent_trajectories_async = sample
    return trainer


def run_policy_phase(trainer, *, is_validation=True):
    """Execute the actual batched ACT block, including cache read and grouping."""
    tree = ast.parse(SOURCE.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_generate_multi_task_batched_full_ace")
    body = next(node.body for node in ast.walk(method) if isinstance(getattr(node, "body", None), list) and any(assignment(child, "reused_validation_rollout") for child in node.body))
    start = next(i for i, node in enumerate(body) if assignment(node, "reused_validation_rollout"))
    end = next(i for i, node in enumerate(body) if assignment(node, "grouped_policy_trajectories"))
    namespace = {
        "self": trainer,
        "is_validation": is_validation,
        "task_ids": list(TASK_INDICES),
        "task_to_indices": TASK_INDICES,
        "timing_raw": {},
        "meta_info": {"validate": is_validation},
        "copy": copy,
        "logger": logging.getLogger(__name__),
        "marked_timer": lambda *args: nullcontext(),
        "raw_reward_metrics": raw_reward_metrics,
        "original_envs": trainer.agent_execution_engine.envs,
    }
    execute(body[start : end + 1], namespace)
    return namespace


def select_next_epoch_cache(trainer, policy_phase):
    """Real selection accepts one capped candidate and rejects the other task's."""
    candidates = [
        {"task_id": "task-a", "task_idx": 0, "traj_idx": 0, "guidebook_uid": "gb-a", "playbook": Playbook("updated-a")},
        {"task_id": "task-b", "task_idx": 1, "traj_idx": 0, "guidebook_uid": "gb-b", "playbook": Playbook("rejected-b")},
    ]
    candidate_playbooks, curator_records = standard_candidate_metadata("standard", candidates)
    namespace = {
        **policy_phase,
        "self": trainer,
        "np": np,
        "Playbook": Playbook,
        "candidate_playbooks": candidate_playbooks,
        "curator_records": curator_records,
        "candidate_rewards": [0.5, 0.0],
        "candidate_eval_trajectories": {0: rows("task-a", 1, reward=0.5), 1: rows("task-b", 1)},
        "current_rewards_by_task": {"task-a": 0.0, "task-b": 1.0},
        "playbook_before_text_by_task": {"task-a": "initial-a", "task-b": "initial-b"},
        "store": {},
    }
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
        namespace[name] = {}
    for name in ("task_playbook_sizes", "task_playbook_bullets", "task_metric_pairs"):
        namespace[name] = []
    execute([local_selection("standard")], namespace)
    trainer._store_local_eval_policy_trajectory_cache_by_task(namespace["next_validation_trajectories"])
    return namespace


def two_epochs():
    trainer = sampler_trainer()
    first = run_policy_phase(trainer)
    selection = select_next_epoch_cache(trainer, first)
    second = run_policy_phase(trainer)
    return trainer, first, selection, second


class ValidationCacheTests(unittest.TestCase):
    def assert_task_ownership(self, phase):
        self.assertEqual(len(phase["policy_trajectories"]), 16)
        for task_id, task_rows in phase["grouped_policy_trajectories"].items():
            self.assertEqual(len(task_rows), 8)
            self.assertEqual({row["source_task"] for row in task_rows}, {task_id})
            self.assertEqual([row["sample"] for row in task_rows], list(range(8)))

    def test_selected_candidate_cache_regenerates_requested_batch(self):
        trainer, first, selection, second = two_epochs()
        self.assertEqual(selection["selected_idx_by_task"], {"task-a": 0, "task-b": -1})
        self.assertEqual({key: len(value) for key, value in selection["next_validation_trajectories"].items()}, {"task-a": 1, "task-b": 8})
        self.assertEqual(selection["playbook_after_text_by_task"], {"task-a": "updated-a", "task-b": "initial-b"})
        self.assertEqual(trainer.generation_calls, 2)
        self.assertFalse(second["reused_validation_rollout"])
        self.assert_task_ownership(first)
        self.assert_task_ownership(second)
        self.assertEqual(trainer._local_eval_policy_trajectory_cache, {})

    def test_compatible_cache_reuses_all_real_samples(self):
        trainer = sampler_trainer()
        trainer._store_local_eval_policy_trajectory_cache_by_task({"task-a": rows("task-a", 8), "task-b": rows("task-b", 8)})
        phase = run_policy_phase(trainer)
        self.assertTrue(phase["reused_validation_rollout"])
        self.assertEqual(trainer.generation_calls, 0)
        self.assert_task_ownership(phase)

    def test_counts_must_match_each_task_even_if_batch_total_matches(self):
        trainer = sampler_trainer()
        trainer._store_local_eval_policy_trajectory_cache_by_task({"task-a": rows("task-a", 4), "task-b": rows("task-b", 12)})
        phase = run_policy_phase(trainer)
        self.assertFalse(phase["reused_validation_rollout"])
        self.assertEqual(trainer.generation_calls, 1)
        self.assert_task_ownership(phase)

    def test_partial_cache_regenerates_and_preserves_other_batch_tasks(self):
        trainer = sampler_trainer()
        trainer._store_local_eval_policy_trajectory_cache_by_task({"task-b": rows("task-b", 8), "another-batch": rows("another-batch", 8)})
        phase = run_policy_phase(trainer)
        self.assertEqual(trainer.generation_calls, 1)
        self.assert_task_ownership(phase)
        self.assertEqual(list(trainer._local_eval_policy_trajectory_cache), ["another-batch"])

    def test_single_task_cache_requires_requested_samples(self):
        trainer = sampler_trainer()
        trainer.agent_execution_engine.envs = trainer.agent_execution_engine.envs[:8]
        trainer._artifact_task_id = "task-a"
        trainer._store_local_eval_policy_trajectory_cache_by_task({"task-a": rows("task-a", 1)})
        self.assertIsNone(trainer._pop_validation_policy_trajectory_cache())
        trainer._store_local_eval_policy_trajectory_cache_by_task({"task-a": rows("task-a", 8)})
        self.assertEqual(len(trainer._pop_validation_policy_trajectory_cache()), 8)
        self.assertEqual(trainer._local_eval_policy_trajectory_cache, {})

    def test_training_does_not_consume_validation_cache(self):
        trainer = sampler_trainer()
        trainer._store_local_eval_policy_trajectory_cache_by_task({"task-a": rows("task-a", 1)})
        phase = run_policy_phase(trainer, is_validation=False)
        self.assertEqual(trainer.generation_calls, 1)
        self.assertFalse(phase["reused_validation_rollout"])
        self.assertEqual(len(trainer._local_eval_policy_trajectory_cache["task-a"]), 1)
        self.assert_task_ownership(phase)

    def test_seed_and_candidates_rows_join_their_task_group(self):
        trainer = production_methods(trainer_namespace("best_rejection_update", "seed_and_candidates", kind="standard")["self"])
        task_to_indices = {"task-a": [0, 1, 2, 3], "task-b": [4, 5, 6, 7]}
        grouped = {"task-a": rows("task-a", 4) + rows("task-a", 3, offset=100), "task-b": rows("task-b", 4, offset=4) + rows("task-b", 1, offset=200)}

        class FakeBatch:
            pass

        batch = FakeBatch()
        batch.non_tensor_batch = {"uid": np.array(["uid-a"] * 4 + ["uid-b"] * 4, dtype=object), "task": np.array(["task-a"] * 4 + ["task-b"] * 4, dtype=object)}
        flattened = trainer._flatten_grouped_trajectories_in_env_order(grouped, task_to_indices)
        base_indices = trainer._flatten_grouped_base_indices_in_env_order(grouped, task_to_indices)
        fake_tensordict = SimpleNamespace(TensorDict=lambda data, batch_size: SimpleNamespace(batch_size=tuple(batch_size)))
        with patch.dict(sys.modules, {"tensordict": fake_tensordict}):
            trainer._append_policy_and_component_rows_to_batch(
                batch, extra_policy_count=4, component_extra_uids=["ref-a"], extra_policy_base_indices=base_indices[8:], share_extra_policy_uids=True
            )

        self.assertEqual(batch.non_tensor_batch["task"][:12].tolist(), [row["source_task"] for row in flattened])
        self.assertEqual(batch.non_tensor_batch["uid"].tolist(), ["uid-a"] * 4 + ["uid-b"] * 4 + ["uid-a"] * 3 + ["uid-b", "ref-a"])

@unittest.skipUnless(os.environ.get("ACE_VALIDATION_REAL_DATAPROTO") == "1", "requires an existing VeRL/TensorDict environment")
class RealDataProtoValidationTests(unittest.TestCase):
    def test_selected_candidate_cache_supports_two_epoch_batch_union(self):
        import torch
        from verl import DataProto

        trainer, first, _, second = two_epochs()
        for phase in (first, second):
            with self.subTest(reused=phase["reused_validation_rollout"]):
                # The validation loop creates the configured eight rows/task
                # every epoch; auxiliary component rows are appended afterwards.
                batch = DataProto.from_dict(
                    tensors={"input_ids": torch.zeros((2, 1), dtype=torch.int64)},
                    non_tensors={"uid": np.array(["uid-a", "uid-b"], dtype=object), "task": np.array(["task-a", "task-b"], dtype=object)},
                ).repeat(repeat_times=8, interleave=True)
                batch.pop(["input_ids"])
                trainer._append_policy_and_component_rows_to_batch(batch, component_extra_uids=["ref-a", "ref-b", "cur-a", "cur-b"])
                policy_rows = phase["policy_trajectories"]
                output = DataProto.from_dict(tensors={"token_level_scores": torch.tensor([[row["trajectory_reward"]] for row in policy_rows] + [[0.0]] * 4)})
                batch = batch.union(output)
                self.assertEqual(len(batch), 20)
                self.assertEqual(batch.non_tensor_batch["task"][:16].tolist(), [row["source_task"] for row in policy_rows])
                self.assertEqual(batch.batch["token_level_scores"][:8].sum().item(), 0.0)
                self.assertEqual(batch.batch["token_level_scores"][8:16].sum().item(), 8.0)


if __name__ == "__main__":
    unittest.main()
