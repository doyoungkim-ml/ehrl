"""CPU regressions executing the production candidate scorer and async engine.

Only model/environment execution is replaced. The real semaphore, synchronous
generator bridge, candidate construction, aggregation and lifecycle run here.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import logging
import random
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Event, Thread, Timer
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def bind_methods(target, filename, names, **extra):
    namespace = {
        "asyncio": asyncio,
        "copy": copy,
        "np": np,
        "logger": logging.getLogger(__name__),
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "Queue": Queue,
        "Thread": Thread,
        "traceback": traceback,
        **extra,
    }
    tree = ast.parse((ROOT / filename).read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert {node.name for node in methods} == names
    for method in methods:
        method.decorator_list = []
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    for name in names:
        setattr(target, name, MethodType(namespace[name], target))
    return target


class FakeEnv:
    created = []

    def __init__(self, task):
        self.task = task
        self.close_calls = 0

    @classmethod
    def from_dict(cls, task):
        env = cls(task)
        cls.created.append(env)
        return env

    @staticmethod
    def is_multithread_safe():
        return True

    def close(self):
        self.close_calls += 1


class FakeAgent:
    pass


class Playbook:
    def __init__(self, text):
        self.text = text

    def to_string(self):
        return self.text


class RolloutEngine:
    def __init__(self):
        self.wakes = 0
        self.sleeps = 0

    async def wake_up(self):
        self.wakes += 1

    async def sleep(self):
        self.sleeps += 1


def make_trainer(*, base_parallel=2, cap=2, batch_size=2, rollouts=0):
    FakeEnv.created = []
    trainer = SimpleNamespace(
        candidate_eval_batch_size=batch_size,
        candidate_eval_max_parallel_agents=cap,
        candidate_eval_rollouts=rollouts,
        env_class=FakeEnv,
        agent_class=FakeAgent,
        _rollout_engine_external_management_depth=0,
    )
    sources = {"a": [FakeEnv({"source": 10}), FakeEnv({"source": 11})], "b": [FakeEnv({"source": 20})]}
    trainer._original_envs = [env for group in sources.values() for env in group]
    trainer._original_agents = [FakeAgent() for _ in trainer._original_envs]
    trainer._get_full_agent_args = lambda: {}
    trainer._get_full_env_args = lambda: {}
    trainer._parse_candidate_eval_rollouts = lambda value: int(value)
    trainer._inject_playbook_into_agents = lambda agents, text: [setattr(agent, "playbook", text) for agent in agents]
    trainer._should_capture_selected_candidate_policy_trajectories = lambda: False
    bind_methods(
        trainer,
        "rllm/algorithms/ace/trainer_base.py",
        {
            "_evaluate_candidate_pool",
            "_evaluate_candidate_playbooks",
            "_get_candidate_eval_parallel_agents",
            "_get_candidate_eval_source_envs",
            "_build_candidate_eval_envs_and_agents",
            "_compute_rollout_token_totals",
        },
    )
    bind_methods(trainer, "rllm/algorithms/ace/trainer.py", {"_evaluate_local_candidate_playbooks", "_build_local_candidate_eval_envs_and_agents"})
    bind_methods(trainer, "rllm/algorithms/ace/trainer_delayed_joint_base.py", {"_evaluate_candidate_playbooks_with_trajectories"})
    bind_methods(trainer, "rllm/trainer/verl/agent_ppo_trainer.py", {"generate_agent_trajectories_async"})
    engine = SimpleNamespace(
        n_parallel_agents=base_parallel,
        engine_name="verl",
        rollout_engine=RolloutEngine(),
        envs=trainer._original_envs,
        agents=trainer._original_agents,
        active=0,
        peak=0,
        started=[],
        finished=[],
    )
    bind_methods(
        engine,
        "rllm/engine/agent_execution_engine.py",
        {"trajectory_generator", "update_envs_and_agents"},
        BaseEnv=FakeEnv,
        colorful_print=lambda *args, **kwargs: None,
        is_fatal_rollout_engine_error=lambda exc: False,
    )

    async def run(idx, **kwargs):
        engine.active += 1
        engine.peak = max(engine.peak, engine.active)
        engine.started.append(idx)
        try:
            await asyncio.sleep(0)
            env, agent = engine.envs[idx], engine.agents[idx]
            env.close()
            return {
                "idx": idx,
                "trajectory_reward": int(agent.playbook) + env.task["source"] / 100,
                "source": env.task["source"],
                "metrics": {"total_prompt_tokens": idx + 10, "total_completion_tokens": idx + 1},
            }
        finally:
            engine.finished.append(idx)
            engine.active -= 1

    engine.run_agent_trajectory_with_retry = run
    trainer.agent_execution_engine = engine
    candidates = [
        {"task_id": task, "guidebook_uid": f"gb-{idx}", "playbook": Playbook(str(idx)), "task_idx": idx, "task_uid": task, "traj_idx": idx}
        for idx, task in enumerate(("a", "b", "a", "b"))
    ]
    return trainer, sources, candidates


def evaluate(trainer, sources, candidates, capture=True):
    return trainer._evaluate_local_candidate_playbooks(candidates, sources, capture_eval_trajectories=capture)


class TestCandidatePool(unittest.TestCase):
    def assert_restored(self, trainer):
        self.assertIs(trainer.agent_execution_engine.envs, trainer._original_envs)
        self.assertIs(trainer.agent_execution_engine.agents, trainer._original_agents)

    def test_straggler_does_not_block_next_logical_batch(self):
        trainer, sources, candidates = make_trainer(batch_size=1)
        engine = trainer.agent_execution_engine
        run = engine.run_agent_trajectory_with_retry
        later_started = None
        controlled_active = 0
        controlled_peak = 0

        async def controlled_run(idx, **kwargs):
            nonlocal later_started, controlled_active, controlled_peak
            controlled_active += 1
            controlled_peak = max(controlled_peak, controlled_active)
            try:
                if later_started is None:
                    later_started = asyncio.Event()
                if idx == 0:
                    # Candidate zero occupies indices 0/1. The next candidate's
                    # index 2 must start while zero is still waiting.
                    await asyncio.wait_for(later_started.wait(), timeout=2)
                elif idx == 2:
                    self.assertNotIn(0, engine.finished)
                    later_started.set()
                return await run(idx, **kwargs)
            finally:
                controlled_active -= 1

        engine.run_agent_trajectory_with_retry = controlled_run
        rewards, records, trajectories, metrics = evaluate(trainer, sources, candidates)
        self.assertEqual(len(rewards), 4)
        self.assertEqual(len(metrics), 6)
        self.assertEqual(engine.n_parallel_agents, 2)
        self.assertLessEqual(engine.peak, 2)
        self.assertEqual(controlled_peak, 2)
        self.assertEqual(controlled_active, 0)
        self.assertEqual([row["candidate_idx"] for row in records], list(range(4)))
        self.assertEqual([row["source"] for row in trajectories[0]], [10, 11])
        self.assertEqual((engine.rollout_engine.wakes, engine.rollout_engine.sleeps), (1, 1))
        self.assert_restored(trainer)

    def test_reordered_results_preserve_mapping_samples_metrics_and_rng(self):
        trainer, sources, candidates = make_trainer()
        seen = []

        def reverse_results(**kwargs):
            engine = trainer.agent_execution_engine
            seen.append((len(engine.envs), engine.n_parallel_agents, kwargs))
            for idx in reversed(range(len(engine.envs))):
                env, agent = engine.envs[idx], engine.agents[idx]
                env.close()
                yield {
                    "idx": idx,
                    "source": env.task["source"],
                    "trajectory_reward": int(agent.playbook) + env.task["source"] / 100,
                    "metrics": {"total_prompt_tokens": idx + 10, "total_completion_tokens": idx + 1},
                }

        trainer.generate_agent_trajectories_async = reverse_results
        state = random.getstate()
        rewards, records, trajectories, metrics = evaluate(trainer, sources, candidates)
        self.assertEqual(random.getstate(), state)
        np.testing.assert_allclose(rewards, [0.105, 1.2, 2.105, 3.2])
        self.assertEqual([row["task_id"] for row in records], ["a", "b", "a", "b"])
        self.assertEqual([row["eval_rollouts"] for row in records], [2, 1, 2, 1])
        self.assertEqual([row["source"] for row in trajectories[2]], [10, 11])
        self.assertEqual([row["total_completion_tokens"] for row in metrics], list(range(1, 7)))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][:2], (6, 2))
        self.assertIs(trainer._last_candidate_eval_trajectories, trajectories)
        self.assertTrue(all(env.close_calls == 1 for env in FakeEnv.created))
        self.assert_restored(trainer)

    def test_capacity_matches_largest_legacy_chunk_not_flattened_count(self):
        for base, cap, batch, expected in ((2, 0, 1, 2), (8, 0, 1, 2), (2, 0, 2, 3), (1, 2, 2, 2), (8, 1, 2, 3)):
            with self.subTest(base=base, cap=cap, batch=batch):
                trainer, sources, candidates = make_trainer(base_parallel=base, cap=cap, batch_size=batch)
                _, records, _, _ = evaluate(trainer, sources, candidates[:3])
                self.assertTrue(all(row["eval_parallel_agents"] == expected for row in records))
                self.assertLessEqual(trainer.agent_execution_engine.peak, expected)
                self.assertEqual(trainer.agent_execution_engine.n_parallel_agents, base)
                self.assertEqual(records[-1]["eval_batch_candidates"], 1)

    def test_uncaptured_evaluation_keeps_all_metrics(self):
        trainer, sources, candidates = make_trainer(rollouts=1)
        rewards, records, trajectories, metrics = evaluate(trainer, sources, candidates, capture=False)
        self.assertEqual(len(rewards), 4)
        self.assertEqual(trajectories, {})
        self.assertIsNone(trainer._last_candidate_eval_trajectories)
        self.assertEqual(sum(row["total_completion_tokens"] for row in metrics), 10)
        self.assertTrue(all(row["eval_rollouts"] == 1 for row in records))
        totals = trainer._compute_rollout_token_totals.__func__(
            policy_trajectories=[{"metrics": {"total_prompt_tokens": 3, "total_completion_tokens": 2, "total_gradient_update_tokens": 1}}],
            candidate_eval_token_metrics=metrics,
        )
        self.assertEqual(totals, {"rollout/total_prompt_tokens": 49, "rollout/total_completion_tokens": 12, "rollout/total_gradient_update_tokens": 1, "rollout/total_trajectories": 5})

    def test_external_scope_owns_wake_sleep(self):
        trainer, sources, candidates = make_trainer()
        trainer._rollout_engine_external_management_depth = 1
        evaluate(trainer, sources, candidates)
        rollout = trainer.agent_execution_engine.rollout_engine
        self.assertEqual((rollout.wakes, rollout.sleeps), (0, 0))

    def test_empty_candidates_and_missing_sources(self):
        trainer, sources, candidates = make_trainer(batch_size=1)
        self.assertEqual(evaluate(trainer, sources, []), ([], [], {}, []))
        self.assertEqual(evaluate(trainer, {}, candidates), ([0.0] * 4, [], {}, []))
        rewards, records, _, metrics = evaluate(trainer, {"b": sources["b"]}, candidates)
        self.assertEqual(rewards, [0, 1.2, 0, 3.2])
        self.assertEqual([row["candidate_idx"] for row in records], [1, 3])
        self.assertEqual(len(metrics), 2)

    def test_bad_result_identity_fails_after_draining_and_restores(self):
        for result_indices in ((0, 0, 2, 3, 4, 5), (0, 1, 3, 4, 5), (0, 1, 2, 3, 4, 99)):
            with self.subTest(indices=result_indices):
                trainer, sources, candidates = make_trainer()
                drained = []

                def bad_results(result_indices=result_indices, drained=drained, **kwargs):
                    for idx in result_indices:
                        yield {"idx": idx, "trajectory_reward": 1}
                    drained.append(True)

                trainer.generate_agent_trajectories_async = bad_results
                with self.assertRaises(ValueError):
                    evaluate(trainer, sources, candidates)
                self.assertEqual(drained, [True])
                self.assert_restored(trainer)
                rollout = trainer.agent_execution_engine.rollout_engine
                self.assertEqual((rollout.wakes, rollout.sleeps), (1, 1))

    def test_generation_failure_closes_unfinished_environments_and_restores(self):
        trainer, sources, candidates = make_trainer()

        def fail(**kwargs):
            raise RuntimeError("broken rollout backend")

        trainer.generate_agent_trajectories_async = fail
        with self.assertRaisesRegex(RuntimeError, "broken rollout backend"):
            evaluate(trainer, sources, candidates)
        self.assertTrue(all(env.close_calls == 1 for env in FakeEnv.created))
        self.assertEqual(trainer.agent_execution_engine.n_parallel_agents, 2)
        self.assert_restored(trainer)
        rollout = trainer.agent_execution_engine.rollout_engine
        self.assertEqual((rollout.wakes, rollout.sleeps), (1, 1))

    def test_global_standard_and_delayed_use_same_pool(self):
        for delayed in (False, True):
            with self.subTest(delayed=delayed):
                trainer, _, candidates = make_trainer(rollouts=1)
                if delayed:
                    rewards, records, trajectories = trainer._evaluate_candidate_playbooks_with_trajectories(candidates)
                    self.assertEqual([row["task_idx"] for row in records], list(range(4)))
                    self.assertEqual(len(trajectories), 4)
                    self.assertTrue(all(len(group) == 1 for group in trajectories))
                else:
                    rewards, records = trainer._evaluate_candidate_playbooks(candidates, meta_info={"validate": True})
                    self.assertEqual(len(trainer._last_candidate_eval_trajectories), 4)
                np.testing.assert_allclose(rewards, [0.1, 1.1, 2.1, 3.1])
                self.assertEqual(len(records), 4)
                self.assertEqual(trainer.agent_execution_engine.started, list(range(4)))
                self.assert_restored(trainer)

    def test_failure_waits_for_cancelled_environment_operation_before_close(self):
        trainer, sources, candidates = make_trainer()
        engine = trainer.agent_execution_engine
        running, release, finished = Event(), Event(), Event()
        cleanup_races = []
        original_close = FakeEnv.close

        def operation():
            running.set()
            release.wait(timeout=2)
            finished.set()

        async def fail_while_other_environment_runs(idx, **kwargs):
            if idx == 1:
                await asyncio.get_running_loop().run_in_executor(engine.executor, operation)
            else:
                while not running.is_set():
                    await asyncio.sleep(0)
                # Failure cancels the coroutine but cannot cancel operation().
                Timer(0.05, release.set).start()
                raise RuntimeError("failed while another environment was running")

        def checked_close(env):
            if running.is_set() and not finished.is_set():
                cleanup_races.append(env)
            original_close(env)

        engine.run_agent_trajectory_with_retry = fail_while_other_environment_runs
        with patch.object(FakeEnv, "close", checked_close):
            with self.assertRaisesRegex(RuntimeError, "failed while another environment was running"):
                evaluate(trainer, sources, candidates)
        self.assertTrue(finished.is_set())
        self.assertEqual(cleanup_races, [])
        self.assertTrue(all(env.close_calls == 1 for env in FakeEnv.created))
        self.assert_restored(trainer)


if __name__ == "__main__":
    unittest.main()
