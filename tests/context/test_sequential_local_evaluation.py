"""Sequential policy visits retain measured outcomes without search or stopping."""

import asyncio
import json
from types import MethodType

import pytest
from test_local_evaluation_concurrency import Curator, Engine, Recorder, Reflector, Rule, setup_run

from rllm.agents.agent import BaseAgent, Step, Trajectory
from rllm.algorithms.ace import evaluation


@pytest.mark.parametrize("visits", [1, 3, 8])
def test_exact_policy_reflection_curator_chain_and_final_visit_reuse(tmp_path, visits):
    async def run():
        events = []

        class OrderedEngine(Engine):
            async def execute_tasks(self, tasks):
                events.append("policy")
                return await super().execute_tasks(tasks)

        class OrderedReflector(Reflector):
            async def analyze(self, payload, before):
                events.append("reflector")
                return await super().analyze(payload, before)

        class OrderedCurator(Curator):
            async def generate_operations(self, payload, insights, before):
                events.append("curator")
                return await super().generate_operations(payload, insights, before)

        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder, concurrency=1, tasks=[{"id": "alpha"}], engine_class=OrderedEngine)
        rule = Rule(recorder)
        rule.reflector = OrderedReflector(recorder, rule)
        rule.curator = OrderedCurator(recorder, rule)
        kwargs["hooks"].create_context_rule = lambda *a, **kw: rule
        kwargs["args"].local_eval_epochs = visits
        kwargs["args"].final_policy_eval = True
        result = await evaluation.run_local_ace(**kwargs)
        entry = result.evaluations[0]
        assert events == [item for _ in range(visits - 1) for item in ("policy", "reflector", "curator")] + ["policy"]
        assert [tips for _, tips, _ in recorder.generator_calls] == list(range(visits))
        assert entry["local_eval_protocol"] == evaluation.SEQUENTIAL_LOCAL_PROTOCOL
        assert entry["policy_visits"] == visits and entry["adaptation_updates"] == visits - 1
        assert entry["rollout_counts"] == {"policy": visits, "reflector": visits - 1, "curator": visits - 1, "candidate": 0, "total": 3 * visits - 2}
        final = entry["final_policy_evaluation"]
        assert final["kind"] == "last_measured_policy_visit"
        assert final["policy_visit"] == visits and final["after_local_eval_epoch"] == visits - 1
        assert final["provenance"]["extra_policy_rollouts"] == 0
        assert final["tasks"] == entry["local_epochs"][-1]["tasks"]
        assert final["metrics"] == entry["local_epochs"][-1]["metrics"]
        assert not final["tasks"][0]["adaptation_applied"]
        assert entry["adaptation_evaluation"]["local_epochs"] == entry["local_epochs"][:-1]
        assert len(list((tmp_path / "artifacts").glob("task_*/chat_completions/eval/*.json"))) == visits
        assert json.loads((tmp_path / "metrics.json").read_text())["local_eval_protocol"] == evaluation.SEQUENTIAL_LOCAL_PROTOCOL

    asyncio.run(run())


def test_success_then_failure_continues_and_preserves_raw_final_policy(tmp_path):
    class SuccessThenFailure(Engine):
        async def execute_tasks(self, tasks):
            trajectories = await super().execute_tasks(tasks)
            reward = 1.0 if len(self.recorder.generator_calls) == 1 else 0.0
            for trajectory in trajectories:
                trajectory.reward = trajectory.steps[0].reward = reward
            return trajectories

    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder, concurrency=1, tasks=[{"id": "alpha"}], engine_class=SuccessThenFailure)
        result = await evaluation.run_local_ace(**kwargs)
        entry = result.evaluations[0]
        assert len(recorder.generator_calls) == 3
        assert len(recorder.reflector_calls) == len(recorder.curator_calls) == 2
        assert [visit["tasks"][0]["reward"] for visit in entry["local_epochs"]] == [1.0, 0.0, 0.0]
        assert [visit["tasks"][0]["samples"][0]["success"] for visit in entry["local_epochs"]] == [True, False, False]
        assert [visit["metrics"]["cumulative_success_rate"] for visit in entry["local_epochs"]] == [1.0] * 3
        final = entry["final_policy_evaluation"]
        assert final["metrics"]["avg_reward"] == final["metrics"]["pass_at_k_success_rate"] == 0.0
        assert final["metrics"]["cumulative_success_rate"] == 1.0
        assert final["metrics"]["max_reward_mean"] == entry["metrics"]["final_policy/max_reward_mean"] == 0.0
        assert [tips for _, tips, _ in recorder.generator_calls] == [0, 1, 2]  # Deterioration does not reject the delta.
        for visit, expected in zip(entry["local_epochs"], (1.0, 0.0, 0.0), strict=True):
            artifact = visit["tasks"][0]["artifacts"]["policy_chat_completions"]
            assert json.loads(open(artifact).read())["trajectories"][0]["reward"] == expected

    asyncio.run(run())


@pytest.mark.parametrize("delta", [
    [], [{"type": "ADD", "content": 7}],
    [{"type": "UPDATE", "bullet_id": "missing", "content": "unused"}],
    [{"type": "ADD", "content": "partial change"}, {"type": "ADD", "content": 7}],
])
def test_invalid_or_noop_delta_keeps_playbook_but_still_runs_next_visit(tmp_path, delta):
    class InvalidCurator(Curator):
        async def generate_operations(self, *args):
            self.recorder.curator_calls.append("alpha")
            return {"result": {"operations": delta}}

    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder, concurrency=1, tasks=[{"id": "alpha"}])
        rule = Rule(recorder)
        rule.curator = InvalidCurator(recorder, rule)
        kwargs["hooks"].create_context_rule = lambda *a, **kw: rule
        result = await evaluation.run_local_ace(**kwargs)
        assert [tips for _, tips, _ in recorder.generator_calls] == [0, 0, 0]
        assert len(recorder.reflector_calls) == len(recorder.curator_calls) == 2
        entry = result.evaluations[0]
        assert all(not visit["tasks"][0]["adaptation_applied"] for visit in entry["local_epochs"])
        assert entry["actually_applied_context_updates"] == 0
        assert entry["adaptation_update_slots_per_task"] == 2

    asyncio.run(run())


def test_real_parallel_executor_retains_all_eight_samples_after_first_success(tmp_path):
    # Exercise the production scheduler, substituting only the environment/model
    # interaction. Every attempt is active before the first successful result.
    from rllm.engine.agent_execution_engine import AgentExecutionEngine

    class Agent(BaseAgent):
        def __init__(self):
            self.reset()

        def reset(self):
            self.messages = []
            self._trajectory = Trajectory()

        @property
        def trajectory(self):
            return self._trajectory

    class Environment:
        created = []

        @classmethod
        def from_dict(cls, task):
            env = cls()
            cls.created.append(env)
            return env

    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder, concurrency=1, tasks=[{"id": "alpha"}])
        args = kwargs["args"]
        args.g_act, args.g_ref, args.g_cur, args.local_eval_epochs = 8, 0, 0, 1
        engine = AgentExecutionEngine.__new__(AgentExecutionEngine)
        engine.agent_class, engine.env_class = Agent, Environment
        engine.agent_args, engine.env_args = {}, {}
        engine.n_parallel_agents = engine.max_env_workers = 8
        engine.envs, engine.agents = [None] * 8, [None] * 8
        started, finished = [], []
        all_started, succeeded = asyncio.Event(), asyncio.Event()

        async def interaction(self, index, application_id):
            started.append(application_id)
            if len(started) == 8:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), 2)
            if application_id == 0:
                succeeded.set()
            else:
                await succeeded.wait()
                await asyncio.sleep(0)
            reward = float(application_id == 0)
            trajectory = self.agents[index].trajectory
            trajectory.reward = reward
            trajectory.steps = [Step(action=f"sample-{application_id}", reward=reward, done=True)]
            finished.append(application_id)
            return trajectory

        engine.run_agent_trajectory_async = MethodType(interaction, engine)
        kwargs["hooks"].build_generator_engine = lambda args: engine
        result = await evaluation.run_local_ace(**kwargs)
        entry = result.evaluations[0]
        assert len(started) == len(finished) == len(Environment.created) == 8
        assert finished[0] == 0 and len({id(env) for env in Environment.created}) == 8
        assert not recorder.reflector_calls and not recorder.curator_calls
        row = entry["tasks"][0]
        assert [sample["sample_index"] for sample in row["samples"]] == list(range(8))
        assert [sample["reward"] for sample in row["samples"]] == [1.0] + [0.0] * 7
        assert len({sample["attempt_id"] for sample in row["samples"]}) == 8
        assert row["reward"] == 1 / 8
        assert entry["metrics"]["pass_at_k_success_rate"] == 1.0
        assert entry["metrics"]["pass_at_1_success_rate"] == 1 / 8
        assert entry["metrics"]["pass_at_4_success_rate"] == 0.5
        artifact = json.loads(open(row["artifacts"]["policy_chat_completions"]).read())
        assert artifact["num_trajectories"] == 8
        assert [trajectory["steps"][0]["action"] for trajectory in artifact["trajectories"]] == [f"sample-{index}" for index in range(8)]
        assert len({trajectory["info"]["evaluation_sample"]["attempt_id"] for trajectory in artifact["trajectories"]}) == 8

    asyncio.run(run())


@pytest.mark.parametrize("settings", [
    {"g_ref": 2}, {"g_cur": 2}, {"g_ref": 0, "g_cur": 1},
    {"freeze_solved": True}, {"warm_start_from_run_dir": "/old/run", "start_epoch": 3},
    {"final_policy_from_run_dir": "/old/run", "final_policy_eval": True},
    {"local_eval_epochs": 0}, {"g_act": 0}, {"local_eval_protocol": "unknown"},
    {"g_act": 8, "n_parallel_agents": 4},
])
def test_invalid_or_legacy_only_configuration_fails_before_resources(tmp_path, settings):
    recorder = Recorder()
    kwargs = setup_run(tmp_path, recorder)
    for key, value in settings.items():
        setattr(kwargs["args"], key, value)
    with pytest.raises(ValueError):
        asyncio.run(evaluation.run_local_ace(**kwargs))
    assert not recorder.engines and not recorder.rules


def test_incomplete_policy_batch_cannot_publish_metrics(tmp_path):
    class EmptyBatch(Engine):
        async def execute_tasks(self, tasks):
            await super().execute_tasks(tasks)
            return []

    kwargs = setup_run(tmp_path, Recorder(), concurrency=1, engine_class=EmptyBatch)
    with pytest.raises(RuntimeError, match="Incomplete trajectory batch"):
        asyncio.run(evaluation.run_local_ace(**kwargs))
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize("pre_receipt_artifacts", [False, True])
def test_retry_reuses_completed_task_visits_and_finishes_partial_run(tmp_path, pre_receipt_artifacts):
    class FailOnSecondBetaVisit(Engine):
        async def execute_tasks(self, tasks):
            task = tasks[0]
            if task["id"] == "beta" and len([item for item in self.recorder.generator_calls if item[0] == "beta"]) == 1:
                raise RuntimeError("interrupted task visit")
            return await super().execute_tasks(tasks)

    tasks = [{"id": "alpha"}, {"id": "beta"}]
    first_recorder = Recorder()
    first = setup_run(tmp_path / "partial", first_recorder, concurrency=1, tasks=tasks, engine_class=FailOnSecondBetaVisit)
    first["args"].resume = True
    with pytest.raises(RuntimeError, match="interrupted task visit"):
        asyncio.run(evaluation.run_local_ace(**first))
    assert not (tmp_path / "partial/metrics.json").exists()
    receipts = list((tmp_path / "partial/artifacts").glob("task_*/completed_visits/*.json"))
    assert len(receipts) == 3
    if pre_receipt_artifacts:
        for receipt in receipts:
            receipt.unlink()

    retry_recorder = Recorder()
    retry = setup_run(tmp_path / "partial", retry_recorder, concurrency=1, tasks=tasks)
    retry["args"].resume = True
    resumed = asyncio.run(evaluation.run_local_ace(**retry))
    assert [(task, visit) for task, visit, _ in retry_recorder.generator_calls] == [("beta", 1), ("alpha", 2), ("beta", 2)]

    fresh = asyncio.run(evaluation.run_local_ace(**setup_run(tmp_path / "fresh", Recorder(), concurrency=1, tasks=tasks)))
    assert [epoch["metrics"] for epoch in resumed.evaluations[0]["local_epochs"]] == [epoch["metrics"] for epoch in fresh.evaluations[0]["local_epochs"]]
    assert [[row["reward"] for row in epoch["tasks"]] for epoch in resumed.evaluations[0]["local_epochs"]] == [[row["reward"] for row in epoch["tasks"]] for epoch in fresh.evaluations[0]["local_epochs"]]


def test_legacy_warm_start_refuses_sequential_playbook_history(tmp_path):
    kwargs = setup_run(tmp_path, Recorder(), concurrency=1, tasks=[{"id": "alpha"}])
    asyncio.run(evaluation.run_local_ace(**kwargs))
    with pytest.raises(ValueError, match="Legacy warm start cannot import sequential"):
        evaluation.load_warm_start_state(kwargs["playbook_snapshot_dir"], ["alpha"], 0, 3)
