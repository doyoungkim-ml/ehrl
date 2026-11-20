"""Concurrent local ACE visits retain isolated state and deterministic aggregation."""

import asyncio
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rllm.agents.agent import Step, Trajectory
from rllm.algorithms.ace import evaluation
from rllm.algorithms.ace.context.playbook import Playbook
from rllm.context.base import ContextUpdateLevel


class Client:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class Recorder:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.active = 0
        self.peak = 0
        self.two_active = asyncio.Event()
        self.engines = []
        self.rules = []
        self.clients = []
        self.prepared = []
        self.generator_calls = []
        self.reflector_calls = []
        self.curator_calls = []
        self.finished = []
        self.events = []

    def client(self):
        client = Client()
        self.clients.append(client)
        return client


class Agent:
    def reset(self):
        self.messages = []


class Environment:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class Engine:
    agent_class = Agent

    def __init__(self, recorder, args):
        self.recorder = recorder
        self.worker_index = getattr(args, "worker_index", 0)
        self.shutdown_calls = 0
        self.inflight = False
        self.envs = [Environment()]
        self.rollout_engine = SimpleNamespace(client=recorder.client())
        recorder.engines.append(self)

    async def execute_tasks(self, tasks):
        assert len(tasks) == 1  # Concurrency must not multiply g_act.
        assert not self.inflight
        self.inflight = True
        task = tasks[0]
        task_id = task["id"]
        recorder = self.recorder
        current_class = self.agent_class
        agent = current_class()
        agent.reset()
        prompt = agent.messages[0]["content"]
        tips = re.findall(r"task=(\w+) visit=(\d+)", prompt)
        assert all(owner == task_id for owner, _ in tips), "Playbook leaked across tasks"
        recorder.generator_calls.append((task_id, len(tips), self.worker_index))
        recorder.events.append(("start", task_id))
        recorder.active += 1
        recorder.peak = max(recorder.peak, recorder.active)
        if recorder.active >= 2:
            recorder.two_active.set()
        try:
            if recorder.failure:
                await recorder.two_active.wait()
                if recorder.failure == "task" and task_id == "alpha":
                    raise RuntimeError("acting failed")
                await asyncio.Event().wait()
            await asyncio.sleep(0.004 if task_id == "alpha" else 0.001)
            assert self.agent_class is current_class, "Runner changed another task's agent class"
            reward = len(tips) / 10
            if task.get("reject") and tips:
                reward = -0.1
            trajectory = Trajectory(
                task=task,
                reward=reward,
                steps=[Step(observation=task_id, action="inspect", reward=reward, done=True)],
            )
            trajectory.info["next_visit"] = len(tips) + 1
            recorder.finished.append(task_id)
            recorder.events.append(("finish", task_id))
            return [trajectory]
        finally:
            recorder.active -= 1
            self.inflight = False

    def shutdown(self):
        assert not self.inflight
        self.shutdown_calls += 1


class Reflector:
    def __init__(self, recorder, rule):
        self.recorder = recorder
        self.rule = rule
        self.rollout_engine = SimpleNamespace(client=recorder.client())

    async def analyze(self, payload, before):
        task = payload["trajectory"].task
        self.recorder.reflector_calls.append(task["id"])
        assert self.rule.get_context() == before
        await asyncio.sleep(0)
        assert self.rule.get_context() == before
        if task.get("reflect_error"):
            raise RuntimeError("reflection failed")
        return {"result": {"key_insight": "Inspect specifications"}}


class Curator:
    def __init__(self, recorder, rule):
        self.recorder = recorder
        self.rule = rule
        self.rollout_engine = SimpleNamespace(client=recorder.client())

    async def generate_operations(self, payload, insights, before):
        task = payload["trajectory"].task
        self.recorder.curator_calls.append(task["id"])
        assert self.rule.get_context() == before
        assert "Task 0" in insights  # Each task remains a logical one-task batch.
        await asyncio.sleep(0)
        assert self.rule.get_context() == before
        if task.get("invalid_curator"):
            return {"result": {"operations": []}}
        visit = payload["trajectory"].info["next_visit"]
        return {"result": {"operations": [{"type": "ADD", "content": f"Advice for task={task['id']} visit={visit}"}]}}


class Rule:
    update_level = ContextUpdateLevel.BATCH

    def __init__(self, recorder):
        self.playbook = Playbook("")
        self.config = SimpleNamespace(reflector_only=False)
        self.reflector = Reflector(recorder, self)
        self.curator = Curator(recorder, self)
        recorder.rules.append(self)

    def get_context(self):
        return self.playbook.to_string()


def setup_run(tmp_path, recorder, *, concurrency=2, tasks=None, hook=True, build_failure=False, engine_class=Engine):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "base_system_prompt.txt").write_text("Advice: {{playbook}}")
    args = SimpleNamespace(
        model_name="test-model",
        local_eval_epochs=3,
        g_act=1,
        g_ref=1,
        g_cur=1,
        selection_mode="best_rejection",
        max_turns=30,
        training_input_parity=True,
        freeze_solved=False,
    )
    if concurrency is not None:
        args.task_concurrency = concurrency

    def prepare_worker(original, index, count):
        recorder.prepared.append((index, count))
        assert original is args
        return SimpleNamespace(**vars(original), worker_index=index)

    def build_engine(worker_args):
        if build_failure and worker_args.worker_index == 1:
            raise RuntimeError("engine construction failed")
        return engine_class(recorder, worker_args)

    hooks = evaluation.ACEExperimentHooks(
        experiment_name="concurrency-test",
        task_id_getter=lambda task: task["id"],
        build_generator_engine=build_engine,
        create_context_rule=lambda *a, **kw: Rule(recorder),
        trajectory_reward=lambda trajectory: trajectory.reward,
        trajectory_success=lambda trajectory: trajectory.reward == 1,
        prepare_local_worker_args=prepare_worker if hook else None,
    )
    tasks = tasks if tasks is not None else [{"id": name} for name in ("alpha", "beta", "gamma", "delta")]
    kwargs = dict(
        args=args,
        hooks=hooks,
        train_tasks=[],
        eval_tasks=tasks,
        prompts_dir=str(tmp_path),
        initial_playbook_path=str(tmp_path / "unused-playbook.txt"),
        playbook_snapshot_dir=str(tmp_path / "artifacts"),
        ace_log_dir=None,
        metrics_file=str(tmp_path / "metrics.json"),
        script_path=__file__,
    )
    return kwargs


def canonical(value, root):
    if isinstance(value, dict):
        return {key: canonical(item, root) for key, item in value.items() if key not in {"timestamp", "run_id"}}
    if isinstance(value, list):
        return [canonical(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<ROOT>")
    return value


def test_concurrent_visits_overlap_without_state_or_artifact_leaks(tmp_path):
    async def run():
        serial_recorder, concurrent_recorder = Recorder(), Recorder()
        serial_root, concurrent_root = tmp_path / "serial", tmp_path / "concurrent"
        serial = await evaluation.run_local_ace(**setup_run(serial_root, serial_recorder, concurrency=1, hook=False))
        concurrent = await evaluation.run_local_ace(**setup_run(concurrent_root, concurrent_recorder))
        assert canonical(serial.evaluations, serial_root) == canonical(concurrent.evaluations, concurrent_root)
        assert serial_recorder.peak == 1
        assert concurrent_recorder.peak == 2
        assert concurrent_recorder.finished[0] == "beta"  # Completion order differs.
        assert concurrent_recorder.prepared == [(0, 2), (1, 2)]
        assert len(concurrent_recorder.engines) == len(concurrent_recorder.rules) == 2
        assert len(concurrent_recorder.generator_calls) == 12
        assert len(concurrent_recorder.reflector_calls) == len(concurrent_recorder.curator_calls) == 8
        for task_id in ("alpha", "beta", "gamma", "delta"):
            # Measured visit k sees exactly k-1 prior updates; there is no scoring pass.
            assert [tips for task, tips, _ in concurrent_recorder.generator_calls if task == task_id] == [0, 1, 2]
        started_counts = dict.fromkeys(("alpha", "beta", "gamma", "delta"), 0)
        finished_counts = dict(started_counts)
        for event, task_id in concurrent_recorder.events:
            if event == "start":
                prior = started_counts[task_id]
                if prior in (1, 2):
                    assert all(count >= prior for count in finished_counts.values()), "Next visit crossed the epoch barrier"
                started_counts[task_id] += 1
            else:
                finished_counts[task_id] += 1
        for epoch in concurrent.evaluations[0]["local_epochs"]:
            assert [row["task_id"] for row in epoch["tasks"]] == ["alpha", "beta", "gamma", "delta"]
        serial_logs = sorted(serial_root.glob("artifacts/task_*/ace_logs/eval/*.json"))
        concurrent_logs = sorted(concurrent_root.glob("artifacts/task_*/ace_logs/eval/*.json"))
        assert len(serial_logs) == len(concurrent_logs) == 12
        for old, new in zip(serial_logs, concurrent_logs, strict=True):
            assert old.relative_to(serial_root) == new.relative_to(concurrent_root)
            assert canonical(json.loads(old.read_text()), serial_root) == canonical(json.loads(new.read_text()), concurrent_root)
        for task_index, task_id in enumerate(("alpha", "beta", "gamma", "delta")):
            for path in (concurrent_root / f"artifacts/task_{task_id}/chat_completions/eval").glob("*.json"):
                assert json.loads(path.read_text())["batch_idx"] == task_index + 1
        assert all(engine.shutdown_calls == 1 and engine.agent_class is Agent for engine in concurrent_recorder.engines)
        assert all(client.close_calls == 1 for client in concurrent_recorder.clients)

    asyncio.run(run())


def test_default_one_keeps_serial_path_without_worker_hook(tmp_path):
    async def run():
        implicit_recorder, explicit_recorder = Recorder(), Recorder()
        implicit_root, explicit_root = tmp_path / "implicit", tmp_path / "explicit"
        implicit = await evaluation.run_local_ace(**setup_run(implicit_root, implicit_recorder, concurrency=None, hook=False))
        explicit = await evaluation.run_local_ace(**setup_run(explicit_root, explicit_recorder, concurrency=1, hook=False))
        assert canonical(implicit.evaluations, implicit_root) == canonical(explicit.evaluations, explicit_root)
        assert not implicit_recorder.prepared and not explicit_recorder.prepared
        assert len(implicit_recorder.engines) == len(explicit_recorder.engines) == 1
        assert implicit_recorder.generator_calls == explicit_recorder.generator_calls

    asyncio.run(run())


def test_conditional_role_and_candidate_work_preserves_budget(tmp_path):
    async def run():
        recorder = Recorder()
        tasks = [{"id": "alpha", "reflect_error": True}, {"id": "beta", "invalid_curator": True}, {"id": "gamma", "reject": True}, {"id": "delta"}]
        result = await evaluation.run_local_ace(**setup_run(tmp_path, recorder, tasks=tasks))
        assert len(recorder.generator_calls) == 12  # Three measured visits for all four tasks.
        assert len(recorder.reflector_calls) == 8
        assert len(recorder.curator_calls) == 6
        rows = {row["task_id"]: row for row in result.evaluations[0]["tasks"]}
        assert rows["alpha"]["num_candidates"] == rows["beta"]["num_candidates"] == 0
        assert rows["gamma"]["reject_candidate"] == 0
        assert rows["gamma"]["playbook_bullets"] == 2
        assert rows["delta"]["playbook_bullets"] == 2

    asyncio.run(run())


def test_task_failure_cancels_siblings_and_closes_all_workers(tmp_path):
    async def run():
        recorder = Recorder(failure="task")
        with pytest.raises(RuntimeError, match="acting failed"):
            await evaluation.run_local_ace(**setup_run(tmp_path, recorder))
        assert recorder.peak == 2 and recorder.active == 0
        assert all(engine.shutdown_calls == 1 and not engine.inflight for engine in recorder.engines)
        assert all(engine.envs[0].close_calls == 1 for engine in recorder.engines)
        assert all(client.close_calls == 1 for client in recorder.clients)
        assert not (tmp_path / "metrics.json").exists()

    asyncio.run(run())


def test_external_cancellation_finishes_worker_cleanup(tmp_path):
    async def run():
        recorder = Recorder(failure="cancel")
        task = asyncio.create_task(evaluation.run_local_ace(**setup_run(tmp_path, recorder)))
        await recorder.two_active.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert recorder.active == 0
        assert all(engine.shutdown_calls == 1 for engine in recorder.engines)
        assert all(client.close_calls == 1 for client in recorder.clients)
        assert not (tmp_path / "metrics.json").exists()

    asyncio.run(run())


def test_partial_worker_construction_closes_created_resources(tmp_path):
    async def run():
        recorder = Recorder()
        with pytest.raises(RuntimeError, match="engine construction failed"):
            await evaluation.run_local_ace(**setup_run(tmp_path, recorder, build_failure=True))
        assert len(recorder.rules) == 2 and len(recorder.engines) == 1
        assert recorder.engines[0].shutdown_calls == 1
        assert all(client.close_calls == 1 for client in recorder.clients)

    asyncio.run(run())


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_invalid_concurrency_fails_before_resource_creation(tmp_path, concurrency):
    async def run():
        recorder = Recorder()
        with pytest.raises(ValueError, match="positive integer"):
            await evaluation.run_local_ace(**setup_run(tmp_path, recorder, concurrency=concurrency))
        assert not recorder.rules and not recorder.engines

    asyncio.run(run())


@pytest.mark.parametrize("duplicate", [False, True])
def test_concurrent_mode_requires_hook_and_unique_task_ids(tmp_path, duplicate):
    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder, hook=duplicate, tasks=[{"id": "alpha"}, {"id": "alpha"}] if duplicate else None)
        with pytest.raises(ValueError, match="unique task IDs" if duplicate else "prepare_local_worker_args"):
            await evaluation.run_local_ace(**kwargs)
        assert not recorder.rules and not recorder.engines

    asyncio.run(run())


def test_cleanup_waits_for_inflight_environment_threads(tmp_path):
    async def run():
        recorder = Recorder()
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        started_count = 0

        class ThreadEnvironment(Environment):
            def __init__(self):
                super().__init__()
                self.finished = False

            def operation(self):
                nonlocal started_count

                def mark_started():
                    nonlocal started_count
                    started_count += 1
                    if started_count == 2:
                        started.set()

                loop.call_soon_threadsafe(mark_started)
                release.wait(timeout=5)
                self.finished = True

            def close(self):
                assert self.finished, "Closed server while its environment call was still running"
                super().close()

        class ThreadEngine(Engine):
            def __init__(self, recorder, args):
                super().__init__(recorder, args)
                self.executor = ThreadPoolExecutor(max_workers=1)
                self.envs = [ThreadEnvironment()]

            async def execute_tasks(self, tasks):
                await loop.run_in_executor(self.executor, self.envs[0].operation)
                raise AssertionError("The outer task should be cancelled before release")

            def shutdown(self):
                self.executor.shutdown(wait=True)
                self.shutdown_calls += 1

        task = asyncio.create_task(evaluation.run_local_ace(**setup_run(tmp_path, recorder, engine_class=ThreadEngine)))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert all(env.close_calls == 0 for engine in recorder.engines for env in engine.envs)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(engine.shutdown_calls == 1 for engine in recorder.engines)
        assert all(env.close_calls == 1 for engine in recorder.engines for env in engine.envs)
        assert all(client.close_calls == 1 for client in recorder.clients)

    asyncio.run(run())


def test_sanitized_task_artifact_collisions_fail_before_execution(tmp_path):
    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder, tasks=[{"id": "a/b"}, {"id": "a_b"}])
        with pytest.raises(ValueError, match="unique task artifact paths"):
            await evaluation.run_local_ace(**kwargs)
        assert not recorder.engines and not recorder.rules

    asyncio.run(run())


def test_vanilla_concurrency_keeps_one_visit_and_no_adaptation(tmp_path):
    async def run():
        recorder = Recorder()
        kwargs = setup_run(tmp_path, recorder)
        kwargs["args"].g_ref = kwargs["args"].g_cur = 0
        kwargs["args"].local_eval_epochs = 1
        result = await evaluation.run_local_ace(**kwargs)
        assert recorder.peak == 2
        assert len(recorder.generator_calls) == 4
        assert recorder.reflector_calls == recorder.curator_calls == []
        assert all(row["playbook_bullets"] == 0 for row in result.evaluations[0]["tasks"])
        assert len(result.evaluations[0]["local_epochs"]) == 1

    asyncio.run(run())
