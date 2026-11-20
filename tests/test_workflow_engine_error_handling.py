"""Tests for AgentWorkflowEngine permanent-failure handling.

Covers rllm.workflow.raise_on_error plumbing:
- raise_on_error=False returns the ERROR episode instead of raising,
- raise_on_error=True (legacy default) still raises,
- every failed attempt is appended to the persistent error-log JSONL
  with the human-readable task instance id.
"""

import asyncio
import json

import pytest

from rllm.agents.agent import Episode
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine, _task_instance_id
from rllm.workflows.workflow import TerminationReason


class _AlwaysFailingWorkflow:
    """Stub workflow whose every rollout terminates with ERROR."""

    def __init__(self, **kwargs):
        pass

    def is_multithread_safe(self):
        return True

    async def run_with_termination_handling(self, task, uid, **kwargs):
        episode = Episode(id=uid)
        episode.termination_reason = TerminationReason.ERROR
        episode.info["error"] = {
            "error_type": "RuntimeError",
            "error_message": "container exploded",
            "traceback": "Traceback (most recent call last): boom",
        }
        return episode


def _make_engine(tmp_path, raise_on_error):
    return AgentWorkflowEngine(
        workflow_cls=_AlwaysFailingWorkflow,
        workflow_args={},
        rollout_engine=None,
        n_parallel_tasks=1,
        retry_limit=3,
        raise_on_error=raise_on_error,
        error_log_path=str(tmp_path / "rollout_errors.jsonl"),
    )


def test_no_raise_returns_error_episode_and_logs(tmp_path):
    engine = _make_engine(tmp_path, raise_on_error=False)

    async def run():
        await engine.initialize_pool()
        return await engine.process_task_with_retry({"instance_id": "getmoto__moto-6387"}, "task-1", 0)

    task_id, rollout_idx, episode = asyncio.run(run())
    assert task_id == "task-1" and rollout_idx == 0
    assert episode.termination_reason == TerminationReason.ERROR

    records = [json.loads(line) for line in (tmp_path / "rollout_errors.jsonl").read_text().splitlines()]
    assert len(records) == 3  # one per attempt
    assert [r["attempt"] for r in records] == [1, 2, 3]
    assert all(r["instance_id"] == "getmoto__moto-6387" for r in records)
    assert all(r["error_message"] == "container exploded" for r in records)
    assert all(r["traceback"] for r in records)


def test_raise_on_error_still_raises(tmp_path):
    engine = _make_engine(tmp_path, raise_on_error=True)

    async def run():
        await engine.initialize_pool()
        return await engine.process_task_with_retry({"instance_id": "getmoto__moto-6387"}, "task-1", 0)

    with pytest.raises(Exception, match="failed permanently"):
        asyncio.run(run())
    # attempts are logged even on the raising path
    records = (tmp_path / "rollout_errors.jsonl").read_text().splitlines()
    assert len(records) == 3


def test_no_error_log_path_is_silent(tmp_path):
    engine = AgentWorkflowEngine(
        workflow_cls=_AlwaysFailingWorkflow,
        workflow_args={},
        rollout_engine=None,
        n_parallel_tasks=1,
        retry_limit=1,
        raise_on_error=False,
        error_log_path=None,
    )

    async def run():
        await engine.initialize_pool()
        return await engine.process_task_with_retry({}, "task-1", 0)

    task_id, _, episode = asyncio.run(run())  # must not throw
    assert episode.termination_reason == TerminationReason.ERROR


def test_task_instance_id_fallbacks():
    assert _task_instance_id({"instance_id": "a"}) == "a"
    assert _task_instance_id({"task_id": "b"}) == "b"
    assert _task_instance_id({"unrelated": 1}) == "unknown"
    assert _task_instance_id(None) == "unknown"
