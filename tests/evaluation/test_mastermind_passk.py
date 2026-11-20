"""Parallel Mastermind sampling with the production runner and task scheduler."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.mastermind.eval import evaluate

REPO = Path(__file__).resolve().parents[2]


def test_sample_expansion_preserves_games_and_ids():
    from rllm.environments.mastermind.mastermind import MastermindEnv

    tasks = [{"uid": "game", "seed": 137, "nested": {"value": 1}}]
    copies = evaluate.replicate_tasks(tasks, 3, 42)
    assert [sample["_evaluation_attempt_id"] for sample in copies] == [0, 1, 2]
    assert [sample["_evaluation_sampling_seed"] for sample in copies] == [42, 43, 44]
    codes = []
    for sample in copies:
        env = MastermindEnv.from_dict(sample)
        env.reset()
        codes.append(env.secret_code)
    assert codes[0] == codes[1] == codes[2]
    copies[0]["nested"]["value"] = 9
    assert tasks[0]["nested"]["value"] == copies[1]["nested"]["value"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate.replicate_tasks(tasks * 2, 3, 42)


@pytest.fixture
def execution(monkeypatch):
    # Replace model/environment execution only. ContextAwareRunner.run and
    # AgentExecutionEngine.execute_tasks retain their real batching/semaphores.
    from transformers import AutoTokenizer

    from rllm.agents.agent import Step
    from rllm.data import DatasetRegistry
    from rllm.engine.agent_execution_engine import AgentExecutionEngine

    state = SimpleNamespace(active=0, maximum=0, started=[], completed=[], requests=[], batches=[], closed=False, engine=None)
    # Registered rows carry their own budget; the evaluator must replace it.
    tasks = [{"uid": "a", "seed": 137, "max_turns": 20}, {"uid": "b", "seed": 138, "max_turns": 20}]
    monkeypatch.setattr(DatasetRegistry, "load_dataset", lambda *args: SimpleNamespace(get_data=lambda: tasks))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args: object())

    async def close():
        state.closed = True

    def initialize(self, **kwargs):
        state.engine = self
        state.init_kwargs = kwargs
        self.agent_class = kwargs["agent_class"]
        self.agent_args = kwargs["agent_args"]
        self.env_class = kwargs["env_class"]
        self.env_args = {}
        self.n_parallel_agents = self.max_env_workers = kwargs["n_parallel_agents"]
        self.agents = [None] * self.n_parallel_agents
        self.envs = [None] * self.n_parallel_agents
        self.rollout_engine = SimpleNamespace(client=SimpleNamespace(close=close))

    async def model_response(self, prompt, application_id, **kwargs):
        assert "_evaluation_sampling_seed" not in kwargs
        state.requests.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(text=f"seed={kwargs['seed']}")

    async def rollout(self, idx, application_id, **kwargs):
        agent = self.agents[idx]
        task = agent.trajectory.task
        attempt = task["_evaluation_attempt_id"]
        identity = (task["uid"], attempt)
        state.started.append(identity)
        state.active += 1
        state.maximum = max(state.maximum, state.active)
        try:
            response = await self.get_model_response(identity, application_id, **kwargs)
            await asyncio.sleep(0.001 if attempt == 0 else 0.01)
            agent.reset()
            agent.trajectory.steps = [
                Step(reward=1.0 if attempt == 0 else 0.5, metadata={"won": attempt == 0, "evaluation_raw_output": {"raw_response": response.text}}, model_response=response.text)
            ]
            agent.trajectory.reward = agent.trajectory.steps[0].reward
            agent.trajectory.info["termination_reason"] = "ENV_DONE"
            state.completed.append(identity)
            return agent.trajectory
        finally:
            state.active -= 1

    execute_tasks = AgentExecutionEngine.execute_tasks

    async def record_batch(self, tasks):
        state.batches.append([(task["uid"], task["_evaluation_attempt_id"]) for task in tasks])
        return await execute_tasks(self, tasks)

    monkeypatch.setattr(AgentExecutionEngine, "__init__", initialize)
    monkeypatch.setattr(AgentExecutionEngine, "get_model_response", model_response)
    monkeypatch.setattr(AgentExecutionEngine, "run_agent_trajectory_async", rollout)
    monkeypatch.setattr(AgentExecutionEngine, "execute_tasks", record_batch)
    return state


def test_parallel_attempts_complete_after_early_success_and_retain_raw_outputs(tmp_path, execution):
    evaluate.main(["--base-url", "http://fixture/v1", "--g-act", "3", "--parallel", "4", "--run-dir", str(tmp_path)])
    metrics = json.loads((tmp_path / "metrics_step0.json").read_text())
    trajectories = json.loads((tmp_path / "trajectories.json").read_text())
    assert execution.maximum == 3
    assert execution.batches == [[("a", 0), ("a", 1), ("a", 2)], [("b", 0), ("b", 1), ("b", 2)]]
    assert execution.completed == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)]
    assert [request["seed"] for request in execution.requests] == [42, 43, 44] * 2
    assert execution.closed and execution.engine.executor is None
    assert metrics["evaluation_protocol"] == "independent_samples_v1"
    assert (metrics["num_tasks"], metrics["num_samples"], metrics["samples_per_task"]) == (2, 6, 3)
    assert metrics["success_rate"] == metrics["pass_at_k_success_rate"] == 1.0
    assert metrics["sample_success_rate"] == pytest.approx(1 / 3)
    assert metrics["avg_reward"] == pytest.approx(2 / 3)
    assert metrics["pass_at_1_success_rate"] == pytest.approx(1 / 3)
    assert metrics["pass_at_2_success_rate"] == pytest.approx(2 / 3)
    assert metrics["pass_at_3_success_rate"] == 1.0
    assert metrics["prefix_success_rates"] == {"1": 1.0, "2": 1.0, "3": 1.0}
    for task in metrics["tasks"]:
        assert [sample["attempt_id"] for sample in task["samples"]] == [0, 1, 2]
        assert [sample["sample_index"] for sample in task["samples"]] == [0, 1, 2]
    assert len(trajectories) == len({trajectory["sample_id"] for trajectory in trajectories}) == 6
    assert [trajectory["steps"][0]["raw_response"] for trajectory in trajectories] == ["seed=42", "seed=43", "seed=44"] * 2
    assert [trajectory["task"]["seed"] for trajectory in trajectories] == [137] * 3 + [138] * 3


@pytest.mark.parametrize("argv,budget", [([], 5), (["--max-turns", "7"], 7)])
def test_max_turns_overrides_dataset_rows_and_is_recorded(tmp_path, execution, argv, budget):
    evaluate.main(["--base-url", "http://fixture/v1", "--g-act", "2", "--parallel", "2", "--run-dir", str(tmp_path), *argv])
    metrics = json.loads((tmp_path / "metrics_step0.json").read_text())
    trajectories = json.loads((tmp_path / "trajectories.json").read_text())
    assert execution.init_kwargs["env_args"] == {"max_turns": budget}
    assert execution.init_kwargs["agent_args"]["max_steps"] == execution.init_kwargs["max_steps"] == budget
    assert metrics["max_turns"] == budget
    # plot_turns reads the padded turn budget from each recorded game's task.
    assert [trajectory["task"]["max_turns"] for trajectory in trajectories] == [budget] * 4
    receipt = json.loads((tmp_path / "completed_batches/batch_000000.json").read_text())
    assert receipt["identity"]["max_turns"] == budget


def test_max_turns_must_be_positive():
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--base-url", "http://fixture/v1", "--max-turns", "0"])


def test_single_sample_keeps_existing_metric_meanings(execution):
    args = evaluate.parse_args(["--base-url", "http://fixture/v1", "--parallel", "2"])
    result = asyncio.run(evaluate.evaluate(args))
    assert result["num_tasks"] == result["num_samples"] == result["success_count"] == 2
    assert result["success_rate"] == result["avg_reward"] == result["pass_at_1_success_rate"] == 1.0
    assert all(task["num_steps"] == 1 and task["termination_reason"] == "ENV_DONE" for task in result["tasks"])
    assert [request["seed"] for request in execution.requests] == [42, 42]
    assert "trajectories" not in result


def test_vanilla_retry_reuses_completed_task_batch(tmp_path, execution, monkeypatch):
    from rllm.engine.agent_execution_engine import AgentExecutionEngine

    recorded = AgentExecutionEngine.execute_tasks
    fail = True

    async def interrupt_second_batch(self, tasks):
        if fail and tasks[0]["uid"] == "b":
            raise RuntimeError("interrupted second batch")
        return await recorded(self, tasks)

    monkeypatch.setattr(AgentExecutionEngine, "execute_tasks", interrupt_second_batch)
    command = ["--base-url", "http://fixture/v1", "--g-act", "3", "--parallel", "4", "--run-dir", str(tmp_path)]
    with pytest.raises(RuntimeError, match="interrupted second batch"):
        evaluate.main(command)
    assert (tmp_path / "completed_batches/batch_000000.json").is_file()
    assert not (tmp_path / "metrics_step0.json").exists()

    fail = False
    execution.started.clear()
    evaluate.main(command)
    assert execution.started == [("b", 0), ("b", 1), ("b", 2)]
    assert json.loads((tmp_path / "metrics_step0.json").read_text())["num_samples"] == 6


def test_incomplete_sample_group_is_rejected():
    task = evaluate.replicate_tasks([{"uid": "a"}], 2, 42)[0]
    trajectory = SimpleNamespace(task=task, steps=[], info={})
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        evaluate.summarize_trajectories([trajectory], 2)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        evaluate.summarize_trajectories([trajectory, trajectory], 2)


@pytest.mark.parametrize("argv", [["--g-act", "0"], ["--g-act", "8", "--parallel", "4"]])
def test_invalid_sampling_capacity_fails_at_parse(argv):
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--base-url", "http://fixture/v1", *argv])


def test_launcher_passes_g_act_and_preserves_all_attempts(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("RLLM_", "MODEL_", "SLURM_", "PRESET", "MAX_TURNS", "G_ACT", "PARALLEL"))}
    env.update(RLLM_DRY_RUN="1", RUNS_BASE=str(tmp_path))
    default = subprocess.run(["bash", str(REPO / "examples/mastermind/eval/vanilla/eval.slurm")], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert default.returncode == 0, default.stderr
    args = evaluate.parse_args(shlex.split(default.stdout.splitlines()[-1])[3:])
    assert (args.g_act, args.parallel, args.max_turns) == (8, 8, 5)
    assert f"{tmp_path}/pass8/" in str(args.run_dir) + "/"
    env.update(G_ACT="8", LOCAL_EVAL_EPOCHS="1", PARALLEL="16", MAX_TURNS="7")
    result = subprocess.run(["bash", str(REPO / "examples/mastermind/eval/vanilla/eval.slurm")], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.splitlines()[-1])
    args = evaluate.parse_args(command[3:])
    assert args.g_act == 8 and args.parallel == 16 and args.save_trajectories
    assert args.max_turns == 7
    env["PARALLEL"] = "4"
    invalid = subprocess.run(["bash", str(REPO / "examples/mastermind/eval/vanilla/eval.slurm")], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert invalid.returncode != 0 and "PARALLEL" in invalid.stderr

    env["PARALLEL"] = "64"
    env["TASK_CONCURRENCY"] = "8"
    ignored = subprocess.run(["bash", str(REPO / "examples/mastermind/eval/vanilla/eval.slurm")], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert ignored.returncode != 0 and "TASK_CONCURRENCY" in ignored.stderr


def test_evaluation_agent_keeps_raw_output_after_env_update():
    """The shared agent replaces step info on update_from_env; raw outputs must survive it."""
    agent = evaluate.build_evaluation_agent_class()(max_steps=3, use_accumulate_history=True, use_accumulate_thinking=False, save_raw_outputs=True)
    agent.reset()
    agent.update_from_env({"result": "Turn 1"}, 0.0, False, {})
    model_output = SimpleNamespace(text="<guess>1234</guess>", content="<guess>1234</guess>", reasoning=None, finish_reason="stop", completion_length=7)
    agent.update_from_model("<guess>1234</guess>", model_output=model_output)
    assert agent.trajectory.steps[-1].info[evaluate.RAW_OUTPUT_KEY]["raw_response"] == "<guess>1234</guess>"
    agent.update_from_env({"result": "2 exact"}, 0.5, False, {"won": False, "secret_code": "1243"})
    step = agent.trajectory.steps[-1]
    assert step.reward == 0.5 and step.info["won"] is False
    assert step.info[evaluate.RAW_OUTPUT_KEY] == {
        "raw_response": "<guess>1234</guess>",
        "engine_text": "<guess>1234</guess>",
        "engine_content": "<guess>1234</guess>",
        "engine_reasoning": None,
        "finish_reason": "stop",
        "completion_length": 7,
    }
    payload = evaluate.trajectory_payload(agent.trajectory, 0)
    assert payload["steps"][-1]["raw_response"] == "<guess>1234</guess>"
    assert "secret_code" not in payload["steps"][-1]["info"] and evaluate.RAW_OUTPUT_KEY not in payload["steps"][-1]["info"]

    plain = evaluate.build_evaluation_agent_class()(max_steps=3, use_accumulate_history=True, use_accumulate_thinking=False)
    plain.reset()
    plain.update_from_env({"result": "Turn 1"}, 0.0, False, {})
    plain.update_from_model("<guess>1234</guess>", model_output=model_output)
    plain.update_from_env({"result": "2 exact"}, 0.5, False, {"won": False})
    assert evaluate.RAW_OUTPUT_KEY not in plain.trajectory.steps[-1].info
