"""Evaluation task identity and AppWorld concurrency checks without benchmark runtimes."""

import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.appworld.eval.task_loading import load_appworld_tasks, load_specific_tasks

ROOT = Path(__file__).resolve().parents[2]


def export_ids(directory, split, ids, filename=None):
    path = directory / (filename or f"appworld_{split}_task_ids.json")
    path.write_text(json.dumps(ids))
    return path


def test_appworld_reads_requested_split_and_never_uses_training_cache(tmp_path, monkeypatch):
    export_ids(tmp_path, "dev", ["dev_first", "dev_second"])
    export_ids(tmp_path, "test_challenge", ["challenge_first", "challenge_second"])
    forbidden_registry = ModuleType("rllm.data.dataset")
    forbidden_registry.DatasetRegistry = SimpleNamespace(
        dataset_exists=Mock(side_effect=AssertionError("must not consult a cached training split")),
        load_dataset=Mock(side_effect=AssertionError("must not load a cached training split")),
        register_dataset=Mock(side_effect=AssertionError("must not mutate the training registry")),
    )
    monkeypatch.setitem(sys.modules, "rllm.data.dataset", forbidden_registry)
    monkeypatch.setenv("APPWORLD_TASK_IDS_DIR", str(tmp_path))
    challenge = load_appworld_tasks(split="test_challenge", num_tasks=None)
    dev = load_appworld_tasks(split="dev", num_tasks=1, task_offset=1)
    assert [task["task_id"] for task in challenge] == ["challenge_first", "challenge_second"]
    assert {task["split"] for task in challenge} == {"test_challenge"}
    assert dev == [{"task_id": "dev_second", "split": "dev", "index": 1}]


@pytest.mark.parametrize("filename", ("appworld_dev_task_ids.json", "dev_task_ids.json", "dev.json"))
def test_appworld_accepts_explicit_split_export_formats(tmp_path, filename):
    export_ids(tmp_path, "dev", ["one", "two"], filename)
    tasks = load_appworld_tasks(split="dev", task_ids_dir=tmp_path)
    assert [task["task_id"] for task in tasks] == ["one", "two"]


def test_appworld_missing_split_does_not_use_another_file(tmp_path):
    export_ids(tmp_path, "dev", ["dev_only"])
    with pytest.raises(FileNotFoundError, match="test_normal"):
        load_appworld_tasks(split="test_normal", task_ids_dir=tmp_path)


@pytest.mark.parametrize("payload", ([], {}, [None], [""], ["same", "same"]))
def test_appworld_rejects_invalid_or_duplicate_task_ids(tmp_path, payload):
    export_ids(tmp_path, "dev", payload)
    with pytest.raises(ValueError):
        load_appworld_tasks(split="dev", task_ids_dir=tmp_path)


def test_appworld_reports_invalid_json_source(tmp_path):
    path = tmp_path / "appworld_dev_task_ids.json"
    path.write_text("not json")
    with pytest.raises(ValueError, match="appworld_dev_task_ids.json"):
        load_appworld_tasks(split="dev", task_ids_dir=tmp_path)


@pytest.mark.parametrize("offset,limit", ((-1, None), (0, 0), (0, -1), (2, None), (1, 2), (False, 1), (0, True)))
def test_appworld_rejects_empty_or_mislabeled_slices(tmp_path, offset, limit):
    export_ids(tmp_path, "dev", ["first", "second"])
    with pytest.raises(ValueError):
        load_appworld_tasks(split="dev", task_offset=offset, num_tasks=limit, task_ids_dir=tmp_path)


def test_explicit_appworld_ids_are_unique_and_ordered():
    assert [task["task_id"] for task in load_specific_tasks(["second", "first"])] == ["second", "first"]
    with pytest.raises(ValueError, match="duplicate"):
        load_specific_tasks(["same", "same"])


@pytest.mark.parametrize("loader", (load_appworld_tasks,))
def test_unknown_split_is_rejected_before_loading(loader):
    with pytest.raises(ValueError, match="Unknown"):
        loader(split="wrong_split")


def build_appworld_engine(args):
    # Execute the real engine factory with only model loading and engine creation
    # replaced. This verifies env_args without importing torch or AppWorld.
    path = ROOT / "examples/appworld/eval/ace/appworld_ace_utils.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_generator_engine")
    tokenizer = Mock(return_value=object())
    namespace = {
        "AgentExecutionEngine": lambda **kwargs: SimpleNamespace(**kwargs),
        "AutoTokenizer": SimpleNamespace(from_pretrained=tokenizer),
        "is_external_model": lambda _model: False,
        "AppWorldReactAgentACE": object,
        "AppWorldEnv": object,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_generator_engine"](args)


def engine_args(**overrides):
    args = dict(
        model_name="test-model",
        max_turns=30,
        n_parallel_agents=4,
        n_servers=4,
        base_port=23000,
        remote_environment_url="http://127.0.0.1:23000",
        base_url="http://127.0.0.1:30000/v1",
        api_key=None,
        debug=False,
        max_model_length=40960,
    )
    return SimpleNamespace(**(args | overrides))


def test_appworld_factory_preserves_parallelism_with_one_server_per_agent():
    engine = build_appworld_engine(engine_args())
    assert engine.n_parallel_agents == 4
    assert engine.env_args == {"max_turns": 30, "remote_environment_url": "http://127.0.0.1:23000", "n_servers": 4, "base_port": 23000}


@pytest.mark.parametrize("overrides", ({"n_servers": 1}, {"n_servers": 0}, {"n_parallel_agents": 0}, {"base_port": 65535}, {"base_port": 0}))
def test_appworld_factory_rejects_unsafe_pool_settings(overrides):
    with pytest.raises(ValueError, match="AppWorld"):
        build_appworld_engine(engine_args(**overrides))


def test_appworld_single_server_runs_serially():
    engine = build_appworld_engine(engine_args(n_parallel_agents=1, n_servers=1, remote_environment_url="http://existing-server:9000"))
    assert engine.env_args["remote_environment_url"] == "http://existing-server:9000"
    assert engine.env_args["n_servers"] == 1
    assert engine.n_parallel_agents == 1
