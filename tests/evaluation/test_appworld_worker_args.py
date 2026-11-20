"""Worker isolation without importing optional GPU dependencies."""

import argparse
import ast
import copy
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]


def worker_args_function():
    path = ROOT / "examples/appworld/eval/ace/appworld_ace_utils.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_local_worker_args")
    namespace = {"copy": copy, "urlsplit": urlsplit}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def settings(**overrides):
    return argparse.Namespace(
        **{
            "n_servers": 4,
            "n_parallel_agents": 4,
            "base_port": 12000,
            "remote_environment_url": "http://127.0.0.1:12000",
            "g_act": 1,
            "g_ref": 1,
            "g_cur": 1,
            "local_eval_epochs": 3,
            "seed": 42,
            "gen_temperature": 0.7,
            "max_model_length": 32768,
            "reflector_max_history_tokens": 20000,
            "nested": {"role_settings": [0.7]},
            **overrides,
        }
    )


def test_workers_have_disjoint_single_server_endpoints_and_preserve_protocol():
    prepare = worker_args_function()
    original = settings()
    before = copy.deepcopy(vars(original))
    workers = [prepare(original, index, 4) for index in range(4)]
    assert [worker.base_port for worker in workers] == [12000, 12001, 12002, 12003]
    assert [worker.remote_environment_url for worker in workers] == [f"http://127.0.0.1:{port}" for port in range(12000, 12004)]
    assert all(worker.n_servers == worker.n_parallel_agents == 1 for worker in workers)
    for worker in workers:
        for key in ("g_act", "g_ref", "g_cur", "local_eval_epochs", "seed", "gen_temperature", "max_model_length", "reflector_max_history_tokens"):
            assert getattr(worker, key) == before[key]
    workers[0].nested["role_settings"].append(1.0)
    assert workers[1].nested == before["nested"]
    assert vars(original) == before


def test_pass8_workers_share_sixteen_server_pool_with_eight_slots_each():
    prepare = worker_args_function()
    original = settings(n_servers=16, n_parallel_agents=16, g_act=8)
    before = copy.deepcopy(vars(original))
    workers = [prepare(original, index, 2) for index in range(2)]
    assert all(worker.n_servers == 16 for worker in workers)
    assert all(worker.n_parallel_agents == 8 for worker in workers)
    assert all(worker.base_port == 12000 for worker in workers)
    assert all(worker.remote_environment_url == "http://127.0.0.1:12000" for worker in workers)
    assert vars(original) == before


@pytest.mark.parametrize("url", [None, "http://[::1]:12000/path", "https://example.test:12000/prefix"])
def test_worker_url_preserves_endpoint_host_and_path(url):
    worker = worker_args_function()(settings(remote_environment_url=url), 2, 4)
    expected = "http://127.0.0.1:12002" if url is None else url.replace(":12000", ":12002")
    assert worker.remote_environment_url == expected


@pytest.mark.parametrize(
    "overrides,index,count",
    [
        ({"n_servers": 2}, 0, 4),
        ({"n_parallel_agents": 2}, 0, 4),
        ({"g_act": 2}, 0, 4),
        ({"base_port": 65534}, 0, 4),
        ({"base_port": 0}, 0, 4),
        ({"remote_environment_url": "http://127.0.0.1:13000"}, 0, 4),
        ({"remote_environment_url": "http://127.0.0.1"}, 0, 4),
        ({}, 4, 4),
        ({}, -1, 4),
        ({}, 0, 0),
    ],
)
def test_invalid_worker_capacity_or_endpoint_fails_before_engine_creation(overrides, index, count):
    with pytest.raises(ValueError):
        worker_args_function()(settings(**overrides), index, count)
