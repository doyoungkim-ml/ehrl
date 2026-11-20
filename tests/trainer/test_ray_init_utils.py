import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_ray_init_utils():
    module_path = Path(__file__).resolve().parents[2] / "rllm" / "trainer" / "ray_init_utils.py"
    spec = importlib.util.spec_from_file_location("rllm_ray_init_utils_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_get_ray_init_settings_defaults_to_local_cluster(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda temp_dir=None: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert "address" not in settings


def test_get_ray_init_settings_attaches_when_ray_address_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("RAY_ADDRESS", "ray://dummy")

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda temp_dir=None: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert settings["address"] == "auto"


def test_get_ray_init_settings_attaches_when_ray_current_cluster_file_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    marker = tmp_path / "ray_current_cluster"
    marker.write_text("dummy")

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda temp_dir=None: marker)

    settings = ray_init_utils.get_ray_init_settings(config=None)
    assert settings["address"] == "auto"


def test_config_address_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("RAY_ADDRESS", "ray://dummy")

    class Cfg:
        ray_init = {"address": "ray://explicit"}

    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils, "_ray_current_cluster_path", lambda temp_dir=None: tmp_path / "missing")

    settings = ray_init_utils.get_ray_init_settings(Cfg())
    assert settings["address"] == "ray://explicit"


def test_job_temp_dir_ignores_marker_in_default_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "shared"))
    shared_ray = tmp_path / "shared" / "ray"
    shared_ray.mkdir(parents=True)
    (shared_ray / "ray_current_cluster").write_text("127.0.0.1:12345")
    ray_init_utils = _load_ray_init_utils()
    assert ray_init_utils.should_attach_to_existing_ray_cluster()

    config = SimpleNamespace(ray_init={"_temp_dir": str(tmp_path / "job"), "num_cpus": 32})
    settings = ray_init_utils.get_ray_init_settings(config)
    assert settings == config.ray_init


def test_job_temp_dir_uses_its_own_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("RAY_TMPDIR", str(tmp_path / "unused"))
    job_root = tmp_path / "job"
    job_root.mkdir()
    (job_root / "ray_current_cluster").write_text("127.0.0.1:12345")
    config = SimpleNamespace(ray_init={"_temp_dir": str(job_root)})
    settings = _load_ray_init_utils().get_ray_init_settings(config)
    assert settings == {"_temp_dir": str(job_root), "address": "auto"}


@pytest.mark.parametrize("env_name", ["RAY_TMPDIR", "TMPDIR"])
def test_environment_temp_root_adds_ray_subdirectory(monkeypatch, tmp_path, env_name):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_TMPDIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "unused"))
    monkeypatch.setenv(env_name, str(tmp_path))
    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils.sys, "platform", "linux")
    assert ray_init_utils._ray_current_cluster_path() == tmp_path / "ray" / "ray_current_cluster"
    assert "address" not in ray_init_utils.get_ray_init_settings()
    (tmp_path / "ray").mkdir()
    (tmp_path / "ray" / "ray_current_cluster").write_text("127.0.0.1:12345")
    assert ray_init_utils.get_ray_init_settings()["address"] == "auto"


def test_linux_default_path_without_environment(monkeypatch):
    monkeypatch.delenv("RAY_TMPDIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    ray_init_utils = _load_ray_init_utils()
    monkeypatch.setattr(ray_init_utils.sys, "platform", "linux")
    assert ray_init_utils._ray_current_cluster_path() == Path("/tmp/ray/ray_current_cluster")


@pytest.mark.parametrize("address", ["local", "auto", "ray://explicit"])
def test_explicit_address_preserved_with_job_temp_dir(monkeypatch, tmp_path, address):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    config = SimpleNamespace(ray_init={"_temp_dir": str(tmp_path), "address": address})
    assert _load_ray_init_utils().get_ray_init_settings(config) == config.ray_init
