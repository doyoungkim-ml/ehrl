"""Exercise the generic task boundary with a non-ACE trainer and fake workers."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rllm.trainer.verl.task import configure_managed_hf_cache, run_agent_task


@pytest.fixture
def backend(monkeypatch):
    events = []
    instances = []

    class AlgorithmTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instances.append(self)

        def init_workers(self):
            events.append("init")
            if self.kwargs.get("fail_at") == "init":
                raise RuntimeError("init failed")

        def fit_agent(self):
            events.append("fit")
            if self.kwargs.get("fail_at") == "fit":
                raise RuntimeError("fit failed")

        def shutdown(self):
            events.append("shutdown")

    modules = {
        "verl": {},
        "verl.trainer": {},
        "verl.trainer.ppo": {},
        "verl.trainer.ppo.reward": {"load_reward_manager": lambda config, tokenizer, **kwargs: dict(kwargs)},
        "verl.utils": {"hf_tokenizer": lambda path, **kwargs: (path, kwargs)},
        "verl.utils.fs": {"copy_to_local": lambda path, **kwargs: path},
        "rllm.experimental.verl.patch": {
            "patch_verl_dynamic_batch_sync": lambda: events.append("dynamic-batch-patch"),
            "patch_verl_training_output_postprocess": lambda: events.append("output-patch"),
        },
        "foundation_test_algorithm": {"Trainer": AlgorithmTrainer, "Agent": type("Agent", (), {}), "Env": type("Env", (), {})},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("RLLM_CACHE_DIR", raising=False)
    runner = SimpleNamespace(
        add_actor_rollout_worker=lambda config: ("actor", "worker_group"),
        add_critic_worker=lambda config: events.append("critic"),
        add_ref_policy_worker=lambda config, actor: events.append(("reference", actor)),
        init_resource_pool_mgr=lambda config: "pool",
        role_worker_mapping={"actor": "worker"},
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"model": {"path": "model", "use_shm": False}},
            "data": {"trust_remote_code": False},
            "reward_model": {"reward_kwargs": {"score": "test"}},
            "rllm": {"env": {"env_args": {"override": "config"}}, "agent": {"agent_args": {"override": "config"}}},
        }
    )
    return runner, config, instances, events


def execute(backend, **kwargs):
    runner, config, _, _ = backend
    return run_agent_task(
        runner,
        config=config,
        trainer_class_path="foundation_test_algorithm.Trainer",
        default_agent_class_path="foundation_test_algorithm.Agent",
        default_env_class_path="foundation_test_algorithm.Env",
        **kwargs,
    )


def test_algorithm_constructor_and_benchmark_precedence(backend):
    defaults = {"keep": 1, "override": "default"}
    algorithm = {"algorithm_config": {"schedule": "custom"}}
    execute(backend, agent_args=defaults, env_args=defaults, trainer_kwargs=algorithm)
    _, config, instances, events = backend
    kwargs = instances[0].kwargs
    assert kwargs["agent_args"] == kwargs["env_args"] == {"keep": 1, "override": "config"}
    assert kwargs["algorithm_config"] == {"schedule": "custom"}
    assert "ace_config" not in kwargs
    assert kwargs["role_worker_mapping"] == {"actor": "worker"}
    assert kwargs["resource_pool_manager"] == "pool"
    assert kwargs["ray_worker_group_cls"] == "worker_group"
    assert kwargs["reward_fn"] == kwargs["val_reward_fn"] == {"score": "test"}
    assert defaults == {"keep": 1, "override": "default"}
    assert algorithm == {"algorithm_config": {"schedule": "custom"}}
    assert config.rllm.env.env_args == {"override": "config"}
    assert events[-5:] == ["dynamic-batch-patch", "output-patch", "init", "fit", "shutdown"]


@pytest.mark.parametrize("failure,tail", [("init", ["init", "shutdown"]), ("fit", ["init", "fit", "shutdown"])])
def test_worker_cleanup_on_failure(backend, failure, tail):
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        execute(backend, trainer_kwargs={"fail_at": failure})
    assert backend[3][-len(tail) :] == tail


def test_explicit_agent_and_environment_override_default_paths(backend):
    agent, env = object(), object()
    execute(backend, agent_class=agent, env_class=env)
    assert backend[2][0].kwargs["agent_class"] is agent
    assert backend[2][0].kwargs["env_class"] is env


@pytest.mark.parametrize("reserved", ["config", "tokenizer", "agent_class", "env_args", "role_worker_mapping"])
def test_algorithm_cannot_override_backend_ownership(backend, reserved):
    with pytest.raises(ValueError, match="cannot override backend arguments"):
        execute(backend, trainer_kwargs={reserved: None})
    assert backend[2] == backend[3] == []


def test_managed_cache_applied_before_model_access(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_CACHE_DIR", str(tmp_path))
    for name in ["XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"]:
        monkeypatch.delenv(name, raising=False)
    assert configure_managed_hf_cache() == str(tmp_path / "huggingface")
    assert (tmp_path / "huggingface/hub").is_dir()
    assert (tmp_path / "huggingface/datasets").is_dir()
    assert (tmp_path / "xdg").is_dir()
