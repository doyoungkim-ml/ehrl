"""Shared Ray/VeRL setup for algorithms implemented as agent PPO trainers.

Algorithms supply a trainer class and constructor arguments; benchmarks supply
agents and environments. Model loading, cache setup and worker ownership stay
in the backend. Importing this module does not import Ray or VeRL.
"""

from __future__ import annotations

import importlib
import os
import socket
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from rllm.trainer.agent_trainer import AgentTrainer


def import_object(import_path: str):
    module_name, sep, attr_name = import_path.rpartition(".")
    if not sep:
        raise ValueError(f"Expected fully-qualified import path, got {import_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def startup_log(message: str) -> None:
    print(f"[startup] {message}", flush=True)


def configure_managed_hf_cache() -> str | None:
    """Pin Ray tasks to the launcher's durable HF cache.

    Ray actors do not necessarily inherit a Slurm login shell's environment.
    In particular, an actor that loses ``HF_HOME`` falls back to
    ``~/.huggingface`` (``/scratch/$USER/.huggingface`` on this cluster), whose
    inode quota is too small for model snapshots.  The GPU launcher forwards
    ``RLLM_CACHE_DIR`` explicitly, so apply its AppWorld-GRPO-style cache layout
    before the task's first Hugging Face download.
    """
    cache_root = os.environ.get("RLLM_CACHE_DIR")
    if not cache_root:
        return None

    root = Path(cache_root).expanduser()
    hf_home = root / "huggingface"
    hf_hub_cache = hf_home / "hub"
    hf_datasets_cache = hf_home / "datasets"
    xdg_cache_home = root / "xdg"
    for path in (hf_hub_cache, hf_datasets_cache, xdg_cache_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "XDG_CACHE_HOME": str(xdg_cache_home),
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_hub_cache),
            "HF_DATASETS_CACHE": str(hf_datasets_cache),
            "HUGGINGFACE_HUB_CACHE": str(hf_hub_cache),
            "TRANSFORMERS_CACHE": str(hf_hub_cache),
        }
    )
    return str(hf_home)


def apply_training_runtime_patches() -> None:
    """Apply the maintained backend repairs before initializing trainer workers."""
    from rllm.experimental.verl.patch import (
        patch_verl_dynamic_batch_sync,
        patch_verl_training_output_postprocess,
    )

    patch_verl_dynamic_batch_sync()
    patch_verl_training_output_postprocess()


def run_agent_task(
    runner,
    *,
    config,
    trainer_class_path: str,
    default_agent_class_path: str,
    default_env_class_path: str,
    agent_class=None,
    env_class=None,
    agent_args: dict[str, Any] | None = None,
    env_args: dict[str, Any] | None = None,
    trainer_kwargs: dict[str, Any] | None = None,
    log_prefix: str = "AgentTaskRunner",
):
    """Construct and run an agent trainer using the caller's worker layout.

    ``trainer_kwargs`` holds algorithm-specific constructor arguments. Backend
    arguments cannot be overridden through it. Hydra agent/environment values
    take precedence over the benchmark defaults, without mutating the caller.
    This entrypoint supports agent/environment training; workflow and SDK
    training continue to use their existing TaskRunner paths.
    """
    trainer_kwargs = dict(trainer_kwargs or {})
    reserved = {
        "config",
        "tokenizer",
        "role_worker_mapping",
        "resource_pool_manager",
        "ray_worker_group_cls",
        "reward_fn",
        "val_reward_fn",
        "env_class",
        "agent_class",
        "env_args",
        "agent_args",
    }
    if overlap := reserved.intersection(trainer_kwargs):
        raise ValueError(f"trainer_kwargs cannot override backend arguments: {sorted(overlap)}")

    from pprint import pprint

    from verl.trainer.ppo.reward import load_reward_manager
    from verl.utils.fs import copy_to_local

    startup_log(f"{log_prefix} hostname={socket.gethostname()} pid={os.getpid()}")
    hf_home = configure_managed_hf_cache()
    if hf_home is not None:
        startup_log(f"using managed Hugging Face cache: {hf_home}")
    try:
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
    except ValueError:
        # Resolver may already be registered by another trainer in the same process.
        pass
    OmegaConf.resolve(config)
    pprint(OmegaConf.to_container(config))

    startup_log("copying model checkpoint to local storage")
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )

    startup_log("loading tokenizer")
    from verl.utils import hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    startup_log("building resource pool and worker role mapping")
    actor_rollout_cls, ray_worker_group_cls = runner.add_actor_rollout_worker(config)
    runner.add_critic_worker(config)
    runner.add_ref_policy_worker(config, actor_rollout_cls)

    reward_fn = load_reward_manager(
        config,
        tokenizer,
        **config.reward_model.get("reward_kwargs", {}),
    )
    val_reward_fn = load_reward_manager(
        config,
        tokenizer,
        **config.reward_model.get("reward_kwargs", {}),
    )
    resource_pool_manager = runner.init_resource_pool_mgr(config)

    if env_class is None:
        env_class = import_object(default_env_class_path)
    if agent_class is None:
        agent_class = import_object(default_agent_class_path)

    resolved_env_args = dict(env_args or {})
    resolved_agent_args = dict(agent_args or {})
    if config.rllm.env.get("env_args") is not None:
        resolved_env_args.update(config.rllm.env.get("env_args"))
    if config.rllm.agent.get("agent_args") is not None:
        resolved_agent_args.update(config.rllm.agent.get("agent_args"))

    trainer_cls = import_object(trainer_class_path)
    startup_log(f"constructing trainer {trainer_class_path}")
    trainer = trainer_cls(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=runner.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        env_class=env_class,
        agent_class=agent_class,
        env_args=resolved_env_args,
        agent_args=resolved_agent_args,
        **trainer_kwargs,
    )

    try:
        apply_training_runtime_patches()
        startup_log("initializing trainer workers")
        trainer.init_workers()
        startup_log("starting training loop")
        trainer.fit_agent()
    finally:
        startup_log("shutting down trainer")
        trainer.shutdown()


class RayAgentTrainer(AgentTrainer):
    """AgentTrainer with an explicitly supplied Ray actor class.

    The actor implements ``run(config, agent_class, env_class, agent_args,
    env_args)`` and typically delegates to :func:`run_agent_task`.
    """

    def __init__(self, *args, task_runner_cls, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_runner_cls = task_runner_cls

    def _train_verl(self):
        import ray

        from rllm.trainer.ray_init_utils import get_ray_init_settings
        from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

        if not ray.is_initialized():
            ray_init_settings = get_ray_init_settings(self.config)
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

        runner = self.task_runner_cls.remote()
        ray.get(
            runner.run.remote(
                config=self.config,
                agent_class=self.agent_class,
                env_class=self.env_class,
                agent_args=self.agent_args,
                env_args=self.env_args,
            )
        )
