"""ACE configuration binding for the algorithm-independent VeRL task runner."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from rllm.trainer.verl.task import RayAgentTrainer, run_agent_task

LocalACERLAgentTrainer = RayAgentTrainer


def _initialize_controller_seed(seed: int | None) -> None:
    """Seed only this Ray trainer process; checkpoint RNG restoration wins later."""
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 32):
        raise ValueError("ace.controller_seed must be null or an integer in [0, 2**32 - 1]")

    import random

    import numpy as np
    import torch

    random.seed(seed)
    # Use NumPy's common seed domain, and do not touch CUDA/worker determinism.
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def run_local_ace_rl_task(
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
    ace_defaults: dict[str, Any] | None = None,
    log_prefix: str = "LocalACERLTaskRunner",
):
    """Bind benchmark defaults and explicit overrides to an ACE trainer."""
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
    ace_config = {"controller_seed": None, **dict(ace_defaults or {})}
    if hasattr(config, "rllm") and config.rllm.get("ace") is not None:
        ace_config.update(OmegaConf.to_container(config.rllm.ace, resolve=True))
    # The old entrypoints treated empty asset paths as requests for defaults.
    for name in ("prompts_dir", "initial_playbook", "reflector_adapter_class"):
        if not ace_config.get(name) and ace_defaults and name in ace_defaults:
            ace_config[name] = ace_defaults[name]
    # Record the same resolved defaults used by the trainer, so the maintained
    # run's resume guard and logs see its actual scientific configuration.
    OmegaConf.update(config, "rllm.ace", ace_config, merge=False, force_add=True)
    _initialize_controller_seed(ace_config["controller_seed"])
    return run_agent_task(
        runner,
        config=config,
        trainer_class_path=trainer_class_path,
        default_agent_class_path=default_agent_class_path,
        default_env_class_path=default_env_class_path,
        agent_class=agent_class,
        env_class=env_class,
        agent_args=agent_args,
        env_args=env_args,
        trainer_kwargs={"ace_config": ace_config},
        log_prefix=log_prefix,
    )
