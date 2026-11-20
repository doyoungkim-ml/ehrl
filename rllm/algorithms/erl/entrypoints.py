"""Benchmark-independent backend binding for original ERL."""

from omegaconf import OmegaConf


def train_original_erl(config, *, agent_class, env_class, train_dataset=None, val_dataset=None, agent_args=None, env_args=None, adapter_module=None):
    import ray

    from rllm.trainer.verl.task import RayAgentTrainer, run_agent_task
    from rllm.trainer.verl.train_agent_ppo import TaskRunner

    if adapter_module:
        import importlib

        adapter = importlib.import_module(adapter_module)
        for name in ("success_adapter", "attempt_runner"):
            function = "succeeded" if name == "success_adapter" else "run_attempt"
            if hasattr(adapter, function):
                OmegaConf.update(config, f"rllm.erl.{name}", f"{adapter_module}.{function}", force_add=True)

    @ray.remote(num_cpus=1)
    class ERLTaskRunner(TaskRunner):
        def run(self, config, agent_class=None, env_class=None, agent_args=None, env_args=None):
            return run_agent_task(
                self,
                config=config,
                trainer_class_path="rllm.algorithms.erl.trainer.OriginalERLAgentPPOTrainer",
                default_agent_class_path=f"{agent_class.__module__}.{agent_class.__name__}",
                default_env_class_path=f"{env_class.__module__}.{env_class.__name__}",
                agent_class=agent_class,
                env_class=env_class,
                agent_args=agent_args,
                env_args=env_args,
                log_prefix="OriginalERLTaskRunner",
            )

    config.rllm.workflow.use_workflow = False
    RayAgentTrainer(
        config=config,
        agent_class=agent_class,
        env_class=env_class,
        agent_args=agent_args,
        env_args=env_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_runner_cls=ERLTaskRunner,
    ).train()
