"""Execute benchmark controller routing without loading distributed GPU workers."""

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
JOINT_PPO_TRAINER = "rllm.algorithms.ppo.trainer_joint.LocalACEJointPPOTrainer"
DELAYED_JOINT_TRAINER = "rllm.algorithms.ace.trainer_delayed_joint.LocalACERLDelayedJointAgentPPOTrainer"


@pytest.mark.parametrize("benchmark", ("appworld", "mbpp", "mastermind", "mini_swe_bench"))
def test_benchmark_dispatches_joint_ppo_to_dedicated_controller(benchmark, monkeypatch):
    """Run the real mode resolver and Ray task's run method with a capture sink."""
    path = ROOT / f"examples/{benchmark}/train/ace_rl/training_entrypoints.py"
    if not path.exists():
        path = path.with_name("train.py")
    tree = ast.parse(path.read_text())
    constants = {
        "TRAINERS", "STANDARD_TRAINER", "ITERATIVE_TRAINER", "DELAYED_JOINT_TRAINER", "JOINT_PPO_TRAINER",
        "APPWORLD_AGENT", "APPWORLD_ENV", "AGENT_CLASS", "ENV_CLASS",
    }
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in {"get_trainer_class_path", "_resolve_trainer_class_path"})
    ]
    # MiniSWE defines its Ray task locally so that importing the binding stays
    # lightweight; other benchmarks define their tasks at module level.
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    for node in classes:
        node.decorator_list = []
    nodes.extend(classes)
    captured = []
    namespace = {
        "TaskRunner": object, "OmegaConf": OmegaConf, "ACE_DEFAULTS": {},
        "run_local_ace_rl_task": lambda runner, **kwargs: captured.append(kwargs),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    config = OmegaConf.create({"rllm": {"ace": {"training_mode": "joint_ppo"}}})
    if benchmark == "appworld":
        main_path = path.with_name("train.py")
        main = next(node for node in ast.parse(main_path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main")
        main.decorator_list = []
        selected = []
        namespace["train_appworld_local_ace_rl"] = lambda config, **kwargs: selected.append(kwargs["task_runner_cls"])
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(main_path), "exec"), namespace)
        namespace["main"](config)
        runner_cls = selected[-1]
        assert runner_cls.__name__ == "AppWorldLocalACEJointPPOTaskRunner"
        config.rllm.ace.training_mode = "delayed_joint"
        namespace["main"](config)
        assert selected[-1].__name__ == "AppWorldLocalACERLDelayedJointTaskRunner"
        config.rllm.ace.training_mode = "joint_ppo"
    else:
        resolver = namespace["_resolve_trainer_class_path" if benchmark == "mini_swe_bench" else "get_trainer_class_path"]
        assert resolver(config) == JOINT_PPO_TRAINER
        config.rllm.ace.training_mode = "delayed_joint"
        assert resolver(config) == DELAYED_JOINT_TRAINER
        config.rllm.ace.training_mode = "joint_ppo"
        runner_cls = namespace[classes[0].name]
    runner_cls().run(config, agent_args={"marker": "preserve"}, env_args={"marker": "preserve"})
    assert captured[-1]["trainer_class_path"] == JOINT_PPO_TRAINER
    assert captured[-1]["config"] is config
    assert captured[-1]["agent_args"] == {"marker": "preserve"}
    assert captured[-1]["env_args"] == {"marker": "preserve"}
