from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _get_class_function(tree: ast.AST, class_name: str, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == function_name:
                    return item
    raise AssertionError(f"Could not find {class_name}.{function_name}")


def test_agent_ppo_trainer_keeps_phase_scope_and_kwargs_support():
    tree = ast.parse((ROOT / "rllm/trainer/verl/agent_ppo_trainer.py").read_text())

    _get_class_function(tree, "AgentPPOTrainer", "_phase_scope")
    _get_class_function(tree, "AgentPPOTrainer", "_build_no_padding_batch_td")
    generate_fn = _get_class_function(tree, "AgentPPOTrainer", "generate_agent_trajectories_async")

    assert generate_fn.args.kwarg is not None
    assert generate_fn.args.kwarg.arg == "trajectory_generator_kwargs"


def test_agent_ppo_trainer_keeps_temporary_padding_for_no_padding_log_prob():
    text = (ROOT / "rllm/trainer/verl/agent_ppo_trainer.py").read_text()

    assert "batch_td, _ = self._build_no_padding_batch_td(" in text
    assert 'metric_prefix="old_log_prob"' in text
    assert "self._slice_padded_batch_tensor(log_probs, original_batch_size)" in text
    assert "self._slice_padded_batch_tensor(ref_lp, original_batch_size)" in text


def test_agent_ppo_trainer_pads_update_actor_batches_to_mini_batch_size():
    text = (ROOT / "rllm/trainer/verl/agent_ppo_trainer.py").read_text()

    assert 'metric_prefix="update_actor"' in text
    assert "size_divisor=ppo_mini_batch_size" in text
    assert 'metrics[f"batch/{key}_pad_divisor"] = divisor' in text


def test_agent_execution_engine_trajectory_generator_keeps_rollout_state_flag():
    tree = ast.parse((ROOT / "rllm/engine/agent_execution_engine.py").read_text())
    trajectory_fn = _get_class_function(tree, "AgentExecutionEngine", "trajectory_generator")

    arg_names = [arg.arg for arg in trajectory_fn.args.args]
    assert "manage_rollout_engine_state" in arg_names
