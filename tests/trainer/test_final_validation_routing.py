"""Execute final validation blocks to verify standalone-evaluation opt-out."""

import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ["agent_ppo_trainer.py", "agent_workflow_trainer.py"])
@pytest.mark.parametrize("setting,enabled", [(None, True), (True, True), (False, False)])
def test_final_validation_defaults_to_on_and_can_be_disabled(filename, setting, enabled):
    path = ROOT / "rllm/trainer/verl" / filename
    tree = ast.parse(path.read_text())
    block = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and "val_after_train" in ast.unparse(node.test)
    )
    calls = []
    validation = {"val/reward": 0.5}
    settings = {} if setting is None else {"val_after_train": setting}
    instance = SimpleNamespace(
        val_reward_fn=object(), global_steps=23, config=SimpleNamespace(trainer=settings),
        agent_execution_engine=SimpleNamespace(set_training_step=lambda *a, **kw: calls.append(("mode", a, kw))),
        _validate_agent=lambda **kwargs: (calls.append(("validate", kwargs)) or validation),
    )
    logger = SimpleNamespace(log=lambda **kw: calls.append(("log", kw)))
    ns = {"self": instance, "logger": logger, "epoch": 4, "pprint": lambda *args: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(block)], type_ignores=[])), str(path), "exec"), ns)
    assert sum(event[0] == "validate" for event in calls) == int(enabled)
    assert sum(event[0] == "log" for event in calls) == int(enabled)
    if not enabled:
        assert calls == []
    else:
        assert ("log", {"data": validation, "step": 23}) in calls


@pytest.mark.parametrize("filename", ["agent_ppo_trainer.py", "agent_workflow_trainer.py"])
@pytest.mark.parametrize("step,total,frequency,save_at_end,expected", [
    (8, 10, 5, True, False),
    (9, 10, 5, True, True),
    (9, 10, 5, None, False),
    (10, 11, 5, True, True),
    (10, 12, 5, None, True),
    (9, 10, 0, True, True),
])
def test_end_checkpoint_opt_in_saves_final_update_once(filename, step, total, frequency, save_at_end, expected):
    path = ROOT / "rllm/trainer/verl" / filename
    tree = ast.parse(path.read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If) and "save_at_end" in ast.unparse(node.test))
    saved = []
    settings = {} if save_at_end is None else {"save_at_end": save_at_end}
    instance = SimpleNamespace(
        global_steps=step, total_training_steps=total,
        config=SimpleNamespace(trainer=SimpleNamespace(save_freq=frequency, get=settings.get)),
        _save_checkpoint=lambda: saved.append(step),
    )
    ns = {"self": instance, "timing_raw": {}, "marked_timer": lambda *args, **kwargs: nullcontext()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(block)], type_ignores=[])), str(path), "exec"), ns)
    assert saved == ([step] if expected else [])
