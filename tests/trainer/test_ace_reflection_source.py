"""Singlemode reflection-source choice, executed from the real trainer source.

The trainer module imports VeRL/Ray/GPU workers, so the CPU gate executes the
production helper functions and method extracted from ``trainer_base.py``.
"""

import ast
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "rllm/algorithms/ace/trainer_base.py"
TREE = ast.parse(SOURCE.read_text())
HELPERS = {"_VALID_REFLECTION_SOURCE_SELECTIONS", "_coerce_reflection_source_selection", "_reflection_source_reward"}


def _production_namespace():
    nodes = [
        node
        for node in TREE.body
        if (isinstance(node, ast.FunctionDef) and node.name in HELPERS)
        or (isinstance(node, ast.Assign) and any(getattr(target, "id", None) in HELPERS for target in node.targets))
    ]
    trainer = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "ACERLAgentPPOTrainer")
    nodes.append(next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_select_reflection_source_indices"))
    namespace = {"random": random}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace


NS = _production_namespace()


def select(task_group, *, singlemode=True, selection="min_reward"):
    trainer = SimpleNamespace(singlemode_enabled=singlemode, reflection_source_selection=selection)
    return NS["_select_reflection_source_indices"](trainer, task_group)


def rollouts(*rewards):
    return [{"idx": index, "trajectory_reward": reward} for index, reward in enumerate(rewards)]


def test_unique_minimum_reward_rollout_is_the_source():
    assert select(rollouts(1.0, 0.25, 0.5, 1.0)) == [1]
    assert select(rollouts(-0.5, 0.0, 2.0)) == [0]


def test_ties_choose_only_among_minimum_rollouts(monkeypatch):
    offered = []

    def choice(candidates):
        offered.append(list(candidates))
        return candidates[-1]

    monkeypatch.setattr(random, "choice", choice)
    assert select(rollouts(0.0, 1.0, 0.0, 0.5, 0.0)) == [4]
    assert offered == [[0, 2, 4]]


def test_ties_are_broken_uniformly_among_minima():
    random.seed(0)
    chosen = {select(rollouts(1.0, 0.0, 1.0, 0.0))[0] for _ in range(200)}
    assert chosen == {1, 3}


def test_missing_or_null_reward_counts_as_zero():
    assert select([{"trajectory_reward": 0.5}, {}, {"trajectory_reward": 0.1}]) == [1]
    assert select([{"trajectory_reward": 0.5}, {"trajectory_reward": None}, {"trajectory_reward": -1.0}]) == [2]


def test_non_singlemode_reflects_on_every_rollout():
    assert select(rollouts(1.0, 0.0, 0.5), singlemode=False) == [0, 1, 2]
    assert select(rollouts(1.0, 0.0, 0.5), singlemode=False, selection="random") == [0, 1, 2]


def test_empty_task_group_has_no_source():
    assert select([]) == []


def test_random_mode_keeps_uniform_index_draw(monkeypatch):
    calls = []
    monkeypatch.setattr(random, "randrange", lambda upper: calls.append(upper) or 2)
    monkeypatch.setattr(random, "choice", lambda candidates: pytest.fail("random mode must not rank rewards"))
    assert select(rollouts(0.0, 1.0, 1.0), selection="random") == [2]
    assert calls == [3]


@pytest.mark.parametrize("value,expected", [(None, "random"), ("min_reward", "min_reward"), (" Random ", "random")])
def test_selection_values_are_normalized(value, expected):
    assert NS["_coerce_reflection_source_selection"](value) == expected


@pytest.mark.parametrize("value", ["max_reward", "", "uniform", 1])
def test_invalid_selection_is_rejected(value):
    with pytest.raises(ValueError, match="reflection_source_selection"):
        NS["_coerce_reflection_source_selection"](value)


def test_trainer_reads_and_validates_the_config_key_with_random_default():
    trainer = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "ACERLAgentPPOTrainer")
    setup = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_init_ace_components")
    expected = "self.reflection_source_selection = _coerce_reflection_source_selection(ace.get('reflection_source_selection', 'random'))"
    assert expected in ast.unparse(setup)
