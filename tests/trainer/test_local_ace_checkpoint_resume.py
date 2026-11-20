"""CPU-only checkpoint tests: real Playbook and AST-isolated trainer hooks.

Heavy Ray/VERL imports are deliberately not replaced globally; only selected real
trainer method bodies are executed against a small parent checkpoint adapter.
These tests do not claim to exercise distributed PPO training.
"""

import ast
import importlib.util
import json
import logging
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "rllm/algorithms/ace/checkpoint.py"
STANDARD = ROOT / "rllm/algorithms/ace/trainer.py"


def load_playbook():
    package = types.ModuleType("_ace_checkpoint_test_playbook")
    package.__path__ = [str(ROOT / "rllm/algorithms/ace/context")]
    sys.modules[package.__name__] = package
    name = package.__name__ + ".playbook"
    spec = importlib.util.spec_from_file_location(name, ROOT / "rllm/algorithms/ace/context/playbook.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Playbook


Playbook = load_playbook()


@pytest.fixture(autouse=True)
def preserve_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def helper():
    assert HELPER.exists(), "ACE checkpoint preservation helper is missing"
    spec = importlib.util.spec_from_file_location("_ace_checkpoint_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trainer():
    return types.SimpleNamespace(
        global_steps=5,
        _training_epoch=0,
        initial_playbook_text="initial playbook",
        playbook=Playbook("last task playbook"),
        local_train_playbooks={"task/a": Playbook("## OTHERS\n[misc-00003] learned α")},
        local_eval_playbooks={"eval": Playbook("discard eval-only cache")},
        local_train_task_seen_counts={"task/a": 2, "unmaterialized": 1},
        _current_task_seen_counts={},
        _current_task_flush_info={},
        flush_playbook_every_n_steps=3,
        flush_playbook_every_n_epochs=2,
        _last_playbook_flush_step=0,
        _steps_per_epoch=5,
    )


def test_sidecar_without_actual_data_epoch_is_rejected(tmp_path):
    state = helper()
    original = trainer()
    original._training_epoch = 2
    state.save_ace_state(original, tmp_path)
    path = tmp_path / "ace_state.json"
    document = json.loads(path.read_text())
    document.pop("_training_epoch")
    path.write_text(json.dumps(document))
    resumed = trainer()
    resumed.local_train_task_seen_counts = {"untouched": 9}
    with pytest.raises(ValueError, match="_training_epoch"):
        state.load_ace_state(resumed, tmp_path)
    assert resumed.local_train_task_seen_counts == {"untouched": 9}


def test_save_without_actual_data_epoch_is_rejected(tmp_path):
    original = trainer()
    if hasattr(original, "_training_epoch"):
        del original._training_epoch
    with pytest.raises(ValueError, match="_training_epoch"):
        helper().save_ace_state(original, tmp_path)
    assert not (tmp_path / "ace_state.json").exists()


@pytest.mark.parametrize("saved_gate", [None, "none", "strict_improvement"])
@pytest.mark.parametrize("current_gate", ["none", "strict_improvement"])
def test_auxiliary_objective_must_match_before_restoring_controller(tmp_path, saved_gate, current_gate):
    state = helper()
    original = trainer()
    original.auxiliary_update_gate = saved_gate or "none"
    state.save_ace_state(original, tmp_path)
    path = tmp_path / "ace_state.json"
    if saved_gate is None:
        document = json.loads(path.read_text())
        document.pop("auxiliary_update_gate")
        path.write_text(json.dumps(document))
    resumed = trainer()
    resumed.auxiliary_update_gate = current_gate
    resumed.local_train_task_seen_counts = {"untouched": 9}
    if (saved_gate or "none") == current_gate:
        assert state.load_ace_state(resumed, tmp_path)
        assert resumed.local_train_task_seen_counts == original.local_train_task_seen_counts
    else:
        with pytest.raises(ValueError, match="incompatible auxiliary_update_gate"):
            state.load_ace_state(resumed, tmp_path)
        assert resumed.local_train_task_seen_counts == {"untouched": 9}


def test_strict_auxiliary_gate_rejects_missing_controller_sidecar(tmp_path):
    resumed = trainer()
    resumed.auxiliary_update_gate = "strict_improvement"
    resumed.local_train_task_seen_counts = {"untouched": 9}
    with pytest.raises(ValueError, match="missing ACE state cannot establish the auxiliary objective"):
        helper().load_ace_state(resumed, tmp_path)
    assert resumed.local_train_task_seen_counts == {"untouched": 9}


def test_roundtrip_preserves_task_playbooks_counts_and_next_bullet_id(tmp_path):
    state = helper()
    original = trainer()
    # Removed high IDs cannot be recovered from the remaining playbook text.
    original.local_train_playbooks["task/a"].next_global_id = 19
    # Explicit mode must survive even when text-based auto-detection disagrees.
    original.playbook = Playbook("global without headers", flat_mode=False)
    original.playbook.next_global_id = 42
    original._last_playbook_flush_step = 3
    state.save_ace_state(original, tmp_path)
    document = json.loads((tmp_path / "ace_state.json").read_text())
    assert document["schema_version"] == 1
    resumed = trainer()
    resumed.local_train_playbooks = {}
    resumed.local_train_task_seen_counts = {}
    state.load_ace_state(resumed, tmp_path)
    assert resumed.local_train_task_seen_counts == original.local_train_task_seen_counts
    book = resumed.local_train_playbooks["task/a"]
    assert book.to_string() == original.local_train_playbooks["task/a"].to_string()
    assert book.flat_mode is False
    assert book.next_global_id == 19
    operation = {"operations": [{"type": "ADD", "section": "others", "content": "next bullet"}]}
    book.apply_operations(operation)
    original.local_train_playbooks["task/a"].apply_operations(operation)
    assert book.to_string() == original.local_train_playbooks["task/a"].to_string()
    assert resumed.playbook.to_string() == original.playbook.to_string()
    assert resumed.playbook.flat_mode is False
    assert resumed.playbook.next_global_id == 42
    assert resumed.initial_playbook_text == original.initial_playbook_text
    assert resumed._last_playbook_flush_step == 3
    assert resumed._steps_per_epoch == 5
    assert resumed.flush_playbook_every_n_steps == 3
    assert resumed.flush_playbook_every_n_epochs == 2
    assert resumed.local_eval_playbooks == {}


def isolated_class(path, name, methods, namespace, bases=None):
    """Compile unchanged production methods without importing Ray/VERL."""
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
    if bases is not None:
        node.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def local_class(parent=object):
    namespace = {"ACERLAgentPPOTrainer": parent, "Playbook": Playbook, "logger": logging.getLogger(__name__), "os": __import__("os")}
    return isolated_class(
        STANDARD,
        "LocalACERLAgentPPOTrainer",
        {
            "_get_task_playbook",
            "_prepare_task_playbooks_for_step",
            "_capture_component_schedule_state",
            "_restore_component_schedule_state",
            "_save_checkpoint",
            "_load_checkpoint",
            "_iter_training_epoch",
        },
        namespace,
    )


def test_interrupted_matches_uninterrupted_next_task_flush_schedule_and_rng(tmp_path):
    state = helper()
    local = local_class()
    iterative = isolated_class(
        ROOT / "rllm/algorithms/ace/trainer_iterative.py",
        "LocalACERLIterativeAgentPPOTrainer",
        {"_get_active_component", "_advance_component_schedule"},
        {"LocalACERLAgentPPOTrainer": local},
    )

    def fresh():
        instance = iterative()
        instance.__dict__.update(trainer().__dict__)
        instance.iterative_schedule = ["policy", "reflector", "curator"]
        instance.iterative_steps_per_role = 2
        instance._iter_role_index = 1
        instance._iter_role_step = 1
        instance._training_epoch = 4
        return instance

    def step(instance, task=None):
        if task is None:
            task = random.choice(["task/a", "unseen", "unmaterialized"])
        instance.global_steps += 1
        flushed = instance._prepare_task_playbooks_for_step([task], is_eval=False)
        before = instance._get_task_playbook(task, is_eval=False).to_string()
        role = instance._get_active_component()
        decision = (
            random.randrange(100000),
            random.gauss(0, 1),
            float(np.random.random()),
            float(np.random.normal()),
            torch.rand(3).tolist(),
        )
        instance.local_train_playbooks[task] = Playbook(before + f"\n{role}:{decision}")
        instance._advance_component_schedule()
        return task, before, flushed, role, decision

    original = fresh()
    random.seed(82)
    np.random.seed(83)
    torch.manual_seed(84)
    random.gauss(0, 1)  # Keep cached Gaussian state, not just integer seeds.
    np.random.normal()
    state.save_ace_state(original, tmp_path)
    expected = [step(original, "task/a")] + [step(original) for _ in range(12)]
    resumed = fresh()
    resumed._iter_role_index = resumed._iter_role_step = 0
    resumed._training_epoch = 0
    resumed.local_train_playbooks = {}
    resumed.local_train_task_seen_counts = {}
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    state.load_ace_state(resumed, tmp_path)
    assert resumed._training_epoch == 4
    actual = [step(resumed, "task/a")] + [step(resumed) for _ in range(12)]
    assert actual == expected
    assert actual[0][2]["task/a"]["task_seen_count"] == 3
    assert actual[0][1] == "initial playbook"
    assert resumed.local_train_task_seen_counts == original.local_train_task_seen_counts
    assert resumed._capture_component_schedule_state() == original._capture_component_schedule_state()


def test_load_clears_transient_eval_caches_with_explicit_cold_fallback(tmp_path, caplog):
    state = helper()
    resumed = trainer()
    resumed._local_eval_policy_trajectory_cache = {"eval": ["trajectory"]}
    resumed._current_validation_policy_tasks = ["eval"]
    resumed._current_validation_policy_sources = ["eval-source"]
    resumed._current_local_eval_epoch = 2
    resumed._in_local_eval = True
    resumed._current_task_seen_counts = {"eval": 7}
    resumed._current_task_flush_info = {"eval": {"flushed": True}}
    resumed._last_playbook_flush_step = 3
    rng_before = snapshot(resumed)[1:]
    with caplog.at_level(logging.WARNING):
        assert state.load_ace_state(resumed, tmp_path) is False
    assert resumed.local_eval_playbooks == {}
    assert resumed._local_eval_policy_trajectory_cache == {}
    assert resumed._current_validation_policy_tasks is None
    assert resumed._current_validation_policy_sources is None
    assert resumed._current_local_eval_epoch is None
    assert resumed._in_local_eval is False
    assert resumed._current_task_seen_counts == {}
    assert resumed._current_task_flush_info == {}
    assert resumed.global_steps == 5
    assert "ace_state.json" in caplog.text
    assert "cold ACE" in caplog.text
    assert resumed.local_train_playbooks == {}
    assert resumed.local_train_task_seen_counts == {}
    assert resumed.playbook.to_string() == resumed.initial_playbook_text
    assert resumed._last_playbook_flush_step == 0
    assert snapshot(resumed)[1:] == rng_before
    assert not (tmp_path / "ace_state.json").exists()


def test_atomic_save_creates_directory_before_publishing_sidecar(tmp_path, monkeypatch):
    state = helper()
    directory = tmp_path / "new" / "global_step_5"
    import os

    replace = os.replace
    fsync = os.fsync
    events = []

    def sync(fd):
        events.append("fsync")
        return fsync(fd)

    def publish(source, target):
        assert Path(source).parent == directory
        assert Path(target) == directory / "ace_state.json"
        assert json.loads(Path(source).read_text())["global_steps"] == 5
        assert events == ["fsync"]
        events.append("replace")
        return replace(source, target)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", publish)
    original = trainer()
    before = snapshot(original)
    assert state.save_ace_state(original, directory) == directory / "ace_state.json"
    assert events[:2] == ["fsync", "replace"]
    assert sorted(p.name for p in directory.iterdir()) == ["ace_state.json"]
    assert snapshot(original) == before


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_failed_atomic_save_leaves_previous_checkpoint_intact(tmp_path, monkeypatch, failure):
    import os

    state = helper()
    original = trainer()
    state.save_ace_state(original, tmp_path)
    path = tmp_path / "ace_state.json"
    previous = path.read_bytes()
    original.local_train_task_seen_counts["task/a"] = 99

    def fail(*args):
        raise OSError("simulated checkpoint I/O failure")

    monkeypatch.setattr(os, failure, fail)
    with pytest.raises(OSError, match="simulated checkpoint"):
        state.save_ace_state(original, tmp_path)
    assert path.read_bytes() == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ace_state.json"]


@pytest.mark.parametrize("raise_during_validation", [False, True])
@pytest.mark.parametrize("has_training_epoch", [False, True])
def test_validation_context_preserves_training_state_even_on_exception(
    tmp_path,
    monkeypatch,
    raise_during_validation,
    has_training_epoch,
):
    state = helper()
    original = trainer()
    original.iterative_schedule = ["policy", "reflector", "curator"]
    original.iterative_steps_per_role = 2
    original._iter_role_index = original._iter_role_step = 1
    if has_training_epoch:
        original._training_epoch = 4
    else:
        del original._training_epoch
    original.local_eval_playbooks = {}
    original._local_eval_policy_trajectory_cache = {}
    original._current_task_seen_counts = {"task/a": 2}
    original._current_task_flush_info = {"task/a": {"flushed": False}}
    original.local_train_playbooks["task/a"].next_global_id = 19
    before = snapshot(original)

    def no_disk(*args, **kwargs):
        raise AssertionError("validation preservation must stay in memory")

    monkeypatch.setattr(state, "save_ace_state", no_disk)
    monkeypatch.setattr(state.tempfile, "NamedTemporaryFile", no_disk)
    try:
        with state.preserve_ace_training_state(original):
            original.local_train_playbooks["task/a"].content = "in-place mutation"
            original.local_train_playbooks["task/a"].next_global_id = 99
            original.local_train_playbooks["eval"] = Playbook("leak")
            original.local_train_task_seen_counts["task/a"] = 100
            original.playbook = Playbook("evaluation playbook")
            original.initial_playbook_text = "wrong template"
            original.iterative_schedule.reverse()
            original._iter_role_index = original._iter_role_step = 0
            original._training_epoch = 99
            original.global_steps += 1
            original._last_playbook_flush_step = 6
            original.flush_playbook_every_n_steps = 9
            original._current_task_seen_counts = {"eval": 1}
            original._current_task_flush_info = {}
            original.local_eval_playbooks["eval"] = Playbook("must be discarded")
            original._local_eval_policy_trajectory_cache["eval"] = ["must be discarded"]
            random.random()
            np.random.random(5)
            torch.rand(5)
            if raise_during_validation:
                raise RuntimeError("validation failed")
    except RuntimeError as exc:
        assert raise_during_validation and str(exc) == "validation failed"
    else:
        assert not raise_during_validation
    assert snapshot(original) == before
    assert list(tmp_path.iterdir()) == []


def test_lazy_epoch_length_is_initialized_when_loading(tmp_path):
    state = helper()
    state.save_ace_state(trainer(), tmp_path)
    resumed = trainer()
    resumed._steps_per_epoch = 0  # Production initialization, before first epoch estimate.
    resumed.train_dataloader = range(5)
    assert state.load_ace_state(resumed, tmp_path) is True
    assert resumed._steps_per_epoch == 5
    assert resumed.local_train_task_seen_counts == trainer().local_train_task_seen_counts


@pytest.mark.parametrize("invalid", ["playbook-id", "task-key", "training-epoch"])
def test_invalid_live_state_does_not_publish_unloadable_checkpoint(tmp_path, invalid):
    state = helper()
    original = trainer()
    state.save_ace_state(original, tmp_path)
    path = tmp_path / "ace_state.json"
    previous = path.read_bytes()
    if invalid == "playbook-id":
        original.playbook.next_global_id = 0
    elif invalid == "task-key":
        original.local_train_task_seen_counts = {1: 2}  # JSON must not silently stringify IDs.
    else:
        original._training_epoch = -1
    before = snapshot(original)
    with pytest.raises(ValueError, match="ACE"):
        state.save_ace_state(original, tmp_path)
    assert path.read_bytes() == previous
    assert snapshot(original) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ace_state.json"]


def test_missing_legacy_state_resets_schedule_not_parent_data_epoch(tmp_path, caplog):
    state = helper()
    resumed = trainer()
    resumed.iterative_schedule = ["policy", "reflector", "curator"]
    resumed.iterative_steps_per_role = 2
    resumed._iter_role_index = resumed._iter_role_step = 1
    resumed._training_epoch = 4
    with caplog.at_level(logging.WARNING):
        assert state.load_ace_state(resumed, tmp_path) is False
    assert resumed._iter_role_index == resumed._iter_role_step == 0
    assert resumed._training_epoch == 4
    assert resumed.iterative_schedule == ["policy", "reflector", "curator"]
    assert resumed.iterative_steps_per_role == 2


@pytest.mark.parametrize("raises", [False, True])
def test_preserve_controller_rng_is_lightweight_and_exception_safe(monkeypatch, raises):
    state = helper()
    before = snapshot(trainer())[1:]

    def no_training_snapshot(*args):
        raise AssertionError("RNG-only preservation must not copy playbooks")

    monkeypatch.setattr(state, "_capture_state", no_training_snapshot)

    def consume():
        with state.preserve_controller_rng():
            random.random()
            np.random.normal(0, 1, 5)
            torch.rand(5)
            assert snapshot(trainer())[1:] != before
            if raises:
                raise RuntimeError("iterator failed")

    if raises:
        with pytest.raises(RuntimeError, match="iterator failed"):
            consume()
    else:
        consume()
    assert snapshot(trainer())[1:] == before


def snapshot(instance):
    """Compare values as well as identities without checkpoint implementation code."""
    numpy_state = np.random.get_state()
    return (
        json.dumps(vars(instance), default=vars, sort_keys=True),
        random.getstate(),
        (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        torch.get_rng_state().tolist(),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        (("schema_version",), 2),
        (("schema_version",), True),
        (("global_steps",), 6),
        (("global_steps",), True),
        (("initial_playbook_text",), {}),
        (("playbook", "content"), []),
        (("playbook", "flat_mode"), 1),
        (("playbook", "next_global_id"), 0),
        (("local_train_playbooks", "task/a", "next_global_id"), 3),
        (("local_train_task_seen_counts", "task/a"), -1),
        (("local_train_task_seen_counts", "task/a"), True),
        (("flush_config", "flush_playbook_every_n_steps"), 4),
        (("flush_config", "flush_playbook_every_n_epochs"), 3),
        (("flush_config", "_steps_per_epoch"), 6),
        (("_last_playbook_flush_step",), 6),
        (("_training_epoch",), -1),
        (("component_schedule",), {}),
        (("rng", "python"), [0]),
        (("rng", "numpy", 1), [-1] * 624),
        (("rng", "numpy", 2), 625),
        (("rng", "torch_cpu"), []),
        (("rng", "torch_cpu", 0), 256),
    ],
)
def test_invalid_state_fails_before_trainer_or_rng_mutation(tmp_path, field, value):
    state = helper()
    state.save_ace_state(trainer(), tmp_path)
    path = tmp_path / "ace_state.json"
    document = json.loads(path.read_text())
    target = document
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    path.write_text(json.dumps(document))
    resumed = trainer()
    resumed.playbook = Playbook("must not be replaced")
    resumed.local_train_playbooks = {"private": Playbook("must survive")}
    resumed.local_train_task_seen_counts = {"private": 9}
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    before = snapshot(resumed)
    old_book = resumed.playbook
    with pytest.raises(ValueError, match="ACE"):
        state.load_ace_state(resumed, tmp_path)
    assert snapshot(resumed) == before
    assert resumed.playbook is old_book


@pytest.mark.parametrize(
    "contents",
    [
        '{"schema_version":',
        "null",
        "[]",
        "{}",
        '{"schema_version":1,"schema_version":2}',
        '{"schema_version":NaN}',
    ],
)
def test_unreadable_schema_fails_without_mutation(tmp_path, contents):
    state = helper()
    (tmp_path / "ace_state.json").write_text(contents)
    resumed = trainer()
    before = snapshot(resumed)
    with pytest.raises(ValueError, match="ACE"):
        state.load_ace_state(resumed, tmp_path)
    assert snapshot(resumed) == before
