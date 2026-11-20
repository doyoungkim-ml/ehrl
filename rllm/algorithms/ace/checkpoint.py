"""JSON controller state accompanying the trainer's model/optimizer checkpoint.

The caller owns checkpoint selection and must restore ``global_steps`` before
loading ACE state. Playbooks are rebuilt using the live playbook class so this
module does not import the heavyweight rllm/VERL trainer dependency tree.
"""

import json
import logging
import math
import os
import random
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ACE_STATE_VERSION = 1
ACE_STATE_FILENAME = "ace_state.json"
logger = logging.getLogger(__name__)
_FLUSH_CONFIG_FIELDS = (
    "flush_playbook_every_n_steps",
    "flush_playbook_every_n_epochs",
    "_steps_per_epoch",
)
_SCHEDULE_FIELDS = (
    "iterative_schedule",
    "iterative_steps_per_role",
    "_iter_role_index",
    "_iter_role_step",
)


def _pack_rng():
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    return {
        "python": [python_state[0], list(python_state[1]), python_state[2]],
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch_cpu": torch.get_rng_state().tolist(),
    }


def _unpack_rng(data):
    _fields(data, {"python", "numpy", "torch_cpu"}, "rng")
    python = data["python"]
    _require(isinstance(python, list) and len(python) == 3, "rng.python")
    _require(type(python[0]) is int and python[0] == 3, "rng.python version")
    _integer_list(python[1], "rng.python state", 625, 2**32 - 1)
    _integer(python[1][-1], "rng.python position", maximum=624)
    _require(python[2] is None or _finite_number(python[2]), "rng.python Gaussian cache")
    numpy = data["numpy"]
    _require(isinstance(numpy, list) and len(numpy) == 5, "rng.numpy")
    _require(numpy[0] == "MT19937", "rng.numpy generator")
    _integer_list(numpy[1], "rng.numpy keys", 624, 2**32 - 1)
    _integer(numpy[2], "rng.numpy position", maximum=624)
    _integer(numpy[3], "rng.numpy has_gauss", maximum=1)
    _require(_finite_number(numpy[4]), "rng.numpy Gaussian cache")
    _integer_list(data["torch_cpu"], "rng.torch_cpu", None, 255)
    python_state = (data["python"][0], tuple(data["python"][1]), data["python"][2])
    numpy_state = (data["numpy"][0], np.array(data["numpy"][1], dtype=np.uint32), *data["numpy"][2:])
    torch_state = torch.tensor(data["torch_cpu"], dtype=torch.uint8)
    # Probe isolated generators: malformed RNG bytes must never reach live state.
    random.Random(0).setstate(python_state)
    np.random.RandomState(0).set_state(numpy_state)
    torch.Generator(device="cpu").set_state(torch_state)
    return python_state, numpy_state, torch_state


def _require(condition, field):
    if not condition:
        raise ValueError(f"Invalid ACE checkpoint: {field}")


def _mapping(value, field):
    _require(isinstance(value, dict) and all(isinstance(key, str) for key in value), field)


def _fields(value, required, field, optional=()):
    _mapping(value, field)
    _require(required <= value.keys() <= required | set(optional), f"{field} fields")


def _integer(value, field, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum, field)
    _require(maximum is None or value <= maximum, field)


def _integer_list(value, field, length, maximum):
    _require(isinstance(value, list) and (length is None or len(value) == length), field)
    for item in value:
        _integer(item, field, maximum=maximum)


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pack_playbook(book):
    return {
        "content": book.content,
        "flat_mode": book.flat_mode,
        "next_global_id": book.next_global_id,
    }


def _unpack_playbook(data, book_type):
    _fields(data, {"content", "flat_mode", "next_global_id"}, "playbook")
    _require(isinstance(data["content"], str), "playbook.content")
    _require(type(data["flat_mode"]) is bool, "playbook.flat_mode")
    _integer(data["next_global_id"], "playbook.next_global_id", minimum=1)
    book = book_type(data["content"], flat_mode=data["flat_mode"])
    _require(data["next_global_id"] >= book.next_global_id, "playbook.next_global_id would reuse an ID")
    book.content = data["content"]
    book.next_global_id = data["next_global_id"]
    return book


def _capture_state(trainer):
    """Capture detached JSON-compatible controller values (never eval caches)."""
    document = {
        "schema_version": ACE_STATE_VERSION,
        "global_steps": trainer.global_steps,
        "initial_playbook_text": trainer.initial_playbook_text,
        "playbook": _pack_playbook(trainer.playbook),
        "local_train_playbooks": {key: _pack_playbook(book) for key, book in trainer.local_train_playbooks.items()},
        "local_train_task_seen_counts": trainer.local_train_task_seen_counts,
        "flush_config": {name: getattr(trainer, name) for name in _FLUSH_CONFIG_FIELDS},
        "_last_playbook_flush_step": trainer._last_playbook_flush_step,
        "component_schedule": {name: getattr(trainer, name) for name in _SCHEDULE_FIELDS} if hasattr(trainer, "_iter_role_index") else None,
        "auxiliary_update_gate": getattr(trainer, "auxiliary_update_gate", "none"),
        "handoff_protocol": getattr(trainer, "handoff_protocol", "legacy"),
        "rng": _pack_rng(),
    }
    if hasattr(trainer, "_training_epoch"):
        document["_training_epoch"] = trainer._training_epoch
    return deepcopy(document)


def save_ace_state(trainer, checkpoint_dir):
    """Atomically save ACE state before the parent publishes its checkpoint."""
    document = _capture_state(trainer)
    _decode_state(trainer, document)  # Never publish a sidecar we cannot restore.
    _require("_training_epoch" in document, "missing _training_epoch in disk checkpoint")
    path = Path(checkpoint_dir) / ACE_STATE_FILENAME
    payload = json.dumps(document, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".ace_state-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _decode_state(trainer, document):
    """Validate and materialize the entire payload without mutating the trainer."""
    _fields(
        document,
        {
            "schema_version",
            "global_steps",
            "initial_playbook_text",
            "playbook",
            "local_train_playbooks",
            "local_train_task_seen_counts",
            "flush_config",
            "_last_playbook_flush_step",
            "component_schedule",
            "rng",
        },
        "root",
        optional={"_training_epoch", "auxiliary_update_gate", "handoff_protocol"},
    )
    _integer(document["schema_version"], "schema_version")
    _require(document["schema_version"] == ACE_STATE_VERSION, "unsupported schema_version")
    _integer(document["global_steps"], "global_steps")
    _require(document["global_steps"] == trainer.global_steps, "global_steps does not match parent checkpoint")
    saved_gate = document.get("auxiliary_update_gate", "none")
    _require(saved_gate in ("none", "strict_improvement"), "auxiliary_update_gate")
    current_gate = getattr(trainer, "auxiliary_update_gate", "none")
    _require(saved_gate == current_gate, "incompatible auxiliary_update_gate; changing the objective requires a separate weight-only run")
    saved_protocol = document.get("handoff_protocol", "legacy")
    _require(saved_protocol in ("legacy", "feedback_v1"), "handoff_protocol")
    _require(saved_protocol == getattr(trainer, "handoff_protocol", "legacy"), "incompatible handoff_protocol; changing experience inputs requires a separate weight-only run")
    _require(isinstance(document["initial_playbook_text"], str), "initial_playbook_text")
    config = document["flush_config"]
    _fields(config, set(_FLUSH_CONFIG_FIELDS), "flush_config")
    for name in _FLUSH_CONFIG_FIELDS:
        _require(type(config[name]) is int, f"flush_config.{name}")
        current = getattr(trainer, name)
        if name == "_steps_per_epoch":
            # This is a lazy cache, not a user-selected flush interval. A fresh
            # trainer starts at zero; compare known lengths, otherwise restore.
            if current == 0 and hasattr(trainer, "train_dataloader"):
                current = len(trainer.train_dataloader)
            if current == 0 or config[name] == 0:
                continue
        _require(config[name] == current, f"incompatible flush_config.{name}")
    _integer(config["_steps_per_epoch"], "flush_config._steps_per_epoch")
    _integer(document["_last_playbook_flush_step"], "_last_playbook_flush_step", maximum=document["global_steps"])
    _mapping(document["local_train_playbooks"], "local_train_playbooks")
    _mapping(document["local_train_task_seen_counts"], "local_train_task_seen_counts")
    for count in document["local_train_task_seen_counts"].values():
        _integer(count, "local_train_task_seen_counts")
    book_type = type(trainer.playbook)
    updates = {
        "initial_playbook_text": document["initial_playbook_text"],
        "playbook": _unpack_playbook(document["playbook"], book_type),
        "local_train_playbooks": {key: _unpack_playbook(book, book_type) for key, book in document["local_train_playbooks"].items()},
        "local_train_task_seen_counts": document["local_train_task_seen_counts"].copy(),
        "_last_playbook_flush_step": document["_last_playbook_flush_step"],
        **config,
    }
    schedule = document["component_schedule"]
    _require((schedule is not None) == hasattr(trainer, "_iter_role_index"), "incompatible component_schedule")
    if schedule is not None:
        _fields(schedule, set(_SCHEDULE_FIELDS), "component_schedule")
        roles = schedule["iterative_schedule"]
        _require(isinstance(roles, list) and bool(roles), "component_schedule.iterative_schedule")
        _require(all(role in ("policy", "reflector", "curator") for role in roles), "component_schedule roles")
        _integer(schedule["iterative_steps_per_role"], "iterative_steps_per_role", minimum=1)
        _integer(schedule["_iter_role_index"], "_iter_role_index", maximum=len(roles) - 1)
        _integer(schedule["_iter_role_step"], "_iter_role_step", maximum=schedule["iterative_steps_per_role"] - 1)
        for name in ("iterative_schedule", "iterative_steps_per_role"):
            _require(schedule[name] == getattr(trainer, name), f"incompatible component_schedule.{name}")
        updates.update(schedule)
    if "_training_epoch" in document:
        _integer(document["_training_epoch"], "_training_epoch")
        updates["_training_epoch"] = document["_training_epoch"]
    return updates, _unpack_rng(document["rng"])


def _apply_state(trainer, updates, rng):
    for name, value in updates.items():
        setattr(trainer, name, value)
    _clear_transient_state(trainer)
    _restore_rng(rng)


def _restore_rng(rng):
    python_state, numpy_state, torch_state = rng
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


@contextmanager
def preserve_controller_rng():
    """Keep local iterator bookkeeping out of controller RNG decisions.

    This CPU-only scope does not snapshot playbooks or access a trainer. Apply it
    to iterator construction on both fresh and resumed training paths.
    """
    rng = _unpack_rng(_pack_rng())
    try:
        yield
    finally:
        _restore_rng(rng)


def _clear_transient_state(trainer):
    for name in ("local_eval_playbooks", "_current_task_seen_counts", "_current_task_flush_info"):
        setattr(trainer, name, {})
    for name, factory in (
        ("_local_eval_policy_trajectory_cache", dict),
        ("_current_validation_policy_tasks", lambda: None),
        ("_current_validation_policy_sources", lambda: None),
        ("_current_local_eval_epoch", lambda: None),
        ("_in_local_eval", lambda: False),
    ):
        if hasattr(trainer, name):
            setattr(trainer, name, factory())


def _cold_start(trainer):
    trainer.playbook = type(trainer.playbook)(trainer.initial_playbook_text)
    trainer.local_train_playbooks = {}
    trainer.local_train_task_seen_counts = {}
    trainer._last_playbook_flush_step = 0
    for name in ("_iter_role_index", "_iter_role_step"):
        if hasattr(trainer, name):
            setattr(trainer, name, 0)
    _clear_transient_state(trainer)


@contextmanager
def preserve_ace_training_state(trainer):
    """Isolate validation from training, restoring even on exceptional exit.

    Use the same snapshot/validation path as checkpoints, entirely in memory.
    Current train-step diagnostics survive; evaluation caches do not. The saved
    ``_training_epoch`` is the actual data epoch supplied by the training loop,
    never an estimate derived from optimizer update counts.
    """
    document = _capture_state(trainer)
    updates, rng = _decode_state(trainer, document)
    updates["global_steps"] = document["global_steps"]
    diagnostics = {name: deepcopy(getattr(trainer, name)) for name in ("_current_task_seen_counts", "_current_task_flush_info")}
    try:
        yield
    finally:
        _apply_state(trainer, updates, rng)
        if "_training_epoch" not in document and hasattr(trainer, "_training_epoch"):
            delattr(trainer, "_training_epoch")
        for name, value in diagnostics.items():
            setattr(trainer, name, value)


def _unique_object(pairs):
    document = {}
    for key, value in pairs:
        _require(key not in document, f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _invalid_constant(value):
    raise ValueError(f"Invalid ACE checkpoint: non-finite JSON number {value}")


def load_ace_state(trainer, checkpoint_dir):
    """Restore ACE state after the parent checkpoint has restored its step.

    The saved initial template is authoritative; flush and iterative scheduling
    configuration must match. Invalid input fails before any trainer/RNG writes.
    """
    path = Path(checkpoint_dir) / ACE_STATE_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        updates, rng = _decode_state(trainer, document)
        _require("_training_epoch" in document, "missing _training_epoch in disk checkpoint")
    except FileNotFoundError:
        _require(getattr(trainer, "auxiliary_update_gate", "none") == "none", "missing ACE state cannot establish the auxiliary objective; use a verified checkpoint or a fresh weight-only run")
        _require(getattr(trainer, "handoff_protocol", "legacy") == "legacy", "missing ACE state cannot establish handoff_protocol; use a verified checkpoint or a fresh weight-only run")
        logger.warning("ACE state missing at %s; cold ACE fallback: playbooks and task visits restart from the initial template, not an exact ACE continuation.", path)
        _cold_start(trainer)
        return False
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RuntimeError) as exc:
        raise ValueError(f"Invalid ACE checkpoint {path}: {exc}") from exc
    _apply_state(trainer, updates, rng)
    return True
