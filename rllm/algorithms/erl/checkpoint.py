"""Atomic full-controller sidecar for original ERL, independent of ACE."""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from rllm.algorithms.erl import UPSTREAM_COMMIT

FILENAME = "erl_state.json"
VERSION = 1


def capture_rng():
    numpy = np.random.get_state()
    python = random.getstate()
    return {
        "python": [python[0], list(python[1]), python[2]],
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "torch_cpu": torch.get_rng_state().tolist(),
    }


def decode_rng(document):
    if not isinstance(document, dict) or set(document) != {"python", "numpy", "torch_cpu"}:
        raise ValueError("Invalid ERL controller RNG fields")
    python, numpy, cpu = document["python"], document["numpy"], document["torch_cpu"]
    if not isinstance(python, list) or len(python) != 3 or not isinstance(numpy, list) or len(numpy) != 5:
        raise ValueError("Invalid ERL controller RNG structure")
    if not isinstance(cpu, list) or any(type(value) is not int or not 0 <= value <= 255 for value in cpu):
        raise ValueError("Invalid ERL torch RNG bytes")
    if not isinstance(numpy[1], list) or any(type(value) is not int or not 0 <= value < 2**32 for value in numpy[1]):
        raise ValueError("Invalid ERL numpy RNG keys")
    states = (
        (python[0], tuple(python[1]), python[2]),
        (numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]),
        torch.tensor(cpu, dtype=torch.uint8),
    )
    # Probe isolated generators before mutating any live controller or RNG.
    random.Random(0).setstate(states[0])
    np.random.RandomState(0).set_state(states[1])
    torch.Generator(device="cpu").set_state(states[2])
    return states


def restore_rng(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])


@contextmanager
def preserve_controller_rng():
    states = decode_rng(capture_rng())
    try:
        yield
    finally:
        restore_rng(states)


def save_state(trainer, checkpoint_dir):
    epoch = getattr(trainer, "_training_epoch", None)
    if type(epoch) is not int or epoch < 0:
        raise ValueError("ERL checkpoint requires the actual consumed data epoch")
    document = {
        "version": VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "global_steps": trainer.global_steps,
        "training_epoch": epoch,
        "configuration": copy.deepcopy(trainer._erl_state_signature),
        "controller": trainer.erl_controller.state_dict(),
        "rng": capture_rng(),
    }
    # Check all serialized fields with the same loader before publishing.
    _validate(trainer, document)
    folder = Path(checkpoint_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{FILENAME}.", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(document, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, folder / FILENAME)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate(trainer, document):
    required = {"version", "upstream_commit", "global_steps", "training_epoch", "configuration", "controller", "rng"}
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("Invalid ERL checkpoint fields")
    if document["version"] != VERSION or document["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("Incompatible ERL checkpoint version or upstream pin")
    if type(document["global_steps"]) is not int or document["global_steps"] != trainer.global_steps:
        raise ValueError("ERL checkpoint step does not match restored model")
    if type(document["training_epoch"]) is not int or document["training_epoch"] < 0:
        raise ValueError("Invalid ERL checkpoint training epoch")
    if document["configuration"] != trainer._erl_state_signature:
        raise ValueError("ERL checkpoint configuration changed")
    controller = copy.deepcopy(trainer.erl_controller)
    controller.load_state_dict(document["controller"])
    return controller, decode_rng(document["rng"])


def load_state(trainer, checkpoint_dir):
    path = Path(checkpoint_dir) / FILENAME
    if not path.is_file():
        raise RuntimeError(f"Cannot resume original ERL without {path}; remembered prompts and FIFO ownership cannot be reconstructed from weights")
    document = json.loads(path.read_text(encoding="utf-8"))
    controller, rng = _validate(trainer, document)
    trainer.erl_controller = controller
    trainer._training_epoch = document["training_epoch"]
    trainer._erl_pending = None
    restore_rng(rng)
