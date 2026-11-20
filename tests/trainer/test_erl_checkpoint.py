"""ERL controller checkpoint hooks, with actual VeRL/torchdata continuation."""

import ast
import copy
import json
import os
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_erl_original import ROOT, collect, trainer_class
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from rllm.algorithms.erl.checkpoint import capture_rng, load_state, preserve_controller_rng, save_state
from rllm.algorithms.erl.controller import ERLController


def subject(cls=object, root=None):
    instance = cls.__new__(cls) if cls is not object else SimpleNamespace()
    instance.global_steps = 5
    instance._training_epoch = 2
    instance.erl_controller = ERLController(2)
    instance.erl_controller.memories = ["learned prompt", None]
    instance.erl_controller.slot_order = [1, 0]
    instance._erl_state_signature = {"upstream": "pin", "role_caps": [4096, 8192]}
    instance._erl_pending = "transient"
    instance.config = OmegaConf.create({"trainer": {"default_local_dir": str(root), "default_hdfs_dir": None, "resume_mode": "auto", "resume_from_path": None}})
    return instance


@pytest.fixture(autouse=True)
def preserve_rng():
    with preserve_controller_rng():
        yield


def test_roundtrip_memory_fifo_data_epoch_and_controller_rng(tmp_path):
    original = subject()
    save_state(original, tmp_path)
    expected = (random.random(), np.random.random(), torch.rand(2))
    restored = subject()
    restored._training_epoch = 0
    restored.erl_controller = ERLController(2)
    load_state(restored, tmp_path)
    actual = (random.random(), np.random.random(), torch.rand(2))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert restored.erl_controller.state_dict() == original.erl_controller.state_dict()
    assert restored._training_epoch == 2
    assert restored._erl_pending is None


@pytest.mark.parametrize(
    "field,value", [("version", 99), ("global_steps", 6), ("training_epoch", -1), ("upstream_commit", "wrong"), ("rng", {"bad": 1}), ("controller", {"memories": [None], "slot_order": [0]})]
)
def test_invalid_checkpoint_does_not_mutate_live_state(tmp_path, field, value):
    original = subject()
    save_state(original, tmp_path)
    file = tmp_path / "erl_state.json"
    document = json.loads(file.read_text())
    document[field] = value
    file.write_text(json.dumps(document))
    before = original.erl_controller.state_dict(), capture_rng(), original._training_epoch
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        load_state(original, tmp_path)
    assert before == (original.erl_controller.state_dict(), capture_rng(), original._training_epoch)


def test_missing_state_refuses_silent_memory_cold_start(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot resume original ERL"):
        load_state(subject(), tmp_path)


def test_actual_save_hook_persists_state_before_parent_marker(tmp_path):
    class Parent:
        def _save_checkpoint(self):
            folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            assert (folder / "erl_state.json").is_file()
            (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(self.global_steps))

    instance = subject(trainer_class(Parent), tmp_path)
    instance._save_checkpoint()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "5"


def test_actual_resume_hook_uses_loaded_explicit_path(tmp_path):
    folder = tmp_path / "source/global_step_5"
    save_state(subject(), folder)

    class Parent:
        def _load_checkpoint(self):
            self.global_steps = 5
            self._dataloader_resume_iterator_finished = False

    instance = subject(trainer_class(Parent), tmp_path / "new")
    instance.config.trainer.resume_mode = "resume_path"
    instance.config.trainer.resume_from_path = str(folder)
    instance.erl_controller.memories = [None, None]
    instance._load_checkpoint()
    assert instance.erl_controller.memories == ["learned prompt", None]


def isolated_class(path, name, methods, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls = copy.deepcopy(cls)
    cls.bases = []
    cls.decorator_list = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert len(cls.body) == len(methods)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def integrated_class(monkeypatch):
    source_text = os.environ.get("VERL_RESUME_SOURCE")
    assert source_text, "VERL_RESUME_SOURCE is required; ERL checkpoint continuation coverage cannot be skipped"
    source = Path(source_text)
    namespace = {"os": os, "torch": torch, "Role": SimpleNamespace(Critic="critic")}
    file = source / "verl/utils/checkpoint/checkpoint_manager.py"
    nodes = [node for node in ast.parse(file.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in {"find_latest_ckpt_path", "get_checkpoint_tracker_filename"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(file), "exec"), namespace)
    parent = isolated_class(source / "verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", {"_save_checkpoint", "_load_checkpoint"}, namespace)
    fs = ModuleType("verl.utils.fs")
    fs.local_mkdir_safe = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(sys.modules, "verl.utils.fs", fs)
    cls = trainer_class(parent)
    loop = isolated_class(ROOT / "rllm/trainer/verl/agent_ppo_trainer.py", "AgentPPOTrainer", {"_get_training_start_epoch"}, {})
    cls._get_training_start_epoch = loop._get_training_start_epoch
    return cls


class CPUWorker:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def save_checkpoint(self, local_path, remote_path, step, max_ckpt_to_keep):
        assert remote_path is None
        Path(local_path).mkdir(parents=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, Path(local_path) / "cpu.pt")

    def load_checkpoint(self, path, del_local_after_load):
        state = torch.load(Path(path) / "cpu.pt", weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def update(self, batch):
        self.optimizer.zero_grad()
        inputs = batch.float().reshape(-1, 1) / 16
        target = inputs * 0.2 + torch.rand_like(inputs) * 0.01
        loss = (self.model(inputs) - target).square().mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fresh(cls, folder, workers):
    instance = subject(cls, folder)
    instance.global_steps = instance._training_epoch = 0
    instance.erl_controller = ERLController(2)
    instance.actor_rollout_wg = CPUWorker()
    instance.use_critic = False
    dataset = list(range(16))
    instance.train_dataloader = StatefulDataLoader(dataset, batch_size=4, num_workers=workers, sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(41)))
    instance.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(folder), "default_hdfs_dir": None, "resume_mode": "auto", "resume_from_path": None, "del_local_ckpt_after_load": False},
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
        }
    )
    return instance


def update(instance, batch):
    before = instance.erl_controller.state_dict()
    decision = random.random(), float(np.random.random())
    first = [int(value) % 2 for value in batch]
    second = [1 if int(value) % 3 else 0.2 for value in batch]
    collect(instance.erl_controller, first, second, [f"{value}:{decision}" for value in batch])
    loss = instance.actor_rollout_wg.update(batch)
    instance.global_steps += 1
    return instance._training_epoch, batch.tolist(), before, decision, loss


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("boundary", [2, 4, "finished"])
def test_real_pinned_resume_matches_memory_data_rng_weights_optimizer(tmp_path, monkeypatch, workers, boundary):
    cls = integrated_class(monkeypatch)
    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    original = fresh(cls, tmp_path / "original", workers)
    expected, checkpoint = [], None
    for epoch in range(4):
        original._training_epoch = epoch
        for batch_index, batch in enumerate(original._iter_training_epoch(), 1):
            record = update(original, batch)
            if checkpoint:
                expected.append(record)
            if (epoch, batch_index) == (1, boundary):
                original._save_checkpoint()
                checkpoint = tmp_path / "original" / f"global_step_{original.global_steps}"
        if epoch == 1 and boundary == "finished":
            original._save_checkpoint()
            checkpoint = tmp_path / "original" / f"global_step_{original.global_steps}"
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    resumed = fresh(cls, tmp_path / "resumed", workers)
    resumed.config.trainer.resume_mode = "resume_path"
    resumed.config.trainer.resume_from_path = str(checkpoint)
    resumed._load_checkpoint()
    actual = []
    for epoch in range(resumed._get_training_start_epoch(), 4):
        resumed._training_epoch = epoch
        for batch in resumed._iter_training_epoch():
            actual.append(update(resumed, batch))
    assert actual == expected
    assert resumed.erl_controller.state_dict() == original.erl_controller.state_dict()
    for key, value in original.actor_rollout_wg.model.state_dict().items():
        assert torch.equal(resumed.actor_rollout_wg.model.state_dict()[key], value)
    expected_opt = original.actor_rollout_wg.optimizer.state_dict()
    actual_opt = resumed.actor_rollout_wg.optimizer.state_dict()
    assert expected_opt["param_groups"] == actual_opt["param_groups"]
    for key, values in expected_opt["state"].items():
        for field, value in values.items():
            assert torch.equal(actual_opt["state"][key][field], value)
