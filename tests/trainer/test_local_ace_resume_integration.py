"""Checkpoint-path integration with real CPU optimization and torchdata.

VERL_RESUME_SOURCE points to a patched VeRL source checkout. Execute its actual
save/load methods and the actual local ACE hooks via AST isolation, replacing
only distributed workers with a tiny CPU Linear+SGD worker. This is not GPU PPO.
"""

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
from rllm.algorithms.ace.entrypoints import _initialize_controller_seed
from test_local_ace_checkpoint_resume import ROOT, isolated_class, local_class, trainer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler


class CPUWorker:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def save_checkpoint(self, local_path, remote_path, step, max_ckpt_to_keep):
        assert remote_path is None
        path = Path(local_path)
        path.mkdir(parents=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path / "cpu.pt")

    def load_checkpoint(self, path, del_local_after_load):
        assert not del_local_after_load
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


def integrated_class(monkeypatch):
    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("set VERL_RESUME_SOURCE to the tested VeRL resume checkout")
    source = Path(source)
    namespace = {"os": os, "torch": torch, "Role": SimpleNamespace(Critic="critic"), "deprecated": lambda message: lambda cls: cls}
    tree = ast.parse((source / "verl/utils/checkpoint/checkpoint_manager.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"find_latest_ckpt_path", "get_checkpoint_tracker_filename"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "checkpoint_manager.py", "exec"), namespace)
    parent = isolated_class(source / "verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", {"_save_checkpoint", "_load_checkpoint"}, namespace, bases=[])
    # The only inline dependency in these methods is mkdir. Scope the adapter
    # to this test; never poison global import state across test collection.
    fs = ModuleType("verl.utils.fs")
    fs.local_mkdir_safe = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(sys.modules, "verl.utils.fs", fs)
    cls = local_class(parent)
    loop = isolated_class(ROOT / "rllm/trainer/verl/agent_ppo_trainer.py", "AgentPPOTrainer", {"_get_training_start_epoch"}, {"RayPPOTrainer": object})
    cls._get_training_start_epoch = loop._get_training_start_epoch
    return cls


def fresh(cls, output_dir, *, workers=0):
    subject = cls()
    subject.__dict__.update(trainer().__dict__)
    subject.global_steps = 0
    subject._training_epoch = 0
    subject.local_train_playbooks = {}
    subject.local_train_task_seen_counts = {}
    subject._steps_per_epoch = 4
    subject.actor_rollout_wg = CPUWorker()
    subject.use_critic = False
    dataset = list(range(16))
    subject.train_dataloader = StatefulDataLoader(dataset, batch_size=4, num_workers=workers, sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(41)))
    subject.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(output_dir), "default_hdfs_dir": None, "resume_mode": "auto", "resume_from_path": None, "del_local_ckpt_after_load": False},
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
        }
    )
    return subject


def update(subject, batch):
    tasks = [str(item) for item in batch.tolist()]
    flushes = subject._prepare_task_playbooks_for_step(tasks, is_eval=False)
    before = [subject._get_task_playbook(task, is_eval=False).to_string() for task in tasks]
    decision = random.random(), float(np.random.random())
    for task in tasks:
        subject.local_train_playbooks[task].apply_operations({"operations": [{"type": "ADD", "section": "others", "content": f"learned {decision}"}]})
    subject.playbook = copy.deepcopy(subject.local_train_playbooks[tasks[0]])
    # Consume one rejected batch per epoch: visits advance, optimizer does not.
    if int(batch[0]) % 3:
        loss = subject.actor_rollout_wg.update(batch)
        subject.global_steps += 1
    else:
        loss = None
    return subject._training_epoch, tasks, before, flushes, decision, loss


def test_empty_data_checkpoint_is_rejected_before_ace_restore(tmp_path, monkeypatch):
    cls = integrated_class(monkeypatch)
    original = fresh(cls, tmp_path / "source")
    original.global_steps = 1
    original._save_checkpoint()
    checkpoint = tmp_path / "source/global_step_1"
    torch.save({}, checkpoint / "data.pt")
    resumed = fresh(cls, tmp_path / "resumed")
    resumed.config.trainer.resume_mode = "resume_path"
    resumed.config.trainer.resume_from_path = str(checkpoint)
    resumed.local_train_task_seen_counts = {"untouched": 9}
    with pytest.raises(ValueError):
        resumed._load_checkpoint()
    assert resumed.local_train_task_seen_counts == {"untouched": 9}


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("checkpoint_batch", [2, 4, "finished"])
@pytest.mark.parametrize("controller_seed", [None, 42])
def test_resumed_next_updates_exactly_match_uninterrupted(tmp_path, monkeypatch, workers, checkpoint_batch, controller_seed):
    cls = integrated_class(monkeypatch)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(21)
        np.random.seed(22)
        torch.manual_seed(23)
        _initialize_controller_seed(controller_seed)
        original = fresh(cls, tmp_path / "original", workers=workers)
        expected = []
        checkpoint = None
        for epoch in range(5):
            original._training_epoch = epoch
            for batch_index, batch in enumerate(original._iter_training_epoch(), 1):
                record = update(original, batch)
                if checkpoint is not None:
                    expected.append(record)
                if (epoch, batch_index) == (2, checkpoint_batch):
                    original._save_checkpoint()
                    checkpoint = tmp_path / "original" / f"global_step_{original.global_steps}"
            if epoch == 2 and checkpoint_batch == "finished":
                original._save_checkpoint()
                checkpoint = tmp_path / "original" / f"global_step_{original.global_steps}"
        assert checkpoint is not None and (checkpoint / "data.pt").is_file()
        assert (checkpoint / "ace_state.json").is_file()
        # Initialization seeds need no sidecar field: actual saved RNG state is
        # authoritative after restoration, including for old sidecar schemas.
        assert "controller_seed" not in json.loads((checkpoint / "ace_state.json").read_text())
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        _initialize_controller_seed(controller_seed)
        random.random(), np.random.random(), torch.rand(3)
        resumed = fresh(cls, tmp_path / "resumed", workers=workers)
        resumed.config.trainer.resume_mode = "resume_path"
        resumed.config.trainer.resume_from_path = str(checkpoint)
        resumed._load_checkpoint()
        actual = []
        for epoch in range(resumed._get_training_start_epoch(), 5):
            resumed._training_epoch = epoch
            for batch in resumed._iter_training_epoch():
                actual.append(update(resumed, batch))
        assert actual == expected
        assert resumed.global_steps == original.global_steps
        assert resumed.local_train_task_seen_counts == original.local_train_task_seen_counts
        assert {k: v.to_string() for k, v in resumed.local_train_playbooks.items()} == {k: v.to_string() for k, v in original.local_train_playbooks.items()}
        for key, value in original.actor_rollout_wg.model.state_dict().items():
            assert torch.equal(resumed.actor_rollout_wg.model.state_dict()[key], value)
        expected_optimizer = original.actor_rollout_wg.optimizer.state_dict()
        actual_optimizer = resumed.actor_rollout_wg.optimizer.state_dict()
        assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
        for key, values in expected_optimizer["state"].items():
            for field, value in values.items():
                assert torch.equal(actual_optimizer["state"][key][field], value)
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
