"""CPU tests of the real fit_agent epoch-loop control (not PPO operations).

The PPO batch body is replaced with a recorder. Original checkpoint initialization,
outer-loop header, epoch assignment and dataloader iteration are executed from AST.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_local_ace_checkpoint_resume import isolated_class
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "rllm/trainer/verl/agent_ppo_trainer.py"


def loop_class():
    node = next(n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "AgentPPOTrainer")
    fit = copy.deepcopy(next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "fit_agent"))
    epoch_loop = next(n for n in fit.body if isinstance(n, ast.For) and ast.unparse(n.target) == "epoch")
    inner = next(n for n in epoch_loop.body if isinstance(n, ast.For))
    inner.body = ast.parse("self.records.append((epoch, getattr(self, '_training_epoch', None), batch_dict.tolist()))").body
    # Retain real loop setup, omitting tracking/validation and PPO dependencies.
    setup = [
        n
        for n in fit.body[: fit.body.index(epoch_loop)]
        if isinstance(n, ast.Assign | ast.AugAssign)
        and (
            any(isinstance(t, ast.Attribute) and t.attr == "global_steps" for t in getattr(n, "targets", []))
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "start_epoch" for t in n.targets))
            or (isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "self.global_steps")
        )
    ]
    # Resume is invoked at exactly the production position, after global_steps=0.
    setup.insert(1, ast.parse("self._load_checkpoint()").body[0])
    fit.body = ast.parse("from rllm.utils.training_reward import initialize_training_reward, complete_training_reward_epoch").body + setup + ast.parse("initialize_training_reward(self, start_epoch)").body + [epoch_loop]
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in {"_get_training_start_epoch", "_iter_training_epoch"}] + [fit]
    node.bases = []
    ns = {"pprint": lambda *args: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(PATH), "exec"), ns)
    return ns["AgentPPOTrainer"]


def loader():
    dataset = list(range(16))
    return StatefulDataLoader(dataset, batch_size=4, sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(42)), num_workers=0)


@pytest.mark.parametrize("consumed,finished", [(2, False), (4, False), (4, True)])
def test_resume_uses_saved_data_epoch_even_when_update_steps_differ(consumed, finished):
    original = loader()
    for _ in range(2):
        list(original)
    iterator = iter(original)
    for _ in range(consumed):
        next(iterator)
    if finished:
        assert next(iterator, None) is None
    state = copy.deepcopy(original.state_dict())
    expected = [(2, 2, b.tolist()) for b in iterator]
    for epoch in range(3, 5):
        expected.extend((epoch, epoch, b.tolist()) for b in original)
    resumed = loop_class()()
    resumed.config = OmegaConf.create({"trainer": {"total_epochs": 5}})
    resumed.train_dataloader = loader()
    resumed.records = []

    def restore():
        # Update count is deliberately not data-epoch * len(loader), as rejected
        # batches consume input and increment visits without an optimizer step.
        resumed.global_steps = 3
        resumed._training_epoch = 2
        resumed._dataloader_resume_iterator_finished = state["_iterator_finished"]
        resumed.train_dataloader.load_state_dict(state)

    resumed._load_checkpoint = restore
    resumed.fit_agent()
    assert resumed.records == expected


def test_task_epoch_estimate_uses_actual_data_epoch_not_optimizer_count():
    local = isolated_class(ROOT / "rllm/algorithms/ace/trainer.py", "LocalACERLAgentPPOTrainer", {"_estimate_epoch"}, {"ACERLAgentPPOTrainer": object})()
    local.global_steps = 3
    local._steps_per_epoch = 4
    local._training_epoch = 2
    assert local._estimate_epoch() == 3
