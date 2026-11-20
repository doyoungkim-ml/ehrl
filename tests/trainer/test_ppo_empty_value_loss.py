"""CPU numerical coverage for masked synthetic critic padding."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_mixed_advantages import verl_math  # noqa: F401 -- shared real pinned-source fixture

from rllm.trainer.verl.losses import mask_safe_value_loss


@pytest.mark.parametrize("nested", [False, True])
def test_padding_only_microbatch_has_finite_zero_loss_and_gradients(nested):
    if nested:
        predictions = torch.nested.nested_tensor([torch.tensor([.2, .7]), torch.tensor([-.4])], layout=torch.jagged, requires_grad=True)
        mask = torch.nested.nested_tensor([torch.zeros(2), torch.zeros(1)], layout=torch.jagged)
    else:
        predictions = torch.tensor([[.2, .7], [.3, -.4]], requires_grad=True)
        mask = torch.zeros_like(predictions)
    loss, metrics = mask_safe_value_loss(
        loss_fn=lambda **kwargs: pytest.fail("empty padding reached dividing backend"),
        model_output={"values": predictions}, data={"response_mask": mask},
    )
    loss.backward()
    gradient = predictions.grad.values() if nested else predictions.grad
    assert loss.item() == 0
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert all(value == 0 for value in metrics.values())


def test_real_critic_rows_keep_pinned_value_loss_and_gradients(verl_math):  # noqa: F811 -- injected pytest fixture
    predictions = torch.tensor([[.2, .7], [.3, -.4]], requires_grad=True)
    data = {"response_mask": torch.tensor([[1., 1.], [0., 0.]]), "values": torch.zeros(2, 2), "returns": torch.tensor([[0., 1.], [0., 0.]])}
    calls = []

    def backend(model_output, data, dp_group):
        calls.append(dp_group)
        loss, clipfrac = verl_math["compute_value_loss"](
            vpreds=model_output["values"], values=data["values"], returns=data["returns"], response_mask=data["response_mask"], cliprange_value=.2, loss_agg_mode="token-mean"
        )
        return loss, {"critic/vf_clipfrac": clipfrac.item()}

    expected, expected_metrics = backend({"values": predictions}, data, "group")
    expected_grad = torch.autograd.grad(expected, predictions)[0]
    actual, actual_metrics = mask_safe_value_loss(loss_fn=backend, model_output={"values": predictions}, data=data, dp_group="group")
    actual_grad = torch.autograd.grad(actual, predictions)[0]
    assert calls == ["group", "group"]
    assert actual.item() == expected.item()
    assert actual_metrics == expected_metrics
    assert torch.equal(actual_grad, expected_grad)
    assert torch.equal(actual_grad[1], torch.zeros(2))


def test_registered_critic_worker_installs_guard_on_its_real_loss_hook():
    path = Path(__file__).resolve().parents[2] / "rllm/trainer/verl/train_agent_ppo.py"
    worker = next(node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.ClassDef) and node.name == "CriticTrainingWorker")
    setter = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "set_loss_fn")
    assert len(setter.decorator_list) == 1
    assert ast.unparse(setter.decorator_list[0]) == "register(dispatch_mode=Dispatch.ONE_TO_ALL)"
    setter.decorator_list = []
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[setter], type_ignores=[])), str(path), "exec"), namespace)
    worker = SimpleNamespace()
    namespace["set_loss_fn"](worker, lambda **kwargs: pytest.fail("unguarded empty critic loss"))
    predictions = torch.tensor([.3], requires_grad=True)
    loss, _ = worker.loss_fn(model_output={"values": predictions}, data={"response_mask": torch.zeros(1)})
    loss.backward()
    assert predictions.grad.item() == 0
