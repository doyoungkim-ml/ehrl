from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rllm.trainer.verl.role_credit import role_credit_metrics


def test_role_credit_counts_only_surviving_action_tokens():
    batch = SimpleNamespace(
        non_tensor_batch={
            "ace_component": np.array(["policy", "reflector", "curator", "curator"], dtype=object),
            "uid": np.array(["p", "r", "c", "c"], dtype=object),
        },
        batch={
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0]]),
            "advantages": torch.tensor([[1., -1., 9.], [0., 0., 8.], [.5, 8., 8.], [-.5, float("nan"), 9.]]),
        },
    )
    m = role_credit_metrics(batch)
    assert m["ace/role_credit/policy_action_tokens"] == 2
    assert m["ace/role_credit/policy_positive_advantage_tokens"] == 1
    assert m["ace/role_credit/policy_negative_advantage_tokens"] == 1
    assert m["ace/role_credit/reflector_nonzero_advantage_rows"] == 0
    assert m["ace/role_credit/curator_groups"] == 1
    assert m["ace/role_credit/curator_singleton_groups"] == 0
    assert m["ace/role_credit/curator_nonzero_advantage_rows"] == 2
    assert m["ace/role_credit/curator_invalid_advantage_tokens"] == 1
    # A rejected curator group is absent from this final batch and adds no credit.
    assert m["ace/role_credit/curator_rows"] == 2


def test_non_ace_batch_has_no_role_metrics():
    assert role_credit_metrics(SimpleNamespace(non_tensor_batch={})) == {}


def test_role_credit_rejects_misaligned_metadata():
    batch = SimpleNamespace(non_tensor_batch={"ace_component": ["policy"], "uid": ["p"]}, batch={"advantages": torch.ones(2, 2), "response_mask": torch.ones(2, 2)})
    with pytest.raises(ValueError, match="align"):
        role_credit_metrics(batch)
