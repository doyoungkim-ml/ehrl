"""Observe role-labelled action-token advantages without changing PPO batches."""

from collections import Counter

import numpy as np
import torch


def role_credit_metrics(batch) -> dict[str, int]:
    """Count surviving rows and signed advantages after rejection/truncation.

    These are loss-eligible action tokens, not measured parameter gradients.
    Call before transport padding so duplicate padding rows are not counted.
    """
    roles = batch.non_tensor_batch.get("ace_component")
    if roles is None:
        return {}
    advantages = batch.batch["advantages"]
    mask = batch.batch["response_mask"].bool()
    if advantages.shape != mask.shape or len(roles) != len(mask):
        raise ValueError("ACE role labels, action masks and advantages must align")
    uids = batch.non_tensor_batch["uid"]
    if len(uids) != len(roles):
        raise ValueError("ACE role labels and group IDs must align")
    metrics = {}
    for role in ("policy", "reflector", "curator"):
        indices = np.flatnonzero(np.asarray(roles, dtype=object) == role)
        role_adv = advantages[indices]
        role_mask = mask[indices]
        finite = torch.isfinite(role_adv)
        positive = role_mask & finite & (role_adv > 1e-8)
        negative = role_mask & finite & (role_adv < -1e-8)
        nonzero = positive | negative
        groups = Counter(str(uids[i]) for i in indices)
        prefix = f"ace/role_credit/{role}"
        metrics.update({
            f"{prefix}_rows": len(indices),
            f"{prefix}_groups": len(groups),
            f"{prefix}_singleton_groups": sum(n == 1 for n in groups.values()),
            f"{prefix}_action_tokens": int(role_mask.sum().item()),
            f"{prefix}_positive_advantage_tokens": int(positive.sum().item()),
            f"{prefix}_negative_advantage_tokens": int(negative.sum().item()),
            f"{prefix}_nonzero_advantage_rows": int(nonzero.any(dim=-1).sum().item()),
            f"{prefix}_invalid_advantage_tokens": int((role_mask & ~finite).sum().item()),
        })
    return metrics
