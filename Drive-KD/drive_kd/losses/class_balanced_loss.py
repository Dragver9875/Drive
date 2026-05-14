from __future__ import annotations

import torch


@torch.no_grad()
def class_weights_from_target(
    target: torch.Tensor,
    num_classes: int,
    min_weight: float = 0.05,
    max_weight: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:

    target = target.long().view(-1)

    counts = torch.bincount(target.clamp(0, num_classes - 1), minlength=num_classes).float()
    freq = counts / (counts.sum() + eps)

    weights = 1.0 / (freq + eps)
    weights = weights / weights.mean().clamp_min(eps)
    weights = weights.clamp(min=min_weight, max=max_weight)

    return weights


@torch.no_grad()
def binary_pos_weight_from_target(
    target: torch.Tensor,
    min_value: float = 1.0,
    max_value: float = 20.0,
    eps: float = 1e-6,
) -> torch.Tensor:

    target = target.float()

    pos = target.sum()
    total = torch.tensor(float(target.numel()), device=target.device)
    neg = total - pos

    pos_weight = neg / (pos + eps)
    pos_weight = pos_weight.clamp(min=min_value, max=max_value)

    return pos_weight