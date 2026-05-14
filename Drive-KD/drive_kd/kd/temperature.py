from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    dim: int = 1,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    return F.softmax(logits / float(temperature), dim=dim)


def sigmoid_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    return torch.sigmoid(logits / float(temperature))


def binary_logits_from_two_class_logits(logits: torch.Tensor) -> torch.Tensor:

    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B,2,H,W], got {logits.shape}")

    if logits.shape[1] != 2:
        raise ValueError(
            f"Expected two-class logits with channel size 2, got {logits.shape[1]}"
        )

    return logits[:, 1] - logits[:, 0]