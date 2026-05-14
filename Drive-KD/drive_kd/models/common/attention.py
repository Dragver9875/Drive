from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_spatial_attention(
    feature: torch.Tensor,
    method: str = "channel_mean_abs",
    keepdim: bool = True,
) -> torch.Tensor:

    if feature.ndim != 4:
        raise ValueError(f"Expected feature tensor [B,C,H,W], got {feature.shape}")

    method = method.lower()

    if method == "channel_mean_abs":
        att = feature.abs().mean(dim=1, keepdim=keepdim)

    elif method == "channel_sum_abs":
        att = feature.abs().sum(dim=1, keepdim=keepdim)

    elif method == "channel_mean_square":
        att = feature.pow(2).mean(dim=1, keepdim=keepdim)

    else:
        raise ValueError(f"Unsupported attention method: {method}")

    return att


def normalize_attention_map(
    attention: torch.Tensor,
    eps: float = 1e-6,
    mode: str = "l2",
) -> torch.Tensor:

    if attention.ndim == 3:
        attention = attention.unsqueeze(1)

    if attention.ndim != 4:
        raise ValueError(f"Expected attention [B,1,H,W] or [B,H,W], got {attention.shape}")

    mode = mode.lower()

    b = attention.shape[0]
    flat = attention.view(b, -1)

    if mode == "l2":
        norm = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
        flat = flat / (norm + eps)

    elif mode == "sum":
        norm = flat.sum(dim=1, keepdim=True)
        flat = flat / (norm + eps)

    elif mode == "minmax":
        min_v = flat.min(dim=1, keepdim=True).values
        max_v = flat.max(dim=1, keepdim=True).values
        flat = (flat - min_v) / (max_v - min_v + eps)

    else:
        raise ValueError(f"Unsupported attention normalization mode: {mode}")

    return flat.view_as(attention)


def resize_attention_like(
    attention: torch.Tensor,
    reference: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:

    if attention.ndim == 3:
        attention = attention.unsqueeze(1)

    if reference.ndim != 4:
        raise ValueError(f"Reference must be [B,C,H,W], got {reference.shape}")

    if attention.shape[-2:] == reference.shape[-2:]:
        return attention

    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        return F.interpolate(
            attention,
            size=reference.shape[-2:],
            mode=mode,
            align_corners=False,
        )

    return F.interpolate(attention, size=reference.shape[-2:], mode=mode)


def attention_l2_loss(
    student_attention: torch.Tensor,
    teacher_attention: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:

    if student_attention.ndim == 3:
        student_attention = student_attention.unsqueeze(1)

    if teacher_attention.ndim == 3:
        teacher_attention = teacher_attention.unsqueeze(1)

    if student_attention.shape[-2:] != teacher_attention.shape[-2:]:
        teacher_attention = F.interpolate(
            teacher_attention,
            size=student_attention.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if normalize:
        student_attention = normalize_attention_map(student_attention, mode="l2")
        teacher_attention = normalize_attention_map(teacher_attention, mode="l2")

    return F.mse_loss(student_attention, teacher_attention)