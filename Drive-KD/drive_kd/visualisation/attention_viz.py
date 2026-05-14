from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from drive_kd.visualization.overlays import denormalize_image, save_rgb_image, to_numpy_mask
from drive_kd.visualization.save_grid import save_image_grid


def attention_to_numpy(attention: torch.Tensor | np.ndarray) -> np.ndarray:

    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().float().numpy()

    if attention.ndim == 4:
        if attention.shape[0] != 1:
            raise ValueError(f"Cannot convert batched attention with B>1: {attention.shape}")
        attention = attention[0]

    if attention.ndim == 3:
        if attention.shape[0] == 1:
            attention = attention[0]
        elif attention.shape[-1] == 1:
            attention = attention[..., 0]
        else:
            attention = attention.mean(axis=0)

    if attention.ndim != 2:
        raise ValueError(f"Expected 2D attention, got shape={attention.shape}")

    return attention.astype(np.float32)


def normalize_to_uint8(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    x = x.astype(np.float32)

    min_v = float(x.min())
    max_v = float(x.max())

    x = (x - min_v) / (max_v - min_v + eps)
    return (x * 255.0).astype(np.uint8)


def attention_to_heatmap(
    attention: torch.Tensor | np.ndarray,
    output_size: tuple[int, int] | None = None,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:

    att = attention_to_numpy(attention)

    if output_size is not None:
        att = cv2.resize(
            att,
            (output_size[1], output_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    att_u8 = normalize_to_uint8(att)

    heat_bgr = cv2.applyColorMap(att_u8, colormap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    return heat_rgb


def overlay_attention(
    image: torch.Tensor | np.ndarray,
    attention: torch.Tensor | np.ndarray,
    alpha: float = 0.50,
) -> np.ndarray:

    img = denormalize_image(image) if isinstance(image, torch.Tensor) else image

    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)

    heat = attention_to_heatmap(attention, output_size=img.shape[:2])

    out = (
        (1.0 - alpha) * img.astype(np.float32)
        + alpha * heat.astype(np.float32)
    ).astype(np.uint8)

    return out


def save_attention_grid(
    *,
    path: str | Path,
    image: torch.Tensor | np.ndarray,
    attention_dict: dict[str, torch.Tensor | np.ndarray],
    title: str = "attention",
    ncols: int = 3,
) -> Path:

    images = []

    base = denormalize_image(image) if isinstance(image, torch.Tensor) else image

    if base.dtype != np.uint8:
        base = np.clip(base, 0.0, 1.0)
        base = (base * 255.0).astype(np.uint8)

    images.append(base)

    for _, att in sorted(attention_dict.items()):
        images.append(overlay_attention(base, att, alpha=0.55))

    return save_image_grid(
        path=path,
        images=images,
        ncols=ncols,
        title=title,
    )