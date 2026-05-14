from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from drive_kd.visualization.overlays import (
    denormalize_image,
    save_rgb_image,
    tensor_to_numpy_image,
    to_numpy_mask,
)


def binary_error_map(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.50,
) -> np.ndarray:

    p = to_numpy_mask(pred, threshold=threshold) > 0
    t = to_numpy_mask(target, threshold=0.5) > 0

    if p.shape != t.shape:
        raise ValueError(f"Error map shape mismatch: pred={p.shape}, target={t.shape}")

    out = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)

    tp = p & t
    fp = p & ~t
    fn = ~p & t

    out[tp] = (0, 220, 0)
    out[fp] = (255, 0, 0)
    out[fn] = (0, 80, 255)

    return out


def overlay_error_map(
    image: torch.Tensor | np.ndarray,
    error_rgb: np.ndarray,
    alpha: float = 0.70,
) -> np.ndarray:

    if isinstance(image, torch.Tensor):
        img = denormalize_image(image)
    else:
        img = tensor_to_numpy_image(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)

    if img.shape != error_rgb.shape:
        raise ValueError(f"Image/error shape mismatch: image={img.shape}, error={error_rgb.shape}")

    out = img.copy()

    mask = error_rgb.sum(axis=-1) > 0

    out[mask] = (
        (1.0 - alpha) * out[mask].astype(np.float32)
        + alpha * error_rgb[mask].astype(np.float32)
    ).astype(np.uint8)

    return out


def save_error_map(
    *,
    path: str | Path,
    image: torch.Tensor | np.ndarray | None,
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.50,
    overlay: bool = True,
    alpha: float = 0.70,
) -> Path:

    err = binary_error_map(pred=pred, target=target, threshold=threshold)

    if overlay:
        if image is None:
            raise ValueError("image must be provided when overlay=True")
        err = overlay_error_map(image=image, error_rgb=err, alpha=alpha)

    return save_rgb_image(path, err)