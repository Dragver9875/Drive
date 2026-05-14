from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def tensor_to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().float().numpy()

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(image)}")

    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={image.shape}")

    if image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.transpose(image, (1, 2, 0))

    return image


def denormalize_image(
    image: torch.Tensor | np.ndarray,
    mean: list[float] | tuple[float, float, float] = DEFAULT_MEAN,
    std: list[float] | tuple[float, float, float] = DEFAULT_STD,
) -> np.ndarray:

    img = tensor_to_numpy_image(image).astype(np.float32)

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    img = img * std_arr + mean_arr
    img = np.clip(img, 0.0, 1.0)

    return (img * 255.0).astype(np.uint8)


def to_numpy_mask(mask: torch.Tensor | np.ndarray, threshold: float | None = None) -> np.ndarray:

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().float().numpy()

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(mask)}")

    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError(f"Cannot convert batched mask with batch > 1: {mask.shape}")
        mask = mask[0]

    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"Unsupported 3D mask shape: {mask.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")

    if threshold is not None:
        mask = (mask > threshold).astype(np.uint8)

    return mask


def mask_to_color(
    mask: torch.Tensor | np.ndarray,
    color: tuple[int, int, int],
    threshold: float | None = None,
) -> np.ndarray:

    m = to_numpy_mask(mask, threshold=threshold)
    m = (m > 0).astype(np.uint8)

    color_mask = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    color_mask[m > 0] = np.array(color, dtype=np.uint8)

    return color_mask


def overlay_mask(
    image_rgb: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
    threshold: float | None = None,
) -> np.ndarray:

    if isinstance(image_rgb, torch.Tensor):
        img = denormalize_image(image_rgb)
    else:
        img = tensor_to_numpy_image(image_rgb)

        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)

    m = to_numpy_mask(mask, threshold=threshold)
    m_bool = m > 0

    overlay = img.copy()
    color_arr = np.array(color, dtype=np.float32)

    overlay[m_bool] = (
        (1.0 - alpha) * overlay[m_bool].astype(np.float32)
        + alpha * color_arr
    ).astype(np.uint8)

    return overlay


def draw_legend(
    image_rgb: np.ndarray,
    items: list[tuple[str, tuple[int, int, int]]],
    x: int = 10,
    y: int = 10,
) -> np.ndarray:

    out = image_rgb.copy()

    for i, (label, color) in enumerate(items):
        yy = y + i * 24

        cv2.rectangle(
            out,
            (x, yy),
            (x + 16, yy + 16),
            color[::-1],
            thickness=-1,
        )
        cv2.putText(
            out,
            label,
            (x + 24, yy + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return out


def save_rgb_image(path: str | Path, image_rgb: np.ndarray) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

    return path


def save_prediction_overlay(
    *,
    path: str | Path,
    image: torch.Tensor | np.ndarray,
    road_mask: torch.Tensor | np.ndarray | None = None,
    lane_mask: torch.Tensor | np.ndarray | None = None,
    edge_mask: torch.Tensor | np.ndarray | None = None,
    road_threshold: float = 0.50,
    lane_threshold: float = 0.35,
    edge_threshold: float = 0.40,
    alpha: float = 0.45,
    add_legend: bool = True,
) -> Path:

    if isinstance(image, torch.Tensor):
        out = denormalize_image(image)
    else:
        out = tensor_to_numpy_image(image)
        if out.dtype != np.uint8:
            out = np.clip(out, 0.0, 1.0)
            out = (out * 255.0).astype(np.uint8)

    if road_mask is not None:
        out = overlay_mask(
            out,
            road_mask,
            color=(0, 180, 0),
            alpha=alpha,
            threshold=road_threshold,
        )

    if lane_mask is not None:
        out = overlay_mask(
            out,
            lane_mask,
            color=(0, 120, 255),
            alpha=alpha,
            threshold=lane_threshold,
        )

    if edge_mask is not None:
        out = overlay_mask(
            out,
            edge_mask,
            color=(255, 0, 0),
            alpha=alpha,
            threshold=edge_threshold,
        )

    if add_legend:
        out = draw_legend(
            out,
            [
                ("road", (0, 180, 0)),
                ("lane", (0, 120, 255)),
                ("edge", (255, 0, 0)),
            ],
        )

    return save_rgb_image(path, out)


def logits_to_prediction_masks(
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:

    road_logits = outputs["road_logits"]
    lane_logits = outputs["lane_logits"]
    edge_logits = outputs["edge_logits"]

    return {
        "road_prob": torch.softmax(road_logits, dim=1)[:, 1],
        "lane_prob": torch.softmax(lane_logits, dim=1)[:, 1],
        "edge_prob": torch.sigmoid(edge_logits).squeeze(1),
    }