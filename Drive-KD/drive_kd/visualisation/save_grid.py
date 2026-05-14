from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drive_kd.visualization.overlays import save_rgb_image


def pad_to_same_size(
    images: list[np.ndarray],
    pad_value: int = 0,
) -> list[np.ndarray]:

    if not images:
        return []

    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    padded = []

    for img in images:
        h, w = img.shape[:2]

        if img.ndim == 2:
            out = np.full((max_h, max_w), pad_value, dtype=img.dtype)
            out[:h, :w] = img
        else:
            out = np.full((max_h, max_w, img.shape[2]), pad_value, dtype=img.dtype)
            out[:h, :w] = img

        padded.append(out)

    return padded


def make_image_grid(
    images: list[np.ndarray],
    ncols: int = 4,
    pad: int = 8,
    pad_value: int = 0,
) -> np.ndarray:

    if not images:
        raise ValueError("Cannot create grid from empty image list.")

    images = pad_to_same_size(images, pad_value=pad_value)

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(images) / ncols))

    h, w = images[0].shape[:2]
    c = images[0].shape[2] if images[0].ndim == 3 else 1

    grid_h = nrows * h + (nrows - 1) * pad
    grid_w = ncols * w + (ncols - 1) * pad

    if c == 1:
        grid = np.full((grid_h, grid_w), pad_value, dtype=images[0].dtype)
    else:
        grid = np.full((grid_h, grid_w, c), pad_value, dtype=images[0].dtype)

    for idx, img in enumerate(images):
        row = idx // ncols
        col = idx % ncols

        y0 = row * (h + pad)
        x0 = col * (w + pad)

        grid[y0 : y0 + h, x0 : x0 + w] = img

    return grid


def add_title_bar(
    image_rgb: np.ndarray,
    title: str,
    height: int = 32,
) -> np.ndarray:

    if image_rgb.ndim != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")

    h, w = image_rgb.shape[:2]

    bar = np.zeros((height, w, 3), dtype=np.uint8)

    cv2.putText(
        bar,
        title,
        (10, int(height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return np.concatenate([bar, image_rgb], axis=0)


def save_image_grid(
    *,
    path: str | Path,
    images: list[np.ndarray],
    ncols: int = 4,
    pad: int = 8,
    pad_value: int = 0,
    title: str | None = None,
) -> Path:

    grid = make_image_grid(
        images=images,
        ncols=ncols,
        pad=pad,
        pad_value=pad_value,
    )

    if title is not None:
        grid = add_title_bar(grid, title=title)

    return save_rgb_image(path, grid)