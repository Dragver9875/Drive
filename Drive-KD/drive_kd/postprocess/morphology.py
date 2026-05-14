from __future__ import annotations

import cv2
import numpy as np


def ensure_uint8_binary(mask: np.ndarray) -> np.ndarray:

    if mask.dtype != np.uint8:
        mask = mask.astype(np.float32)

    if mask.max() <= 1:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 127).astype(np.uint8) * 255

    return mask


def make_kernel(kernel_size: int = 3, shape: str = "ellipse") -> np.ndarray:

    kernel_size = int(kernel_size)

    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")

    if kernel_size % 2 == 0:
        kernel_size += 1

    shape = shape.lower()

    if shape == "rect":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    if shape == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    raise ValueError(f"Unsupported kernel shape: {shape}")


def open_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:

    m = ensure_uint8_binary(mask)
    kernel = make_kernel(kernel_size)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=int(iterations))


def close_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:

    m = ensure_uint8_binary(mask)
    kernel = make_kernel(kernel_size)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=int(iterations))


def erode_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:

    m = ensure_uint8_binary(mask)
    kernel = make_kernel(kernel_size)
    return cv2.erode(m, kernel, iterations=int(iterations))


def dilate_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:

    m = ensure_uint8_binary(mask)
    kernel = make_kernel(kernel_size)
    return cv2.dilate(m, kernel, iterations=int(iterations))


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 64,
    connectivity: int = 8,
) -> np.ndarray:

    m = ensure_uint8_binary(mask)
    binary = (m > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=int(connectivity),
    )

    out = np.zeros_like(binary)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            out[labels == label] = 1

    return out.astype(np.uint8) * 255


def fill_small_holes(
    mask: np.ndarray,
    max_hole_area: int = 128,
) -> np.ndarray:

    m = ensure_uint8_binary(mask)
    binary = (m > 0).astype(np.uint8)

    inv = 1 - binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    h, w = binary.shape
    out = binary.copy()

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        ww = stats[label, cv2.CC_STAT_WIDTH]
        hh = stats[label, cv2.CC_STAT_HEIGHT]

        touches_border = x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h

        if not touches_border and area <= max_hole_area:
            out[labels == label] = 1

    return out.astype(np.uint8) * 255


def keep_largest_components(
    mask: np.ndarray,
    k: int = 1,
    connectivity: int = 8,
) -> np.ndarray:

    m = ensure_uint8_binary(mask)
    binary = (m > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=int(connectivity),
    )

    components = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        components.append((area, label))

    components.sort(reverse=True)

    keep = {label for _, label in components[: int(k)]}

    out = np.zeros_like(binary)

    for label in keep:
        out[labels == label] = 1

    return out.astype(np.uint8) * 255