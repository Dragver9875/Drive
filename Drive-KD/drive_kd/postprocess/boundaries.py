from __future__ import annotations

import cv2
import numpy as np

from drive_kd.postprocess.morphology import (
    close_mask,
    ensure_uint8_binary,
    make_kernel,
    open_mask,
    remove_small_components,
)


def extract_boundary_from_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:

    m = ensure_uint8_binary(mask)
    kernel = make_kernel(kernel_size)

    boundary = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)

    return ensure_uint8_binary(boundary)


def extract_combined_boundary(
    road_mask: np.ndarray | None = None,
    lane_mask: np.ndarray | None = None,
    edge_mask: np.ndarray | None = None,
    kernel_size: int = 3,
) -> np.ndarray:

    boundaries = []

    if road_mask is not None:
        boundaries.append(extract_boundary_from_mask(road_mask, kernel_size=kernel_size))

    if lane_mask is not None:
        boundaries.append(extract_boundary_from_mask(lane_mask, kernel_size=kernel_size))

    if edge_mask is not None:
        boundaries.append(ensure_uint8_binary(edge_mask))

    if not boundaries:
        raise ValueError("At least one mask must be provided.")

    out = np.zeros_like(boundaries[0], dtype=np.uint8)

    for b in boundaries:
        if b.shape != out.shape:
            b = cv2.resize(b, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, b)

    return ensure_uint8_binary(out)


def postprocess_edge_mask(
    edge_mask: np.ndarray,
    min_area: int = 16,
    open_kernel: int = 3,
    close_kernel: int = 3,
) -> np.ndarray:

    m = ensure_uint8_binary(edge_mask)
    m = remove_small_components(m, min_area=min_area)
    m = open_mask(m, kernel_size=open_kernel)
    m = close_mask(m, kernel_size=close_kernel)

    return ensure_uint8_binary(m)


def edge_mask_to_contours(
    edge_mask: np.ndarray,
    min_length: int = 10,
) -> list[np.ndarray]:

    m = ensure_uint8_binary(edge_mask)

    contours, _ = cv2.findContours(
        m,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    filtered = []

    for contour in contours:
        length = cv2.arcLength(contour, closed=False)

        if length >= min_length:
            filtered.append(contour)

    return filtered