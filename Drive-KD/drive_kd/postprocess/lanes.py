from __future__ import annotations

import cv2
import numpy as np

from drive_kd.postprocess.morphology import (
    close_mask,
    ensure_uint8_binary,
    keep_largest_components,
    open_mask,
    remove_small_components,
)


def clean_lane_mask(
    lane_mask: np.ndarray,
    min_area: int = 24,
    open_kernel: int = 3,
    close_kernel: int = 5,
) -> np.ndarray:

    m = ensure_uint8_binary(lane_mask)
    m = remove_small_components(m, min_area=min_area)
    m = open_mask(m, kernel_size=open_kernel)
    m = close_mask(m, kernel_size=close_kernel)

    return ensure_uint8_binary(m)


def keep_largest_lane_components(
    lane_mask: np.ndarray,
    k: int = 12,
) -> np.ndarray:

    return keep_largest_components(lane_mask, k=k)


def thin_lane_mask(
    lane_mask: np.ndarray,
    max_iterations: int = 64,
) -> np.ndarray:

    img = ensure_uint8_binary(lane_mask)
    img = (img > 0).astype(np.uint8)

    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    iterations = 0

    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        iterations += 1

        if cv2.countNonZero(img) == 0:
            break

        if iterations >= max_iterations:
            break

    return skeleton.astype(np.uint8) * 255


def lane_mask_to_polylines(
    lane_mask: np.ndarray,
    min_length: int = 20,
    epsilon_ratio: float = 0.01,
) -> list[np.ndarray]:

    m = clean_lane_mask(lane_mask)

    contours, _ = cv2.findContours(
        m,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    polylines = []

    for contour in contours:
        length = cv2.arcLength(contour, closed=False)

        if length < min_length:
            continue

        epsilon = float(epsilon_ratio) * length
        approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=False)

        polyline = approx.reshape(-1, 2)

        if len(polyline) >= 2:
            polylines.append(polyline)

    return polylines


def draw_lane_polylines(
    image_rgb: np.ndarray,
    polylines: list[np.ndarray],
    color: tuple[int, int, int] = (0, 120, 255),
    thickness: int = 2,
) -> np.ndarray:

    out = image_rgb.copy()

    for polyline in polylines:
        pts = polyline.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            out,
            [pts],
            isClosed=False,
            color=color[::-1],
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return out