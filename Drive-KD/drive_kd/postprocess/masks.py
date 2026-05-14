from __future__ import annotations

import numpy as np
import torch

from drive_kd.postprocess.boundaries import postprocess_edge_mask
from drive_kd.postprocess.lanes import clean_lane_mask
from drive_kd.postprocess.morphology import (
    close_mask,
    fill_small_holes,
    remove_small_components,
)
from drive_kd.postprocess.thresholds import DriveThresholds


def logits_to_probabilities(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    out = {}

    if "road_prob" in outputs:
        out["road_prob"] = outputs["road_prob"]
    else:
        out["road_prob"] = torch.softmax(outputs["road_logits"], dim=1)[:, 1]

    if "lane_prob" in outputs:
        out["lane_prob"] = outputs["lane_prob"]
    else:
        out["lane_prob"] = torch.softmax(outputs["lane_logits"], dim=1)[:, 1]

    if "edge_prob" in outputs:
        out["edge_prob"] = outputs["edge_prob"]
    else:
        out["edge_prob"] = torch.sigmoid(outputs["edge_logits"]).squeeze(1)

    return out


def tensor_prob_to_uint8_mask(
    prob: torch.Tensor | np.ndarray,
    threshold: float,
) -> np.ndarray:

    if isinstance(prob, torch.Tensor):
        arr = prob.detach().cpu().float().numpy()
    else:
        arr = prob.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unsupported probability shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D probability map, got: {arr.shape}")

    return (arr > float(threshold)).astype(np.uint8) * 255


def probabilities_to_binary_masks(
    probs: dict[str, torch.Tensor | np.ndarray],
    thresholds: DriveThresholds | None = None,
) -> dict[str, np.ndarray]:

    thresholds = thresholds or DriveThresholds()

    return {
        "road_mask": tensor_prob_to_uint8_mask(probs["road_prob"], thresholds.road),
        "lane_mask": tensor_prob_to_uint8_mask(probs["lane_prob"], thresholds.lane),
        "edge_mask": tensor_prob_to_uint8_mask(probs["edge_prob"], thresholds.edge),
    }


def clean_road_mask(
    road_mask: np.ndarray,
    min_area: int = 2048,
    close_kernel: int = 7,
    fill_holes: bool = True,
) -> np.ndarray:

    m = remove_small_components(road_mask, min_area=min_area)
    m = close_mask(m, kernel_size=close_kernel)

    if fill_holes:
        m = fill_small_holes(m, max_hole_area=512)

    return m


def postprocess_prediction_dict(
    outputs: dict[str, torch.Tensor],
    thresholds: DriveThresholds | None = None,
    clean: bool = True,
    sample_index: int = 0,
) -> dict[str, np.ndarray]:

    thresholds = thresholds or DriveThresholds()
    probs = logits_to_probabilities(outputs)

    single_probs = {
        "road_prob": probs["road_prob"][sample_index],
        "lane_prob": probs["lane_prob"][sample_index],
        "edge_prob": probs["edge_prob"][sample_index],
    }

    masks = probabilities_to_binary_masks(single_probs, thresholds=thresholds)

    if clean:
        masks["road_mask"] = clean_road_mask(masks["road_mask"])
        masks["lane_mask"] = clean_lane_mask(masks["lane_mask"])
        masks["edge_mask"] = postprocess_edge_mask(masks["edge_mask"])

    return masks