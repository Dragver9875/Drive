from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from drive_kd.visualization.error_maps import binary_error_map, overlay_error_map
from drive_kd.visualization.overlays import (
    denormalize_image,
    overlay_mask,
    save_prediction_overlay,
    save_rgb_image,
)
from drive_kd.visualization.save_grid import save_image_grid


def extract_probabilities(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    result = {}

    if "road_prob" in outputs:
        result["road"] = outputs["road_prob"]
    else:
        result["road"] = torch.softmax(outputs["road_logits"], dim=1)[:, 1]

    if "lane_prob" in outputs:
        result["lane"] = outputs["lane_prob"]
    else:
        result["lane"] = torch.softmax(outputs["lane_logits"], dim=1)[:, 1]

    if "edge_prob" in outputs:
        result["edge"] = outputs["edge_prob"]
    else:
        result["edge"] = torch.sigmoid(outputs["edge_logits"]).squeeze(1)

    return result


def make_single_sample_teacher_student_panel(
    *,
    image: torch.Tensor,
    gt_road: torch.Tensor,
    gt_lane: torch.Tensor,
    gt_edge: torch.Tensor,
    teacher_probs: dict[str, torch.Tensor],
    student_probs: dict[str, torch.Tensor],
    index: int = 0,
    road_threshold: float = 0.50,
    lane_threshold: float = 0.35,
    edge_threshold: float = 0.40,
) -> list[np.ndarray]:

    img = image[index]
    base = denormalize_image(img)

    gt_overlay = base.copy()
    gt_overlay = overlay_mask(gt_overlay, gt_road[index], color=(0, 180, 0), alpha=0.45, threshold=0.5)
    gt_overlay = overlay_mask(gt_overlay, gt_lane[index], color=(0, 120, 255), alpha=0.55, threshold=0.5)
    gt_overlay = overlay_mask(gt_overlay, gt_edge[index], color=(255, 0, 0), alpha=0.60, threshold=0.5)

    teacher_overlay = base.copy()
    teacher_overlay = overlay_mask(
        teacher_overlay,
        teacher_probs["road"][index],
        color=(0, 180, 0),
        alpha=0.45,
        threshold=road_threshold,
    )
    teacher_overlay = overlay_mask(
        teacher_overlay,
        teacher_probs["lane"][index],
        color=(0, 120, 255),
        alpha=0.55,
        threshold=lane_threshold,
    )
    teacher_overlay = overlay_mask(
        teacher_overlay,
        teacher_probs["edge"][index],
        color=(255, 0, 0),
        alpha=0.60,
        threshold=edge_threshold,
    )

    student_overlay = base.copy()
    student_overlay = overlay_mask(
        student_overlay,
        student_probs["road"][index],
        color=(0, 180, 0),
        alpha=0.45,
        threshold=road_threshold,
    )
    student_overlay = overlay_mask(
        student_overlay,
        student_probs["lane"][index],
        color=(0, 120, 255),
        alpha=0.55,
        threshold=lane_threshold,
    )
    student_overlay = overlay_mask(
        student_overlay,
        student_probs["edge"][index],
        color=(255, 0, 0),
        alpha=0.60,
        threshold=edge_threshold,
    )

    road_err = overlay_error_map(
        base,
        binary_error_map(student_probs["road"][index], gt_road[index], threshold=road_threshold),
        alpha=0.75,
    )

    lane_err = overlay_error_map(
        base,
        binary_error_map(student_probs["lane"][index], gt_lane[index], threshold=lane_threshold),
        alpha=0.75,
    )

    edge_err = overlay_error_map(
        base,
        binary_error_map(student_probs["edge"][index], gt_edge[index], threshold=edge_threshold),
        alpha=0.75,
    )

    return [
        base,
        gt_overlay,
        teacher_overlay,
        student_overlay,
        road_err,
        lane_err,
        edge_err,
    ]


def save_teacher_student_comparison(
    *,
    path: str | Path,
    batch: dict,
    teacher_outputs: dict[str, torch.Tensor] | None,
    student_outputs: dict[str, torch.Tensor],
    max_samples: int = 2,
    road_threshold: float = 0.50,
    lane_threshold: float = 0.35,
    edge_threshold: float = 0.40,
) -> Path:

    image = batch["image"].detach().cpu()
    gt_road = batch["road_mask"].detach().cpu()
    gt_lane = batch["lane_mask"].detach().cpu()
    gt_edge = batch["edge_mask"].detach().cpu()

    student_probs = extract_probabilities(
        {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in student_outputs.items()}
    )

    if teacher_outputs is not None:
        teacher_probs = extract_probabilities(
            {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in teacher_outputs.items()}
        )
    else:
        teacher_probs = {
            "road": batch["teacher_road_prob"].detach().cpu(),
            "lane": batch["teacher_lane_prob"].detach().cpu(),
            "edge": batch["teacher_edge_prob"].detach().cpu(),
        }

    panels = []

    n = min(max_samples, image.shape[0])

    for i in range(n):
        panels.extend(
            make_single_sample_teacher_student_panel(
                image=image,
                gt_road=gt_road,
                gt_lane=gt_lane,
                gt_edge=gt_edge,
                teacher_probs=teacher_probs,
                student_probs=student_probs,
                index=i,
                road_threshold=road_threshold,
                lane_threshold=lane_threshold,
                edge_threshold=edge_threshold,
            )
        )

    return save_image_grid(
        path=path,
        images=panels,
        ncols=7,
        title="original | GT | teacher | student | road error | lane error | edge error",
    )