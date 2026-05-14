from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from drive_kd.engine.infer_teacher import run_teacher_inference_batch
from drive_kd.utils.tensor_io import tensor_to_numpy


def probability_to_boundary_uint8(
    *,
    edge_prob: np.ndarray,
    road_prob: np.ndarray | None = None,
    lane_prob: np.ndarray | None = None,
    edge_threshold: float = 0.40,
    road_threshold: float = 0.50,
    lane_threshold: float = 0.35,
    kernel_size: int = 3,
) -> np.ndarray:

    edge_mask = (edge_prob > edge_threshold).astype(np.uint8)

    combined = edge_mask.copy()

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if road_prob is not None:
        road_mask = (road_prob > road_threshold).astype(np.uint8)
        road_grad = cv2.morphologyEx(road_mask, cv2.MORPH_GRADIENT, kernel)
        combined = np.maximum(combined, road_grad)

    if lane_prob is not None:
        lane_mask = (lane_prob > lane_threshold).astype(np.uint8)
        lane_grad = cv2.morphologyEx(lane_mask, cv2.MORPH_GRADIENT, kernel)
        combined = np.maximum(combined, lane_grad)

    return (combined > 0).astype(np.uint8) * 255


def save_teacher_probabilities(
    *,
    path: str | Path,
    road_prob: np.ndarray,
    lane_prob: np.ndarray,
    edge_prob: np.ndarray,
    dtype: str = "float16",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if dtype == "float16" else np.float32

    np.savez_compressed(
        path,
        road_prob=road_prob.astype(np_dtype),
        lane_prob=lane_prob.astype(np_dtype),
        edge_prob=edge_prob.astype(np_dtype),
    )


def save_teacher_attention(
    *,
    path: str | Path,
    attention: dict[str, torch.Tensor],
    sample_index: int,
    dtype: str = "float16",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if dtype == "float16" else np.float32

    arrays = {}

    for key, tensor in attention.items():
        # tensor expected [B,1,h,w] or [B,h,w]
        item = tensor[sample_index].detach().cpu().float().numpy()

        if item.ndim == 3 and item.shape[0] == 1:
            item = item[0]

        arrays[key] = item.astype(np_dtype)

    np.savez_compressed(path, **arrays)


@torch.no_grad()
def generate_teacher_cache(
    *,
    model: torch.nn.Module,
    loader,
    split: str,
    output_root: str | Path,
    device: torch.device | str,
    amp_enabled: bool = True,
    overwrite_existing: bool = False,
    probability_dtype: str = "float16",
    attention_dtype: str = "float16",
    save_probabilities: bool = True,
    save_boundaries: bool = True,
    save_attention: bool = True,
    road_threshold: float = 0.50,
    lane_threshold: float = 0.35,
    edge_threshold: float = 0.40,
    boundary_kernel_size: int = 3,
) -> dict[str, Any]:

    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    output_root = Path(output_root)

    prob_dir = output_root / "probabilities" / split
    boundary_dir = output_root / "boundaries" / split
    attention_dir = output_root / "attention" / split
    reports_dir = output_root / "reports"

    prob_dir.mkdir(parents=True, exist_ok=True)
    boundary_dir.mkdir(parents=True, exist_ok=True)
    attention_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    report: dict[str, Any] = {
        "split": split,
        "num_seen": 0,
        "num_probability_saved": 0,
        "num_boundary_saved": 0,
        "num_attention_saved": 0,
        "num_skipped_existing": 0,
        "missing_attention_batches": 0,
        "output_root": str(output_root),
    }

    pbar = tqdm(loader, desc=f"teacher cache {split}")

    for batch in pbar:
        image_ids = batch["image_id"]

        outputs = run_teacher_inference_batch(
            model=model,
            batch=batch,
            device=device,
            amp_enabled=amp_enabled,
            return_attention=save_attention,
        )

        road_prob_b = tensor_to_numpy(outputs["road_prob"], dtype=np.float32)
        lane_prob_b = tensor_to_numpy(outputs["lane_prob"], dtype=np.float32)
        edge_prob_b = tensor_to_numpy(outputs["edge_prob"], dtype=np.float32)

        attention = outputs.get("attention", {})

        if save_attention and not attention:
            report["missing_attention_batches"] += 1

        batch_size = len(image_ids)

        for i in range(batch_size):
            image_id = str(image_ids[i])

            prob_path = prob_dir / f"{image_id}.npz"
            boundary_path = boundary_dir / f"{image_id}.png"
            attention_path = attention_dir / f"{image_id}.npz"

            all_existing = True

            if save_probabilities and not prob_path.exists():
                all_existing = False

            if save_boundaries and not boundary_path.exists():
                all_existing = False

            if save_attention and not attention_path.exists():
                all_existing = False

            if all_existing and not overwrite_existing:
                report["num_skipped_existing"] += 1
                report["num_seen"] += 1
                continue

            road_prob = road_prob_b[i]
            lane_prob = lane_prob_b[i]
            edge_prob = edge_prob_b[i]

            if save_probabilities:
                save_teacher_probabilities(
                    path=prob_path,
                    road_prob=road_prob,
                    lane_prob=lane_prob,
                    edge_prob=edge_prob,
                    dtype=probability_dtype,
                )
                report["num_probability_saved"] += 1

            if save_boundaries:
                boundary = probability_to_boundary_uint8(
                    edge_prob=edge_prob,
                    road_prob=road_prob,
                    lane_prob=lane_prob,
                    edge_threshold=edge_threshold,
                    road_threshold=road_threshold,
                    lane_threshold=lane_threshold,
                    kernel_size=boundary_kernel_size,
                )
                cv2.imwrite(str(boundary_path), boundary)
                report["num_boundary_saved"] += 1

            if save_attention and attention:
                save_teacher_attention(
                    path=attention_path,
                    attention=attention,
                    sample_index=i,
                    dtype=attention_dtype,
                )
                report["num_attention_saved"] += 1

            report["num_seen"] += 1

        pbar.set_postfix(
            {
                "seen": report["num_seen"],
                "prob": report["num_probability_saved"],
                "boundary": report["num_boundary_saved"],
                "attention": report["num_attention_saved"],
            }
        )

    report_path = reports_dir / f"teacher_cache_{split}_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report