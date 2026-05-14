from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from drive_kd.datasets.drive_supervised_dataset import DriveSupervisedDataset
from drive_kd.datasets.manifest_utils import read_manifest, required_columns
from drive_kd.datasets.transforms import build_drive_transforms


class DriveKDDataset(DriveSupervisedDataset):

    REQUIRED_KD_COLUMNS = [
        "image_id",
        "image_path",
        "road_mask_path",
        "lane_mask_path",
        "edge_mask_path",
        "teacher_prob_path",
        "teacher_boundary_path",
        "teacher_attention_path",
        "split",
        "height",
        "width",
    ]

    def __init__(
        self,
        manifest_path: str | Path,
        image_height: int,
        image_width: int,
        mean: list[float],
        std: list[float],
        train: bool,
        transform_config: dict[str, Any] | None = None,
        load_attention: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.df = read_manifest(self.manifest_path).reset_index(drop=True)

        required_columns(self.df, self.REQUIRED_KD_COLUMNS, manifest_name=str(self.manifest_path))

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.mean = mean
        self.std = std
        self.train = bool(train)
        self.load_attention = bool(load_attention)

        self.transforms = build_drive_transforms(
            image_height=self.image_height,
            image_width=self.image_width,
            mean=self.mean,
            std=self.std,
            train=self.train,
            transform_config=transform_config,
            include_teacher_targets=True,
        )

    @staticmethod
    def read_teacher_probs(path: str | Path) -> dict[str, np.ndarray]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Teacher probability cache not found: {path}")

        data = np.load(path)

        required = ["road_prob", "lane_prob", "edge_prob"]
        missing = [k for k in required if k not in data.files]

        if missing:
            raise KeyError(f"Teacher probability cache {path} missing keys: {missing}")

        return {
            "road_prob": data["road_prob"].astype(np.float32),
            "lane_prob": data["lane_prob"].astype(np.float32),
            "edge_prob": data["edge_prob"].astype(np.float32),
        }

    @staticmethod
    def read_teacher_boundary(path: str | Path) -> np.ndarray:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Teacher boundary cache not found: {path}")

        boundary = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if boundary is None:
            raise FileNotFoundError(f"Could not read teacher boundary: {path}")

        return (boundary > 127).astype(np.float32)

    @staticmethod
    def read_teacher_attention(path: str | Path) -> dict[str, torch.Tensor]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Teacher attention cache not found: {path}")

        data = np.load(path)
        out: dict[str, torch.Tensor] = {}

        for key in data.files:
            arr = data[key].astype(np.float32)

            if arr.ndim == 2:
                arr = arr[None, :, :]
            elif arr.ndim == 3 and arr.shape[0] != 1:
                # Keep [C,H,W] attention-like tensors if provided.
                pass
            elif arr.ndim != 3:
                raise ValueError(
                    f"Unsupported attention shape for key={key}, path={path}: {arr.shape}"
                )

            out[key] = torch.from_numpy(arr).float()

        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        image = self.read_rgb(row["image_path"])

        road_mask = self.read_mask_binary(row["road_mask_path"])
        lane_mask = self.read_mask_binary(row["lane_mask_path"])
        edge_mask = self.read_mask_binary(row["edge_mask_path"])

        teacher_probs = self.read_teacher_probs(row["teacher_prob_path"])
        teacher_boundary = self.read_teacher_boundary(row["teacher_boundary_path"])

        teacher_road_prob = teacher_probs["road_prob"]
        teacher_lane_prob = teacher_probs["lane_prob"]
        teacher_edge_prob = teacher_probs["edge_prob"]

        if image.shape[:2] != road_mask.shape[:2]:
            image = cv2.resize(
                image,
                (road_mask.shape[1], road_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        expected_shape = road_mask.shape[:2]

        def resize_map_if_needed(x: np.ndarray, interpolation: int) -> np.ndarray:
            if x.shape[:2] == expected_shape:
                return x
            return cv2.resize(
                x,
                (expected_shape[1], expected_shape[0]),
                interpolation=interpolation,
            )

        lane_mask = resize_map_if_needed(lane_mask, cv2.INTER_NEAREST)
        edge_mask = resize_map_if_needed(edge_mask, cv2.INTER_NEAREST)

        teacher_road_prob = resize_map_if_needed(teacher_road_prob, cv2.INTER_LINEAR)
        teacher_lane_prob = resize_map_if_needed(teacher_lane_prob, cv2.INTER_LINEAR)
        teacher_edge_prob = resize_map_if_needed(teacher_edge_prob, cv2.INTER_LINEAR)
        teacher_boundary = resize_map_if_needed(teacher_boundary, cv2.INTER_NEAREST)

        transformed = self.transforms(
            image=image,
            road_mask=road_mask,
            lane_mask=lane_mask,
            edge_mask=edge_mask,
            teacher_road_prob=teacher_road_prob,
            teacher_lane_prob=teacher_lane_prob,
            teacher_edge_prob=teacher_edge_prob,
            teacher_boundary=teacher_boundary,
        )

        image_t = transformed["image"].float()

        road_t = transformed["road_mask"].long()
        lane_t = transformed["lane_mask"].long()
        edge_t = transformed["edge_mask"].float().unsqueeze(0)

        teacher_road_t = transformed["teacher_road_prob"].float().clamp(0.0, 1.0)
        teacher_lane_t = transformed["teacher_lane_prob"].float().clamp(0.0, 1.0)
        teacher_edge_t = transformed["teacher_edge_prob"].float().clamp(0.0, 1.0)
        teacher_boundary_t = transformed["teacher_boundary"].float().unsqueeze(0).clamp(0.0, 1.0)

        teacher_attention: dict[str, torch.Tensor] = {}

        if self.load_attention:
            teacher_attention = self.read_teacher_attention(row["teacher_attention_path"])

        return {
            "image": image_t,
            "road_mask": road_t,
            "lane_mask": lane_t,
            "edge_mask": edge_t,
            "teacher_road_prob": teacher_road_t,
            "teacher_lane_prob": teacher_lane_t,
            "teacher_edge_prob": teacher_edge_t,
            "teacher_boundary": teacher_boundary_t,
            "teacher_attention": teacher_attention,
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "teacher_prob_path": str(row["teacher_prob_path"]),
            "teacher_boundary_path": str(row["teacher_boundary_path"]),
            "teacher_attention_path": str(row["teacher_attention_path"]),
        }

    def sample_summary(self, idx: int = 0) -> dict[str, Any]:
        sample = self[idx]

        return {
            "image_id": sample["image_id"],
            "image_shape": list(sample["image"].shape),
            "road_shape": list(sample["road_mask"].shape),
            "lane_shape": list(sample["lane_mask"].shape),
            "edge_shape": list(sample["edge_mask"].shape),
            "teacher_road_shape": list(sample["teacher_road_prob"].shape),
            "teacher_lane_shape": list(sample["teacher_lane_prob"].shape),
            "teacher_edge_shape": list(sample["teacher_edge_prob"].shape),
            "teacher_boundary_shape": list(sample["teacher_boundary"].shape),
            "attention_keys": sorted(list(sample["teacher_attention"].keys())),
            "road_sum": int(sample["road_mask"].sum().item()),
            "lane_sum": int(sample["lane_mask"].sum().item()),
            "edge_sum": float(sample["edge_mask"].sum().item()),
            "teacher_road_mean": float(sample["teacher_road_prob"].mean().item()),
            "teacher_lane_mean": float(sample["teacher_lane_prob"].mean().item()),
            "teacher_edge_mean": float(sample["teacher_edge_prob"].mean().item()),
        }