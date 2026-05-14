from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from drive_kd.datasets.manifest_utils import read_manifest, required_columns
from drive_kd.datasets.transforms import build_drive_transforms


class DriveSupervisedDataset(Dataset):
    """
    Dataset for supervised road/lane/edge training.

    Used for:
      1. SegFormer-B1 teacher training.
      2. Evaluation.
      3. Optional non-KD student training.

    Returns:
      image:      FloatTensor [3, H, W]
      road_mask:  LongTensor  [H, W]
      lane_mask:  LongTensor  [H, W]
      edge_mask:  FloatTensor [1, H, W]
    """

    REQUIRED_COLUMNS = [
        "image_id",
        "image_path",
        "road_mask_path",
        "lane_mask_path",
        "edge_mask_path",
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
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.df = read_manifest(self.manifest_path).reset_index(drop=True)

        required_columns(self.df, self.REQUIRED_COLUMNS, manifest_name=str(self.manifest_path))

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.mean = mean
        self.std = std
        self.train = bool(train)

        self.transforms = build_drive_transforms(
            image_height=self.image_height,
            image_width=self.image_width,
            mean=self.mean,
            std=self.std,
            train=self.train,
            transform_config=transform_config,
            include_teacher_targets=False,
        )

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def read_rgb(path: str | Path) -> np.ndarray:
        path = Path(path)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def read_mask_binary(path: str | Path) -> np.ndarray:
        path = Path(path)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {path}")

        return (mask > 127).astype(np.uint8)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        image = self.read_rgb(row["image_path"])
        road_mask = self.read_mask_binary(row["road_mask_path"])
        lane_mask = self.read_mask_binary(row["lane_mask_path"])
        edge_mask = self.read_mask_binary(row["edge_mask_path"])

        # Raw images can be 720x1280 while masks are already 384x640.
        # Albumentations checks shape consistency before transforms.
        if image.shape[:2] != road_mask.shape[:2]:
            image = cv2.resize(
                image,
                (road_mask.shape[1], road_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        transformed = self.transforms(
            image=image,
            road_mask=road_mask,
            lane_mask=lane_mask,
            edge_mask=edge_mask,
        )

        image_t = transformed["image"].float()
        road_t = transformed["road_mask"].long()
        lane_t = transformed["lane_mask"].long()
        edge_t = transformed["edge_mask"].float().unsqueeze(0)

        return {
            "image": image_t,
            "road_mask": road_t,
            "lane_mask": lane_t,
            "edge_mask": edge_t,
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
        }

    def sample_summary(self, idx: int = 0) -> dict[str, Any]:
        sample = self[idx]

        return {
            "image_id": sample["image_id"],
            "image_shape": list(sample["image"].shape),
            "road_shape": list(sample["road_mask"].shape),
            "lane_shape": list(sample["lane_mask"].shape),
            "edge_shape": list(sample["edge_mask"].shape),
            "road_sum": int(sample["road_mask"].sum().item()),
            "lane_sum": int(sample["lane_mask"].sum().item()),
            "edge_sum": float(sample["edge_mask"].sum().item()),
        }