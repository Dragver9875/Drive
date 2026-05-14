from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from drive_kd.datasets import (
    DriveKDDataset,
    DriveSupervisedDataset,
    drive_kd_collate,
    drive_supervised_collate,
)


def write_rgb(path: Path, shape: tuple[int, int] = (64, 96)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 80
    img[..., 1] = 120
    img[..., 2] = 160

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def write_mask(path: Path, shape: tuple[int, int] = (64, 96), kind: str = "road") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if kind == "road":
        mask[h // 2 :, :] = 255
    elif kind == "lane":
        mask[:, w // 2 - 1 : w // 2 + 1] = 255
    elif kind == "edge":
        mask[h // 2 - 1 : h // 2 + 1, :] = 255
    else:
        raise ValueError(kind)

    cv2.imwrite(str(path), mask)


def create_supervised_manifest(tmp_path: Path) -> Path:
    image_path = tmp_path / "images" / "sample.jpg"
    road_path = tmp_path / "road_masks" / "sample.png"
    lane_path = tmp_path / "lane_masks" / "sample.png"
    edge_path = tmp_path / "edge_masks" / "sample.png"

    write_rgb(image_path)
    write_mask(road_path, kind="road")
    write_mask(lane_path, kind="lane")
    write_mask(edge_path, kind="edge")

    manifest = tmp_path / "manifest.csv"

    df = pd.DataFrame(
        [
            {
                "image_id": "sample",
                "image_name": "sample.jpg",
                "image_path": str(image_path),
                "road_mask_path": str(road_path),
                "lane_mask_path": str(lane_path),
                "edge_mask_path": str(edge_path),
                "split": "train",
                "height": 64,
                "width": 96,
                "has_road": 1,
                "has_lane": 1,
                "road_pixels": int((64 // 2) * 96),
                "lane_pixels": int(64 * 2),
                "edge_pixels": int(2 * 96),
            }
        ]
    )

    df.to_csv(manifest, index=False)
    return manifest


def create_kd_manifest(tmp_path: Path, supervised_manifest: Path) -> Path:
    df = pd.read_csv(supervised_manifest)

    cache_root = tmp_path / "cache"
    prob_path = cache_root / "probabilities" / "train" / "sample.npz"
    boundary_path = cache_root / "boundaries" / "train" / "sample.png"
    attention_path = cache_root / "attention" / "train" / "sample.npz"

    prob_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    attention_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = 64, 96

    np.savez_compressed(
        prob_path,
        road_prob=np.random.rand(h, w).astype(np.float16),
        lane_prob=np.random.rand(h, w).astype(np.float16),
        edge_prob=np.random.rand(h, w).astype(np.float16),
    )

    boundary = np.zeros((h, w), dtype=np.uint8)
    boundary[h // 2 - 1 : h // 2 + 1, :] = 255
    cv2.imwrite(str(boundary_path), boundary)

    np.savez_compressed(
        attention_path,
        s2=np.random.rand(h // 4, w // 4).astype(np.float16),
        s4=np.random.rand(h // 16, w // 16).astype(np.float16),
    )

    df["teacher_prob_path"] = str(prob_path)
    df["teacher_boundary_path"] = str(boundary_path)
    df["teacher_attention_path"] = str(attention_path)

    kd_manifest = tmp_path / "manifest_kd.csv"
    df.to_csv(kd_manifest, index=False)

    return kd_manifest


def test_drive_supervised_dataset(tmp_path: Path) -> None:
    manifest = create_supervised_manifest(tmp_path)

    ds = DriveSupervisedDataset(
        manifest_path=manifest,
        image_height=64,
        image_width=96,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        train=False,
        transform_config={},
    )

    sample = ds[0]

    assert sample["image"].shape == (3, 64, 96)
    assert sample["road_mask"].shape == (64, 96)
    assert sample["lane_mask"].shape == (64, 96)
    assert sample["edge_mask"].shape == (1, 64, 96)

    assert sample["road_mask"].sum() > 0
    assert sample["lane_mask"].sum() > 0
    assert sample["edge_mask"].sum() > 0


def test_drive_supervised_collate(tmp_path: Path) -> None:
    manifest = create_supervised_manifest(tmp_path)

    ds = DriveSupervisedDataset(
        manifest_path=manifest,
        image_height=64,
        image_width=96,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        train=False,
        transform_config={},
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=drive_supervised_collate,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (1, 3, 64, 96)
    assert batch["road_mask"].shape == (1, 64, 96)
    assert batch["lane_mask"].shape == (1, 64, 96)
    assert batch["edge_mask"].shape == (1, 1, 64, 96)


def test_drive_kd_dataset(tmp_path: Path) -> None:
    supervised_manifest = create_supervised_manifest(tmp_path)
    kd_manifest = create_kd_manifest(tmp_path, supervised_manifest)

    ds = DriveKDDataset(
        manifest_path=kd_manifest,
        image_height=64,
        image_width=96,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        train=False,
        transform_config={},
        load_attention=True,
    )

    sample = ds[0]

    assert sample["image"].shape == (3, 64, 96)
    assert sample["teacher_road_prob"].shape == (64, 96)
    assert sample["teacher_lane_prob"].shape == (64, 96)
    assert sample["teacher_edge_prob"].shape == (64, 96)
    assert sample["teacher_boundary"].shape == (1, 64, 96)

    assert "s2" in sample["teacher_attention"]
    assert "s4" in sample["teacher_attention"]


def test_drive_kd_collate(tmp_path: Path) -> None:
    supervised_manifest = create_supervised_manifest(tmp_path)
    kd_manifest = create_kd_manifest(tmp_path, supervised_manifest)

    ds = DriveKDDataset(
        manifest_path=kd_manifest,
        image_height=64,
        image_width=96,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        train=False,
        transform_config={},
        load_attention=True,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=drive_kd_collate,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (1, 3, 64, 96)
    assert batch["teacher_road_prob"].shape == (1, 64, 96)
    assert batch["teacher_lane_prob"].shape == (1, 64, 96)
    assert batch["teacher_edge_prob"].shape == (1, 64, 96)
    assert batch["teacher_boundary"].shape == (1, 1, 64, 96)

    assert "s2" in batch["teacher_attention"]
    assert "s4" in batch["teacher_attention"]