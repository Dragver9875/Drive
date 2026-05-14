from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drive_kd.datasets import DriveSupervisedDataset, drive_supervised_collate
from drive_kd.datasets.manifest_utils import load_yaml, read_manifest
from drive_kd.datasets.path_resolver import DatasetPathResolver
from drive_kd.utils.file_io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Drive-KD dataset integrity.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/bdd100k_local.yaml",
        help="Dataset config path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of samples to inspect for positive mask ratios.",
    )
    return parser.parse_args()


def check_manifest_paths(df: pd.DataFrame, name: str) -> dict:
    required = [
        "image_path",
        "road_mask_path",
        "lane_mask_path",
        "edge_mask_path",
    ]

    report = {
        "name": name,
        "rows": len(df),
        "missing": {},
    }

    for col in required:
        missing = 0

        for value in df[col].head(min(len(df), 5000)):
            if not Path(str(value)).exists():
                missing += 1

        report["missing"][col] = missing

    return report


def estimate_positive_ratios(dataset: DriveSupervisedDataset, num_samples: int) -> dict:
    n = min(num_samples, len(dataset))

    road_sum = 0.0
    lane_sum = 0.0
    edge_sum = 0.0
    pixels = 0.0

    for idx in range(n):
        sample = dataset[idx]

        road = sample["road_mask"]
        lane = sample["lane_mask"]
        edge = sample["edge_mask"]

        road_sum += float(road.sum().item())
        lane_sum += float(lane.sum().item())
        edge_sum += float(edge.sum().item())

        pixels += float(road.numel())

    return {
        "samples": n,
        "road_pos_ratio": road_sum / max(1.0, pixels),
        "lane_pos_ratio": lane_sum / max(1.0, pixels),
        "edge_pos_ratio": edge_sum / max(1.0, pixels),
    }


def main() -> None:
    args = parse_args()

    resolver = DatasetPathResolver(args.config)
    dataset_cfg = load_yaml(args.config)["dataset"]

    train_manifest = resolver.train_absolute_manifest
    val_manifest = resolver.val_absolute_manifest

    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Train absolute manifest not found: {train_manifest}\n"
            "Run scripts/00_build_absolute_manifests.py first."
        )

    if not val_manifest.exists():
        raise FileNotFoundError(
            f"Val absolute manifest not found: {val_manifest}\n"
            "Run scripts/00_build_absolute_manifests.py first."
        )

    train_df = read_manifest(train_manifest)
    val_df = read_manifest(val_manifest)

    print("[manifest rows]")
    print("  train:", len(train_df))
    print("  val:  ", len(val_df))

    expected = dataset_cfg.get("expected_counts", {})
    train_min = int(expected.get("train_min_rows", 60000))
    val_min = int(expected.get("val_min_rows", 9000))

    if len(train_df) < train_min:
        raise RuntimeError(f"Train rows too low: {len(train_df)} < {train_min}")

    if len(val_df) < val_min:
        raise RuntimeError(f"Val rows too low: {len(val_df)} < {val_min}")

    train_path_report = check_manifest_paths(train_df, "train")
    val_path_report = check_manifest_paths(val_df, "val")

    print("[path check]")
    print(json.dumps(train_path_report, indent=2))
    print(json.dumps(val_path_report, indent=2))

    train_ds = DriveSupervisedDataset(
        manifest_path=train_manifest,
        image_height=resolver.image_height,
        image_width=resolver.image_width,
        mean=resolver.mean,
        std=resolver.std,
        train=True,
        transform_config=dataset_cfg.get("transforms", {}),
    )

    val_ds = DriveSupervisedDataset(
        manifest_path=val_manifest,
        image_height=resolver.image_height,
        image_width=resolver.image_width,
        mean=resolver.mean,
        std=resolver.std,
        train=False,
        transform_config=dataset_cfg.get("transforms", {}),
    )

    print("[sample summaries]")
    print("  train:", train_ds.sample_summary(0))
    print("  val:  ", val_ds.sample_summary(0))

    batch_size = 2
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=drive_supervised_collate,
    )

    batch = next(iter(loader))

    print("[batch shapes]")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")

    stats = {
        "train": estimate_positive_ratios(train_ds, args.num_samples),
        "val": estimate_positive_ratios(val_ds, min(args.num_samples, 128)),
    }

    print("[positive ratios]")
    print(json.dumps(stats, indent=2))

    if stats["train"]["road_pos_ratio"] <= 0:
        raise RuntimeError("Train road positive ratio is zero.")

    if stats["train"]["lane_pos_ratio"] <= 0:
        raise RuntimeError("Train lane positive ratio is zero.")

    if stats["train"]["edge_pos_ratio"] <= 0:
        raise RuntimeError("Train edge positive ratio is zero.")

    out = {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "path_report": {
            "train": train_path_report,
            "val": val_path_report,
        },
        "positive_ratios": stats,
    }

    write_json(out, "outputs/evaluation/dataset_check_report.json")

    print("[done] Dataset check passed.")


if __name__ == "__main__":
    main()