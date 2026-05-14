from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drive_kd.datasets import TeacherCacheDataset, drive_supervised_collate
from drive_kd.datasets.manifest_utils import build_kd_manifest, load_yaml
from drive_kd.engine import generate_teacher_cache, load_checkpoint
from drive_kd.models.teachers import build_teacher
from drive_kd.utils import get_amp_enabled, get_device, load_config, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SegFormer-B1 teacher cache.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp_001_teacher_cache.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override teacher checkpoint path.",
    )
    return parser.parse_args()


def make_loader(dataset, batch_size: int, num_workers: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": drive_supervised_collate,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, include_linked_configs=True)

    dataset_cfg = cfg["resolved"]["dataset"]["dataset"]
    teacher_cfg = cfg["resolved"]["teacher"]
    cache_cfg = cfg["resolved"]["cache"]["teacher_cache"]

    seed_everything(int(cfg["experiment"].get("seed", 42)))

    device = get_device()
    amp_enabled = get_amp_enabled(device, bool(cfg["cache"].get("amp", True)))

    model = build_teacher(teacher_cfg).to(device)

    checkpoint = args.checkpoint or cfg["teacher"].get("checkpoint") or cache_cfg["checkpoint"]["path"]

    print(f"[load teacher checkpoint] {checkpoint}")
    load_checkpoint(
        path=checkpoint,
        model=model,
        map_location=device,
        strict=bool(cache_cfg.get("checkpoint", {}).get("strict", True)),
    )

    output_root = Path(cfg["output"]["cache_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg["cache"].get("batch_size", cache_cfg.get("batch_size", 4)))
    num_workers = int(cfg["cache"].get("num_workers", cache_cfg.get("num_workers", 2)))

    reports = {}

    for split in cfg["cache"].get("splits", ["train", "val"]):
        manifest = dataset_cfg["absolute_manifests"][split]

        dataset = TeacherCacheDataset(
            manifest_path=manifest,
            image_height=int(dataset_cfg["image_height"]),
            image_width=int(dataset_cfg["image_width"]),
            mean=dataset_cfg["normalization"]["mean"],
            std=dataset_cfg["normalization"]["std"],
            split=split,
            transform_config=dataset_cfg.get("transforms", {}),
        )

        loader = make_loader(dataset, batch_size=batch_size, num_workers=num_workers)

        report = generate_teacher_cache(
            model=model,
            loader=loader,
            split=split,
            output_root=output_root,
            device=device,
            amp_enabled=amp_enabled,
            overwrite_existing=bool(cfg["cache"].get("overwrite_existing", False)),
            probability_dtype=str(cfg["cache"].get("probability_dtype", "float16")),
            attention_dtype=str(cfg["cache"].get("attention_dtype", "float16")),
            save_probabilities=bool(cfg["cache"].get("save_probabilities", True)),
            save_boundaries=bool(cfg["cache"].get("save_boundaries", True)),
            save_attention=bool(cfg["cache"].get("save_attention", True)),
            road_threshold=float(cfg["boundary_generation"]["thresholds"].get("road", 0.50)),
            lane_threshold=float(cfg["boundary_generation"]["thresholds"].get("lane", 0.35)),
            edge_threshold=float(cfg["boundary_generation"]["thresholds"].get("edge", 0.40)),
            boundary_kernel_size=int(cfg["boundary_generation"].get("kernel_size", 3)),
        )

        reports[split] = report

    print("[build kd manifests]")

    train_kd = build_kd_manifest(
        absolute_manifest_path=dataset_cfg["absolute_manifests"]["train"],
        output_manifest_path=cfg["kd_manifest"]["train_output"],
        teacher_cache_root=output_root,
        split="train",
        require_attention=bool(cfg["cache"].get("save_attention", True)),
        require_boundary=bool(cfg["cache"].get("save_boundaries", True)),
    )

    val_kd = build_kd_manifest(
        absolute_manifest_path=dataset_cfg["absolute_manifests"]["val"],
        output_manifest_path=cfg["kd_manifest"]["val_output"],
        teacher_cache_root=output_root,
        split="val",
        require_attention=bool(cfg["cache"].get("save_attention", True)),
        require_boundary=bool(cfg["cache"].get("save_boundaries", True)),
    )

    inventory = {
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "reports": reports,
        "train_kd_rows": len(train_kd),
        "val_kd_rows": len(val_kd),
        "train_kd_manifest": str(cfg["kd_manifest"]["train_output"]),
        "val_kd_manifest": str(cfg["kd_manifest"]["val_output"]),
    }

    write_json(inventory, output_root / "reports" / "teacher_cache_inventory.json")

    print(json.dumps(inventory, indent=2))
    print("[done] Teacher cache generation complete.")


if __name__ == "__main__":
    main()