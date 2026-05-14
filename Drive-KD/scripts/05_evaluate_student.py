from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drive_kd.datasets import DriveKDDataset, drive_kd_collate
from drive_kd.engine import evaluate_model, load_checkpoint
from drive_kd.metrics import DriveEvaluator
from drive_kd.models.students import build_student
from drive_kd.utils import get_amp_enabled, get_device, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Drive-EffB0-BiFPN-KD student.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp_002_student_effb0_full_kd.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/student_effb0_kd/checkpoints/best.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, include_linked_configs=True)

    dataset_cfg = cfg["resolved"]["dataset"]["dataset"]
    student_cfg = cfg["resolved"]["student"]

    device = get_device()
    amp_enabled = get_amp_enabled(device, bool(cfg["training"].get("amp", True)))

    val_ds = DriveKDDataset(
        manifest_path=cfg["data"]["val_manifest"],
        image_height=int(dataset_cfg["image_height"]),
        image_width=int(dataset_cfg["image_width"]),
        mean=dataset_cfg["normalization"]["mean"],
        std=dataset_cfg["normalization"]["std"],
        train=False,
        transform_config=dataset_cfg.get("transforms", {}),
        load_attention=True,
    )

    num_workers = int(cfg["training"].get("num_workers", 2))
    kwargs = {
        "batch_size": int(cfg["training"].get("batch_size", 4)),
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": drive_kd_collate,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    val_loader = DataLoader(val_ds, **kwargs)

    model = build_student(student_cfg).to(device)

    load_checkpoint(
        path=args.checkpoint,
        model=model,
        map_location=device,
        strict=True,
    )

    evaluator = DriveEvaluator.from_config(cfg)

    output_json = Path("outputs/evaluation/metrics_student_effb0_kd.json")
    metrics = evaluate_model(
        model=model,
        loader=val_loader,
        evaluator=evaluator,
        device=device,
        amp_enabled=amp_enabled,
        output_json=output_json,
        desc="evaluate student",
    )

    summary = {
        "checkpoint": args.checkpoint,
        "metrics": metrics,
    }

    write_json(summary, "outputs/evaluation/evaluation_summary.json")

    print("[metrics]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"[done] Saved metrics: {output_json}")


if __name__ == "__main__":
    main()