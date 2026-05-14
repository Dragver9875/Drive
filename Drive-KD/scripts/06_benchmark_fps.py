from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drive_kd.engine import load_checkpoint
from drive_kd.metrics import benchmark_fps, count_parameters, estimate_model_size_mb
from drive_kd.models.students import build_student
from drive_kd.utils import get_device, load_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Drive-EffB0-BiFPN-KD FPS.")
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, include_linked_configs=True)

    student_cfg = cfg["resolved"]["student"]

    device = get_device()

    model = build_student(student_cfg).to(device)

    load_checkpoint(
        path=args.checkpoint,
        model=model,
        map_location=device,
        strict=True,
    )

    params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    result = benchmark_fps(
        model=model,
        image_height=args.image_height,
        image_width=args.image_width,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        device=device,
        use_amp=True,
    )

    report = {
        "checkpoint": args.checkpoint,
        "params": params,
        "trainable_params": trainable_params,
        "model_size_fp32_mb": estimate_model_size_mb(model, bytes_per_param=4),
        "model_size_fp16_mb": estimate_model_size_mb(model, bytes_per_param=2),
        **result.as_dict(),
    }

    out_path = Path("outputs/evaluation/fps_report.json")
    write_json(report, out_path)

    print("[fps report]")
    for k, v in report.items():
        print(f"  {k}: {v}")

    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()