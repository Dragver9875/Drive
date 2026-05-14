from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drive_kd.engine import load_checkpoint
from drive_kd.models.students import build_student
from drive_kd.utils import get_device, load_config, write_json


class ExportWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(
            x,
            return_features=False,
            return_attention=False,
            return_probabilities=False,
        )
        return out["road_logits"], out["lane_logits"], out["edge_logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Drive-EffB0-BiFPN-KD student.")
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
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["pt", "torchscript", "onnx", "all"],
    )
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=640)
    return parser.parse_args()


def save_plain_pt(model: nn.Module, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def export_torchscript(wrapper: nn.Module, example: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapper.eval()

    traced = torch.jit.trace(wrapper, example)
    traced.save(str(path))


def export_onnx(wrapper: nn.Module, example: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapper.eval()

    torch.onnx.export(
        wrapper,
        example,
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["road_logits", "lane_logits", "edge_logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "road_logits": {0: "batch"},
            "lane_logits": {0: "batch"},
            "edge_logits": {0: "batch"},
        },
    )


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

    model.eval()

    wrapper = ExportWrapper(model).to(device).eval()

    example = torch.randn(
        1,
        3,
        args.image_height,
        args.image_width,
        device=device,
    )

    out_dir = Path("outputs/exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "name": "Drive-EffB0-BiFPN-KD",
        "checkpoint": args.checkpoint,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "outputs": ["road_logits", "lane_logits", "edge_logits"],
    }

    exported = {}

    if args.format in {"pt", "all"}:
        pt_path = out_dir / "drive_effb0_bifpn_kd_best.pt"
        save_plain_pt(model, pt_path, metadata)
        exported["pt"] = str(pt_path)

    if args.format in {"torchscript", "all"}:
        ts_path = out_dir / "drive_effb0_bifpn_kd.torchscript.pt"
        export_torchscript(wrapper, example, ts_path)
        exported["torchscript"] = str(ts_path)

    if args.format in {"onnx", "all"}:
        onnx_path = out_dir / "drive_effb0_bifpn_kd.onnx"
        export_onnx(wrapper, example, onnx_path)
        exported["onnx"] = str(onnx_path)

    model_card = f"""# Drive-EffB0-BiFPN-KD

## Architecture

EfficientNet-B0 encoder + SPPF + BiFPN-lite + road/lane/edge heads.

## Input

- Shape: `[B, 3, {args.image_height}, {args.image_width}]`
- Normalization: ImageNet mean/std

## Outputs

1. `road_logits`: `[B, 2, H, W]`
2. `lane_logits`: `[B, 2, H, W]`
3. `edge_logits`: `[B, 1, H, W]`

## Source checkpoint

`{args.checkpoint}`
"""

    model_card_path = out_dir / "model_card.md"
    model_card_path.write_text(model_card, encoding="utf-8")
    exported["model_card"] = str(model_card_path)

    write_json(
        {
            "metadata": metadata,
            "exported": exported,
        },
        out_dir / "export_report.json",
    )

    print("[exported]")
    for k, v in exported.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()