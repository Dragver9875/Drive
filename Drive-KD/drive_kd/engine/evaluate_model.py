from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from drive_kd.engine.amp_utils import get_autocast_context
from drive_kd.metrics import DriveEvaluator
from drive_kd.utils.device import move_batch_to_device


@torch.no_grad()
def evaluate_model(
    *,
    model: torch.nn.Module,
    loader,
    evaluator: DriveEvaluator,
    device: torch.device | str,
    amp_enabled: bool = True,
    output_json: str | Path | None = None,
    desc: str = "evaluate",
) -> dict[str, float]:

    device = torch.device(device)

    model.eval()
    evaluator.reset()

    pbar = tqdm(loader, desc=desc)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        with get_autocast_context(enabled=amp_enabled, device=device):
            outputs = model(
                batch["image"],
                return_features=False,
                return_attention=False,
                return_probabilities=False,
            )

        evaluator.update(outputs, batch)

        metrics_now = evaluator.compute()

        pbar.set_postfix(
            {
                "road_iou": metrics_now.get("road_iou", 0.0),
                "lane_f1": metrics_now.get("lane_f1", 0.0),
                "edge_f1": metrics_now.get("edge_f1", 0.0),
            }
        )

    metrics = evaluator.compute()

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        with output_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics