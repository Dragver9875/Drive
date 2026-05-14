from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from tqdm import tqdm

from drive_kd.engine.amp_utils import get_autocast_context
from drive_kd.metrics import DriveEvaluator
from drive_kd.utils.device import move_batch_to_device


@torch.no_grad()
def validate_one_epoch(
    *,
    model: torch.nn.Module,
    loader,
    criterion,
    evaluator: DriveEvaluator,
    device: torch.device | str,
    epoch: int,
    amp_enabled: bool = True,
    desc: str | None = None,
) -> dict[str, float]:

    device = torch.device(device)
    model.eval()
    evaluator.reset()

    running = Counter()
    n_batches = 0

    pbar = tqdm(loader, desc=desc or f"val epoch {epoch}")

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        with get_autocast_context(enabled=amp_enabled, device=device):
            outputs = model(batch["image"])

            try:
                loss, logs = criterion(outputs, batch, epoch=epoch)
            except TypeError:
                loss, logs = criterion(outputs, batch)

        evaluator.update(outputs, batch)

        for key, value in logs.items():
            if isinstance(value, int | float):
                running[key] += float(value)

        n_batches += 1

        metrics_now = evaluator.compute()
        pbar.set_postfix(
            {
                "loss": running.get("loss_total", float(loss.detach().cpu())) / max(1, n_batches),
                "road_iou": metrics_now.get("road_iou", 0.0),
                "lane_f1": metrics_now.get("lane_f1", 0.0),
                "edge_f1": metrics_now.get("edge_f1", 0.0),
            }
        )

    out: dict[str, float] = {}

    for key, value in running.items():
        out[f"val/{key}"] = float(value) / max(1, n_batches)

    metric_values = evaluator.compute()

    for key, value in metric_values.items():
        out[f"val/{key}"] = float(value)

    return out