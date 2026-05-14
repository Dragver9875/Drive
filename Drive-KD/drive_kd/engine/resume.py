from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from drive_kd.engine.checkpointing import find_checkpoint, load_checkpoint


def resume_if_available(
    *,
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: torch.device | str = "cpu",
    enabled: bool = True,
    strict: bool = True,
) -> tuple[int, float | None, list[dict[str, Any]], dict[str, Any] | None]:

    if not enabled:
        return 1, None, [], None

    ckpt_path = find_checkpoint(checkpoint_dir, names=("last.pt", "best.pt"))

    if ckpt_path is None:
        return 1, None, [], None

    ckpt = load_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=device,
        strict=strict,
    )

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = ckpt.get("best_metric")
    history = ckpt.get("history", [])

    if best_metric is not None:
        best_metric = float(best_metric)

    print(f"[resume] Loaded checkpoint: {ckpt_path}")
    print(f"[resume] start_epoch={start_epoch}, best_metric={best_metric}")

    return start_epoch, best_metric, history, ckpt