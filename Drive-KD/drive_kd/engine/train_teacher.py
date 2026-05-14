from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from tqdm import tqdm

from drive_kd.engine.amp_utils import get_autocast_context
from drive_kd.utils.device import move_batch_to_device


def train_teacher_one_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    criterion,
    device: torch.device | str,
    epoch: int,
    grad_accum_steps: int = 1,
    amp_enabled: bool = True,
    max_grad_norm: float | None = 5.0,
) -> dict[str, float]:
    
    device = torch.device(device)
    model.train()

    grad_accum_steps = max(1, int(grad_accum_steps))

    running = Counter()
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"teacher train epoch {epoch}")

    for step, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)

        with get_autocast_context(enabled=amp_enabled, device=device):
            outputs = model(batch["image"])
            loss, logs = criterion(outputs, batch)
            loss_for_backward = loss / grad_accum_steps

        scaler.scale(loss_for_backward).backward()

        should_step = step % grad_accum_steps == 0 or step == len(loader)

        if should_step:
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for key, value in logs.items():
            if isinstance(value, int | float):
                running[key] += float(value)

        n_batches += 1

        pbar.set_postfix(
            {
                "loss": running.get("loss_total", float(loss.detach().cpu())) / max(1, n_batches),
                "road": running.get("loss_road", 0.0) / max(1, n_batches),
                "lane": running.get("loss_lane", 0.0) / max(1, n_batches),
                "edge": running.get("loss_edge", 0.0) / max(1, n_batches),
            }
        )

    return {f"train/{k}": float(v) / max(1, n_batches) for k, v in running.items()}