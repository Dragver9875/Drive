from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def make_grad_scaler(
    *,
    enabled: bool,
    device: torch.device | str,
) -> torch.cuda.amp.GradScaler:

    device = torch.device(device)
    amp_enabled = bool(enabled and device.type == "cuda")

    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def get_autocast_context(
    *,
    enabled: bool,
    device: torch.device | str,
    dtype: torch.dtype = torch.float16,
) -> Any:

    device = torch.device(device)

    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True, dtype=dtype)

    return nullcontext()


def autocast_enabled(
    *,
    enabled: bool,
    device: torch.device | str,
) -> bool:
    device = torch.device(device)
    return bool(enabled and device.type == "cuda")