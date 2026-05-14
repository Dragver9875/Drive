from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    best_metric: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
    }

    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    if config is not None:
        ckpt["config"] = config

    if metrics is not None:
        ckpt["metrics"] = metrics

    if history is not None:
        ckpt["history"] = history

    if best_metric is not None:
        ckpt["best_metric"] = float(best_metric)

    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, path)
    return path


def load_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)

    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {path}")

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt


def find_checkpoint(
    checkpoint_dir: str | Path,
    names: tuple[str, ...] = ("last.pt", "best.pt"),
) -> Path | None:

    checkpoint_dir = Path(checkpoint_dir)

    for name in names:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate

    return None


def copy_checkpoint(
    src: str | Path,
    dst: str | Path,
) -> Path:
    
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Checkpoint source not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    return dst


def checkpoint_metric_is_better(
    current: float,
    best: float | None,
    mode: str = "max",
) -> bool:

    mode = mode.lower()

    if best is None:
        return True

    if mode == "max":
        return current > best

    if mode == "min":
        return current < best

    raise ValueError(f"Unsupported checkpoint monitor mode: {mode}")