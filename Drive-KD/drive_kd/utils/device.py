from __future__ import annotations

from typing import Any

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_amp_enabled(device: torch.device | str, requested: bool = True) -> bool:

    device = torch.device(device)

    return bool(requested and device.type == "cuda")


def move_to_device(
    value: Any,
    device: torch.device | str,
    non_blocking: bool = True,
) -> Any:

    device = torch.device(device)

    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, dict):
        return {
            k: move_to_device(v, device, non_blocking=non_blocking)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [move_to_device(v, device, non_blocking=non_blocking) for v in value]

    if isinstance(value, tuple):
        return tuple(move_to_device(v, device, non_blocking=non_blocking) for v in value)

    return value


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device | str,
    non_blocking: bool = True,
) -> dict[str, Any]:

    return {
        key: move_to_device(value, device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def cuda_memory_summary() -> dict[str, float]:

    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024.0**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024.0**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024.0**2),
    }


def empty_cuda_cache() -> None:

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def describe_device(device: torch.device | str | None = None) -> dict[str, Any]:

    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    info: dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()

        props = torch.cuda.get_device_properties(index)

        info.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": torch.cuda.get_device_name(index),
                "total_memory_mb": props.total_memory / (1024.0**2),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
        )

    return info