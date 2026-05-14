from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def tensor_to_numpy(
    x: torch.Tensor | np.ndarray,
    dtype: np.dtype | str | None = None,
) -> np.ndarray:

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

    if dtype is not None:
        arr = arr.astype(dtype)

    return arr


def save_npz(
    path: str | Path,
    compressed: bool = True,
    **arrays: torch.Tensor | np.ndarray,
) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np_arrays = {key: tensor_to_numpy(value) for key, value in arrays.items()}

    if compressed:
        np.savez_compressed(path, **np_arrays)
    else:
        np.savez(path, **np_arrays)

    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    data = np.load(path)

    return {key: data[key] for key in data.files}


def save_tensor(
    tensor: torch.Tensor,
    path: str | Path,
) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(tensor.detach().cpu(), path)

    return path


def load_tensor(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> torch.Tensor:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    obj = torch.load(path, map_location=map_location)

    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(obj)}")

    return obj


def to_float16_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:

    return tensor_to_numpy(x, dtype=np.float16)


def to_uint8_mask(x: torch.Tensor | np.ndarray, threshold: float = 0.5) -> np.ndarray:

    arr = tensor_to_numpy(x)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    mask = (arr > threshold).astype(np.uint8) * 255

    return mask


def detach_dict_to_cpu(
    data: dict[str, Any],
) -> dict[str, Any]:

    out: dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu()
        elif isinstance(value, dict):
            out[key] = detach_dict_to_cpu(value)
        else:
            out[key] = value

    return out