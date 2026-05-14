from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> dict[str, Any]:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
) -> Path:

    path = ensure_parent(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    return path


def write_text(
    text: str,
    path: str | Path,
) -> Path:

    path = ensure_parent(path)

    with path.open("w", encoding="utf-8") as f:
        f.write(text)

    return path


def copy_file(
    src: str | Path,
    dst: str | Path,
    overwrite: bool = True,
) -> Path:

    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    if dst.exists() and not overwrite:
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    return dst


def safe_unlink(path: str | Path) -> None:

    path = Path(path)

    if path.exists() and path.is_file():
        path.unlink()


def list_files(
    root: str | Path,
    pattern: str = "*",
    recursive: bool = True,
) -> list[Path]:

    root = Path(root)

    if not root.exists():
        return []

    iterator = root.rglob(pattern) if recursive else root.glob(pattern)

    return sorted([p for p in iterator if p.is_file()])


def directory_size_bytes(root: str | Path) -> int:

    root = Path(root)

    if not root.exists():
        return 0

    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def format_bytes(num_bytes: int) -> str:

    value = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} PB"