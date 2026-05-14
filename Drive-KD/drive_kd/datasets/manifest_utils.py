from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML config is empty: {path}")

    return data


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError(f"Manifest has zero rows: {path}")

    return df


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = ensure_parent(path)

    if not rows:
        raise ValueError(f"Cannot write empty manifest: {path}")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_images(root: str | Path) -> int:
    root = Path(root)

    if not root.exists():
        return 0

    return sum(
        1
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def build_image_index(image_root: str | Path) -> dict[str, Path]:
    image_root = Path(image_root)

    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    index: dict[str, Path] = {}
    duplicates: dict[str, int] = {}

    for p in image_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        if p.name in index:
            duplicates[p.name] = duplicates.get(p.name, 0) + 1
            continue

        index[p.name] = p

    if not index:
        raise RuntimeError(f"No images found under: {image_root}")

    if duplicates:
        print(f"[warn] Duplicate image basenames ignored: {len(duplicates)}")

    return index


def required_columns(df: pd.DataFrame, columns: list[str], manifest_name: str = "manifest") -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(f"{manifest_name} is missing required columns: {missing}")


def build_absolute_manifest_from_portable(
    portable_manifest_path: str | Path,
    output_manifest_path: str | Path,
    image_root: str | Path,
    processed_masks_root: str | Path,
    split: str,
) -> pd.DataFrame:
    
    portable_manifest_path = Path(portable_manifest_path)
    output_manifest_path = Path(output_manifest_path)
    image_root = Path(image_root)
    processed_masks_root = Path(processed_masks_root)

    df = read_manifest(portable_manifest_path)

    required_columns(
        df,
        [
            "image_id",
            "image_name",
            "road_mask_relpath",
            "lane_mask_relpath",
            "edge_mask_relpath",
            "split",
            "height",
            "width",
            "has_road",
            "has_lane",
            "road_pixels",
            "lane_pixels",
            "edge_pixels",
        ],
        manifest_name=str(portable_manifest_path),
    )

    image_index = build_image_index(image_root)

    rows: list[dict[str, Any]] = []
    missing_images = 0
    missing_masks = 0

    for _, row in df.iterrows():
        image_name = str(row["image_name"])
        image_path = image_index.get(image_name)

        road_mask_path = processed_masks_root / str(row["road_mask_relpath"])
        lane_mask_path = processed_masks_root / str(row["lane_mask_relpath"])
        edge_mask_path = processed_masks_root / str(row["edge_mask_relpath"])

        if image_path is None:
            missing_images += 1
            continue

        if not road_mask_path.exists() or not lane_mask_path.exists() or not edge_mask_path.exists():
            missing_masks += 1
            continue

        rows.append(
            {
                "image_id": str(row["image_id"]),
                "image_name": image_name,
                "image_path": str(image_path),
                "road_mask_path": str(road_mask_path),
                "lane_mask_path": str(lane_mask_path),
                "edge_mask_path": str(edge_mask_path),
                "split": split,
                "height": int(row["height"]),
                "width": int(row["width"]),
                "has_road": int(row["has_road"]),
                "has_lane": int(row["has_lane"]),
                "road_pixels": int(row["road_pixels"]),
                "lane_pixels": int(row["lane_pixels"]),
                "edge_pixels": int(row["edge_pixels"]),
            }
        )

    if not rows:
        raise RuntimeError(
            "Absolute manifest construction produced zero rows. "
            f"missing_images={missing_images}, missing_masks={missing_masks}"
        )

    out = pd.DataFrame(rows)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_manifest_path, index=False)

    print(f"[manifest] split={split}")
    print(f"  portable:       {portable_manifest_path}")
    print(f"  output:         {output_manifest_path}")
    print(f"  rows:           {len(out)}")
    print(f"  missing_images: {missing_images}")
    print(f"  missing_masks:  {missing_masks}")

    return out


def build_kd_manifest(
    absolute_manifest_path: str | Path,
    output_manifest_path: str | Path,
    teacher_cache_root: str | Path,
    split: str,
    require_attention: bool = True,
    require_boundary: bool = True,
) -> pd.DataFrame:

    absolute_manifest_path = Path(absolute_manifest_path)
    output_manifest_path = Path(output_manifest_path)
    teacher_cache_root = Path(teacher_cache_root)

    df = read_manifest(absolute_manifest_path)

    required_columns(
        df,
        [
            "image_id",
            "image_name",
            "image_path",
            "road_mask_path",
            "lane_mask_path",
            "edge_mask_path",
            "split",
            "height",
            "width",
        ],
        manifest_name=str(absolute_manifest_path),
    )

    rows: list[dict[str, Any]] = []
    missing_prob = 0
    missing_boundary = 0
    missing_attention = 0

    for _, row in df.iterrows():
        image_id = str(row["image_id"])

        teacher_prob_path = teacher_cache_root / "probabilities" / split / f"{image_id}.npz"
        teacher_boundary_path = teacher_cache_root / "boundaries" / split / f"{image_id}.png"
        teacher_attention_path = teacher_cache_root / "attention" / split / f"{image_id}.npz"

        if not teacher_prob_path.exists():
            missing_prob += 1
            continue

        if require_boundary and not teacher_boundary_path.exists():
            missing_boundary += 1
            continue

        if require_attention and not teacher_attention_path.exists():
            missing_attention += 1
            continue

        rows.append(
            {
                "image_id": image_id,
                "image_name": str(row["image_name"]),
                "image_path": str(row["image_path"]),
                "road_mask_path": str(row["road_mask_path"]),
                "lane_mask_path": str(row["lane_mask_path"]),
                "edge_mask_path": str(row["edge_mask_path"]),
                "teacher_prob_path": str(teacher_prob_path),
                "teacher_boundary_path": str(teacher_boundary_path),
                "teacher_attention_path": str(teacher_attention_path),
                "split": split,
                "height": int(row["height"]),
                "width": int(row["width"]),
            }
        )

    if not rows:
        raise RuntimeError(
            "KD manifest construction produced zero rows. "
            f"missing_prob={missing_prob}, "
            f"missing_boundary={missing_boundary}, "
            f"missing_attention={missing_attention}"
        )

    out = pd.DataFrame(rows)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_manifest_path, index=False)

    print(f"[kd-manifest] split={split}")
    print(f"  absolute:          {absolute_manifest_path}")
    print(f"  output:            {output_manifest_path}")
    print(f"  rows:              {len(out)}")
    print(f"  missing_prob:      {missing_prob}")
    print(f"  missing_boundary:  {missing_boundary}")
    print(f"  missing_attention: {missing_attention}")

    return out