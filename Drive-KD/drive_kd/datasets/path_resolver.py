from __future__ import annotations

from pathlib import Path
from typing import Any

from drive_kd.datasets.manifest_utils import load_yaml


def as_path(x: str | Path) -> Path:
    return Path(str(x)).expanduser()


class DatasetPathResolver:

    def __init__(self, dataset_config_path: str | Path):
        self.dataset_config_path = Path(dataset_config_path)
        self.config = load_yaml(self.dataset_config_path)

        if "dataset" not in self.config:
            raise KeyError(f"Missing top-level 'dataset' key in {self.dataset_config_path}")

        self.dataset = self.config["dataset"]

    @property
    def image_height(self) -> int:
        return int(self.dataset["image_height"])

    @property
    def image_width(self) -> int:
        return int(self.dataset["image_width"])

    @property
    def processed_masks_root(self) -> Path:
        return as_path(self.dataset["processed_masks_root"])

    @property
    def train_image_root(self) -> Path:
        return as_path(self.dataset["image_roots"]["train"])

    @property
    def val_image_root(self) -> Path:
        return as_path(self.dataset["image_roots"]["val"])

    @property
    def train_portable_manifest(self) -> Path:
        return as_path(self.dataset["portable_manifests"]["train"])

    @property
    def val_portable_manifest(self) -> Path:
        return as_path(self.dataset["portable_manifests"]["val"])

    @property
    def train_absolute_manifest(self) -> Path:
        return as_path(self.dataset["absolute_manifests"]["train"])

    @property
    def val_absolute_manifest(self) -> Path:
        return as_path(self.dataset["absolute_manifests"]["val"])

    @property
    def train_kd_manifest(self) -> Path:
        return as_path(self.dataset["kd_manifests"]["train"])

    @property
    def val_kd_manifest(self) -> Path:
        return as_path(self.dataset["kd_manifests"]["val"])

    @property
    def mean(self) -> list[float]:
        return list(self.dataset["normalization"]["mean"])

    @property
    def std(self) -> list[float]:
        return list(self.dataset["normalization"]["std"])

    def absolute_manifest_for_split(self, split: str) -> Path:
        if split == "train":
            return self.train_absolute_manifest
        if split == "val":
            return self.val_absolute_manifest
        raise ValueError(f"Unsupported split: {split}")

    def kd_manifest_for_split(self, split: str) -> Path:
        if split == "train":
            return self.train_kd_manifest
        if split == "val":
            return self.val_kd_manifest
        raise ValueError(f"Unsupported split: {split}")

    def image_root_for_split(self, split: str) -> Path:
        if split == "train":
            return self.train_image_root
        if split == "val":
            return self.val_image_root
        raise ValueError(f"Unsupported split: {split}")

    def portable_manifest_for_split(self, split: str) -> Path:
        if split == "train":
            return self.train_portable_manifest
        if split == "val":
            return self.val_portable_manifest
        raise ValueError(f"Unsupported split: {split}")

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_config_path": str(self.dataset_config_path),
            "image_height": self.image_height,
            "image_width": self.image_width,
            "processed_masks_root": str(self.processed_masks_root),
            "train_image_root": str(self.train_image_root),
            "val_image_root": str(self.val_image_root),
            "train_portable_manifest": str(self.train_portable_manifest),
            "val_portable_manifest": str(self.val_portable_manifest),
            "train_absolute_manifest": str(self.train_absolute_manifest),
            "val_absolute_manifest": str(self.val_absolute_manifest),
            "train_kd_manifest": str(self.train_kd_manifest),
            "val_kd_manifest": str(self.val_kd_manifest),
        }