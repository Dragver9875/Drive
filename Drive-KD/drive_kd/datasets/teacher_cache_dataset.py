from __future__ import annotations

from pathlib import Path
from typing import Any

from drive_kd.datasets.drive_supervised_dataset import DriveSupervisedDataset


class TeacherCacheDataset(DriveSupervisedDataset):

    def __init__(
        self,
        manifest_path: str | Path,
        image_height: int,
        image_width: int,
        mean: list[float],
        std: list[float],
        split: str,
        transform_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            manifest_path=manifest_path,
            image_height=image_height,
            image_width=image_width,
            mean=mean,
            std=std,
            train=False,
            transform_config=transform_config,
        )

        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split for teacher cache: {split}")

        self.split = split

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)
        sample["split"] = self.split
        return sample