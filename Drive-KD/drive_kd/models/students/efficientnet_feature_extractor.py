from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


@dataclass
class FeatureInfo:

    name: str
    channels: int
    reduction: int


class TimmFeatureExtractor(nn.Module):

    PYRAMID_NAMES = ("p1", "p2", "p3", "p4", "p5")

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        out_indices: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4),
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = out_indices

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        raw_info = self.backbone.feature_info

        self.feature_info: list[FeatureInfo] = []

        for i, name in enumerate(self.PYRAMID_NAMES):
            self.feature_info.append(
                FeatureInfo(
                    name=name,
                    channels=int(raw_info.channels()[i]),
                    reduction=int(raw_info.reduction()[i]),
                )
            )

    @property
    def out_channels(self) -> dict[str, int]:
        return {item.name: item.channels for item in self.feature_info}

    @property
    def reductions(self) -> dict[str, int]:
        return {item.name: item.reduction for item in self.feature_info}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)

        if len(feats) != 5:
            raise RuntimeError(
                f"{self.model_name} expected to return 5 feature maps, got {len(feats)}"
            )

        out = {name: feat for name, feat in zip(self.PYRAMID_NAMES, feats)}

        return out


class PyramidFeatureResizer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = ["p1", "p2", "p3", "p4", "p5"]

        for key in required:
            if key not in feats:
                raise KeyError(f"Missing pyramid feature: {key}")

        return feats