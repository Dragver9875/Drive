from __future__ import annotations

import torch
import torch.nn as nn

from drive_kd.models.students.efficientnet_feature_extractor import TimmFeatureExtractor


class EfficientNetB0Encoder(nn.Module):

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = False,
        model_name: str = "efficientnet_b0",
    ) -> None:
        super().__init__()

        self.extractor = TimmFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            out_indices=(0, 1, 2, 3, 4),
        )

        if freeze:
            self.freeze()

    @property
    def out_channels(self) -> dict[str, int]:
        return self.extractor.out_channels

    @property
    def reductions(self) -> dict[str, int]:
        return self.extractor.reductions

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.extractor(x)