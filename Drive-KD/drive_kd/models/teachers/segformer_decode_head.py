from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drive_kd.models.common.conv import ConvBNAct


class SegFormerMLPProjection(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
    ) -> None:
        super().__init__()

        self.proj = ConvBNAct(
            in_channels,
            hidden_dim,
            k=1,
            s=1,
            norm=norm,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SegFormerDecodeHead(nn.Module):

    REQUIRED_LEVELS = ("s1", "s2", "s3", "s4")

    def __init__(
        self,
        in_channels: dict[str, int],
        hidden_dim: int = 256,
        dropout: float = 0.1,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
    ) -> None:
        super().__init__()

        missing = [level for level in self.REQUIRED_LEVELS if level not in in_channels]
        if missing:
            raise KeyError(f"Missing SegFormer decoder input channels for: {missing}")

        self.hidden_dim = int(hidden_dim)

        self.projections = nn.ModuleDict(
            {
                level: SegFormerMLPProjection(
                    in_channels=in_channels[level],
                    hidden_dim=self.hidden_dim,
                    norm=norm,
                    act=act,
                )
                for level in self.REQUIRED_LEVELS
            }
        )

        self.fuse = nn.Sequential(
            ConvBNAct(
                self.hidden_dim * len(self.REQUIRED_LEVELS),
                self.hidden_dim,
                k=1,
                s=1,
                norm=norm,
                act=act,
            ),
            ConvBNAct(
                self.hidden_dim,
                self.hidden_dim,
                k=3,
                s=1,
                norm=norm,
                act=act,
            ),
            nn.Dropout2d(p=float(dropout)) if dropout > 0 else nn.Identity(),
        )

    @staticmethod
    def _upsample_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x

        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [level for level in self.REQUIRED_LEVELS if level not in features]
        if missing:
            raise KeyError(f"SegFormerDecodeHead missing features: {missing}")

        reference = features["s1"]

        projected = []

        for level in self.REQUIRED_LEVELS:
            x = self.projections[level](features[level])
            x = self._upsample_to(x, reference)
            projected.append(x)

        fused = torch.cat(projected, dim=1)
        decoded = self.fuse(fused)

        return decoded