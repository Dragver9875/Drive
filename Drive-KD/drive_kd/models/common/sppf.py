from __future__ import annotations

import torch
import torch.nn as nn

from drive_kd.models.common.conv import ConvBNAct


class SPPF(nn.Module):

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 5,
        hidden_channels: int | None = None,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
    ) -> None:
        super().__init__()

        if k % 2 == 0:
            raise ValueError("SPPF kernel size must be odd.")

        hidden = hidden_channels if hidden_channels is not None else max(1, c1 // 2)

        self.cv1 = ConvBNAct(c1, hidden, k=1, s=1, norm=norm, act=act)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBNAct(hidden * 4, c2, k=1, s=1, norm=norm, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)

        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))