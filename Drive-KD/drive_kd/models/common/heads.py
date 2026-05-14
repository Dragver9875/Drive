from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drive_kd.models.common.conv import ConvBNAct


class SegmentationHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_convs: int = 2,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_convs < 1:
            raise ValueError("SegmentationHead requires at least one conv block.")

        layers: list[nn.Module] = []

        c_in = in_channels

        for _ in range(num_convs):
            layers.append(
                ConvBNAct(
                    c_in,
                    hidden_channels,
                    k=3,
                    s=1,
                    norm=norm,
                    act=act,
                )
            )
            c_in = hidden_channels

            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))

        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        logits = self.net(x)

        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


class MultiTaskSegmentationHeads(nn.Module):

    def __init__(
        self,
        in_channels: int,
        road_hidden_channels: int = 96,
        lane_hidden_channels: int = 96,
        edge_hidden_channels: int = 48,
        num_convs: int = 2,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.road_head = SegmentationHead(
            in_channels=in_channels,
            hidden_channels=road_hidden_channels,
            out_channels=2,
            num_convs=num_convs,
            norm=norm,
            act=act,
            dropout=dropout,
        )

        self.lane_head = SegmentationHead(
            in_channels=in_channels,
            hidden_channels=lane_hidden_channels,
            out_channels=2,
            num_convs=num_convs,
            norm=norm,
            act=act,
            dropout=dropout,
        )

        self.edge_head = SegmentationHead(
            in_channels=in_channels,
            hidden_channels=edge_hidden_channels,
            out_channels=1,
            num_convs=num_convs,
            norm=norm,
            act=act,
            dropout=dropout,
        )

    def forward(
        self,
        fused: torch.Tensor,
        output_size: tuple[int, int],
        return_probabilities: bool = False,
    ) -> dict[str, torch.Tensor]:
        road_logits = self.road_head(fused, output_size=output_size)
        lane_logits = self.lane_head(fused, output_size=output_size)
        edge_logits = self.edge_head(fused, output_size=output_size)

        out = {
            "road_logits": road_logits,
            "lane_logits": lane_logits,
            "edge_logits": edge_logits,
        }

        if return_probabilities:
            out.update(
                {
                    "road_prob": torch.softmax(road_logits, dim=1)[:, 1],
                    "lane_prob": torch.softmax(lane_logits, dim=1)[:, 1],
                    "edge_prob": torch.sigmoid(edge_logits).squeeze(1),
                }
            )

        return out