from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drive_kd.models.common.conv import ConvBNAct, DepthwiseSeparableConv
from drive_kd.models.common.weighted_fusion import WeightedFusion


class BiFPNLite(nn.Module):

    REQUIRED_LEVELS = ("p1", "p2", "p3", "p4", "p5")

    def __init__(
        self,
        channels: int = 96,
        eps: float = 1e-4,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
        use_depthwise_refine: bool = False,
    ) -> None:
        super().__init__()

        self.channels = int(channels)

        refine_cls = DepthwiseSeparableConv if use_depthwise_refine else ConvBNAct

        self.fuse_td4 = WeightedFusion(2, eps=eps)
        self.fuse_td3 = WeightedFusion(2, eps=eps)
        self.fuse_td2 = WeightedFusion(2, eps=eps)
        self.fuse_td1 = WeightedFusion(2, eps=eps)

        self.refine_td4 = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)
        self.refine_td3 = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)
        self.refine_td2 = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)
        self.refine_td1 = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)

        self.down_1_to_2 = ConvBNAct(channels, channels, k=3, s=2, norm=norm, act=act)
        self.down_2_to_3 = ConvBNAct(channels, channels, k=3, s=2, norm=norm, act=act)
        self.down_3_to_4 = ConvBNAct(channels, channels, k=3, s=2, norm=norm, act=act)
        self.down_4_to_5 = ConvBNAct(channels, channels, k=3, s=2, norm=norm, act=act)

        self.fuse_bu2 = WeightedFusion(3, eps=eps)
        self.fuse_bu3 = WeightedFusion(3, eps=eps)
        self.fuse_bu4 = WeightedFusion(3, eps=eps)
        self.fuse_bu5 = WeightedFusion(2, eps=eps)

        self.refine_bu2 = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)
        self.refine_bu3 = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)
        self.refine_bu4 = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)
        self.refine_bu5 = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)

        self.final_fusion = WeightedFusion(5, eps=eps)
        self.final_refine = refine_cls(channels, channels, k=3, s=1, norm=norm, act=act)

    @staticmethod
    def up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    @staticmethod
    def resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def _check_inputs(self, feats: dict[str, torch.Tensor]) -> None:
        missing = [level for level in self.REQUIRED_LEVELS if level not in feats]

        if missing:
            raise KeyError(f"BiFPNLite missing required feature levels: {missing}")

        for level in self.REQUIRED_LEVELS:
            x = feats[level]

            if x.ndim != 4:
                raise ValueError(f"Feature {level} must be 4D [B,C,H,W], got {x.shape}")

            if x.shape[1] != self.channels:
                raise ValueError(
                    f"Feature {level} has {x.shape[1]} channels, "
                    f"but BiFPNLite expects {self.channels}. "
                    "Use 1x1 adapters before BiFPNLite."
                )

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._check_inputs(feats)

        p1 = feats["p1"]
        p2 = feats["p2"]
        p3 = feats["p3"]
        p4 = feats["p4"]
        p5 = feats["p5"]

        td5 = p5

        td4 = self.fuse_td4([p4, self.up_to(td5, p4)])
        td4 = self.refine_td4(td4)

        td3 = self.fuse_td3([p3, self.up_to(td4, p3)])
        td3 = self.refine_td3(td3)

        td2 = self.fuse_td2([p2, self.up_to(td3, p2)])
        td2 = self.refine_td2(td2)

        td1 = self.fuse_td1([p1, self.up_to(td2, p1)])
        td1 = self.refine_td1(td1)

        bu1 = td1

        d12 = self.resize_to(self.down_1_to_2(bu1), p2)
        bu2 = self.fuse_bu2([p2, td2, d12])
        bu2 = self.refine_bu2(bu2)

        d23 = self.resize_to(self.down_2_to_3(bu2), p3)
        bu3 = self.fuse_bu3([p3, td3, d23])
        bu3 = self.refine_bu3(bu3)

        d34 = self.resize_to(self.down_3_to_4(bu3), p4)
        bu4 = self.fuse_bu4([p4, td4, d34])
        bu4 = self.refine_bu4(bu4)

        d45 = self.resize_to(self.down_4_to_5(bu4), p5)
        bu5 = self.fuse_bu5([p5, d45])
        bu5 = self.refine_bu5(bu5)

        fused = self.final_fusion(
            [
                bu1,
                self.up_to(bu2, bu1),
                self.up_to(bu3, bu1),
                self.up_to(bu4, bu1),
                self.up_to(bu5, bu1),
            ]
        )
        fused = self.final_refine(fused)

        return {
            "fused": fused,
            "p1": bu1,
            "p2": bu2,
            "p3": bu3,
            "p4": bu4,
            "p5": bu5,
        }