from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from drive_kd.models.common import (
    BiFPNLite,
    ConvBNAct,
    MultiTaskSegmentationHeads,
    SPPF,
    compute_spatial_attention,
)
from drive_kd.models.students.effb0_outputs import StudentOutput
from drive_kd.models.students.efficientnet_b0_encoder import EfficientNetB0Encoder


class FeatureAdapters(nn.Module):

    REQUIRED_LEVELS = ("p1", "p2", "p3", "p4", "p5")

    def __init__(
        self,
        in_channels: dict[str, int],
        out_channels: int = 96,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
    ) -> None:
        super().__init__()

        missing = [k for k in self.REQUIRED_LEVELS if k not in in_channels]

        if missing:
            raise KeyError(f"Missing input channel definitions for levels: {missing}")

        self.adapters = nn.ModuleDict(
            {
                level: ConvBNAct(
                    in_channels[level],
                    out_channels,
                    k=1,
                    s=1,
                    norm=norm,
                    act=act,
                )
                for level in self.REQUIRED_LEVELS
            }
        )

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        for level, adapter in self.adapters.items():
            if level not in feats:
                raise KeyError(f"FeatureAdapters missing input feature: {level}")
            out[level] = adapter(feats[level])

        return out


class DriveEffB0BiFPN(nn.Module):

    def __init__(
        self,
        *,
        image_height: int = 384,
        image_width: int = 640,
        encoder_pretrained: bool = True,
        encoder_freeze: bool = False,
        encoder_model_name: str = "efficientnet_b0",
        neck_channels: int = 96,
        sppf_enabled: bool = True,
        sppf_kernel_size: int = 5,
        bifpn_repeats: int = 1,
        use_depthwise_bifpn: bool = False,
        return_features_default: bool = True,
        return_attention_default: bool = True,
        return_probabilities_default: bool = True,
    ) -> None:
        super().__init__()

        self.image_height = int(image_height)
        self.image_width = int(image_width)

        self.neck_channels = int(neck_channels)
        self.sppf_enabled = bool(sppf_enabled)

        self.return_features_default = bool(return_features_default)
        self.return_attention_default = bool(return_attention_default)
        self.return_probabilities_default = bool(return_probabilities_default)

        self.encoder = EfficientNetB0Encoder(
            pretrained=encoder_pretrained,
            freeze=encoder_freeze,
            model_name=encoder_model_name,
        )

        self.adapters = FeatureAdapters(
            in_channels=self.encoder.out_channels,
            out_channels=self.neck_channels,
        )

        if self.sppf_enabled:
            self.sppf = SPPF(
                c1=self.neck_channels,
                c2=self.neck_channels,
                k=sppf_kernel_size,
            )
        else:
            self.sppf = nn.Identity()

        if bifpn_repeats < 1:
            raise ValueError("bifpn_repeats must be >= 1")

        self.neck = nn.Sequential(
            *[
                BiFPNLite(
                    channels=self.neck_channels,
                    use_depthwise_refine=use_depthwise_bifpn,
                )
                for _ in range(bifpn_repeats)
            ]
        )

        self.heads = MultiTaskSegmentationHeads(
            in_channels=self.neck_channels,
            road_hidden_channels=self.neck_channels,
            lane_hidden_channels=self.neck_channels,
            edge_hidden_channels=max(32, self.neck_channels // 2),
            num_convs=2,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DriveEffB0BiFPN":

        student_cfg = cfg.get("student", cfg)

        input_cfg = student_cfg.get("input", {})
        encoder_cfg = student_cfg.get("encoder", {})
        neck_cfg = student_cfg.get("neck", {})
        sppf_cfg = student_cfg.get("sppf", {})
        output_cfg = student_cfg.get("output", {})

        return cls(
            image_height=int(input_cfg.get("height", 384)),
            image_width=int(input_cfg.get("width", 640)),
            encoder_pretrained=bool(encoder_cfg.get("pretrained", True)),
            encoder_freeze=bool(encoder_cfg.get("freeze", False)),
            encoder_model_name=str(encoder_cfg.get("timm_model_name", "efficientnet_b0")),
            neck_channels=int(neck_cfg.get("channels", 96)),
            sppf_enabled=bool(sppf_cfg.get("enabled", True)),
            sppf_kernel_size=int(sppf_cfg.get("kernel_size", 5)),
            bifpn_repeats=int(neck_cfg.get("num_repeats", 1)),
            use_depthwise_bifpn=bool(neck_cfg.get("use_depthwise_refine", False)),
            return_features_default=bool(output_cfg.get("return_features", True)),
            return_attention_default=bool(output_cfg.get("return_attention", True)),
            return_probabilities_default=bool(output_cfg.get("return_probabilities", True)),
        )

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_feats = self.encoder(x)
        feats = self.adapters(encoder_feats)

        feats["p5"] = self.sppf(feats["p5"])

        return feats

    def forward_neck(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        neck_out: dict[str, torch.Tensor] | None = None

        current = feats

        for neck_block in self.neck:
            neck_out = neck_block(current)
            current = {
                "p1": neck_out["p1"],
                "p2": neck_out["p2"],
                "p3": neck_out["p3"],
                "p4": neck_out["p4"],
                "p5": neck_out["p5"],
            }

        if neck_out is None:
            raise RuntimeError("Neck produced no output.")

        return neck_out

    def build_attention(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        attention: dict[str, torch.Tensor] = {}

        for level in ["p2", "p4"]:
            if level in features:
                attention[level] = compute_spatial_attention(
                    features[level],
                    method="channel_mean_abs",
                    keepdim=True,
                )

        return attention

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool | None = None,
        return_attention: bool | None = None,
        return_probabilities: bool | None = None,
        as_dataclass: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]] | StudentOutput:
        output_size = x.shape[-2:]

        if return_features is None:
            return_features = self.return_features_default

        if return_attention is None:
            return_attention = self.return_attention_default

        if return_probabilities is None:
            return_probabilities = self.return_probabilities_default

        feats = self.forward_features(x)
        neck_out = self.forward_neck(feats)

        fused = neck_out["fused"]

        head_out = self.heads(
            fused,
            output_size=output_size,
            return_probabilities=return_probabilities,
        )

        features_for_output: dict[str, torch.Tensor] = {}
        attention_for_output: dict[str, torch.Tensor] = {}

        if return_features:
            features_for_output = {
                "p1": neck_out["p1"],
                "p2": neck_out["p2"],
                "p3": neck_out["p3"],
                "p4": neck_out["p4"],
                "p5": neck_out["p5"],
                "fused": fused,
            }

        if return_attention:
            attention_for_output = self.build_attention(neck_out)

        structured = StudentOutput(
            road_logits=head_out["road_logits"],
            lane_logits=head_out["lane_logits"],
            edge_logits=head_out["edge_logits"],
            road_prob=head_out.get("road_prob"),
            lane_prob=head_out.get("lane_prob"),
            edge_prob=head_out.get("edge_prob"),
            features=features_for_output,
            attention=attention_for_output,
        )

        if as_dataclass:
            return structured

        return structured.to_dict()

    def freeze_encoder(self) -> None:
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        self.encoder.unfreeze()

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())