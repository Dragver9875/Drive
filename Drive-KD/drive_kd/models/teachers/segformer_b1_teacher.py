from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel

from drive_kd.models.common import (
    MultiTaskSegmentationHeads,
    compute_spatial_attention,
)
from drive_kd.models.teachers.base_teacher import BaseTeacher
from drive_kd.models.teachers.segformer_decode_head import SegFormerDecodeHead
from drive_kd.models.teachers.teacher_outputs import TeacherOutput


class SegFormerB1Teacher(BaseTeacher):

    DEFAULT_MODEL_NAME = "nvidia/mit-b1"

    DEFAULT_IN_CHANNELS = {
        "s1": 64,
        "s2": 128,
        "s3": 320,
        "s4": 512,
    }

    def __init__(
        self,
        *,
        image_height: int = 384,
        image_width: int = 640,
        pretrained: bool = True,
        model_name: str = DEFAULT_MODEL_NAME,
        decoder_hidden_dim: int = 256,
        decoder_dropout: float = 0.1,
        freeze_encoder: bool = False,
        return_features_default: bool = True,
        return_attention_default: bool = True,
        return_probabilities_default: bool = True,
    ) -> None:
        super().__init__(name="SegFormer-B1-DriveTeacher")

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.pretrained = bool(pretrained)
        self.model_name = str(model_name)

        self.return_features_default = bool(return_features_default)
        self.return_attention_default = bool(return_attention_default)
        self.return_probabilities_default = bool(return_probabilities_default)

        self.encoder = self._build_encoder(
            pretrained=self.pretrained,
            model_name=self.model_name,
        )

        self.decoder = SegFormerDecodeHead(
            in_channels=self.DEFAULT_IN_CHANNELS,
            hidden_dim=decoder_hidden_dim,
            dropout=decoder_dropout,
        )

        self.heads = MultiTaskSegmentationHeads(
            in_channels=decoder_hidden_dim,
            road_hidden_channels=decoder_hidden_dim,
            lane_hidden_channels=decoder_hidden_dim,
            edge_hidden_channels=max(64, decoder_hidden_dim // 2),
            num_convs=2,
            dropout=0.0,
        )

        if freeze_encoder:
            self.freeze_encoder()

    @staticmethod
    def _build_encoder(pretrained: bool, model_name: str) -> SegformerModel:
        if pretrained:
            try:
                return SegformerModel.from_pretrained(
                    model_name,
                    output_hidden_states=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load pretrained SegFormer encoder '{model_name}'. "
                    "Use pretrained=False for offline/local tests, or make sure "
                    "the checkpoint is available."
                ) from exc

        config = SegformerConfig(
            num_channels=3,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[64, 128, 320, 512],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.1,
            output_hidden_states=True,
        )

        return SegformerModel(config)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SegFormerB1Teacher":

        teacher_cfg = cfg.get("teacher", cfg)

        input_cfg = teacher_cfg.get("input", {})
        pretrained_cfg = teacher_cfg.get("pretrained", {})
        encoder_cfg = teacher_cfg.get("encoder", {})
        decoder_cfg = teacher_cfg.get("decoder", {})
        output_cfg = teacher_cfg.get("output", {})

        return cls(
            image_height=int(input_cfg.get("height", 384)),
            image_width=int(input_cfg.get("width", 640)),
            pretrained=bool(pretrained_cfg.get("enabled", True)),
            model_name=str(pretrained_cfg.get("model_name", cls.DEFAULT_MODEL_NAME)),
            decoder_hidden_dim=int(decoder_cfg.get("hidden_dim", 256)),
            decoder_dropout=float(decoder_cfg.get("dropout", 0.1)),
            freeze_encoder=bool(encoder_cfg.get("freeze", False)),
            return_features_default=bool(output_cfg.get("return_features", True)),
            return_attention_default=bool(output_cfg.get("return_attention", True)),
            return_probabilities_default=bool(output_cfg.get("return_probabilities", True)),
        )

    @staticmethod
    def _hidden_states_to_feature_dict(hidden_states: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:

        spatial_states = [h for h in hidden_states if h.ndim == 4]

        if len(spatial_states) < 4:
            raise RuntimeError(
                f"Expected at least 4 spatial hidden states from SegFormer, "
                f"got {len(spatial_states)}."
            )

        s1, s2, s3, s4 = spatial_states[-4:]

        return {
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
        }

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        if outputs.hidden_states is None:
            raise RuntimeError("SegFormer encoder did not return hidden_states.")

        return self._hidden_states_to_feature_dict(outputs.hidden_states)

    def decode(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decoder(features)

    def build_attention(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        attention: dict[str, torch.Tensor] = {}

        for level in ["s2", "s4"]:
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
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]] | TeacherOutput:
        output_size = x.shape[-2:]

        if return_features is None:
            return_features = self.return_features_default

        if return_attention is None:
            return_attention = self.return_attention_default

        if return_probabilities is None:
            return_probabilities = self.return_probabilities_default

        features = self.encode(x)
        decoded = self.decode(features)

        head_out = self.heads(
            decoded,
            output_size=output_size,
            return_probabilities=return_probabilities,
        )

        features_for_output: dict[str, torch.Tensor] = {}
        attention_for_output: dict[str, torch.Tensor] = {}

        if return_features:
            features_for_output = {
                **features,
                "decoded": decoded,
            }

        if return_attention:
            attention_for_output = self.build_attention(features)

        structured = TeacherOutput(
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
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True