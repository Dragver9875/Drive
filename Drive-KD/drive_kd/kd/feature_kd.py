from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from drive_kd.models.common.attention import (
    compute_spatial_attention,
    normalize_attention_map,
)


class FeatureSpatialKDLoss(nn.Module):

    def __init__(
        self,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.normalize = bool(normalize)

    def forward(self, student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
        s_att = compute_spatial_attention(student_feature, method="channel_mean_abs", keepdim=True)
        t_att = compute_spatial_attention(teacher_feature, method="channel_mean_abs", keepdim=True)

        if t_att.shape[-2:] != s_att.shape[-2:]:
            t_att = F.interpolate(
                t_att,
                size=s_att.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.normalize:
            s_att = normalize_attention_map(s_att, mode="l2")
            t_att = normalize_attention_map(t_att, mode="l2")

        return F.mse_loss(s_att, t_att)


class FeatureKDLoss(nn.Module):

    def __init__(
        self,
        layer_pairs: list[dict[str, Any]] | None = None,
        weight: float = 0.0,
        enabled: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.layer_pairs = layer_pairs or []
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.spatial_loss = FeatureSpatialKDLoss(normalize=normalize)

    def forward(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        teacher_outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = outputs["edge_logits"].device

        if not self.enabled or self.weight <= 0:
            zero = torch.zeros((), device=device)
            return zero, {"kd_feature_total": 0.0}

        if "features" not in outputs:
            raise KeyError("Student outputs missing 'features' for FeatureKDLoss.")

        if "features" not in teacher_outputs:
            raise KeyError("Teacher outputs missing 'features' for FeatureKDLoss.")

        student_features = outputs["features"]
        teacher_features = teacher_outputs["features"]

        if not isinstance(student_features, dict):
            raise TypeError("outputs['features'] must be a dict.")

        if not isinstance(teacher_features, dict):
            raise TypeError("teacher_outputs['features'] must be a dict.")

        total = torch.zeros((), device=device)
        logs: dict[str, float] = {}

        active = 0

        for pair in self.layer_pairs:
            teacher_key = str(pair["teacher"])
            student_key = str(pair["student"])
            pair_weight = float(pair.get("weight", 1.0))

            if teacher_key not in teacher_features:
                raise KeyError(f"Teacher features missing key: {teacher_key}")

            if student_key not in student_features:
                raise KeyError(f"Student features missing key: {student_key}")

            loss = self.spatial_loss(
                student_features[student_key].to(device),
                teacher_features[teacher_key].to(device),
            )

            total = total + pair_weight * loss
            active += 1

            logs[f"kd_feature_{teacher_key}_to_{student_key}"] = float(loss.detach().cpu())

        if active == 0:
            raise RuntimeError("FeatureKDLoss enabled but no active layer pairs were provided.")

        total = total / float(active)
        weighted_total = self.weight * total

        logs["kd_feature_raw"] = float(total.detach().cpu())
        logs["kd_feature_total"] = float(weighted_total.detach().cpu())

        return weighted_total, logs