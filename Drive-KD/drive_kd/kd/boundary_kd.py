from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from drive_kd.losses.boundary_loss import BoundaryBCEDiceLoss


class BoundaryKDLoss(nn.Module):

    def __init__(
        self,
        weight: float = 0.3,
        pos_weight: float = 4.0,
        smooth: float = 1.0,
        enabled: bool = True,
    ) -> None:
        super().__init__()

        self.weight = float(weight)
        self.enabled = bool(enabled)

        self.loss = BoundaryBCEDiceLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            pos_weight=pos_weight,
            smooth=smooth,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "BoundaryKDLoss":
        kd_cfg = cfg.get("kd", cfg)
        boundary_cfg = kd_cfg.get("boundary_kd", {})

        return cls(
            weight=float(boundary_cfg.get("weight", kd_cfg.get("weights", {}).get("boundary_kd", 0.3))),
            pos_weight=float(boundary_cfg.get("pos_weight", 4.0)),
            smooth=float(boundary_cfg.get("smooth", 1.0)),
            enabled=bool(boundary_cfg.get("enabled", True)),
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        edge_logits = outputs["edge_logits"]
        device = edge_logits.device

        if not self.enabled:
            zero = torch.zeros((), device=device)
            return zero, {"kd_boundary_total": 0.0}

        if "teacher_boundary" not in batch:
            raise KeyError("Batch missing teacher_boundary for BoundaryKDLoss.")

        teacher_boundary = batch["teacher_boundary"].to(device).float()

        loss = self.loss(edge_logits, teacher_boundary)
        total = self.weight * loss

        logs = {
            "kd_boundary_raw": float(loss.detach().cpu()),
            "kd_boundary_total": float(total.detach().cpu()),
        }

        return total, logs