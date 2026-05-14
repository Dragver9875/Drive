from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from drive_kd.models.common.attention import attention_l2_loss


class AttentionKDLoss(nn.Module):

    def __init__(
        self,
        layer_pairs: list[dict[str, Any]] | None = None,
        weight: float = 0.1,
        normalize: bool = True,
        enabled: bool = True,
    ) -> None:
        super().__init__()

        self.layer_pairs = layer_pairs or [
            {"teacher": "s2", "student": "p2", "weight": 1.0},
            {"teacher": "s4", "student": "p4", "weight": 1.0},
        ]
        self.weight = float(weight)
        self.normalize = bool(normalize)
        self.enabled = bool(enabled)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AttentionKDLoss":
        kd_cfg = cfg.get("kd", cfg)
        att_cfg = kd_cfg.get("attention_kd", {})

        return cls(
            layer_pairs=list(att_cfg.get("layer_pairs", [
                {"teacher": "s2", "student": "p2", "weight": 1.0},
                {"teacher": "s4", "student": "p4", "weight": 1.0},
            ])),
            weight=float(att_cfg.get("weight", kd_cfg.get("weights", {}).get("attention_kd", 0.1))),
            normalize=bool(att_cfg.get("normalize", True)),
            enabled=bool(att_cfg.get("enabled", True)),
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        edge_logits = outputs["edge_logits"]  # used only for device reference
        device = edge_logits.device

        if not self.enabled:
            zero = torch.zeros((), device=device)
            return zero, {"kd_attention_total": 0.0}

        if "attention" not in outputs:
            raise KeyError("Model outputs missing 'attention' for AttentionKDLoss.")

        if "teacher_attention" not in batch:
            raise KeyError("Batch missing 'teacher_attention' for AttentionKDLoss.")

        student_attention = outputs["attention"]
        teacher_attention = batch["teacher_attention"]

        if not isinstance(student_attention, dict):
            raise TypeError("outputs['attention'] must be a dict.")

        if not isinstance(teacher_attention, dict):
            raise TypeError("batch['teacher_attention'] must be a dict.")

        total = torch.zeros((), device=device)
        logs: dict[str, float] = {}

        active_pairs = 0

        for pair in self.layer_pairs:
            teacher_key = str(pair["teacher"])
            student_key = str(pair["student"])
            pair_weight = float(pair.get("weight", 1.0))

            if student_key not in student_attention:
                raise KeyError(f"Student attention missing key: {student_key}")

            if teacher_key not in teacher_attention:
                raise KeyError(f"Teacher attention missing key: {teacher_key}")

            s_att = student_attention[student_key].to(device).float()
            t_att = teacher_attention[teacher_key].to(device).float()

            loss = attention_l2_loss(
                student_attention=s_att,
                teacher_attention=t_att,
                normalize=self.normalize,
            )

            total = total + pair_weight * loss
            active_pairs += 1

            logs[f"kd_attention_{teacher_key}_to_{student_key}"] = float(loss.detach().cpu())

        if active_pairs == 0:
            raise RuntimeError("AttentionKDLoss had zero active layer pairs.")

        total = total / float(active_pairs)
        weighted_total = self.weight * total

        logs["kd_attention_raw"] = float(total.detach().cpu())
        logs["kd_attention_total"] = float(weighted_total.detach().cpu())

        return weighted_total, logs