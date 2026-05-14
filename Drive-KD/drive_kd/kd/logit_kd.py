from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from drive_kd.kd.temperature import binary_logits_from_two_class_logits


class MulticlassLogitKDLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 3.0,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Logit KD shape mismatch: student={student_logits.shape}, "
                f"teacher={teacher_logits.shape}"
            )

        t = self.temperature

        student_log_prob = F.log_softmax(student_logits / t, dim=1)
        teacher_prob = F.softmax(teacher_logits / t, dim=1)

        kl = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="none",
        )

        kl = kl.sum(dim=1).mean()

        return kl * (t * t)


class BinaryLogitKDLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = float(temperature)

    @staticmethod
    def _squeeze(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            return x[:, 0]
        return x

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        student_logits = self._squeeze(student_logits).float()
        teacher_logits = self._squeeze(teacher_logits).float()

        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Binary logit KD shape mismatch: student={student_logits.shape}, "
                f"teacher={teacher_logits.shape}"
            )

        t = self.temperature

        teacher_prob = torch.sigmoid(teacher_logits / t)

        loss = F.binary_cross_entropy_with_logits(
            student_logits / t,
            teacher_prob,
        )

        return loss * (t * t)


class MultiTaskLogitKDLoss(nn.Module):

    def __init__(
        self,
        road_temperature: float = 3.0,
        lane_temperature: float = 4.0,
        edge_temperature: float = 2.0,
        road_weight: float = 1.0,
        lane_weight: float = 1.5,
        edge_weight: float = 1.5,
    ) -> None:
        super().__init__()

        self.road_loss = MulticlassLogitKDLoss(temperature=road_temperature)
        self.lane_loss = MulticlassLogitKDLoss(temperature=lane_temperature)
        self.edge_loss = BinaryLogitKDLoss(temperature=edge_temperature)

        self.road_weight = float(road_weight)
        self.lane_weight = float(lane_weight)
        self.edge_weight = float(edge_weight)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MultiTaskLogitKDLoss":
        kd_cfg = cfg.get("kd", cfg)
        temp_cfg = kd_cfg.get("temperature", {})
        tasks_cfg = kd_cfg.get("tasks", {})

        return cls(
            road_temperature=float(temp_cfg.get("road", 3.0)),
            lane_temperature=float(temp_cfg.get("lane", 4.0)),
            edge_temperature=float(temp_cfg.get("edge", 2.0)),
            road_weight=float(tasks_cfg.get("road", {}).get("kd_weight", 1.0)),
            lane_weight=float(tasks_cfg.get("lane", {}).get("kd_weight", 1.5)),
            edge_weight=float(tasks_cfg.get("edge", {}).get("kd_weight", 1.5)),
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        teacher_logits: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        road = self.road_loss(outputs["road_logits"], teacher_logits["road_logits"])
        lane = self.lane_loss(outputs["lane_logits"], teacher_logits["lane_logits"])
        edge = self.edge_loss(outputs["edge_logits"], teacher_logits["edge_logits"])

        total = (
            self.road_weight * road
            + self.lane_weight * lane
            + self.edge_weight * edge
        )

        logs = {
            "kd_logit_road": float(road.detach().cpu()),
            "kd_logit_lane": float(lane.detach().cpu()),
            "kd_logit_edge": float(edge.detach().cpu()),
            "kd_logit_total": float(total.detach().cpu()),
        }

        return total, logs