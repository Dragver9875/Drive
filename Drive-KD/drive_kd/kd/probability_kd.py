from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from drive_kd.kd.temperature import binary_logits_from_two_class_logits


def _squeeze_binary_map(x: torch.Tensor) -> torch.Tensor:

    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]

    return x


class SoftTargetBCELoss(nn.Module):

    def __init__(
        self,
        reduction: str = "mean",
        target_clamp_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.reduction = reduction
        self.target_clamp_eps = float(target_clamp_eps)

    def forward(self, student_logits: torch.Tensor, teacher_prob: torch.Tensor) -> torch.Tensor:
        student_logits = _squeeze_binary_map(student_logits).float()
        teacher_prob = _squeeze_binary_map(teacher_prob).float()

        if student_logits.shape != teacher_prob.shape:
            raise ValueError(
                f"SoftTargetBCELoss shape mismatch: "
                f"student_logits={student_logits.shape}, "
                f"teacher_prob={teacher_prob.shape}"
            )

        teacher_prob = teacher_prob.clamp(
            min=self.target_clamp_eps,
            max=1.0 - self.target_clamp_eps,
        )

        return F.binary_cross_entropy_with_logits(
            student_logits,
            teacher_prob,
            reduction=self.reduction,
        )


@dataclass
class ProbabilityKDWeights:
    road: float = 1.0
    lane: float = 1.5
    edge: float = 1.5


class ProbabilityKDLoss(nn.Module):

    def __init__(
        self,
        road_weight: float = 1.0,
        lane_weight: float = 1.5,
        edge_weight: float = 1.5,
        enabled_road: bool = True,
        enabled_lane: bool = True,
        enabled_edge: bool = True,
    ) -> None:
        super().__init__()

        self.weights = ProbabilityKDWeights(
            road=float(road_weight),
            lane=float(lane_weight),
            edge=float(edge_weight),
        )

        self.enabled_road = bool(enabled_road)
        self.enabled_lane = bool(enabled_lane)
        self.enabled_edge = bool(enabled_edge)

        self.soft_bce = SoftTargetBCELoss()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ProbabilityKDLoss":
        kd_cfg = cfg.get("kd", cfg)
        prob_cfg = kd_cfg.get("probability_kd", {})
        tasks_cfg = kd_cfg.get("tasks", {})

        road_cfg = prob_cfg.get("road", {})
        lane_cfg = prob_cfg.get("lane", {})
        edge_cfg = prob_cfg.get("edge", {})

        return cls(
            road_weight=float(road_cfg.get("weight", tasks_cfg.get("road", {}).get("kd_weight", 1.0))),
            lane_weight=float(lane_cfg.get("weight", tasks_cfg.get("lane", {}).get("kd_weight", 1.5))),
            edge_weight=float(edge_cfg.get("weight", tasks_cfg.get("edge", {}).get("kd_weight", 1.5))),
            enabled_road=bool(road_cfg.get("enabled", tasks_cfg.get("road", {}).get("kd", True))),
            enabled_lane=bool(lane_cfg.get("enabled", tasks_cfg.get("lane", {}).get("kd", True))),
            enabled_edge=bool(edge_cfg.get("enabled", tasks_cfg.get("edge", {}).get("kd", True))),
        )

    @staticmethod
    def _required_tensor(container: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in container:
            raise KeyError(f"Missing required key for ProbabilityKDLoss: {key}")
        return container[key]

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = self._required_tensor(outputs, "edge_logits").device
        total = torch.zeros((), device=device)

        logs: dict[str, float] = {}

        if self.enabled_road:
            road_logits = self._required_tensor(outputs, "road_logits")
            teacher_road = self._required_tensor(batch, "teacher_road_prob").to(device)

            road_binary_logits = binary_logits_from_two_class_logits(road_logits)
            road_loss = self.soft_bce(road_binary_logits, teacher_road)

            total = total + self.weights.road * road_loss
            logs["kd_prob_road"] = float(road_loss.detach().cpu())

        if self.enabled_lane:
            lane_logits = self._required_tensor(outputs, "lane_logits")
            teacher_lane = self._required_tensor(batch, "teacher_lane_prob").to(device)

            lane_binary_logits = binary_logits_from_two_class_logits(lane_logits)
            lane_loss = self.soft_bce(lane_binary_logits, teacher_lane)

            total = total + self.weights.lane * lane_loss
            logs["kd_prob_lane"] = float(lane_loss.detach().cpu())

        if self.enabled_edge:
            edge_logits = self._required_tensor(outputs, "edge_logits")
            teacher_edge = self._required_tensor(batch, "teacher_edge_prob").to(device)

            edge_loss = self.soft_bce(edge_logits, teacher_edge)

            total = total + self.weights.edge * edge_loss
            logs["kd_prob_edge"] = float(edge_loss.detach().cpu())

        logs["kd_prob_total"] = float(total.detach().cpu())

        return total, logs