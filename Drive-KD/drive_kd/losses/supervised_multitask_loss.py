from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from drive_kd.losses.boundary_loss import BoundaryBCEDiceLoss
from drive_kd.losses.dice_loss import BinaryDiceLoss
from drive_kd.losses.focal_loss import FocalCrossEntropyLoss


class SupervisedMultiTaskLoss(nn.Module):

    def __init__(
        self,
        road_weight: float = 1.0,
        lane_weight: float = 1.0,
        edge_weight: float = 1.5,
        road_class_weights: list[float] | tuple[float, float] | None = (0.40, 1.00),
        lane_focal_alpha: float = 0.75,
        lane_focal_gamma: float = 2.0,
        edge_pos_weight: float = 4.0,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.road_weight = float(road_weight)
        self.lane_weight = float(lane_weight)
        self.edge_weight = float(edge_weight)

        if road_class_weights is not None:
            self.register_buffer(
                "road_class_weights",
                torch.tensor(road_class_weights, dtype=torch.float32),
            )
        else:
            self.road_class_weights = None

        self.road_dice = BinaryDiceLoss(smooth=dice_smooth)

        self.lane_focal = FocalCrossEntropyLoss(
            alpha=lane_focal_alpha,
            gamma=lane_focal_gamma,
        )
        self.lane_dice = BinaryDiceLoss(smooth=dice_smooth)

        self.edge_loss = BoundaryBCEDiceLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            pos_weight=edge_pos_weight,
            smooth=dice_smooth,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SupervisedMultiTaskLoss":

        loss_cfg = cfg.get("loss", cfg)

        road_cfg = loss_cfg.get("road", {})
        lane_cfg = loss_cfg.get("lane", {})
        edge_cfg = loss_cfg.get("edge", {})

        road_ce_cfg = road_cfg.get("ce", {})
        lane_focal_cfg = lane_cfg.get("focal_ce", {})
        edge_bce_cfg = edge_cfg.get("bce", {})

        dice_smooth = (
            road_cfg.get("dice", {}).get(
                "smooth",
                lane_cfg.get("dice", {}).get("smooth", edge_cfg.get("dice", {}).get("smooth", 1.0)),
            )
        )

        return cls(
            road_weight=float(loss_cfg.get("road_weight", 1.0)),
            lane_weight=float(loss_cfg.get("lane_weight", 1.0)),
            edge_weight=float(loss_cfg.get("edge_weight", 1.5)),
            road_class_weights=road_ce_cfg.get("class_weights", [0.40, 1.00]),
            lane_focal_alpha=float(lane_focal_cfg.get("alpha", 0.75)),
            lane_focal_gamma=float(lane_focal_cfg.get("gamma", 2.0)),
            edge_pos_weight=float(edge_bce_cfg.get("pos_weight", 4.0)),
            dice_smooth=float(dice_smooth),
        )

    @staticmethod
    def _get_required(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in outputs:
            raise KeyError(f"Model outputs missing required key: {key}")
        return outputs[key]

    @staticmethod
    def _get_required_batch(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"Batch missing required key: {key}")
        return batch[key]

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        road_logits = self._get_required(outputs, "road_logits")
        lane_logits = self._get_required(outputs, "lane_logits")
        edge_logits = self._get_required(outputs, "edge_logits")

        road_target = self._get_required_batch(batch, "road_mask").long()
        lane_target = self._get_required_batch(batch, "lane_mask").long()
        edge_target = self._get_required_batch(batch, "edge_mask").float()

        road_ce = F.cross_entropy(
            road_logits,
            road_target,
            weight=self.road_class_weights.to(road_logits.device)
            if self.road_class_weights is not None
            else None,
        )

        road_prob = torch.softmax(road_logits, dim=1)[:, 1]
        road_dice = self.road_dice(road_prob, road_target.float())

        road_loss = road_ce + road_dice

        lane_focal = self.lane_focal(lane_logits, lane_target)

        lane_prob = torch.softmax(lane_logits, dim=1)[:, 1]
        lane_dice = self.lane_dice(lane_prob, lane_target.float())

        lane_loss = lane_focal + lane_dice

        edge_loss = self.edge_loss(edge_logits, edge_target)

        total = (
            self.road_weight * road_loss
            + self.lane_weight * lane_loss
            + self.edge_weight * edge_loss
        )

        log_items = OrderedDict(
            {
                "loss_total": float(total.detach().cpu()),
                "loss_road": float(road_loss.detach().cpu()),
                "loss_road_ce": float(road_ce.detach().cpu()),
                "loss_road_dice": float(road_dice.detach().cpu()),
                "loss_lane": float(lane_loss.detach().cpu()),
                "loss_lane_focal": float(lane_focal.detach().cpu()),
                "loss_lane_dice": float(lane_dice.detach().cpu()),
                "loss_edge": float(edge_loss.detach().cpu()),
            }
        )

        return total, dict(log_items)