from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from drive_kd.kd.attention_kd import AttentionKDLoss
from drive_kd.kd.boundary_kd import BoundaryKDLoss
from drive_kd.kd.kd_weight_scheduler import KDWeightScheduler
from drive_kd.kd.probability_kd import ProbabilityKDLoss
from drive_kd.losses.supervised_multitask_loss import SupervisedMultiTaskLoss


class MultiTaskKDLoss(nn.Module):

    def __init__(
        self,
        supervised_loss: SupervisedMultiTaskLoss,
        probability_kd_loss: ProbabilityKDLoss,
        boundary_kd_loss: BoundaryKDLoss,
        attention_kd_loss: AttentionKDLoss,
        scheduler: KDWeightScheduler | None = None,
    ) -> None:
        super().__init__()

        self.supervised_loss = supervised_loss
        self.probability_kd_loss = probability_kd_loss
        self.boundary_kd_loss = boundary_kd_loss
        self.attention_kd_loss = attention_kd_loss
        self.scheduler = scheduler or KDWeightScheduler()

    @classmethod
    def from_config(
        cls,
        *,
        experiment_config: dict[str, Any] | None = None,
        kd_config: dict[str, Any] | None = None,
    ) -> "MultiTaskKDLoss":

        experiment_config = experiment_config or {}
        kd_config = kd_config or {}

        supervised = SupervisedMultiTaskLoss.from_config(experiment_config)

        probability_kd = ProbabilityKDLoss.from_config(kd_config)
        boundary_kd = BoundaryKDLoss.from_config(kd_config)
        attention_kd = AttentionKDLoss.from_config(kd_config)

        scheduler = KDWeightScheduler.from_config(kd_config)

        return cls(
            supervised_loss=supervised,
            probability_kd_loss=probability_kd,
            boundary_kd_loss=boundary_kd,
            attention_kd_loss=attention_kd,
            scheduler=scheduler,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        *,
        epoch: int = 1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = self.scheduler.get_weights(epoch)

        supervised_raw, supervised_logs = self.supervised_loss(outputs, batch)
        probability_raw, probability_logs = self.probability_kd_loss(outputs, batch)
        boundary_raw, boundary_logs = self.boundary_kd_loss(outputs, batch)
        attention_raw, attention_logs = self.attention_kd_loss(outputs, batch)

        total = (
            weights.supervised * supervised_raw
            + weights.probability_kd * probability_raw
            + weights.boundary_kd * boundary_raw
            + weights.attention_kd * attention_raw
        )

        logs = OrderedDict()

        logs["loss_total"] = float(total.detach().cpu())

        logs["loss_supervised_weighted"] = float(
            (weights.supervised * supervised_raw).detach().cpu()
        )
        logs["loss_probability_kd_weighted"] = float(
            (weights.probability_kd * probability_raw).detach().cpu()
        )
        logs["loss_boundary_kd_weighted"] = float(
            (weights.boundary_kd * boundary_raw).detach().cpu()
        )
        logs["loss_attention_kd_weighted"] = float(
            (weights.attention_kd * attention_raw).detach().cpu()
        )

        logs["weight_supervised"] = float(weights.supervised)
        logs["weight_probability_kd"] = float(weights.probability_kd)
        logs["weight_boundary_kd"] = float(weights.boundary_kd)
        logs["weight_attention_kd"] = float(weights.attention_kd)

        for k, v in supervised_logs.items():
            logs[f"sup/{k}"] = v

        for k, v in probability_logs.items():
            logs[f"prob/{k}"] = v

        for k, v in boundary_logs.items():
            logs[f"boundary/{k}"] = v

        for k, v in attention_logs.items():
            logs[f"attention/{k}"] = v

        return total, dict(logs)