from __future__ import annotations

from dataclasses import dataclass

import torch

from drive_kd.metrics.boundary_metrics import BoundaryMetrics
from drive_kd.metrics.lane_metrics import LaneMetrics
from drive_kd.metrics.road_metrics import RoadMetrics


@dataclass
class EvaluatorThresholds:

    road: float = 0.50
    lane: float = 0.35
    edge: float = 0.40


class DriveEvaluator:

    def __init__(
        self,
        thresholds: EvaluatorThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or EvaluatorThresholds()

        self.road = RoadMetrics(threshold=self.thresholds.road)
        self.lane = LaneMetrics(threshold=self.thresholds.lane)
        self.edge = BoundaryMetrics(threshold=self.thresholds.edge)

    @classmethod
    def from_config(cls, cfg: dict) -> "DriveEvaluator":

        metrics_cfg = cfg.get("metrics", {})
        thresholds_cfg = metrics_cfg.get("thresholds", {})

        thresholds = EvaluatorThresholds(
            road=float(thresholds_cfg.get("road", 0.50)),
            lane=float(thresholds_cfg.get("lane", 0.35)),
            edge=float(thresholds_cfg.get("edge", 0.40)),
        )

        return cls(thresholds=thresholds)

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> None:
        required_outputs = ["road_logits", "lane_logits", "edge_logits"]
        required_batch = ["road_mask", "lane_mask", "edge_mask"]

        for key in required_outputs:
            if key not in outputs:
                raise KeyError(f"Evaluator missing model output: {key}")

        for key in required_batch:
            if key not in batch:
                raise KeyError(f"Evaluator missing batch key: {key}")

        self.road.update_from_logits(outputs["road_logits"], batch["road_mask"])
        self.lane.update_from_logits(outputs["lane_logits"], batch["lane_mask"])
        self.edge.update_from_logits(outputs["edge_logits"], batch["edge_mask"])

    def compute(self) -> dict[str, float]:
        metrics = {}
        metrics.update(self.road.compute(prefix="road/"))
        metrics.update(self.lane.compute(prefix="lane/"))
        metrics.update(self.edge.compute(prefix="edge/"))

        metrics["road_iou"] = metrics["road/iou"]
        metrics["road_dice"] = metrics["road/dice"]
        metrics["lane_f1"] = metrics["lane/f1"]
        metrics["lane_iou"] = metrics["lane/iou"]
        metrics["edge_f1"] = metrics["edge/f1"]
        metrics["edge_iou"] = metrics["edge/iou"]

        return metrics

    def reset(self) -> None:
        self.road.reset()
        self.lane.reset()
        self.edge.reset()

    def state_dict(self) -> dict[str, dict[str, int]]:
        return {
            "road": self.road.counts.as_dict(),
            "lane": self.lane.counts.as_dict(),
            "edge": self.edge.counts.as_dict(),
        }