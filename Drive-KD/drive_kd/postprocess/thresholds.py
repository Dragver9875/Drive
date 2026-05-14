from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DriveThresholds:

    road: float = 0.50
    lane: float = 0.35
    edge: float = 0.40

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DriveThresholds":

        if "metrics" in cfg and "thresholds" in cfg["metrics"]:
            t = cfg["metrics"]["thresholds"]
            return cls(
                road=float(t.get("road", 0.50)),
                lane=float(t.get("lane", 0.35)),
                edge=float(t.get("edge", 0.40)),
            )

        if "student" in cfg and "inference" in cfg["student"]:
            t = cfg["student"]["inference"]
            return cls(
                road=float(t.get("road_threshold", 0.50)),
                lane=float(t.get("lane_threshold", 0.35)),
                edge=float(t.get("edge_threshold", 0.40)),
            )

        return cls(
            road=float(cfg.get("road", 0.50)),
            lane=float(cfg.get("lane", 0.35)),
            edge=float(cfg.get("edge", 0.40)),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "road": float(self.road),
            "lane": float(self.lane),
            "edge": float(self.edge),
        }