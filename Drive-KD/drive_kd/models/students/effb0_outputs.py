from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class StudentOutput:

    road_logits: torch.Tensor
    lane_logits: torch.Tensor
    edge_logits: torch.Tensor

    road_prob: torch.Tensor | None = None
    lane_prob: torch.Tensor | None = None
    edge_prob: torch.Tensor | None = None

    features: dict[str, torch.Tensor] = field(default_factory=dict)
    attention: dict[str, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "road_logits": self.road_logits,
            "lane_logits": self.lane_logits,
            "edge_logits": self.edge_logits,
        }

        if self.road_prob is not None:
            out["road_prob"] = self.road_prob

        if self.lane_prob is not None:
            out["lane_prob"] = self.lane_prob

        if self.edge_prob is not None:
            out["edge_prob"] = self.edge_prob

        if self.features:
            out["features"] = self.features

        if self.attention:
            out["attention"] = self.attention

        return out