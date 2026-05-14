from __future__ import annotations

from dataclasses import dataclass

import torch

from drive_kd.metrics.segmentation_metrics import (
    BinaryConfusionCounts,
    binary_confusion_counts,
    binary_metrics_from_counts,
    foreground_prob_from_logits,
)


@dataclass
class RoadMetrics:

    threshold: float = 0.50
    counts: BinaryConfusionCounts = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = BinaryConfusionCounts()

    @torch.no_grad()
    def update_from_logits(self, road_logits: torch.Tensor, road_target: torch.Tensor) -> None:
        road_prob = foreground_prob_from_logits(road_logits)
        self.update_from_prob(road_prob, road_target)

    @torch.no_grad()
    def update_from_prob(self, road_prob: torch.Tensor, road_target: torch.Tensor) -> None:
        batch_counts = road_counts(
            road_prob=road_prob,
            road_target=road_target,
            threshold=self.threshold,
        )
        self.counts.update(batch_counts)

    def compute(self, prefix: str = "road/") -> dict[str, float]:
        return road_metrics_from_counts(self.counts, prefix=prefix)

    def reset(self) -> None:
        self.counts = BinaryConfusionCounts()


@torch.no_grad()
def road_counts(
    road_prob: torch.Tensor,
    road_target: torch.Tensor,
    threshold: float = 0.50,
) -> BinaryConfusionCounts:

    return binary_confusion_counts(
        pred=road_prob,
        target=road_target,
        threshold=threshold,
    )


def road_metrics_from_counts(
    counts: BinaryConfusionCounts,
    prefix: str = "road/",
) -> dict[str, float]:

    return binary_metrics_from_counts(counts, prefix=prefix)