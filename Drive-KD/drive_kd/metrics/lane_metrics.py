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
class LaneMetrics:

    threshold: float = 0.35
    counts: BinaryConfusionCounts = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = BinaryConfusionCounts()

    @torch.no_grad()
    def update_from_logits(self, lane_logits: torch.Tensor, lane_target: torch.Tensor) -> None:
        lane_prob = foreground_prob_from_logits(lane_logits)
        self.update_from_prob(lane_prob, lane_target)

    @torch.no_grad()
    def update_from_prob(self, lane_prob: torch.Tensor, lane_target: torch.Tensor) -> None:
        batch_counts = lane_counts(
            lane_prob=lane_prob,
            lane_target=lane_target,
            threshold=self.threshold,
        )
        self.counts.update(batch_counts)

    def compute(self, prefix: str = "lane/") -> dict[str, float]:
        return lane_metrics_from_counts(self.counts, prefix=prefix)

    def reset(self) -> None:
        self.counts = BinaryConfusionCounts()


@torch.no_grad()
def lane_counts(
    lane_prob: torch.Tensor,
    lane_target: torch.Tensor,
    threshold: float = 0.35,
) -> BinaryConfusionCounts:

    return binary_confusion_counts(
        pred=lane_prob,
        target=lane_target,
        threshold=threshold,
    )


def lane_metrics_from_counts(
    counts: BinaryConfusionCounts,
    prefix: str = "lane/",
) -> dict[str, float]:

    return binary_metrics_from_counts(counts, prefix=prefix)