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
class BoundaryMetrics:

    threshold: float = 0.40
    counts: BinaryConfusionCounts = None

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = BinaryConfusionCounts()

    @torch.no_grad()
    def update_from_logits(self, edge_logits: torch.Tensor, edge_target: torch.Tensor) -> None:
        edge_prob = foreground_prob_from_logits(edge_logits)
        self.update_from_prob(edge_prob, edge_target)

    @torch.no_grad()
    def update_from_prob(self, edge_prob: torch.Tensor, edge_target: torch.Tensor) -> None:
        batch_counts = boundary_counts(
            edge_prob=edge_prob,
            edge_target=edge_target,
            threshold=self.threshold,
        )
        self.counts.update(batch_counts)

    def compute(self, prefix: str = "edge/") -> dict[str, float]:
        return boundary_metrics_from_counts(self.counts, prefix=prefix)

    def reset(self) -> None:
        self.counts = BinaryConfusionCounts()


@torch.no_grad()
def boundary_counts(
    edge_prob: torch.Tensor,
    edge_target: torch.Tensor,
    threshold: float = 0.40,
) -> BinaryConfusionCounts:

    return binary_confusion_counts(
        pred=edge_prob,
        target=edge_target,
        threshold=threshold,
    )


def boundary_metrics_from_counts(
    counts: BinaryConfusionCounts,
    prefix: str = "edge/",
) -> dict[str, float]:

    return binary_metrics_from_counts(counts, prefix=prefix)