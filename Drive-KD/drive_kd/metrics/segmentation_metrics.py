from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryConfusionCounts:

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, other: "BinaryConfusionCounts") -> None:
        self.tp += int(other.tp)
        self.fp += int(other.fp)
        self.fn += int(other.fn)
        self.tn += int(other.tn)

    def as_dict(self, prefix: str = "") -> dict[str, int]:
        return {
            f"{prefix}tp": int(self.tp),
            f"{prefix}fp": int(self.fp),
            f"{prefix}fn": int(self.fn),
            f"{prefix}tn": int(self.tn),
        }


@torch.no_grad()
def binary_confusion_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float | None = None,
) -> BinaryConfusionCounts:

    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    if pred.shape != target.shape:
        raise ValueError(f"binary_confusion_counts shape mismatch: pred={pred.shape}, target={target.shape}")

    if threshold is not None:
        pred_bool = pred > float(threshold)
    else:
        pred_bool = pred.bool()

    target_bool = target > 0.5

    tp = (pred_bool & target_bool).sum().item()
    fp = (pred_bool & ~target_bool).sum().item()
    fn = (~pred_bool & target_bool).sum().item()
    tn = (~pred_bool & ~target_bool).sum().item()

    return BinaryConfusionCounts(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def precision_from_counts(counts: BinaryConfusionCounts, eps: float = 1e-7) -> float:
    return counts.tp / (counts.tp + counts.fp + eps)


def recall_from_counts(counts: BinaryConfusionCounts, eps: float = 1e-7) -> float:
    return counts.tp / (counts.tp + counts.fn + eps)


def f1_from_counts(counts: BinaryConfusionCounts, eps: float = 1e-7) -> float:
    precision = precision_from_counts(counts, eps=eps)
    recall = recall_from_counts(counts, eps=eps)
    return 2.0 * precision * recall / (precision + recall + eps)


def iou_from_counts(counts: BinaryConfusionCounts, eps: float = 1e-7) -> float:
    return counts.tp / (counts.tp + counts.fp + counts.fn + eps)


def dice_from_counts(counts: BinaryConfusionCounts, eps: float = 1e-7) -> float:
    return (2.0 * counts.tp) / (2.0 * counts.tp + counts.fp + counts.fn + eps)


def accuracy_from_counts(counts: BinaryConfusionCounts, eps: float = 1e-7) -> float:
    total = counts.tp + counts.fp + counts.fn + counts.tn
    return (counts.tp + counts.tn) / (total + eps)


def specificity_from_counts(counts: BinaryConfusionCounts, eps: float = 1e-7) -> float:
    return counts.tn / (counts.tn + counts.fp + eps)


def binary_metrics_from_counts(
    counts: BinaryConfusionCounts,
    *,
    prefix: str = "",
    eps: float = 1e-7,
) -> dict[str, float]:

    return {
        f"{prefix}precision": precision_from_counts(counts, eps=eps),
        f"{prefix}recall": recall_from_counts(counts, eps=eps),
        f"{prefix}f1": f1_from_counts(counts, eps=eps),
        f"{prefix}iou": iou_from_counts(counts, eps=eps),
        f"{prefix}dice": dice_from_counts(counts, eps=eps),
        f"{prefix}accuracy": accuracy_from_counts(counts, eps=eps),
        f"{prefix}specificity": specificity_from_counts(counts, eps=eps),
    }


@torch.no_grad()
def foreground_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:

    if logits.ndim == 4 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]

    if logits.ndim == 4 and logits.shape[1] == 1:
        return torch.sigmoid(logits)[:, 0]

    if logits.ndim == 3:
        return logits

    raise ValueError(f"Unsupported logits/probability shape: {logits.shape}")