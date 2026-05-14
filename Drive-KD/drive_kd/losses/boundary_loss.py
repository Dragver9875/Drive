from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drive_kd.losses.dice_loss import BinaryDiceLoss


class BoundaryBCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = 4.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.pos_weight_value = pos_weight
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()

        if logits.ndim == 4 and logits.shape[1] == 1:
            logits_2d = logits[:, 0]
        else:
            logits_2d = logits

        if target.ndim == 4 and target.shape[1] == 1:
            target_2d = target[:, 0]
        else:
            target_2d = target

        if logits_2d.shape != target_2d.shape:
            raise ValueError(
                f"Boundary loss shape mismatch: logits={logits_2d.shape}, "
                f"target={target_2d.shape}"
            )

        if self.pos_weight_value is not None:
            pos_weight = torch.tensor(
                float(self.pos_weight_value),
                dtype=logits_2d.dtype,
                device=logits_2d.device,
            )
            bce = F.binary_cross_entropy_with_logits(
                logits_2d,
                target_2d,
                pos_weight=pos_weight,
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits_2d, target_2d)

        probs = torch.sigmoid(logits_2d)
        dice = self.dice(probs, target_2d)

        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def boundary_f1_soft(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.40,
    eps: float = 1e-7,
) -> dict[str, float]:

    if pred_prob.ndim == 4 and pred_prob.shape[1] == 1:
        pred_prob = pred_prob[:, 0]

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    pred = pred_prob > threshold
    tgt = target > 0.5

    tp = (pred & tgt).sum().item()
    fp = (pred & ~tgt).sum().item()
    fn = (~pred & tgt).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }