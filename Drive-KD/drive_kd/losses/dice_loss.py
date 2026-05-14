from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    dims: tuple[int, ...] | None = None,
) -> torch.Tensor:

    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        raise ValueError(f"Dice shape mismatch: pred={pred.shape}, target={target.shape}")

    if dims is None:
        dims = tuple(range(1, pred.ndim))

    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return dice


class BinaryDiceLoss(nn.Module):

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = probs.float()
        target = target.float()

        if probs.ndim == 4 and probs.shape[1] == 1:
            probs = probs[:, 0]

        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]

        if probs.shape != target.shape:
            raise ValueError(
                f"BinaryDiceLoss shape mismatch: probs={probs.shape}, target={target.shape}"
            )

        dice = dice_coefficient(probs, target, smooth=self.smooth, dims=(1, 2))
        return 1.0 - dice.mean()


class MulticlassDiceLoss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        include_background: bool = True,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("MulticlassDiceLoss requires num_classes >= 2")

        self.num_classes = int(num_classes)
        self.smooth = float(smooth)
        self.include_background = bool(include_background)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"logits must be [B,C,H,W], got {logits.shape}")

        if target.ndim != 3:
            raise ValueError(f"target must be [B,H,W], got {target.shape}")

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits channel count {logits.shape[1]} does not match "
                f"num_classes={self.num_classes}"
            )

        probs = torch.softmax(logits, dim=1)

        target_one_hot = F.one_hot(
            target.long().clamp(min=0, max=self.num_classes - 1),
            num_classes=self.num_classes,
        )
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        start_class = 0 if self.include_background else 1

        dice_values = []

        for cls_idx in range(start_class, self.num_classes):
            cls_dice = dice_coefficient(
                probs[:, cls_idx],
                target_one_hot[:, cls_idx],
                smooth=self.smooth,
                dims=(1, 2),
            )
            dice_values.append(cls_dice)

        if not dice_values:
            raise RuntimeError("No classes selected for MulticlassDiceLoss.")

        dice = torch.stack(dice_values, dim=0).mean()
        return 1.0 - dice