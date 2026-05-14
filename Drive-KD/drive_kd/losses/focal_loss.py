from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropyLoss(nn.Module):

    def __init__(
        self,
        alpha: float | torch.Tensor = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha_tensor", alpha.float())
            self.alpha_float: float | None = None
        else:
            self.alpha_float = float(alpha)
            self.alpha_tensor = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"logits must be [B,C,H,W], got {logits.shape}")

        if target.ndim != 3:
            raise ValueError(f"target must be [B,H,W], got {target.shape}")

        ce_kwargs = {
            "input": logits,
            "target": target.long(),
            "reduction": "none",
        }

        if self.ignore_index is not None:
            ce_kwargs["ignore_index"] = self.ignore_index

        ce = F.cross_entropy(**ce_kwargs)

        pt = torch.exp(-ce)

        if self.alpha_tensor is not None:
            alpha = self.alpha_tensor.to(logits.device)
            alpha_factor = alpha[target.long().clamp(0, len(alpha) - 1)]
        else:
            alpha_factor = self.alpha_float

        loss = alpha_factor * (1.0 - pt).pow(self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class BinaryFocalLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()

        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]

        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]

        if logits.shape != target.shape:
            raise ValueError(
                f"BinaryFocalLoss shape mismatch: logits={logits.shape}, target={target.shape}"
            )

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        prob = torch.sigmoid(logits)
        pt = prob * target + (1.0 - prob) * (1.0 - target)

        alpha_factor = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_factor * (1.0 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss