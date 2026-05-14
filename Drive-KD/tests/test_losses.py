from __future__ import annotations

import torch

from drive_kd.losses import (
    BinaryDiceLoss,
    BoundaryBCEDiceLoss,
    FocalCrossEntropyLoss,
    SupervisedMultiTaskLoss,
)


def test_binary_dice_loss_forward() -> None:
    probs = torch.rand(2, 64, 96)
    target = torch.randint(0, 2, (2, 64, 96)).float()

    loss = BinaryDiceLoss()(probs, target)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_focal_cross_entropy_forward() -> None:
    logits = torch.randn(2, 2, 64, 96)
    target = torch.randint(0, 2, (2, 64, 96))

    loss = FocalCrossEntropyLoss()(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_boundary_bce_dice_forward() -> None:
    logits = torch.randn(2, 1, 64, 96)
    target = torch.randint(0, 2, (2, 1, 64, 96)).float()

    loss = BoundaryBCEDiceLoss()(logits, target)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_supervised_multitask_loss_forward() -> None:
    b, h, w = 2, 64, 96

    outputs = {
        "road_logits": torch.randn(b, 2, h, w),
        "lane_logits": torch.randn(b, 2, h, w),
        "edge_logits": torch.randn(b, 1, h, w),
    }

    batch = {
        "road_mask": torch.randint(0, 2, (b, h, w)),
        "lane_mask": torch.randint(0, 2, (b, h, w)),
        "edge_mask": torch.randint(0, 2, (b, 1, h, w)).float(),
    }

    loss, logs = SupervisedMultiTaskLoss()(outputs, batch)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert "loss_total" in logs
    assert "loss_road" in logs
    assert "loss_lane" in logs
    assert "loss_edge" in logs