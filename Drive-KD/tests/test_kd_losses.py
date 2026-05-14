from __future__ import annotations

import torch

from drive_kd.kd import (
    AttentionKDLoss,
    BoundaryKDLoss,
    KDWeightScheduler,
    MultiTaskKDLoss,
    ProbabilityKDLoss,
)
from drive_kd.losses import SupervisedMultiTaskLoss


def make_fake_outputs_and_batch() -> tuple[dict, dict]:
    b, h, w = 2, 64, 96

    outputs = {
        "road_logits": torch.randn(b, 2, h, w),
        "lane_logits": torch.randn(b, 2, h, w),
        "edge_logits": torch.randn(b, 1, h, w),
        "attention": {
            "p2": torch.randn(b, 1, h // 4, w // 4),
            "p4": torch.randn(b, 1, h // 16, w // 16),
        },
    }

    batch = {
        "road_mask": torch.randint(0, 2, (b, h, w)),
        "lane_mask": torch.randint(0, 2, (b, h, w)),
        "edge_mask": torch.randint(0, 2, (b, 1, h, w)).float(),
        "teacher_road_prob": torch.rand(b, h, w),
        "teacher_lane_prob": torch.rand(b, h, w),
        "teacher_edge_prob": torch.rand(b, h, w),
        "teacher_boundary": torch.randint(0, 2, (b, 1, h, w)).float(),
        "teacher_attention": {
            "s2": torch.randn(b, 1, h // 4, w // 4),
            "s4": torch.randn(b, 1, h // 16, w // 16),
        },
    }

    return outputs, batch


def test_probability_kd_loss_forward() -> None:
    outputs, batch = make_fake_outputs_and_batch()

    loss, logs = ProbabilityKDLoss()(outputs, batch)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert "kd_prob_total" in logs


def test_boundary_kd_loss_forward() -> None:
    outputs, batch = make_fake_outputs_and_batch()

    loss, logs = BoundaryKDLoss()(outputs, batch)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert "kd_boundary_total" in logs


def test_attention_kd_loss_forward() -> None:
    outputs, batch = make_fake_outputs_and_batch()

    loss, logs = AttentionKDLoss()(outputs, batch)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert "kd_attention_total" in logs


def test_kd_weight_scheduler() -> None:
    scheduler = KDWeightScheduler(
        supervised=1.0,
        probability_kd=0.5,
        boundary_kd=0.3,
        attention_kd=0.1,
        kd_warmup_epochs=3,
    )

    w1 = scheduler.get_weights(epoch=1)
    w3 = scheduler.get_weights(epoch=3)
    w4 = scheduler.get_weights(epoch=4)

    assert w1.supervised == 1.0
    assert 0.0 < w1.probability_kd < w3.probability_kd
    assert w3.probability_kd == w4.probability_kd


def test_multitask_kd_loss_forward() -> None:
    outputs, batch = make_fake_outputs_and_batch()

    criterion = MultiTaskKDLoss(
        supervised_loss=SupervisedMultiTaskLoss(),
        probability_kd_loss=ProbabilityKDLoss(),
        boundary_kd_loss=BoundaryKDLoss(),
        attention_kd_loss=AttentionKDLoss(),
        scheduler=KDWeightScheduler(kd_warmup_epochs=3),
    )

    loss, logs = criterion(outputs, batch, epoch=1)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    assert "loss_total" in logs
    assert "loss_supervised_weighted" in logs
    assert "loss_probability_kd_weighted" in logs
    assert "loss_boundary_kd_weighted" in logs
    assert "loss_attention_kd_weighted" in logs