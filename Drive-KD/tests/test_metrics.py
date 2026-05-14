from __future__ import annotations

import torch

from drive_kd.metrics import (
    BinaryConfusionCounts,
    DriveEvaluator,
    binary_confusion_counts,
    binary_metrics_from_counts,
    count_parameters,
    estimate_model_size_mb,
)
from drive_kd.models.students import DriveEffB0BiFPNKD


def test_binary_confusion_counts() -> None:
    pred = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)
    target = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

    counts = binary_confusion_counts(pred, target, threshold=0.5)

    assert counts.tp == 1
    assert counts.fp == 1
    assert counts.fn == 1
    assert counts.tn == 1


def test_binary_metrics_from_counts() -> None:
    counts = BinaryConfusionCounts(tp=10, fp=5, fn=5, tn=80)

    metrics = binary_metrics_from_counts(counts, prefix="test/")

    assert 0.0 <= metrics["test/precision"] <= 1.0
    assert 0.0 <= metrics["test/recall"] <= 1.0
    assert 0.0 <= metrics["test/f1"] <= 1.0
    assert 0.0 <= metrics["test/iou"] <= 1.0
    assert 0.0 <= metrics["test/dice"] <= 1.0


def test_drive_evaluator_update_compute() -> None:
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

    evaluator = DriveEvaluator()
    evaluator.update(outputs, batch)
    metrics = evaluator.compute()

    assert "road_iou" in metrics
    assert "lane_f1" in metrics
    assert "edge_f1" in metrics

    assert 0.0 <= metrics["road_iou"] <= 1.0
    assert 0.0 <= metrics["lane_f1"] <= 1.0
    assert 0.0 <= metrics["edge_f1"] <= 1.0


def test_efficiency_metrics() -> None:
    model = DriveEffB0BiFPNKD(
        image_height=128,
        image_width=192,
        encoder_pretrained=False,
        neck_channels=32,
        bifpn_repeats=1,
    )

    params = count_parameters(model)
    size_mb = estimate_model_size_mb(model)

    assert params > 0
    assert size_mb > 0