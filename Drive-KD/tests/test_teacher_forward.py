from __future__ import annotations

import torch

from drive_kd.models.teachers import SegFormerB1Teacher


def test_teacher_forward_shapes_offline_init() -> None:

    model = SegFormerB1Teacher(
        image_height=128,
        image_width=192,
        pretrained=False,
        decoder_hidden_dim=64,
        return_features_default=True,
        return_attention_default=True,
        return_probabilities_default=True,
    )

    model.eval()

    x = torch.randn(1, 3, 128, 192)

    with torch.no_grad():
        out = model(x)

    assert out["road_logits"].shape == (1, 2, 128, 192)
    assert out["lane_logits"].shape == (1, 2, 128, 192)
    assert out["edge_logits"].shape == (1, 1, 128, 192)

    assert out["road_prob"].shape == (1, 128, 192)
    assert out["lane_prob"].shape == (1, 128, 192)
    assert out["edge_prob"].shape == (1, 128, 192)

    assert "features" in out
    assert "s1" in out["features"]
    assert "s2" in out["features"]
    assert "s3" in out["features"]
    assert "s4" in out["features"]
    assert "decoded" in out["features"]

    assert "attention" in out
    assert "s2" in out["attention"]
    assert "s4" in out["attention"]


def test_teacher_freeze_unfreeze_encoder() -> None:
    model = SegFormerB1Teacher(
        image_height=128,
        image_width=192,
        pretrained=False,
        decoder_hidden_dim=64,
    )

    model.freeze_encoder()
    assert all(not p.requires_grad for p in model.encoder.parameters())

    model.unfreeze_encoder()
    assert all(p.requires_grad for p in model.encoder.parameters())