from __future__ import annotations

import torch

from drive_kd.models.students import DriveEffB0BiFPNKD


def test_student_forward_shapes() -> None:
    model = DriveEffB0BiFPNKD(
        image_height=128,
        image_width=192,
        encoder_pretrained=False,
        neck_channels=32,
        bifpn_repeats=1,
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
    assert "attention" in out

    assert "p2" in out["attention"]
    assert "p4" in out["attention"]


def test_student_export_mode_tuple_wrapper_behavior() -> None:
    model = DriveEffB0BiFPNKD(
        image_height=128,
        image_width=192,
        encoder_pretrained=False,
        neck_channels=32,
        bifpn_repeats=1,
        return_features_default=False,
        return_attention_default=False,
        return_probabilities_default=False,
    )

    model.eval()

    x = torch.randn(1, 3, 128, 192)

    with torch.no_grad():
        out = model(
            x,
            return_features=False,
            return_attention=False,
            return_probabilities=False,
        )

    assert "road_logits" in out
    assert "lane_logits" in out
    assert "edge_logits" in out
    assert "features" not in out
    assert "attention" not in out