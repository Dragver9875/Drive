from __future__ import annotations

from typing import Any

from drive_kd.models.students.drive_effb0_bifpn import DriveEffB0BiFPN


class DriveEffB0BiFPNKD(DriveEffB0BiFPN):

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DriveEffB0BiFPNKD":
        student_cfg = cfg.get("student", cfg)

        input_cfg = student_cfg.get("input", {})
        encoder_cfg = student_cfg.get("encoder", {})
        neck_cfg = student_cfg.get("neck", {})
        sppf_cfg = student_cfg.get("sppf", {})
        output_cfg = student_cfg.get("output", {})

        return cls(
            image_height=int(input_cfg.get("height", 384)),
            image_width=int(input_cfg.get("width", 640)),
            encoder_pretrained=bool(encoder_cfg.get("pretrained", True)),
            encoder_freeze=bool(encoder_cfg.get("freeze", False)),
            encoder_model_name=str(encoder_cfg.get("timm_model_name", "efficientnet_b0")),
            neck_channels=int(neck_cfg.get("channels", 96)),
            sppf_enabled=bool(sppf_cfg.get("enabled", True)),
            sppf_kernel_size=int(sppf_cfg.get("kernel_size", 5)),
            bifpn_repeats=int(neck_cfg.get("num_repeats", 1)),
            use_depthwise_bifpn=bool(neck_cfg.get("use_depthwise_refine", False)),
            return_features_default=bool(output_cfg.get("return_features", True)),
            return_attention_default=bool(output_cfg.get("return_attention", True)),
            return_probabilities_default=bool(output_cfg.get("return_probabilities", True)),
        )