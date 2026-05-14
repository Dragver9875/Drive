from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def _build_affine_transform(cfg: dict[str, Any]) -> A.BasicTransform:

    translate_percent = {
        "x": tuple(cfg.get("translate_percent_x", [-0.05, 0.05])),
        "y": tuple(cfg.get("translate_percent_y", [-0.05, 0.05])),
    }

    scale = tuple(cfg.get("scale", [0.90, 1.10]))
    rotate = tuple(cfg.get("rotate", [-3.0, 3.0]))
    p = float(cfg.get("p", 0.35))

    try:
        return A.Affine(
            translate_percent=translate_percent,
            scale=scale,
            rotate=rotate,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=p,
        )
    except TypeError:
        return A.Affine(
            translate_percent=translate_percent,
            scale=scale,
            rotate=rotate,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            mode=cv2.BORDER_CONSTANT,
            cval=0,
            cval_mask=0,
            p=p,
        )


def build_drive_transforms(
    *,
    image_height: int,
    image_width: int,
    mean: list[float],
    std: list[float],
    train: bool,
    transform_config: dict[str, Any] | None = None,
    include_teacher_targets: bool = False,
) -> A.Compose:

    transform_config = transform_config or {}

    transforms: list[A.BasicTransform] = [
        A.Resize(
            height=image_height,
            width=image_width,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        )
    ]

    if train:
        train_cfg = transform_config.get("train", transform_config)

        if train_cfg.get("horizontal_flip", {}).get("enabled", True):
            transforms.append(
                A.HorizontalFlip(
                    p=float(train_cfg.get("horizontal_flip", {}).get("p", 0.5))
                )
            )

        if train_cfg.get("brightness_contrast", {}).get("enabled", True):
            bc = train_cfg.get("brightness_contrast", {})
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=float(bc.get("brightness_limit", 0.20)),
                    contrast_limit=float(bc.get("contrast_limit", 0.20)),
                    p=float(bc.get("p", 0.35)),
                )
            )

        if train_cfg.get("hue_saturation_value", {}).get("enabled", True):
            hsv = train_cfg.get("hue_saturation_value", {})
            transforms.append(
                A.HueSaturationValue(
                    hue_shift_limit=int(hsv.get("hue_shift_limit", 10)),
                    sat_shift_limit=int(hsv.get("sat_shift_limit", 25)),
                    val_shift_limit=int(hsv.get("val_shift_limit", 20)),
                    p=float(hsv.get("p", 0.25)),
                )
            )

        if train_cfg.get("motion_blur", {}).get("enabled", True):
            mb = train_cfg.get("motion_blur", {})
            transforms.append(
                A.MotionBlur(
                    blur_limit=int(mb.get("blur_limit", 5)),
                    p=float(mb.get("p", 0.15)),
                )
            )

        if train_cfg.get("gaussian_blur", {}).get("enabled", True):
            gb = train_cfg.get("gaussian_blur", {})
            blur_limit = int(gb.get("blur_limit", 3))
            transforms.append(
                A.GaussianBlur(
                    blur_limit=(blur_limit, blur_limit),
                    p=float(gb.get("p", 0.10)),
                )
            )

        if train_cfg.get("affine", {}).get("enabled", True):
            transforms.append(_build_affine_transform(train_cfg.get("affine", {})))

    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    additional_targets = {
        "road_mask": "mask",
        "lane_mask": "mask",
        "edge_mask": "mask",
    }

    if include_teacher_targets:
        additional_targets.update(
            {
                "teacher_road_prob": "mask",
                "teacher_lane_prob": "mask",
                "teacher_edge_prob": "mask",
                "teacher_boundary": "mask",
            }
        )

    return A.Compose(transforms, additional_targets=additional_targets)