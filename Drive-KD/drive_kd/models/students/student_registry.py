from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from drive_kd.models.students.drive_effb0_bifpn import DriveEffB0BiFPN
from drive_kd.models.students.drive_effb0_bifpn_kd import DriveEffB0BiFPNKD


STUDENT_REGISTRY: dict[str, Callable[[dict[str, Any]], nn.Module]] = {}


def register_student(name: str):

    def decorator(fn: Callable[[dict[str, Any]], nn.Module]):
        key = name.lower()
        STUDENT_REGISTRY[key] = fn
        return fn

    return decorator


@register_student("drive_effb0_bifpn")
def _build_drive_effb0_bifpn(cfg: dict[str, Any]) -> nn.Module:
    return DriveEffB0BiFPN.from_config(cfg)


@register_student("drive_effb0_bifpn_kd")
def _build_drive_effb0_bifpn_kd(cfg: dict[str, Any]) -> nn.Module:
    return DriveEffB0BiFPNKD.from_config(cfg)


@register_student("Drive-EffB0-BiFPN-KD")
def _build_drive_effb0_bifpn_kd_pretty(cfg: dict[str, Any]) -> nn.Module:
    return DriveEffB0BiFPNKD.from_config(cfg)


def build_student(cfg: dict[str, Any]) -> nn.Module:

    student_cfg = cfg.get("student", cfg)
    student_type = str(student_cfg.get("type", "drive_effb0_bifpn_kd")).lower()

    if student_type not in STUDENT_REGISTRY:
        available = sorted(STUDENT_REGISTRY.keys())
        raise KeyError(f"Unknown student type: {student_type}. Available: {available}")

    return STUDENT_REGISTRY[student_type](cfg)