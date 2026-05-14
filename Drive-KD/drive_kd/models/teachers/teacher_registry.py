from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from drive_kd.models.teachers.segformer_b1_teacher import SegFormerB1Teacher


TEACHER_REGISTRY: dict[str, Callable[[dict[str, Any]], nn.Module]] = {}


def register_teacher(name: str):

    def decorator(fn: Callable[[dict[str, Any]], nn.Module]):
        key = name.lower()
        TEACHER_REGISTRY[key] = fn
        return fn

    return decorator


@register_teacher("segformer_b1")
def _build_segformer_b1(cfg: dict[str, Any]) -> nn.Module:
    return SegFormerB1Teacher.from_config(cfg)


@register_teacher("SegFormer-B1-DriveTeacher")
def _build_segformer_b1_pretty(cfg: dict[str, Any]) -> nn.Module:
    return SegFormerB1Teacher.from_config(cfg)


def build_teacher(cfg: dict[str, Any]) -> nn.Module:

    teacher_cfg = cfg.get("teacher", cfg)
    teacher_type = str(teacher_cfg.get("type", "segformer_b1")).lower()

    if teacher_type not in TEACHER_REGISTRY:
        available = sorted(TEACHER_REGISTRY.keys())
        raise KeyError(f"Unknown teacher type: {teacher_type}. Available: {available}")

    return TEACHER_REGISTRY[teacher_type](cfg)