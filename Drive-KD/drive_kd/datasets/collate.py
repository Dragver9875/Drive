from __future__ import annotations

from typing import Any

import torch


def _stack_optional_tensor(batch: list[dict[str, Any]], key: str) -> torch.Tensor | None:
    if key not in batch[0] or batch[0][key] is None:
        return None
    return torch.stack([b[key] for b in batch], dim=0)


def drive_supervised_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "road_mask": torch.stack([b["road_mask"] for b in batch], dim=0),
        "lane_mask": torch.stack([b["lane_mask"] for b in batch], dim=0),
        "edge_mask": torch.stack([b["edge_mask"] for b in batch], dim=0),
        "image_id": [b["image_id"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }

    return out


def drive_kd_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = drive_supervised_collate(batch)

    out["teacher_road_prob"] = torch.stack([b["teacher_road_prob"] for b in batch], dim=0)
    out["teacher_lane_prob"] = torch.stack([b["teacher_lane_prob"] for b in batch], dim=0)
    out["teacher_edge_prob"] = torch.stack([b["teacher_edge_prob"] for b in batch], dim=0)
    out["teacher_boundary"] = torch.stack([b["teacher_boundary"] for b in batch], dim=0)

    attention_keys = set()

    for b in batch:
        for key in b.get("teacher_attention", {}).keys():
            attention_keys.add(key)

    out["teacher_attention"] = {}

    for key in sorted(attention_keys):
        values = [b["teacher_attention"].get(key) for b in batch]

        if any(v is None for v in values):
            continue

        out["teacher_attention"][key] = torch.stack(values, dim=0)

    out["teacher_prob_path"] = [b["teacher_prob_path"] for b in batch]
    out["teacher_boundary_path"] = [b["teacher_boundary_path"] for b in batch]
    out["teacher_attention_path"] = [b["teacher_attention_path"] for b in batch]

    return out