from __future__ import annotations

import torch
import torch.nn.functional as F

from drive_kd.engine.amp_utils import get_autocast_context
from drive_kd.utils.device import move_batch_to_device


@torch.no_grad()
def run_teacher_inference_batch(
    *,
    model: torch.nn.Module,
    batch: dict,
    device: torch.device | str,
    amp_enabled: bool = True,
    return_attention: bool = True,
) -> dict:
    
    device = torch.device(device)
    model.eval()

    batch = move_batch_to_device(batch, device)

    with get_autocast_context(enabled=amp_enabled, device=device):
        outputs = model(
            batch["image"],
            return_features=return_attention,
            return_attention=return_attention,
            return_probabilities=True,
        )

    if "road_prob" not in outputs:
        outputs["road_prob"] = F.softmax(outputs["road_logits"], dim=1)[:, 1]

    if "lane_prob" not in outputs:
        outputs["lane_prob"] = F.softmax(outputs["lane_logits"], dim=1)[:, 1]

    if "edge_prob" not in outputs:
        outputs["edge_prob"] = torch.sigmoid(outputs["edge_logits"]).squeeze(1)

    return outputs