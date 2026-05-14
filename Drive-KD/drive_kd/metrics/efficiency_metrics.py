from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(
    model: nn.Module,
    bytes_per_param: int = 4,
    trainable_only: bool = False,
) -> float:

    params = count_parameters(model, trainable_only=trainable_only)
    return params * bytes_per_param / (1024.0**2)


@dataclass
class FPSBenchmarkResult:

    batch_size: int
    image_height: int
    image_width: int
    warmup_iters: int
    benchmark_iters: int
    latency_ms_mean: float
    latency_ms_median: float
    fps: float
    device: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "batch_size": int(self.batch_size),
            "image_height": int(self.image_height),
            "image_width": int(self.image_width),
            "warmup_iters": int(self.warmup_iters),
            "benchmark_iters": int(self.benchmark_iters),
            "latency_ms_mean": float(self.latency_ms_mean),
            "latency_ms_median": float(self.latency_ms_median),
            "fps": float(self.fps),
            "device": str(self.device),
        }


@torch.no_grad()
def benchmark_fps(
    model: nn.Module,
    *,
    image_height: int = 384,
    image_width: int = 640,
    batch_size: int = 1,
    channels: int = 3,
    warmup_iters: int = 20,
    benchmark_iters: int = 100,
    device: torch.device | str | None = None,
    use_amp: bool = True,
) -> FPSBenchmarkResult:

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, channels, image_height, image_width, device=device)

    amp_enabled = bool(use_amp and device.type == "cuda")

    for _ in range(warmup_iters):
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            _ = model(x, return_features=False, return_attention=False, return_probabilities=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms: list[float] = []

    for _ in range(benchmark_iters):
        start = time.perf_counter()

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            _ = model(x, return_features=False, return_attention=False, return_probabilities=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)

    latencies = torch.tensor(latencies_ms)
    latency_mean = float(latencies.mean().item())
    latency_median = float(latencies.median().item())

    fps = 1000.0 * batch_size / latency_mean

    return FPSBenchmarkResult(
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        latency_ms_mean=latency_mean,
        latency_ms_median=latency_median,
        fps=fps,
        device=str(device),
    )