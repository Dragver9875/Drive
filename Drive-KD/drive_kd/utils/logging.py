from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


def setup_logger(
    name: str = "drive_kd",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        existing_files = [
            getattr(h, "baseFilename", None)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]

        if str(log_file) not in existing_files:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class AverageMeter:

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        value = float(value)
        n = int(n)

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

    def as_dict(self) -> dict[str, float]:
        return {
            f"{self.name}/val": float(self.val),
            f"{self.name}/avg": float(self.avg),
        }


class MetricLogger:

    def __init__(self) -> None:
        self.meters: OrderedDict[str, AverageMeter] = OrderedDict()

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter(key)
            self.meters[key].update(float(value))

    def update_dict(self, data: dict[str, Any]) -> None:
        numeric = {}

        for key, value in data.items():
            if isinstance(value, int | float):
                numeric[key] = float(value)

        self.update(**numeric)

    def averages(self) -> dict[str, float]:
        return {key: meter.avg for key, meter in self.meters.items()}

    def latest(self) -> dict[str, float]:
        return {key: meter.val for key, meter in self.meters.items()}

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def format(self, precision: int = 4) -> str:
        parts = []

        for key, meter in self.meters.items():
            parts.append(f"{key}: {meter.avg:.{precision}f}")

        return " | ".join(parts)