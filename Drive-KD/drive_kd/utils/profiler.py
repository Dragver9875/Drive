from __future__ import annotations

import time
from dataclasses import dataclass


def format_seconds(seconds: float) -> str:

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes = seconds / 60.0

    if minutes < 60:
        return f"{minutes:.2f}min"

    hours = minutes / 60.0

    return f"{hours:.2f}h"


@dataclass
class Timer:

    start_time: float | None = None
    end_time: float | None = None

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        return self.elapsed()

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time

    def elapsed_str(self) -> str:
        return format_seconds(self.elapsed())

    def reset(self) -> None:
        self.start_time = None
        self.end_time = None

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()