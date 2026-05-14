from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int = 42,
    deterministic: bool = False,
    benchmark: bool = True,
) -> None:

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark and not deterministic)

    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass