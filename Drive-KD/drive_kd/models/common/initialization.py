from __future__ import annotations

import torch.nn as nn


def initialize_module(
    module: nn.Module,
    conv_init: str = "kaiming_normal",
    linear_init: str = "xavier_uniform",
    norm_init: str = "ones",
) -> None:

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            _init_conv(m, conv_init)

        elif isinstance(m, nn.Linear):
            _init_linear(m, linear_init)

        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            _init_norm(m, norm_init)


def _init_conv(m: nn.Conv2d, init: str) -> None:
    init = init.lower()

    if init == "kaiming_normal":
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    elif init == "kaiming_uniform":
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")

    elif init == "xavier_uniform":
        nn.init.xavier_uniform_(m.weight)

    elif init == "xavier_normal":
        nn.init.xavier_normal_(m.weight)

    else:
        raise ValueError(f"Unsupported conv initialization: {init}")

    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _init_linear(m: nn.Linear, init: str) -> None:
    init = init.lower()

    if init == "xavier_uniform":
        nn.init.xavier_uniform_(m.weight)

    elif init == "xavier_normal":
        nn.init.xavier_normal_(m.weight)

    elif init == "kaiming_normal":
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    else:
        raise ValueError(f"Unsupported linear initialization: {init}")

    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _init_norm(m: nn.Module, init: str) -> None:
    init = init.lower()

    if hasattr(m, "weight") and m.weight is not None:
        if init == "ones":
            nn.init.ones_(m.weight)
        elif init == "zeros":
            nn.init.zeros_(m.weight)
        else:
            raise ValueError(f"Unsupported norm initialization: {init}")

    if hasattr(m, "bias") and m.bias is not None:
        nn.init.zeros_(m.bias)