from __future__ import annotations

import torch
import torch.nn as nn


def autopad(k: int | tuple[int, int], p: int | tuple[int, int] | None = None) -> int | tuple[int, int]:

    if p is not None:
        return p

    if isinstance(k, tuple):
        return (k[0] // 2, k[1] // 2)

    return k // 2


def get_activation(name: str | None = "silu", inplace: bool = True) -> nn.Module:

    if name is None or name.lower() in {"none", "identity", "linear"}:
        return nn.Identity()

    name = name.lower()

    if name == "silu":
        return nn.SiLU(inplace=inplace)

    if name == "relu":
        return nn.ReLU(inplace=inplace)

    if name == "gelu":
        return nn.GELU()

    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)

    raise ValueError(f"Unsupported activation: {name}")


def get_norm(name: str | None, num_channels: int) -> nn.Module:

    if name is None or name.lower() in {"none", "identity"}:
        return nn.Identity()

    name = name.lower()

    if name in {"batchnorm", "bn", "batch_norm"}:
        return nn.BatchNorm2d(num_channels)

    if name in {"groupnorm", "gn", "group_norm"}:
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    raise ValueError(f"Unsupported norm: {name}")


class ConvBNAct(nn.Module):

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
        bias: bool | None = None,
    ) -> None:
        super().__init__()

        if bias is None:
            bias = norm is None or str(norm).lower() in {"none", "identity"}

        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias=bias,
        )
        self.norm = get_norm(norm, c2)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        norm: str | None = "batchnorm",
        act: str | None = "silu",
    ) -> None:
        super().__init__()

        self.depthwise = nn.Conv2d(
            c1,
            c1,
            kernel_size=k,
            stride=s,
            padding=autopad(k),
            groups=c1,
            bias=False,
        )
        self.pointwise = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = get_norm(norm, c2)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x