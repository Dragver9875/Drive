from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFusion(nn.Module):

    def __init__(
        self,
        n_inputs: int,
        eps: float = 1e-4,
        init_value: float = 1.0,
    ) -> None:
        super().__init__()

        if n_inputs < 2:
            raise ValueError("WeightedFusion requires at least two input tensors.")

        self.n_inputs = int(n_inputs)
        self.eps = float(eps)

        self.weights = nn.Parameter(
            torch.full((self.n_inputs,), float(init_value), dtype=torch.float32)
        )

    def normalized_weights(self) -> torch.Tensor:
        weights = F.relu(self.weights)
        return weights / (weights.sum() + self.eps)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} inputs, got {len(inputs)}.")

        ref_shape = inputs[0].shape

        for i, x in enumerate(inputs):
            if x.shape != ref_shape:
                raise ValueError(
                    f"WeightedFusion input shape mismatch at index {i}: "
                    f"expected {ref_shape}, got {x.shape}"
                )

        weights = self.normalized_weights()

        out = torch.zeros_like(inputs[0])

        for w, x in zip(weights, inputs):
            out = out + w * x

        return out