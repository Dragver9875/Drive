from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseTeacher(nn.Module, abc.ABC):

    def __init__(self, name: str = "base_teacher") -> None:
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool | None = None,
        return_attention: bool | None = None,
        return_probabilities: bool | None = None,
        as_dataclass: bool = False,
    ):
        raise NotImplementedError

    def freeze_encoder(self) -> None:

        if not hasattr(self, "encoder"):
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'encoder'.")

        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:

        if not hasattr(self, "encoder"):
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'encoder'.")

        for p in self.encoder.parameters():
            p.requires_grad = True

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())