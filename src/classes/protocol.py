from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from typing import Self

from classes.training import TrainParameters


class PushForwardOperator(ABC):
    @abstractmethod
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes `u` to response space given condition `x`."""

    @abstractmethod
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes response `y` to latent space given condition `x`."""

    @abstractmethod
    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        *args,
        **kwargs,
    ) -> "PushForwardOperator":
        """Fits the operator."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the operator."""

    @abstractmethod
    def load(
        self,
        path: str,
        map_location: torch.device = torch.device("cpu"),
    ) -> Self:
        """Loads the operator state from disk."""

    @classmethod
    @abstractmethod
    def load_class(
        cls,
        path: str,
        map_location: torch.device = torch.device("cpu"),
    ) -> Self:
        """Builds and loads the operator from disk."""
