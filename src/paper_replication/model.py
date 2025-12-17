from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Holds architecture-specific parameters to be filled from the paper."""

    input_dim: int | None = None
    output_dim: int | None = None
    hidden_dims: Iterable[int] = field(default_factory=list)
    dropout: float | None = None
    extra_hyperparameters: dict[str, Any] = field(default_factory=dict)


class PaperModel(nn.Module):
    """Stub model; implement layers and forward pass based on the paper."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: Define layers precisely as described in the paper (e.g., CNN/Transformer blocks).
        raise NotImplementedError("Implement the model architecture from the paper.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass mirroring the paper's computational graph."""
        # TODO: Replace with the real forward pass.
        raise NotImplementedError("Implement the forward pass from the paper.")
