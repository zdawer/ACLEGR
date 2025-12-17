from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import warnings

import torch
from torch import Tensor, nn


def _activation_factory(name: str) -> nn.Module:
    """Return an activation module matching the requested name."""
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation '{name}'.")


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding that matches Transformer expectations."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, sequence_length, dim)

        Returns:
            Tensor with positional information added, matching input shape.
        """
        batch, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {dim}.")
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=x.device) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, dim)
        x = x + pe
        return self.dropout(x)


@dataclass
class ModelConfig:
    """Hyperparameters mirroring the paper architecture."""

    input_channels: int = 3
    cnn_channels: Sequence[int] = (32, 64)
    cnn_kernel_size: int = 3
    cnn_stride: int = 1
    cnn_pool: int = 2
    embedding_dim: int = 128
    num_heads: int = 4
    depth: int = 4
    transformer_mlp_dim: int = 256
    mlp_hidden_dim: int = 256
    num_classes: int = 10
    dropout: float = 0.1
    activation: str = "gelu"


class ReplicatedModel(nn.Module):
    """
    Hybrid CNN + Transformer encoder followed by an MLP classifier.

    The forward pass follows the paper's computational graph:
    1) Convolutional feature extractor.
    2) Flatten to a token sequence and project to embeddings.
    3) Add positional encodings and run Transformer encoder blocks.
    4) Sequence pooling (mean) and classification MLP head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        activation_name = config.activation

        cnn_layers: list[nn.Module] = []
        in_channels = config.input_channels
        for out_channels in config.cnn_channels:
            cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=config.cnn_kernel_size,
                        stride=config.cnn_stride,
                        padding=config.cnn_kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    _activation_factory(activation_name),
                    nn.Dropout2d(config.dropout),
                    nn.MaxPool2d(kernel_size=config.cnn_pool) if config.cnn_pool > 1 else nn.Identity(),
                )
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)

        self.projection = nn.Linear(config.cnn_channels[-1], config.embedding_dim)
        self.position = SinusoidalPositionalEncoding(config.embedding_dim, dropout=config.dropout)

        normalized_activation = activation_name.lower()
        if normalized_activation not in {"relu", "gelu"}:
            warnings.warn(
                "Transformer encoder only supports 'relu' or 'gelu'; "
                f"falling back to 'gelu' for activation '{activation_name}'."
            )
            normalized_activation = "gelu"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.transformer_mlp_dim,
            dropout=config.dropout,
            activation=normalized_activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)

        self.norm = nn.LayerNorm(config.embedding_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.mlp_hidden_dim),
            _activation_factory(activation_name),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input with shape (batch, channels, height, width); got {tuple(x.shape)}.")
        batch, channels, height, width = x.shape
        if channels != self.config.input_channels:
            raise ValueError(f"Expected {self.config.input_channels} channels but received {channels}.")

        features = self.cnn(x)
        _, conv_channels, conv_height, conv_width = features.shape
        if conv_height == 0 or conv_width == 0:
            raise ValueError("Convolutional backbone produced empty spatial dimensions.")

        tokens = features.flatten(2).transpose(1, 2)  # (batch, seq_len, conv_channels)
        tokens = self.projection(tokens)  # (batch, seq_len, embedding_dim)
        tokens = self.position(tokens)

        encoded = self.transformer(tokens)  # (batch, seq_len, embedding_dim)
        encoded = self.norm(encoded)
        pooled = encoded.mean(dim=1)
        logits = self.mlp_head(pooled)
        if logits.shape != (batch, self.config.num_classes):
            raise AssertionError(
                f"Unexpected logits shape {tuple(logits.shape)}; expected {(batch, self.config.num_classes)}."
            )
        return logits


__all__: Iterable[str] = ["ModelConfig", "ReplicatedModel"]
