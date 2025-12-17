from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .data import DataConfig, PaperDataset
from .model import ModelConfig, PaperModel


@dataclass
class TrainingConfig:
    """Holds optimization details that should mirror the paper."""

    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    grad_clip: float | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler_factory: Callable[[optim.Optimizer], optim.lr_scheduler._LRScheduler] | None = None
    extra_parameters: dict[str, object] = field(default_factory=dict)


def create_dataloader(config: DataConfig, batch_size: int, shuffle: bool) -> DataLoader:
    """Instantiate a dataloader matching the paper's batching strategy."""
    dataset = PaperDataset(config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    criterion: nn.Module | None = None,
) -> None:
    """
    Skeleton training loop. Fill in the details to match the paper exactly.

    - Define the loss function (criterion) used in the paper.
    - Choose the optimizer and learning-rate scheduler described in the paper.
    - Implement evaluation and logging as reported by the authors.
    """

    device = training_config.device
    model = PaperModel(model_config).to(device)

    if criterion is None:
        # TODO: replace with the loss function from the paper.
        raise NotImplementedError("Specify the loss function used in the paper.")

    # TODO: replace Adam with the optimizer from the paper.
    optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate or 1e-3)

    if training_config.scheduler_factory:
        scheduler = training_config.scheduler_factory(optimizer)
    else:
        scheduler = None

    train_loader = create_dataloader(data_config, training_config.batch_size or 1, shuffle=True)

    for epoch in range(training_config.epochs or 1):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            if training_config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
            optimizer.step()

        if scheduler:
            scheduler.step()

        # TODO: add validation, logging, checkpointing, and metric computation per the paper.
