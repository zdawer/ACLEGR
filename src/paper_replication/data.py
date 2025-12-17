from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from torch.utils.data import Dataset


@dataclass
class DataConfig:
    """Describe dataset sources and preprocessing."""

    root: Path | str
    splits: dict[str, Iterable[str]] = field(default_factory=dict)
    transform: Callable | None = None
    target_transform: Callable | None = None
    download: bool = False
    extra_parameters: dict[str, object] = field(default_factory=dict)


class PaperDataset(Dataset):
    """Stub dataset; fill in loading, preprocessing, and target construction."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        # TODO: load files, apply preprocessing, and build label/target fields.
        raise NotImplementedError("Implement dataset loading and preprocessing from the paper.")

    def __len__(self) -> int:
        # TODO: return dataset size after loading.
        raise NotImplementedError("Return the dataset size.")

    def __getitem__(self, idx: int):
        # TODO: return a single sample and its target, matching the paper's formatting.
        raise NotImplementedError("Return a processed sample and target.")
