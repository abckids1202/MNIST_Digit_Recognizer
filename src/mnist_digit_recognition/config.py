from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    root_dir: str
    download: bool
    validation_fraction: float
    num_workers: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str


@dataclass(frozen=True)
class ModelConfig:
    dropout: float


@dataclass(frozen=True)
class OutputConfig:
    model_dir: str
    report_dir: str
    checkpoint_name: str


@dataclass(frozen=True)
class AppConfig:
    seed: int
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    outputs: OutputConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    return AppConfig(
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        model=ModelConfig(**raw["model"]),
        outputs=OutputConfig(**raw["outputs"]),
    )
