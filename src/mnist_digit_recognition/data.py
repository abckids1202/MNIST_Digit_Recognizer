from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import AppConfig


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def build_train_val_loaders(config: AppConfig) -> tuple[DataLoader, DataLoader]:
    dataset = datasets.MNIST(
        root=Path(config.data.root_dir),
        train=True,
        download=config.data.download,
        transform=mnist_transform(),
    )

    val_size = int(len(dataset) * config.data.validation_fraction)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    return train_loader, val_loader


def build_test_loader(config: AppConfig) -> DataLoader:
    dataset = datasets.MNIST(
        root=Path(config.data.root_dir),
        train=False,
        download=config.data.download,
        transform=mnist_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
