from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import TrainConfig


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader


def _subset(dataset, size: int):
    return Subset(dataset, list(range(min(size, len(dataset)))))


def build_dataloaders(config: TrainConfig) -> DataBundle:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    if config.use_fake_data:
        train_dataset = datasets.FakeData(
            size=config.train_size,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
        val_dataset = datasets.FakeData(
            size=config.val_size,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
        )
    else:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)
        train_dataset = _subset(train_dataset, config.train_size)
        val_dataset = _subset(val_dataset, config.val_size)
    return DataBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        ),
        val_loader=DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    )
