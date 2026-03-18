from __future__ import annotations

from dataclasses import dataclass

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
    train_ops = []
    if config.use_augmentation:
        train_ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    train_transform = transforms.Compose(train_ops + [transforms.ToTensor(), normalization])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalization])
    if config.use_fake_data:
        train_dataset = datasets.FakeData(
            size=config.train_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
        )
        val_dataset = datasets.FakeData(
            size=config.val_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_transform,
        )
    else:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=eval_transform)
        train_dataset = _subset(train_dataset, config.train_size)
        val_dataset = _subset(val_dataset, config.val_size)
    return DataBundle(
        train_loader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers),
        val_loader=DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers),
    )
