from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(inputs) + self.shortcut(inputs))


class SmallResNet(nn.Module):
    def __init__(self, width: int = 32, blocks_per_stage: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        widths = [width, width * 2, width * 4]
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )
        layers: list[nn.Module] = []
        in_channels = widths[0]
        for stage_index, out_channels in enumerate(widths):
            for block_index in range(blocks_per_stage):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                layers.append(ResidualBlock(in_channels, out_channels, stride=stride, dropout=dropout))
                in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.stem(inputs)))
