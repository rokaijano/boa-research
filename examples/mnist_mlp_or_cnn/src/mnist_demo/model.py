from __future__ import annotations

import torch
from torch import nn


class MnistConvNet(nn.Module):
    def __init__(self, channels: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        hidden = max(16, channels)
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((hidden * 2) * 7 * 7, hidden * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))
