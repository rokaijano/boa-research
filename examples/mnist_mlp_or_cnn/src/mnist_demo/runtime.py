from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from .config import TrainConfig


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_metrics(config: TrainConfig, metrics: dict[str, object]) -> None:
    reports_path = config.reports_dir / "metrics.json"
    ensure_parent(reports_path)
    reports_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(config: TrainConfig, state: dict[str, object]) -> None:
    ensure_parent(config.checkpoint_path)
    torch.save(state, config.checkpoint_path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device)


class Timer:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.started
