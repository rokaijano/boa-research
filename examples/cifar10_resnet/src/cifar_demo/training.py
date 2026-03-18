from __future__ import annotations

import torch
from torch import nn

from .config import TrainConfig
from .data import build_dataloaders
from .model import SmallResNet
from .runtime import (
    Timer,
    accuracy_from_logits,
    build_optimizer,
    build_scheduler,
    choose_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_metrics,
)


def _evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += float(criterion(logits, labels).item())
            total_accuracy += accuracy_from_logits(logits, labels)
            total_batches += 1
    return {
        "loss": total_loss / max(1, total_batches),
        "accuracy": total_accuracy / max(1, total_batches),
    }


def train_model(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    loaders = build_dataloaders(config)
    device = choose_device(config.device)
    model = SmallResNet(width=config.width, blocks_per_stage=config.blocks_per_stage, dropout=config.dropout).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, config.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    best_metrics = {"accuracy": 0.0, "loss": float("inf")}
    history: list[dict[str, float]] = []
    with Timer() as timer:
        for epoch in range(config.epochs):
            model.train()
            for inputs, labels in loaders.train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            metrics = _evaluate(model, loaders.val_loader, device)
            metrics["epoch"] = epoch + 1
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            history.append(metrics)
            if metrics["accuracy"] >= best_metrics["accuracy"]:
                best_metrics = {"accuracy": metrics["accuracy"], "loss": metrics["loss"]}
                save_checkpoint(
                    config,
                    {
                        "model_state": model.state_dict(),
                        "config": config.serializable(),
                        "metrics": best_metrics,
                    },
                )
    payload = {
        "accuracy": round(float(best_metrics["accuracy"]), 6),
        "loss": round(float(best_metrics["loss"]), 6),
        "runtime_seconds": round(float(timer.elapsed), 4),
        "device": str(device),
        "history": history,
        "checkpoint_path": str(config.checkpoint_path),
    }
    write_metrics(config, payload)
    return payload


def evaluate_checkpoint(config: TrainConfig) -> dict[str, object]:
    device = choose_device(config.device)
    loaders = build_dataloaders(config)
    checkpoint = load_checkpoint(config.checkpoint_path, device)
    saved = checkpoint["config"]
    model = SmallResNet(
        width=int(saved["width"]),
        blocks_per_stage=int(saved["blocks_per_stage"]),
        dropout=float(saved["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    with Timer() as timer:
        metrics = _evaluate(model, loaders.val_loader, device)
    payload = {
        "accuracy": round(float(metrics["accuracy"]), 6),
        "loss": round(float(metrics["loss"]), 6),
        "runtime_seconds": round(float(timer.elapsed), 4),
        "device": str(device),
        "checkpoint_path": str(config.checkpoint_path),
    }
    write_metrics(config, payload)
    return payload
