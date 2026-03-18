from __future__ import annotations

import torch
from torch import nn

from .config import TrainConfig
from .data import build_dataloaders
from .model import TinyTransformerClassifier
from .runtime import (
    Timer,
    choose_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_metrics,
)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def _evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            total_loss += float(criterion(logits, labels).item())
            total_accuracy += _accuracy(logits, labels)
            total_batches += 1
    return {
        "loss": total_loss / max(1, total_batches),
        "accuracy": total_accuracy / max(1, total_batches),
    }


def train_model(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    bundle = build_dataloaders(config)
    device = choose_device(config.device)
    model = TinyTransformerClassifier(
        vocab_size=len(bundle.vocab),
        num_classes=bundle.num_classes,
        max_length=config.max_length,
        embedding_dim=config.embedding_dim,
        ff_dim=config.ff_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(bundle.train_loader))
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    criterion = nn.CrossEntropyLoss()
    best_metrics = {"accuracy": 0.0, "loss": float("inf")}
    history: list[dict[str, float]] = []
    step = 0
    with Timer() as timer:
        for epoch in range(config.epochs):
            model.train()
            for batch in bundle.train_loader:
                step += 1
                scale = min(1.0, step / warmup_steps)
                optimizer.param_groups[0]["lr"] = config.learning_rate * scale
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            metrics = _evaluate(model, bundle.val_loader, device)
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
                        "vocab": bundle.vocab,
                        "num_classes": bundle.num_classes,
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
    checkpoint = load_checkpoint(config.checkpoint_path, device)
    bundle = build_dataloaders(config)
    saved = checkpoint["config"]
    model = TinyTransformerClassifier(
        vocab_size=len(checkpoint["vocab"]),
        num_classes=int(checkpoint["num_classes"]),
        max_length=int(saved["max_length"]),
        embedding_dim=int(saved["embedding_dim"]),
        ff_dim=int(saved["ff_dim"]),
        num_heads=int(saved["num_heads"]),
        num_layers=int(saved["num_layers"]),
        dropout=float(saved["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    with Timer() as timer:
        metrics = _evaluate(model, bundle.val_loader, device)
    payload = {
        "accuracy": round(float(metrics["accuracy"]), 6),
        "loss": round(float(metrics["loss"]), 6),
        "runtime_seconds": round(float(timer.elapsed), 4),
        "device": str(device),
        "checkpoint_path": str(config.checkpoint_path),
    }
    write_metrics(config, payload)
    return payload
