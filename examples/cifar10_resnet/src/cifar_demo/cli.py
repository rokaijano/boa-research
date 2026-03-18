from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainConfig
from .training import evaluate_checkpoint, train_model


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--checkpoint-path", default="artifacts/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--use-fake-data", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_dir=Path(args.data_dir),
        reports_dir=Path(args.reports_dir),
        checkpoint_path=Path(args.checkpoint_path),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        dropout=args.dropout,
        width=args.width,
        blocks_per_stage=args.blocks_per_stage,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_augmentation=not args.disable_augmentation,
        use_fake_data=args.use_fake_data,
        train_size=args.train_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


def run_train() -> dict[str, object]:
    return train_model(_config_from_args(_base_parser().parse_args()))


def run_eval() -> dict[str, object]:
    return evaluate_checkpoint(_config_from_args(_base_parser().parse_args()))
