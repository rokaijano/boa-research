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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
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
        dropout=args.dropout,
        channels=args.channels,
        weight_decay=args.weight_decay,
        use_fake_data=args.use_fake_data,
        train_size=args.train_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )


def run_train() -> dict[str, object]:
    args = _base_parser().parse_args()
    return train_model(_config_from_args(args))


def run_eval() -> dict[str, object]:
    parser = _base_parser()
    args = parser.parse_args()
    return evaluate_checkpoint(_config_from_args(args))
