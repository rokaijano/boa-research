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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--min-token-frequency", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=19)
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
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        min_token_frequency=args.min_token_frequency,
        use_fake_data=args.use_fake_data,
        train_size=args.train_size,
        val_size=args.val_size,
        device=args.device,
        seed=args.seed,
    )


def run_train() -> dict[str, object]:
    return train_model(_config_from_args(_base_parser().parse_args()))


def run_eval() -> dict[str, object]:
    return evaluate_checkpoint(_config_from_args(_base_parser().parse_args()))
