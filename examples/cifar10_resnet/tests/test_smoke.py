from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fake_data_train_and_eval(tmp_path):
    train_cmd = [
        sys.executable,
        "train.py",
        "--use-fake-data",
        "--train-size",
        "64",
        "--val-size",
        "32",
        "--epochs",
        "1",
        "--batch-size",
        "16",
        "--eval-batch-size",
        "16",
        "--data-dir",
        str(tmp_path / "data"),
        "--reports-dir",
        str(tmp_path / "reports"),
        "--checkpoint-path",
        str(tmp_path / "artifacts" / "best_model.pt"),
    ]
    subprocess.run(train_cmd, cwd=ROOT, check=True)
    eval_cmd = [
        sys.executable,
        "eval.py",
        "--use-fake-data",
        "--val-size",
        "32",
        "--eval-batch-size",
        "16",
        "--data-dir",
        str(tmp_path / "data"),
        "--reports-dir",
        str(tmp_path / "reports"),
        "--checkpoint-path",
        str(tmp_path / "artifacts" / "best_model.pt"),
    ]
    subprocess.run(eval_cmd, cwd=ROOT, check=True)
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert "accuracy" in metrics
    assert "loss" in metrics
