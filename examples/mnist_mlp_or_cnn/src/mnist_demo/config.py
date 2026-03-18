from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    checkpoint_path: Path = Path("artifacts/best_model.pt")
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 3
    optimizer: str = "adam"
    dropout: float = 0.15
    channels: int = 32
    weight_decay: float = 0.0
    use_fake_data: bool = False
    train_size: int = 2048
    val_size: int = 512
    num_workers: int = 0
    device: str = "auto"
    seed: int = 7

    def serializable(self) -> dict[str, object]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}

