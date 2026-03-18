from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    checkpoint_path: Path = Path("artifacts/best_model.pt")
    batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    epochs: int = 8
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    dropout: float = 0.1
    width: int = 32
    blocks_per_stage: int = 2
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    use_augmentation: bool = True
    use_fake_data: bool = False
    train_size: int = 4096
    val_size: int = 1024
    num_workers: int = 0
    device: str = "auto"
    seed: int = 11

    def serializable(self) -> dict[str, object]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
