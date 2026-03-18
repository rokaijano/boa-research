from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    checkpoint_path: Path = Path("artifacts/best_model.pt")
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    epochs: int = 5
    max_length: int = 64
    embedding_dim: int = 128
    ff_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.15
    vocab_size: int = 12000
    min_token_frequency: int = 2
    use_fake_data: bool = False
    train_size: int = 2000
    val_size: int = 500
    device: str = "auto"
    seed: int = 19

    def serializable(self) -> dict[str, object]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
