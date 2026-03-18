from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig


PAD_ID = 0
UNK_ID = 1


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().replace("\n", " ").split(" ") if token]


class TextClassificationDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = tokenize(self.texts[index])[: self.max_length]
        ids = [self.vocab.get(token, UNK_ID) for token in tokens]
        padded = ids + [PAD_ID] * max(0, self.max_length - len(ids))
        attention = [1] * min(len(ids), self.max_length) + [0] * max(0, self.max_length - len(ids))
        return {
            "input_ids": torch.tensor(padded[: self.max_length], dtype=torch.long),
            "attention_mask": torch.tensor(attention[: self.max_length], dtype=torch.float32),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    vocab: dict[str, int]
    num_classes: int


def _fake_samples(train_size: int, val_size: int):
    positive = "team wins championship with dramatic comeback"
    negative = "company reports lower revenue after weak quarter"
    science = "researchers discover new particle in collider study"
    world = "leaders discuss sanctions and diplomatic response"
    texts = [positive, negative, science, world]
    labels = [0, 1, 2, 3]
    train_texts = [texts[index % 4] for index in range(train_size)]
    train_labels = [labels[index % 4] for index in range(train_size)]
    val_texts = [texts[index % 4] for index in range(val_size)]
    val_labels = [labels[index % 4] for index in range(val_size)]
    return train_texts, train_labels, val_texts, val_labels


def _ag_news_samples(config: TrainConfig):
    dataset = load_dataset("ag_news", cache_dir=str(config.data_dir))
    train_split = dataset["train"]
    test_split = dataset["test"]
    train_size = min(config.train_size, len(train_split))
    val_size = min(config.val_size, len(test_split))
    train_texts = [train_split[index]["text"] for index in range(train_size)]
    train_labels = [int(train_split[index]["label"]) for index in range(train_size)]
    val_texts = [test_split[index]["text"] for index in range(val_size)]
    val_labels = [int(test_split[index]["label"]) for index in range(val_size)]
    return train_texts, train_labels, val_texts, val_labels


def build_vocab(texts: list[str], config: TrainConfig) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for token, count in counter.most_common():
        if count < config.min_token_frequency:
            continue
        if len(vocab) >= config.vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def build_dataloaders(config: TrainConfig) -> DataBundle:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    if config.use_fake_data:
        train_texts, train_labels, val_texts, val_labels = _fake_samples(config.train_size, config.val_size)
    else:
        train_texts, train_labels, val_texts, val_labels = _ag_news_samples(config)
    vocab = build_vocab(train_texts, config)
    train_dataset = TextClassificationDataset(train_texts, train_labels, vocab, config.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, vocab, config.max_length)
    return DataBundle(
        train_loader=DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False),
        vocab=vocab,
        num_classes=len(set(train_labels)),
    )
