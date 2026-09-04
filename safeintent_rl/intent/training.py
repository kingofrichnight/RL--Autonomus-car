from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from safeintent_rl.intent.model import IntentGRU


@dataclass
class IntentTrainingResult:
    best_validation_accuracy: float
    test_accuracy: float
    best_epoch: int
    split_mode: str
    train_samples: int
    validation_samples: int
    test_samples: int
    dataset_sha256: str
    checkpoint_path: Path


@dataclass(frozen=True)
class IntentDataSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    mode: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_data_split(
    sample_count: int,
    *,
    seed: int,
    episode_ids: np.ndarray | None = None,
) -> IntentDataSplit:
    """Create deterministic 70/15/15 splits, grouped by episode when possible."""
    if sample_count < 3:
        raise ValueError("At least three samples are required to create data splits")
    rng = np.random.default_rng(seed)

    if episode_ids is not None:
        groups = np.asarray(episode_ids, dtype=np.int64)
        if len(groups) != sample_count:
            raise ValueError("episode_ids must contain one ID per sample")
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 3:
            shuffled_groups = rng.permutation(unique_groups)
            train_count, validation_count = _partition_counts(len(shuffled_groups))
            train_groups = shuffled_groups[:train_count]
            validation_groups = shuffled_groups[
                train_count : train_count + validation_count
            ]
            test_groups = shuffled_groups[train_count + validation_count :]
            return IntentDataSplit(
                train_indices=np.flatnonzero(np.isin(groups, train_groups)),
                validation_indices=np.flatnonzero(np.isin(groups, validation_groups)),
                test_indices=np.flatnonzero(np.isin(groups, test_groups)),
                mode="episode",
            )

    indices = rng.permutation(sample_count)
    train_count, validation_count = _partition_counts(sample_count)
    return IntentDataSplit(
        train_indices=indices[:train_count],
        validation_indices=indices[train_count : train_count + validation_count],
        test_indices=indices[train_count + validation_count :],
        mode="sample",
    )


def _partition_counts(item_count: int) -> tuple[int, int]:
    train_count = max(1, int(0.70 * item_count))
    validation_count = max(1, int(0.15 * item_count))
    if train_count + validation_count >= item_count:
        train_count = item_count - 2
        validation_count = 1
    return train_count, validation_count


def _clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy a model state without sharing CPU tensor storage with future training steps."""
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def train_intent_model(
    data_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> IntentTrainingResult:
    with np.load(data_path) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.int64)
        episode_ids = (
            archive["episode_ids"].astype(np.int64) if "episode_ids" in archive else None
        )
    if x.ndim != 3 or x.shape[-1] != 6:
        raise ValueError("x must have shape [samples, timesteps, 6]")
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of samples")
    if len(x) < 30:
        raise ValueError("Collect at least 30 trajectory samples before training")
    if set(np.unique(y).tolist()) != {0, 1, 2}:
        raise ValueError("The dataset must contain cautious, normal, and aggressive samples")

    split = create_data_split(len(x), seed=seed, episode_ids=episode_ids)
    train_idx = split.train_indices
    val_idx = split.validation_indices
    test_idx = split.test_indices
    expected_classes = {0, 1, 2}
    for split_name, split_indices in (
        ("training", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ):
        if set(np.unique(y[split_indices]).tolist()) != expected_classes:
            raise ValueError(
                f"The {split_name} split must contain cautious, normal, and aggressive samples"
            )

    mean = x[train_idx].mean(axis=(0, 1), keepdims=True)
    std = x[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    x = (x - mean) / std

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentGRU(input_size=x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_generator = torch.Generator().manual_seed(seed)

    def loader(index: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(x[index]), torch.from_numpy(y[index]))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=train_generator if shuffle else None,
        )

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)

        val_accuracy = _accuracy(model, val_loader, device)
        print(
            f"epoch={epoch:03d} train_loss={total_loss / len(train_idx):.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = _clone_state_dict(model.state_dict())
            best_epoch = epoch

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_accuracy = _accuracy(model, loader(test_idx, False), device)
    dataset_sha256 = file_sha256(data_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "input_size": int(x.shape[-1]),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "label_names": ["cautious", "normal", "aggressive"],
            "validation_accuracy": best_accuracy,
            "test_accuracy": test_accuracy,
            "best_epoch": best_epoch,
            "training_seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "split_mode": split.mode,
            "train_indices": train_idx,
            "validation_indices": val_idx,
            "test_indices": test_idx,
            "dataset_sha256": dataset_sha256,
            "class_counts": np.bincount(y, minlength=3),
        },
        output,
    )
    return IntentTrainingResult(
        best_validation_accuracy=best_accuracy,
        test_accuracy=test_accuracy,
        best_epoch=best_epoch,
        split_mode=split.mode,
        train_samples=len(train_idx),
        validation_samples=len(val_idx),
        test_samples=len(test_idx),
        dataset_sha256=dataset_sha256,
        checkpoint_path=output,
    )


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        predictions = model(inputs.to(device)).argmax(dim=1).cpu()
        correct += int((predictions == targets).sum())
        total += len(targets)
    return correct / max(total, 1)
