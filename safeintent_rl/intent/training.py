from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from safeintent_rl.intent.evaluation import classification_metrics
from safeintent_rl.intent.model import IntentGRU


@dataclass
class IntentTrainingResult:
    best_validation_accuracy: float
    test_accuracy: float | None
    best_epoch: int
    split_mode: str
    train_samples: int
    validation_samples: int
    test_samples: int
    dataset_sha256: str
    checkpoint_sha256: str
    history_length: int
    source_history_length: int
    device: str
    checkpoint_path: Path
    summary_path: Path | None
    validation_metrics: dict[str, Any]
    validation_gate_passed: bool


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


def project_histories(x: np.ndarray, history_length: int) -> np.ndarray:
    """Return the causal suffix used by a shorter-history classifier."""
    histories = np.asarray(x)
    if histories.ndim != 3 or histories.shape[-1] != 6:
        raise ValueError("x must have shape [samples, timesteps, 6]")
    if not 1 <= history_length <= histories.shape[1]:
        raise ValueError(
            "history_length must be between 1 and the source history length "
            f"({histories.shape[1]})"
        )
    return histories[:, -history_length:, :]


def _index_sha256(indices: np.ndarray) -> str:
    normalized = np.asarray(indices, dtype=np.dtype("<i8"))
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _resolve_device(requested: str | None) -> torch.device:
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def train_intent_model(
    data_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    history_length: int | None = None,
    evaluate_test: bool = True,
    expected_data_sha256: str | None = None,
    summary_output_path: str | Path | None = None,
    device: str | None = None,
    refuse_overwrite: bool = False,
    minimum_validation_accuracy_advantage: float = 0.05,
    minimum_validation_macro_f1: float = 0.50,
    minimum_validation_class_recall: float = 0.40,
) -> IntentTrainingResult:
    data_path = Path(data_path)
    output = Path(output_path)
    summary_output = Path(summary_output_path) if summary_output_path is not None else None
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    for name, threshold in (
        ("minimum_validation_accuracy_advantage", minimum_validation_accuracy_advantage),
        ("minimum_validation_macro_f1", minimum_validation_macro_f1),
        ("minimum_validation_class_recall", minimum_validation_class_recall),
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(f"{name} must be between zero and one")
    if refuse_overwrite and output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    if refuse_overwrite and summary_output is not None and summary_output.exists():
        raise FileExistsError(f"Refusing to overwrite training summary: {summary_output}")

    dataset_sha256 = file_sha256(data_path)
    if (
        expected_data_sha256 is not None
        and dataset_sha256.lower() != expected_data_sha256.lower()
    ):
        raise ValueError("Dataset fingerprint does not match --data-sha256")

    with np.load(data_path) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.int64)
        episode_ids = (
            archive["episode_ids"].astype(np.int64) if "episode_ids" in archive else None
        )
    if x.ndim != 3 or x.shape[-1] != 6:
        raise ValueError("x must have shape [samples, timesteps, 6]")
    source_history_length = int(x.shape[1])
    selected_history_length = (
        source_history_length if history_length is None else int(history_length)
    )
    x = project_histories(x, selected_history_length)
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
    checked_splits = [("training", train_idx), ("validation", val_idx)]
    if evaluate_test:
        checked_splits.append(("test", test_idx))
    for split_name, split_indices in checked_splits:
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
    training_device = _resolve_device(device)
    model = IntentGRU(input_size=x.shape[-1]).to(training_device)
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
            inputs, targets = inputs.to(training_device), targets.to(training_device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)

        val_accuracy = _accuracy(model, val_loader, training_device)
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
    validation_targets, validation_predictions = _predictions(
        model, val_loader, training_device
    )
    validation_metrics = classification_metrics(
        validation_targets,
        validation_predictions,
        ["cautious", "normal", "aggressive"],
    )
    if not np.isclose(validation_metrics["accuracy"], best_accuracy):
        raise RuntimeError("Frozen checkpoint does not reproduce its validation accuracy")
    validation_accuracy_advantage = float(
        validation_metrics["accuracy"] - validation_metrics["majority_class_accuracy"]
    )
    minimum_observed_recall = min(
        float(result["recall"])
        for result in validation_metrics["per_class"].values()
    )
    validation_gate_passed = bool(
        validation_accuracy_advantage >= minimum_validation_accuracy_advantage
        and validation_metrics["macro_f1"] >= minimum_validation_macro_f1
        and minimum_observed_recall >= minimum_validation_class_recall
    )
    validation_gate = {
        "requirements": {
            "minimum_accuracy_advantage": minimum_validation_accuracy_advantage,
            "minimum_macro_f1": minimum_validation_macro_f1,
            "minimum_class_recall": minimum_validation_class_recall,
        },
        "observed": {
            "accuracy_advantage": validation_accuracy_advantage,
            "macro_f1": float(validation_metrics["macro_f1"]),
            "minimum_class_recall": minimum_observed_recall,
        },
        "passed": validation_gate_passed,
    }
    test_accuracy = (
        _accuracy(model, loader(test_idx, False), training_device)
        if evaluate_test
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "input_size": int(x.shape[-1]),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "label_names": ["cautious", "normal", "aggressive"],
        "validation_accuracy": best_accuracy,
        "validation_metrics": validation_metrics,
        "validation_gate": validation_gate,
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
        "history_length": selected_history_length,
        "source_history_length": source_history_length,
        "history_projection": "causal_suffix",
        "test_metrics_sealed": not evaluate_test,
        "training_device": str(training_device),
    }
    if test_accuracy is not None:
        checkpoint["test_accuracy"] = test_accuracy
    torch.save(checkpoint, output)
    checkpoint_sha256 = file_sha256(output)

    if summary_output is not None:
        split_indices = {
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx,
        }
        split_episode_ids = (
            {
                name: np.unique(episode_ids[index]).astype(int).tolist()
                for name, index in split_indices.items()
            }
            if episode_ids is not None
            else None
        )
        summary = {
            "status": "complete",
            "data_path": str(data_path),
            "dataset_sha256": dataset_sha256,
            "checkpoint_path": str(output),
            "checkpoint_sha256": checkpoint_sha256,
            "source_samples": int(len(x)),
            "source_history_length": source_history_length,
            "history_length": selected_history_length,
            "history_projection": "causal_suffix",
            "feature_count": int(x.shape[-1]),
            "label_names": ["cautious", "normal", "aggressive"],
            "class_counts": np.bincount(y, minlength=3).astype(int).tolist(),
            "training_seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_device": str(training_device),
            "split_mode": split.mode,
            "split_samples": {
                name: int(len(index)) for name, index in split_indices.items()
            },
            "split_indices_sha256": {
                name: _index_sha256(index) for name, index in split_indices.items()
            },
            "split_episode_ids": split_episode_ids,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "validation_metrics": validation_metrics,
            "validation_gate": validation_gate,
            "test_metrics_sealed": not evaluate_test,
            "test_accuracy": test_accuracy,
        }
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return IntentTrainingResult(
        best_validation_accuracy=best_accuracy,
        test_accuracy=test_accuracy,
        best_epoch=best_epoch,
        split_mode=split.mode,
        train_samples=len(train_idx),
        validation_samples=len(val_idx),
        test_samples=len(test_idx),
        dataset_sha256=dataset_sha256,
        checkpoint_sha256=checkpoint_sha256,
        history_length=selected_history_length,
        source_history_length=source_history_length,
        device=str(training_device),
        checkpoint_path=output,
        summary_path=summary_output,
        validation_metrics=validation_metrics,
        validation_gate_passed=validation_gate_passed,
    )


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    targets, predictions = _predictions(model, loader, device)
    return float((predictions == targets).mean())


@torch.no_grad()
def _predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    target_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    for inputs, target_batch in loader:
        prediction_batches.append(model(inputs.to(device)).argmax(dim=1).cpu().numpy())
        target_batches.append(target_batch.numpy())
    if not target_batches:
        raise ValueError("Cannot evaluate an empty data loader")
    return np.concatenate(target_batches), np.concatenate(prediction_batches)
