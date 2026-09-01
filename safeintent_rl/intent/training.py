from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from safeintent_rl.intent.model import IntentGRU


@dataclass
class IntentTrainingResult:
    best_validation_accuracy: float
    test_accuracy: float
    checkpoint_path: Path


def train_intent_model(
    data_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> IntentTrainingResult:
    archive = np.load(data_path)
    x = archive["x"].astype(np.float32)
    y = archive["y"].astype(np.int64)
    if len(x) < 30:
        raise ValueError("Collect at least 30 trajectory samples before training")

    rng = np.random.default_rng(seed)
    if "episode_ids" in archive and len(np.unique(archive["episode_ids"])) >= 3:
        groups = archive["episode_ids"].astype(np.int64)
        unique_groups = rng.permutation(np.unique(groups))
        train_group_end = max(1, int(0.70 * len(unique_groups)))
        validation_group_end = max(train_group_end + 1, int(0.85 * len(unique_groups)))
        train_groups = unique_groups[:train_group_end]
        validation_groups = unique_groups[train_group_end:validation_group_end]
        test_groups = unique_groups[validation_group_end:]
        if len(test_groups) == 0:
            test_groups = validation_groups[-1:]
            validation_groups = validation_groups[:-1]
        train_idx = np.flatnonzero(np.isin(groups, train_groups))
        val_idx = np.flatnonzero(np.isin(groups, validation_groups))
        test_idx = np.flatnonzero(np.isin(groups, test_groups))
    else:
        indices = rng.permutation(len(x))
        train_end = int(0.70 * len(x))
        validation_end = int(0.85 * len(x))
        train_idx, val_idx, test_idx = (
            indices[:train_end],
            indices[train_end:validation_end],
            indices[validation_end:],
        )

    mean = x[train_idx].mean(axis=(0, 1), keepdims=True)
    std = x[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    x = (x - mean) / std

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntentGRU(input_size=x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def loader(index: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(x[index]), torch.from_numpy(y[index]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

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
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_accuracy = _accuracy(model, loader(test_idx, False), device)

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
            "test_indices": test_idx,
        },
        output,
    )
    return IntentTrainingResult(best_accuracy, test_accuracy, output)


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
