from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safeintent_rl.intent.model import IntentGRU

EXPECTED_LABEL_NAMES = ["cautious", "normal", "aggressive"]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_intent_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load the project's checkpoint with only its required NumPy types allowlisted."""
    numpy_safe_globals = [
        np._core.multiarray._reconstruct,
        np._core.multiarray.scalar,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(numpy_safe_globals):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("Intent checkpoint must contain a dictionary")

    required = {"input_size", "model_state", "mean", "std", "label_names"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Intent checkpoint is missing fields: {sorted(missing)}")

    input_size = int(checkpoint["input_size"])
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    if input_size != 6:
        raise ValueError(f"Intent checkpoint input size must be 6, got {input_size}")
    if mean.shape != (1, 1, input_size) or std.shape != mean.shape:
        raise ValueError("Intent checkpoint normalization arrays have invalid shapes")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Intent checkpoint normalization values must be finite and positive")
    if list(checkpoint["label_names"]) != EXPECTED_LABEL_NAMES:
        raise ValueError("Intent checkpoint label order does not match the project classes")
    if not isinstance(checkpoint["model_state"], dict):
        raise ValueError("Intent checkpoint model_state must be a dictionary")
    return checkpoint


class IntentPredictor:
    """Load a trained GRU and return softmax behavior probabilities."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        if expected_sha256 is not None:
            actual_sha256 = file_sha256(checkpoint_path)
            if actual_sha256.lower() != expected_sha256.lower():
                raise ValueError(
                    "Intent checkpoint fingerprint does not match --intent-model-sha256"
                )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = load_intent_checkpoint(checkpoint_path, map_location=self.device)
        self.model = IntentGRU(input_size=int(checkpoint["input_size"])).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["std"], dtype=np.float32)
        self.label_names = list(checkpoint["label_names"])

    @torch.no_grad()
    def predict_proba(self, histories: np.ndarray) -> np.ndarray:
        array = np.asarray(histories, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        normalized = (array - self.mean) / self.std
        logits = self.model(torch.from_numpy(normalized).to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()
