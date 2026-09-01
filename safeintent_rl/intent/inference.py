from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from safeintent_rl.intent.model import IntentGRU


class IntentPredictor:
    """Load a trained GRU and return calibrated-style softmax behavior probabilities."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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

