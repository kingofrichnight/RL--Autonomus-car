from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from safeintent_rl.intent.evaluation import classification_metrics
from safeintent_rl.intent.inference import IntentPredictor
from safeintent_rl.intent.training import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the intent GRU on a labeled archive")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/intent_gru_seed42.metrics.json")
    args = parser.parse_args()

    with np.load(args.data) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.int64)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    if "test_indices" not in checkpoint:
        raise ValueError("Checkpoint does not contain a held-out test split")
    expected_hash = checkpoint.get("dataset_sha256")
    if not expected_hash:
        raise ValueError("Checkpoint does not contain a dataset fingerprint")
    actual_hash = file_sha256(args.data)
    if actual_hash != expected_hash:
        raise ValueError("Dataset fingerprint does not match the training checkpoint")

    test_indices = np.asarray(checkpoint["test_indices"], dtype=np.int64)
    if len(test_indices) == 0 or np.any((test_indices < 0) | (test_indices >= len(x))):
        raise ValueError("Checkpoint contains invalid test indices")
    if len(np.unique(test_indices)) != len(test_indices):
        raise ValueError("Checkpoint test indices contain duplicates")

    test_x = x[test_indices]
    test_y = y[test_indices]
    predictor = IntentPredictor(args.model)
    predictions = predictor.predict_proba(test_x).argmax(axis=1)
    metrics = classification_metrics(test_y, predictions, predictor.label_names)
    checkpoint_accuracy = float(checkpoint["test_accuracy"])
    if not np.isclose(metrics["accuracy"], checkpoint_accuracy):
        raise RuntimeError("Recomputed test accuracy does not match the checkpoint")

    metrics.update(
        {
            "data_path": str(Path(args.data)),
            "model_path": str(Path(args.model)),
            "dataset_sha256": actual_hash,
            "split_mode": str(checkpoint.get("split_mode", "unknown")),
            "training_seed": int(checkpoint.get("training_seed", -1)),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "validation_accuracy": float(checkpoint["validation_accuracy"]),
            "checkpoint_test_accuracy": checkpoint_accuracy,
            "train_samples": int(len(checkpoint.get("train_indices", []))),
            "validation_samples": int(len(checkpoint.get("validation_indices", []))),
            "test_samples": int(len(test_indices)),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"majority_class_accuracy={metrics['majority_class_accuracy']:.4f}")
    print(f"balanced_accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"macro_precision={metrics['macro_precision']:.4f}")
    print(f"macro_recall={metrics['macro_recall']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print("rows=true, columns=predicted")
    print(np.asarray(metrics["confusion_matrix"], dtype=np.int64))
    print(f"Saved intent metrics to {output}")


if __name__ == "__main__":
    main()
