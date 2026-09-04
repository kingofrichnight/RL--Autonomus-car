from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, Any]:
    """Return JSON-serializable multiclass metrics and a confusion matrix."""
    target_array = np.asarray(targets, dtype=np.int64).reshape(-1)
    prediction_array = np.asarray(predictions, dtype=np.int64).reshape(-1)
    class_count = len(label_names)
    if len(target_array) == 0:
        raise ValueError("At least one labeled sample is required")
    if len(target_array) != len(prediction_array):
        raise ValueError("targets and predictions must have the same length")
    if class_count == 0:
        raise ValueError("label_names must not be empty")
    if np.any((target_array < 0) | (target_array >= class_count)):
        raise ValueError("targets contain an invalid class index")
    if np.any((prediction_array < 0) | (prediction_array >= class_count)):
        raise ValueError("predictions contain an invalid class index")

    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (target_array, prediction_array), 1)

    per_class: dict[str, dict[str, float | int]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_index, label_name in enumerate(label_names):
        true_positive = int(confusion[class_index, class_index])
        predicted_positive = int(confusion[:, class_index].sum())
        actual_positive = int(confusion[class_index, :].sum())
        precision = true_positive / max(predicted_positive, 1)
        recall = true_positive / max(actual_positive, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[str(label_name)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": actual_positive,
            "predicted": predicted_positive,
        }
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))

    return {
        "samples": int(len(target_array)),
        "accuracy": float((prediction_array == target_array).mean()),
        "majority_class_accuracy": float(confusion.sum(axis=1).max() / len(target_array)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "label_names": [str(name) for name in label_names],
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
