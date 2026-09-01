from __future__ import annotations

import argparse

import numpy as np
import torch

from safeintent_rl.intent.inference import IntentPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the intent GRU on a labeled archive")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    archive = np.load(args.data)
    x = archive["x"].astype(np.float32)
    y = archive["y"].astype(np.int64)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    test_indices = np.asarray(checkpoint.get("test_indices", np.arange(len(x))), dtype=np.int64)
    x = x[test_indices]
    y = y[test_indices]
    predictor = IntentPredictor(args.model)
    predictions = predictor.predict_proba(x).argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for target, prediction in zip(y, predictions, strict=True):
        confusion[target, prediction] += 1
    accuracy = float((predictions == y).mean())
    precision = []
    recall = []
    f1 = []
    for class_index in range(3):
        true_positive = confusion[class_index, class_index]
        predicted_positive = confusion[:, class_index].sum()
        actual_positive = confusion[class_index, :].sum()
        class_precision = true_positive / max(predicted_positive, 1)
        class_recall = true_positive / max(actual_positive, 1)
        class_f1 = 2 * class_precision * class_recall / max(class_precision + class_recall, 1e-12)
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
    print(f"accuracy={accuracy:.4f}")
    print(f"macro_precision={np.mean(precision):.4f}")
    print(f"macro_recall={np.mean(recall):.4f}")
    print(f"macro_f1={np.mean(f1):.4f}")
    print("rows=true, columns=predicted")
    print(confusion)


if __name__ == "__main__":
    main()
