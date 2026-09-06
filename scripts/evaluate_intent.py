from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from safeintent_rl.intent.evaluation import classification_metrics
from safeintent_rl.intent.inference import IntentPredictor, load_intent_checkpoint
from safeintent_rl.intent.training import file_sha256, project_histories


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the intent GRU on a labeled archive")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/intent_gru_seed42.metrics.json")
    parser.add_argument("--data-sha256")
    parser.add_argument("--model-sha256")
    parser.add_argument("--expected-history-length", type=int)
    parser.add_argument("--refuse-overwrite", action="store_true")
    parser.add_argument("--require-sealed-test", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--minimum-accuracy-advantage", type=float, default=0.05)
    parser.add_argument("--minimum-macro-f1", type=float, default=0.50)
    parser.add_argument("--minimum-class-recall", type=float, default=0.40)
    parser.add_argument("--reference-accuracy", type=float)
    parser.add_argument("--maximum-reference-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--coverage-json")
    parser.add_argument("--coverage-json-sha256")
    parser.add_argument("--minimum-coverage", type=float, default=0.70)
    args = parser.parse_args()

    for name in (
        "minimum_accuracy_advantage",
        "minimum_macro_f1",
        "minimum_class_recall",
        "maximum_reference_accuracy_drop",
        "minimum_coverage",
    ):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between zero and one")
    if args.reference_accuracy is not None and not 0 <= args.reference_accuracy <= 1:
        raise ValueError("--reference-accuracy must be between zero and one")

    output = Path(args.output)
    if args.refuse_overwrite and output.exists():
        raise FileExistsError(f"Refusing to overwrite intent metrics: {output}")

    with np.load(args.data) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.int64)
    checkpoint = load_intent_checkpoint(args.model, map_location="cpu")
    checkpoint_accuracy = (
        float(checkpoint["test_accuracy"]) if "test_accuracy" in checkpoint else None
    )
    test_was_sealed = bool(
        checkpoint.get("test_metrics_sealed", checkpoint_accuracy is None)
    )
    if args.require_sealed_test and (not test_was_sealed or checkpoint_accuracy is not None):
        raise ValueError("Checkpoint does not contain a sealed held-out test split")
    if "test_indices" not in checkpoint:
        raise ValueError("Checkpoint does not contain a held-out test split")
    expected_hash = checkpoint.get("dataset_sha256")
    if not expected_hash:
        raise ValueError("Checkpoint does not contain a dataset fingerprint")
    actual_hash = file_sha256(args.data)
    if args.data_sha256 is not None and actual_hash.lower() != args.data_sha256.lower():
        raise ValueError("Dataset fingerprint does not match --data-sha256")
    if actual_hash != expected_hash:
        raise ValueError("Dataset fingerprint does not match the training checkpoint")
    model_hash = file_sha256(args.model)
    if args.model_sha256 is not None and model_hash.lower() != args.model_sha256.lower():
        raise ValueError("Intent checkpoint fingerprint does not match --model-sha256")

    source_history_length = int(checkpoint.get("source_history_length", x.shape[1]))
    if x.shape[1] != source_history_length:
        raise ValueError("Dataset source history length does not match the checkpoint")
    history_length = int(checkpoint.get("history_length", source_history_length))
    if (
        args.expected_history_length is not None
        and history_length != args.expected_history_length
    ):
        raise ValueError("Checkpoint history length does not match --expected-history-length")
    x = project_histories(x, history_length)

    coverage_evidence = None
    if args.coverage_json is not None:
        coverage_hash = file_sha256(args.coverage_json)
        if args.coverage_json_sha256 is None:
            raise ValueError("--coverage-json-sha256 is required with --coverage-json")
        if coverage_hash.lower() != args.coverage_json_sha256.lower():
            raise ValueError("Coverage artifact fingerprint does not match")
        coverage_report = json.loads(Path(args.coverage_json).read_text(encoding="utf-8"))
        curve = coverage_report.get("history_coverage_curve", {})
        if coverage_report.get("status") != "complete":
            raise ValueError("Coverage artifact is not complete")
        if int(curve.get("selected_history_length", -1)) != history_length:
            raise ValueError("Coverage artifact selected a different history length")
        coverage_row = curve.get("coverage_by_history_length", {}).get(
            str(history_length)
        )
        if not isinstance(coverage_row, dict) or "coverage" not in coverage_row:
            raise ValueError("Coverage artifact does not contain the selected curve row")
        coverage_evidence = {
            "path": str(Path(args.coverage_json)),
            "sha256": coverage_hash,
            "history_length": history_length,
            "coverage": float(coverage_row["coverage"]),
        }
    elif args.coverage_json_sha256 is not None:
        raise ValueError("--coverage-json is required with --coverage-json-sha256")

    test_indices = np.asarray(checkpoint["test_indices"], dtype=np.int64)
    if len(test_indices) == 0 or np.any((test_indices < 0) | (test_indices >= len(x))):
        raise ValueError("Checkpoint contains invalid test indices")
    if len(np.unique(test_indices)) != len(test_indices):
        raise ValueError("Checkpoint test indices contain duplicates")

    test_x = x[test_indices]
    test_y = y[test_indices]
    predictor = IntentPredictor(args.model, device=args.device)
    predictions = predictor.predict_proba(test_x).argmax(axis=1)
    metrics = classification_metrics(test_y, predictions, predictor.label_names)
    if checkpoint_accuracy is not None and not np.isclose(
        metrics["accuracy"], checkpoint_accuracy
    ):
        raise RuntimeError("Recomputed test accuracy does not match the checkpoint")

    accuracy_advantage = float(
        metrics["accuracy"] - metrics["majority_class_accuracy"]
    )
    minimum_observed_recall = min(
        float(result["recall"]) for result in metrics["per_class"].values()
    )
    all_classes_observed = all(
        int(result["support"]) > 0 for result in metrics["per_class"].values()
    )
    accuracy_floor = (
        args.reference_accuracy - args.maximum_reference_accuracy_drop
        if args.reference_accuracy is not None
        else None
    )
    acceptance_conditions = [
        all_classes_observed,
        accuracy_advantage >= args.minimum_accuracy_advantage,
        metrics["macro_f1"] >= args.minimum_macro_f1,
        minimum_observed_recall >= args.minimum_class_recall,
    ]
    if accuracy_floor is not None:
        acceptance_conditions.append(metrics["accuracy"] >= accuracy_floor)
    if coverage_evidence is not None:
        acceptance_conditions.append(
            coverage_evidence["coverage"] >= args.minimum_coverage
        )
    acceptance_gate = {
        "requirements": {
            "all_classes_observed": True,
            "minimum_accuracy_advantage": args.minimum_accuracy_advantage,
            "minimum_macro_f1": args.minimum_macro_f1,
            "minimum_class_recall": args.minimum_class_recall,
            "reference_accuracy": args.reference_accuracy,
            "maximum_reference_accuracy_drop": args.maximum_reference_accuracy_drop,
            "minimum_accuracy_from_reference": accuracy_floor,
            "minimum_coverage": (
                args.minimum_coverage if coverage_evidence is not None else None
            ),
        },
        "observed": {
            "all_classes_observed": all_classes_observed,
            "accuracy_advantage": accuracy_advantage,
            "macro_f1": float(metrics["macro_f1"]),
            "minimum_class_recall": minimum_observed_recall,
            "coverage": (
                coverage_evidence["coverage"] if coverage_evidence is not None else None
            ),
        },
        "passed": bool(all(acceptance_conditions)),
    }

    metrics.update(
        {
            "data_path": str(Path(args.data)),
            "model_path": str(Path(args.model)),
            "model_sha256": model_hash,
            "dataset_sha256": actual_hash,
            "source_history_length": source_history_length,
            "history_length": history_length,
            "history_projection": str(
                checkpoint.get("history_projection", "causal_suffix")
            ),
            "split_mode": str(checkpoint.get("split_mode", "unknown")),
            "training_seed": int(checkpoint.get("training_seed", -1)),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "validation_accuracy": float(checkpoint["validation_accuracy"]),
            "checkpoint_test_accuracy": checkpoint_accuracy,
            "test_metrics_were_sealed": test_was_sealed,
            "evaluation_device": args.device,
            "coverage_evidence": coverage_evidence,
            "acceptance_gate": acceptance_gate,
            "train_samples": int(len(checkpoint.get("train_indices", []))),
            "validation_samples": int(len(checkpoint.get("validation_indices", []))),
            "test_samples": int(len(test_indices)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"majority_class_accuracy={metrics['majority_class_accuracy']:.4f}")
    print(f"balanced_accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"macro_precision={metrics['macro_precision']:.4f}")
    print(f"macro_recall={metrics['macro_recall']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"acceptance_gate_passed={acceptance_gate['passed']}")
    print("rows=true, columns=predicted")
    print(np.asarray(metrics["confusion_matrix"], dtype=np.int64))
    print(f"Saved intent metrics to {output}")


if __name__ == "__main__":
    main()
