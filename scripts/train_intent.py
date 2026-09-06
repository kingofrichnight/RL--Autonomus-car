from __future__ import annotations

import argparse

from safeintent_rl.intent.training import train_intent_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GRU driver-intent classifier")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/intent_gru.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-length", type=int)
    parser.add_argument("--data-sha256")
    parser.add_argument("--summary-output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--seal-test",
        action="store_true",
        help="Do not calculate or store held-out test metrics during training",
    )
    parser.add_argument("--refuse-overwrite", action="store_true")
    parser.add_argument("--minimum-validation-accuracy-advantage", type=float, default=0.05)
    parser.add_argument("--minimum-validation-macro-f1", type=float, default=0.50)
    parser.add_argument("--minimum-validation-class-recall", type=float, default=0.40)
    args = parser.parse_args()

    result = train_intent_model(
        args.data,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        history_length=args.history_length,
        evaluate_test=not args.seal_test,
        expected_data_sha256=args.data_sha256,
        summary_output_path=args.summary_output,
        device=args.device,
        refuse_overwrite=args.refuse_overwrite,
        minimum_validation_accuracy_advantage=(
            args.minimum_validation_accuracy_advantage
        ),
        minimum_validation_macro_f1=args.minimum_validation_macro_f1,
        minimum_validation_class_recall=args.minimum_validation_class_recall,
    )
    print(f"best_validation_accuracy={result.best_validation_accuracy:.4f}")
    if result.test_accuracy is None:
        print("test_accuracy=sealed")
    else:
        print(f"test_accuracy={result.test_accuracy:.4f}")
    print(f"best_epoch={result.best_epoch}")
    print(f"split_mode={result.split_mode}")
    print(
        "split_samples="
        f"{result.train_samples}/{result.validation_samples}/{result.test_samples}"
    )
    print(f"dataset_sha256={result.dataset_sha256}")
    print(f"history_length={result.history_length}")
    print(f"source_history_length={result.source_history_length}")
    print(f"device={result.device}")
    print(f"checkpoint_sha256={result.checkpoint_sha256}")
    print(f"validation_macro_f1={result.validation_metrics['macro_f1']:.4f}")
    print(f"validation_gate_passed={result.validation_gate_passed}")
    print(f"checkpoint={result.checkpoint_path}")
    if result.summary_path is not None:
        print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
