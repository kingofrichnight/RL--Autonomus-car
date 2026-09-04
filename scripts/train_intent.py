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
    args = parser.parse_args()

    result = train_intent_model(
        args.data,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(f"best_validation_accuracy={result.best_validation_accuracy:.4f}")
    print(f"test_accuracy={result.test_accuracy:.4f}")
    print(f"best_epoch={result.best_epoch}")
    print(f"split_mode={result.split_mode}")
    print(
        "split_samples="
        f"{result.train_samples}/{result.validation_samples}/{result.test_samples}"
    )
    print(f"dataset_sha256={result.dataset_sha256}")
    print(f"checkpoint={result.checkpoint_path}")


if __name__ == "__main__":
    main()
