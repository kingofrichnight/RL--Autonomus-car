"""Fresh, uniquely named retry of the interrupted synchronized PPO experiment.

Only output paths and provenance change. The original runner, model protocol,
training/evaluation settings and verification logic stay byte-identical. This
is not checkpoint continuation, a sensor experiment, or callback-best selection.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from scripts import run_synchronized_predictive as original
from scripts.audit_geometry_development import sha

STEM = "ppo_v3_predictive_sync_v1_retry01_seed42"
MODEL = f"models/{STEM}.zip"
TRAIN_SUMMARY = f"results/{STEM}.training.json"
EVAL_OUTPUT = "results/ppo_v3_predictive_sync_v1_retry01_development_seed40042.csv"
ORIGINAL_RECORD = "results/ppo_v3_predictive_sync_v1_seed42.train.run.json"
INTERRUPTION_RECORD = "results/ppo_v3_predictive_sync_v1_seed42.interruption.json"
LINEAGE_HASHES = {
    "scripts/run_synchronized_predictive.py":
        "9ab2dfbbb90b89f425cdca8e52f7c61f11e24f481efb2f5e0a8b87dc82fe4823",
    ORIGINAL_RECORD: "4a884209aaeb320255416eedbf9d170bbecaee946a6e9fa998005618f1ec60a7",
    INTERRUPTION_RECORD: "ab64f59b8d8b891213612948cc1cb7817cca39b30501c4b321ed14e9b3fc1ea7",
}
ORIGINAL_FINGERPRINT = original.fingerprint_inputs


def retry_inputs() -> dict[str, str]:
    """Bind this fresh attempt to all 42 original inputs and interruption evidence."""
    for path, digest in LINEAGE_HASHES.items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Retry lineage changed: {path}")
    prior = json.loads(Path(ORIGINAL_RECORD).read_text())
    interruption = json.loads(Path(INTERRUPTION_RECORD).read_text())
    sources = dict(ORIGINAL_FINGERPRINT())
    if sources != prior["source_sha256"]:
        raise ValueError("Original training input snapshot differs")
    if (prior["observation_protocol"] != original.PROTOCOL
            or interruption["status"] != "interrupted"
            or interruption["training_stem"] != "ppo_v3_predictive_sync_v1_seed42"):
        raise ValueError("Retry requires the recorded interrupted predecessor")
    sources.update(LINEAGE_HASHES)
    sources["scripts/run_synchronized_retry.py"] = sha(Path(__file__))
    return sources


@contextmanager
def retry_paths():
    """Process-local path/provenance adapter; always restore historical globals.

    Use in the dedicated CLI process, not concurrently from multiple threads.
    The original fingerprint implementation is captured before adaptation to
    avoid recursion and to retain the historical module's source identity.
    """
    replacements = {"STEM": STEM, "MODEL": MODEL, "TRAIN_SUMMARY": TRAIN_SUMMARY,
                    "EVAL_OUTPUT": EVAL_OUTPUT, "fingerprint_inputs": retry_inputs}
    saved = {name: getattr(original, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(original, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(original, name, value)


def run(mode: str) -> None:
    if mode not in ("train", "evaluate"):
        raise ValueError("Only fresh train or completed-model evaluate is supported")
    with retry_paths():
        original.run(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--refuse-overwrite", action="store_true", required=True)
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
