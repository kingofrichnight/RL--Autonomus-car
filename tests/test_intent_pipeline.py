import json
import sys

import numpy as np
import pytest
import torch

from safeintent_rl.intent.dataset import TrajectoryCollector, save_dataset
from safeintent_rl.intent.evaluation import classification_metrics
from safeintent_rl.intent.inference import IntentPredictor, load_intent_checkpoint
from safeintent_rl.intent.training import (
    _clone_state_dict,
    create_data_split,
    file_sha256,
    project_histories,
    train_intent_model,
)
from scripts.evaluate_intent import main as evaluate_intent_main


def test_trajectory_collector_rejects_invalid_window_settings() -> None:
    with pytest.raises(ValueError, match="history_length"):
        TrajectoryCollector(history_length=0)
    with pytest.raises(ValueError, match="sample_stride"):
        TrajectoryCollector(sample_stride=0)


def test_save_dataset_preserves_metadata(tmp_path) -> None:
    output = tmp_path / "intent.npz"
    samples = [np.zeros((10, 6), dtype=np.float32)]
    save_dataset(
        output,
        samples,
        [2],
        [17],
        metadata={"first_seed": 42, "collector": "seeded_random_ego_policy"},
    )

    with np.load(output) as archive:
        assert archive["x"].shape == (1, 10, 6)
        assert archive["y"].tolist() == [2]
        assert archive["episode_ids"].tolist() == [17]
        metadata = json.loads(str(archive["metadata_json"]))
    assert metadata == {"collector": "seeded_random_ego_policy", "first_seed": 42}


def test_save_dataset_rejects_empty_samples(tmp_path) -> None:
    with pytest.raises(ValueError, match="shape"):
        save_dataset(tmp_path / "empty.npz", [], [], [])


def test_episode_split_is_deterministic_and_has_no_group_leakage() -> None:
    groups = np.repeat(np.arange(20), 3)
    first = create_data_split(len(groups), seed=42, episode_ids=groups)
    second = create_data_split(len(groups), seed=42, episode_ids=groups)

    assert first.mode == "episode"
    assert np.array_equal(first.train_indices, second.train_indices)
    train_groups = set(groups[first.train_indices])
    validation_groups = set(groups[first.validation_indices])
    test_groups = set(groups[first.test_indices])
    assert len(train_groups) == 14
    assert len(validation_groups) == 3
    assert len(test_groups) == 3
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_cloned_state_dict_does_not_share_cpu_tensor_storage() -> None:
    parameter = torch.tensor([1.0])
    copied = _clone_state_dict({"weight": parameter})
    parameter.add_(2.0)
    assert copied["weight"].item() == pytest.approx(1.0)


def test_project_histories_uses_causal_suffix() -> None:
    x = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)

    projected = project_histories(x, 3)

    assert projected.shape == (2, 3, 6)
    assert np.array_equal(projected, x[:, -3:, :])
    with pytest.raises(ValueError, match="history_length"):
        project_histories(x, 5)


def test_classification_metrics_include_per_class_results() -> None:
    targets = np.asarray([0, 0, 1, 1, 2, 2])
    predictions = np.asarray([0, 1, 1, 1, 2, 0])
    metrics = classification_metrics(targets, predictions, ["cautious", "normal", "aggressive"])

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["majority_class_accuracy"] == pytest.approx(1 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(metrics["macro_recall"])
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert metrics["per_class"]["aggressive"]["support"] == 2


def test_intent_training_rejects_a_split_missing_a_driver_class(tmp_path) -> None:
    x = np.zeros((30, 4, 6), dtype=np.float32)
    y = np.zeros(30, dtype=np.int64)
    y[3:6] = 1
    y[6:9] = 2
    episode_ids = np.repeat(np.arange(10), 3)
    data_path = tmp_path / "missing_split_class.npz"
    np.savez_compressed(data_path, x=x, y=y, episode_ids=episode_ids)

    with pytest.raises(ValueError, match="split must contain"):
        train_intent_model(data_path, tmp_path / "unused.pt", epochs=1, seed=42)


def test_intent_training_checkpoint_records_reproducibility_metadata(tmp_path) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 4, 6)).astype(np.float32)
    y = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 10)
    episode_ids = np.repeat(np.arange(10), 3)
    data_path = tmp_path / "training.npz"
    model_path = tmp_path / "intent.pt"
    np.savez_compressed(data_path, x=x, y=y, episode_ids=episode_ids)

    result = train_intent_model(data_path, model_path, epochs=1, batch_size=8, seed=7)
    checkpoint = load_intent_checkpoint(model_path, map_location="cpu")

    assert result.best_epoch == 1
    assert result.split_mode == "episode"
    assert result.dataset_sha256 == file_sha256(data_path)
    assert checkpoint["dataset_sha256"] == result.dataset_sha256
    assert len(checkpoint["train_indices"]) == result.train_samples
    assert len(checkpoint["validation_indices"]) == result.validation_samples
    assert len(checkpoint["test_indices"]) == result.test_samples
    assert checkpoint["history_length"] == 4
    assert checkpoint["source_history_length"] == 4
    assert checkpoint["history_projection"] == "causal_suffix"
    assert checkpoint["test_metrics_sealed"] is False
    assert checkpoint["test_accuracy"] == pytest.approx(result.test_accuracy)

    with pytest.raises(ValueError, match="fingerprint"):
        IntentPredictor(model_path, device="cpu", expected_sha256="0" * 64)


def test_development_training_projects_history_and_seals_test_metrics(
    tmp_path, monkeypatch
) -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(30, 4, 6)).astype(np.float32)
    y = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 10)
    episode_ids = np.repeat(np.arange(10), 3)
    data_path = tmp_path / "training.npz"
    model_path = tmp_path / "intent_h3.pt"
    summary_path = tmp_path / "intent_h3.training.json"
    np.savez_compressed(data_path, x=x, y=y, episode_ids=episode_ids)

    result = train_intent_model(
        data_path,
        model_path,
        epochs=1,
        batch_size=8,
        seed=7,
        history_length=3,
        evaluate_test=False,
        expected_data_sha256=file_sha256(data_path),
        summary_output_path=summary_path,
        device="cpu",
        refuse_overwrite=True,
    )
    checkpoint = load_intent_checkpoint(model_path, map_location="cpu")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = create_data_split(len(x), seed=7, episode_ids=episode_ids)
    projected = x[:, -3:, :]
    expected_mean = projected[split.train_indices].mean(axis=(0, 1), keepdims=True)
    expected_std = projected[split.train_indices].std(axis=(0, 1), keepdims=True) + 1e-6

    assert result.test_accuracy is None
    assert result.history_length == 3
    assert result.source_history_length == 4
    assert result.device == "cpu"
    assert checkpoint["history_length"] == 3
    assert checkpoint["source_history_length"] == 4
    assert checkpoint["history_projection"] == "causal_suffix"
    assert checkpoint["test_metrics_sealed"] is True
    assert "test_accuracy" not in checkpoint
    assert checkpoint["validation_metrics"]["samples"] == result.validation_samples
    assert checkpoint["validation_gate"]["passed"] == result.validation_gate_passed
    assert np.allclose(checkpoint["mean"], expected_mean)
    assert np.allclose(checkpoint["std"], expected_std)
    assert summary["checkpoint_sha256"] == file_sha256(model_path)
    assert summary["dataset_sha256"] == file_sha256(data_path)
    assert summary["history_length"] == 3
    assert summary["source_history_length"] == 4
    assert summary["test_metrics_sealed"] is True
    assert summary["test_accuracy"] is None
    assert summary["validation_metrics"]["samples"] == result.validation_samples
    assert summary["validation_gate"]["passed"] == result.validation_gate_passed
    assert set(summary["split_episode_ids"]) == {"train", "validation", "test"}

    predictor = IntentPredictor(model_path, device="cpu")
    assert predictor.predict_proba(projected[:2]).shape == (2, 3)
    with pytest.raises(ValueError, match="expects history length"):
        predictor.predict_proba(x[:2])

    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "history_coverage_curve": {
                    "selected_history_length": 3,
                    "coverage_by_history_length": {"3": {"coverage": 0.75}},
                },
            }
        ),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "intent_h3.metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_intent.py",
            "--data",
            str(data_path),
            "--data-sha256",
            file_sha256(data_path),
            "--model",
            str(model_path),
            "--model-sha256",
            file_sha256(model_path),
            "--expected-history-length",
            "3",
            "--require-sealed-test",
            "--device",
            "cpu",
            "--coverage-json",
            str(coverage_path),
            "--coverage-json-sha256",
            file_sha256(coverage_path),
            "--reference-accuracy",
            "0.0",
            "--output",
            str(metrics_path),
            "--refuse-overwrite",
        ],
    )
    evaluate_intent_main()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["history_length"] == 3
    assert metrics["test_metrics_were_sealed"] is True
    assert metrics["checkpoint_test_accuracy"] is None
    assert metrics["coverage_evidence"]["coverage"] == pytest.approx(0.75)
    assert "passed" in metrics["acceptance_gate"]
    with pytest.raises(FileExistsError, match="intent metrics"):
        evaluate_intent_main()

    with pytest.raises(FileExistsError, match="checkpoint"):
        train_intent_model(
            data_path,
            model_path,
            epochs=1,
            history_length=3,
            evaluate_test=False,
            summary_output_path=summary_path,
            refuse_overwrite=True,
        )


def test_intent_training_checks_expected_dataset_fingerprint(tmp_path) -> None:
    data_path = tmp_path / "training.npz"
    np.savez_compressed(
        data_path,
        x=np.zeros((30, 4, 6), dtype=np.float32),
        y=np.tile(np.asarray([0, 1, 2], dtype=np.int64), 10),
    )

    with pytest.raises(ValueError, match="--data-sha256"):
        train_intent_model(
            data_path,
            tmp_path / "unused.pt",
            epochs=1,
            expected_data_sha256="0" * 64,
        )
