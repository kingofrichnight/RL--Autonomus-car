import json

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
    train_intent_model,
)


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

    with pytest.raises(ValueError, match="fingerprint"):
        IntentPredictor(model_path, device="cpu", expected_sha256="0" * 64)
