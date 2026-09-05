from dataclasses import asdict

import pytest

from safeintent_rl.intent.diagnostics import OnlineIntentDiagnostics


def _labeled_snapshot() -> dict:
    return {
        "slot_count": 2,
        "missing_slots": 1,
        "obstacle_slots": 0,
        "vehicle_slots": 1,
        "warmup_vehicle_slots": 0,
        "predicted_vehicle_slots": 1,
        "labeled_predictions": 1,
        "correct_predictions": 1,
        "confidence_sum": 0.8,
        "entropy_sum": 0.5,
        "true_labels": [0],
        "predicted_labels": [0],
    }


def test_online_diagnostics_aggregate_coverage_and_classification() -> None:
    diagnostics = OnlineIntentDiagnostics()
    diagnostics.update(
        {
            "slot_count": 5,
            "missing_slots": 1,
            "obstacle_slots": 1,
            "vehicle_slots": 3,
            "warmup_vehicle_slots": 1,
            "predicted_vehicle_slots": 2,
            "labeled_predictions": 2,
            "correct_predictions": 1,
            "confidence_sum": 1.5,
            "entropy_sum": 0.8,
            "true_labels": [0, 2],
            "predicted_labels": [0, 1],
        }
    )

    result = diagnostics.summarize(["cautious", "normal", "aggressive"])

    assert result["decisions"] == 1
    assert result["vehicle_slot_rate"] == pytest.approx(3 / 5)
    assert result["prediction_coverage"] == pytest.approx(2 / 3)
    assert result["mean_confidence"] == pytest.approx(0.75)
    assert result["classification"]["accuracy"] == pytest.approx(0.5)
    assert result["classification"]["confusion_matrix"] == [
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ]


def test_online_diagnostics_reject_inconsistent_slot_counts() -> None:
    diagnostics = OnlineIntentDiagnostics()
    with pytest.raises(ValueError, match="slot counts"):
        diagnostics.update(
            {
                "slot_count": 5,
                "missing_slots": 2,
                "obstacle_slots": 0,
                "vehicle_slots": 2,
                "warmup_vehicle_slots": 1,
                "predicted_vehicle_slots": 1,
                "labeled_predictions": 1,
                "correct_predictions": 1,
                "confidence_sum": 0.9,
                "entropy_sum": 0.2,
                "true_labels": [0],
                "predicted_labels": [0],
            }
        )


def test_online_diagnostics_preserve_zero_ready_prediction_coverage() -> None:
    diagnostics = OnlineIntentDiagnostics()
    diagnostics.update(
        {
            "slot_count": 5,
            "missing_slots": 2,
            "obstacle_slots": 1,
            "vehicle_slots": 2,
            "warmup_vehicle_slots": 2,
            "predicted_vehicle_slots": 0,
            "labeled_predictions": 0,
            "correct_predictions": 0,
            "confidence_sum": 0.0,
            "entropy_sum": 0.0,
            "true_labels": [],
            "predicted_labels": [],
        }
    )

    result = diagnostics.summarize(["cautious", "normal", "aggressive"])

    assert result["decisions"] == 1
    assert result["vehicle_slot_rate"] == pytest.approx(2 / 5)
    assert result["warmup_vehicle_slots"] == 2
    assert result["prediction_coverage"] == 0.0
    assert result["labeled_prediction_rate"] is None
    assert result["classification_available"] is False
    assert result["all_classes_observed"] is False
    assert result["classification"] is None
    assert result["mean_confidence"] is None
    assert result["mean_entropy"] is None
    assert result["mean_normalized_entropy"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slot_count", 2.5),
        ("missing_slots", -1),
        ("vehicle_slots", 2),
        ("labeled_predictions", 2),
        ("correct_predictions", 0),
        ("true_labels", [0.5]),
        ("predicted_labels", [3]),
        ("confidence_sum", float("nan")),
        ("confidence_sum", float("inf")),
        ("confidence_sum", -0.1),
        ("confidence_sum", 1.1),
        ("entropy_sum", float("nan")),
        ("entropy_sum", -0.1),
        ("entropy_sum", 1.2),
    ],
)
def test_invalid_snapshot_does_not_mutate_accumulated_evidence(field, value) -> None:
    diagnostics = OnlineIntentDiagnostics()
    diagnostics.update(_labeled_snapshot())
    before = asdict(diagnostics)
    invalid = _labeled_snapshot()
    invalid[field] = value

    with pytest.raises(ValueError):
        diagnostics.update(invalid)

    assert asdict(diagnostics) == before
