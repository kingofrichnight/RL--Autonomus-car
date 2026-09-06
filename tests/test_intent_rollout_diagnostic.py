import json
from types import SimpleNamespace

import pandas as pd
import pytest

from safeintent_rl.evaluation import EpisodeMetrics
from safeintent_rl.intent.diagnostics import HistoryCoverageComparison, OnlineIntentDiagnostics
from scripts import diagnose_intent_rollout
from scripts.diagnose_intent_rollout import (
    _parse_history_lengths,
    _save_report,
    _verify_diagnostic_reference,
    _verify_reference,
)


def _episode(reward: float = 1.5) -> EpisodeMetrics:
    return EpisodeMetrics(
        reward=reward,
        length=12,
        success=True,
        collision=False,
        travel_time=2.4,
        min_ttc=0.8,
        unsafe_ttc_events=3,
        safety_interventions=0,
    )


def test_reference_verification_accepts_exact_episode_results(tmp_path) -> None:
    episodes = [_episode()]
    reference = tmp_path / "reference.csv"
    pd.DataFrame([episodes[0].as_dict()]).to_csv(reference, index=False)

    _verify_reference(episodes, str(reference))


def test_reference_verification_rejects_changed_episode_results(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    pd.DataFrame([_episode().as_dict()]).to_csv(reference, index=False)

    with pytest.raises(RuntimeError, match="changed"):
        _verify_reference([_episode(reward=2.0)], str(reference))


def test_save_report_refuses_to_overwrite_existing_evidence(tmp_path) -> None:
    output = tmp_path / "reports" / "diagnostics.json"
    _save_report(output, {"status": "failed", "error": "original evidence"})
    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        _save_report(output, {"status": "complete"})

    assert output.read_bytes() == original


def test_diagnostic_reference_accepts_nested_numeric_equivalence() -> None:
    reference = {"count": 3, "rate": 0.25, "matrix": [[1, 2], [3, 4]]}
    observed = {"count": 3, "rate": 0.25 + 5e-13, "matrix": [[1, 2], [3, 4]]}

    _verify_diagnostic_reference(observed, reference)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("4,5,6,7", (4, 5, 6, 7)), ("4, 6", (4, 6))],
)
def test_parse_history_lengths(value, expected) -> None:
    assert _parse_history_lengths(value) == expected


@pytest.mark.parametrize("value", ["", "4,4", "6,4", "0,4", "four,5"])
def test_parse_history_lengths_rejects_invalid_candidates(value) -> None:
    with pytest.raises(Exception):
        _parse_history_lengths(value)


@pytest.mark.parametrize(
    ("observed", "match"),
    [
        ({"count": 4}, "structure"),
        ({"count": 3, "rate": 0.3, "matrix": [[1, 2], [3, 4]]}, "value"),
        ({"count": 3, "rate": 0.25, "matrix": [[1, 2]]}, "sequence"),
    ],
)
def test_diagnostic_reference_rejects_changed_evidence(observed, match) -> None:
    reference = {"count": 3, "rate": 0.25, "matrix": [[1, 2], [3, 4]]}

    with pytest.raises(RuntimeError, match=match):
        _verify_diagnostic_reference(observed, reference)


def test_reference_failure_saves_diagnostics_and_raises(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "reference.csv"
    expected = _episode().as_dict()
    expected.update(length=1, travel_time=0.2, unsafe_ttc_events=1)
    pd.DataFrame([expected]).to_csv(reference, index=False)
    output = tmp_path / "failed_diagnostics.json"
    snapshot = {
        "slot_count": 2,
        "missing_slots": 1,
        "obstacle_slots": 0,
        "vehicle_slots": 1,
        "warmup_vehicle_slots": 1,
        "predicted_vehicle_slots": 0,
        "labeled_predictions": 0,
        "correct_predictions": 0,
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "true_labels": [],
        "predicted_labels": [],
    }
    closed = []
    base = SimpleNamespace(
        vehicle=SimpleNamespace(crashed=False),
        road=SimpleNamespace(vehicles=[]),
        config={"policy_frequency": 5},
    )
    env = SimpleNamespace(
        unwrapped=base,
        last_intent_diagnostics=snapshot,
        predictor=SimpleNamespace(label_names=["cautious", "normal", "aggressive"]),
        reset=lambda **kwargs: ([0.0], {}),
        step=lambda action: ([0.0], 2.0, True, False, {"min_ttc": 0.8}),
        close=lambda: closed.append(True),
    )
    model = SimpleNamespace(predict=lambda observation, deterministic: (0, None))
    monkeypatch.setattr(
        diagnose_intent_rollout, "PPO", SimpleNamespace(load=lambda *args, **kwargs: model)
    )
    monkeypatch.setattr(diagnose_intent_rollout, "make_intersection_env", lambda **kwargs: env)
    monkeypatch.setattr(diagnose_intent_rollout, "_intent_wrapper", lambda value: value)
    monkeypatch.setattr(diagnose_intent_rollout, "_require_sha256", lambda *args: "a" * 64)
    monkeypatch.setattr(diagnose_intent_rollout, "minimum_ttc", lambda *args: 0.8)
    monkeypatch.setattr(diagnose_intent_rollout, "detect_success", lambda *args: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_intent_rollout.py",
            "--model", "unused.zip", "--model-sha256", "a" * 64,
            "--config", "unused.yaml", "--config-sha256", "a" * 64,
            "--intent-model", "unused.pt", "--intent-model-sha256", "a" * 64,
            "--episodes", "1", "--intent-neighbors", "2",
            "--reference-csv", str(reference), "--reference-csv-sha256", "a" * 64,
            "--output", str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="Diagnostic failed; preserve"):
        diagnose_intent_rollout.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert closed == [True]
    assert report["status"] == "failed"
    assert "reference episode result" in report["error"]
    assert report["driving_reference_reproduced"] is False
    assert report["episodes_completed"] == 1
    assert report["failed_episode_results"][0]["reward"] == 2.0
    assert report["online_intent"]["decisions"] == 1
    assert report["online_intent"]["prediction_coverage"] == 0.0
    assert report["online_intent"]["classification"] is None


def test_shadow_run_reproduces_original_diagnostics_and_saves_comparison(
    tmp_path,
    monkeypatch,
) -> None:
    reference_csv = tmp_path / "reference.csv"
    expected = _episode().as_dict()
    expected.update(length=1, travel_time=0.2, unsafe_ttc_events=1)
    pd.DataFrame([expected]).to_csv(reference_csv, index=False)
    current_snapshot = {
        "slot_count": 2,
        "missing_slots": 1,
        "obstacle_slots": 0,
        "vehicle_slots": 1,
        "warmup_vehicle_slots": 1,
        "predicted_vehicle_slots": 0,
        "labeled_predictions": 0,
        "correct_predictions": 0,
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "true_labels": [],
        "predicted_labels": [],
        "vehicle_rows": [0],
        "predicted_rows": [],
    }
    shadow_snapshot = {
        **current_snapshot,
        "warmup_vehicle_slots": 0,
        "predicted_vehicle_slots": 1,
        "labeled_predictions": 1,
        "correct_predictions": 1,
        "confidence_sum": 0.8,
        "entropy_sum": 0.5,
        "true_labels": [2],
        "predicted_labels": [2],
        "vehicle_rows": [0],
        "predicted_rows": [0],
        "history_lengths": [1],
    }
    labels = ["cautious", "normal", "aggressive"]
    current = OnlineIntentDiagnostics()
    current.update(current_snapshot)
    reference_diagnostic = tmp_path / "reference.json"
    reference_diagnostic.write_text(
        json.dumps(
            {
                "status": "complete",
                "episodes": 1,
                "first_seed": 10042,
                "last_seed": 10042,
                "model_sha256": "a" * 64,
                "config_sha256": "a" * 64,
                "intent_model_sha256": "a" * 64,
                "reference_csv_sha256": "a" * 64,
                "online_intent": current.summarize(labels),
            }
        ),
        encoding="utf-8",
    )
    shadow_aggregate = OnlineIntentDiagnostics()
    shadow_aggregate.update(shadow_snapshot)
    comparison = HistoryCoverageComparison()
    comparison.update(current_snapshot, shadow_snapshot)
    reference_shadow_diagnostic = tmp_path / "reference_shadow.json"
    reference_shadow_diagnostic.write_text(
        json.dumps(
            {
                "status": "complete",
                "episodes": 1,
                "first_seed": 10042,
                "last_seed": 10042,
                "shadow_history_neighbors": 3,
                "model_sha256": "a" * 64,
                "config_sha256": "a" * 64,
                "intent_model_sha256": "a" * 64,
                "reference_csv_sha256": "a" * 64,
                "reference_diagnostic_json_sha256": "a" * 64,
                "online_intent": current.summarize(labels),
                "shadow_online_intent": shadow_aggregate.summarize(labels),
                "coverage_comparison": comparison.summarize(),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "shadow.json"
    base = SimpleNamespace(
        vehicle=SimpleNamespace(crashed=False),
        road=SimpleNamespace(vehicles=[]),
        config={"policy_frequency": 5},
    )
    env = SimpleNamespace(
        unwrapped=base,
        last_intent_diagnostics=current_snapshot,
        last_shadow_intent_diagnostics=shadow_snapshot,
        predictor=SimpleNamespace(label_names=labels),
        reset=lambda **kwargs: ([0.0], {}),
        step=lambda action: ([0.0], 1.5, True, False, {"min_ttc": 0.8}),
        close=lambda: None,
    )
    model = SimpleNamespace(predict=lambda observation, deterministic: (0, None))
    monkeypatch.setattr(
        diagnose_intent_rollout, "PPO", SimpleNamespace(load=lambda *args, **kwargs: model)
    )
    monkeypatch.setattr(diagnose_intent_rollout, "make_intersection_env", lambda **kwargs: env)
    monkeypatch.setattr(diagnose_intent_rollout, "_intent_wrapper", lambda value: value)
    monkeypatch.setattr(diagnose_intent_rollout, "_require_sha256", lambda *args: "a" * 64)
    monkeypatch.setattr(diagnose_intent_rollout, "minimum_ttc", lambda *args: 0.8)
    monkeypatch.setattr(diagnose_intent_rollout, "detect_success", lambda *args: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_intent_rollout.py",
            "--model", "unused.zip", "--model-sha256", "a" * 64,
            "--config", "unused.yaml", "--config-sha256", "a" * 64,
            "--intent-model", "unused.pt", "--intent-model-sha256", "a" * 64,
            "--episodes", "1", "--intent-neighbors", "2",
            "--shadow-history-neighbors", "3",
            "--reference-csv", str(reference_csv), "--reference-csv-sha256", "a" * 64,
            "--reference-diagnostic-json", str(reference_diagnostic),
            "--reference-diagnostic-json-sha256", "a" * 64,
            "--reference-shadow-diagnostic-json", str(reference_shadow_diagnostic),
            "--reference-shadow-diagnostic-json-sha256", "a" * 64,
            "--history-coverage-lengths", "1,2",
            "--history-coverage-minimum", "0.7",
            "--output", str(output),
        ],
    )

    diagnose_intent_rollout.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["driving_reference_reproduced"] is True
    assert report["online_intent_reference_reproduced"] is True
    assert report["online_intent"]["prediction_coverage"] == 0.0
    assert report["shadow_online_intent"]["prediction_coverage"] == 1.0
    assert report["coverage_comparison"]["shadow_only"] == 1
    assert report["coverage_comparison"]["shadow_is_readiness_superset"] is True
    assert report["shadow_diagnostic_reference_reproduced"] is True
    assert report["history_coverage_curve"]["selected_history_length"] == 1
    assert report["shadow_history_neighbors"] == 3
