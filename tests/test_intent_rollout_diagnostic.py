import json
from types import SimpleNamespace

import pandas as pd
import pytest

from safeintent_rl.evaluation import EpisodeMetrics
from scripts import diagnose_intent_rollout
from scripts.diagnose_intent_rollout import _save_report, _verify_reference


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
