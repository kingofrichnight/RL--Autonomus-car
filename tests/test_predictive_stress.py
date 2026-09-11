import copy
import json
import sys

import numpy as np
import pytest

from safeintent_rl.config import load_config
from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.intent.inference import file_sha256
from scripts import evaluate_predictive_stress as stress


def protocol():
    with open("configs/v3_predictive_stress_v1.json", encoding="utf-8") as stream:
        return json.load(stream)


def test_protocol_is_valid_and_changes_only_one_factor():
    spec = protocol()
    stress.validate_protocol(spec)
    base = load_config(spec["arms"]["control"]["config"])
    original = copy.deepcopy(base)
    for scenario in spec["scenarios"]:
        effective = stress.scenario_config(base, scenario)
        effective["spawn_probability"] = base["spawn_probability"]
        assert effective == base
    assert base == original


@pytest.mark.parametrize("field,value", [
    ("episodes_per_scenario", 99), ("first_seed", 40042), ("duration", 60),
    ("stopped_speed_threshold", 0), ("unsafe_ttc_threshold", 3),
])
def test_rejects_protocol_changes(field, value):
    spec = protocol()
    spec[field] = value
    with pytest.raises(ValueError):
        stress.validate_protocol(spec)


@pytest.mark.parametrize("probabilities", [[-1, 1, 1], [1, 1, 1], [0, 1],
                                           [float("nan"), 0, 1]])
def test_rejects_invalid_probabilities(probabilities):
    spec = protocol()
    spec["scenarios"][0]["driver_probabilities"] = probabilities
    with pytest.raises(ValueError):
        stress.validate_protocol(spec)


def test_default_profile_override_preserves_dynamics():
    a = make_intersection_env("configs/intersection_reward_v3_collision_first.yaml")
    b = make_intersection_env("configs/intersection_reward_v3_collision_first.yaml",
                              driver_probabilities=(0.3, 0.45, 0.25))
    try:
        x, _ = a.reset(seed=7)
        y, _ = b.reset(seed=7)
        np.testing.assert_array_equal(x, y)
        for action in [0, 1, 2, 1]:
            x, y = a.step(action), b.step(action)
            np.testing.assert_array_equal(x[0], y[0])
            assert x[1:4] == y[1:4]
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("label,probabilities", [
    ("cautious", (1, 0, 0)), ("aggressive", (0, 0, 1)),
])
def test_profile_override_is_applied_to_npcs_not_ego(label, probabilities):
    env = make_intersection_env("configs/intersection_reward_v3_collision_first.yaml",
                                driver_probabilities=probabilities)
    try:
        env.reset(seed=7)
        for _ in range(3):
            assert not hasattr(env.unwrapped.vehicle, "safeintent_driver_label")
            for actor in env.unwrapped.road.vehicles:
                if actor is not env.unwrapped.vehicle:
                    assert actor.safeintent_driver_label == label
            env.step(1)
    finally:
        env.close()


def test_summary_reports_incomplete_and_nullable_ttc():
    row = dict(success=False, collision=False, incomplete=True, travel_time=30,
               stopped_time=20, zero_target_time=19, unsafe_ttc_events=0,
               max_npcs=10, reward=-1, min_ttc=None)
    result = stress.summarize([row])
    assert result["incomplete_rate"] == 1
    assert result["success_count"] == 0
    assert result["mean_min_ttc"] is None
    assert result["mean_stopped_time"] == 20


def test_failed_run_preserves_report_and_refuses_overwrite(tmp_path, monkeypatch):
    spec = protocol()
    fake_model = tmp_path / "model.zip"
    fake_model.write_bytes(b"not a model")
    spec["arms"]["control"]["model"] = str(fake_model)
    spec["arms"]["control"]["model_sha256"] = file_sha256(fake_model)
    spec_path = tmp_path / "protocol.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["check", "--arm", "control", "--protocol", str(spec_path),
                        "--protocol-sha256", file_sha256(spec_path), "--output-dir", str(output)])

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(stress.PPO, "load", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        stress.main()
    saved = (output / "report.json").read_bytes()
    assert json.loads(saved)["status"] == "failed"
    with pytest.raises(FileExistsError):
        stress.main()
    assert (output / "report.json").read_bytes() == saved
