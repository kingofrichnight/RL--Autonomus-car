from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from safeintent_rl.evaluation import EpisodeMetrics
from scripts import diagnose_fusion_timing as diagnostic


def frame(outcomes: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"success": [x == "S" for x in outcomes], "collision": [x == "C" for x in outcomes]}
    )


def test_selection_includes_all_discordances_and_first_controls_by_row() -> None:
    selected = diagnostic.select_cases(frame("SSCCCIIISS"), frame("SCSCSSIISS"), 100, 1)
    assert [row["row_index"] for row in selected] == [0, 1, 2, 3, 4, 5]
    assert [row["seed"] for row in selected] == [100, 101, 102, 103, 104, 105]
    assert [row["transition"] for row in selected] == [
        "success_to_success",
        "success_to_collision",
        "collision_to_success",
        "collision_to_collision",
        "collision_to_success",
        "incomplete_to_success",
    ]
    assert [row["stratum"] for row in selected].count("control") == 2
    assert diagnostic.select_cases(frame("IC"), frame("CI"), 42, 0)[0]["transition"] == (
        "incomplete_to_collision"
    )


@pytest.mark.parametrize(
    "left,right,seed,controls",
    [("S", "SS", 42, 1), ("", "", 42, 1), ("S", "S", -1, 1), ("S", "S", 42, -1)],
)
def test_selection_rejects_invalid_bounds(left, right, seed, controls) -> None:
    with pytest.raises(ValueError):
        diagnostic.select_cases(frame(left), frame(right), seed, controls)


def test_collision_first_outcome() -> None:
    assert diagnostic.outcome(pd.Series({"success": True, "collision": True})) == "collision"


def metrics() -> EpisodeMetrics:
    return EpisodeMetrics(1.25, 2, True, False, 0.4, math.inf, 0, 0)


def test_reference_comparison_preserves_exact_counts_and_small_float_tolerance() -> None:
    observed = metrics()
    expected = pd.Series(observed.as_dict())
    assert diagnostic.verify_episode(observed, expected) == []
    expected["reward"] += 5e-13
    assert diagnostic.verify_episode(observed, expected) == []
    expected["length"] += 5e-13
    assert diagnostic.verify_episode(observed, expected) == ["length"]
    expected["reward"] += 2e-12
    expected["success"] = False
    assert diagnostic.verify_episode(observed, expected) == ["reward", "length", "success"]


def test_reference_nan_does_not_match() -> None:
    observed = metrics()
    observed.min_ttc = math.nan
    expected = pd.Series(observed.as_dict())
    assert diagnostic.verify_episode(observed, expected) == ["min_ttc"]


def record(action="IDLE", speed=0.0, before=0.0, after=0.0, risk=False, obs="same") -> dict:
    return dict(
        action=action,
        speed=speed,
        target_speed_before=before,
        target_speed_after=after,
        cpa_risk=risk,
        observation_sha256=obs,
    )


def test_trace_summary_separates_idle_target_and_stopped_streaks() -> None:
    trace = [
        record(speed=0.3),
        record(speed=-0.3, risk=True),
        record("FASTER", speed=2, before=4.5, after=9, risk=True),
        record("SLOWER", speed=0.4, before=9, after=4.5),
        record(speed=1, before=9, after=9),
    ]
    result = diagnostic.summarize_trace(trace, 1.1, 0.5, 2)
    assert result["action_counts"] == {"IDLE": 3, "FASTER": 1, "SLOWER": 1}
    assert result["risk_action_counts"] == {"IDLE": 1, "FASTER": 1}
    assert result["stopped_steps"] == 3
    assert result["longest_stopped_run_steps"] == 2
    assert result["low_speed_without_cpa_flag_steps"] == 2
    assert result["target_increase_steps"] == result["target_decrease_steps"] == 1
    assert result["terminal_window"]["decision_count"] == 3
    assert result["terminal_window"]["longest_stopped_run_steps"] == 1


@pytest.mark.parametrize(
    "v1,v2,action_step,obs_step,prefix",
    [
        ([record(), record()], [record(), record()], None, None, True),
        ([record(), record(obs="left")], [record("FASTER"), record(obs="right")], 0, 1, True),
        ([record(), record(obs="left")], [record(), record("FASTER", obs="right")], 1, 1, False),
        ([record(obs="left")], [record(obs="right")], None, 0, False),
        ([], [], None, None, False),
    ],
)
def test_trace_comparison_detects_action_and_observation_order(
    v1, v2, action_step, obs_step, prefix
) -> None:
    result = diagnostic.compare_traces(v1, v2)
    assert result["first_action_divergence_step"] == action_step
    assert result["first_observation_divergence_step"] == obs_step
    assert result["common_observation_prefix_until_action_divergence"] is prefix
    assert result["overlap_steps"] == min(len(v1), len(v2))


def test_save_report_is_exclusive_and_rejects_nan_before_file_creation(tmp_path) -> None:
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        diagnostic.save_report(path, {"speed": math.nan})
    assert not path.exists()
    diagnostic.save_report(path, {"original": True})
    with pytest.raises(FileExistsError):
        diagnostic.save_report(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


class StubEnv:
    def __init__(self):
        self.unwrapped = self
        self.config = {"policy_frequency": 5}
        self.action_type = SimpleNamespace(actions={0: "SLOWER", 1: "IDLE", 2: "FASTER"})
        self.reset(seed=7)

    def reset(self, *, seed):
        self.np_random = np.random.default_rng(seed)
        self.vehicle = SimpleNamespace(
            speed=1.0,
            target_speed=4.5,
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            lane_index=("o0", "ir0", 0),
            target_lane_index=("o0", "ir0", 0),
            route=[("o0", "ir0", 0)],
            lane=SimpleNamespace(length=100.0, local_coordinates=lambda pos: (pos[0], pos[1])),
            crashed=False,
        )
        self.road = SimpleNamespace(vehicles=[self.vehicle])
        self.actions = []
        return np.array([1.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(int(action))
        self.vehicle.speed += 0.5
        self.vehicle.position[0] += 0.3
        if int(action) == 2:
            self.vehicle.target_speed = 9.0
        done = len(self.actions) == 2
        return (
            np.array([1.0, 0.0], dtype=np.float32),
            0.625,
            done,
            False,
            {
                "route_progress": len(self.actions) / 2,
                "is_success": done,
            },
        )


DIAGNOSTIC = dict(
    cpa_horizon=3.0,
    cpa_max_range=60.0,
    cpa_time_threshold=2.0,
    cpa_distance_threshold=3.0,
    stopped_speed=0.5,
    terminal_window_s=2.0,
)


def test_snapshot_preserves_observation_route_vehicle_and_rng() -> None:
    env = StubEnv()
    observation = np.array([1.0, 0.0], dtype=np.float32)
    rng_state = copy.deepcopy(env.np_random.bit_generator.state)
    route = copy.deepcopy(env.vehicle.route)
    position = env.vehicle.position.copy()
    original_observation = observation.copy()
    snapshot = diagnostic._capture_decision(env, observation, 1, 0, math.inf, DIAGNOSTIC)
    assert env.np_random.bit_generator.state == rng_state
    assert env.vehicle.route == route
    np.testing.assert_array_equal(env.vehicle.position, position)
    np.testing.assert_array_equal(observation, original_observation)
    assert snapshot["target_speed_before"] == env.vehicle.target_speed == 4.5
    assert snapshot["cpa_risk"] is False
    assert snapshot["closest_cpa_time_s"] is None


@pytest.mark.parametrize("bad_observation", [True, False])
def test_nonfinite_trace_inputs_fail(bad_observation) -> None:
    env = StubEnv()
    observation = np.array([1.0, 0.0])
    if bad_observation:
        observation[0] = math.nan
    else:
        env.vehicle.speed = math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        diagnostic._capture_decision(env, observation, 1, 0, math.inf, DIAGNOSTIC)


def test_replay_forwards_one_deterministic_action_per_step_and_matches_metrics() -> None:
    env = StubEnv()
    calls = []

    def predict(observation, *, deterministic):
        calls.append((observation.copy(), deterministic))
        return np.array(1 if len(calls) == 1 else 2), None

    observed, trace = diagnostic.replay_episode(
        SimpleNamespace(predict=predict), env, 7, 2.0, DIAGNOSTIC
    )
    assert diagnostic.verify_episode(observed, pd.Series(metrics().as_dict())) == []
    assert env.actions == [1, 2]
    assert len(calls) == len(trace) == 2 and all(call[1] is True for call in calls)
    assert trace[0]["action"] == "IDLE"
    assert trace[0]["speed_after"] > trace[0]["speed"]
    assert trace[0]["target_speed_after"] == trace[0]["target_speed_before"]
    assert trace[1]["target_speed_after"] == 9
    assert diagnostic._compact_trace(trace)[0][diagnostic.TRACE_COLUMNS.index("radial_ttc_s")] is (
        None
    )


def protocol() -> dict:
    return json.loads(Path("configs/fusion_timing_v1.json").read_text())


@pytest.mark.parametrize(
    "field,value", [("episodes", 0), ("first_seed", -1), ("controls_per_group", True)]
)
def test_protocol_invalid_counts_fail(field, value) -> None:
    data = protocol()
    data[field] = value
    with pytest.raises(ValueError, match="integer"):
        diagnostic.validate_protocol(data)


def test_protocol_invalid_geometry_fails() -> None:
    data = protocol()
    data["diagnostic"]["cpa_time_threshold"] = 4
    with pytest.raises(ValueError, match="horizon"):
        diagnostic.validate_protocol(data)


def test_run_preserves_integrity_failure_and_existing_output(tmp_path, monkeypatch) -> None:
    data = protocol()
    monkeypatch.setattr(
        diagnostic,
        "_require_hash",
        lambda *args: (_ for _ in ()).throw(ValueError("Fingerprint mismatch")),
    )
    output = tmp_path / "failure.json"
    with pytest.raises(RuntimeError, match="preserve"):
        diagnostic.run_diagnostic(data, output, "frozen")
    saved = output.read_bytes()
    report = json.loads(saved)
    assert report["reference_reproduced"] is False
    assert report["cases"] == []
    assert report["failure"]["message"] == "Fingerprint mismatch"
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(data, output, "frozen")
    assert output.read_bytes() == saved


def test_main_preserves_protocol_loading_failure(tmp_path, monkeypatch) -> None:
    output = tmp_path / "failure.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnostic",
            "--protocol",
            str(tmp_path / "missing"),
            "--protocol-sha256",
            "0" * 64,
            "--output",
            str(output),
        ],
    )
    with pytest.raises(RuntimeError, match="Protocol loading failed"):
        diagnostic.main()
    report = json.loads(output.read_text())
    assert report["phase"] == "protocol_loading" and report["reference_reproduced"] is False


@pytest.mark.parametrize("failure_kind", [None, "metrics", "cleanup", "both", "prefix"])
def test_paired_runner_saves_success_or_failing_case_and_closes_envs(
    tmp_path, monkeypatch, failure_kind
) -> None:
    data = protocol()
    data.update(
        episodes=1,
        controls_per_group=1,
        expected_selected_transition_counts={"success_to_success": 1},
    )
    monkeypatch.setattr(diagnostic, "_require_hash", lambda *args: None)

    def read_reference(spec, _protocol):
        row = metrics().as_dict()
        if failure_kind in {"metrics", "both"} and spec is data["policies"]["v2"]:
            row["reward"] += 1
        return pd.DataFrame([row])

    monkeypatch.setattr(diagnostic, "_read_reference", read_reference)
    model = SimpleNamespace(
        observation_space="obs",
        action_space="actions",
        device="cpu",
        predict=lambda observation, deterministic: (np.array(1), None),
    )
    monkeypatch.setattr(diagnostic.PPO, "load", lambda *args, **kwargs: model)
    created = []
    closed = []

    def make_env(**kwargs):
        env = StubEnv()
        env.observation_space = "obs"
        env.action_space = "actions"
        env.tag = len(created)

        def close():
            closed.append(env.tag)
            if failure_kind in {"cleanup", "both"} and env.tag == 0:
                raise RuntimeError("close failure")

        env.close = close
        created.append(env)
        return env

    monkeypatch.setattr(diagnostic, "make_intersection_env", make_env)
    capture = diagnostic._capture_decision

    def snapshot(env, *args):
        row = capture(env, *args)
        if failure_kind == "prefix" and env.tag == 1:
            row["observation_sha256"] = "different"
        return row

    monkeypatch.setattr(diagnostic, "_capture_decision", snapshot)
    output = tmp_path / "pair.json"
    if failure_kind is None:
        diagnostic.run_diagnostic(data, output, "frozen")
    else:
        with pytest.raises(RuntimeError, match="preserve"):
            diagnostic.run_diagnostic(data, output, "frozen")
    report = json.loads(output.read_text())
    assert closed == [0, 1]
    assert report["reference_reproduced"] is (failure_kind is None)
    case = report["cases"][0]
    assert case["seed"] == data["first_seed"]
    if failure_kind is None:
        assert report["verified_episode_count"] == 2
        assert case["comparison"]["shared_initial_observation"] is True
    if failure_kind in {"metrics", "both"}:
        assert case["policies"]["v2"]["mismatched_fields"] == ["reward"]
        assert "Reference mismatch for v2" in report["failure"]["message"]
    if failure_kind in {"cleanup", "both"}:
        assert report["cleanup_errors"] == {"v1": "close failure"}
    if failure_kind == "prefix":
        assert case["comparison"]["shared_initial_observation"] is False
