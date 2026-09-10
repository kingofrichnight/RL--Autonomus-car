import copy
import json

import numpy as np
import pytest
from highway_env.vehicle.kinematics import Vehicle

from safeintent_rl.config import load_config
from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.sensors.predictive import PredictiveSafetyObservation, rectangle_separation

CONFIG = "configs/intersection_v3_predictive_safety_v1.yaml"
CONTROL = "configs/intersection_reward_v3_collision_first.yaml"


def test_config_preserves_v3_dynamics_and_coefficients():
    new = load_config(CONFIG)
    assert new.pop("predictive_safety_observation")["horizon"] == 3.0
    assert new == load_config(CONTROL)


@pytest.mark.parametrize("position,heading,expected", [
    ([10, 0], 0, 5), ([0, 4], 0, 2), ([5, 0], 0, 0),
    ([0, 0], 0, -2), ([0, 4], np.pi / 2, 0.5),
])
def test_rectangle_geometry(position, heading, expected):
    gap = rectangle_separation(np.zeros(2), 0, (5, 2), np.array([position]),
                               np.array([heading]), np.array([[5, 2]]))
    assert gap[0] == pytest.approx(expected)


@pytest.mark.parametrize("option,value", [
    ("horizon", 0), ("horizon", 0.01), ("horizon", float("nan")),
    ("margin", -1), ("uncertainty_growth", float("inf")),
    ("clearance_scale", 0), ("speed_scale", -1),
    ("max_neighbors", 15), ("max_neighbors", True),
])
def test_invalid_parameters(option, value):
    base = make_intersection_env(CONTROL)
    try:
        with pytest.raises(ValueError):
            PredictiveSafetyObservation(base, **{option: value})
    finally:
        base.close()


def test_prediction_is_read_only_and_preserves_entire_action_rollout():
    plain = make_intersection_env(CONTROL)
    wrapped = make_intersection_env(CONFIG)
    try:
        a, _ = plain.reset(seed=7)
        b, _ = wrapped.reset(seed=7)
        assert b.shape == (115,)
        assert wrapped.observation_space.contains(b)
        np.testing.assert_array_equal(a.reshape(-1), b[:105])
        base = wrapped.unwrapped
        rng = json.dumps(base.np_random.bit_generator.state, sort_keys=True)
        positions = np.array([v.position.copy() for v in base.road.vehicles])
        routes = copy.deepcopy([getattr(v, "route", None) for v in base.road.vehicles])
        f = wrapped.forecast()
        np.testing.assert_array_equal(f, wrapped.forecast())
        np.testing.assert_array_equal(positions, [v.position for v in base.road.vehicles])
        assert routes == [getattr(v, "route", None) for v in base.road.vehicles]
        assert rng == json.dumps(base.np_random.bit_generator.state, sort_keys=True)
        for action in [2, 1, 0, 1] * 5:
            x, y = plain.step(action), wrapped.step(action)
            np.testing.assert_array_equal(x[0].reshape(-1), y[0][:105])
            assert x[1:4] == y[1:4]
            assert wrapped.observation_space.contains(y[0])
            if x[2] or x[3]:
                break
    finally:
        plain.close()
        wrapped.close()


def test_action_forecasts_distinguish_braking_from_acceleration():
    env = make_intersection_env(CONFIG)
    try:
        env.reset(seed=7)
        base = env.unwrapped
        ego = base.vehicle
        ego.speed = ego.target_speed = 4.5
        direction = np.array([np.cos(ego.heading), np.sin(ego.heading)])
        obstacle = Vehicle(base.road, ego.position + 12 * direction,
                           heading=ego.heading, speed=0)
        base.road.vehicles = [ego, obstacle]
        forecasts = env.forecast()
        assert forecasts[0, 0] > forecasts[2, 0]
        assert forecasts[0, 1] > forecasts[2, 1]
        assert forecasts[0, 2] < forecasts[2, 2]
        assert forecasts[2, 0] < 0
    finally:
        env.close()


def test_empty_scene_has_no_predicted_conflict():
    env = make_intersection_env(CONFIG)
    try:
        env.reset(seed=7)
        env.unwrapped.road.vehicles = [env.unwrapped.vehicle]
        np.testing.assert_array_equal(env.forecast()[:, :2], np.ones((3, 2)))
        with pytest.raises(ValueError, match="overrides"):
            env.reset(options={"config": {"duration": 1}})
    finally:
        env.close()


@pytest.mark.parametrize("extra", ["risk_fusion", "target_speed_observation", "safety_shield",
                                   "cpa_safety_shield"])
def test_protocol_rejects_unplanned_combinations(extra):
    with pytest.raises(ValueError, match="without other"):
        make_intersection_env(CONFIG, **{extra: True})


def test_other_hidden_behavior_and_routes_do_not_enter_forecast():
    env = make_intersection_env(CONFIG)
    try:
        env.reset(seed=7)
        before = env.forecast()
        for actor in env.unwrapped.road.vehicles:
            if actor is not env.unwrapped.vehicle:
                actor.route = None
                actor.ACC_MAX = 12345
                actor.intent_label = "unobserved test label"
        np.testing.assert_array_equal(before, env.forecast())
    finally:
        env.close()
