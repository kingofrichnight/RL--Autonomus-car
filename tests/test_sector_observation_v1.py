"""Engineering checks for the same-actor, derived sector observation ablation."""

import copy
import math

import gymnasium as gym
import numpy as np
import pytest

from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
from scripts.sector_observation_v1 import (
    CELLS,
    CLOSING_SPEED_SCALE,
    INPUT_SIZE,
    OUTPUT_SIZE,
    PROTOCOL,
    RANGE_SCALE,
    SectorPredictiveObservation,
    sector_features,
)

CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"


def observation(yaw=0.0):
    result = np.zeros(INPUT_SIZE, dtype=np.float32)
    result[:7] = [1, 0, 0, 0, 0, math.cos(yaw), math.sin(yaw)]
    result[105:] = np.linspace(-0.5, 0.5, INPUT_SIZE - 105, dtype=np.float32)
    return result


def actor(obs, row, position, velocity=(0.0, 0.0)):
    obs[row * 7:row * 7 + 7] = [
        1, position[0] / 200, position[1] / 200,
        velocity[0] / 80, velocity[1] / 80, 1, 0,
    ]


def test_frozen_dimensions_scales_and_empty_encoding():
    assert CELLS == 16 and INPUT_SIZE == 115 and OUTPUT_SIZE == 163
    assert PROTOCOL == "predictive_post_spawn_sector_v1"
    assert RANGE_SCALE == 200 * math.sqrt(2)
    assert CLOSING_SPEED_SCALE == 80 * math.sqrt(2)
    features = sector_features(observation())
    assert features.shape == (16, 3)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(features, np.tile([0, 1, 0], (16, 1)))


def test_cardinal_sectors_range_and_approach_recede_sign():
    obs = observation()
    actor(obs, 1, (20, 0), (-10, 0))
    actor(obs, 2, (0, 40), (0, 10))
    actor(obs, 3, (-60, 0), (0, 10))
    actor(obs, 4, (0, -80), (0, 0))
    features = sector_features(obs)
    np.testing.assert_array_equal(np.flatnonzero(features[:, 0]), [0, 4, 8, 12])
    np.testing.assert_allclose(
        features[[0, 4, 8, 12], 1], np.array([20, 40, 60, 80]) / RANGE_SCALE,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        features[[0, 4, 8, 12], 2], np.array([10, -10, 0, 0]) / CLOSING_SPEED_SCALE,
        rtol=1e-6, atol=1e-7,
    )


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(-math.pi / 16 - 1e-4, 15), (-math.pi / 16 + 1e-4, 0),
     (math.pi / 16 - 1e-4, 0), (math.pi / 16 + 1e-4, 1)],
)
def test_forward_sector_is_centered_and_wraps(angle, expected):
    obs = observation()
    actor(obs, 1, (20 * math.cos(angle), 20 * math.sin(angle)))
    np.testing.assert_array_equal(np.flatnonzero(sector_features(obs)[:, 0]), [expected])


def test_joint_rotation_preserves_ego_aligned_features():
    before = observation()
    after = observation(math.pi / 2)
    for row, position, velocity in [
        (1, (20, 10), (-10, 3)), (2, (-40, 20), (2, -5)),
        (3, (0, -30), (0, 4)),
    ]:
        actor(before, row, position, velocity)
        actor(after, row, (-position[1], position[0]), (-velocity[1], velocity[0]))
    np.testing.assert_allclose(sector_features(before), sector_features(after), atol=1e-7)


def test_nearest_actor_wins_and_exact_tie_keeps_first_native_row():
    obs = observation()
    actor(obs, 1, (20, 0), (-5, 0))
    actor(obs, 2, (10, 0), (-3, 0))
    actor(obs, 3, (10, 0), (4, 0))
    np.testing.assert_allclose(
        sector_features(obs)[0], [1, 10 / RANGE_SCALE, 3 / CLOSING_SPEED_SCALE],
        rtol=1e-6,
    )


def test_colocated_actor_is_present_in_forward_sector_with_zero_closing():
    obs = observation(math.pi)
    actor(obs, 1, (0, 0), (80, -80))
    features = sector_features(obs)
    np.testing.assert_array_equal(features[0], [1, 0, 0])
    np.testing.assert_array_equal(features[1:], np.tile([0, 1, 0], (15, 1)))


def test_extreme_valid_geometry_fits_normalized_ranges():
    obs = observation()
    actor(obs, 1, (200, 200), (-80, -80))
    actor(obs, 2, (-200, -200), (-80, -80))
    features = sector_features(obs)
    np.testing.assert_allclose(features[2], [1, 1, 1], atol=1e-7)
    np.testing.assert_allclose(features[10], [1, 1, -1], atol=1e-7)
    assert np.all(np.abs(features) <= 1)


def test_absent_padding_is_ignored_and_read_only_input_is_not_mutated():
    obs = observation()
    actor(obs, 1, (15, -7), (3, -2))
    expected = sector_features(obs)
    obs[14:105].reshape(13, 7)[:, 1:] = 12345
    snapshot = obs.copy()
    obs.flags.writeable = False
    np.testing.assert_array_equal(sector_features(obs), expected)
    np.testing.assert_array_equal(obs, snapshot)


@pytest.mark.parametrize("shape", [(114,), (116,), (1, 115), (115, 1)])
def test_rejects_wrong_input_shape(shape):
    with pytest.raises(ValueError):
        sector_features(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize(
    ("index", "value"),
    [(105, np.nan), (110, np.inf), (0, 0), (7, 0.5), (14, -1),
     (5, 2), (6, 1), (8, 1.001), (9, -1.001), (10, 1.001), (11, -1.001)],
)
def test_rejects_nonfinite_or_incompatible_present_input(index, value):
    obs = observation()
    actor(obs, 1, (20, 0))
    obs[index] = value
    with pytest.raises(ValueError):
        sector_features(obs)


def assert_same_scene(left, right):
    a, b = left.unwrapped, right.unwrapped
    assert a.time == b.time and a.steps == b.steps
    assert a.np_random.bit_generator.state == b.np_random.bit_generator.state
    assert len(a.road.vehicles) == len(b.road.vehicles)
    for x, y in zip(a.road.vehicles, b.road.vehicles, strict=True):
        np.testing.assert_array_equal(x.position, y.position)
        np.testing.assert_array_equal(x.velocity, y.velocity)
        for field in (
            "heading", "speed", "crashed", "lane_index", "action", "route",
            "target_speed", "safeintent_driver_label",
        ):
            assert getattr(x, field, None) == getattr(y, field, None)


@pytest.mark.parametrize("seed", [7, 42, 80042])
def test_short_rollout_preserves_prefix_scene_rewards_rng_and_actions(seed):
    control = SynchronizedPredictiveObservation(make_intersection_env(CONFIG))
    candidate = SectorPredictiveObservation(
        SynchronizedPredictiveObservation(make_intersection_env(CONFIG))
    )
    try:
        a, ia = control.reset(seed=seed)
        b, ib = candidate.reset(seed=seed)
        ia.pop("action")  # Native unexecuted random reset placeholder is not a policy action.
        ib.pop("action")
        assert ia == ib
        assert candidate.action_space == control.action_space
        assert candidate.observation_space.shape == (OUTPUT_SIZE,)
        np.testing.assert_array_equal(a, b[:INPUT_SIZE])
        assert candidate.observation_space.contains(b)
        for action in [2, 1, 0, 1] * 3:
            a, *left = control.step(action)
            b, *right = candidate.step(action)
            assert left == right
            assert_same_scene(control, candidate)
            np.testing.assert_array_equal(a, b[:INPUT_SIZE])
            np.testing.assert_array_equal(sector_features(a).reshape(-1), b[INPUT_SIZE:])
            assert candidate.observation_space.contains(b)
            if left[1] or left[2]:
                break
    finally:
        control.close()
        candidate.close()


def test_augmentation_uses_no_extra_observation_forecast_or_rng(monkeypatch):
    sync = SynchronizedPredictiveObservation(make_intersection_env(CONFIG))
    env = SectorPredictiveObservation(sync)
    try:
        obs, _ = sync.reset(seed=7)
        snapshot = obs.copy()
        rng = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)

        def forbidden(*args, **kwargs):
            pytest.fail("Sector projection must not observe, forecast, step, or sample again")

        monkeypatch.setattr(env.unwrapped.observation_type, "observe", forbidden)
        monkeypatch.setattr(sync.env, "forecast", forbidden)
        monkeypatch.setattr(sync, "step", forbidden)
        first = env._augment(obs)
        repeated = env._augment(obs)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(obs, snapshot)
        np.testing.assert_array_equal(first[:INPUT_SIZE], obs)
        assert env.unwrapped.np_random.bit_generator.state == rng
    finally:
        env.close()


def test_requires_direct_synchronized_parent():
    raw = make_intersection_env(CONFIG)
    indirect = gym.Wrapper(SynchronizedPredictiveObservation(make_intersection_env(CONFIG)))
    try:
        for invalid in (raw, indirect):
            with pytest.raises(TypeError):
                SectorPredictiveObservation(invalid)
    finally:
        raw.close()
        indirect.close()


def test_rejects_wrong_geometry_at_construction_and_during_augmentation():
    sync = SynchronizedPredictiveObservation(make_intersection_env(CONFIG))
    try:
        obs, _ = sync.reset(seed=7)
        env = SectorPredictiveObservation(sync)
        sync.unwrapped.observation_type.features_range["x"] = [-100, 100]
        with pytest.raises(ValueError):
            env._augment(obs)
        with pytest.raises(ValueError):
            SectorPredictiveObservation(sync)
    finally:
        sync.close()


def test_native_reset_cannot_replace_the_frozen_geometry_contract():
    sync = SynchronizedPredictiveObservation(make_intersection_env(CONFIG))
    env = SectorPredictiveObservation(sync)
    try:
        env.unwrapped.config["observation"]["features_range"]["y"] = [-4, 4]
        with pytest.raises(ValueError):
            env.reset(seed=7)
    finally:
        env.close()
