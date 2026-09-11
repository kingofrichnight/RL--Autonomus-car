"""Geometry v2 changes only y scaling; it is not a new forecast or reward."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from safeintent_rl.config import load_config
from safeintent_rl.envs.intersection import make_intersection_env

LEGACY = "configs/intersection_v3_predictive_safety_v1.yaml"
GEOMETRY = "configs/intersection_v3_predictive_geometry_v2.yaml"
RANGES = {"x": [-200, 200], "y": [-200, 200], "vx": [-80, 80], "vy": [-80, 80]}
UNCHANGED_COLUMNS = [0, 1, 3, 4, 5, 6]


def test_geometry_config_adds_only_explicit_normalization_ranges():
    legacy = load_config(LEGACY)
    geometry = load_config(GEOMETRY)
    assert "features_range" not in legacy["observation"]
    assert geometry["observation"].pop("features_range") == RANGES
    assert geometry == legacy


def test_effective_geometry_contract_has_115_inputs_and_only_y_range_changes():
    legacy = make_intersection_env(LEGACY)
    geometry = make_intersection_env(GEOMETRY)
    try:
        old_obs, _ = legacy.reset(seed=80042)
        new_obs, _ = geometry.reset(seed=80042)
        old_type = legacy.unwrapped.observation_type
        new_type = geometry.unwrapped.observation_type
        assert old_type.features_range == dict(RANGES, y=[-4, 4])
        assert new_type.features_range == RANGES
        assert old_type.clip is new_type.clip is True
        assert old_type.see_behind is new_type.see_behind is False
        assert old_type.absolute is new_type.absolute is False
        assert old_type.normalize is new_type.normalize is True
        assert old_type.order == new_type.order == "sorted"
        assert old_type.vehicles_count == new_type.vehicles_count == 15
        assert old_type.features == new_type.features == [
            "presence", "x", "y", "vx", "vy", "cos_h", "sin_h",
        ]
        assert old_obs.shape == new_obs.shape == (115,)
        assert old_obs.dtype == new_obs.dtype == np.float32
        assert legacy.observation_space == geometry.observation_space
        assert legacy.action_space == geometry.action_space
        assert legacy.actions == geometry.actions == ("SLOWER", "IDLE", "FASTER")
        effective_new = copy.deepcopy(geometry.unwrapped.config)
        assert effective_new["observation"].pop("features_range") == RANGES
        assert effective_new == legacy.unwrapped.config
    finally:
        legacy.close()
        geometry.close()


def test_native_normalizer_distinguishes_cross_traffic_distances_without_other_changes():
    legacy = make_intersection_env(LEGACY)
    geometry = make_intersection_env(GEOMETRY)
    try:
        legacy.reset(seed=80042)
        geometry.reset(seed=80042)
        # Exercise the actual HighwayEnv normalizer, not a reimplemented formula.
        raw = pd.DataFrame({
            "presence": [1.0] * 6,
            "x": [-250.0, -100.0, -10.0, 10.0, 100.0, 250.0],
            "y": [-250.0, -50.0, -10.0, 10.0, 50.0, 250.0],
            "vx": [-100.0, -40.0, -10.0, 10.0, 40.0, 100.0],
            "vy": [100.0, 40.0, 10.0, -10.0, -40.0, -100.0],
            "cos_h": [0.0, 1.0, -1.0, 0.0, 1.0, -1.0],
            "sin_h": [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        })
        old = legacy.unwrapped.observation_type.normalize_obs(raw.copy())
        new = geometry.unwrapped.observation_type.normalize_obs(raw.copy())
        np.testing.assert_array_equal(old["y"], [-1, -1, -1, 1, 1, 1])
        np.testing.assert_allclose(new["y"], [-1, -0.25, -0.05, 0.05, 0.25, 1])
        pd.testing.assert_frame_equal(old.drop(columns="y"), new.drop(columns="y"))
        # Presence and headings have no new ranges and remain untransformed.
        pd.testing.assert_frame_equal(
            new[["presence", "cos_h", "sin_h"]], raw[["presence", "cos_h", "sin_h"]],
        )
    finally:
        legacy.close()
        geometry.close()


def _assert_equal_dynamics(legacy, geometry):
    old = legacy.unwrapped
    new = geometry.unwrapped
    assert old.time == new.time
    assert old.steps == new.steps
    assert json.dumps(old.np_random.bit_generator.state, sort_keys=True) == json.dumps(
        new.np_random.bit_generator.state, sort_keys=True,
    )
    assert len(old.road.vehicles) == len(new.road.vehicles)
    for left, right in zip(old.road.vehicles, new.road.vehicles, strict=True):
        np.testing.assert_array_equal(left.position, right.position)
        np.testing.assert_array_equal(left.velocity, right.velocity)
        assert left.heading == right.heading
        assert left.speed == right.speed
        assert left.crashed == right.crashed
        assert left.lane_index == right.lane_index
        assert left.action == right.action
        for name in ("route", "target_speed", "speed_index", "safeintent_driver_label",
                     "ACC_MAX", "COMFORT_ACC_MAX", "DISTANCE_WANTED", "TIME_WANTED"):
            assert getattr(left, name, None) == getattr(right, name, None)


@pytest.mark.parametrize("seed", [7, 42, 80042])
def test_geometry_preserves_seeded_dynamics_reward_and_all_non_y_inputs(seed):
    legacy = make_intersection_env(LEGACY)
    geometry = make_intersection_env(GEOMETRY)
    try:
        old_obs, old_info = legacy.reset(seed=seed)
        new_obs, new_info = geometry.reset(seed=seed)
        # Native reset reports an independently sampled, unexecuted action from
        # its newly reconstructed space. This placeholder is not simulator RNG
        # or a policy action; all subsequently executed action info must match.
        assert legacy.action_space.contains(old_info.pop("action"))
        assert geometry.action_space.contains(new_info.pop("action"))
        assert old_info == new_info
        _assert_equal_dynamics(legacy, geometry)
        saw_changed_y = False
        for action in [2, 1, 0, 1] * 40:
            old_matrix = old_obs[:105].reshape(15, 7)
            new_matrix = new_obs[:105].reshape(15, 7)
            np.testing.assert_array_equal(
                old_matrix[:, UNCHANGED_COLUMNS], new_matrix[:, UNCHANGED_COLUMNS],
            )
            # The original target-speed scalar and all nine forecasts are identical.
            np.testing.assert_array_equal(old_obs[105:], new_obs[105:])
            assert geometry.observation_space.contains(new_obs)
            saw_changed_y |= not np.array_equal(old_matrix[:, 2], new_matrix[:, 2])
            left = legacy.step(action)
            right = geometry.step(action)
            assert left[1:] == right[1:]
            _assert_equal_dynamics(legacy, geometry)
            old_obs, new_obs = left[0], right[0]
            if left[2] or left[3]:
                np.testing.assert_array_equal(
                    old_obs[:105].reshape(15, 7)[:, UNCHANGED_COLUMNS],
                    new_obs[:105].reshape(15, 7)[:, UNCHANGED_COLUMNS],
                )
                assert geometry.observation_space.contains(new_obs)
                np.testing.assert_array_equal(old_obs[105:], new_obs[105:])
                break
        else:
            pytest.fail("Fixed-action contract rollout did not terminate within the deadline")
        assert saw_changed_y, "The rollout must actually exercise the geometry change"
    finally:
        legacy.close()
        geometry.close()
