import copy

import numpy as np
import pytest
from highway_env.vehicle.kinematics import Vehicle

from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation

CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"


def assert_same_scene(left, right):
    a, b = left.unwrapped, right.unwrapped
    assert a.time == b.time and a.steps == b.steps
    assert a.np_random.bit_generator.state == b.np_random.bit_generator.state
    assert len(a.road.vehicles) == len(b.road.vehicles)
    for x, y in zip(a.road.vehicles, b.road.vehicles, strict=True):
        np.testing.assert_array_equal(x.position, y.position)
        np.testing.assert_array_equal(x.velocity, y.velocity)
        for field in ("heading", "speed", "crashed", "lane_index", "action", "route",
                      "target_speed", "safeintent_driver_label"):
            assert getattr(x, field, None) == getattr(y, field, None)


@pytest.mark.parametrize("seed", [7, 42, 80042])
def test_complete_rollout_preserves_dynamics_rewards_rng_and_forecasts(seed):
    old = make_intersection_env(CONFIG)
    new = SynchronizedPredictiveObservation(make_intersection_env(CONFIG))
    try:
        a, ia = old.reset(seed=seed)
        b, ib = new.reset(seed=seed)
        ia.pop("action")  # Unexecuted random native reset placeholder, as in geometry tests.
        ib.pop("action")
        assert ia == ib
        np.testing.assert_array_equal(a, b)
        differences = 0
        for action in [2, 1, 0, 1] * 40:
            a, *left = old.step(action)
            b, *right = new.step(action)
            assert left == right
            assert_same_scene(old, new)
            np.testing.assert_array_equal(a[105:], b[105:])
            np.testing.assert_array_equal(b[:105],
                                          new.unwrapped.observation_type.observe().reshape(-1))
            differences += int(not np.array_equal(a[:105], b[:105]))
            if left[1] or left[2]:
                break
        else:
            pytest.fail("Engineering trajectory did not finish")
        assert differences > 0
    finally:
        old.close()
        new.close()


def test_forced_post_step_spawn_aligns_inputs_without_second_forecast(monkeypatch):
    inner = make_intersection_env(CONFIG)
    env = SynchronizedPredictiveObservation(inner)
    try:
        env.reset(seed=7)
        base = env.unwrapped
        captured = {}
        original_augment = inner._augment

        def capture(observation):
            captured["before"] = original_augment(observation)
            return captured["before"]

        def spawn(*args, **kwargs):
            actor = Vehicle(base.road, base.vehicle.position + 12 * base.vehicle.direction,
                            heading=base.vehicle.heading, speed=0)
            base.road.vehicles.append(actor)

        monkeypatch.setattr(inner, "_augment", capture)
        monkeypatch.setattr(base, "_spawn_vehicle", spawn)
        calls = []
        forecast = inner.forecast

        def counted():
            calls.append(1)
            return forecast()

        monkeypatch.setattr(inner, "forecast", counted)
        observation, *_ = env.step(1)
        assert len(calls) == 1
        assert not np.array_equal(observation[:105], captured["before"][:105])
        np.testing.assert_array_equal(observation[105:], captured["before"][105:])
        rng = copy.deepcopy(base.np_random.bit_generator.state)
        repeated = env._refresh(observation)
        np.testing.assert_array_equal(repeated, observation)
        assert base.np_random.bit_generator.state == rng
        assert len(calls) == 1
    finally:
        env.close()


def test_rejects_incompatible_or_stochastic_observation_contract():
    raw = make_intersection_env("configs/intersection_reward_v3.yaml")
    inner = make_intersection_env(CONFIG)
    try:
        with pytest.raises(TypeError, match="directly"):
            SynchronizedPredictiveObservation(raw)
        inner.unwrapped.observation_type.order = "shuffled"
        with pytest.raises(ValueError, match="contract"):
            SynchronizedPredictiveObservation(inner)
    finally:
        raw.close()
        inner.close()
