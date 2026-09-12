import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from highway_env.envs.common.observation import KinematicObservation
from highway_env.road.road import Road

from safeintent_rl.sensors.predictive import PredictiveSafetyObservation
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
from scripts import sector_timeout_trace_v1 as trace
from scripts.run_sector_comparison_v1 import PaddingPredictiveObservation
from scripts.sector_observation_v1 import SectorPredictiveObservation


def global_rng_state():
    return random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def restore_global_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def assert_global_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def class_hooks():
    return (KinematicObservation.observe, KinematicObservation.normalize_obs,
            PredictiveSafetyObservation.forecast, Road.close_objects_to)


def assert_same_dynamics(left, right):
    first, second = left.unwrapped, right.unwrapped
    assert first.time == second.time and first.steps == second.steps
    assert first.np_random.bit_generator.state == second.np_random.bit_generator.state
    assert len(first.road.vehicles) == len(second.road.vehicles)
    for a, b in zip(first.road.vehicles, second.road.vehicles, strict=True):
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.velocity, b.velocity)
        for name in (
            "heading", "speed", "target_speed", "crashed", "lane_index", "route", "action",
        ):
            assert getattr(a, name, None) == getattr(b, name, None)


@pytest.mark.parametrize("arm,outer,size", [
    ("geometry", PredictiveSafetyObservation, 115),
    ("padding", PaddingPredictiveObservation, 163),
    ("sector", SectorPredictiveObservation, 163),
])
def test_diagnostic_factory_preserves_exact_historical_wrapper_and_action_contract(
        arm, outer, size):
    env = trace.make_diagnostic_env(arm, bootstrap_seed=7)
    try:
        assert type(env) is outer
        if arm != "geometry":
            assert type(env.env) is SynchronizedPredictiveObservation
            assert type(env.env.env) is PredictiveSafetyObservation
        assert env.observation_space.shape == (size,)
        assert env.action_space.n == 3
        observation, _ = env.reset(seed=7)
        assert observation.shape == (size,) and observation.dtype == np.float32
        if arm == "padding":
            np.testing.assert_array_equal(observation[115:], np.zeros(48, dtype=np.float32))
    finally:
        env.close()


@pytest.mark.parametrize("arm,native_calls", [("geometry", 1), ("padding", 2), ("sector", 2)])
def test_capture_preserves_short_fixed_action_rollout_and_all_rng_streams(arm, native_calls):
    plain = trace.make_diagnostic_env(arm, bootstrap_seed=7)
    captured = trace.make_diagnostic_env(arm, bootstrap_seed=7)
    before_hooks = class_hooks()
    global_before = global_rng_state()
    try:
        plain.reset(seed=7)
        with trace.ObservationCapture(captured) as capture:
            capture.reset_frame(new_episode=True)
            observation, info = captured.reset(seed=7)
            reset_frame = capture.frame(observation, info)
            np.testing.assert_array_equal(reset_frame["policy_observation"], observation)
            assert_same_dynamics(plain, captured)
            for action in [2, 1, 0]:
                shared_rng = global_rng_state()
                expected, *expected_outcome = plain.step(action)
                expected_rng = global_rng_state()
                restore_global_rng(shared_rng)
                capture.reset_frame()
                actual, *actual_outcome = captured.step(action)
                frame = capture.frame(actual, actual_outcome[-1])
                assert_global_rng_equal(global_rng_state(), expected_rng)
                np.testing.assert_array_equal(actual, expected)
                assert actual_outcome == expected_outcome
                assert_same_dynamics(plain, captured)
                assert len(frame["native_events"]) == native_calls
                assert len(frame["forecast_events"]) == 1
                np.testing.assert_array_equal(frame["policy_observation"], actual)
                np.testing.assert_array_equal(
                    np.asarray(frame["forecast_events"][0]["values"]).ravel(), actual[106:115],
                )
                assert frame["state"]["time"] == captured.unwrapped.time
                assert frame["state"]["crashed"] == captured.unwrapped.vehicle.crashed
                if actual_outcome[1] or actual_outcome[2]:
                    break
        assert class_hooks() == before_hooks
    finally:
        restore_global_rng(global_before)
        plain.close()
        captured.close()


def test_capture_hooks_restore_after_exception():
    env = trace.make_diagnostic_env("geometry", bootstrap_seed=7)
    before = class_hooks()
    try:
        with pytest.raises(RuntimeError, match="deliberate"):
            with trace.ObservationCapture(env):
                raise RuntimeError("deliberate capture failure")
        assert class_hooks() == before
    finally:
        env.close()


@pytest.mark.parametrize("arm", ["geometry", "padding", "sector"])
def test_capture_follows_reset_recreated_observer_but_ignores_other_environment(arm):
    env = trace.make_diagnostic_env(arm, bootstrap_seed=7)
    unrelated = trace.make_diagnostic_env(arm, bootstrap_seed=7)
    original_observer = env.unwrapped.observation_type
    try:
        with trace.ObservationCapture(env) as capture:
            capture.reset_frame(new_episode=True)
            observation, info = env.reset(seed=7)
            assert env.unwrapped.observation_type is not original_observer
            expected = capture.frame(observation, info)
            unrelated.reset(seed=7)
            unrelated.step(1)
            repeated = capture.frame(observation, info)
            assert repeated == expected
            assert expected["native_events"] and expected["forecast_events"]
    finally:
        env.close()
        unrelated.close()


def test_capture_distinguishes_actual_raw_clipping_from_normalized_boundaries(monkeypatch):
    env = trace.make_diagnostic_env("geometry", bootstrap_seed=7)
    try:
        env.reset(seed=7)
        original = env.unwrapped.vehicle.to_dict

        def controlled_raw(*args, **kwargs):
            result = original(*args, **kwargs)
            result.update(x=400.0, y=-200.0, vx=-80.0, vy=100.0)
            return result

        monkeypatch.setattr(env.unwrapped.vehicle, "to_dict", controlled_raw)
        with trace.ObservationCapture(env) as capture:
            capture.reset_frame()
            observation, _, _, _, info = env.step(1)
            frame = capture.frame(observation, info)
            event = frame["native_events"][0]
            np.testing.assert_array_equal(event["raw_rows"][0][1:5], [400, -200, -80, 100])
            np.testing.assert_array_equal(event["normalized_rows"][0][1:5], [1, -1, -1, 1])
            np.testing.assert_array_equal(event["clipped_mask"][0][1:5], [True, False, False, True])
            np.testing.assert_array_equal(event["boundary_mask"][0][1:5], [True] * 4)
            assert not event["clipped_mask"][0][0]
            assert event["boundary_mask"][0][0]
            assert np.asarray(event["boundary_mask"]).shape == (15, 7)
            assert np.asarray(event["clipped_mask"]).shape == np.asarray(event["raw_rows"]).shape
    finally:
        env.close()


def test_frame_returns_detached_copies_and_episode_first_sighting_actor_ids():
    env = trace.make_diagnostic_env("sector", bootstrap_seed=7)
    try:
        with trace.ObservationCapture(env) as capture:
            capture.reset_frame(new_episode=True)
            observation, info = env.reset(seed=7)
            expected = capture.frame(observation, info)
            ids = expected["native_events"][-1]["selected_birth_ids"]
            assert ids and len(ids) == len(set(ids))
            assert all(value > 0 for value in ids)
            assert ids == expected["forecast_events"][0]["selected_birth_ids"]
            assert expected["native_forecast_ids_match"]
            actual = capture.frame(observation, info)
            actual["policy_observation"][0] = -123
            actual["native_events"][0]["raw_rows"][0][1] = -123
            actual["forecast_events"][0]["selected_actors"][0]["position"][0] = -123
            assert capture.frame(observation, info) == expected
            capture.reset_frame(new_episode=True)
            observation, info = env.reset(seed=7)
            repeated = capture.frame(observation, info)
            assert repeated["native_events"][-1]["selected_birth_ids"] == ids
    finally:
        env.close()


@pytest.mark.parametrize("arm", ["geometry", "sector"])
@pytest.mark.parametrize("mutation", ["stale", "reordered"])
def test_frame_rejects_stale_or_reordered_capture_events(arm, mutation):
    env = trace.make_diagnostic_env(arm, bootstrap_seed=7)
    try:
        with trace.ObservationCapture(env) as capture:
            capture.reset_frame(new_episode=True)
            observation, info = env.reset(seed=7)
            capture.frame(observation, info)
            if mutation == "stale":
                capture.events[0]["time"] -= 1
            else:
                capture.events[0], capture.events[1] = capture.events[1], capture.events[0]
            with pytest.raises(ValueError):
                capture.frame(observation, info)
    finally:
        env.close()


def test_overlapping_captures_are_rejected_without_disturbing_active_hooks():
    env = trace.make_diagnostic_env("geometry", bootstrap_seed=7)
    other = trace.make_diagnostic_env("sector", bootstrap_seed=7)
    before = class_hooks()
    try:
        with trace.ObservationCapture(env):
            active = class_hooks()
            with pytest.raises(RuntimeError, match="Overlapping"):
                with trace.ObservationCapture(other):
                    pytest.fail("Second capture entered")
            assert class_hooks() == active
        assert class_hooks() == before
        with trace.ObservationCapture(other):
            pass
        assert class_hooks() == before
    finally:
        env.close()
        other.close()


def test_partial_hook_install_failure_restores_hooks_and_releases_capture(monkeypatch):
    env = trace.make_diagnostic_env("geometry", bootstrap_seed=7)
    original = trace.patch.object
    calls = []
    before = class_hooks()

    def fail_third(*args, **kwargs):
        calls.append(args)
        if len(calls) == 3:
            raise RuntimeError("deliberate hook installation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(trace.patch, "object", fail_third)
    try:
        with pytest.raises(RuntimeError, match="deliberate hook installation failure"):
            with trace.ObservationCapture(env):
                pytest.fail("Partial installation entered")
        assert class_hooks() == before
        monkeypatch.setattr(trace.patch, "object", original)
        with trace.ObservationCapture(env):
            pass
        assert class_hooks() == before
    finally:
        env.close()


@pytest.mark.parametrize("arm", ["padding", "sector"])
def test_factory_passes_bootstrap_to_original_before_adding_observation_wrappers(monkeypatch, arm):
    inner = trace.make_diagnostic_env("geometry", bootstrap_seed=7)
    calls = []
    synchronized, outer = object(), object()

    def factory(*, config_path, seed):
        calls.append(("factory", config_path, seed))
        return inner

    def sync(env):
        assert env is inner
        calls.append(("synchronized",))
        return synchronized

    def wrapper(env):
        assert env is synchronized
        calls.append(("outer",))
        return outer

    monkeypatch.setattr(trace, "make_intersection_env", factory)
    monkeypatch.setattr(trace, "SynchronizedPredictiveObservation", sync)
    monkeypatch.setattr(trace, "PaddingPredictiveObservation" if arm == "padding"
                        else "SectorPredictiveObservation", wrapper)
    try:
        assert trace.make_diagnostic_env(arm, "fixed-config", bootstrap_seed=7) is outer
        assert calls == [("factory", "fixed-config", 7), ("synchronized",), ("outer",)]
    finally:
        inner.close()


class PredictOnce:
    def __init__(self, probabilities=(.2, .5, .3), action=1, distribution_calls=1, error=None):
        self.probabilities = probabilities
        self.action = action
        self.distribution_calls = distribution_calls
        self.error = error
        self.predict_calls = []
        self.network_calls = []
        self.policy = SimpleNamespace(get_distribution=self.distribution)
        self.result = (np.array(action), ("preserved-state",))

    def distribution(self, observation):
        self.network_calls.append(observation)
        return SimpleNamespace(distribution=SimpleNamespace(
            probs=torch.tensor([self.probabilities], dtype=torch.float32),
        ))

    def predict(self, observation, deterministic=False):
        self.predict_calls.append((observation, deterministic))
        for _ in range(self.distribution_calls):
            self.policy.get_distribution(observation)
        if self.error is not None:
            raise self.error
        return self.result


def test_captured_predict_reuses_single_actual_network_pass_and_preserves_return_and_rng():
    model = PredictOnce()
    observation = np.zeros(115, dtype=np.float32)
    before = model.policy.get_distribution
    rng = global_rng_state()
    result, probabilities = trace.captured_predict(model, observation)
    assert result is model.result
    np.testing.assert_allclose(probabilities, [.2, .5, .3], atol=1e-7)
    assert len(model.predict_calls) == len(model.network_calls) == 1
    assert model.predict_calls[0][0] is observation
    assert model.predict_calls[0][1] is True
    assert model.network_calls[0] is observation
    assert model.policy.get_distribution == before
    assert_global_rng_equal(global_rng_state(), rng)


@pytest.mark.parametrize("calls", [0, 2])
def test_captured_predict_rejects_missing_or_extra_network_pass_and_restores(calls):
    model = PredictOnce(distribution_calls=calls)
    before = model.policy.get_distribution
    with pytest.raises(ValueError):
        trace.captured_predict(model, np.zeros(115, dtype=np.float32))
    assert len(model.predict_calls) == 1
    assert model.policy.get_distribution == before


@pytest.mark.parametrize("probabilities,action", [
    ((.2, .5, .3), 0), ((.4, .4, .2), 1), ((math.nan, .5, .5), 1),
    ((-.1, .8, .3), 1), ((0, 1.1, -.1), 1), ((.2, .4, .3), 1),
])
def test_captured_predict_rejects_invalid_argmax_or_nonfinite_distribution(probabilities, action):
    model = PredictOnce(probabilities=probabilities, action=action)
    before = model.policy.get_distribution
    with pytest.raises(ValueError):
        trace.captured_predict(model, np.zeros(115, dtype=np.float32))
    assert model.policy.get_distribution == before


def test_captured_predict_restores_actual_distribution_hook_on_predict_exception():
    model = PredictOnce(error=RuntimeError("deliberate predict failure"))
    before = model.policy.get_distribution
    with pytest.raises(RuntimeError, match="deliberate predict failure"):
        trace.captured_predict(model, np.zeros(115, dtype=np.float32))
    assert model.policy.get_distribution == before
    assert len(model.predict_calls) == len(model.network_calls) == 1


def test_real_tiny_cpu_ppo_prediction_is_single_pass_with_unchanged_action_and_rng(monkeypatch):
    from stable_baselines3 import PPO

    env = trace.make_diagnostic_env("geometry", bootstrap_seed=7)
    try:
        model = PPO("MlpPolicy", env, seed=7, n_steps=8, batch_size=8, n_epochs=1,
                    policy_kwargs={"net_arch": [8]}, device="cpu", verbose=0)
        observation, _ = env.reset(seed=7)
        expected_action, expected_state = model.predict(observation, deterministic=True)
        original_distribution = model.policy.get_distribution
        calls = []

        def counted(*args, **kwargs):
            calls.append(True)
            return original_distribution(*args, **kwargs)

        monkeypatch.setattr(model.policy, "get_distribution", counted)
        before = global_rng_state()
        (action, state), probabilities = trace.captured_predict(model, observation)
        np.testing.assert_array_equal(action, expected_action)
        assert state is expected_state is None
        assert int(action) == int(np.asarray(probabilities).argmax())
        assert len(calls) == 1
        assert model.policy.get_distribution is counted
        assert_global_rng_equal(global_rng_state(), before)
        assert model.num_timesteps == 0
    finally:
        env.close()
