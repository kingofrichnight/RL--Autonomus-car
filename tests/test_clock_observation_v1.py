import copy
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from highway_env.envs.common.observation import KinematicObservation
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from safeintent_rl.envs.driver_behavior import DriverBehaviorWrapper
from safeintent_rl.envs.reward import RouteProgressRewardWrapper
from safeintent_rl.sensors.predictive import PredictiveSafetyObservation
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
from scripts.clock_observation_v1 import ClockPredictiveObservation
from scripts.run_sector_comparison_v1 import initial_policy_fingerprint


def forbidden(*args, **kwargs):
    pytest.fail("The clock wrapper must not drive, observe, forecast, or draw random numbers")


class FabricatedIntersection(gym.Env):
    """Contract-only object: no HighwayEnv road, vehicles, reset or physics exists."""

    def __init__(self):
        self.config = {
            "duration": 30, "simulation_frequency": 15, "policy_frequency": 5,
            "spawn_probability": .6, "initial_vehicle_count": 10,
            "collision_reward": -10.0, "arrived_reward": 5.0,
            "high_speed_reward": .05, "normalize_reward": False, "controlled_vehicles": 1,
            "observation": {
                "type": "Kinematics", "vehicles_count": 15,
                "features": list(SynchronizedPredictiveObservation.FEATURES),
                "absolute": False, "normalize": True, "order": "sorted",
                "features_range": {"x": [-200, 200], "y": [-200, 200],
                                   "vx": [-80, 80], "vy": [-80, 80]},
            },
            "action": {"type": "DiscreteMetaAction", "longitudinal": True,
                       "lateral": False, "target_speeds": [0, 4.5, 9]},
        }
        self.spec = gym.envs.registration.EnvSpec("intersection-v2")
        self.time = 0.0
        self.np_random = np.random.default_rng(7)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(15, 7), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.action_type = SimpleNamespace(actions={0: "SLOWER", 1: "IDLE", 2: "FASTER"},
                                           target_speeds=np.array([0, 4.5, 9]),
                                           longitudinal=True, lateral=False)
        self.observation_type = KinematicObservation(self, **self.config["observation"])
        self.observation_type.observe = forbidden

    reset = forbidden
    step = forbidden
    _agent_rewards = forbidden


def fabricated_parent():
    base = FabricatedIntersection()
    reward = RouteProgressRewardWrapper(DriverBehaviorWrapper(base), collision_first=True)
    predictive = PredictiveSafetyObservation(reward)
    predictive.forecast = forbidden
    parent = SynchronizedPredictiveObservation(predictive)
    parent.reset = forbidden
    parent.step = forbidden
    return parent


def prefix():
    result = np.linspace(-.75, .75, 115, dtype=np.float32)
    result[105:] = .5
    result[1] = np.float32(-0.0)
    return result


@pytest.mark.parametrize("arm", ["zero", "clock"])
@pytest.mark.parametrize("elapsed", [0.0, .2, 15.0, np.nextafter(30.0, 0.0), 30.0, 30.2])
def test_appended_clock_uses_actual_time_and_preserves_all_460_prefix_bytes(arm, elapsed):
    parent = fabricated_parent()
    wrapper = ClockPredictiveObservation(parent, arm=arm)
    parent.unwrapped.time = elapsed
    observation = prefix()
    before = observation.tobytes()
    observation.flags.writeable = False
    actual = wrapper._augment(observation)
    expected = np.float32(max(0.0, min(1.0, (30 - elapsed) / 30))) if arm == "clock" else 0
    assert actual.shape == (116,) and actual.dtype == np.float32
    assert actual[:115].tobytes() == before and len(before) == 460
    assert observation.tobytes() == before
    assert not np.shares_memory(actual, observation)
    assert actual[-1] == expected
    assert wrapper.observation_space.contains(actual)
    if arm == "clock" and elapsed == np.nextafter(30.0, 0.0):
        assert actual[-1] > 0


@pytest.mark.parametrize("arm", ["zero", "clock"])
def test_bounds_actions_and_rng_are_preserved_without_calls_or_source_mutation(arm, monkeypatch):
    parent = fabricated_parent()
    base = parent.unwrapped
    source = prefix()
    config = copy.deepcopy(base.config)
    rng = copy.deepcopy(base.np_random.bit_generator.state)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(np.random, "random", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(torch, "rand", forbidden)
    wrapper = ClockPredictiveObservation(parent, arm=arm)
    wrapper._check_contract()
    wrapper._augment(source)
    np.testing.assert_array_equal(wrapper.observation_space.low[:115], parent.observation_space.low)
    np.testing.assert_array_equal(
        wrapper.observation_space.high[:115], parent.observation_space.high,
    )
    assert wrapper.observation_space.low[-1] == 0 and wrapper.observation_space.high[-1] == 1
    assert wrapper.action_space is parent.action_space
    assert base.config == config and base.time == 0
    assert base.np_random.bit_generator.state == rng
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert wrapper.arm == arm
    expected_protocol = ("predictive_post_spawn_clock_zero_v1" if arm == "zero"
                         else "predictive_post_spawn_clock_remaining_v1")
    assert wrapper.protocol == expected_protocol


@pytest.mark.parametrize("arm", ["", "padding", "remaining", None, True, 0])
def test_invalid_arm_is_rejected(arm):
    with pytest.raises((ValueError, TypeError)):
        ClockPredictiveObservation(fabricated_parent(), arm=arm)


@pytest.mark.parametrize("kind", ["base", "predictive", "extra_wrapper"])
def test_parent_must_be_the_direct_synchronized_predictive_wrapper(kind):
    parent = fabricated_parent()
    incompatible = {"base": parent.unwrapped, "predictive": parent.env,
                    "extra_wrapper": gym.Wrapper(parent)}[kind]
    with pytest.raises((ValueError, TypeError)):
        ClockPredictiveObservation(incompatible, arm="clock")


@pytest.mark.parametrize("elapsed", [-.2, float("nan"), float("inf"), -float("inf"), True, "0"])
@pytest.mark.parametrize("arm", ["zero", "clock"])
def test_both_arms_reject_invalid_simulation_time(arm, elapsed):
    wrapper = ClockPredictiveObservation(fabricated_parent(), arm=arm)
    wrapper.unwrapped.time = elapsed
    with pytest.raises((ValueError, TypeError)):
        wrapper._augment(prefix())


@pytest.mark.parametrize("mutation", [
    "dtype", "short", "long", "matrix", "nan", "inf", "list", "native_high", "native_low",
    "forecast_high",
])
def test_rejects_invalid_supplied_prefix_without_coercion(mutation):
    wrapper = ClockPredictiveObservation(fabricated_parent(), arm="clock")
    observation = prefix()
    if mutation == "dtype":
        observation = observation.astype(np.float64)
    elif mutation == "short":
        observation = observation[:-1]
    elif mutation == "long":
        observation = np.concatenate([observation, np.array([0], dtype=np.float32)])
    elif mutation == "matrix":
        observation = observation.reshape(1, 115)
    elif mutation == "nan":
        observation[0] = np.nan
    elif mutation == "inf":
        observation[0] = np.inf
    elif mutation == "list":
        observation = observation.tolist()
    elif mutation == "native_high":
        observation[0] = np.nextafter(np.float32(1), np.float32(2))
    elif mutation == "native_low":
        observation[0] = np.nextafter(np.float32(-1), np.float32(-2))
    else:
        observation[-1] = 1.1
    with pytest.raises((ValueError, TypeError)):
        wrapper._augment(observation)


@pytest.mark.parametrize("arm", ["zero", "clock"])
@pytest.mark.parametrize("field,value", [
    ("duration", 29.0), ("duration", 31), ("duration", float("nan")),
    ("duration", True), ("duration", "30"),
    ("simulation_frequency", 30), ("policy_frequency", 10),
])
def test_contract_rejects_changed_configuration_after_construction(arm, field, value):
    wrapper = ClockPredictiveObservation(fabricated_parent(), arm=arm)
    wrapper.unwrapped.config[field] = value
    with pytest.raises((ValueError, TypeError)):
        wrapper._augment(prefix())


@pytest.mark.parametrize("field,value", [
    ("order", "shuffled"), ("vehicles_count", 14), ("absolute", True),
    ("normalize", False), ("clip", False), ("see_behind", True),
    ("include_obstacles", False), ("features", ["presence", "x"]),
    ("features_range", {"x": [-100, 100], "y": [-200, 200],
                        "vx": [-80, 80], "vy": [-80, 80]}),
])
def test_contract_rechecks_recreated_or_changed_native_observation(field, value):
    wrapper = ClockPredictiveObservation(fabricated_parent(), arm="clock")
    native = KinematicObservation(wrapper.unwrapped, **wrapper.unwrapped.config["observation"])
    setattr(native, field, value)
    native.observe = forbidden
    wrapper.unwrapped.observation_type = native
    with pytest.raises((ValueError, TypeError)):
        wrapper._augment(prefix())


@pytest.mark.parametrize("field,value", [
    ("horizon", 4.0), ("margin", .6), ("uncertainty_growth", .5),
    ("clearance_scale", 20.0), ("speed_scale", 10.0), ("max_neighbors", 13), ("frequency", 30),
    ("actions", ("FASTER", "IDLE", "SLOWER")),
])
def test_contract_rejects_changed_predictive_settings(field, value):
    wrapper = ClockPredictiveObservation(fabricated_parent(), arm="clock")
    setattr(wrapper.env.env, field, value)
    with pytest.raises((ValueError, TypeError)):
        wrapper._check_contract()


@pytest.mark.parametrize("mutation", ["action_count", "target_speeds", "parent_dtype",
                                      "parent_shape", "parent_bounds"])
def test_contract_rejects_incompatible_action_or_observation_spaces(mutation):
    wrapper = ClockPredictiveObservation(fabricated_parent(), arm="clock")
    if mutation == "action_count":
        wrapper.unwrapped.action_space = gym.spaces.Discrete(4)
    elif mutation == "target_speeds":
        wrapper.unwrapped.action_type.target_speeds = np.array([0, 5, 10])
    else:
        space = wrapper.env.observation_space
        low, high = space.low.copy(), space.high.copy()
        dtype = np.float32
        if mutation == "parent_dtype":
            dtype = np.float64
        elif mutation == "parent_shape":
            low, high = low[:-1], high[:-1]
        else:
            low[0] = -10
        wrapper.env.observation_space = gym.spaces.Box(low, high, dtype=dtype)
    with pytest.raises((ValueError, TypeError)):
        wrapper._check_contract()


@pytest.mark.parametrize("field,value", [("arm", "padding"), ("protocol", "old_predictive")])
def test_mutated_arm_or_protocol_identity_is_rejected(field, value):
    wrapper = ClockPredictiveObservation(fabricated_parent(), arm="clock")
    setattr(wrapper, field, value)
    with pytest.raises((ValueError, TypeError)):
        wrapper._augment(prefix())


@pytest.mark.parametrize("arm", ["zero", "clock"])
@pytest.mark.parametrize("terminated,truncated,elapsed", [
    (False, False, .2), (True, False, 15.0), (False, True, 30.2), (True, True, 30.2),
])
def test_one_delegation_preserves_action_reward_info_flags_and_reads_returned_time(
        arm, terminated, truncated, elapsed):
    parent = fabricated_parent()
    source = prefix()
    source.flags.writeable = False
    reset_info, step_info = {"reset": object()}, {"collision": terminated, "details": object()}
    reward = np.float64(-1.25)
    calls = []
    options = {}
    action = object()

    def reset(**kwargs):
        calls.append(("reset", kwargs))
        parent.unwrapped.time = 0
        return source, reset_info

    def step(actual_action):
        calls.append(("step", actual_action))
        parent.unwrapped.time = elapsed
        return source, reward, terminated, truncated, step_info

    parent.reset, parent.step = reset, step
    wrapper = ClockPredictiveObservation(parent, arm=arm)
    parent.unwrapped.time = 17
    reset_observation, info = wrapper.reset(seed=7, options=options)
    assert info is reset_info
    assert reset_observation[-1] == (1 if arm == "clock" else 0)
    observation, actual_reward, actual_terminated, actual_truncated, info = wrapper.step(action)
    assert calls == [("reset", {"seed": 7, "options": options}), ("step", action)]
    assert calls[0][1]["options"] is options and calls[1][1] is action
    assert actual_reward is reward and info is step_info
    assert actual_terminated is terminated and actual_truncated is truncated
    assert observation[:115].tobytes() == reset_observation[:115].tobytes() == source.tobytes()
    expected = np.float32(max(0, (30 - elapsed) / 30)) if arm == "clock" else 0
    assert observation[-1] == expected
    if terminated and elapsed == 15 and arm == "clock":
        assert observation[-1] == .5


def ending_env(arm="clock", elapsed=30.2, terminated=False, truncated=True):
    parent = fabricated_parent()
    source = prefix()
    calls = []

    def reset(**kwargs):
        calls.append("reset")
        parent.unwrapped.time = 0
        return source.copy(), {"reset": True}

    def step(action):
        calls.append("step")
        parent.unwrapped.time = elapsed
        return source.copy(), 1.0, terminated, truncated, {"synthetic": True}

    parent.reset, parent.step = reset, step
    return ClockPredictiveObservation(parent, arm=arm), calls


@pytest.mark.parametrize("arm", ["zero", "clock"])
@pytest.mark.parametrize("elapsed,terminated,truncated", [
    (30.2, False, True), (15.0, True, False), (30.2, True, True),
])
def test_dummy_vec_env_saves_terminal_clock_separately_from_autoreset(
        arm, elapsed, terminated, truncated):
    env, calls = ending_env(arm, elapsed, terminated, truncated)
    vector = DummyVecEnv([lambda: env])
    try:
        initial = vector.reset()
        returned, rewards, dones, infos = vector.step(np.array([1]))
        assert initial.shape == returned.shape == (1, 116)
        assert returned[0, -1] == initial[0, -1] == (1 if arm == "clock" else 0)
        terminal = infos[0]["terminal_observation"]
        assert terminal[-1] == (np.float32(max(0, (30 - elapsed) / 30)) if arm == "clock" else 0)
        assert infos[0]["TimeLimit.truncated"] is (truncated and not terminated)
        assert rewards.tolist() == [1] and dones.tolist() == [True]
        assert calls == ["reset", "step", "reset"]
    finally:
        vector.close()


class SyntheticValuePolicy:
    def __init__(self):
        self.value_observations = []

    def set_training_mode(self, mode):
        assert mode is False

    def __call__(self, observation):
        return torch.tensor([1]), torch.tensor([[0.0]]), torch.tensor([0.0])

    def obs_to_tensor(self, observation):
        return torch.as_tensor(observation).reshape(1, -1), False

    def predict_values(self, observation):
        self.value_observations.append(observation.clone())
        return (2 + 10 * observation[:, -1]).reshape(-1, 1)


@pytest.mark.parametrize("terminated,truncated,elapsed,expected_reward", [
    (False, True, 30.2, 1 + .99 * 2),
    (True, False, 15.0, 1.0),
    (True, True, 30.2, 1.0),
])
def test_installed_sb3_collector_bootstraps_only_pure_truncation_from_terminal_clock(
        terminated, truncated, elapsed, expected_reward):
    env, calls = ending_env("clock", elapsed, terminated, truncated)
    vector = DummyVecEnv([lambda: env])
    policy = SyntheticValuePolicy()
    saved_rewards = []
    buffer = SimpleNamespace(
        reset=lambda: None,
        add=lambda obs, actions, rewards, *args: saved_rewards.append(rewards.copy()),
        compute_returns_and_advantage=lambda **kwargs: None,
    )
    callback = SimpleNamespace(on_rollout_start=lambda: None, update_locals=lambda values: None,
                               on_step=lambda: True, on_rollout_end=lambda: None)
    collector = SimpleNamespace(
        _last_obs=vector.reset(), policy=policy, use_sde=False, sde_sample_freq=-1,
        device=torch.device("cpu"), action_space=env.action_space, num_timesteps=0,
        gamma=.99, _last_episode_starts=np.array([True]), _update_info_buffer=lambda *args: None,
    )
    try:
        assert OnPolicyAlgorithm.collect_rollouts(collector, vector, callback, buffer, 1)
        assert saved_rewards[0][0] == pytest.approx(expected_reward)
        pure_timeout = truncated and not terminated
        assert len(policy.value_observations) == (2 if pure_timeout else 1)
        if pure_timeout:
            assert policy.value_observations[0][0, -1] == 0
        assert policy.value_observations[-1][0, -1] == 1
        assert collector._last_obs[0, -1] == 1
        assert calls == ["reset", "step", "reset"]
    finally:
        vector.close()


def test_fresh_116_input_cpu_ppo_initializations_match_without_reset_learning_or_driving(
        monkeypatch):
    monkeypatch.setattr(PPO, "learn", forbidden)
    monkeypatch.setattr(PPO, "train", forbidden)
    fingerprints = []
    environments = []
    with torch.random.fork_rng():
        try:
            for arm in ("zero", "clock"):
                env = ClockPredictiveObservation(fabricated_parent(), arm=arm)
                environments.append(env)
                model = PPO(
                    "MlpPolicy", env, learning_rate=.0003, n_steps=1024, batch_size=64,
                    n_epochs=10, gamma=.99, gae_lambda=.95, ent_coef=.01, vf_coef=.5,
                    max_grad_norm=.5, clip_range=.2, normalize_advantage=True,
                    target_kl=None, clip_range_vf=None, seed=42,
                    policy_kwargs={"net_arch": [256, 256]}, device="cpu", verbose=0,
                )
                assert model.num_timesteps == 0
                assert model.observation_space.shape == (116,)
                assert model.action_space == gym.spaces.Discrete(3)
                assert model.device.type == "cpu"
                fingerprints.append(initial_policy_fingerprint(model))
            assert fingerprints[0] == fingerprints[1]
            assert fingerprints[0]["parameter_count"] == 192516
            assert len(fingerprints[0]["sha256"]) == 64
        finally:
            for env in environments:
                env.close()
