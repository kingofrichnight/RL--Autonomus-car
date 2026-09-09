from __future__ import annotations

import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.sensors import EgoTargetSpeedObservation


class _ControllerEnv(gym.Env):
    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(-10, 10, shape=(2, 2), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.vehicle = SimpleNamespace(target_speed=4.5, speed=2.0, route=[("o0", "ir0", 0)])
        self.actions = []
        self.prefix = np.array([[1, 2], [3, 4]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicle.target_speed = 4.5
        self.actions = []
        return self.prefix.copy(), {"reset": True}

    def step(self, action):
        self.actions.append(action)
        self.vehicle.speed += 1
        if int(action) != 1:
            self.vehicle.target_speed = 0.0 if int(action) == 0 else 9.0
        return self.prefix.copy(), 1.25, False, False, {"unchanged": True}


def test_target_state_is_controller_target_not_measured_speed() -> None:
    base = _ControllerEnv()
    env = EgoTargetSpeedObservation(base)
    observation, info = env.reset(seed=7)
    assert observation.tolist() == [1, 2, 3, 4, 0.5]
    assert observation.dtype == np.float32 and env.observation_space.contains(observation)
    assert info == {"reset": True}
    assert base.vehicle.speed == 2.0


def test_target_wrapper_is_read_only_and_preserves_transition() -> None:
    base = _ControllerEnv()
    env = EgoTargetSpeedObservation(base)
    env.reset(seed=7)
    rng = copy.deepcopy(base.np_random.bit_generator.state)
    prefix = base.prefix.copy()
    route = copy.deepcopy(base.vehicle.route)
    observation, reward, terminated, truncated, info = env.step(1)
    assert base.actions == [1]
    np.testing.assert_array_equal(observation[:-1], prefix.flatten())
    assert observation[-1] == 0.5 and base.vehicle.speed == 3
    assert reward == 1.25 and terminated is False and truncated is False
    assert info == {"unchanged": True}
    assert base.np_random.bit_generator.state == rng and base.vehicle.route == route


def test_target_state_updates_after_action_and_reset() -> None:
    env = EgoTargetSpeedObservation(_ControllerEnv())
    env.reset()
    assert env.step(0)[0][-1] == 0.0
    assert env.step(1)[0][-1] == 0.0
    assert env.step(2)[0][-1] == 1.0
    assert env.reset()[0][-1] == 0.5


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_target_wrapper_rejects_invalid_scale(scale) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        EgoTargetSpeedObservation(_ControllerEnv(), scale=scale)


@pytest.mark.parametrize("target", [-1, 10, float("nan"), float("inf")])
def test_target_wrapper_rejects_invalid_target_instead_of_clipping(target) -> None:
    base = _ControllerEnv()
    env = EgoTargetSpeedObservation(base)
    base.vehicle.target_speed = target
    with pytest.raises(ValueError, match="within the recorded scale"):
        env._augment(base.prefix)


@pytest.mark.parametrize("fusion", [False, True])
def test_factory_target_feature_preserves_reset_observation_prefix(fusion) -> None:
    plain = make_intersection_env(
        config_path="configs/intersection_reward_v3_collision_first.yaml", risk_fusion=fusion
    )
    target = make_intersection_env(
        config_path="configs/intersection_reward_v3_collision_first.yaml", risk_fusion=fusion,
        target_speed_observation=True,
    )
    try:
        left, _ = plain.reset(seed=7)
        right, _ = target.reset(seed=7)
        assert left.size == (175 if fusion else 105)
        assert right.size == left.size + 1
        np.testing.assert_array_equal(left.reshape(-1), right[:-1])
        assert right[-1] == target.unwrapped.vehicle.target_speed / 9
    finally:
        plain.close()
        target.close()
