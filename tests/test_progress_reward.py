from types import SimpleNamespace

import gymnasium as gym
import pytest
from gymnasium import spaces
from highway_env.envs.intersection_env import IntersectionEnv

from safeintent_rl.config import load_config
from safeintent_rl.envs.reward import RouteProgressRewardWrapper


class _Lane:
    def __init__(self, length: float) -> None:
        self.length = length

    def local_coordinates(self, position: list[float]) -> tuple[float, float]:
        return float(position[0]), float(position[1])


class _Network:
    def __init__(self, lanes: dict[tuple[str, str, int], _Lane]) -> None:
        self.lanes = lanes

    def get_lane(self, lane_index: tuple[str, str, int]) -> _Lane:
        return self.lanes[lane_index]


class _ProgressEnv(gym.Env):
    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(low=0.0, high=200.0, shape=(1,))

    def __init__(self) -> None:
        super().__init__()
        first = ("a", "b", 0)
        second = ("b", "c", 0)
        lanes = {first: _Lane(100.0), second: _Lane(80.0)}
        self.vehicle = SimpleNamespace(
            route=[first, second],
            lane_index=first,
            lane=lanes[first],
            position=[60.0, 0.0],
            velocity=[10.0, 0.0],
        )
        self.other_vehicle = SimpleNamespace(
            position=[63.0, 0.0],
            velocity=[8.0, 0.0],
        )
        self.road = SimpleNamespace(
            network=_Network(lanes),
            vehicles=[self.vehicle, self.other_vehicle],
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.vehicle.position = [60.0, 0.0]
        self.other_vehicle.position = [63.0, 0.0]
        return [60.0], {}

    def step(self, action: int):
        if action == 1:
            self.vehicle.position[0] += 10.0
        return [self.vehicle.position[0]], 1.0, False, False, {}


def test_progress_reward_adds_dense_feedback() -> None:
    env = RouteProgressRewardWrapper(
        _ProgressEnv(), progress_weight=2.0, time_penalty=0.005
    )
    env.reset()
    _, reward, _, _, info = env.step(1)

    expected_delta = 10.0 / 65.0
    assert info["route_progress_delta"] == pytest.approx(expected_delta)
    assert reward == pytest.approx(1.0 + 2.0 * expected_delta - 0.005)


def test_stall_receives_time_cost_without_progress_reward() -> None:
    env = RouteProgressRewardWrapper(
        _ProgressEnv(), progress_weight=2.0, time_penalty=0.005
    )
    env.reset()
    _, reward, _, _, info = env.step(0)

    assert info["route_progress_delta"] == 0.0
    assert info["progress_reward"] == 0.0
    assert reward == pytest.approx(0.995)


def test_ttc_risk_penalty_is_bounded_and_actionable() -> None:
    env = RouteProgressRewardWrapper(
        _ProgressEnv(),
        progress_weight=2.0,
        time_penalty=0.005,
        risk_weight=0.2,
        risk_ttc_threshold=2.0,
    )
    env.reset()
    _, reward, _, _, info = env.step(0)

    assert info["reward_min_ttc"] == pytest.approx(1.5)
    assert info["risk_fraction"] == pytest.approx(0.25)
    assert info["risk_penalty"] == pytest.approx(0.05)
    assert reward == pytest.approx(0.945)


class _TerminalRewardEnv(_ProgressEnv):
    def __init__(self, *, crashed: bool, arrived: bool) -> None:
        super().__init__()
        self.vehicle.crashed = crashed
        self.arrived = arrived
        self.config = {
            "controlled_vehicles": 1,
            "normalize_reward": False,
            "collision_reward": -10.0,
            "arrived_reward": 5.0,
            "high_speed_reward": 0.05,
        }

    def _agent_rewards(self, action, vehicle):
        assert vehicle is self.vehicle
        return {
            "collision_reward": vehicle.crashed,
            "arrived_reward": self.arrived,
            "high_speed_reward": 1.0,
            "on_road_reward": True,
        }

    def step(self, action):
        observation, _, _, _, info = super().step(action)
        # Exercise the installed upstream implementation, including its arrival precedence.
        reward = IntersectionEnv._agent_reward(self, action, self.vehicle)
        return observation, reward, self.vehicle.crashed or self.arrived, False, info


def test_upstream_arrival_overrides_simultaneous_collision_reward() -> None:
    env = _TerminalRewardEnv(crashed=True, arrived=True)
    _, reward, _, _, _ = env.step(0)
    assert reward == 5.0


@pytest.mark.parametrize("crashed,arrived", [(False, False), (False, True), (True, False)])
def test_collision_first_preserves_nonoverlap_rewards(crashed, arrived) -> None:
    legacy = RouteProgressRewardWrapper(_TerminalRewardEnv(crashed=crashed, arrived=arrived))
    corrected = RouteProgressRewardWrapper(
        _TerminalRewardEnv(crashed=crashed, arrived=arrived), collision_first=True
    )
    legacy.reset()
    corrected.reset()
    left = legacy.step(1)
    right = corrected.step(1)
    assert left[:4] == right[:4]
    assert right[4]["collision_arrival_reward_corrected"] is False
    assert right[4]["uncorrected_base_reward"] == left[4]["base_reward"]


def test_collision_first_removes_only_arrival_override_with_progress_preserved() -> None:
    legacy = RouteProgressRewardWrapper(_TerminalRewardEnv(crashed=True, arrived=True))
    corrected = RouteProgressRewardWrapper(
        _TerminalRewardEnv(crashed=True, arrived=True), collision_first=True
    )
    legacy.reset()
    corrected.reset()
    left = legacy.step(1)
    right = corrected.step(1)
    assert left[0] == right[0] and left[2:4] == right[2:4]
    assert left[4]["base_reward"] == right[4]["uncorrected_base_reward"] == 5.0
    assert right[4]["base_reward"] == pytest.approx(-9.95)
    assert right[4]["collision_arrival_reward_corrected"] is True
    assert left[4]["progress_reward"] == right[4]["progress_reward"]
    assert left[1] - right[1] == pytest.approx(14.95)
    assert right[1] < 0
    assert "collision_arrival_reward_corrected" not in left[4]


@pytest.mark.parametrize("key,value", [("normalize_reward", True), ("controlled_vehicles", 2),
                                     ("collision_reward", 0.0), ("collision_reward", float("nan"))])
def test_collision_first_rejects_unsupported_reward_settings(key, value) -> None:
    env = _TerminalRewardEnv(crashed=True, arrived=True)
    env.config[key] = value
    with pytest.raises(ValueError):
        RouteProgressRewardWrapper(env, collision_first=True)


def test_collision_first_requires_explicit_boolean() -> None:
    with pytest.raises(ValueError, match="boolean"):
        RouteProgressRewardWrapper(_ProgressEnv(), collision_first="false")


def test_collision_first_configuration_changes_only_one_wrapper_option() -> None:
    old = load_config("configs/intersection_reward_v3.yaml")
    new = load_config("configs/intersection_reward_v3_collision_first.yaml")
    assert new["reward_wrapper"].pop("collision_first") is True
    assert new == old
