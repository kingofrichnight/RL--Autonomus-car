from types import SimpleNamespace

import gymnasium as gym
import pytest
from gymnasium import spaces

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
        )
        self.road = SimpleNamespace(network=_Network(lanes))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.vehicle.position = [60.0, 0.0]
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
