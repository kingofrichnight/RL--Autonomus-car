from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from safeintent_rl.sensors import KinematicRiskFusionWrapper


class StubRoad:
    def __init__(self, vehicles: list[object]) -> None:
        self.vehicles = vehicles

    def close_objects_to(self, _ego: object, _distance: float, **kwargs):
        count = int(kwargs["count"])
        return self.vehicles[1 : count + 1]


class StubKinematicsEnv(gym.Env):
    action_space = gym.spaces.Discrete(3)

    def __init__(self, *, order: str = "sorted") -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(3, 7),
            dtype=np.float32,
        )
        self.vehicle = SimpleNamespace(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
        )
        crossing = SimpleNamespace(
            position=np.array([2.0, 2.0]),
            velocity=np.array([0.0, -1.0]),
        )
        self.road = StubRoad([self.vehicle, crossing])
        self.PERCEPTION_DISTANCE = 200.0
        self.observation_type = SimpleNamespace(
            observer_vehicle=self.vehicle,
            order=order,
            vehicles_count=3,
            see_behind=False,
            include_obstacles=False,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros((3, 7), dtype=np.float32), {}

    def step(self, action: int):
        return np.ones((3, 7), dtype=np.float32), 0.0, False, False, {}


def test_sensor_fusion_appends_bounded_aligned_risk_features() -> None:
    env = KinematicRiskFusionWrapper(
        StubKinematicsEnv(),
        max_neighbors=2,
        range_scale=10.0,
        relative_speed_scale=2.0,
        ttc_scale=10.0,
        cpa_horizon=5.0,
        cpa_distance_scale=10.0,
    )

    observation, _ = env.reset()

    assert observation.shape == (31,)
    assert env.observation_space.shape == (31,)
    crossing = observation[21:26]
    assert crossing[0] == pytest.approx(np.sqrt(8.0) / 10.0)
    assert crossing[1] == pytest.approx(np.sqrt(2.0) / 2.0)
    assert crossing[2] == pytest.approx(0.2)
    assert crossing[3] == pytest.approx(0.4)
    assert crossing[4] == pytest.approx(0.0)
    assert observation[26:].tolist() == pytest.approx([1.0, 0.0, 1.0, 1.0, 1.0])
    assert env.observation_space.contains(observation)


def test_sensor_fusion_preserves_native_observation_values_on_step() -> None:
    env = KinematicRiskFusionWrapper(StubKinematicsEnv(), max_neighbors=2)

    observation, _, _, _, _ = env.step(1)

    assert observation[:21].tolist() == pytest.approx([1.0] * 21)


def test_sensor_fusion_requires_sorted_slots() -> None:
    with pytest.raises(ValueError, match="requires sorted"):
        KinematicRiskFusionWrapper(StubKinematicsEnv(order="shuffled"), max_neighbors=2)


def test_sensor_fusion_rejects_too_many_neighbors() -> None:
    with pytest.raises(ValueError, match="exceeds observed"):
        KinematicRiskFusionWrapper(StubKinematicsEnv(), max_neighbors=3)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("range_scale", 0.0),
        ("relative_speed_scale", float("inf")),
        ("ttc_scale", -1.0),
        ("cpa_horizon", float("nan")),
        ("cpa_distance_scale", 0.0),
    ],
)
def test_sensor_fusion_rejects_invalid_scales(keyword: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        KinematicRiskFusionWrapper(
            StubKinematicsEnv(),
            max_neighbors=2,
            **{keyword: value},
        )
