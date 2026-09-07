from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from safeintent_rl.safety import CPAAccelerationShield


class StubIntersection(gym.Env):
    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))

    def __init__(self, other_position: tuple[float, float]) -> None:
        super().__init__()
        self.vehicle = SimpleNamespace(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
        )
        other = SimpleNamespace(
            position=np.array(other_position),
            velocity=np.array([0.0, -1.0]),
        )
        self.road = SimpleNamespace(vehicles=[self.vehicle, other])
        self.action_type = SimpleNamespace(
            actions={0: "SLOWER", 1: "IDLE", 2: "FASTER"}
        )
        self.last_action: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(1), {}

    def step(self, action: int):
        self.last_action = int(action)
        return np.zeros(1), 0.0, False, False, {}


def test_cpa_shield_vetoes_only_faster_for_matching_conflict() -> None:
    base = StubIntersection((2.0, 2.0))
    env = CPAAccelerationShield(base)
    env.reset()

    _, _, _, _, info = env.step(2)

    assert base.last_action == 1
    assert info["safety_intervened"] is True
    assert info["proposed_action"] == 2
    assert info["executed_action"] == 1
    assert info["cpa_conflict"] is True
    assert info["cpa_conflict_time"] == pytest.approx(2.0)
    assert info["cpa_conflict_distance"] == pytest.approx(0.0)

    _, _, _, _, info = env.step(1)

    assert base.last_action == 1
    assert info["safety_intervened"] is False
    assert info["safety_intervention_rate"] == pytest.approx(0.5)


def test_cpa_shield_releases_when_geometry_does_not_match() -> None:
    base = StubIntersection((20.0, 20.0))
    env = CPAAccelerationShield(base)
    env.reset()

    _, _, _, _, info = env.step(2)

    assert base.last_action == 2
    assert info["safety_intervened"] is False
    assert info["cpa_conflict"] is False


def test_cpa_shield_can_selectively_brake_faster() -> None:
    base = StubIntersection((2.0, 2.0))
    env = CPAAccelerationShield(base, override_action="SLOWER")
    env.reset()

    _, _, _, _, info = env.step(2)

    assert base.last_action == 0
    assert info["safety_intervened"] is True
    assert info["proposed_action"] == 2
    assert info["executed_action"] == 0


def test_cpa_shield_rejects_unknown_override_action() -> None:
    with pytest.raises(ValueError, match="must be IDLE or SLOWER"):
        CPAAccelerationShield(StubIntersection((2.0, 2.0)), override_action="BRAKE")


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("time_threshold", 0.0),
        ("distance_threshold", -1.0),
        ("horizon", float("inf")),
        ("max_range", float("nan")),
    ],
)
def test_cpa_shield_rejects_invalid_parameters(keyword: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CPAAccelerationShield(StubIntersection((2.0, 2.0)), **{keyword: value})


def test_cpa_shield_rejects_threshold_beyond_horizon() -> None:
    with pytest.raises(ValueError, match="cannot exceed horizon"):
        CPAAccelerationShield(
            StubIntersection((2.0, 2.0)),
            time_threshold=3.1,
            horizon=3.0,
        )
