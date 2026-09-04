from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import safeintent_rl.intent.wrapper as wrapper_module
from safeintent_rl.intent.wrapper import IntentObservationWrapper


class _Vehicle:
    def __init__(self, position: tuple[float, float], velocity: tuple[float, float]) -> None:
        self.position = np.asarray(position, dtype=np.float32)
        self.velocity = np.asarray(velocity, dtype=np.float32)


class _Road:
    def __init__(self, ego: _Vehicle, ordered_neighbors: list[_Vehicle]) -> None:
        self.ordered_neighbors = ordered_neighbors
        self.vehicles = [ego, *ordered_neighbors]
        self.last_query: dict | None = None

    def close_objects_to(
        self,
        vehicle,
        distance,
        count=None,
        see_behind=True,
        sort=True,
        vehicles_only=False,
    ):
        self.last_query = {
            "vehicle": vehicle,
            "distance": distance,
            "count": count,
            "see_behind": see_behind,
            "sort": sort,
            "vehicles_only": vehicles_only,
        }
        return self.ordered_neighbors[:count]


class _KinematicsEnv(gym.Env):
    def __init__(self, order: str = "sorted") -> None:
        self.vehicle = _Vehicle((0.0, 0.0), (1.0, 0.0))
        first = _Vehicle((4.0, 1.0), (2.0, 0.0))
        second = _Vehicle((7.0, -2.0), (0.5, 0.5))
        self.road = _Road(self.vehicle, [first, second])
        self.PERCEPTION_DISTANCE = 60.0
        self.observation_type = SimpleNamespace(
            order=order,
            include_obstacles=False,
            vehicles_count=15,
            see_behind=True,
            observer_vehicle=self.vehicle,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.ones((15, 7), dtype=np.float32),
            high=np.ones((15, 7), dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((15, 7), dtype=np.float32), {}


class _Predictor:
    instances: list["_Predictor"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[np.ndarray] = []
        self.instances.append(self)

    def predict_proba(self, histories: np.ndarray) -> np.ndarray:
        self.calls.append(histories.copy())
        values = np.asarray([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], dtype=np.float32)
        return values[: len(histories)]


def test_wrapper_aligns_visible_neighbors_and_batches_inference(monkeypatch) -> None:
    _Predictor.instances.clear()
    monkeypatch.setattr(wrapper_module, "IntentPredictor", _Predictor)
    env = _KinematicsEnv()
    wrapped = IntentObservationWrapper(
        env,
        "unused.pt",
        max_neighbors=2,
        history_length=1,
    )

    observation, _ = wrapped.reset(seed=42)

    predictor = _Predictor.instances[-1]
    assert len(predictor.calls) == 1
    assert predictor.calls[0].shape == (2, 1, 6)
    np.testing.assert_allclose(
        observation[-6:],
        [0.1, 0.2, 0.7, 0.8, 0.1, 0.1],
    )
    assert env.road.last_query == {
        "vehicle": env.vehicle,
        "distance": 60.0,
        "count": 2,
        "see_behind": True,
        "sort": True,
        "vehicles_only": True,
    }
    expected_first = np.asarray([4.0, 1.0, 1.0, 0.0, 0.0, np.sqrt(17.0)])
    np.testing.assert_allclose(predictor.calls[0][0, 0], expected_first)


def test_wrapper_rejects_observation_order_that_cannot_be_aligned(monkeypatch) -> None:
    monkeypatch.setattr(wrapper_module, "IntentPredictor", _Predictor)
    with pytest.raises(ValueError, match="sorted"):
        IntentObservationWrapper(_KinematicsEnv(order="shuffled"), "unused.pt")


def test_wrapper_preserves_an_obstacle_slot_without_assigning_intent(monkeypatch) -> None:
    _Predictor.instances.clear()
    monkeypatch.setattr(wrapper_module, "IntentPredictor", _Predictor)
    env = _KinematicsEnv()
    obstacle = _Vehicle((1.0, 0.0), (0.0, 0.0))
    env.road.ordered_neighbors.insert(0, obstacle)
    env.observation_type.include_obstacles = True
    wrapped = IntentObservationWrapper(
        env,
        "unused.pt",
        max_neighbors=2,
        history_length=1,
    )

    observation, _ = wrapped.reset(seed=42)

    predictor = _Predictor.instances[-1]
    assert predictor.calls[0].shape == (1, 1, 6)
    np.testing.assert_allclose(observation[-6:], [0.0, 0.0, 0.0, 0.1, 0.2, 0.7])
    assert env.road.last_query["vehicles_only"] is False
