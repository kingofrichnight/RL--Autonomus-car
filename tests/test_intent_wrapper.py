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
        first.safeintent_driver_label = "cautious"
        second.safeintent_driver_label = "aggressive"
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
        self.label_names = ["cautious", "normal", "aggressive"]
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


def test_diagnostics_do_not_change_augmented_observation(monkeypatch) -> None:
    _Predictor.instances.clear()
    monkeypatch.setattr(wrapper_module, "IntentPredictor", _Predictor)
    plain = IntentObservationWrapper(
        _KinematicsEnv(),
        "unused.pt",
        max_neighbors=2,
        history_length=1,
    )
    diagnostic = IntentObservationWrapper(
        _KinematicsEnv(),
        "unused.pt",
        max_neighbors=2,
        history_length=1,
        collect_diagnostics=True,
    )

    plain_observation, _ = plain.reset(seed=42)
    diagnostic_observation, _ = diagnostic.reset(seed=42)

    np.testing.assert_array_equal(plain_observation, diagnostic_observation)
    snapshot = diagnostic.last_intent_diagnostics
    assert snapshot["predicted_vehicle_slots"] == 2
    assert snapshot["labeled_predictions"] == 2
    assert snapshot["true_labels"] == [0, 2]
    assert snapshot["predicted_labels"] == [2, 0]


def test_shadow_history_uses_wider_observed_slots_without_changing_output(
    monkeypatch,
) -> None:
    _Predictor.instances.clear()
    monkeypatch.setattr(wrapper_module, "IntentPredictor", _Predictor)
    plain_env = _KinematicsEnv()
    shadow_env = _KinematicsEnv()
    for env in (plain_env, shadow_env):
        third = _Vehicle((9.0, 3.0), (1.5, -0.5))
        third.safeintent_driver_label = "normal"
        env.road.ordered_neighbors.append(third)
        env.road.vehicles.append(third)
    plain = IntentObservationWrapper(
        plain_env,
        "unused.pt",
        max_neighbors=2,
        history_length=2,
        collect_diagnostics=True,
    )
    shadow = IntentObservationWrapper(
        shadow_env,
        "unused.pt",
        max_neighbors=2,
        history_length=2,
        collect_diagnostics=True,
        shadow_history_neighbors=3,
    )

    plain_initial, _ = plain.reset(seed=42)
    shadow_initial, _ = shadow.reset(seed=42)
    np.testing.assert_array_equal(plain_initial, shadow_initial)

    plain_env.road.ordered_neighbors = [
        plain_env.road.ordered_neighbors[2],
        plain_env.road.ordered_neighbors[0],
        plain_env.road.ordered_neighbors[1],
    ]
    shadow_env.road.ordered_neighbors = [
        shadow_env.road.ordered_neighbors[2],
        shadow_env.road.ordered_neighbors[0],
        shadow_env.road.ordered_neighbors[1],
    ]
    base_observation = np.zeros((15, 7), dtype=np.float32)
    plain_next = plain._augment(base_observation)
    shadow_next = shadow._augment(base_observation)

    np.testing.assert_array_equal(plain_next, shadow_next)
    assert shadow.last_intent_diagnostics["warmup_vehicle_slots"] == 1
    assert shadow.last_intent_diagnostics["predicted_vehicle_slots"] == 1
    assert shadow.last_shadow_intent_diagnostics["warmup_vehicle_slots"] == 0
    assert shadow.last_shadow_intent_diagnostics["predicted_vehicle_slots"] == 2
    assert shadow.last_shadow_intent_diagnostics["history_lengths"] == [2, 2]
    assert len(shadow.shadow_histories) == 3


@pytest.mark.parametrize("shadow_neighbors", [1, 15])
def test_shadow_history_requires_valid_observed_slot_count(
    monkeypatch,
    shadow_neighbors,
) -> None:
    monkeypatch.setattr(wrapper_module, "IntentPredictor", _Predictor)
    with pytest.raises(ValueError, match="shadow_history_neighbors"):
        IntentObservationWrapper(
            _KinematicsEnv(),
            "unused.pt",
            max_neighbors=2,
            collect_diagnostics=True,
            shadow_history_neighbors=shadow_neighbors,
        )


def test_shadow_history_requires_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(wrapper_module, "IntentPredictor", _Predictor)
    with pytest.raises(ValueError, match="requires diagnostics"):
        IntentObservationWrapper(
            _KinematicsEnv(),
            "unused.pt",
            max_neighbors=2,
            shadow_history_neighbors=3,
        )
