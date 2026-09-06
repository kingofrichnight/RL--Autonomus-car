from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from safeintent_rl.intent.inference import IntentPredictor


class IntentObservationWrapper(gym.Wrapper):
    """Append GRU behavior probabilities for the nearest NPCs to a flat observation."""

    def __init__(
        self,
        env: gym.Env,
        checkpoint_path: str | Path,
        max_neighbors: int = 5,
        history_length: int | None = None,
        device: str = "cpu",
        expected_checkpoint_sha256: str | None = None,
        collect_diagnostics: bool = False,
        shadow_history_neighbors: int | None = None,
        history_tracking_neighbors: int | None = None,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("IntentObservationWrapper requires a Box observation space")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")

        observation_type = getattr(self.unwrapped, "observation_type", None)
        if observation_type is None:
            raise TypeError("IntentObservationWrapper requires a kinematics observation")
        if getattr(observation_type, "order", None) != "sorted":
            raise ValueError("IntentObservationWrapper requires sorted vehicle observations")
        observed_slots = int(getattr(observation_type, "vehicles_count", 1)) - 1
        if max_neighbors > observed_slots:
            raise ValueError("max_neighbors exceeds the number of observed traffic slots")
        tracking_neighbors = (
            max_neighbors
            if history_tracking_neighbors is None
            else history_tracking_neighbors
        )
        if not max_neighbors <= tracking_neighbors <= observed_slots:
            raise ValueError(
                "history_tracking_neighbors must cover the intent slots without "
                "exceeding observed traffic slots"
            )
        if shadow_history_neighbors is not None:
            if not collect_diagnostics:
                raise ValueError("shadow history tracking requires diagnostics")
            if not max_neighbors <= shadow_history_neighbors <= observed_slots:
                raise ValueError(
                    "shadow_history_neighbors must cover the intent slots without "
                    "exceeding observed traffic slots"
                )

        self.predictor = IntentPredictor(
            checkpoint_path,
            device=device,
            expected_sha256=expected_checkpoint_sha256,
        )
        checkpoint_history_length = getattr(self.predictor, "history_length", None)
        if history_length is None:
            history_length = checkpoint_history_length or 10
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if (
            checkpoint_history_length is not None
            and history_length != checkpoint_history_length
        ):
            raise ValueError(
                f"Intent checkpoint expects history length {checkpoint_history_length}, "
                f"got {history_length}"
            )
        self.max_neighbors = max_neighbors
        self.history_length = history_length
        self.history_tracking_neighbors = tracking_neighbors
        self.collect_diagnostics = collect_diagnostics
        self.shadow_history_neighbors = shadow_history_neighbors
        self.last_intent_diagnostics: dict[str, Any] = {}
        self.last_shadow_intent_diagnostics: dict[str, Any] = {}
        self.histories: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.previous_velocity: dict[int, np.ndarray] = {}
        self.shadow_histories: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.shadow_previous_velocity: dict[int, np.ndarray] = {}
        original_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        original_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([original_low, np.zeros(max_neighbors * 3, dtype=np.float32)]),
            high=np.concatenate([original_high, np.ones(max_neighbors * 3, dtype=np.float32)]),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.histories.clear()
        self.previous_velocity.clear()
        self.shadow_histories.clear()
        self.shadow_previous_velocity.clear()
        return self._augment(observation), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info

    def _augment(self, observation: Any) -> np.ndarray:
        flat_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        probabilities = np.zeros((self.max_neighbors, 3), dtype=np.float32)
        diagnostics: dict[str, Any] = {
            "slot_count": self.max_neighbors,
            "missing_slots": self.max_neighbors,
            "obstacle_slots": 0,
            "vehicle_slots": 0,
            "warmup_vehicle_slots": 0,
            "predicted_vehicle_slots": 0,
            "labeled_predictions": 0,
            "correct_predictions": 0,
            "confidence_sum": 0.0,
            "entropy_sum": 0.0,
            "true_labels": [],
            "predicted_labels": [],
            "vehicle_rows": [],
            "predicted_rows": [],
        }
        base = self.unwrapped
        ego = getattr(base, "vehicle", None)
        road = getattr(base, "road", None)
        if ego is None or road is None:
            if self.collect_diagnostics:
                self.last_intent_diagnostics = diagnostics
            return np.concatenate([flat_observation, probabilities.reshape(-1)])

        ego_position = np.asarray(ego.position, dtype=np.float32)
        ego_velocity = np.asarray(ego.velocity, dtype=np.float32)
        observation_type = base.observation_type
        neighbors = road.close_objects_to(
            getattr(observation_type, "observer_vehicle", ego),
            float(base.PERCEPTION_DISTANCE),
            count=self.max_neighbors,
            see_behind=bool(observation_type.see_behind),
            sort=True,
            vehicles_only=not bool(observation_type.include_obstacles),
        )
        tracked_neighbors = neighbors
        if self.history_tracking_neighbors != self.max_neighbors:
            tracked_neighbors = road.close_objects_to(
                getattr(observation_type, "observer_vehicle", ego),
                float(base.PERCEPTION_DISTANCE),
                count=self.history_tracking_neighbors,
                see_behind=bool(observation_type.see_behind),
                sort=True,
                vehicles_only=not bool(observation_type.include_obstacles),
            )
            if [id(value) for value in tracked_neighbors[: len(neighbors)]] != [
                id(value) for value in neighbors
            ]:
                raise RuntimeError(
                    "Tracked history slots do not align with policy intent slots"
                )
        diagnostics["missing_slots"] = self.max_neighbors - len(neighbors)
        road_vehicle_ids = {id(vehicle) for vehicle in road.vehicles}
        active_ids: set[int] = set()
        ready_rows: list[int] = []
        ready_histories: list[np.ndarray] = []
        ready_vehicles: list[Any] = []
        for vehicle in tracked_neighbors:
            key = id(vehicle)
            if key not in road_vehicle_ids:
                continue
            active_ids.add(key)
            velocity = np.asarray(vehicle.velocity, dtype=np.float32)
            previous = self.previous_velocity.get(key, velocity)
            rel_position = np.asarray(vehicle.position, dtype=np.float32) - ego_position
            rel_velocity = velocity - ego_velocity
            feature = np.asarray(
                [
                    rel_position[0],
                    rel_position[1],
                    rel_velocity[0],
                    rel_velocity[1],
                    np.linalg.norm(velocity - previous),
                    np.linalg.norm(rel_position),
                ],
                dtype=np.float32,
            )
            self.histories[key].append(feature)
            self.previous_velocity[key] = velocity

        for row, vehicle in enumerate(neighbors):
            key = id(vehicle)
            if key not in road_vehicle_ids:
                diagnostics["obstacle_slots"] += 1
                continue
            diagnostics["vehicle_slots"] += 1
            diagnostics["vehicle_rows"].append(row)
            if len(self.histories[key]) == self.history_length:
                ready_rows.append(row)
                ready_histories.append(np.stack(self.histories[key]))
                ready_vehicles.append(vehicle)
            else:
                diagnostics["warmup_vehicle_slots"] += 1
                probabilities[row] = 1.0 / 3.0

        if ready_histories:
            predicted = np.asarray(
                self.predictor.predict_proba(np.stack(ready_histories)),
                dtype=np.float32,
            )
            if predicted.shape != (len(ready_rows), 3) or not np.isfinite(predicted).all():
                raise RuntimeError("Intent predictor returned invalid probabilities")
            if np.any((predicted < 0) | (predicted > 1)) or not np.allclose(
                predicted.sum(axis=1), 1.0, atol=1e-5
            ):
                raise RuntimeError("Intent probabilities must be in [0, 1] and sum to one")
            probabilities[ready_rows] = predicted
            diagnostics["predicted_vehicle_slots"] = len(ready_rows)
            diagnostics["predicted_rows"] = ready_rows.copy()
            if self.collect_diagnostics:
                label_names = list(self.predictor.label_names)
                for vehicle, prediction in zip(ready_vehicles, predicted, strict=True):
                    true_name = getattr(vehicle, "safeintent_driver_label", None)
                    if true_name not in label_names:
                        continue
                    true_index = label_names.index(true_name)
                    predicted_index = int(np.argmax(prediction))
                    diagnostics["labeled_predictions"] += 1
                    diagnostics["correct_predictions"] += int(
                        true_index == predicted_index
                    )
                    diagnostics["confidence_sum"] += float(np.max(prediction))
                    clipped = np.clip(prediction, 1e-12, 1.0)
                    diagnostics["entropy_sum"] += float(-np.sum(clipped * np.log(clipped)))
                    diagnostics["true_labels"].append(true_index)
                    diagnostics["predicted_labels"].append(predicted_index)

        if self.shadow_history_neighbors is not None:
            self.last_shadow_intent_diagnostics = self._shadow_diagnostics(
                ego,
                road,
                neighbors,
                ego_position,
                ego_velocity,
            )

        for stale in set(self.histories) - active_ids:
            self.histories.pop(stale, None)
            self.previous_velocity.pop(stale, None)
        if self.collect_diagnostics:
            self.last_intent_diagnostics = diagnostics
        return np.concatenate([flat_observation, probabilities.reshape(-1)]).astype(np.float32)

    def _shadow_diagnostics(
        self,
        ego: Any,
        road: Any,
        target_neighbors: list[Any],
        ego_position: np.ndarray,
        ego_velocity: np.ndarray,
    ) -> dict[str, Any]:
        """Measure wider history tracking without changing the policy observation."""
        if self.shadow_history_neighbors is None:
            raise RuntimeError("Shadow diagnostics are not enabled")
        observation_type = self.unwrapped.observation_type
        tracked_neighbors = road.close_objects_to(
            getattr(observation_type, "observer_vehicle", ego),
            float(self.unwrapped.PERCEPTION_DISTANCE),
            count=self.shadow_history_neighbors,
            see_behind=bool(observation_type.see_behind),
            sort=True,
            vehicles_only=not bool(observation_type.include_obstacles),
        )
        if [id(value) for value in tracked_neighbors[: len(target_neighbors)]] != [
            id(value) for value in target_neighbors
        ]:
            raise RuntimeError("Shadow history slots do not align with policy intent slots")

        road_vehicle_ids = {id(vehicle) for vehicle in road.vehicles}
        active_ids: set[int] = set()
        for vehicle in tracked_neighbors:
            key = id(vehicle)
            if key not in road_vehicle_ids:
                continue
            active_ids.add(key)
            velocity = np.asarray(vehicle.velocity, dtype=np.float32)
            previous = self.shadow_previous_velocity.get(key, velocity)
            rel_position = np.asarray(vehicle.position, dtype=np.float32) - ego_position
            rel_velocity = velocity - ego_velocity
            feature = np.asarray(
                [
                    rel_position[0],
                    rel_position[1],
                    rel_velocity[0],
                    rel_velocity[1],
                    np.linalg.norm(velocity - previous),
                    np.linalg.norm(rel_position),
                ],
                dtype=np.float32,
            )
            self.shadow_histories[key].append(feature)
            self.shadow_previous_velocity[key] = velocity

        for stale in set(self.shadow_histories) - active_ids:
            self.shadow_histories.pop(stale, None)
            self.shadow_previous_velocity.pop(stale, None)

        diagnostics: dict[str, Any] = {
            "slot_count": self.max_neighbors,
            "missing_slots": self.max_neighbors - len(target_neighbors),
            "obstacle_slots": 0,
            "vehicle_slots": 0,
            "warmup_vehicle_slots": 0,
            "predicted_vehicle_slots": 0,
            "labeled_predictions": 0,
            "correct_predictions": 0,
            "confidence_sum": 0.0,
            "entropy_sum": 0.0,
            "true_labels": [],
            "predicted_labels": [],
            "vehicle_rows": [],
            "predicted_rows": [],
            "history_lengths": [],
        }
        ready_histories: list[np.ndarray] = []
        ready_vehicles: list[Any] = []
        ready_rows: list[int] = []
        for row, vehicle in enumerate(target_neighbors):
            key = id(vehicle)
            if key not in road_vehicle_ids:
                diagnostics["obstacle_slots"] += 1
                continue
            diagnostics["vehicle_slots"] += 1
            diagnostics["vehicle_rows"].append(row)
            diagnostics["history_lengths"].append(len(self.shadow_histories[key]))
            if len(self.shadow_histories[key]) == self.history_length:
                ready_rows.append(row)
                ready_histories.append(np.stack(self.shadow_histories[key]))
                ready_vehicles.append(vehicle)
            else:
                diagnostics["warmup_vehicle_slots"] += 1

        if ready_histories:
            predicted = np.asarray(
                self.predictor.predict_proba(np.stack(ready_histories)),
                dtype=np.float32,
            )
            if predicted.shape != (len(ready_histories), 3) or not np.isfinite(predicted).all():
                raise RuntimeError("Intent predictor returned invalid shadow probabilities")
            if np.any((predicted < 0) | (predicted > 1)) or not np.allclose(
                predicted.sum(axis=1), 1.0, atol=1e-5
            ):
                raise RuntimeError(
                    "Shadow intent probabilities must be in [0, 1] and sum to one"
                )
            diagnostics["predicted_vehicle_slots"] = len(ready_histories)
            diagnostics["predicted_rows"] = ready_rows
            label_names = list(self.predictor.label_names)
            for vehicle, prediction in zip(ready_vehicles, predicted, strict=True):
                true_name = getattr(vehicle, "safeintent_driver_label", None)
                if true_name not in label_names:
                    continue
                true_index = label_names.index(true_name)
                predicted_index = int(np.argmax(prediction))
                diagnostics["labeled_predictions"] += 1
                diagnostics["correct_predictions"] += int(true_index == predicted_index)
                diagnostics["confidence_sum"] += float(np.max(prediction))
                clipped = np.clip(prediction, 1e-12, 1.0)
                diagnostics["entropy_sum"] += float(-np.sum(clipped * np.log(clipped)))
                diagnostics["true_labels"].append(true_index)
                diagnostics["predicted_labels"].append(predicted_index)
        return diagnostics
