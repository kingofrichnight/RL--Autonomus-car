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
        history_length: int = 10,
        device: str = "cpu",
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("IntentObservationWrapper requires a Box observation space")
        if max_neighbors <= 0 or history_length <= 0:
            raise ValueError("max_neighbors and history_length must be positive")

        observation_type = getattr(self.unwrapped, "observation_type", None)
        if observation_type is None:
            raise TypeError("IntentObservationWrapper requires a kinematics observation")
        if getattr(observation_type, "order", None) != "sorted":
            raise ValueError("IntentObservationWrapper requires sorted vehicle observations")
        observed_slots = int(getattr(observation_type, "vehicles_count", 1)) - 1
        if max_neighbors > observed_slots:
            raise ValueError("max_neighbors exceeds the number of observed traffic slots")

        self.predictor = IntentPredictor(
            checkpoint_path,
            device=device,
            expected_sha256=expected_checkpoint_sha256,
        )
        self.max_neighbors = max_neighbors
        self.history_length = history_length
        self.histories: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.previous_velocity: dict[int, np.ndarray] = {}
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
        return self._augment(observation), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info

    def _augment(self, observation: Any) -> np.ndarray:
        flat_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        probabilities = np.zeros((self.max_neighbors, 3), dtype=np.float32)
        base = self.unwrapped
        ego = getattr(base, "vehicle", None)
        road = getattr(base, "road", None)
        if ego is None or road is None:
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
        road_vehicle_ids = {id(vehicle) for vehicle in road.vehicles}
        active_ids: set[int] = set()
        ready_rows: list[int] = []
        ready_histories: list[np.ndarray] = []
        for row, vehicle in enumerate(neighbors):
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
            if len(self.histories[key]) == self.history_length:
                ready_rows.append(row)
                ready_histories.append(np.stack(self.histories[key]))
            else:
                probabilities[row] = 1.0 / 3.0

        if ready_histories:
            predicted = np.asarray(
                self.predictor.predict_proba(np.stack(ready_histories)),
                dtype=np.float32,
            )
            if predicted.shape != (len(ready_rows), 3) or not np.isfinite(predicted).all():
                raise RuntimeError("Intent predictor returned invalid probabilities")
            probabilities[ready_rows] = predicted

        for stale in set(self.histories) - active_ids:
            self.histories.pop(stale, None)
            self.previous_velocity.pop(stale, None)
        return np.concatenate([flat_observation, probabilities.reshape(-1)]).astype(np.float32)
