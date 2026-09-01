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
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("IntentObservationWrapper requires a Box observation space")
        if max_neighbors <= 0 or history_length <= 0:
            raise ValueError("max_neighbors and history_length must be positive")

        self.predictor = IntentPredictor(checkpoint_path)
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
        neighbors = sorted(
            (vehicle for vehicle in road.vehicles if vehicle is not ego),
            key=lambda vehicle: float(
                np.linalg.norm(np.asarray(vehicle.position, dtype=np.float32) - ego_position)
            ),
        )[: self.max_neighbors]
        active_ids: set[int] = set()
        for row, vehicle in enumerate(neighbors):
            key = id(vehicle)
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
                probabilities[row] = self.predictor.predict_proba(
                    np.stack(self.histories[key])
                )[0]
            else:
                probabilities[row] = 1.0 / 3.0

        for stale in set(self.histories) - active_ids:
            self.histories.pop(stale, None)
            self.previous_velocity.pop(stale, None)
        return np.concatenate([flat_observation, probabilities.reshape(-1)]).astype(np.float32)

