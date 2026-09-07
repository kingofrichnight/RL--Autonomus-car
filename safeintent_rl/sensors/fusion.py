from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from safeintent_rl.safety.conflict import pairwise_closest_approach


class KinematicRiskFusionWrapper(gym.Wrapper):
    """Fuse kinematics with radar-like range/range-rate and CPA risk features."""

    FEATURES_PER_NEIGHBOR = 5

    def __init__(
        self,
        env: gym.Env,
        *,
        max_neighbors: int = 14,
        range_scale: float = 200.0,
        relative_speed_scale: float = 20.0,
        ttc_scale: float = 10.0,
        cpa_horizon: float = 5.0,
        cpa_distance_scale: float = 20.0,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("KinematicRiskFusionWrapper requires a Box observation space")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")

        observation_type = getattr(self.unwrapped, "observation_type", None)
        if observation_type is None:
            raise TypeError("KinematicRiskFusionWrapper requires a kinematics observation")
        if getattr(observation_type, "order", None) != "sorted":
            raise ValueError("KinematicRiskFusionWrapper requires sorted observations")
        observed_slots = int(getattr(observation_type, "vehicles_count", 1)) - 1
        if max_neighbors > observed_slots:
            raise ValueError("max_neighbors exceeds observed traffic slots")

        scales = {
            "range_scale": range_scale,
            "relative_speed_scale": relative_speed_scale,
            "ttc_scale": ttc_scale,
            "cpa_horizon": cpa_horizon,
            "cpa_distance_scale": cpa_distance_scale,
        }
        for name, value in scales.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.max_neighbors = max_neighbors
        self.range_scale = float(range_scale)
        self.relative_speed_scale = float(relative_speed_scale)
        self.ttc_scale = float(ttc_scale)
        self.cpa_horizon = float(cpa_horizon)
        self.cpa_distance_scale = float(cpa_distance_scale)

        original_low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        original_high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        feature_low = np.tile(
            np.array([0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            max_neighbors,
        )
        feature_high = np.ones(
            max_neighbors * self.FEATURES_PER_NEIGHBOR,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([original_low, feature_low]),
            high=np.concatenate([original_high, feature_high]),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info

    def _augment(self, observation: Any) -> np.ndarray:
        flat_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        features = np.tile(
            np.array([1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            (self.max_neighbors, 1),
        )
        base = self.unwrapped
        ego = getattr(base, "vehicle", None)
        road = getattr(base, "road", None)
        if ego is None or road is None:
            return np.concatenate([flat_observation, features.reshape(-1)])

        observation_type = base.observation_type
        neighbors = road.close_objects_to(
            getattr(observation_type, "observer_vehicle", ego),
            float(base.PERCEPTION_DISTANCE),
            count=self.max_neighbors,
            see_behind=bool(observation_type.see_behind),
            sort=True,
            vehicles_only=not bool(observation_type.include_obstacles),
        )
        ego_position = np.asarray(ego.position, dtype=np.float64)
        ego_velocity = np.asarray(ego.velocity, dtype=np.float64)
        for row, vehicle in enumerate(neighbors[-self.max_neighbors :]):
            features[row] = self._risk_features(
                ego_position,
                ego_velocity,
                np.asarray(vehicle.position, dtype=np.float64),
                np.asarray(vehicle.velocity, dtype=np.float64),
            )
        return np.concatenate([flat_observation, features.reshape(-1)]).astype(np.float32)

    def _risk_features(
        self,
        ego_position: np.ndarray,
        ego_velocity: np.ndarray,
        other_position: np.ndarray,
        other_velocity: np.ndarray,
    ) -> np.ndarray:
        relative_position = other_position - ego_position
        relative_velocity = other_velocity - ego_velocity
        distance = float(np.linalg.norm(relative_position))
        if distance <= np.finfo(np.float64).eps:
            closing_speed = 0.0
            radial_ttc = 0.0
        else:
            closing_speed = -float(np.dot(relative_position, relative_velocity)) / distance
            radial_ttc = distance / closing_speed if closing_speed > 0 else math.inf

        closest = pairwise_closest_approach(
            ego_position,
            ego_velocity,
            other_position,
            other_velocity,
            horizon=self.cpa_horizon,
        )
        ttc_normalized = (
            min(radial_ttc / self.ttc_scale, 1.0) if math.isfinite(radial_ttc) else 1.0
        )
        cpa_time_normalized = (
            min(closest.time / self.cpa_horizon, 1.0)
            if math.isfinite(closest.time)
            else 1.0
        )
        cpa_distance_normalized = (
            min(closest.distance / self.cpa_distance_scale, 1.0)
            if math.isfinite(closest.distance)
            else 1.0
        )
        return np.array(
            [
                np.clip(distance / self.range_scale, 0.0, 1.0),
                np.clip(closing_speed / self.relative_speed_scale, -1.0, 1.0),
                ttc_normalized,
                cpa_time_normalized,
                cpa_distance_normalized,
            ],
            dtype=np.float32,
        )
