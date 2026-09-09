from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np


class EgoTargetSpeedObservation(gym.Wrapper):
    """Append the ego controller's current speed target without changing its actions."""

    def __init__(self, env: gym.Env, *, scale: float = 9.0) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("EgoTargetSpeedObservation requires a Box observation space")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Target-speed scale must be finite and positive")
        self.scale = float(scale)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([
                np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1),
                np.array([0.0], dtype=np.float32),
            ]),
            high=np.concatenate([
                np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1),
                np.array([1.0], dtype=np.float32),
            ]),
            dtype=np.float32,
        )

    def _augment(self, observation: Any) -> np.ndarray:
        target = float(self.unwrapped.vehicle.target_speed)
        if not math.isfinite(target) or not 0 <= target <= self.scale:
            raise ValueError("Ego target speed must be finite and within the recorded scale")
        prefix = np.asarray(observation, dtype=np.float32).reshape(-1)
        return np.concatenate([prefix, np.array([target / self.scale], dtype=np.float32)])

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info
