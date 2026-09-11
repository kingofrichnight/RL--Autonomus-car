"""Opt-in scene alignment; historical predictive observations remain unchanged."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from safeintent_rl.sensors.predictive import PredictiveSafetyObservation


class SynchronizedPredictiveObservation(gym.Wrapper):
    """Refresh native kinematics after clearing/spawning, beside the same forecast.

    HighwayEnv IntersectionEnv returns kinematics sampled before clearing and
    spawning vehicles. The inner predictive wrapper forecasts after those
    operations. Refresh only the 105 native inputs from that post-step scene;
    retain its target speed and nine forecasts byte-for-byte. No second forecast,
    physics step, action override, reward change or learned component is added.

    This changes policy-input semantics and requires a separately trained model.
    It must not be silently applied when scoring historical checkpoints.
    """

    FEATURES = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]

    def __init__(self, env: PredictiveSafetyObservation) -> None:
        if not isinstance(env, PredictiveSafetyObservation):
            raise TypeError("Wrap PredictiveSafetyObservation directly")
        super().__init__(env)
        self._check_contract()

    def _check_contract(self) -> None:
        obs_type = self.unwrapped.observation_type
        if (type(obs_type).__name__ != "KinematicObservation"
                or obs_type.features != self.FEATURES
                or obs_type.vehicles_count != 15 or obs_type.order != "sorted"
                or obs_type.absolute or not obs_type.normalize
                or not obs_type.clip or not obs_type.include_obstacles
                or obs_type.see_behind
                or self.env.max_neighbors != 14
                or self.env.observation_space.shape != (115,)):
            raise ValueError("Requires frozen sorted 15-vehicle predictive input contract")

    def _refresh(self, observation: np.ndarray) -> np.ndarray:
        self._check_contract()  # Native reset recreates its observation type.
        original = np.asarray(observation)
        if original.shape != (115,) or not np.isfinite(original).all():
            raise ValueError("Invalid predictive observation")
        native = np.asarray(self.unwrapped.observation_type.observe(), dtype=np.float32)
        if native.shape != (15, 7) or not np.isfinite(native).all():
            raise ValueError("Invalid refreshed kinematics")
        result = original.copy()
        result[:105] = native.reshape(-1)
        if not self.observation_space.contains(result):
            raise ValueError("Refreshed observation outside declared space")
        return result

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        observation, info = self.env.reset(**kwargs)
        return self._refresh(observation), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._refresh(observation), reward, terminated, truncated, info
