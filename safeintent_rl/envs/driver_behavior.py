from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class DriverProfile:
    name: str
    desired_speed_scale: float
    acceleration_scale: float
    following_distance_scale: float
    time_gap_scale: float
    yield_probability: float


PROFILES: tuple[DriverProfile, ...] = (
    DriverProfile("cautious", 0.80, 0.75, 1.35, 1.30, 0.85),
    DriverProfile("normal", 1.00, 1.00, 1.00, 1.00, 0.55),
    DriverProfile("aggressive", 1.20, 1.30, 0.70, 0.70, 0.20),
)


def apply_profile(vehicle: Any, profile: DriverProfile) -> None:
    """Apply behavior parameters to one HighwayEnv NPC without exposing its label to PPO."""
    if hasattr(vehicle, "target_speed"):
        vehicle.target_speed = float(vehicle.target_speed) * profile.desired_speed_scale

    for attr in ("ACC_MAX", "COMFORT_ACC_MAX"):
        if hasattr(vehicle, attr):
            setattr(vehicle, attr, float(getattr(vehicle, attr)) * profile.acceleration_scale)
    if hasattr(vehicle, "DISTANCE_WANTED"):
        vehicle.DISTANCE_WANTED = (
            float(vehicle.DISTANCE_WANTED) * profile.following_distance_scale
        )
    if hasattr(vehicle, "TIME_WANTED"):
        vehicle.TIME_WANTED = float(vehicle.TIME_WANTED) * profile.time_gap_scale

    # Kept on the simulated vehicle for supervised labels; observations do not include it.
    vehicle.safeintent_driver_label = profile.name
    vehicle.safeintent_yield_probability = profile.yield_probability


class DriverBehaviorWrapper(gym.Wrapper):
    """Randomize hidden NPC personalities after each environment reset."""

    def __init__(
        self,
        env: gym.Env,
        probabilities: tuple[float, float, float] = (0.30, 0.45, 0.25),
    ) -> None:
        super().__init__(env)
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (3,) or np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
            raise ValueError("probabilities must be three non-negative values summing to one")
        self.probabilities = probs
        self.profile_counts: dict[str, int] = {}

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.profile_counts = {profile.name: 0 for profile in PROFILES}
        self._assign_unlabeled_profiles()
        info = dict(info)
        info["driver_profile_counts"] = dict(self.profile_counts)
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._assign_unlabeled_profiles()
        info = dict(info)
        info["driver_profile_counts"] = dict(self.profile_counts)
        return observation, reward, terminated, truncated, info

    def _assign_unlabeled_profiles(self) -> None:
        road = getattr(self.unwrapped, "road", None)
        ego = getattr(self.unwrapped, "vehicle", None)
        if road is None:
            return

        for vehicle in road.vehicles:
            if vehicle is ego or hasattr(vehicle, "safeintent_driver_label"):
                continue
            index = int(self.np_random.choice(len(PROFILES), p=self.probabilities))
            profile = PROFILES[index]
            apply_profile(vehicle, profile)
            self.profile_counts[profile.name] += 1
