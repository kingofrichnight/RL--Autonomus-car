"""Engineering-only, matched-information polar features for synchronized PPO.

This is NOT a ray-cast LiDAR or an independent sensor. It projects the existing
clipped kinematic rows into sectors; no extra actors, hidden state, occlusion
model or measurement noise are introduced. There is deliberately no training
entry point: the ongoing synchronized comparison must finish first.

Kept outside safeintent_rl while that experiment fingerprints all package Python
files. Importing this opt-in prototype cannot change its historical factories.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation

PROTOCOL = "predictive_post_spawn_sector_v1"
CELLS = 16
INPUT_SIZE = 115
OUTPUT_SIZE = INPUT_SIZE + 3 * CELLS
RANGE_SCALE = 200 * math.sqrt(2)
CLOSING_SPEED_SCALE = 80 * math.sqrt(2)
FEATURE_NAMES = ("presence", "normalized_center_range", "normalized_closing_speed")
FEATURE_RANGES = {"x": [-200, 200], "y": [-200, 200],
                  "vx": [-80, 80], "vy": [-80, 80]}


def sector_features(observation: np.ndarray) -> np.ndarray:
    """Return (16, 3) float32 features using ONLY the supplied 115-vector.

    Sector zero is centered on ego heading, with increasing angles following
    the simulator's x/y coordinate convention. Half-open bins have width pi/8.
    A sector retains the nearest present actor's CENTER, with exact-distance
    ties resolved by the lower native row index. This is not surface clearance.
    Positive closing speed means approach; empty sectors are (0, 1, 0).
    Coincident centers use sector zero and closing speed zero because a bearing
    and radial velocity are undefined there. Presence disambiguates this case.

    Decoding cannot recover clipping losses or unseen actors. Scales are fixed
    to the diagonal bounds implied by the frozen geometry observation ranges.
    """
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (INPUT_SIZE,) or not np.isfinite(values).all():
        raise ValueError("Requires a finite 115-value synchronized observation")
    native = values[:105].reshape(15, 7)
    presence = native[:, 0]
    if presence[0] != 1 or not np.isin(presence, [0, 1]).all():
        raise ValueError("Requires binary presence and a present ego row")
    present = native[presence == 1]
    if (np.abs(present[:, 1:5]) > 1).any():
        raise ValueError("Kinematic coordinates exceed the normalized bounds")
    cosine, sine = native[0, 5:7]
    if not math.isclose(math.hypot(cosine, sine), 1, rel_tol=0, abs_tol=1e-5):
        raise ValueError("Requires a unit ego heading")
    heading = math.atan2(sine, cosine)
    width = 2 * math.pi / CELLS
    result = np.tile([0.0, 1.0, 0.0], (CELLS, 1)).astype(np.float32)
    nearest = np.full(CELLS, np.inf)
    for row in native[1:]:
        if row[0] == 0:
            continue
        x, y = row[1:3] * 200
        vx, vy = row[3:5] * 80
        distance = math.hypot(x, y)
        if distance == 0:
            cell, closing_speed = 0, 0.0
        else:
            angle = (math.atan2(y, x) - heading + width / 2) % (2 * math.pi)
            cell = min(int(math.floor(angle / width)), CELLS - 1)
            closing_speed = -(x * vx + y * vy) / distance
        if distance < nearest[cell]:
            nearest[cell] = distance
            result[cell] = (1, np.clip(distance / RANGE_SCALE, 0, 1),
                            np.clip(closing_speed / CLOSING_SPEED_SCALE, -1, 1))
    return result


class SectorPredictiveObservation(gym.Wrapper):
    """Append polar features without changing the 115 existing inputs or dynamics.

    Opt-in and engineering-only. A future policy must be trained separately;
    the 163-value input is incompatible with existing 105/115-input checkpoints.
    """

    def __init__(self, env: SynchronizedPredictiveObservation) -> None:
        if not isinstance(env, SynchronizedPredictiveObservation):
            raise TypeError("Wrap SynchronizedPredictiveObservation directly")
        super().__init__(env)
        self._check_contract()
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low,
                                np.tile([0, 0, -1], CELLS)]).astype(np.float32),
            high=np.concatenate([env.observation_space.high,
                                 np.ones(3 * CELLS)]).astype(np.float32),
            dtype=np.float32,
        )

    def _check_contract(self) -> None:
        self.env._check_contract()
        if (self.unwrapped.observation_type.features_range != FEATURE_RANGES
                or self.env.observation_space.shape != (INPUT_SIZE,)
                or self.env.observation_space.dtype != np.float32):
            raise ValueError("Requires the frozen geometry normalization contract")

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        self._check_contract()  # Native reset recreates the observation type.
        original = np.asarray(observation)
        if original.dtype != np.float32:
            raise ValueError("Requires float32 inputs to preserve the prefix exactly")
        features = sector_features(original)
        result = np.concatenate([original, features.reshape(-1)])
        if not self.observation_space.contains(result):
            raise ValueError("Sector observation outside declared space")
        return result

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info
