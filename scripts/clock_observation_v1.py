"""Opt-in 116-input zero/remaining-time observations; no experiment entry point.

The clock is information for a newly trained PPO policy, not an urgency reward,
action override, new sensor, or change to native termination/bootstrapping.
See V3_CLOCK_OBSERVATION_V1.md. Training still requires its later release gate.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import gymnasium as gym
import numpy as np

from safeintent_rl.sensors.predictive import PredictiveSafetyObservation
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation

INPUT_SIZE = 115
OUTPUT_SIZE = 116
DURATION = 30.0
PROTOCOLS = {
    "zero": "predictive_post_spawn_clock_zero_v1",
    "clock": "predictive_post_spawn_clock_remaining_v1",
}
FEATURE_RANGES = {"x": [-200, 200], "y": [-200, 200],
                  "vx": [-80, 80], "vy": [-80, 80]}
PREDICTIVE_SETTINGS = {
    "horizon": 3.0, "margin": 0.5, "uncertainty_growth": 0.25,
    "clearance_scale": 10.0, "speed_scale": 9.0, "max_neighbors": 14, "frequency": 15,
}


def _finite_real(value: Any) -> bool:
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value))


class ClockPredictiveObservation(gym.Wrapper):
    """Append one value to the frozen synchronized predictive parent.

    All 115 float32 inputs (460 bytes) are copied exactly, including signed zero.
    Only actual elapsed simulation time is read, after the delegate returns.
    Validation is identical in both arms. Full reward/config/runtime provenance
    and checkpoint compatibility remain responsibilities of the future runner.
    """

    def __init__(self, env: SynchronizedPredictiveObservation, arm: str = "zero") -> None:
        if not isinstance(arm, str) or arm not in PROTOCOLS:
            raise ValueError("Clock arm must be 'zero' or 'clock'")
        if not isinstance(env, SynchronizedPredictiveObservation):
            raise TypeError("Wrap SynchronizedPredictiveObservation directly")
        super().__init__(env)
        self.arm = arm
        self.protocol = PROTOCOLS[arm]
        self._check_contract()
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, np.array([0], dtype=np.float32)]),
            high=np.concatenate([env.observation_space.high, np.array([1], dtype=np.float32)]),
            dtype=np.float32,
        )

    def _check_contract(self) -> None:
        if not isinstance(self.env, SynchronizedPredictiveObservation) or not isinstance(
            self.env.env, PredictiveSafetyObservation
        ):
            raise TypeError("Requires the direct synchronized/predictive parent chain")
        self.env._check_contract()  # Native reset recreates its observation type.
        predictive = self.env.env
        base = self.unwrapped
        space = self.env.observation_space
        if (not isinstance(space, gym.spaces.Box) or space.shape != (INPUT_SIZE,)
                or space.dtype != np.float32
                or base.observation_type.features_range != FEATURE_RANGES):
            raise ValueError("Requires frozen float32 geometry normalization and 115-input Box")
        if self._observation_space is not None and (
            not np.array_equal(space.low, self.observation_space.low[:INPUT_SIZE])
            or not np.array_equal(space.high, self.observation_space.high[:INPUT_SIZE])
        ):
            raise ValueError("Parent observation bounds changed after construction")
        for name, expected in {"duration": DURATION, "simulation_frequency": 15,
                               "policy_frequency": 5}.items():
            value = base.config.get(name)
            if not _finite_real(value) or value != expected:
                raise ValueError(f"Requires frozen {name}={expected}")
        for name, expected in PREDICTIVE_SETTINGS.items():
            value = getattr(predictive, name, None)
            if not _finite_real(value) or value != expected:
                raise ValueError(f"Requires frozen predictive {name}={expected}")
        action = base.action_type
        if (not isinstance(self.action_space, gym.spaces.Discrete)
                or self.action_space.n != 3 or self.action_space.start != 0
                or action.actions != {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
                or predictive.actions != ("SLOWER", "IDLE", "FASTER")
                or not action.longitudinal or action.lateral
                or not np.array_equal(action.target_speeds, [0, 4.5, 9])):
            raise ValueError("Requires frozen longitudinal SLOWER/IDLE/FASTER actions")
        if self.arm not in PROTOCOLS or self.protocol != PROTOCOLS[self.arm]:
            raise ValueError("Clock arm/protocol identity changed")

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        self._check_contract()
        original = np.asarray(observation)
        if (original.dtype != np.float32 or original.shape != (INPUT_SIZE,)
                or not np.isfinite(original).all()
                or not self.env.observation_space.contains(original)
                or (np.abs(original[:105]) > 1).any()):
            raise ValueError("Requires finite, in-bounds float32 predictive observation")
        elapsed = self.unwrapped.time
        if not _finite_real(elapsed) or elapsed < 0:
            raise ValueError("Simulation time must be finite, numeric and nonnegative")
        remaining = np.float32(0.0)
        if self.arm == "clock":
            remaining = np.float32(min(1.0, max(0.0, (DURATION - elapsed) / DURATION)))
        result = np.empty(OUTPUT_SIZE, dtype=np.float32)
        result[:INPUT_SIZE] = original
        result[-1] = remaining
        return result

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info
