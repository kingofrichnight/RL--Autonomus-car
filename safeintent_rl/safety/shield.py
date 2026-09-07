from __future__ import annotations

import math
from typing import Any

import gymnasium as gym

from safeintent_rl.safety.conflict import (
    ClosestApproach,
    nearby_closest_approaches,
)
from safeintent_rl.safety.ttc import minimum_ttc


class TTCSafetyShield(gym.Wrapper):
    """Override unsafe accelerate/idle meta-actions with SLOWER when TTC is critical."""

    def __init__(self, env: gym.Env, ttc_threshold: float = 2.0) -> None:
        super().__init__(env)
        if ttc_threshold <= 0:
            raise ValueError("ttc_threshold must be positive")
        self.ttc_threshold = float(ttc_threshold)
        self.interventions = 0
        self.decisions = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.interventions = 0
        self.decisions = 0
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.decisions += 1
        ttc = self.current_min_ttc()
        original_action = int(action) if hasattr(action, "__int__") else action
        safe_action, intervened = self._safe_action(original_action, ttc)
        if intervened:
            self.interventions += 1

        observation, reward, terminated, truncated, info = self.env.step(safe_action)
        info = dict(info)
        info.update(
            {
                "min_ttc": ttc,
                "safety_intervened": intervened,
                "proposed_action": original_action,
                "executed_action": safe_action,
                "safety_intervention_rate": self.interventions / self.decisions,
            }
        )
        return observation, reward, terminated, truncated, info

    def current_min_ttc(self) -> float:
        base = self.unwrapped
        ego = getattr(base, "vehicle", None)
        road = getattr(base, "road", None)
        if ego is None or road is None:
            return float("inf")
        return minimum_ttc(ego, list(road.vehicles))

    def _safe_action(self, action: Any, ttc: float) -> tuple[Any, bool]:
        if ttc > self.ttc_threshold:
            return action, False

        action_type = getattr(self.unwrapped, "action_type", None)
        actions = getattr(action_type, "actions", {})
        action_name = str(actions.get(action, "")).upper()
        slower_index = next(
            (index for index, name in actions.items() if str(name).upper() == "SLOWER"),
            None,
        )
        unsafe_names = {"FASTER", "IDLE"}
        if action_name in unsafe_names and slower_index is not None:
            return int(slower_index), True
        return action, False


class CPAAccelerationShield(gym.Wrapper):
    """Override CPA-conflicted acceleration with a configured longitudinal action."""

    def __init__(
        self,
        env: gym.Env,
        *,
        time_threshold: float = 2.0,
        distance_threshold: float = 3.0,
        horizon: float = 3.0,
        max_range: float = 60.0,
        override_action: str = "IDLE",
    ) -> None:
        super().__init__(env)
        parameters = {
            "time_threshold": time_threshold,
            "distance_threshold": distance_threshold,
            "horizon": horizon,
            "max_range": max_range,
        }
        for name, value in parameters.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if time_threshold > horizon:
            raise ValueError("time_threshold cannot exceed horizon")
        normalized_override = override_action.upper()
        if normalized_override not in {"IDLE", "SLOWER"}:
            raise ValueError("override_action must be IDLE or SLOWER")

        self.time_threshold = float(time_threshold)
        self.distance_threshold = float(distance_threshold)
        self.horizon = float(horizon)
        self.max_range = float(max_range)
        self.override_action = normalized_override
        self.interventions = 0
        self.decisions = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.interventions = 0
        self.decisions = 0
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.decisions += 1
        original_action = int(action) if hasattr(action, "__int__") else action
        approaches = self.current_closest_approaches()
        matching = tuple(
            item
            for item in approaches
            if item.time <= self.time_threshold
            and item.distance <= self.distance_threshold
        )
        safe_action, intervened = self._safe_action(original_action, matching)
        if intervened:
            self.interventions += 1

        observation, reward, terminated, truncated, info = self.env.step(safe_action)
        info = dict(info)
        closest_match = min(matching, key=lambda item: (item.time, item.distance), default=None)
        info.update(
            {
                "safety_intervened": intervened,
                "proposed_action": original_action,
                "executed_action": safe_action,
                "safety_intervention_rate": self.interventions / self.decisions,
                "cpa_conflict": bool(matching),
                "cpa_conflict_time": (
                    closest_match.time if closest_match is not None else math.inf
                ),
                "cpa_conflict_distance": (
                    closest_match.distance if closest_match is not None else math.inf
                ),
            }
        )
        return observation, reward, terminated, truncated, info

    def current_closest_approaches(self) -> tuple[ClosestApproach, ...]:
        base = self.unwrapped
        ego = getattr(base, "vehicle", None)
        road = getattr(base, "road", None)
        if ego is None or road is None:
            return ()
        return nearby_closest_approaches(
            ego,
            list(road.vehicles),
            horizon=self.horizon,
            max_range=self.max_range,
        )

    def _safe_action(
        self,
        action: Any,
        matching: tuple[ClosestApproach, ...],
    ) -> tuple[Any, bool]:
        if not matching:
            return action, False

        action_type = getattr(self.unwrapped, "action_type", None)
        actions = getattr(action_type, "actions", {})
        action_name = str(actions.get(action, "")).upper()
        override_index = next(
            (
                index
                for index, name in actions.items()
                if str(name).upper() == self.override_action
            ),
            None,
        )
        if action_name == "FASTER" and override_index is not None:
            return int(override_index), True
        return action, False
