from __future__ import annotations

from typing import Any

import gymnasium as gym

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

