from __future__ import annotations

from typing import Any


class RuleBasedAgent:
    """Simple TTC baseline using HighwayEnv high-level action names."""

    def __init__(self, env: Any, brake_ttc: float = 2.0, accelerate_ttc: float = 4.0) -> None:
        self.env = env
        self.brake_ttc = brake_ttc
        self.accelerate_ttc = accelerate_ttc

    def predict(self, _observation: Any, min_ttc: float = float("inf")) -> int:
        actions = getattr(self.env.unwrapped.action_type, "actions", {})
        names = {str(name).upper(): int(index) for index, name in actions.items()}
        if min_ttc <= self.brake_ttc and "SLOWER" in names:
            return names["SLOWER"]
        if min_ttc >= self.accelerate_ttc and "FASTER" in names:
            return names["FASTER"]
        return names.get("IDLE", next(iter(actions), 0))

