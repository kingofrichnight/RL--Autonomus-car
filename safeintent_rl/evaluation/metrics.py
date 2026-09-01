from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    reward: float
    length: int
    success: bool
    collision: bool
    travel_time: float
    min_ttc: float
    unsafe_ttc_events: int
    safety_interventions: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_success(env: Any, final_info: dict[str, Any]) -> bool:
    """Detect arrival across HighwayEnv versions and other Gymnasium environments."""
    for key in ("is_success", "arrived"):
        if key in final_info:
            return bool(final_info[key])

    base = getattr(env, "unwrapped", env)
    has_arrived = getattr(base, "has_arrived", None)
    vehicle = getattr(base, "vehicle", None)
    if callable(has_arrived) and vehicle is not None:
        return bool(has_arrived(vehicle))
    return False


def summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    if not episodes:
        raise ValueError("At least one episode is required")

    def mean(field: str) -> float:
        return float(np.mean([float(getattr(item, field)) for item in episodes]))

    finite_ttc = [item.min_ttc for item in episodes if np.isfinite(item.min_ttc)]
    return {
        "episodes": float(len(episodes)),
        "mean_reward": mean("reward"),
        "mean_length": mean("length"),
        "success_rate": mean("success"),
        "collision_rate": mean("collision"),
        "mean_travel_time": mean("travel_time"),
        "mean_min_ttc": float(np.mean(finite_ttc)) if finite_ttc else float("inf"),
        "mean_unsafe_ttc_events": mean("unsafe_ttc_events"),
        "mean_safety_interventions": mean("safety_interventions"),
    }
