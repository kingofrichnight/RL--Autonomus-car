from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def pairwise_ttc(
    ego_position: ArrayLike,
    ego_velocity: ArrayLike,
    other_position: ArrayLike,
    other_velocity: ArrayLike,
    *,
    minimum_distance: float = 0.1,
) -> float:
    """Estimate TTC using radial closing speed; return infinity when not closing."""
    relative_position = np.asarray(other_position, dtype=np.float64) - np.asarray(
        ego_position, dtype=np.float64
    )
    relative_velocity = np.asarray(other_velocity, dtype=np.float64) - np.asarray(
        ego_velocity, dtype=np.float64
    )
    distance = float(np.linalg.norm(relative_position))
    if distance <= minimum_distance:
        return 0.0

    line_of_sight = relative_position / distance
    closing_speed = -float(np.dot(relative_velocity, line_of_sight))
    if closing_speed <= 0:
        return math.inf
    return distance / closing_speed


def minimum_ttc(ego: object, others: list[object], max_range: float = 60.0) -> float:
    """Return the smallest finite radial TTC among nearby simulated vehicles."""
    result = math.inf
    ego_position = np.asarray(getattr(ego, "position"), dtype=np.float64)
    ego_velocity = np.asarray(getattr(ego, "velocity"), dtype=np.float64)
    for other in others:
        if other is ego:
            continue
        other_position = np.asarray(getattr(other, "position"), dtype=np.float64)
        if np.linalg.norm(other_position - ego_position) > max_range:
            continue
        value = pairwise_ttc(
            ego_position,
            ego_velocity,
            other_position,
            getattr(other, "velocity"),
        )
        result = min(result, value)
    return result

