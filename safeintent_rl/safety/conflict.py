from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ClosestApproach:
    """Constant-velocity closest-approach geometry for one traffic object."""

    time: float
    distance: float
    current_distance: float


def pairwise_closest_approach(
    ego_position: ArrayLike,
    ego_velocity: ArrayLike,
    other_position: ArrayLike,
    other_velocity: ArrayLike,
    *,
    horizon: float = 3.0,
    minimum_distance: float = 0.1,
) -> ClosestApproach:
    """Predict closest approach within a finite horizon using constant velocity."""
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(minimum_distance) or minimum_distance <= 0:
        raise ValueError("minimum_distance must be finite and positive")

    relative_position = np.asarray(other_position, dtype=np.float64) - np.asarray(
        ego_position, dtype=np.float64
    )
    relative_velocity = np.asarray(other_velocity, dtype=np.float64) - np.asarray(
        ego_velocity, dtype=np.float64
    )
    current_distance = float(np.linalg.norm(relative_position))
    if current_distance <= minimum_distance:
        return ClosestApproach(time=0.0, distance=0.0, current_distance=current_distance)

    relative_speed_squared = float(np.dot(relative_velocity, relative_velocity))
    if relative_speed_squared <= np.finfo(np.float64).eps:
        return ClosestApproach(
            time=math.inf,
            distance=math.inf,
            current_distance=current_distance,
        )

    time = -float(np.dot(relative_position, relative_velocity)) / relative_speed_squared
    if time < 0 or time > horizon:
        return ClosestApproach(
            time=math.inf,
            distance=math.inf,
            current_distance=current_distance,
        )

    distance = float(np.linalg.norm(relative_position + relative_velocity * time))
    return ClosestApproach(time=time, distance=distance, current_distance=current_distance)


def minimum_closest_approach(
    ego: object,
    others: list[object],
    *,
    horizon: float = 3.0,
    max_range: float = 60.0,
) -> ClosestApproach:
    """Return the smallest predicted miss distance among nearby traffic objects."""
    candidates = nearby_closest_approaches(
        ego,
        others,
        horizon=horizon,
        max_range=max_range,
    )
    if not candidates:
        return ClosestApproach(time=math.inf, distance=math.inf, current_distance=math.inf)
    return min(candidates, key=lambda item: (item.distance, item.time))


def nearby_closest_approaches(
    ego: object,
    others: list[object],
    *,
    horizon: float = 3.0,
    max_range: float = 60.0,
) -> tuple[ClosestApproach, ...]:
    """Return every finite closest approach for traffic within the requested range."""
    if not math.isfinite(max_range) or max_range <= 0:
        raise ValueError("max_range must be finite and positive")

    ego_position = np.asarray(getattr(ego, "position"), dtype=np.float64)
    ego_velocity = np.asarray(getattr(ego, "velocity"), dtype=np.float64)
    candidates: list[ClosestApproach] = []
    for other in others:
        if other is ego:
            continue
        other_position = np.asarray(getattr(other, "position"), dtype=np.float64)
        current_distance = float(np.linalg.norm(other_position - ego_position))
        if current_distance > max_range:
            continue
        candidate = pairwise_closest_approach(
            ego_position,
            ego_velocity,
            other_position,
            getattr(other, "velocity"),
            horizon=horizon,
        )
        if math.isfinite(candidate.time) and math.isfinite(candidate.distance):
            candidates.append(candidate)
    return tuple(candidates)
