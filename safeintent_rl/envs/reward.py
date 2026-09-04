from __future__ import annotations

import math
from typing import Any

import gymnasium as gym

from safeintent_rl.safety.ttc import minimum_ttc


class RouteProgressRewardWrapper(gym.Wrapper):
    """Add route progress, time cost, and optional TTC risk shaping."""

    def __init__(
        self,
        env: gym.Env,
        *,
        progress_weight: float = 2.0,
        time_penalty: float = 0.005,
        arrival_distance: float = 25.0,
        risk_weight: float = 0.0,
        risk_ttc_threshold: float = 2.0,
    ) -> None:
        super().__init__(env)
        if progress_weight < 0:
            raise ValueError("progress_weight must be non-negative")
        if time_penalty < 0:
            raise ValueError("time_penalty must be non-negative")
        if arrival_distance <= 0:
            raise ValueError("arrival_distance must be positive")
        if risk_weight < 0:
            raise ValueError("risk_weight must be non-negative")
        if risk_ttc_threshold <= 0:
            raise ValueError("risk_ttc_threshold must be positive")

        self.progress_weight = float(progress_weight)
        self.time_penalty = float(time_penalty)
        self.arrival_distance = float(arrival_distance)
        self.risk_weight = float(risk_weight)
        self.risk_ttc_threshold = float(risk_ttc_threshold)
        self._route: list[tuple[Any, Any, int]] = []
        self._lane_lengths: list[float] = []
        self._goal_progress = 0.0
        self._start_progress = 0.0
        self._previous_progress = 0.0
        self._remaining_distance = 1.0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._capture_route()
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        current_progress = self._absolute_progress()
        distance_delta = max(0.0, current_progress - self._previous_progress)
        normalized_delta = distance_delta / self._remaining_distance
        progress_reward = self.progress_weight * normalized_delta
        risk_ttc, risk_fraction, risk_penalty = self._risk_penalty()
        shaped_reward = (
            float(base_reward)
            + progress_reward
            - self.time_penalty
            - risk_penalty
        )
        self._previous_progress = max(self._previous_progress, current_progress)

        details = dict(info)
        details.update(
            {
                "base_reward": float(base_reward),
                "route_progress": self._normalized_progress(current_progress),
                "route_progress_delta": normalized_delta,
                "progress_reward": progress_reward,
                "time_penalty": self.time_penalty,
                "reward_min_ttc": risk_ttc,
                "risk_fraction": risk_fraction,
                "risk_penalty": risk_penalty,
                "shaped_reward": shaped_reward,
            }
        )
        return observation, shaped_reward, terminated, truncated, details

    def _risk_penalty(self) -> tuple[float, float, float]:
        if self.risk_weight == 0.0:
            return math.inf, 0.0, 0.0

        base = self.env.unwrapped
        ttc = minimum_ttc(base.vehicle, list(base.road.vehicles))
        if not math.isfinite(ttc):
            return ttc, 0.0, 0.0

        risk_fraction = max(
            0.0,
            1.0 - (ttc / self.risk_ttc_threshold),
        )
        return ttc, risk_fraction, self.risk_weight * risk_fraction

    def _capture_route(self) -> None:
        base = self.env.unwrapped
        vehicle = base.vehicle
        route = list(getattr(vehicle, "route", None) or [vehicle.lane_index])
        self._route = [tuple(index) for index in route]
        self._lane_lengths = []
        for lane_index in self._route:
            lane = base.road.network.get_lane(lane_index)
            self._lane_lengths.append(float(lane.length))

        if self._lane_lengths:
            final_length = min(self._lane_lengths[-1], self.arrival_distance)
            self._goal_progress = sum(self._lane_lengths[:-1]) + final_length
        else:
            self._goal_progress = 0.0

        self._start_progress = self._absolute_progress()
        self._previous_progress = self._start_progress
        self._remaining_distance = max(
            self._goal_progress - self._start_progress,
            1.0,
        )

    def _absolute_progress(self) -> float:
        if not self._route:
            return self._previous_progress

        base = self.env.unwrapped
        vehicle = base.vehicle
        current_lane = tuple(vehicle.lane_index)
        route_position = next(
            (
                index
                for index, lane_index in enumerate(self._route)
                if lane_index[:2] == current_lane[:2]
            ),
            None,
        )
        if route_position is None:
            return self._previous_progress

        longitudinal = float(vehicle.lane.local_coordinates(vehicle.position)[0])
        lane_limit = self._lane_lengths[route_position]
        if route_position == len(self._route) - 1:
            lane_limit = min(lane_limit, self.arrival_distance)
        longitudinal = min(max(longitudinal, 0.0), lane_limit)
        progress = sum(self._lane_lengths[:route_position]) + longitudinal
        return min(progress, self._goal_progress)

    def _normalized_progress(self, absolute_progress: float) -> float:
        return min(
            max((absolute_progress - self._start_progress) / self._remaining_distance, 0.0),
            1.0,
        )
