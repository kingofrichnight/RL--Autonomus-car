"""Action-conditioned forecasts, not a shield or calibrated collision probability."""

from __future__ import annotations

import copy
import math
from typing import Any

import gymnasium as gym
import numpy as np
from highway_env.road.road import Road
from highway_env.vehicle.controller import MDPVehicle


def rectangle_separation(
    ego_position: np.ndarray,
    ego_heading: float,
    ego_size: tuple[float, float],
    positions: np.ndarray,
    headings: np.ndarray,
    sizes: np.ndarray,
) -> np.ndarray:
    """Signed maximum separating-axis gap (metres), not Euclidean clearance.

    Nonpositive means intersecting/touching rectangles. Positive is a lower
    bound on Euclidean separation. All inputs describe current/predicted poses.
    """
    if len(positions) == 0:
        return np.empty(0)
    eu = np.array([math.cos(ego_heading), math.sin(ego_heading)])
    ev = np.array([-eu[1], eu[0]])
    ou = np.column_stack([np.cos(headings), np.sin(headings)])
    ov = np.column_stack([-ou[:, 1], ou[:, 0]])
    axes = np.stack([np.broadcast_to(eu, ou.shape), np.broadcast_to(ev, ou.shape),
                     ou, ov], axis=1)
    offset = np.abs(np.einsum("ni,nai->na", positions - ego_position, axes))
    ego_radius = (ego_size[0] * np.abs(axes @ eu)
                  + ego_size[1] * np.abs(axes @ ev)) / 2
    other_radius = (sizes[:, 0, None] * np.abs(np.einsum("ni,nai->na", ou, axes))
                    + sizes[:, 1, None] * np.abs(np.einsum("ni,nai->na", ov, axes))) / 2
    return np.max(offset - ego_radius - other_radius, axis=1)


class PredictiveSafetyObservation(gym.Wrapper):
    """Append target speed and three forecasts for each longitudinal action.

    Forecast: apply action once, then maintain its target for H seconds. Ego
    uses isolated HighwayEnv controller/dynamics and its known navigation route.
    Other actors use only observed poses, velocities and known dimensions.
    Their hidden routes, intent labels, controllers and future RNG are not used.
    """

    FEATURE_NAMES = ("minimum_margin_gap", "first_margin_conflict", "travel_distance")

    def __init__(self, env: gym.Env, *, horizon: float = 3.0,
                 margin: float = 0.5, uncertainty_growth: float = 0.25,
                 clearance_scale: float = 10.0, speed_scale: float = 9.0,
                 max_neighbors: int = 14) -> None:
        super().__init__(env)
        for name, value in {"horizon": horizon, "clearance_scale": clearance_scale,
                            "speed_scale": speed_scale}.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in {"margin": margin, "uncertainty_growth": uncertainty_growth}.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(max_neighbors) is not int or max_neighbors < 1:
            raise ValueError("max_neighbors must be a positive integer")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("PredictiveSafetyObservation requires Box observations")
        self.horizon = float(horizon)
        self.margin = float(margin)
        self.uncertainty_growth = float(uncertainty_growth)
        self.clearance_scale = float(clearance_scale)
        self.speed_scale = float(speed_scale)
        self.max_neighbors = max_neighbors
        base = self.unwrapped
        self.actions = tuple(base.action_type.actions[i] for i in range(env.action_space.n))
        if self.actions != ("SLOWER", "IDLE", "FASTER"):
            raise ValueError("predictive v1 requires SLOWER/IDLE/FASTER action order")
        self.frequency = int(base.config["simulation_frequency"])
        if self.frequency <= 0 or not math.isclose(horizon * self.frequency,
                                                  round(horizon * self.frequency)):
            raise ValueError("horizon must contain an integer number of simulation steps")
        obs_type = base.observation_type
        if (getattr(obs_type, "order", None) != "sorted"
                or max_neighbors > obs_type.vehicles_count - 1):
            raise ValueError("requires sorted kinematics with sufficient observed slots")
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low.reshape(-1), [0],
                                np.tile([-1, 0, 0], 3)]).astype(np.float32),
            high=np.concatenate([env.observation_space.high.reshape(-1), np.ones(10)])
            .astype(np.float32), dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if kwargs.get("options"):
            raise ValueError("predictive protocol forbids reset configuration overrides")
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        details = dict(info)
        details["predictive_collision_cost"] = float(self.unwrapped.vehicle.crashed)
        # No intervention or extra reward term: PPO chooses the executed action.
        return self._augment(obs), reward, terminated, truncated, details

    def _augment(self, observation: Any) -> np.ndarray:
        ego = self.unwrapped.vehicle
        if not isinstance(ego, MDPVehicle):
            raise TypeError("predictive v1 requires HighwayEnv MDPVehicle")
        target = np.clip(ego.target_speed / self.speed_scale, 0, 1)
        return np.concatenate([np.asarray(observation, dtype=np.float32).reshape(-1),
                               [target], self.forecast().reshape(-1)]).astype(np.float32)

    def forecast(self) -> np.ndarray:
        base = self.unwrapped
        ego = base.vehicle
        obs_type = base.observation_type
        neighbors = base.road.close_objects_to(
            obs_type.observer_vehicle, float(base.PERCEPTION_DISTANCE),
            count=self.max_neighbors, see_behind=bool(obs_type.see_behind), sort=True,
            vehicles_only=not bool(obs_type.include_obstacles),
        )
        positions = np.array([v.position for v in neighbors], dtype=float).reshape(-1, 2)
        velocities = np.array([v.velocity for v in neighbors], dtype=float).reshape(-1, 2)
        headings = np.array([v.heading for v in neighbors], dtype=float)
        sizes = np.array([[v.LENGTH, v.WIDTH] for v in neighbors], dtype=float).reshape(-1, 2)
        if not all(np.isfinite(a).all() for a in (positions, velocities, headings, sizes)):
            raise ValueError("nonfinite observed traffic state")
        # Deliberately exclude actual traffic objects, behavior models and RNG.
        road = Road(network=copy.deepcopy(base.road.network),
                    np_random=np.random.default_rng(0), record_history=False)
        template = copy.deepcopy(ego, memo={id(base.road): road})
        result = []
        for action in self.actions:
            ghost = copy.deepcopy(template)
            min_gap = math.inf
            first_conflict = self.horizon
            travel = 0.0
            for frame in range(round(self.horizon * self.frequency) + 1):
                t = frame / self.frequency
                if frame:
                    before = ghost.position.copy()
                    # One initial meta-action; later control updates maintain its target.
                    ghost.act(action if frame == 1 else None)
                    ghost.step(1 / self.frequency)
                    travel += float(np.linalg.norm(ghost.position - before))
                gaps = rectangle_separation(
                    ghost.position, ghost.heading, (ghost.LENGTH, ghost.WIDTH),
                    positions + velocities * t, headings, sizes,
                ) - (self.margin + self.uncertainty_growth * t)
                gap = float(np.min(gaps)) if gaps.size else math.inf
                min_gap = min(min_gap, gap)
                if gap <= 0:
                    first_conflict = min(first_conflict, t)
            result.append([np.clip(min_gap / self.clearance_scale, -1, 1),
                           first_conflict / self.horizon,
                           np.clip(travel / (self.speed_scale * self.horizon), 0, 1)])
        return np.asarray(result, dtype=np.float32)
