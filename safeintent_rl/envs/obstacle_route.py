"""Isolated reference stage 1: two lanes, static blockages, no NPCs/pedestrians.

This is a new benchmark, not an easier replacement for intersection-v2.
Full road occupancy is known; there is no camera or occlusion model.
"""

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Obstacle

from safeintent_rl.agents.route_graph import RouteEdge, shortest_route
from safeintent_rl.config import load_config

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2] / "configs" / "obstacle_route_v1.json"


class LaneBlockage(Obstacle):
    LENGTH = 4.0
    WIDTH = 4.0


class ObstacleRouteEnv(AbstractEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 3,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "absolute": False,
                    "normalize": True,
                    "order": "sorted",
                    "observe_intentions": False,
                    "include_obstacles": True,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": True,
                    "target_speeds": [0.0, 4.5, 9.0],
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
            }
        )
        return config

    def __init__(self, protocol=None, *, alternate_available=True, render_mode=None):
        self.protocol = deepcopy(
            protocol if protocol is not None else load_config(DEFAULT_PROTOCOL)
        )
        if set(self.protocol) != set(load_config(DEFAULT_PROTOCOL)):
            raise ValueError("Protocol fields must exactly match obstacle_route_v1")
        if self.protocol["schema_version"] != 1 or self.protocol["family"] != "obstacle_route_v1":
            raise ValueError("Unsupported obstacle protocol")
        if type(alternate_available) is not bool:
            raise ValueError("alternate_available must be boolean")
        self.alternate_available = alternate_available
        p = self.protocol
        numeric = [v for k, v in p.items() if k not in {"family", "obstacle_x_range_m"}]
        if not all(np.isfinite(v) for v in numeric):
            raise ValueError("Protocol values must be finite")
        lo, hi = p["obstacle_x_range_m"]
        if not (p["ego_x_m"] + 30 < lo <= hi < p["goal_x_m"] < p["road_length_m"]):
            raise ValueError("Invalid road, goal, or blockage positions")
        positive = ("duration", "safe_stop_speed_mps", "safe_stop_hold_s", "clearance_margin_m")
        if any(p[key] <= 0 for key in positive) or not 0 < p["ego_speed_mps"] <= 9:
            raise ValueError("Invalid speed, duration, or clearance")
        if p["simulation_frequency"] != 15 or p["policy_frequency"] != 5:
            raise ValueError("V1 requires 15 Hz simulation and 5 Hz decisions")
        if p["collision_reward"] >= 0 or p["offroad_reward"] >= 0:
            raise ValueError("Collision and offroad rewards must be negative")
        if min(p[k] for k in ("progress_reward", "time_cost_per_s", "lane_change_cost_s")) < 0:
            raise ValueError("Progress weight and costs must be nonnegative")
        if p["ego_x_m"] < 0 or p["duration"] * 5 < 1:
            raise ValueError("Initial position and duration must permit a valid episode")
        super().__init__(
            config={k: p[k] for k in ("simulation_frequency", "policy_frequency")},
            render_mode=render_mode,
        )

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Reset options cannot override the versioned obstacle protocol")
        obs, info = super().reset(seed=seed)
        # Upstream samples an arbitrary action solely for reset info. Do not expose RNG noise.
        info["action"] = None
        return obs, info

    def _reset(self):
        p = self.protocol
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                lanes=2, length=p["road_length_m"], speed_limit=9
            ),
            np_random=self.np_random,
        )
        self.vehicle = self.action_type.vehicle_class(
            self.road, [p["ego_x_m"], 0], speed=p["ego_speed_mps"]
        )
        self.road.vehicles = [self.vehicle]
        self.obstacle_x = float(self.np_random.uniform(*p["obstacle_x_range_m"]))
        self.blocked_lanes = (True, not self.alternate_available)
        for lane_id, blocked in enumerate(self.blocked_lanes):
            if blocked:
                self.road.objects.append(LaneBlockage(self.road, [self.obstacle_x, 4 * lane_id]))
        edges = []
        for lane_id, blocked in enumerate(self.blocked_lanes):
            node = f"lane{lane_id}"
            edges.append(
                RouteEdge(
                    "start",
                    node,
                    self.obstacle_x - p["ego_x_m"],
                    9,
                    lane_change_cost_s=lane_id * p["lane_change_cost_s"],
                )
            )
            edges.append(
                RouteEdge(node, "goal", p["goal_x_m"] - self.obstacle_x, 9, blocked=blocked)
            )
        self.route_plan = shortest_route(edges, "start", "goal")
        self.route_target_lane = int(self.route_plan[1][-1]) if self.route_plan else None
        self._highwater_x = p["ego_x_m"]
        self._progress_delta = 0.0
        self._stopped_steps = 0
        self.min_clearance = self.clearance()

    def clearance(self):
        """Conservative signed circumcircle clearance, in metres (not polygon distance)."""
        return float(
            min(
                np.linalg.norm(o.position - self.vehicle.position)
                - (o.diagonal + self.vehicle.diagonal) / 2
                for o in self.road.objects
            )
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended; reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError("Invalid meta-action")
        result = super().step(action)
        self.done = result[2] or result[3]
        return result

    def _simulate(self, action=None):
        # Track clearance at simulation rate, including between policy decisions.
        for _ in range(3):
            if self.steps % 3 == 0:
                self.action_type.act(action)
            self.road.act()
            self.road.step(1 / 15)
            self.steps += 1
            self.min_clearance = min(self.min_clearance, self.clearance())
        x = float(self.vehicle.position[0])
        capped_x = min(x, self.protocol["goal_x_m"])
        self._progress_delta = max(0.0, capped_x - self._highwater_x)
        self._highwater_x = max(self._highwater_x, capped_x)
        stopped = abs(self.vehicle.speed) <= self.protocol["safe_stop_speed_mps"]
        self._stopped_steps = self._stopped_steps + 1 if stopped else 0

    def has_arrived(self, vehicle):
        return bool(
            not vehicle.crashed
            and vehicle.on_road
            and vehicle.position[0] >= self.protocol["goal_x_m"]
        )

    def safe_blocked_stop(self):
        return bool(
            self.route_plan is None
            and not self.vehicle.crashed
            and self.vehicle.on_road
            and self.vehicle.position[0] < self.obstacle_x
            and self.clearance() >= self.protocol["clearance_margin_m"]
            and self._stopped_steps / 5 >= self.protocol["safe_stop_hold_s"]
        )

    def _reward(self, action):
        p = self.protocol
        if self.vehicle.crashed:
            return p["collision_reward"]
        if not self.vehicle.on_road:
            return p["offroad_reward"]
        if self.has_arrived(self.vehicle):
            return p["arrival_reward"]
        return (
            p["progress_reward"] * self._progress_delta / (p["goal_x_m"] - p["ego_x_m"])
            - p["time_cost_per_s"] / 5
        )

    def _is_terminated(self):
        return bool(
            self.vehicle.crashed
            or not self.vehicle.on_road
            or self.has_arrived(self.vehicle)
            or self.safe_blocked_stop()
        )

    def _is_truncated(self):
        return self.steps >= round(self.protocol["duration"] * 15)

    def _info(self, obs, action=None):
        info = super()._info(obs, action)
        info.update(
            {
                "is_success": self.has_arrived(self.vehicle),
                "collision_obstacle": bool(self.vehicle.crashed),
                "safe_blocked_stop": self.safe_blocked_stop(),
                "route_available": self.route_plan is not None,
                "route_target_lane": self.route_target_lane,
                "reroute_success": bool(
                    self.has_arrived(self.vehicle) and self.route_target_lane == 1
                ),
                "min_obstacle_clearance_m": self.min_clearance,
                "offroad": not self.vehicle.on_road,
            }
        )
        return info


class RouteObservation(gym.ObservationWrapper):
    """21 native kinematics + target speed, route lane, availability, goal distance, two blocks."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            np.concatenate([env.observation_space.low.ravel(), np.zeros(6)]).astype(np.float32),
            np.concatenate([env.observation_space.high.ravel(), np.ones(6)]).astype(np.float32),
        )

    def observation(self, observation):
        base = self.unwrapped
        p = base.protocol
        extra = [
            base.vehicle.target_speed / 9,
            base.route_target_lane or 0,
            float(base.route_plan is not None),
            np.clip((p["goal_x_m"] - base.vehicle.position[0]) / p["goal_x_m"], 0, 1),
            *base.blocked_lanes,
        ]
        return np.concatenate([observation.ravel(), extra]).astype(np.float32)


def make_obstacle_route_env(
    protocol_path=DEFAULT_PROTOCOL, *, alternate_available=True, render_mode=None
):
    return RouteObservation(
        ObstacleRouteEnv(
            load_config(protocol_path),
            alternate_available=alternate_available,
            render_mode=render_mode,
        )
    )


def route_baseline_action(env):
    """B0 for empty adjacent lanes only; not a general traffic-aware safety shield."""
    base = env.unwrapped
    if not isinstance(base, ObstacleRouteEnv):
        raise ValueError("B0 supports only the static obstacle_route_v1 task")
    if len(base.road.vehicles) != 1:
        raise ValueError("B0 is not validated for surrounding traffic")
    if base.route_plan is None:
        name = "SLOWER"
    elif base.vehicle.target_lane_index[2] != base.route_target_lane:
        name = (
            "LANE_RIGHT"
            if base.route_target_lane > base.vehicle.target_lane_index[2]
            else "LANE_LEFT"
        )
    else:
        name = "FASTER"
    return base.action_type.actions_indexes[name]
