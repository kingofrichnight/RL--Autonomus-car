import copy
import json
import sys

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from safeintent_rl.agents.route_graph import RouteEdge, shortest_route
from safeintent_rl.envs.obstacle_route import (
    DEFAULT_PROTOCOL,
    ObstacleRouteEnv,
    make_obstacle_route_env,
)
from scripts.evaluate_obstacle_route import main, run_case


def test_route_avoids_blocked_edge_and_handles_no_path():
    edges = [
        RouteEdge("a", "b", 1, 1, blocked=True),
        RouteEdge("a", "c", 3, 1),
        RouteEdge("c", "b", 3, 1),
    ]
    assert shortest_route(edges, "a", "b") == ("a", "c", "b")
    assert shortest_route(edges[:1], "a", "b") is None
    assert shortest_route([], "a", "a") == ("a",)


def test_route_ties_and_cycles_are_deterministic():
    edges = [
        RouteEdge("a", "c", 1, 1),
        RouteEdge("a", "b", 1, 1),
        RouteEdge("b", "a", 0, 1),
        RouteEdge("b", "z", 1, 1),
        RouteEdge("c", "z", 1, 1),
    ]
    assert shortest_route(edges, "a", "z") == ("a", "b", "z")
    assert shortest_route(edges[::-1], "a", "z") == ("a", "b", "z")


@pytest.mark.parametrize(
    "kwargs", [{"length_m": -1}, {"speed_mps": 0}, {"risk_cost_s": float("nan")}, {"blocked": 1}]
)
def test_route_rejects_invalid_costs(kwargs):
    values = dict(source="a", target="b", length_m=1, speed_mps=1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        RouteEdge(**values)


@pytest.mark.parametrize("seed", [7, 8, 9])
@pytest.mark.parametrize("alternate", [True, False])
def test_b0_scenario_gate(seed, alternate):
    row = run_case(DEFAULT_PROTOCOL, seed, alternate)
    assert not row["collision"] and not row["offroad"] and not row["incomplete"]
    assert row["success"] is alternate
    assert row["safe_blocked_stop"] is not alternate
    assert row["reroute_success"] is alternate


def test_observation_and_rng_noninterference():
    raw = ObstacleRouteEnv()
    wrapped = make_obstacle_route_env()
    try:
        a, _ = raw.reset(seed=7)
        b, _ = wrapped.reset(seed=7)
        assert b.shape == (27,)
        assert np.array_equal(a.ravel(), b[:21])
        state = copy.deepcopy(wrapped.unwrapped.np_random.bit_generator.state)
        wrapped.observation(a)
        assert state == wrapped.unwrapped.np_random.bit_generator.state
        for action in [2, 3, 1, 1, 4]:
            a, ar, at, ax, _ = raw.step(action)
            b, br, bt, bx, _ = wrapped.step(action)
            assert np.array_equal(a.ravel(), b[:21])
            assert (ar, at, ax) == (br, bt, bx)
            assert wrapped.observation_space.contains(b)
    finally:
        raw.close()
        wrapped.close()


def test_collision_first_even_at_goal():
    env = ObstacleRouteEnv()
    try:
        env.vehicle.position[0] = env.protocol["goal_x_m"]
        env.vehicle.crashed = True
        assert not env.has_arrived(env.vehicle)
        assert env._reward(1) == -10
        assert not env.safe_blocked_stop()
    finally:
        env.close()


def test_straight_driving_really_collides_with_blockage():
    env = make_obstacle_route_env()
    try:
        env.reset(seed=7)
        for _ in range(150):
            _, _, terminated, truncated, info = env.step(3)
            if terminated or truncated:
                break
        assert info["collision_obstacle"] and not info["is_success"]
        with pytest.raises(RuntimeError):
            env.step(1)
    finally:
        env.close()


def test_gym_interface():
    env = ObstacleRouteEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_blockage_observation_and_scenario_pairing():
    scenes = [make_obstacle_route_env(alternate_available=a) for a in (True, False)]
    try:
        observations = [env.reset(seed=7)[0] for env in scenes]
        assert scenes[0].unwrapped.obstacle_x == scenes[1].unwrapped.obstacle_x
        assert observations[0][:21].reshape(3, 7)[:, 0].sum() == 2
        assert observations[1][:21].reshape(3, 7)[:, 0].sum() == 3
        assert list(observations[0][-2:]) == [1, 0]
        assert list(observations[1][-2:]) == [1, 1]
        with pytest.raises(ValueError):
            scenes[0].reset(options={"config": {"policy_frequency": 1}})
    finally:
        for env in scenes:
            env.close()


def test_failed_report_and_overwrite_refusal(tmp_path, monkeypatch):
    output = tmp_path / "failed.json"
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "--protocol-sha256", "wrong", "--output", str(output)]
    )
    with pytest.raises(ValueError, match="fingerprint"):
        main()
    original = output.read_bytes()
    assert json.loads(original)["status"] == "failed"
    assert json.loads(original)["episodes"] == []
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == original
