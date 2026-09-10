"""Run the isolated B0 feasibility gate; never compare these rates to intersection PPO."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from safeintent_rl.envs.obstacle_route import (
    DEFAULT_PROTOCOL,
    ObstacleRouteEnv,
    RouteObservation,
    make_obstacle_route_env,
    route_baseline_action,
)
from safeintent_rl.evaluation.metrics import detect_collision, detect_success


def run_case(protocol, seed, alternate_available):
    env = (
        RouteObservation(ObstacleRouteEnv(protocol, alternate_available=alternate_available))
        if isinstance(protocol, dict)
        else make_obstacle_route_env(protocol, alternate_available=alternate_available)
    )
    try:
        env.reset(seed=seed)
        done = False
        steps = 0
        reward_sum = 0.0
        while not done:
            _, reward, terminated, truncated, info = env.step(route_baseline_action(env))
            reward_sum += reward
            steps += 1
            done = terminated or truncated
        success = detect_success(env, info)
        collision = detect_collision(env, info)
        return {
            "seed": seed,
            "alternate_available": alternate_available,
            "obstacle_x_m": env.unwrapped.obstacle_x,
            "success": success,
            "collision": collision,
            "safe_blocked_stop": info["safe_blocked_stop"],
            "incomplete": not (
                success or collision or info["safe_blocked_stop"] or info["offroad"]
            ),
            "offroad": info["offroad"],
            "length": steps,
            "travel_time_s": steps / 5,
            "reward": reward_sum,
            "reroute_success": info["reroute_success"],
            "min_obstacle_clearance_m": info["min_obstacle_clearance_m"],
            "collision_obstacle": info["collision_obstacle"],
            "safety_interventions": 0,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--episodes", type=int, default=10, help="Cases per scenario")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output before simulation. Failed reports are preserved, never overwritten.
    with args.output.open("x", encoding="utf-8") as handle:
        report = {
            "schema_version": 1,
            "status": "running",
            "role": "engineering_feasibility",
            "controller": "B0_static_route_v1",
            "episodes": [],
            "error": None,
        }
        try:
            raw = args.protocol.read_bytes()
            actual_hash = hashlib.sha256(raw).hexdigest()
            report["protocol_sha256"] = actual_hash
            if actual_hash != args.protocol_sha256.lower():
                raise ValueError("Protocol fingerprint mismatch")
            report["protocol"] = json.loads(raw)
            report["source_sha256"] = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    Path(__file__).resolve().parents[1] / "safeintent_rl/envs/obstacle_route.py",
                    Path(__file__).resolve().parents[1] / "safeintent_rl/agents/route_graph.py",
                )
            }
            report["versions"] = {
                name: version(name) for name in ("highway-env", "gymnasium", "numpy")
            }
            report["first_seed"] = args.seed
            report["last_seed"] = args.seed + args.episodes - 1
            for alternate in (True, False):
                for seed in range(args.seed, args.seed + args.episodes):
                    report["episodes"].append(run_case(report["protocol"], seed, alternate))
            report["gate_passed"] = all(
                not row["collision"]
                and not row["offroad"]
                and (
                    row["reroute_success"]
                    if row["alternate_available"]
                    else row["safe_blocked_stop"]
                )
                for row in report["episodes"]
            )
            report["status"] = "complete"
        except BaseException as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(f"Saved {len(report['episodes'])} cases; feasibility gate={report['gate_passed']}")
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
