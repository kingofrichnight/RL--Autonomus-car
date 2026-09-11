"""Frozen, paired stress scenarios; no training and no historical output writes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO

from safeintent_rl.config import load_config
from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.evaluation import detect_collision, detect_success
from safeintent_rl.intent.inference import file_sha256
from safeintent_rl.safety.ttc import minimum_ttc


def validate_protocol(protocol: dict) -> None:
    if protocol["name"] != "v3_predictive_stress_v1":
        raise ValueError("Unsupported protocol")
    if protocol["episodes_per_scenario"] != 100 or protocol["first_seed"] != 80042:
        raise ValueError("Frozen count/seeds differ")
    if protocol["duration"] != 30 or protocol["stopped_speed_threshold"] != 0.5:
        raise ValueError("Frozen duration/stopped threshold differ")
    if protocol["unsafe_ttc_threshold"] != 2.0:
        raise ValueError("Frozen TTC threshold differs")
    if [s["name"] for s in protocol["scenarios"]] != [
        "light", "dense", "aggressive", "cautious", "mixed"
    ]:
        raise ValueError("Missing or reordered scenarios")
    for scenario in protocol["scenarios"]:
        p = np.asarray(scenario["driver_probabilities"], dtype=float)
        if (p.shape != (3,) or not np.isfinite(p).all() or np.any(p < 0)
                or not np.isclose(p.sum(), 1)):
            raise ValueError("Invalid profile probabilities")
        if not 0 <= scenario["spawn_probability"] <= 1:
            raise ValueError("Invalid spawn probability")


def scenario_config(base: dict, scenario: dict) -> dict:
    config = copy.deepcopy(base)
    if config["duration"] != 30:
        raise ValueError("Do not change the 30-second deadline")
    config["spawn_probability"] = scenario["spawn_probability"]
    return config


def episode(env, model, seed: int, stopped_threshold: float = 0.5) -> dict:
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    steps = stopped_steps = zero_target_steps = unsafe_steps = 0
    total_reward = 0.0
    min_ttc = math.inf
    max_npcs = 0
    terminated = truncated = False
    while not (terminated or truncated):
        pre_ttc = minimum_ttc(base.vehicle, list(base.road.vehicles))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        ttc = float(info.get("min_ttc", pre_ttc))
        min_ttc = min(min_ttc, ttc)
        unsafe_steps += int(ttc <= 2.0)
        stopped_steps += int(abs(base.vehicle.speed) < stopped_threshold)
        zero_target_steps += int(abs(base.vehicle.target_speed) < 1e-6)
        max_npcs = max(max_npcs, len(base.road.vehicles) - 1)
        if steps > math.ceil(base.config["duration"] * base.config["policy_frequency"]) + 1:
            raise RuntimeError("Environment exceeded frozen deadline")
    collision = detect_collision(env, info)
    success = detect_success(env, info)
    frequency = float(base.config["policy_frequency"])
    return {"seed": seed, "success": success, "collision": collision,
            "incomplete": not (success or collision), "reward": total_reward,
            "steps": steps, "travel_time": steps / frequency,
            "stopped_time": stopped_steps / frequency,
            "zero_target_time": zero_target_steps / frequency,
            "min_ttc": min_ttc if math.isfinite(min_ttc) else None,
            "unsafe_ttc_events": unsafe_steps, "max_npcs": max_npcs,
            "final_route_progress": info.get("route_progress"),
            "driver_profile_counts": info.get("driver_profile_counts", {})}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize empty results")
    summary = {"episodes": len(rows)}
    for key in ("success", "collision", "incomplete"):
        summary[key + "_count"] = sum(int(row[key]) for row in rows)
        summary[key + "_rate"] = summary[key + "_count"] / len(rows)
    for key in ("travel_time", "stopped_time", "zero_target_time", "unsafe_ttc_events",
                "max_npcs", "reward"):
        summary["mean_" + key] = float(np.mean([row[key] for row in rows]))
    finite = [r["min_ttc"] for r in rows if r["min_ttc"] is not None]
    summary["mean_min_ttc"] = float(np.mean(finite)) if finite else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--protocol", default="configs/v3_predictive_stress_v1.json")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if file_sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Protocol fingerprint mismatch")
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    arm = protocol["arms"][args.arm]
    for field in ("config", "model"):
        if file_sha256(arm[field]) != arm[field + "_sha256"]:
            raise ValueError(f"Frozen {field} fingerprint mismatch")
    config = load_config(arm["config"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {"arm": args.arm, "protocol": protocol, "protocol_sha256": args.protocol_sha256,
              "status": "running", "scenarios": {}, "versions": {
                  name: version(name) for name in ("numpy", "highway-env", "stable-baselines3")},
              "source_sha256": {p: file_sha256(p) for p in (
                  "scripts/evaluate_predictive_stress.py", "safeintent_rl/envs/intersection.py",
                  "safeintent_rl/envs/driver_behavior.py", "safeintent_rl/sensors/predictive.py",
                  "safeintent_rl/envs/reward.py")}}
    start = time.perf_counter()
    try:
        model = PPO.load(arm["model"], device="cpu")
        for scenario in protocol["scenarios"]:
            name = scenario["name"]
            effective = scenario_config(config, scenario)
            rows = []
            with tempfile.TemporaryDirectory(prefix="v3_stress_config_") as directory:
                path = Path(directory) / "effective.yaml"
                path.write_text(yaml.safe_dump(effective), encoding="utf-8")
                report["scenarios"][name] = {"effective_config": effective,
                                            "scenario": scenario, "status": "running"}
                env = make_intersection_env(
                    path, target_speed_observation=arm["target_speed_observation"],
                    target_speed_scale=9.0,
                    driver_probabilities=tuple(scenario["driver_probabilities"]),
                )
                try:
                    if list(env.observation_space.shape) != arm["observation_shape"]:
                        raise ValueError("Observation shape mismatch")
                    with (output / f"{name}.episodes.jsonl").open("x", encoding="utf-8") as stream:
                        for i in range(protocol["episodes_per_scenario"]):
                            row = episode(env, model, protocol["first_seed"] + i,
                                          protocol["stopped_speed_threshold"])
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                            rows.append(row)
                            if (i + 1) % 10 == 0:
                                print(f"{args.arm} {name}: {i + 1}/100", flush=True)
                finally:
                    env.close()
            report["scenarios"][name].update({"status": "complete", "summary": summarize(rows),
                "episodes_sha256": file_sha256(output / f"{name}.episodes.jsonl")})
            with (output / f"{name}.summary.json").open("x", encoding="utf-8") as stream:
                json.dump(report["scenarios"][name], stream, indent=2, allow_nan=False)
            print(json.dumps({"arm": args.arm, "scenario": name,
                              **summarize(rows)}), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(f"{args.arm}: {report['status']}; report={output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
