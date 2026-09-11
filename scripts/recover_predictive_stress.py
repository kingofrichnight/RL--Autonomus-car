"""Preserve a stopped candidate prefix; verify replay, then finish missing seeds."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import yaml
from stable_baselines3 import PPO

from safeintent_rl.config import load_config
from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.intent.inference import file_sha256
from scripts.analyze_predictive_stress import verify_rows
from scripts.evaluate_predictive_stress import (
    episode,
    scenario_config,
    summarize,
    validate_protocol,
)


def rows_match(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-9)
               if isinstance(a[k], float) and isinstance(b[k], (int, float))
               else a[k] == b[k] for k in a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    control = json.loads(Path(args.control_report).read_text(encoding="utf-8"))
    if control["status"] != "complete" or control["arm"] != "control":
        raise ValueError("Require completed control as frozen-source provenance")
    protocol = control["protocol"]
    validate_protocol(protocol)
    if file_sha256("configs/v3_predictive_stress_v1.json") != args.protocol_sha256:
        raise ValueError("Protocol fingerprint mismatch")
    if control["protocol_sha256"] != args.protocol_sha256:
        raise ValueError("Control protocol differs")
    for path, digest in control["source_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError("Do not resume with changed experiment sources")
    arm = protocol["arms"]["candidate"]
    for kind in ("config", "model"):
        if file_sha256(arm[kind]) != arm[kind + "_sha256"]:
            raise ValueError("Frozen candidate artifact changed")
    prefixes = {}
    for scenario in protocol["scenarios"]:
        path = source / (scenario["name"] + ".episodes.jsonl")
        rows = ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                if path.exists() else [])
        if len(rows) > protocol["episodes_per_scenario"]:
            raise ValueError("Too many source rows")
        if rows:
            verify_rows(rows, {**protocol, "episodes_per_scenario": len(rows)})
        prefixes[scenario["name"]] = rows
    output.mkdir(parents=True, exist_ok=False)
    report = copy.deepcopy(control)
    report.update({"arm": "candidate", "status": "running", "scenarios": {}, "recovery": {
        "source_dir": str(source), "source_prefix_sha256": {
            s["name"]: file_sha256(source / (s["name"] + ".episodes.jsonl"))
            for s in protocol["scenarios"] if prefixes[s["name"]]},
        "copied_episode_counts": {name: len(rows) for name, rows in prefixes.items()},
        "script_sha256": file_sha256(__file__), "replay_checks": [],
        "reason": "Original process absent after 194 saved rows; stderr empty; cause unknown",
        "elapsed_time_scope": "recovery process only; original incomplete runtime unavailable"}})
    started = time.perf_counter()
    try:
        model = PPO.load(arm["model"], device="cpu")
        base_config = load_config(arm["config"])
        for scenario in protocol["scenarios"]:
            name = scenario["name"]
            rows = prefixes[name]
            effective = scenario_config(base_config, scenario)
            destination = output / f"{name}.episodes.jsonl"
            if rows:
                shutil.copyfile(source / destination.name, destination)
                if file_sha256(destination) != report["recovery"]["source_prefix_sha256"][name]:
                    raise ValueError("Prefix copy mismatch")
            else:
                destination.touch(exist_ok=False)
            if len(rows) < protocol["episodes_per_scenario"]:
                with tempfile.TemporaryDirectory(prefix="stress_recovery_config_") as directory:
                    config_path = Path(directory) / "effective.yaml"
                    config_path.write_text(yaml.safe_dump(effective), encoding="utf-8")
                    env = make_intersection_env(
                        config_path, target_speed_observation=False,
                        driver_probabilities=tuple(scenario["driver_probabilities"]),
                    )
                    try:
                        if rows:
                            replay = episode(env, model, rows[-1]["seed"])
                            matched = rows_match(rows[-1], replay)
                            report["recovery"]["replay_checks"].append({
                                "scenario": name, "seed": rows[-1]["seed"], "matched": matched})
                            if not matched:
                                raise ValueError("Last saved episode did not reproduce")
                            print(f"recovery {name}: last saved seed reproduced", flush=True)
                        with destination.open("a", encoding="utf-8") as stream:
                            for i in range(len(rows), protocol["episodes_per_scenario"]):
                                row = episode(env, model, protocol["first_seed"] + i)
                                stream.write(json.dumps(row, allow_nan=False) + "\n")
                                stream.flush()
                                rows.append(row)
                                if (i + 1) % 10 == 0:
                                    print(f"candidate recovery {name}: {i + 1}/100", flush=True)
                    finally:
                        env.close()
            report["scenarios"][name] = {
                "effective_config": effective, "scenario": scenario, "status": "complete",
                "summary": summarize(rows), "episodes_sha256": file_sha256(destination)}
            with (output / f"{name}.summary.json").open("x", encoding="utf-8") as stream:
                json.dump(report["scenarios"][name], stream, indent=2, allow_nan=False)
            print(json.dumps({"scenario": name, **summarize(rows)}), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(f"Recovery status: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
