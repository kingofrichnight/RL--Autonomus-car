"""Non-interventional replay of 46 frozen failure cases in both historical arms."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from safeintent_rl.config import load_config
from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.intent.inference import file_sha256
from scripts.analyze_predictive_stress import analyze, verify_rows
from scripts.evaluate_predictive_stress import episode, scenario_config
from scripts.recover_predictive_stress import rows_match

PROTOCOL_SHA256 = "6c3851362df84a0aef76ee76dfad33bc203573aa843dde6c0d3c21dcd1142d47"
SELECTED_SEEDS = {
    "dense": (80044, 80050, 80059, 80083, 80085, 80102, 80121, 80125, 80126),
    "aggressive": (
        80044, 80046, 80050, 80051, 80054, 80059, 80060, 80062, 80066, 80068,
        80073, 80075, 80077, 80081, 80084, 80086, 80090, 80094, 80095, 80103,
        80104, 80105, 80110, 80113, 80114, 80117, 80121, 80122, 80123, 80125,
        80126, 80129, 80130, 80134, 80135, 80137, 80138,
    ),
}
FEATURES = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]


def select_cases(rows: dict[str, list[dict]], protocol: dict) -> dict[str, list[int]]:
    """Select by historical outcome, and reject any deviation from frozen seed lists."""
    selected = {}
    for scenario, outcome in (("dense", "incomplete"), ("aggressive", "collision")):
        verify_rows(rows[scenario], protocol)
        seeds = [row["seed"] for row in rows[scenario] if row[outcome]]
        if seeds != list(SELECTED_SEEDS[scenario]):
            raise ValueError(f"Frozen {scenario} failure selection differs")
        selected[scenario] = seeds
    return selected


def write_line(stream, value: dict) -> None:
    stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


class TracePolicy:
    """Delegate the executed action unchanged; extra evaluations only observe state.

    Birth IDs are assigned on first diagnostic sighting in road-list order, scoped
    to one episode. They are not Python ids, cross-arm identities, or policy inputs.
    Live actors are post-spawn state; they are not claimed to match native slots.
    """

    def __init__(self, model, env, stream, *, arm: str, scenario: str, seed: int):
        self.model = model
        self.env = env
        self.stream = stream
        self.context = {"arm": arm, "scenario": scenario, "seed": seed}
        self.birth_ids: dict[object, int] = {}
        self.decisions = 0
        self.alignment_mismatches = 0
        self.metrics = {"zero_target_decisions": 0, "stopped_decisions": 0,
                        "stopped_faster_margin_positive": 0,
                        "stopped_faster_margin_positive_not_chosen": 0,
                        "all_actions_margin_conflict": 0,
                        "observed_y_boundary_slots": 0, "present_npc_slots": 0}

    def _birth(self, actor) -> int:
        if actor not in self.birth_ids:
            self.birth_ids[actor] = len(self.birth_ids)
        return self.birth_ids[actor]

    def snapshot(self, observation: np.ndarray) -> dict:
        base = self.env.unwrapped
        ego = base.vehicle
        obs_type = base.observation_type
        if (obs_type.features != FEATURES or obs_type.order != "sorted"
                or obs_type.vehicles_count != 15 or obs_type.absolute):
            raise ValueError("Diagnostic requires the frozen relative sorted observation")
        flat = np.asarray(observation).reshape(-1)
        expected_size = 115 if self.context["arm"] == "candidate" else 106
        if flat.size != expected_size or not np.isfinite(flat).all():
            raise ValueError("Invalid frozen observation shape or values")
        observed = flat[:105].reshape(15, 7)
        selected = base.road.close_objects_to(
            ego, float(base.PERCEPTION_DISTANCE), count=14,
            see_behind=bool(obs_type.see_behind), sort=True,
            vehicles_only=not bool(obs_type.include_obstacles),
        )
        self._birth(ego)
        live_actors = list(base.road.vehicles) + list(base.road.objects)
        # Assign before filtering, so one actor retains its ID when leaving/reentering range.
        for actor in live_actors:
            self._birth(actor)
        live = np.zeros((15, 7), dtype=np.float64)
        for index, actor in enumerate([ego, *selected]):
            position = np.asarray(actor.position, dtype=float).copy()
            velocity = np.asarray(actor.velocity, dtype=float).copy()
            if index:
                position -= ego.position
                velocity -= ego.velocity
            live[index] = [1, *position, *velocity,
                           math.cos(actor.heading), math.sin(actor.heading)]
        raw_live = live.copy()
        occupied = len(selected) + 1
        if obs_type.normalize:
            for index, name in enumerate(FEATURES):
                if name in obs_type.features_range:
                    low, high = obs_type.features_range[name]
                    live[:occupied, index] = (
                        -1 + 2 * (live[:occupied, index] - low) / (high - low))
                    if obs_type.clip:
                        live[:occupied, index] = np.clip(live[:occupied, index], -1, 1)
        slot_matches = np.isclose(observed, live.astype(np.float32), atol=1e-6, rtol=1e-6)
        matching_rows = slot_matches.all(axis=1)
        self.alignment_mismatches += int(not matching_rows.all())
        present = observed[1:, 0] != 0
        y_boundary = int(np.sum(present & (np.abs(observed[1:, 2]) >= 1)))
        self.metrics["observed_y_boundary_slots"] += y_boundary
        self.metrics["present_npc_slots"] += int(present.sum())
        current_actors = []
        for actor in live_actors:
            if actor is ego:
                continue
            if np.linalg.norm(actor.position - ego.position) >= base.PERCEPTION_DISTANCE:
                continue
            current_actors.append([self._birth(actor), *np.round(actor.position, 6).tolist(),
                                   *np.round(actor.velocity, 6).tolist(),
                                   round(float(actor.heading), 6)])
        return {
            "policy_step": self.decisions, "simulation_time": float(base.time),
            "ego": [*np.round(ego.position, 6).tolist(), float(ego.speed),
                    float(ego.target_speed), float(ego.heading)],
            "target_feature": float(flat[105]),
            "forecast_features": (flat[106:].reshape(3, 3).tolist()
                                  if flat.size == 115 else None),
            "native_npc_slots": int(present.sum()),
            "native_y_boundary_slots": y_boundary,
            "native_base_observation": observed.tolist(),
            "live_matching_native_rows": matching_rows.tolist(),
            "live_selected_birth_ids": [self._birth(actor) for actor in selected],
            "live_selected_raw_y": raw_live[1:occupied, 2].tolist(),
            "live_actors": current_actors,
        }

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("Diagnostic permits deterministic predictions only")
        # This is exactly the action/state pair that the historical evaluator receives.
        result = self.model.predict(observation, deterministic=True)
        action = int(np.asarray(result[0]).item())
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            probabilities = self.model.policy.get_distribution(tensor).distribution.probs
            probabilities = probabilities[0].detach().cpu().numpy()
        if (probabilities.shape != (3,) or not np.isfinite(probabilities).all()
                or not np.isclose(probabilities.sum(), 1) or action != int(probabilities.argmax())):
            raise ValueError("Categorical probabilities do not match deterministic action")
        details = self.snapshot(observation)
        speed, target = details["ego"][2:4]
        stopped = abs(speed) < 0.5
        self.metrics["zero_target_decisions"] += int(abs(target) < 1e-6)
        self.metrics["stopped_decisions"] += int(stopped)
        forecasts = details["forecast_features"]
        if forecasts is not None:
            all_conflict = all(row[0] <= 0 for row in forecasts)
            faster_open = forecasts[2][0] > 0
            self.metrics["all_actions_margin_conflict"] += int(all_conflict)
            self.metrics["stopped_faster_margin_positive"] += int(stopped and faster_open)
            self.metrics["stopped_faster_margin_positive_not_chosen"] += int(
                stopped and faster_open and action != 2)
        write_line(self.stream, {**self.context, **details, "action": action,
                                 "action_probabilities": probabilities.tolist()})
        self.decisions += 1
        return result


def replay_episode(env, model, reference: dict, stream, *, arm: str, scenario: str) -> dict:
    tracer = TracePolicy(model, env, stream, arm=arm, scenario=scenario, seed=reference["seed"])
    actual = episode(env, tracer, reference["seed"])
    return {"arm": arm, "scenario": scenario, "seed": reference["seed"],
            "matched": rows_match(reference, actual), "reference": reference, "actual": actual,
            "decisions": tracer.decisions, "alignment_mismatches": tracer.alignment_mismatches,
            "diagnostic_counts": tracer.metrics}


def run(root: Path, candidate_dir: Path, output: Path, protocol_sha256: str) -> dict:
    if output.exists():
        raise FileExistsError("Preserve existing diagnostic output")
    if protocol_sha256 != PROTOCOL_SHA256:
        raise ValueError("Frozen protocol hash differs")
    verified = analyze(root, candidate_dir)
    if verified["protocol_sha256"] != PROTOCOL_SHA256 or verified["episodes"] != 1000:
        raise ValueError("Require the complete frozen 1,000-episode stress reference")
    directories = {"candidate": candidate_dir, "control": root / "control"}
    control = json.loads((directories["control"] / "report.json").read_text(encoding="utf-8"))
    protocol = control["protocol"]
    current_versions = {name: version(name) for name in control["versions"]}
    if current_versions != control["versions"]:
        raise ValueError("Replay package versions differ from the historical experiment")
    records = {}
    for arm, path in directories.items():
        records[arm] = {}
        for name in SELECTED_SEEDS:
            lines = (path / f"{name}.episodes.jsonl").read_text(encoding="utf-8").splitlines()
            records[arm][name] = [json.loads(line) for line in lines]
    selected = select_cases(records["candidate"], protocol)
    source_paths = [*control["source_sha256"], "scripts/diagnose_predictive_failures.py",
                    "scripts/analyze_predictive_stress.py", "scripts/recover_predictive_stress.py"]
    sources = {path: file_sha256(path) for path in source_paths}
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "name": "v3_predictive_failure_diagnostic_v1", "status": "running",
        "protocol_sha256": PROTOCOL_SHA256, "selected_seeds": selected,
        "expected_replays": 92, "completed_replays": 0, "matched_replays": 0,
        "arm_report_sha256": verified["arm_report_sha256"], "source_sha256": sources,
        "versions": {**current_versions, "torch": version("torch")},
        "arms": protocol["arms"], "scenarios": {}, "artifacts_sha256": {},
        "trace_semantics": {
            "actors": "episode-first-sighting ID,x,y,vx,vy,heading; poses rounded to 1e-6",
            "ego": "world x,y,speed,target_speed,heading; ego ID is 0",
            "actor_scope": "live post-spawn actors inside native perception radius, including rear",
            "native_slots": "original policy input; no actor IDs asserted for native slots",
            "alignment": "native rows compared with live selected actors; tolerance 1e-6",
            "forecasts": "existing observation only, no extra ghost rollout; null for control",
            "probs": "extra deterministic no-grad forward pass; original predict action unchanged",
            "counts": "pre-action decision samples, not historical post-step stopped-time metric",
            "interpretation": "selected failures; positive predicted margin is not proven safety",
        },
    }
    started = time.perf_counter()
    try:
        with (output / "episode_checks.jsonl").open("x", encoding="utf-8") as checks:
            for arm in ("candidate", "control"):
                arm_config = protocol["arms"][arm]
                model = PPO.load(arm_config["model"], device="cpu")
                for scenario in protocol["scenarios"]:
                    name = scenario["name"]
                    if name not in selected:
                        continue
                    effective = scenario_config(load_config(arm_config["config"]), scenario)
                    key = f"{arm}.{name}"
                    report["scenarios"][key] = {"effective_config": effective,
                                                "scenario": scenario, "status": "running"}
                    with tempfile.TemporaryDirectory(prefix="v3_failure_diagnostic_") as directory:
                        config_path = Path(directory) / "effective.yaml"
                        config_path.write_text(yaml.safe_dump(effective), encoding="utf-8")
                        env = make_intersection_env(
                            config_path,
                            target_speed_observation=arm_config["target_speed_observation"],
                            target_speed_scale=9.0,
                            driver_probabilities=tuple(scenario["driver_probabilities"]),
                        )
                        try:
                            if list(env.observation_space.shape) != arm_config["observation_shape"]:
                                raise ValueError("Frozen policy observation shape changed")
                            with gzip.open(output / f"{key}.trace.jsonl.gz", "xt",
                                           encoding="utf-8") as traces:
                                references = {row["seed"]: row for row in records[arm][name]}
                                for seed in selected[name]:
                                    result = replay_episode(env, model, references[seed], traces,
                                                            arm=arm, scenario=name)
                                    write_line(checks, result)
                                    report["completed_replays"] += 1
                                    report["matched_replays"] += int(result["matched"])
                                    if not result["matched"]:
                                        raise ValueError(
                                            f"Historical episode did not reproduce: {key} {seed}")
                                    print(f"{key} seed={seed} matched; "
                                          f"{report['completed_replays']}/92", flush=True)
                            ranges = env.unwrapped.observation_type.features_range
                            report["scenarios"][key].update({
                                "status": "complete", "replays": len(selected[name]),
                                "effective_observation_ranges": ranges,
                            })
                        finally:
                            env.close()
        if report["completed_replays"] != 92 or report["matched_replays"] != 92:
            raise ValueError("Incomplete diagnostic replay ledger")
        if any(file_sha256(path) != digest for path, digest in sources.items()):
            raise ValueError("Source changed while diagnostic was running")
        report["status"] = "complete"
    except BaseException as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["artifacts_sha256"] = {path.name: file_sha256(path) for path in output.iterdir()
                                      if path.is_file() and path.name != "report.json"}
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(f"Diagnostic {report['status']}: {report['matched_replays']}/92 matched", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/v3_predictive_stress_v1"))
    parser.add_argument("--candidate-dir", type=Path,
                        default=Path("results/v3_predictive_stress_v1/candidate_recovery_01"))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.candidate_dir, args.output_dir, args.protocol_sha256)


if __name__ == "__main__":
    main()
