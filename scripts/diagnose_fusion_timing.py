"""Replay frozen fusion cases without intervening in either policy's actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.evaluation import (
    EpisodeMetrics,
    detect_collision,
    detect_success,
    summarize_episodes,
)
from safeintent_rl.intent.inference import file_sha256
from safeintent_rl.safety.conflict import nearby_closest_approaches
from safeintent_rl.safety.ttc import minimum_ttc

METRIC_FIELDS = list(EpisodeMetrics.__dataclass_fields__)
TRACE_COLUMNS = [
    "step",
    "time_s",
    "action",
    "observation_sha256",
    "speed",
    "target_speed_before",
    "position",
    "lane_index",
    "target_lane_index",
    "lane_longitudinal_m",
    "lane_remaining_m",
    "radial_ttc_s",
    "closest_cpa_time_s",
    "closest_cpa_distance_m",
    "cpa_risk",
    "cpa_risk_pairs",
    "target_speed_after",
    "speed_after",
    "route_progress_after",
    "terminated",
    "truncated",
]


def outcome(row: pd.Series) -> str:
    if bool(row["collision"]):
        return "collision"
    return "success" if bool(row["success"]) else "incomplete"


def select_cases(
    v1: pd.DataFrame, v2: pd.DataFrame, first_seed: int, controls_per_group: int
) -> list[dict[str, Any]]:
    if len(v1) != len(v2) or not len(v1):
        raise ValueError("References must contain the same positive number of rows")
    if first_seed < 0 or controls_per_group < 0:
        raise ValueError("Seed and control count must be non-negative")
    controls: Counter[str] = Counter()
    selected = []
    for index in range(len(v1)):
        left, right = outcome(v1.iloc[index]), outcome(v2.iloc[index])
        transition = f"{left}_to_{right}"
        if left != right:
            stratum = "discordant"
        elif left in {"success", "collision"} and controls[left] < controls_per_group:
            controls[left] += 1
            stratum = "control"
        else:
            continue
        selected.append(
            dict(row_index=index, seed=first_seed + index, transition=transition, stratum=stratum)
        )
    return selected


def verify_episode(metrics: EpisodeMetrics, expected: pd.Series) -> list[str]:
    mismatches = []
    for field, observed in metrics.as_dict().items():
        reference = expected[field]
        if field in {"success", "collision", "length", "unsafe_ttc_events", "safety_interventions"}:
            matches = observed == reference
        else:
            matches = bool(np.isclose(observed, reference, rtol=0.0, atol=1e-12))
        if not matches:
            mismatches.append(field)
    return mismatches


def summarize_trace(
    trace: list[dict[str, Any]],
    frequency: float,
    stopped_speed: float,
    terminal_window_s: float,
) -> dict[str, Any]:
    def counts(records: list[dict[str, Any]]) -> dict[str, Any]:
        longest = current = 0
        for row in records:
            current = current + 1 if abs(row["speed"]) <= stopped_speed else 0
            longest = max(longest, current)
        return {
            "decision_count": len(records),
            "action_counts": dict(Counter(row["action"] for row in records)),
            "risk_action_counts": dict(
                Counter(row["action"] for row in records if row["cpa_risk"])
            ),
            "stopped_steps": sum(abs(row["speed"]) <= stopped_speed for row in records),
            "longest_stopped_run_steps": longest,
            "low_speed_without_cpa_flag_steps": sum(
                abs(row["speed"]) <= stopped_speed and not row["cpa_risk"] for row in records
            ),
            "target_increase_steps": sum(
                row["target_speed_after"] > row["target_speed_before"] for row in records
            ),
            "target_decrease_steps": sum(
                row["target_speed_after"] < row["target_speed_before"] for row in records
            ),
        }

    result = counts(trace)
    result["terminal_window"] = counts(trace[-max(1, math.ceil(frequency * terminal_window_s)) :])
    return result


def compare_traces(v1: list[dict[str, Any]], v2: list[dict[str, Any]]) -> dict[str, Any]:
    overlap = min(len(v1), len(v2))
    action_difference = next(
        (i for i in range(overlap) if v1[i]["action"] != v2[i]["action"]), None
    )
    observation_difference = next(
        (i for i in range(overlap) if v1[i]["observation_sha256"] != v2[i]["observation_sha256"]),
        None,
    )
    shared = bool(overlap and v1[0]["observation_sha256"] == v2[0]["observation_sha256"])
    return {
        "overlap_steps": overlap,
        "first_action_divergence_step": action_difference,
        "first_observation_divergence_step": observation_difference,
        "shared_initial_observation": shared,
        "common_observation_prefix_until_action_divergence": shared
        and (
            observation_difference is None
            or (action_difference is not None and observation_difference > action_difference)
        ),
    }


def save_report(output: Path, report: dict[str, Any]) -> None:
    # Serialize before opening, so a nonfinite value cannot leave a partial file.
    payload = json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _json_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _require_hash(path: str, expected: str) -> None:
    if file_sha256(path).lower() != expected.lower():
        raise ValueError(f"Fingerprint mismatch: {path}")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol["schema_version"] != 1 or set(protocol["policies"]) != {"v1", "v2"}:
        raise ValueError("Unsupported protocol schema or policies")
    for name, minimum in (("episodes", 1), ("first_seed", 0), ("controls_per_group", 0)):
        value = protocol[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid protocol integer: {name}")
    for name, value in {
        "policy_frequency": protocol["policy_frequency"],
        "unsafe_ttc_threshold": protocol["unsafe_ttc_threshold"],
        **protocol["diagnostic"],
    }.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Diagnostic parameter must be finite and positive: {name}")
    if protocol["diagnostic"]["cpa_time_threshold"] > protocol["diagnostic"]["cpa_horizon"]:
        raise ValueError("CPA threshold exceeds diagnostic horizon")


def _read_reference(spec: dict[str, Any], protocol: dict[str, Any]) -> pd.DataFrame:
    for key in ("model", "reference_csv", "reference_summary"):
        _require_hash(spec[key], spec[f"{key}_sha256"])
    summary = json.loads(Path(spec["reference_summary"]).read_text(encoding="utf-8"))
    expected = {
        "episodes": protocol["episodes"],
        "first_seed": protocol["first_seed"],
        "last_seed": protocol["first_seed"] + protocol["episodes"] - 1,
        "model_sha256": spec["model_sha256"],
        "config_sha256": protocol["config_sha256"],
        "safety_shield": False,
        "safety_shield_type": None,
        "intent_model_path": None,
        "intent_model_sha256": None,
        "intent_neighbors": 0,
        "risk_fusion": True,
        "fusion_features_per_neighbor": 5,
        "unsafe_ttc_threshold": protocol["unsafe_ttc_threshold"],
        **protocol["fusion"],
    }
    for key, value in expected.items():
        if key not in summary or summary[key] != value:
            raise ValueError(f"Reference summary disagrees with protocol: {key}")
    frame = pd.read_csv(spec["reference_csv"])
    if list(frame.columns) != METRIC_FIELDS or len(frame) != protocol["episodes"]:
        raise ValueError("Reference CSV structure or episode count differs from protocol")
    for name in ("success", "collision"):
        if frame[name].dtype != bool:
            raise ValueError(f"Reference {name} must contain booleans")
    if (frame.success & frame.collision).any():
        raise ValueError("Reference outcomes must already use collision-first classification")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Frozen references must contain finite numeric values")
    if (frame.length <= 0).any() or (frame.safety_interventions != 0).any():
        raise ValueError("Reference has invalid length or safety interventions")
    for name in ("length", "unsafe_ttc_events", "safety_interventions"):
        if (frame[name] < 0).any() or (frame[name] % 1 != 0).any():
            raise ValueError(f"Reference must contain non-negative integer counts: {name}")
    aggregate = summarize_episodes([EpisodeMetrics(**row) for row in frame.to_dict("records")])
    for key, value in aggregate.items():
        if not math.isclose(value, summary[key], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Reference aggregate mismatch: {key}")
    return frame


def _capture_decision(
    env: Any,
    observation: Any,
    action: Any,
    step: int,
    radial_ttc: float,
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    base = env.unwrapped
    ego = base.vehicle
    if not np.isfinite(np.asarray(observation)).all():
        raise ValueError("Nonfinite policy observation")
    approaches = nearby_closest_approaches(
        ego,
        list(base.road.vehicles),
        horizon=diagnostic["cpa_horizon"],
        max_range=diagnostic["cpa_max_range"],
    )
    closest = min(approaches, key=lambda item: (item.distance, item.time), default=None)
    risk_pairs = sum(
        pair.time <= diagnostic["cpa_time_threshold"]
        and pair.distance <= diagnostic["cpa_distance_threshold"]
        for pair in approaches
    )
    longitudinal = float(ego.lane.local_coordinates(ego.position)[0])
    if not np.isfinite(
        [
            ego.speed,
            ego.target_speed,
            *ego.position,
            longitudinal,
            ego.lane.length,
        ]
    ).all():
        raise ValueError("Nonfinite physical state before action")
    return {
        "step": step,
        "time_s": step / float(base.config["policy_frequency"]),
        "action": str(base.action_type.actions[int(action)]),
        "observation_sha256": hashlib.sha256(
            np.ascontiguousarray(observation).tobytes()
        ).hexdigest(),
        "speed": float(ego.speed),
        "target_speed_before": float(ego.target_speed),
        "position": np.asarray(ego.position).tolist(),
        "lane_index": list(ego.lane_index),
        "target_lane_index": list(ego.target_lane_index),
        "lane_longitudinal_m": longitudinal,
        "lane_remaining_m": float(ego.lane.length) - longitudinal,
        "radial_ttc_s": radial_ttc,
        "closest_cpa_time_s": closest.time if closest else None,
        "closest_cpa_distance_m": closest.distance if closest else None,
        "cpa_risk": bool(risk_pairs),
        "cpa_risk_pairs": risk_pairs,
    }


def replay_episode(
    model: Any,
    env: Any,
    seed: int,
    unsafe_ttc_threshold: float,
    diagnostic: dict[str, Any],
) -> tuple[EpisodeMetrics, list[dict[str, Any]]]:
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    reward_sum = 0.0
    min_ttc = math.inf
    unsafe_events = interventions = 0
    final_info: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    while not (terminated or truncated):
        base = env.unwrapped
        radial_ttc = minimum_ttc(base.vehicle, list(base.road.vehicles))
        action, _ = model.predict(observation, deterministic=True)
        record = _capture_decision(env, observation, action, len(trace), radial_ttc, diagnostic)
        observation, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(
            [
                base.vehicle.target_speed,
                base.vehicle.speed,
                info["route_progress"],
            ]
        ).all():
            raise ValueError("Nonfinite physical state after action")
        record.update(
            {
                "target_speed_after": float(base.vehicle.target_speed),
                "speed_after": float(base.vehicle.speed),
                "route_progress_after": float(info["route_progress"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        trace.append(record)
        reward_sum += float(reward)
        step_ttc = float(info.get("min_ttc", radial_ttc))
        min_ttc = min(min_ttc, step_ttc)
        unsafe_events += int(step_ttc <= unsafe_ttc_threshold)
        interventions += int(info.get("safety_intervened", False))
        final_info = info
    return EpisodeMetrics(
        reward=reward_sum,
        length=len(trace),
        success=detect_success(env, final_info),
        collision=detect_collision(env, final_info),
        travel_time=len(trace) / float(env.unwrapped.config["policy_frequency"]),
        min_ttc=min_ttc,
        unsafe_ttc_events=unsafe_events,
        safety_interventions=interventions,
    ), trace


def _compact_trace(trace: list[dict[str, Any]]) -> list[list[Any]]:
    def rounded(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 6) if math.isfinite(value) else None
        if isinstance(value, list):
            return [rounded(item) for item in value]
        return value

    return [[rounded(row[key]) for key in TRACE_COLUMNS] for row in trace]


def run_diagnostic(protocol: dict[str, Any], output: Path, protocol_sha256: str) -> None:
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnostic: {output}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic": "fusion_action_timing",
        "non_interventional": True,
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
        "reference_reproduced": False,
        "trace_columns": TRACE_COLUMNS,
        "trace_float_decimals": 6,
        "nonfinite_trace_values": "null means no finite prediction; not a safety guarantee",
        "cases": [],
        "runtime": {
            "python": platform.python_version(),
            "script_sha256": file_sha256(__file__),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "torch",
                    "gymnasium",
                    "highway-env",
                    "stable-baselines3",
                )
            },
            "model_devices": {},
        },
    }
    models: dict[str, Any] = {}
    envs: dict[str, Any] = {}
    failure: BaseException | None = None
    try:
        validate_protocol(protocol)
        _require_hash(protocol["config"], protocol["config_sha256"])
        references = {
            name: _read_reference(spec, protocol) for name, spec in protocol["policies"].items()
        }
        selected = select_cases(
            references["v1"],
            references["v2"],
            protocol["first_seed"],
            protocol["controls_per_group"],
        )
        transition_counts = dict(Counter(case["transition"] for case in selected))
        if transition_counts != protocol["expected_selected_transition_counts"]:
            raise ValueError("Selected transition counts disagree with frozen protocol")
        report["selection"] = selected
        report["selected_transition_counts"] = transition_counts
        diagnostic = protocol["diagnostic"]
        for name, spec in protocol["policies"].items():
            models[name] = PPO.load(spec["model"])
            report["runtime"]["model_devices"][name] = str(models[name].device)
            envs[name] = make_intersection_env(
                config_path=protocol["config"],
                seed=protocol["first_seed"],
                risk_fusion=True,
                **protocol["fusion"],
            )
            if models[name].observation_space != envs[name].observation_space:
                raise ValueError(f"Observation space mismatch: {name}")
            if models[name].action_space != envs[name].action_space:
                raise ValueError(f"Action space mismatch: {name}")
            if envs[name].unwrapped.config["policy_frequency"] != protocol["policy_frequency"]:
                raise ValueError("Policy frequency differs from frozen protocol")
        for case in selected:
            recorded: dict[str, Any] = {**case, "policies": {}}
            report["cases"].append(recorded)
            traces = {}
            for name in ("v1", "v2"):
                metrics, trace = replay_episode(
                    models[name],
                    envs[name],
                    case["seed"],
                    protocol["unsafe_ttc_threshold"],
                    diagnostic,
                )
                traces[name] = trace
                reference = references[name].iloc[case["row_index"]]
                mismatches = verify_episode(metrics, reference)
                recorded["policies"][name] = {
                    "episode_metrics": metrics.as_dict(),
                    "expected_episode_metrics": reference.to_dict(),
                    "mismatched_fields": mismatches,
                    "summary": summarize_trace(
                        trace,
                        protocol["policy_frequency"],
                        diagnostic["stopped_speed"],
                        diagnostic["terminal_window_s"],
                    ),
                    "trace": _compact_trace(trace),
                }
                if mismatches:
                    raise RuntimeError(f"Reference mismatch for {name}, seed {case['seed']}")
            comparison = compare_traces(traces["v1"], traces["v2"])
            recorded["comparison"] = comparison
            if not comparison["common_observation_prefix_until_action_divergence"]:
                raise RuntimeError(f"Observation prefix mismatch for seed {case['seed']}")
            print(
                f"verified {len(report['cases'])}/{len(selected)} seed={case['seed']}", flush=True
            )
        report["reference_reproduced"] = True
        report["verified_episode_count"] = 2 * len(selected)
    except (Exception, KeyboardInterrupt) as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        failure = error
    finally:
        cleanup_errors = {}
        for name, env in envs.items():
            try:
                env.close()
            except Exception as error:
                cleanup_errors[name] = str(error)
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            report["reference_reproduced"] = False
            if failure is None:
                failure = RuntimeError("Environment cleanup failed")
                report["failure"] = {"type": type(failure).__name__, "message": str(failure)}
    save_report(output, _json_value(report))
    if failure is not None:
        raise RuntimeError(f"Diagnostic failed; preserve {output}") from failure
    print(f"Saved verified diagnostic to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Preserve existing diagnostic: {output}")
    try:
        _require_hash(args.protocol, args.protocol_sha256)
        protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    except Exception as error:
        save_report(
            output,
            {
                "schema_version": 1,
                "diagnostic": "fusion_action_timing",
                "reference_reproduced": False,
                "phase": "protocol_loading",
                "protocol_path": args.protocol,
                "protocol_sha256": args.protocol_sha256,
                "failure": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise RuntimeError(f"Protocol loading failed; preserve {output}") from error
    run_diagnostic(protocol, output, args.protocol_sha256.lower())


if __name__ == "__main__":
    main()
