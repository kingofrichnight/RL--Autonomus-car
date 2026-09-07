from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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

CPA_TIME_THRESHOLDS = (0.5, 0.75, 1.0, 1.5, 2.0)
CPA_DISTANCE_THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


@dataclass(frozen=True)
class DecisionRecord:
    action_name: str
    radial_ttc: float
    cpa_pairs: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class EpisodeTrace:
    success: bool
    collision: bool
    decisions: tuple[DecisionRecord, ...]


def _require_sha256(path: str, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"{label} fingerprint mismatch: expected {expected}, got {actual}")
    return actual


def _verify_reference(episodes: list[EpisodeMetrics], reference_path: str) -> None:
    observed = pd.DataFrame([episode.as_dict() for episode in episodes])
    reference = pd.read_csv(reference_path)
    if list(reference.columns) != list(observed.columns) or len(reference) != len(observed):
        raise RuntimeError("Conflict diagnostic does not match the reference CSV structure")
    if not np.allclose(
        observed.to_numpy(dtype=np.float64),
        reference.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError("Conflict diagnostic changed at least one reference episode result")


def _window_profile(
    traces: list[EpisodeTrace],
    predicate: Callable[[DecisionRecord], bool],
    *,
    window_steps: int | None,
) -> dict[str, float | int]:
    total_decisions = 0
    trigger_decisions = 0
    success_trigger_decisions = 0
    collision_trigger_decisions = 0
    success_episodes = 0
    collision_episodes = 0
    success_episodes_triggered = 0
    collision_episodes_triggered = 0

    for trace in traces:
        if trace.success:
            success_episodes += 1
        elif trace.collision:
            collision_episodes += 1
        selected = (
            trace.decisions
            if window_steps is None
            else trace.decisions[-window_steps:]
        )
        total_decisions += len(selected)
        triggered = sum(predicate(decision) for decision in selected)
        trigger_decisions += triggered
        if trace.success:
            success_trigger_decisions += triggered
            success_episodes_triggered += int(triggered > 0)
        elif trace.collision:
            collision_trigger_decisions += triggered
            collision_episodes_triggered += int(triggered > 0)

    return {
        "window_decisions": total_decisions,
        "trigger_decisions": trigger_decisions,
        "trigger_rate": trigger_decisions / total_decisions if total_decisions else 0.0,
        "success_trigger_decisions": success_trigger_decisions,
        "collision_trigger_decisions": collision_trigger_decisions,
        "success_episodes_triggered": success_episodes_triggered,
        "success_episode_burden": (
            success_episodes_triggered / success_episodes if success_episodes else 0.0
        ),
        "collision_episodes_triggered": collision_episodes_triggered,
        "collision_episode_coverage": (
            collision_episodes_triggered / collision_episodes if collision_episodes else 0.0
        ),
    }


def _profile_windows(
    traces: list[EpisodeTrace],
    predicate: Callable[[DecisionRecord], bool],
    *,
    policy_frequency: float,
) -> dict[str, dict[str, float | int]]:
    one_second = max(1, math.ceil(policy_frequency))
    two_seconds = max(1, math.ceil(2 * policy_frequency))
    return {
        "all": _window_profile(traces, predicate, window_steps=None),
        "last_1_second": _window_profile(traces, predicate, window_steps=one_second),
        "last_2_seconds": _window_profile(traces, predicate, window_steps=two_seconds),
    }


def build_conflict_profiles(
    traces: list[EpisodeTrace],
    *,
    policy_frequency: float,
    legacy_ttc_threshold: float,
) -> dict[str, Any]:
    legacy = _profile_windows(
        traces,
        lambda decision: (
            decision.action_name in {"FASTER", "IDLE"}
            and decision.radial_ttc <= legacy_ttc_threshold
        ),
        policy_frequency=policy_frequency,
    )
    grid: list[dict[str, Any]] = []
    for action_scope, allowed_actions in (
        ("FASTER_ONLY", {"FASTER"}),
        ("FASTER_OR_IDLE", {"FASTER", "IDLE"}),
    ):
        for time_threshold in CPA_TIME_THRESHOLDS:
            for distance_threshold in CPA_DISTANCE_THRESHOLDS:
                grid.append(
                    {
                        "action_scope": action_scope,
                        "cpa_time_threshold": time_threshold,
                        "cpa_distance_threshold": distance_threshold,
                        "windows": _profile_windows(
                            traces,
                            lambda decision, actions=allowed_actions, time=time_threshold,
                            distance=distance_threshold: (
                                decision.action_name in actions
                                and any(
                                    cpa_time <= time and cpa_distance <= distance
                                    for cpa_time, cpa_distance in decision.cpa_pairs
                                )
                            ),
                            policy_frequency=policy_frequency,
                        ),
                    }
                )
    return {
        "legacy_radial_ttc_rule": {
            "action_scope": "FASTER_OR_IDLE",
            "ttc_threshold": legacy_ttc_threshold,
            "windows": legacy,
        },
        "cpa_grid": grid,
    }


def _action_name(env: Any, action: Any) -> str:
    base = env.unwrapped
    actions = getattr(getattr(base, "action_type", None), "actions", {})
    value = int(action) if hasattr(action, "__int__") else action
    return str(actions.get(value, value)).upper()


def _save_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure action-conditioned conflict geometry without changing PPO actions"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--unsafe-ttc", type=float, default=2.0)
    parser.add_argument("--cpa-horizon", type=float, default=3.0)
    parser.add_argument("--max-range", type=float, default=60.0)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--reference-csv-sha256", required=True)
    parser.add_argument(
        "--output",
        default="results/ppo_reward_v3_conflict_diagnostics_seed10042.json",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    for name, value in (
        ("--unsafe-ttc", args.unsafe_ttc),
        ("--cpa-horizon", args.cpa_horizon),
        ("--max-range", args.max_range),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.cpa_horizon < max(CPA_TIME_THRESHOLDS):
        parser.error(
            f"--cpa-horizon must be at least {max(CPA_TIME_THRESHOLDS)} seconds"
        )

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Existing diagnostic result must be preserved: {output}")

    model_sha256 = _require_sha256(args.model, args.model_sha256, "PPO model")
    config_sha256 = _require_sha256(args.config, args.config_sha256, "configuration")
    reference_sha256 = _require_sha256(
        args.reference_csv,
        args.reference_csv_sha256,
        "reference CSV",
    )

    model = PPO.load(args.model, device="cpu")
    env = make_intersection_env(config_path=args.config, seed=args.seed)
    episodes: list[EpisodeMetrics] = []
    traces: list[EpisodeTrace] = []
    policy_frequency: float | None = None
    try:
        for episode_index in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode_index)
            terminated = truncated = False
            reward_sum = 0.0
            steps = 0
            min_ttc = math.inf
            unsafe_events = 0
            final_info: dict[str, Any] = {}
            decisions: list[DecisionRecord] = []
            while not (terminated or truncated):
                base = env.unwrapped
                ego = base.vehicle
                vehicles = list(base.road.vehicles)
                radial_ttc = minimum_ttc(ego, vehicles)
                approaches = nearby_closest_approaches(
                    ego,
                    vehicles,
                    horizon=args.cpa_horizon,
                    max_range=args.max_range,
                )
                action, _ = model.predict(observation, deterministic=True)
                decisions.append(
                    DecisionRecord(
                        action_name=_action_name(env, action),
                        radial_ttc=radial_ttc,
                        cpa_pairs=tuple(
                            (approach.time, approach.distance) for approach in approaches
                        ),
                    )
                )
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                steps += 1
                step_ttc = float(info.get("min_ttc", radial_ttc))
                min_ttc = min(min_ttc, step_ttc)
                unsafe_events += int(step_ttc <= args.unsafe_ttc)
                final_info = info

            base = env.unwrapped
            collision = detect_collision(env, final_info)
            success = detect_success(env, final_info)
            current_frequency = float(base.config.get("policy_frequency", 1))
            if policy_frequency is None:
                policy_frequency = current_frequency
            elif current_frequency != policy_frequency:
                raise RuntimeError("Policy frequency changed during diagnostic rollout")
            episodes.append(
                EpisodeMetrics(
                    reward=reward_sum,
                    length=steps,
                    success=success,
                    collision=collision,
                    travel_time=steps / current_frequency,
                    min_ttc=min_ttc,
                    unsafe_ttc_events=unsafe_events,
                    safety_interventions=0,
                )
            )
            traces.append(
                EpisodeTrace(
                    success=success,
                    collision=collision,
                    decisions=tuple(decisions),
                )
            )
    finally:
        env.close()

    report: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic": "action_conditioned_closest_approach",
        "non_interventional": True,
        "model_path": str(Path(args.model)),
        "model_sha256": model_sha256,
        "config_path": str(Path(args.config)),
        "config_sha256": config_sha256,
        "reference_csv_path": str(Path(args.reference_csv)),
        "reference_csv_sha256": reference_sha256,
        "episodes": args.episodes,
        "first_seed": args.seed,
        "last_seed": args.seed + args.episodes - 1,
        "unsafe_ttc_threshold": args.unsafe_ttc,
        "cpa_horizon": args.cpa_horizon,
        "max_range": args.max_range,
        "policy_frequency": policy_frequency,
        "episode_summary": summarize_episodes(episodes),
    }
    try:
        _verify_reference(episodes, args.reference_csv)
    except RuntimeError as error:
        report.update(
            {
                "reference_reproduced": False,
                "failure": str(error),
                "observed_episode_results": [asdict(episode) for episode in episodes],
            }
        )
        _save_report(output, report)
        raise RuntimeError(f"Diagnostic failed; preserve {output}") from error

    report.update(
        {
            "reference_reproduced": True,
            "decision_count": sum(len(trace.decisions) for trace in traces),
            "profiles": build_conflict_profiles(
                traces,
                policy_frequency=float(policy_frequency or 1),
                legacy_ttc_threshold=args.unsafe_ttc,
            ),
        }
    )
    _save_report(output, report)
    print(
        json.dumps(
            {
                "reference_reproduced": report["reference_reproduced"],
                "episodes": report["episodes"],
                "decision_count": report["decision_count"],
                "profile_count": len(report["profiles"]["cpa_grid"]),
            },
            indent=2,
        )
    )
    print(f"Saved conflict diagnostic to {output}")


if __name__ == "__main__":
    main()
