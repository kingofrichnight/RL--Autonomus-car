from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.evaluation import EpisodeMetrics, detect_success, summarize_episodes
from safeintent_rl.intent.diagnostics import HistoryCoverageComparison, OnlineIntentDiagnostics
from safeintent_rl.intent.inference import file_sha256
from safeintent_rl.intent.wrapper import IntentObservationWrapper
from safeintent_rl.safety.ttc import minimum_ttc


def _require_sha256(path: str, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"{label} fingerprint mismatch: expected {expected}, got {actual}")
    return actual


def _intent_wrapper(env: Any) -> IntentObservationWrapper:
    current = env
    while True:
        if isinstance(current, IntentObservationWrapper):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise TypeError("Environment does not contain an IntentObservationWrapper")


def _verify_reference(episodes: list[EpisodeMetrics], reference_path: str) -> None:
    observed = pd.DataFrame([episode.as_dict() for episode in episodes])
    reference = pd.read_csv(reference_path)
    if list(reference.columns) != list(observed.columns) or len(reference) != len(observed):
        raise RuntimeError("Diagnostic rollout does not match the reference CSV structure")
    if not np.allclose(
        observed.to_numpy(dtype=np.float64),
        reference.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError("Diagnostic rollout changed at least one reference episode result")


def _verify_diagnostic_reference(
    observed: Any,
    reference: Any,
    *,
    path: str = "online_intent",
) -> None:
    """Require the repeated five-slot diagnostic to reproduce its frozen JSON."""
    if isinstance(reference, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(reference):
            raise RuntimeError(f"Diagnostic reference structure changed at {path}")
        for key in reference:
            _verify_diagnostic_reference(
                observed[key],
                reference[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
        if (
            not isinstance(observed, Sequence)
            or isinstance(observed, (str, bytes))
            or len(observed) != len(reference)
        ):
            raise RuntimeError(f"Diagnostic reference sequence changed at {path}")
        for index, (actual, expected) in enumerate(zip(observed, reference, strict=True)):
            _verify_diagnostic_reference(actual, expected, path=f"{path}[{index}]")
        return
    if (
        isinstance(reference, Real)
        and not isinstance(reference, bool)
        and isinstance(observed, Real)
        and not isinstance(observed, bool)
    ):
        if not math.isclose(float(observed), float(reference), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Diagnostic reference value changed at {path}")
        return
    if observed != reference:
        raise RuntimeError(f"Diagnostic reference value changed at {path}")


def _save_report(output: Path, result: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure online intent coverage and accuracy without changing PPO actions"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--intent-model", required=True)
    parser.add_argument("--intent-model-sha256", required=True)
    parser.add_argument("--intent-neighbors", type=int, default=5)
    parser.add_argument("--shadow-history-neighbors", type=int)
    parser.add_argument("--intent-device", default="cpu")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--unsafe-ttc", type=float, default=2.0)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--reference-csv-sha256", required=True)
    parser.add_argument("--reference-diagnostic-json")
    parser.add_argument("--reference-diagnostic-json-sha256")
    parser.add_argument(
        "--output",
        default="results/ppo_intent_v1_online_diagnostics_seed10042.json",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.intent_neighbors <= 0:
        parser.error("--intent-neighbors must be positive")
    shadow_requested = args.shadow_history_neighbors is not None
    if shadow_requested and args.shadow_history_neighbors < args.intent_neighbors:
        parser.error("--shadow-history-neighbors must cover all intent neighbors")
    if shadow_requested != bool(args.reference_diagnostic_json):
        parser.error("shadow tracking and --reference-diagnostic-json must be used together")
    if shadow_requested != bool(args.reference_diagnostic_json_sha256):
        parser.error(
            "shadow tracking and --reference-diagnostic-json-sha256 must be used together"
        )
    if not math.isfinite(args.unsafe_ttc) or args.unsafe_ttc < 0:
        parser.error("--unsafe-ttc must be finite and non-negative")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Existing diagnostic result must be preserved: {output}")

    model_sha256 = _require_sha256(args.model, args.model_sha256, "PPO model")
    config_sha256 = _require_sha256(args.config, args.config_sha256, "configuration")
    intent_model_sha256 = _require_sha256(
        args.intent_model,
        args.intent_model_sha256,
        "intent model",
    )
    reference_sha256 = _require_sha256(
        args.reference_csv,
        args.reference_csv_sha256,
        "reference CSV",
    )
    reference = pd.read_csv(args.reference_csv)
    if (
        list(reference.columns) != list(EpisodeMetrics.__dataclass_fields__)
        or len(reference) != args.episodes
    ):
        raise ValueError("Reference CSV must have the episode-metric schema and requested rows")
    reference_diagnostic_sha256: str | None = None
    reference_online_intent: dict[str, Any] | None = None
    if shadow_requested:
        reference_diagnostic_sha256 = _require_sha256(
            args.reference_diagnostic_json,
            args.reference_diagnostic_json_sha256,
            "reference diagnostic JSON",
        )
        with Path(args.reference_diagnostic_json).open(encoding="utf-8") as handle:
            reference_diagnostic = json.load(handle)
        if (
            reference_diagnostic.get("status") != "complete"
            or reference_diagnostic.get("episodes") != args.episodes
            or reference_diagnostic.get("first_seed") != args.seed
            or reference_diagnostic.get("last_seed") != args.seed + args.episodes - 1
            or reference_diagnostic.get("model_sha256", "").lower()
            != model_sha256.lower()
            or reference_diagnostic.get("config_sha256", "").lower()
            != config_sha256.lower()
            or reference_diagnostic.get("intent_model_sha256", "").lower()
            != intent_model_sha256.lower()
            or reference_diagnostic.get("reference_csv_sha256", "").lower()
            != reference_sha256.lower()
            or not isinstance(reference_diagnostic.get("online_intent"), dict)
        ):
            raise ValueError("Reference diagnostic JSON does not match the frozen run")
        reference_online_intent = reference_diagnostic["online_intent"]

    model = PPO.load(args.model, device="cpu")
    env = make_intersection_env(
        config_path=args.config,
        seed=args.seed,
        safety_shield=False,
        intent_model=args.intent_model,
        intent_neighbors=args.intent_neighbors,
        intent_device=args.intent_device,
        intent_model_sha256=intent_model_sha256,
        intent_diagnostics=True,
        intent_shadow_history_neighbors=args.shadow_history_neighbors,
    )
    wrapper = _intent_wrapper(env)
    diagnostics = OnlineIntentDiagnostics()
    shadow_diagnostics = OnlineIntentDiagnostics() if shadow_requested else None
    coverage_comparison = HistoryCoverageComparison() if shadow_requested else None
    episodes: list[EpisodeMetrics] = []
    failure: str | None = None
    try:
        for episode_index in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode_index)
            terminated = truncated = False
            reward_sum = 0.0
            steps = 0
            min_episode_ttc = math.inf
            unsafe_events = 0
            final_info: dict[str, Any] = {}
            while not (terminated or truncated):
                diagnostics.update(wrapper.last_intent_diagnostics)
                if shadow_diagnostics is not None:
                    shadow_diagnostics.update(wrapper.last_shadow_intent_diagnostics)
                    coverage_comparison.update(
                        wrapper.last_intent_diagnostics,
                        wrapper.last_shadow_intent_diagnostics,
                    )
                base = env.unwrapped
                pre_step_ttc = minimum_ttc(base.vehicle, list(base.road.vehicles))
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                steps += 1
                step_ttc = float(info.get("min_ttc", pre_step_ttc))
                min_episode_ttc = min(min_episode_ttc, step_ttc)
                unsafe_events += int(step_ttc <= args.unsafe_ttc)
                final_info = info

            base = env.unwrapped
            collision = bool(final_info.get("crashed", getattr(base.vehicle, "crashed", False)))
            policy_frequency = float(base.config.get("policy_frequency", 1))
            episodes.append(
                EpisodeMetrics(
                    reward=reward_sum,
                    length=steps,
                    success=detect_success(env, final_info),
                    collision=collision,
                    travel_time=steps / policy_frequency,
                    min_ttc=min_episode_ttc,
                    unsafe_ttc_events=unsafe_events,
                    safety_interventions=0,
                )
            )
            if (episode_index + 1) % 25 == 0:
                print(f"episode={episode_index + 1}", flush=True)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        env.close()

    online_summary = diagnostics.summarize(wrapper.predictor.label_names)
    shadow_summary = (
        shadow_diagnostics.summarize(wrapper.predictor.label_names)
        if shadow_diagnostics is not None
        else None
    )
    driving_reference_reproduced = False
    online_intent_reference_reproduced = (
        False if reference_online_intent is not None else None
    )
    if failure is None:
        try:
            if diagnostics.decisions != sum(episode.length for episode in episodes):
                raise RuntimeError("Diagnostic decision count differs from episode lengths")
            if diagnostics.slot_observations != diagnostics.decisions * args.intent_neighbors:
                raise RuntimeError("Diagnostic slot count differs from decisions times neighbors")
            if shadow_diagnostics is not None:
                if shadow_diagnostics.decisions != diagnostics.decisions:
                    raise RuntimeError("Shadow and policy diagnostic decision counts differ")
                if shadow_diagnostics.slot_observations != diagnostics.slot_observations:
                    raise RuntimeError("Shadow and policy diagnostic slot counts differ")
                if coverage_comparison.decisions != diagnostics.decisions:
                    raise RuntimeError("Coverage comparison and diagnostic decisions differ")
            _verify_reference(episodes, args.reference_csv)
            driving_reference_reproduced = True
            if reference_online_intent is not None:
                _verify_diagnostic_reference(online_summary, reference_online_intent)
                online_intent_reference_reproduced = True
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
    result = {
        "status": "complete" if failure is None else "failed",
        "error": failure,
        "episodes": args.episodes,
        "episodes_completed": len(episodes),
        "first_seed": args.seed,
        "last_seed": args.seed + args.episodes - 1,
        "driving_reference_reproduced": driving_reference_reproduced,
        "online_intent_reference_reproduced": online_intent_reference_reproduced,
        "driving_metrics": summarize_episodes(episodes) if episodes else None,
        "failed_episode_results": (
            [episode.as_dict() for episode in episodes] if failure is not None else None
        ),
        "online_intent": online_summary,
        "shadow_online_intent": shadow_summary,
        "coverage_comparison": (
            coverage_comparison.summarize() if coverage_comparison is not None else None
        ),
        "model_path": str(Path(args.model)),
        "model_sha256": model_sha256,
        "config_path": str(Path(args.config)),
        "config_sha256": config_sha256,
        "intent_model_path": str(Path(args.intent_model)),
        "intent_model_sha256": intent_model_sha256,
        "intent_neighbors": args.intent_neighbors,
        "shadow_history_neighbors": args.shadow_history_neighbors,
        "intent_device": args.intent_device,
        "policy_device": "cpu",
        "safety_shield": False,
        "unsafe_ttc_threshold": args.unsafe_ttc,
        "measurement_unit": (
            "pre-action visible slot; classification requires ready history and label"
        ),
        "reference_csv": str(Path(args.reference_csv)),
        "reference_csv_sha256": reference_sha256,
        "reference_diagnostic_json": (
            str(Path(args.reference_diagnostic_json))
            if args.reference_diagnostic_json is not None
            else None
        ),
        "reference_diagnostic_json_sha256": reference_diagnostic_sha256,
    }
    _save_report(output, result)
    print(json.dumps(result["online_intent"], indent=2))
    if shadow_summary is not None:
        print(json.dumps({"shadow_online_intent": shadow_summary}, indent=2))
    print(f"Saved online intent diagnostics to {output}")
    if failure is not None:
        raise RuntimeError(f"Diagnostic failed; preserve {output}: {failure}")


if __name__ == "__main__":
    main()
