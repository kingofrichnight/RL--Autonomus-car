from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.evaluation import EpisodeMetrics, detect_success, summarize_episodes
from safeintent_rl.intent.diagnostics import OnlineIntentDiagnostics
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
    parser.add_argument("--intent-device", default="cpu")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--unsafe-ttc", type=float, default=2.0)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--reference-csv-sha256", required=True)
    parser.add_argument(
        "--output",
        default="results/ppo_intent_v1_online_diagnostics_seed10042.json",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.intent_neighbors <= 0:
        parser.error("--intent-neighbors must be positive")
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
    )
    wrapper = _intent_wrapper(env)
    diagnostics = OnlineIntentDiagnostics()
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

    if failure is None:
        try:
            if diagnostics.decisions != sum(episode.length for episode in episodes):
                raise RuntimeError("Diagnostic decision count differs from episode lengths")
            if diagnostics.slot_observations != diagnostics.decisions * args.intent_neighbors:
                raise RuntimeError("Diagnostic slot count differs from decisions times neighbors")
            _verify_reference(episodes, args.reference_csv)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
    result = {
        "status": "complete" if failure is None else "failed",
        "error": failure,
        "episodes": args.episodes,
        "episodes_completed": len(episodes),
        "first_seed": args.seed,
        "last_seed": args.seed + args.episodes - 1,
        "driving_reference_reproduced": failure is None,
        "driving_metrics": summarize_episodes(episodes) if episodes else None,
        "failed_episode_results": (
            [episode.as_dict() for episode in episodes] if failure is not None else None
        ),
        "online_intent": diagnostics.summarize(wrapper.predictor.label_names),
        "model_path": str(Path(args.model)),
        "model_sha256": model_sha256,
        "config_path": str(Path(args.config)),
        "config_sha256": config_sha256,
        "intent_model_path": str(Path(args.intent_model)),
        "intent_model_sha256": intent_model_sha256,
        "intent_neighbors": args.intent_neighbors,
        "intent_device": args.intent_device,
        "policy_device": "cpu",
        "safety_shield": False,
        "unsafe_ttc_threshold": args.unsafe_ttc,
        "measurement_unit": (
            "pre-action visible slot; classification requires ready history and label"
        ),
        "reference_csv": str(Path(args.reference_csv)),
        "reference_csv_sha256": reference_sha256,
    }
    _save_report(output, result)
    print(json.dumps(result["online_intent"], indent=2))
    print(f"Saved online intent diagnostics to {output}")
    if failure is not None:
        raise RuntimeError(f"Diagnostic failed; preserve {output}: {failure}")


if __name__ == "__main__":
    main()
