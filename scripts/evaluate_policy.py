from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.evaluation import (
    EpisodeMetrics,
    detect_collision,
    detect_success,
    summarize_episodes,
)
from safeintent_rl.intent.inference import file_sha256, load_intent_checkpoint
from safeintent_rl.safety.ttc import minimum_ttc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", default=None)
    parser.add_argument("--config", default=None, help="Environment YAML configuration")
    parser.add_argument("--config-sha256", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    shield_group = parser.add_mutually_exclusive_group()
    shield_group.add_argument("--safety-shield", action="store_true")
    shield_group.add_argument("--cpa-shield", action="store_true")
    parser.add_argument("--unsafe-ttc", type=float, default=2.0)
    parser.add_argument("--cpa-time-threshold", type=float, default=2.0)
    parser.add_argument("--cpa-distance-threshold", type=float, default=3.0)
    parser.add_argument("--cpa-horizon", type=float, default=3.0)
    parser.add_argument("--cpa-max-range", type=float, default=60.0)
    parser.add_argument("--intent-model", default=None)
    parser.add_argument("--intent-neighbors", type=int, default=5)
    parser.add_argument("--intent-history-length", type=int, default=None)
    parser.add_argument("--intent-history-tracking-neighbors", type=int, default=None)
    parser.add_argument("--intent-device", default="cpu")
    parser.add_argument("--intent-model-sha256", default=None)
    parser.add_argument("--output", default="results/evaluation.csv")
    parser.add_argument("--refuse-overwrite", action="store_true")
    args = parser.parse_args()

    if args.intent_model_sha256 is not None and args.intent_model is None:
        parser.error("--intent-model-sha256 requires --intent-model")
    if args.config_sha256 is not None and args.config is None:
        parser.error("--config-sha256 requires --config")
    if (
        args.intent_model is None
        and (
            args.intent_history_length is not None
            or args.intent_history_tracking_neighbors is not None
        )
    ):
        parser.error("intent history settings require --intent-model")

    model_sha256 = file_sha256(args.model)
    if args.model_sha256 is not None and model_sha256.lower() != args.model_sha256.lower():
        raise ValueError("PPO checkpoint fingerprint does not match --model-sha256")
    config_sha256 = file_sha256(args.config) if args.config is not None else None
    if (
        args.config_sha256 is not None
        and config_sha256 is not None
        and config_sha256.lower() != args.config_sha256.lower()
    ):
        raise ValueError("Configuration fingerprint does not match --config-sha256")
    intent_model_sha256 = (
        file_sha256(args.intent_model) if args.intent_model is not None else None
    )
    if (
        args.intent_model_sha256 is not None
        and intent_model_sha256 is not None
        and intent_model_sha256.lower() != args.intent_model_sha256.lower()
    ):
        raise ValueError("Intent checkpoint fingerprint does not match the requested model")
    intent_history_length = None
    intent_history_tracking_neighbors = None
    if args.intent_model is not None:
        checkpoint = load_intent_checkpoint(args.intent_model, map_location="cpu")
        checkpoint_history_length = int(checkpoint.get("history_length", 10))
        intent_history_length = (
            checkpoint_history_length
            if args.intent_history_length is None
            else args.intent_history_length
        )
        if intent_history_length != checkpoint_history_length:
            raise ValueError("Intent history length does not match the checkpoint")
        intent_history_tracking_neighbors = (
            args.intent_neighbors
            if args.intent_history_tracking_neighbors is None
            else args.intent_history_tracking_neighbors
        )

    output = Path(args.output)
    summary_path = output.with_suffix(".summary.json")
    if args.refuse_overwrite and output.exists():
        raise FileExistsError(f"Refusing to overwrite episode results: {output}")
    if args.refuse_overwrite and summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation summary: {summary_path}")

    model = PPO.load(args.model)
    env = make_intersection_env(
        config_path=args.config,
        seed=args.seed,
        safety_shield=args.safety_shield,
        cpa_safety_shield=args.cpa_shield,
        cpa_time_threshold=args.cpa_time_threshold,
        cpa_distance_threshold=args.cpa_distance_threshold,
        cpa_horizon=args.cpa_horizon,
        cpa_max_range=args.cpa_max_range,
        intent_model=args.intent_model,
        intent_neighbors=args.intent_neighbors,
        intent_history_length=intent_history_length,
        intent_history_tracking_neighbors=intent_history_tracking_neighbors,
        intent_device=args.intent_device,
        intent_model_sha256=intent_model_sha256,
    )
    episodes: list[EpisodeMetrics] = []
    try:
        for episode_index in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode_index)
            terminated = truncated = False
            reward_sum = 0.0
            steps = 0
            min_ttc = math.inf
            unsafe_events = 0
            interventions = 0
            final_info: dict = {}
            while not (terminated or truncated):
                base = env.unwrapped
                pre_step_ttc = minimum_ttc(base.vehicle, list(base.road.vehicles))
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                steps += 1
                step_ttc = float(info.get("min_ttc", pre_step_ttc))
                min_ttc = min(min_ttc, step_ttc)
                unsafe_events += int(step_ttc <= args.unsafe_ttc)
                interventions += int(info.get("safety_intervened", False))
                final_info = info

            base = env.unwrapped
            collision = detect_collision(env, final_info)
            success = detect_success(env, final_info)
            policy_frequency = float(getattr(base, "config", {}).get("policy_frequency", 1))
            episodes.append(
                EpisodeMetrics(
                    reward=reward_sum,
                    length=steps,
                    success=success,
                    collision=collision,
                    travel_time=steps / policy_frequency,
                    min_ttc=min_ttc,
                    unsafe_ttc_events=unsafe_events,
                    safety_interventions=interventions,
                )
            )
    finally:
        env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.as_dict() for item in episodes]).to_csv(output, index=False)
    summary = summarize_episodes(episodes)
    summary.update(
        {
            "model_path": str(Path(args.model)),
            "model_sha256": model_sha256,
            "config_path": str(Path(args.config)) if args.config is not None else None,
            "config_sha256": config_sha256,
            "first_seed": args.seed,
            "last_seed": args.seed + args.episodes - 1,
            "safety_shield": bool(args.safety_shield or args.cpa_shield),
            "safety_shield_type": (
                "radial_ttc"
                if args.safety_shield
                else "cpa_acceleration_veto"
                if args.cpa_shield
                else None
            ),
            "unsafe_ttc_threshold": args.unsafe_ttc,
            "cpa_time_threshold": (
                args.cpa_time_threshold if args.cpa_shield else None
            ),
            "cpa_distance_threshold": (
                args.cpa_distance_threshold if args.cpa_shield else None
            ),
            "cpa_horizon": args.cpa_horizon if args.cpa_shield else None,
            "cpa_max_range": args.cpa_max_range if args.cpa_shield else None,
            "intent_model_path": (
                str(Path(args.intent_model)) if args.intent_model is not None else None
            ),
            "intent_model_sha256": intent_model_sha256,
            "intent_neighbors": args.intent_neighbors if args.intent_model is not None else 0,
            "intent_history_length": intent_history_length,
            "intent_history_tracking_neighbors": intent_history_tracking_neighbors,
            "intent_device": args.intent_device if args.intent_model is not None else None,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved episode results to {output}")


if __name__ == "__main__":
    main()
