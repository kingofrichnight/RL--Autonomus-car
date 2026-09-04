from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.evaluation import EpisodeMetrics, detect_success, summarize_episodes
from safeintent_rl.intent.inference import file_sha256
from safeintent_rl.safety.ttc import minimum_ttc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None, help="Environment YAML configuration")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--unsafe-ttc", type=float, default=2.0)
    parser.add_argument("--intent-model", default=None)
    parser.add_argument("--intent-neighbors", type=int, default=5)
    parser.add_argument("--intent-device", default="cpu")
    parser.add_argument("--intent-model-sha256", default=None)
    parser.add_argument("--output", default="results/evaluation.csv")
    args = parser.parse_args()

    if args.intent_model_sha256 is not None and args.intent_model is None:
        parser.error("--intent-model-sha256 requires --intent-model")

    model_sha256 = file_sha256(args.model)
    config_sha256 = file_sha256(args.config) if args.config is not None else None
    intent_model_sha256 = (
        file_sha256(args.intent_model) if args.intent_model is not None else None
    )
    if (
        args.intent_model_sha256 is not None
        and intent_model_sha256 is not None
        and intent_model_sha256.lower() != args.intent_model_sha256.lower()
    ):
        raise ValueError("Intent checkpoint fingerprint does not match the requested model")

    model = PPO.load(args.model)
    env = make_intersection_env(
        config_path=args.config,
        seed=args.seed,
        safety_shield=args.safety_shield,
        intent_model=args.intent_model,
        intent_neighbors=args.intent_neighbors,
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
            collision = bool(final_info.get("crashed", getattr(base.vehicle, "crashed", False)))
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

    output = Path(args.output)
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
            "safety_shield": args.safety_shield,
            "unsafe_ttc_threshold": args.unsafe_ttc,
            "intent_model_path": (
                str(Path(args.intent_model)) if args.intent_model is not None else None
            ),
            "intent_model_sha256": intent_model_sha256,
            "intent_neighbors": args.intent_neighbors if args.intent_model is not None else 0,
            "intent_device": args.intent_device if args.intent_model is not None else None,
        }
    )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved episode results to {output}")


if __name__ == "__main__":
    main()
