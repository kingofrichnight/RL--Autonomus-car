from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from safeintent_rl.agents import RuleBasedAgent
from safeintent_rl.envs import make_intersection_env
from safeintent_rl.evaluation import EpisodeMetrics, detect_success, summarize_episodes
from safeintent_rl.safety.ttc import minimum_ttc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TTC rule-based baseline")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brake-ttc", type=float, default=2.0)
    parser.add_argument("--accelerate-ttc", type=float, default=4.0)
    parser.add_argument("--unsafe-ttc", type=float, default=2.0)
    parser.add_argument("--output", default="results/rule_based_seed42.csv")
    args = parser.parse_args()

    env = make_intersection_env(seed=args.seed)
    agent = RuleBasedAgent(
        env,
        brake_ttc=args.brake_ttc,
        accelerate_ttc=args.accelerate_ttc,
    )
    episodes: list[EpisodeMetrics] = []
    action_counts: Counter[int] = Counter()

    try:
        for episode_index in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode_index)
            terminated = truncated = False
            reward_sum = 0.0
            steps = 0
            min_episode_ttc = math.inf
            unsafe_events = 0
            final_info: dict = {}

            while not (terminated or truncated):
                base = env.unwrapped
                step_ttc = minimum_ttc(base.vehicle, list(base.road.vehicles))
                action = agent.predict(observation, min_ttc=step_ttc)
                action_counts[action] += 1
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                steps += 1
                min_episode_ttc = min(min_episode_ttc, step_ttc)
                unsafe_events += int(step_ttc <= args.unsafe_ttc)
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
                    min_ttc=min_episode_ttc,
                    unsafe_ttc_events=unsafe_events,
                    safety_interventions=0,
                )
            )
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.as_dict() for item in episodes]).to_csv(output, index=False)

    summary: dict[str, object] = summarize_episodes(episodes)
    summary.update(
        {
            "controller": "ttc_rule_based",
            "brake_ttc": args.brake_ttc,
            "accelerate_ttc": args.accelerate_ttc,
            "unsafe_ttc": args.unsafe_ttc,
            "action_counts": {str(action): count for action, count in sorted(action_counts.items())},
        }
    )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved episode results to {output}")


if __name__ == "__main__":
    main()
