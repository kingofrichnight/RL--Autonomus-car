from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.intent.dataset import LABELS, TrajectoryCollector, save_dataset
from safeintent_rl.intent.training import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled NPC trajectory histories")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/intent_trajectories.npz")
    parser.add_argument(
        "--summary-output",
        default="results/intent_dataset_seed42.summary.json",
    )
    args = parser.parse_args()

    env = make_intersection_env(seed=args.seed, driver_behaviors=True)
    collector = TrajectoryCollector(args.history_length, args.sample_stride)
    samples = []
    labels = []
    episode_ids = []
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            collector.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                new_samples, new_labels = collector.observe(env)
                samples.extend(new_samples)
                labels.extend(new_labels)
                episode_ids.extend([episode] * len(new_samples))
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if (episode + 1) % 25 == 0:
                print(f"episode={episode + 1} samples={len(samples)}")
    finally:
        env.close()

    metadata = {
        "collector": "seeded_random_ego_policy",
        "episodes": args.episodes,
        "first_seed": args.seed,
        "last_seed": args.seed + args.episodes - 1,
        "history_length": args.history_length,
        "sample_stride": args.sample_stride,
    }
    save_dataset(args.output, samples, labels, episode_ids, metadata=metadata)
    label_counts = {name: labels.count(index) for name, index in LABELS.items()}
    episode_sample_counts = Counter(episode_ids)
    summary = {
        **metadata,
        "samples": len(samples),
        "feature_count": 6,
        "label_counts": label_counts,
        "label_fractions": {
            name: count / max(len(samples), 1) for name, count in label_counts.items()
        },
        "episodes_with_samples": len(episode_sample_counts),
        "minimum_samples_per_episode": min(episode_sample_counts.values()),
        "maximum_samples_per_episode": max(episode_sample_counts.values()),
        "dataset_sha256": file_sha256(args.output),
        "dataset_path": str(Path(args.output)),
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} histories to {args.output}")
    print(f"Saved collection summary to {summary_output}")
    print(f"label_counts={label_counts}")
    print(f"episode_seeds={args.seed}-{args.seed + args.episodes - 1}")


if __name__ == "__main__":
    main()
