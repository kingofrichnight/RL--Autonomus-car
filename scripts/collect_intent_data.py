from __future__ import annotations

import argparse

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.intent.dataset import TrajectoryCollector, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled NPC trajectory histories")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/intent_trajectories.npz")
    args = parser.parse_args()

    env = make_intersection_env(seed=args.seed, driver_behaviors=True)
    collector = TrajectoryCollector(args.history_length, args.sample_stride)
    samples = []
    labels = []
    episode_ids = []
    try:
        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
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

    save_dataset(args.output, samples, labels, episode_ids)
    print(f"Saved {len(samples)} histories to {args.output}")


if __name__ == "__main__":
    main()
