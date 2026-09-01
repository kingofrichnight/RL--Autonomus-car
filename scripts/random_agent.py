from __future__ import annotations

import argparse

from safeintent_rl.envs import make_intersection_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a random HighwayEnv intersection agent")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = make_intersection_env(render_mode="human" if args.render else None, seed=args.seed)
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            del observation, info
            terminated = truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                total_reward += float(reward)
            print(f"episode={episode + 1} reward={total_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

