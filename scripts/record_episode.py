from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from safeintent_rl.envs import make_intersection_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one PPO episode as MP4")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="results/videos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--intent-model", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_env = make_intersection_env(
        render_mode="rgb_array",
        seed=args.seed,
        safety_shield=args.safety_shield,
        intent_model=args.intent_model,
    )
    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(output_dir),
        episode_trigger=lambda episode: episode == 0,
        name_prefix="safeintent-ppo",
    )
    model = PPO.load(args.model)
    try:
        observation, _ = env.reset(seed=args.seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()
    print(f"Saved video in {output_dir}")


if __name__ == "__main__":
    main()
