from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from safeintent_rl.envs import make_intersection_env
from safeintent_rl.evaluation import detect_success


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained PPO policy drive live")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--intent-model", default=None)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = make_intersection_env(
        render_mode="human",
        seed=args.seed,
        safety_shield=args.safety_shield,
        intent_model=args.intent_model,
    )
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            terminated = truncated = False
            total_reward = 0.0
            final_info: dict = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)

            crashed = bool(getattr(env.unwrapped.vehicle, "crashed", False))
            arrived = detect_success(env, final_info)
            print(
                f"episode={episode + 1} reward={total_reward:.3f} "
                f"arrived={arrived} crashed={crashed}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
