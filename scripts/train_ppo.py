from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from safeintent_rl.envs.intersection import make_intersection_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO intersection policy")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--ttc-threshold", type=float, default=2.0)
    parser.add_argument("--intent-model", default=None)
    parser.add_argument("--intent-neighbors", type=int, default=5)
    parser.add_argument("--output", default="models/ppo_intersection")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs") / output.stem
    log_dir.mkdir(parents=True, exist_ok=True)

    def build_env(seed_offset: int = 0):
        env = make_intersection_env(
            seed=args.seed + seed_offset,
            safety_shield=args.safety_shield,
            ttc_threshold=args.ttc_threshold,
            intent_model=args.intent_model,
            intent_neighbors=args.intent_neighbors,
        )
        return Monitor(env)

    train_env = DummyVecEnv([lambda: build_env(0)])
    eval_env = DummyVecEnv([lambda: build_env(10_000)])
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        policy_kwargs={"net_arch": [256, 256]},
    )
    callbacks = [
        CheckpointCallback(save_freq=25_000, save_path=str(log_dir / "checkpoints")),
        EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best"),
            log_path=str(log_dir / "evaluation"),
            eval_freq=10_000,
            n_eval_episodes=20,
            deterministic=True,
        ),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(output)
    train_env.close()
    eval_env.close()
    print(f"Saved PPO policy to {output.with_suffix('.zip')}")


if __name__ == "__main__":
    main()
