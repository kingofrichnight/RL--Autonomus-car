from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from safeintent_rl.config import load_config
from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.intent.inference import (
    file_sha256,
    load_intent_checkpoint,
)


def training_seed_offsets(n_envs: int, seed_stride: int) -> list[int]:
    """Return deterministic, non-overlapping initial seed offsets."""
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if seed_stride <= 0:
        raise ValueError("env_seed_stride must be positive")
    return [index * seed_stride for index in range(n_envs)]


def callback_frequency(vector_timesteps: int, n_envs: int) -> int:
    """Convert a total-timestep interval to Stable-Baselines callback calls."""
    if vector_timesteps <= 0:
        raise ValueError("callback timestep interval must be positive")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    return max(vector_timesteps // n_envs, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO intersection policy")
    parser.add_argument("--config", default=None, help="Environment YAML configuration")
    parser.add_argument("--config-sha256", default=None)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--env-seed-stride", type=int, default=1_000)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--ttc-threshold", type=float, default=2.0)
    parser.add_argument("--intent-model", default=None)
    parser.add_argument("--intent-neighbors", type=int, default=5)
    parser.add_argument("--intent-history-length", type=int, default=None)
    parser.add_argument("--intent-history-tracking-neighbors", type=int, default=None)
    parser.add_argument("--intent-device", default="cpu")
    parser.add_argument("--intent-model-sha256", default=None)
    parser.add_argument("--eval-seed-offset", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--evaluation-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--risk-fusion", action="store_true")
    parser.add_argument("--fusion-neighbors", type=int, default=14)
    parser.add_argument("--fusion-range-scale", type=float, default=200.0)
    parser.add_argument("--fusion-relative-speed-scale", type=float, default=20.0)
    parser.add_argument("--fusion-ttc-scale", type=float, default=10.0)
    parser.add_argument("--fusion-cpa-horizon", type=float, default=5.0)
    parser.add_argument("--fusion-cpa-distance-scale", type=float, default=20.0)
    parser.add_argument("--target-speed-observation", action="store_true")
    parser.add_argument("--target-speed-scale", type=float, default=9.0)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--output", default="models/ppo_intersection")
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
    if args.eval_seed_offset == 0:
        parser.error("--eval-seed-offset must separate training and internal evaluation seeds")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")

    try:
        train_seed_offsets = training_seed_offsets(args.n_envs, args.env_seed_stride)
        evaluation_callback_frequency = callback_frequency(
            args.evaluation_freq, args.n_envs
        )
        checkpoint_callback_frequency = callback_frequency(
            args.checkpoint_freq, args.n_envs
        )
    except ValueError as error:
        parser.error(str(error))
    if args.eval_seed_offset in train_seed_offsets:
        parser.error("--eval-seed-offset overlaps a training environment seed offset")

    intent_model_sha256 = None
    intent_history_length = None
    intent_history_tracking_neighbors = None
    if args.intent_model is not None:
        intent_model_sha256 = file_sha256(args.intent_model)
        if (
            args.intent_model_sha256 is not None
            and intent_model_sha256.lower() != args.intent_model_sha256.lower()
        ):
            raise ValueError("Intent checkpoint fingerprint does not match the requested model")
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
    config_sha256 = file_sha256(args.config) if args.config is not None else None
    if (
        args.config_sha256 is not None
        and config_sha256 is not None
        and config_sha256.lower() != args.config_sha256.lower()
    ):
        raise ValueError("Configuration fingerprint does not match --config-sha256")

    output = Path(args.output)
    model_path = output if str(output).endswith(".zip") else Path(f"{output}.zip")
    summary_output = Path(args.summary_output) if args.summary_output is not None else None
    if args.refuse_overwrite and model_path.exists():
        raise FileExistsError(f"Refusing to overwrite PPO checkpoint: {model_path}")
    if args.refuse_overwrite and summary_output is not None and summary_output.exists():
        raise FileExistsError(f"Refusing to overwrite training summary: {summary_output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs") / output.stem
    log_dir.mkdir(parents=True, exist_ok=True)

    def build_env(seed_offset: int = 0):
        env = make_intersection_env(
            config_path=args.config,
            seed=args.seed + seed_offset,
            safety_shield=args.safety_shield,
            ttc_threshold=args.ttc_threshold,
            risk_fusion=args.risk_fusion,
            fusion_neighbors=args.fusion_neighbors,
            fusion_range_scale=args.fusion_range_scale,
            fusion_relative_speed_scale=args.fusion_relative_speed_scale,
            fusion_ttc_scale=args.fusion_ttc_scale,
            fusion_cpa_horizon=args.fusion_cpa_horizon,
            fusion_cpa_distance_scale=args.fusion_cpa_distance_scale,
            target_speed_observation=args.target_speed_observation,
            target_speed_scale=args.target_speed_scale,
            intent_model=args.intent_model,
            intent_neighbors=args.intent_neighbors,
            intent_history_length=intent_history_length,
            intent_history_tracking_neighbors=intent_history_tracking_neighbors,
            intent_device=args.intent_device,
            intent_model_sha256=intent_model_sha256,
        )
        return Monitor(env)

    train_env = DummyVecEnv(
        [lambda offset=offset: build_env(offset) for offset in train_seed_offsets]
    )
    eval_env = DummyVecEnv([lambda: build_env(args.eval_seed_offset)])
    observation_shape = list(train_env.observation_space.shape)
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
        CheckpointCallback(
            save_freq=checkpoint_callback_frequency,
            save_path=str(log_dir / "checkpoints"),
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best"),
            log_path=str(log_dir / "evaluation"),
            eval_freq=evaluation_callback_frequency,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        ),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(output)
    if summary_output is not None:
        summary = {
            "algorithm": "PPO",
            "config_path": str(Path(args.config)) if args.config is not None else None,
            "config_sha256": config_sha256,
            "training_seed": args.seed,
            "internal_evaluation_seed_offset": args.eval_seed_offset,
            "timesteps_requested": args.timesteps,
            "timesteps_collected": int(model.num_timesteps),
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_envs": args.n_envs,
            "env_seed_stride": args.env_seed_stride,
            "training_seed_offsets": train_seed_offsets,
            "training_initial_seeds": [args.seed + offset for offset in train_seed_offsets],
            "rollout_size": args.n_steps * args.n_envs,
            "eval_episodes": args.eval_episodes,
            "evaluation_freq_timesteps": args.evaluation_freq,
            "evaluation_callback_frequency": evaluation_callback_frequency,
            "checkpoint_freq_timesteps": args.checkpoint_freq,
            "checkpoint_callback_frequency": checkpoint_callback_frequency,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "policy_network": [256, 256],
            "observation_shape": observation_shape,
            "safety_shield": args.safety_shield,
            "ttc_threshold": args.ttc_threshold,
            "risk_fusion": args.risk_fusion,
            "target_speed_observation": args.target_speed_observation,
            "target_speed_scale": (
                args.target_speed_scale if args.target_speed_observation else None
            ),
            "collision_first_reward": bool(
                load_config(args.config).get("reward_wrapper", {}).get("collision_first", False)
            ),
            "fusion_neighbors": args.fusion_neighbors if args.risk_fusion else 0,
            "fusion_features_per_neighbor": 5 if args.risk_fusion else 0,
            "fusion_range_scale": args.fusion_range_scale if args.risk_fusion else None,
            "fusion_relative_speed_scale": (
                args.fusion_relative_speed_scale if args.risk_fusion else None
            ),
            "fusion_ttc_scale": args.fusion_ttc_scale if args.risk_fusion else None,
            "fusion_cpa_horizon": (
                args.fusion_cpa_horizon if args.risk_fusion else None
            ),
            "fusion_cpa_distance_scale": (
                args.fusion_cpa_distance_scale if args.risk_fusion else None
            ),
            "intent_model_path": (
                str(Path(args.intent_model)) if args.intent_model is not None else None
            ),
            "intent_model_sha256": intent_model_sha256,
            "intent_neighbors": args.intent_neighbors if args.intent_model is not None else 0,
            "intent_history_length": intent_history_length,
            "intent_history_tracking_neighbors": intent_history_tracking_neighbors,
            "intent_device": args.intent_device if args.intent_model is not None else None,
            "model_path": str(model_path),
            "model_sha256": file_sha256(model_path),
        }
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved training summary to {summary_output}")
    train_env.close()
    eval_env.close()
    print(f"Saved PPO policy to {model_path}")


if __name__ == "__main__":
    main()
