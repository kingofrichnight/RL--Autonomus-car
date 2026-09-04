from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401  Registers HighwayEnv environments with Gymnasium.

from safeintent_rl.config import load_config, split_env_config
from safeintent_rl.envs.driver_behavior import DriverBehaviorWrapper
from safeintent_rl.envs.reward import RouteProgressRewardWrapper
from safeintent_rl.intent.wrapper import IntentObservationWrapper
from safeintent_rl.safety.shield import TTCSafetyShield

FALLBACK_INTERSECTION_IDS = ("intersection-v2", "intersection-v1", "intersection-v0")


def _available_intersection_id(preferred: str) -> str:
    candidates = (preferred,) + tuple(x for x in FALLBACK_INTERSECTION_IDS if x != preferred)
    for candidate in candidates:
        if candidate in gym.registry:
            return candidate
    raise gym.error.Error(
        "No HighwayEnv intersection environment is registered. Install/upgrade highway-env."
    )


def make_intersection_env(
    config_path: str | Path | None = None,
    *,
    render_mode: str | None = None,
    seed: int | None = None,
    driver_behaviors: bool = True,
    safety_shield: bool = False,
    ttc_threshold: float = 2.0,
    intent_model: str | Path | None = None,
    intent_neighbors: int = 5,
    intent_device: str = "cpu",
    intent_model_sha256: str | None = None,
) -> gym.Env:
    """Create the project's intersection environment with optional research wrappers."""
    loaded = load_config(config_path)
    reward_wrapper_config = loaded.pop("reward_wrapper", None)
    preferred_id, env_config = split_env_config(loaded)
    env_id = _available_intersection_id(preferred_id)
    env = gym.make(env_id, render_mode=render_mode, config=env_config)

    if driver_behaviors:
        env = DriverBehaviorWrapper(env)
    if reward_wrapper_config is not None:
        wrapper_config = dict(reward_wrapper_config)
        wrapper_type = wrapper_config.pop("type", "RouteProgressReward")
        if wrapper_type != "RouteProgressReward":
            raise ValueError(f"Unsupported reward wrapper: {wrapper_type}")
        env = RouteProgressRewardWrapper(env, **wrapper_config)
    if intent_model is not None:
        env = IntentObservationWrapper(
            env,
            checkpoint_path=intent_model,
            max_neighbors=intent_neighbors,
            device=intent_device,
            expected_checkpoint_sha256=intent_model_sha256,
        )
    if safety_shield:
        env = TTCSafetyShield(env, ttc_threshold=ttc_threshold)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env_factory(**kwargs: Any) -> Callable[[], gym.Env]:
    """Return a picklable-style zero-argument factory for vectorized training."""

    def _factory() -> gym.Env:
        return make_intersection_env(**kwargs)

    return _factory
