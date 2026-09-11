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
from safeintent_rl.safety.shield import CPAAccelerationShield, TTCSafetyShield
from safeintent_rl.sensors import EgoTargetSpeedObservation, KinematicRiskFusionWrapper
from safeintent_rl.sensors.predictive import PredictiveSafetyObservation

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
    driver_probabilities: tuple[float, float, float] | None = None,
    safety_shield: bool = False,
    ttc_threshold: float = 2.0,
    cpa_safety_shield: bool = False,
    cpa_time_threshold: float = 2.0,
    cpa_distance_threshold: float = 3.0,
    cpa_horizon: float = 3.0,
    cpa_max_range: float = 60.0,
    cpa_override_action: str = "IDLE",
    risk_fusion: bool = False,
    fusion_neighbors: int = 14,
    fusion_range_scale: float = 200.0,
    fusion_relative_speed_scale: float = 20.0,
    fusion_ttc_scale: float = 10.0,
    fusion_cpa_horizon: float = 5.0,
    fusion_cpa_distance_scale: float = 20.0,
    target_speed_observation: bool = False,
    target_speed_scale: float = 9.0,
    intent_model: str | Path | None = None,
    intent_neighbors: int = 5,
    intent_device: str = "cpu",
    intent_model_sha256: str | None = None,
    intent_diagnostics: bool = False,
    intent_shadow_history_neighbors: int | None = None,
    intent_history_length: int | None = None,
    intent_history_tracking_neighbors: int | None = None,
) -> gym.Env:
    """Create the project's intersection environment with optional research wrappers."""
    if safety_shield and cpa_safety_shield:
        raise ValueError("TTC and CPA safety shields cannot be combined")

    loaded = load_config(config_path)
    reward_wrapper_config = loaded.pop("reward_wrapper", None)
    predictive_config = loaded.pop("predictive_safety_observation", None)
    if predictive_config is not None and (
        risk_fusion or intent_model is not None or target_speed_observation
        or safety_shield or cpa_safety_shield
    ):
        raise ValueError("predictive v1 must be evaluated without other feature/shield options")
    preferred_id, env_config = split_env_config(loaded)
    env_id = _available_intersection_id(preferred_id)
    env = gym.make(env_id, render_mode=render_mode, config=env_config)

    if driver_behaviors:
        env = (DriverBehaviorWrapper(env) if driver_probabilities is None
               else DriverBehaviorWrapper(env, probabilities=driver_probabilities))
    elif driver_probabilities is not None:
        env.close()
        raise ValueError("driver_probabilities requires driver_behaviors")
    if reward_wrapper_config is not None:
        wrapper_config = dict(reward_wrapper_config)
        wrapper_type = wrapper_config.pop("type", "RouteProgressReward")
        if wrapper_type != "RouteProgressReward":
            raise ValueError(f"Unsupported reward wrapper: {wrapper_type}")
        env = RouteProgressRewardWrapper(env, **wrapper_config)
    if predictive_config is not None:
        env = PredictiveSafetyObservation(env, **predictive_config)
    if risk_fusion:
        env = KinematicRiskFusionWrapper(
            env,
            max_neighbors=fusion_neighbors,
            range_scale=fusion_range_scale,
            relative_speed_scale=fusion_relative_speed_scale,
            ttc_scale=fusion_ttc_scale,
            cpa_horizon=fusion_cpa_horizon,
            cpa_distance_scale=fusion_cpa_distance_scale,
        )
    if intent_model is not None:
        env = IntentObservationWrapper(
            env,
            checkpoint_path=intent_model,
            max_neighbors=intent_neighbors,
            history_length=intent_history_length,
            device=intent_device,
            expected_checkpoint_sha256=intent_model_sha256,
            collect_diagnostics=intent_diagnostics,
            shadow_history_neighbors=intent_shadow_history_neighbors,
            history_tracking_neighbors=intent_history_tracking_neighbors,
        )
    if target_speed_observation:
        env = EgoTargetSpeedObservation(env, scale=target_speed_scale)
    if safety_shield:
        env = TTCSafetyShield(env, ttc_threshold=ttc_threshold)
    if cpa_safety_shield:
        env = CPAAccelerationShield(
            env,
            time_threshold=cpa_time_threshold,
            distance_threshold=cpa_distance_threshold,
            horizon=cpa_horizon,
            max_range=cpa_max_range,
            override_action=cpa_override_action,
        )
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env_factory(**kwargs: Any) -> Callable[[], gym.Env]:
    """Return a picklable-style zero-argument factory for vectorized training."""

    def _factory() -> gym.Env:
        return make_intersection_env(**kwargs)

    return _factory
