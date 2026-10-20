"""Causal training-only command partition; preserves the Sector observation.

Native factory imports are deferred. Importing this module does not create an
environment, run a forecast, or load a policy. No partition is a policy input.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from action_entropy_math_v1 import partition_from_state

PROTOCOL = "predictive_post_spawn_sector163_command_class_entropy_v1"


class ActionPartitionMetadata(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if (env.observation_space.shape != (163,)
                or env.observation_space.dtype != np.float32
                or not isinstance(env.action_space, gym.spaces.Discrete)
                or env.action_space.n != 3):
            raise ValueError("Requires the original float32 Sector163 three-label stack")

    def action_entropy_partition(self):
        """Read before env.step; never use the current env from PPO minibatches."""
        ego = self.unwrapped.vehicle
        return partition_from_state(ego.speed, ego.speed_index, ego.target_speed,
                                    ego.target_speeds).copy()


def wrap_base_env(inner):
    from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
    from scripts.sector_observation_v1 import SectorPredictiveObservation
    return ActionPartitionMetadata(
        SectorPredictiveObservation(SynchronizedPredictiveObservation(inner)))
