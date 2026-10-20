"""Deterministic saved QR-DQN proxy with one inference-only CAGE decision."""

import hashlib

import numpy as np

from cage_observation_adapter_v1 import select_from_observation


class CagePolicyProxy:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.decisions = []

    def reset_episode(self):
        self.decisions = []

    def predict(self, observation, *, deterministic=True):
        if deterministic is not True:
            raise ValueError("Only frozen deterministic inference is registered")
        original, state = self.model.predict(observation, deterministic=True)
        scalar = np.asarray(original)
        if scalar.shape != () or not np.issubdtype(scalar.dtype, np.integer):
            raise ValueError("QR-DQN action must be an integer scalar")
        baseline = int(scalar)
        partition = self.env.action_entropy_partition()
        selected, changed, _ = select_from_observation(baseline, partition, observation)
        raw = observation.tobytes(order="C")
        self.decisions.append(dict(index=len(self.decisions) + 1,
            observation_sha256=hashlib.sha256(raw).hexdigest(),
            observation_float32_bits=raw.hex(),
            partition=[int(x) for x in partition],
            forecast_float32_bits=observation[106:115].tobytes(order="C").hex(),
            baseline=baseline, selected=selected, changed=changed))
        if changed:
            return np.asarray(selected, dtype=scalar.dtype), state
        return original, state
