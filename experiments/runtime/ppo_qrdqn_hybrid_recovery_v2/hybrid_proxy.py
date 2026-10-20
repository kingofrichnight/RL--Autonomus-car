"""Two frozen experts, one action and an exact decision receipt per step."""
import hashlib
import time
from hybrid_rule_v2 import select


class HybridProxy:
    def __init__(self, ppo, qr, env, weight):
        self.ppo, self.qr, self.env, self.weight = ppo, qr, env, weight
        self.total_inference_seconds = 0.0
        self.reset_episode()

    def reset_episode(self):
        self.decisions = []

    def predict(self, observation, deterministic=True):
        import numpy as np
        import torch
        if deterministic is not True or observation.dtype != np.dtype('float32') or observation.shape != (163,) or not np.isfinite(observation).all():
            raise ValueError('Hybrid observation/inference contract')
        started = time.perf_counter()
        with torch.inference_mode():
            tensor, _ = self.ppo.policy.obs_to_tensor(observation)
            p = self.ppo.policy.get_distribution(tensor).distribution.probs
            qtensor, _ = self.qr.policy.obs_to_tensor(observation)
            q = self.qr.policy.quantile_net(qtensor)
            if p.shape != (1, 3) or q.shape != (1, 200, 3):
                raise ValueError('Expert dimensions')
            baseline = int(q.mean(dim=1).argmax(dim=1).item())
            p = p[0].cpu().numpy().astype('<f4', copy=False)
            q = np.ascontiguousarray(q[0].cpu().numpy().T, dtype='<f4')
        partition = self.env.action_entropy_partition()
        if partition.shape != (3,) or partition.dtype != np.dtype('uint8'):
            raise ValueError('Command partition contract')
        part = [int(a) for a in partition]
        result = select(p.astype(float).tolist(), q.astype(float).tolist(), part,
                        observation[106:115].reshape(3, 3).astype(float).tolist(), self.weight, baseline)
        self.total_inference_seconds += time.perf_counter() - started
        raw = observation.astype('<f4', copy=False).tobytes()
        self.decisions.append(dict(index=len(self.decisions) + 1,
                                   observation_sha256=hashlib.sha256(raw).hexdigest(), observation_bits=raw.hex(),
                                   probability_bits=p.tobytes().hex(), quantile_bits=q.tobytes().hex(),
                                   partition=part, baseline=baseline, ppo_action=result['ppo_action'],
                                   selected=result['selected'], changed=result['changed']))
        return np.asarray(result['selected'], dtype=np.int64), None
