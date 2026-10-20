"""Source-only import; shared deterministic initialization byte fingerprints.

Function calls inspect an already constructed CPU policy/RNG. They never draw
randomness, construct a model, perform inference, or import a simulator.
"""
import hashlib
import json
import random


def initial_policy_fingerprint(model):
    """Exactly the historical sorted, length-delimited policy tensor format."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite policy tensor")
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                              separators=(",", ":")).encode()
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return {"sha256": digest.hexdigest(),
            "parameter_count": sum(p.numel() for p in model.policy.parameters())}


def _framed_hash(metadata, payload):
    encoded = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256()
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)
    return digest.hexdigest()


def initial_rng_fingerprint():
    """Hash complete Python/global NumPy/Torch CPU states without RNG draws.

    Python: compact JSON of random.getstate() (version, integer state, cached
    Gaussian). NumPy: framed JSON (algorithm, key dtype/shape, position,
    Gaussian-cache flag/value) plus C-order key bytes. Torch CPU: framed JSON
    (dtype/shape) plus its complete contiguous get_rng_state() byte payload.
    These formats intentionally do not include per-space or CUDA generators.
    """
    import numpy as np
    import torch
    python_state = json.dumps(random.getstate(), separators=(",", ":"),
                              allow_nan=False).encode()
    numpy_state = np.random.get_state()
    keys = numpy_state[1]
    numpy_metadata = [numpy_state[0], str(keys.dtype), list(keys.shape),
                      int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])]
    torch_state = torch.get_rng_state().cpu().contiguous()
    return {
        "python_random_sha256": hashlib.sha256(python_state).hexdigest(),
        "numpy_global_sha256": _framed_hash(numpy_metadata, keys.tobytes(order="C")),
        "torch_cpu_sha256": _framed_hash(
            [str(torch_state.dtype), list(torch_state.shape)],
            torch_state.numpy().tobytes(order="C")),
    }
