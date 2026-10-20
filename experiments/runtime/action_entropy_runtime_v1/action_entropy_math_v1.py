"""Pure, frozen-grid command partition and categorical entropy arithmetic.

No simulator, policy, PPO, optimizer, or environment is imported here.
Raw native scalar precision is deliberately retained for partition construction.
"""

from __future__ import annotations

import numpy as np
import torch


GRID = (0.0, 4.5, 9.0)
INTEGER_TORCH_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _native_number(value: object, name: str) -> np.float64:
    """Accept native float64/Python floats and integer scalars, excluding bool.

    Native subtraction by the float64 target-grid endpoint promotes integer
    speed to float64. The same promotion is explicit here; integers too large
    to have a finite float64 representation are rejected, without range caps.
    """
    if isinstance(value, (bool, np.bool_)) or not (
        type(value) in (float, np.float64) or isinstance(value, (int, np.integer))
    ):
        raise TypeError(f"{name} must be a native real numeric scalar, excluding bool")
    try:
        result = np.float64(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must have a finite float64 representation") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def partition_from_state(speed, speed_index, target_speed, target_speeds) -> np.ndarray:
    """Return SLOWER/IDLE/FASTER resultant target indices, uint8 shape (3,).

    Speed may be any native finite real scalar; no artificial cap is imposed.
    The native quantizer clips the index. Integer scalar indices include NumPy
    integer types but exclude bool. The only supported target grid is 0/4.5/9.
    Exact target/index consistency is mandatory. No native state is mutated.
    """
    raw_speed = _native_number(speed, "speed")
    raw_target = _native_number(target_speed, "target_speed")
    if isinstance(speed_index, (bool, np.bool_)) or not isinstance(speed_index, (int, np.integer)):
        raise TypeError("speed_index must be an integer scalar, excluding bool")
    index = int(speed_index)
    if index < 0 or index > 2:
        raise ValueError("speed_index must lie in 0..2")
    if isinstance(target_speeds, np.ndarray):
        if target_speeds.dtype != np.dtype(np.float64) or target_speeds.shape != (3,):
            raise TypeError("target_speeds must be a float64 array of shape (3,)")
        grid = target_speeds
    elif type(target_speeds) in (tuple, list) and len(target_speeds) == 3:
        grid = np.asarray([_native_number(x, "target_speeds item") for x in target_speeds], dtype=np.float64)
    else:
        raise TypeError("target_speeds must contain exactly three float64 scalars")
    if not np.array_equal(grid, np.asarray(GRID, dtype=np.float64)):
        raise ValueError("target_speeds differs from the frozen 0/4.5/9 grid")
    if raw_target != grid[index]:
        raise ValueError("target_speed and speed_index are inconsistent")
    # Keep the installed native arithmetic ordering; do not simplify speed/4.5.
    q = int(np.clip(np.round((raw_speed - grid[0]) / (grid[-1] - grid[0]) * (len(grid) - 1)), 0, len(grid) - 1))
    return np.asarray((max(q - 1, 0), index, min(q + 1, 2)), dtype=np.uint8)


def entropy_components(logits: torch.Tensor, classes: torch.Tensor):
    """Return (label, class, within-class) entropy tensors of shape (batch,).

    Inputs are finite raw float32/float64 logits, shape (batch,3), and same-device
    integer IDs in 0..2 with matching shape. IDs denote groups, not ordinal ranks.
    Feed original raw actor logits rather than already-normalized distribution
    logits to avoid an extra normalization relative to a preexisting Categorical.
    Empty groups are excluded without undefined all-minus-infinity derivatives.
    All-distinct rows return stock Categorical entropy exactly, including gradient.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(classes, torch.Tensor):
        raise TypeError("logits and classes must be torch tensors")
    if logits.dtype not in (torch.float32, torch.float64):
        raise TypeError("logits must use float32 or float64")
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != 3:
        raise ValueError("logits must have nonempty shape (batch,3)")
    if classes.shape != logits.shape or classes.device != logits.device:
        raise ValueError("classes must match logits shape and device")
    if classes.dtype not in INTEGER_TORCH_DTYPES:
        raise TypeError("classes must use an integer dtype, excluding bool")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if not bool(((classes >= 0) & (classes <= 2)).all()):
        raise ValueError("class IDs must lie in 0..2")

    label_entropy = torch.distributions.Categorical(logits=logits).entropy()
    normalizer = torch.logsumexp(logits, dim=-1)
    terms = []
    for group in range(3):
        members = classes == group
        present = members.any(dim=-1)
        masked = torch.where(members, logits, torch.full_like(logits, -torch.inf))
        # Empty rows get finite dummy inputs; their entropy term is then zero.
        safe_masked = torch.where(present[:, None], masked, torch.zeros_like(logits))
        log_mass = torch.logsumexp(safe_masked, dim=-1) - normalizer
        log_mass = torch.where(present, log_mass, torch.zeros_like(log_mass))
        safe_log_mass = torch.clamp(log_mass, min=torch.finfo(logits.dtype).min)
        term = -torch.exp(log_mass) * safe_log_mass
        terms.append(torch.where(present, term, torch.zeros_like(term)))
    grouped_entropy = torch.stack(terms, dim=-1).sum(dim=-1)
    all_distinct = ((classes[:, 0] != classes[:, 1]) &
                    (classes[:, 0] != classes[:, 2]) &
                    (classes[:, 1] != classes[:, 2]))
    class_entropy = torch.where(all_distinct, label_entropy, grouped_entropy)
    within_entropy = label_entropy - class_entropy
    return label_entropy, class_entropy, within_entropy
