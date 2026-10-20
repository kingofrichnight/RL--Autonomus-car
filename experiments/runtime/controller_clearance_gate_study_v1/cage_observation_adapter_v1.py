"""Apply frozen CAGE to the exact existing Sector163 observation."""

import numpy as np

from cage_pure_v1 import select


def select_from_observation(label, partition, observation):
    if type(label) is not int or label not in (0, 1, 2):
        raise ValueError("Saved QR-DQN label differs")
    if (not isinstance(partition, np.ndarray) or partition.shape != (3,)
            or partition.dtype != np.uint8):
        raise ValueError("Pre-action command partition differs")
    if (not isinstance(observation, np.ndarray) or observation.shape != (163,)
            or observation.dtype != np.float32 or not np.isfinite(observation).all()):
        raise ValueError("Sector163 observation differs")
    forecast = observation[106:115].reshape(3, 3)
    rows = tuple(tuple(float(x) for x in row) for row in forecast)
    selected = select(label, tuple(int(x) for x in partition), rows)
    return selected, selected != label, rows
