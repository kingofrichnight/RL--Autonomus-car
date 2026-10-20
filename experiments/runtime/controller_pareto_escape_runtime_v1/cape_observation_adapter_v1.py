"""Map the frozen Sector163 observation and exact command classes to CAPE.

This module makes no forecast or environment call. It consumes the same
float32 observation passed to the saved deterministic QR-DQN policy.
"""

from __future__ import annotations

import numpy as np

from cape_pure_v1 import select


FORECAST_START = 106
FORECAST_STOP = 115


def select_from_observation(label, partition, observation):
    if type(label) is not int or label not in (0, 1, 2):
        raise ValueError("QR-DQN label differs")
    if not isinstance(partition, np.ndarray) or partition.shape != (3,) or partition.dtype != np.uint8:
        raise ValueError("Pre-action command partition differs")
    if not isinstance(observation, np.ndarray) or observation.shape != (163,) or observation.dtype != np.float32:
        raise ValueError("Sector163 observation differs")
    if not np.isfinite(observation).all():
        raise ValueError("Nonfinite observation")
    # The synchronized predictive observation appends target speed at 105,
    # then SLOWER/IDLE/FASTER forecast triplets at 106:115. Sector appends 48.
    forecast = observation[FORECAST_START:FORECAST_STOP].reshape(3, 3)
    classes = tuple(int(x) for x in partition)
    rows = tuple(tuple(float(x) for x in row) for row in forecast)
    selected = select(label, classes, rows)
    return selected, selected != label, rows
