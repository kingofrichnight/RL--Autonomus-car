"""One frozen positive-clearance successor to the CAPE alias escape rule."""

from __future__ import annotations

import math
from collections.abc import Sequence


def select(label: int, classes: Sequence[int], forecast: Sequence[Sequence[float]]) -> int:
    if type(label) is not int or label not in (0, 1, 2):
        raise ValueError("label must be an integer in 0..2")
    if len(classes) != 3 or any(type(value) is not int or value not in (0, 1, 2)
                                for value in classes):
        raise ValueError("classes must be three integer identifiers in 0..2")
    if len(forecast) != 3:
        raise ValueError("forecast must have three action rows")
    values = []
    for row in forecast:
        if len(row) != 3:
            raise ValueError("forecast row must have three values")
        triplet = []
        for value in row:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError("forecast values must be finite numeric scalars")
            triplet.append(float(value))
        if not (-1 <= triplet[0] <= 1 and 0 <= triplet[1] <= 1
                and 0 <= triplet[2] <= 1):
            raise ValueError("forecast values outside observation bounds")
        values.append(triplet)
    if label != 0 or classes[0] != classes[1] or classes[2] == classes[0]:
        return label
    slower, faster = values[0], values[2]
    if (faster[0] > 0 and faster[0] >= slower[0]
            and faster[1] >= slower[1] and faster[2] > slower[2]):
        return 2
    return label
