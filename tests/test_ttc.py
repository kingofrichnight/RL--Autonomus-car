import math

import pytest

from safeintent_rl.safety.ttc import pairwise_ttc


def test_ttc_for_closing_vehicle() -> None:
    value = pairwise_ttc([0, 0], [10, 0], [20, 0], [5, 0])
    assert value == pytest.approx(4.0)


def test_ttc_is_infinite_when_separating() -> None:
    value = pairwise_ttc([0, 0], [5, 0], [20, 0], [10, 0])
    assert math.isinf(value)


def test_ttc_is_zero_for_contact() -> None:
    assert pairwise_ttc([0, 0], [0, 0], [0, 0], [0, 0]) == 0.0

