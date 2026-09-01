import pytest

from safeintent_rl.envs.driver_behavior import PROFILES


def test_profiles_have_expected_order_and_names() -> None:
    assert [profile.name for profile in PROFILES] == ["cautious", "normal", "aggressive"]


def test_aggressive_profile_is_faster_and_accepts_smaller_gaps() -> None:
    cautious, _, aggressive = PROFILES
    assert aggressive.desired_speed_scale > cautious.desired_speed_scale
    assert aggressive.following_distance_scale < cautious.following_distance_scale
    assert aggressive.yield_probability < cautious.yield_probability


def test_profile_probabilities_are_valid() -> None:
    probabilities = (0.30, 0.45, 0.25)
    assert sum(probabilities) == pytest.approx(1.0)

