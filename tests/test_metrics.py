import pytest

from safeintent_rl.evaluation.metrics import EpisodeMetrics, detect_success, summarize_episodes


def test_episode_summary() -> None:
    episodes = [
        EpisodeMetrics(10.0, 20, True, False, 4.0, 3.0, 1, 0),
        EpisodeMetrics(4.0, 30, False, True, 6.0, 1.0, 3, 2),
    ]
    summary = summarize_episodes(episodes)
    assert summary["mean_reward"] == pytest.approx(7.0)
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["collision_rate"] == pytest.approx(0.5)
    assert summary["mean_min_ttc"] == pytest.approx(2.0)


class _IntersectionEnv:
    unwrapped: "_IntersectionEnv"
    vehicle: object

    def __init__(self, arrived: bool) -> None:
        self.unwrapped = self
        self.vehicle = object()
        self.arrived = arrived

    def has_arrived(self, vehicle: object) -> bool:
        assert vehicle is self.vehicle
        return self.arrived


def test_detect_success_uses_highway_env_arrival_method() -> None:
    assert detect_success(_IntersectionEnv(True), {}) is True
    assert detect_success(_IntersectionEnv(False), {}) is False


def test_detect_success_prefers_explicit_info() -> None:
    assert detect_success(_IntersectionEnv(False), {"is_success": True}) is True
