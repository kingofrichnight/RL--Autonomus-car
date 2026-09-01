import pytest

from safeintent_rl.evaluation.metrics import EpisodeMetrics, summarize_episodes


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

