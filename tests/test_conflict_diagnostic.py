import math
from types import SimpleNamespace

import pytest

from safeintent_rl.safety.conflict import (
    minimum_closest_approach,
    nearby_closest_approaches,
    pairwise_closest_approach,
)
from scripts.diagnose_conflicts import (
    DecisionRecord,
    EpisodeTrace,
    build_conflict_profiles,
)


def test_pairwise_closest_approach_detects_crossing_paths() -> None:
    result = pairwise_closest_approach(
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, -1.0],
        [0.0, 1.0],
        horizon=2.0,
    )
    assert result.time == pytest.approx(1.0)
    assert result.distance == pytest.approx(0.0)
    assert result.current_distance == pytest.approx(math.sqrt(2.0))


def test_pairwise_closest_approach_excludes_past_or_beyond_horizon() -> None:
    past = pairwise_closest_approach([0, 0], [1, 0], [-2, 0], [0, 0], horizon=3)
    future = pairwise_closest_approach([0, 0], [1, 0], [5, 0], [0, 0], horizon=3)
    assert math.isinf(past.time) and math.isinf(past.distance)
    assert math.isinf(future.time) and math.isinf(future.distance)


def test_minimum_closest_approach_uses_smallest_predicted_miss_distance() -> None:
    ego = SimpleNamespace(position=[0.0, 0.0], velocity=[1.0, 0.0])
    crossing = SimpleNamespace(position=[1.0, -1.0], velocity=[0.0, 1.0])
    passing = SimpleNamespace(position=[1.0, 3.0], velocity=[0.0, 1.0])
    result = minimum_closest_approach(ego, [ego, passing, crossing], horizon=2.0)
    assert result.time == pytest.approx(1.0)
    assert result.distance == pytest.approx(0.0)


def test_nearby_closest_approaches_retains_earlier_wider_conflict() -> None:
    ego = SimpleNamespace(position=[0.0, 0.0], velocity=[1.0, 0.0])
    later_direct = SimpleNamespace(position=[2.0, -2.0], velocity=[0.0, 1.0])
    earlier_wide = SimpleNamespace(position=[0.5, 1.0], velocity=[0.0, -1.0])
    results = nearby_closest_approaches(
        ego,
        [ego, later_direct, earlier_wide],
        horizon=3.0,
    )
    assert len(results) == 2
    assert any(result.time < 1.0 and result.distance > 0.0 for result in results)


def test_conflict_profiles_separate_collision_coverage_from_success_burden() -> None:
    safe_faster = DecisionRecord("FASTER", 1.0, ((1.0, 4.0),))
    risky_faster = DecisionRecord("FASTER", 1.0, ((0.5, 1.0),))
    risky_idle = DecisionRecord("IDLE", 1.0, ((0.5, 1.0),))
    traces = [
        EpisodeTrace(True, False, (safe_faster, risky_idle)),
        EpisodeTrace(False, True, (safe_faster, risky_faster)),
    ]
    profiles = build_conflict_profiles(
        traces,
        policy_frequency=1.0,
        legacy_ttc_threshold=2.0,
    )
    candidate = next(
        item
        for item in profiles["cpa_grid"]
        if item["action_scope"] == "FASTER_ONLY"
        and item["cpa_time_threshold"] == 0.5
        and item["cpa_distance_threshold"] == 1.5
    )
    aggregate = candidate["windows"]["all"]
    assert aggregate["trigger_decisions"] == 1
    assert aggregate["success_episodes_triggered"] == 0
    assert aggregate["collision_episodes_triggered"] == 1
    assert aggregate["success_episode_burden"] == 0.0
    assert aggregate["collision_episode_coverage"] == 1.0

    legacy = profiles["legacy_radial_ttc_rule"]["windows"]["all"]
    assert legacy["trigger_decisions"] == 4
    assert legacy["success_episode_burden"] == 1.0
