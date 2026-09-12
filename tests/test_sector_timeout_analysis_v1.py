import copy
import gzip
import json
import math
import sys

import pytest

from scripts import analyze_sector_timeouts_v1 as analysis


def synthetic_frame(index, target, progress, arm="sector", speed=2.0):
    now = index / 5
    raw = [[1.0, progress * 4, 0.0, speed, 0.0, 1.0, 0.0],
           [1.0, 10.0, 20.0, 0.0, 0.0, 1.0, 0.0]]
    normalized = [[row[0], row[1] / 200, row[2] / 200, row[3] / 80,
                   row[4] / 80, row[5], row[6]] for row in raw]
    normalized += [[0.0] * 7 for _ in range(13)]
    actors = [{"birth_id": 1, "position": [progress * 4 + 10, 20.0],
               "velocity": [speed, 0.0], "heading": 0.0, "length": 5.0, "width": 2.0}]
    native = {
        "kind": "native", "time": now, "raw_rows": raw, "normalized_rows": normalized,
        "clipped_mask": [[False] * 7 for _ in raw],
        "boundary_mask": [[abs(value) == 1 for value in row] for row in normalized],
        "normalization_ranges": {"x": [-200, 200], "y": [-200, 200],
                                 "vx": [-80, 80], "vy": [-80, 80]},
        "selected_birth_ids": [1], "selected_actors": actors,
    }
    forecasts = [[.2, 1.0, .1], [.3, 1.0, .2], [.4, 1.0, .3]]
    forecast = {"kind": "forecast", "time": now, "values": forecasts,
                "selected_birth_ids": [1], "selected_actors": copy.deepcopy(actors)}
    observation = [value for row in normalized for value in row]
    observation += [target / 9] + [value for row in forecasts for value in row]
    if arm == "sector":
        sectors = [[0.0, 1.0, 0.0] for _ in range(16)]
        sectors[3] = [1.0, math.hypot(10, 20) / (200 * math.sqrt(2)), 0.0]
        observation += [value for row in sectors for value in row]
    elif arm == "padding":
        observation += [0.0] * 48
    return {
        "arm": arm, "policy_observation": observation,
        "native_events": [copy.deepcopy(native) for _ in range(1 if arm == "geometry" else 2)],
        "forecast_events": [forecast], "native_forecast_ids_match": True,
        "state": {"time": now, "speed": speed, "target_speed": target,
                  "position": [progress * 4, 0.0], "heading": 0.0, "crashed": False,
                  "lane_index": ["a", "b", 0], "route_progress": progress,
                  "remaining_route_fraction": 1 - progress},
        "call_order": ["native", "forecast"] + ([] if arm == "geometry" else ["native"]),
    }


def synthetic_episode(post_targets=(0, 0, 4.5, 9, 4.5, 0), actions=(0, 1, 2, 2, 0, 0),
                      initial_target=4.5, arm="sector"):
    count = len(post_targets)
    assert len(actions) == count
    frames = [synthetic_frame(index, target, index / 10, arm)
              for index, target in enumerate((initial_target, *post_targets))]
    decisions, transitions = [], []
    for index, action in enumerate(actions):
        probabilities = [.1, .1, .1]
        probabilities[action] = .8
        decisions.append({
            "kind": "decision", "arm": arm, "seed": 7, "decision_index": index,
            "observation_frame_index": index, "proposed_action": action,
            "action_probabilities": probabilities, "pre_step_ttc": 3.0, "frame": frames[index],
        })
        transitions.append({
            "kind": "transition", "arm": arm, "seed": 7, "decision_index": index,
            "observation_frame_index": index + 1, "executed_action": action, "reward": 1.0,
            "metric_ttc": 3.0, "terminated": False, "truncated": index == count - 1,
        })
    terminal = {"kind": "terminal", "arm": arm, "seed": 7, "observation_frame_index": count,
                "terminated": False, "truncated": True, "frame": frames[-1]}
    metrics = {"reward": float(count), "length": count, "success": False, "collision": False,
               "travel_time": count / 5, "min_ttc": 3.0, "unsafe_ttc_events": 0,
               "safety_interventions": 0}
    return decisions, transitions, terminal, metrics


def test_target_duration_is_attributed_to_post_frames_with_exact_discrete_speed_bins():
    episode = synthetic_episode()
    before = copy.deepcopy(episode)
    result = analysis.summarize_episode(*episode)
    assert result["arm"] == "sector" and result["seed"] == 7
    assert result["outcome"] == "incomplete"
    assert result["metrics"] == episode[-1]
    assert result["elapsed_seconds"] == pytest.approx(1.2)
    assert result["target_seconds"] == pytest.approx({"0": .6, "4.5": .4, "9": .2})
    assert sum(result["target_seconds"].values()) == pytest.approx(result["elapsed_seconds"])
    assert episode == before


def test_idle_actions_do_not_imply_a_zero_target_or_stopped_vehicle():
    episode = synthetic_episode(post_targets=(4.5, 4.5), actions=(1, 1))
    result = analysis.summarize_episode(*episode)
    assert result["actions"]["counts"] == {"SLOWER": 0, "IDLE": 2, "FASTER": 0}
    assert result["target_seconds"] == pytest.approx({"0": 0, "4.5": .4, "9": 0})
    assert result["zero_target"]["total_seconds"] == 0
    assert result["zero_target"]["runs"] == []
    assert result["speed"]["post_transition_samples"]["mean"] == 2.0


def test_zero_target_runs_longest_and_terminal_tail_use_returned_frames():
    result = analysis.summarize_episode(*synthetic_episode())
    zero = result["zero_target"]
    assert zero["total_seconds"] == pytest.approx(.6)
    assert zero["longest_seconds"] == pytest.approx(.4)
    assert zero["tail_seconds"] == pytest.approx(.2)
    assert len(zero["runs"]) == 2
    assert zero["runs"][0]["start_time"] == 0
    assert zero["runs"][0]["end_time"] == pytest.approx(.4)
    assert zero["runs"][0]["duration"] == pytest.approx(.4)
    assert zero["runs"][0]["start_progress"] == 0
    assert zero["runs"][0]["end_progress"] == pytest.approx(.2)
    assert zero["runs"][1]["end_time"] == pytest.approx(1.2)


def test_progress_and_action_distribution_statistics_are_episode_local():
    result = analysis.summarize_episode(*synthetic_episode())
    progress = result["progress"]
    assert progress["start"] == 0
    assert progress["final"] == pytest.approx(.6)
    assert progress["remaining"] == pytest.approx(.4)
    assert progress["net_change"] == pytest.approx(.6)
    assert progress["positive_target_seconds"] == pytest.approx(.6)
    assert progress["positive_target_net_change"] == pytest.approx(.3)
    assert progress["positive_target_progress_per_second"] == pytest.approx(.5)
    assert result["actions"]["counts"] == {"SLOWER": 3, "IDLE": 1, "FASTER": 2}
    assert result["actions"]["entropy"]["mean"] == pytest.approx(
        -.8 * math.log(.8) - .2 * math.log(.1),
    )
    assert result["actions"]["chosen_probability"]["mean"] == pytest.approx(.8)
    assert result["actions"]["top_two_probability_gap"]["mean"] == pytest.approx(.7)


def test_relaunch_is_recorded_at_interval_start_and_all_zero_progress_rate_is_unavailable():
    result = analysis.summarize_episode(*synthetic_episode())
    assert result["zero_target"]["relaunches"] == [
        {"time": .4, "progress": .2, "target_speed": 4.5},
    ]
    all_zero = analysis.summarize_episode(*synthetic_episode(
        post_targets=(0, 0), actions=(1, 1), initial_target=0,
    ))
    assert all_zero["zero_target"]["tail_seconds"] == pytest.approx(.4)
    assert all_zero["zero_target"]["longest_seconds"] == pytest.approx(.4)
    assert all_zero["progress"]["positive_target_seconds"] == 0
    assert all_zero["progress"]["positive_target_progress_per_second"] is None


def test_initial_zero_target_does_not_count_as_an_extra_interval():
    result = analysis.summarize_episode(*synthetic_episode(
        post_targets=(4.5,), actions=(2,), initial_target=0,
    ))
    assert result["target_seconds"] == pytest.approx({"0": 0, "4.5": .2, "9": 0})
    assert result["zero_target"]["runs"] == []
    assert result["zero_target"]["relaunches"] == [
        {"time": 0, "progress": 0, "target_speed": 4.5},
    ]


def test_terminal_speed_sample_is_post_transition_only_not_an_extra_decision():
    episode = synthetic_episode(post_targets=(4.5,), actions=(1,))
    frame = episode[2]["frame"]
    frame["state"]["speed"] = 1.0
    for native in frame["native_events"]:
        native["raw_rows"][0][3] = 1.0
        native["normalized_rows"][0][3] = 1 / 80
    frame["policy_observation"][3] = 1 / 80
    result = analysis.summarize_episode(*episode)
    assert result["speed"]["decision_samples"] == {
        "count": 1, "min": 2, "median": 2, "mean": 2, "max": 2,
    }
    assert result["speed"]["post_transition_samples"] == {
        "count": 1, "min": 1, "median": 1, "mean": 1, "max": 1,
    }


@pytest.mark.parametrize("raw_y,actually_clipped", [(200.0, False), (201.0, True)])
def test_actual_raw_clipping_is_distinct_from_normalized_boundary_occupancy(
        raw_y, actually_clipped):
    episode = synthetic_episode(post_targets=(4.5,), actions=(1,), arm="geometry")
    frame = episode[0][0]["frame"]
    native = frame["native_events"][-1]
    native["raw_rows"][1][2] = raw_y
    native["normalized_rows"][1][2] = 1.0
    native["clipped_mask"][1][2] = actually_clipped
    native["boundary_mask"][1][2] = True
    frame["policy_observation"][9] = 1.0
    result = analysis.summarize_episode(*episode)
    clipped = result["observations"]["clipped_values_by_feature"]
    assert clipped["neighbors"]["y"] == int(actually_clipped)
    assert clipped["neighbors"]["presence"] == clipped["neighbors"]["cos_h"] == 0
    assert all(value == 0 for value in clipped["ego"].values())


def test_forecast_saturation_is_reported_in_normalized_units_without_uncensoring():
    episode = synthetic_episode(post_targets=(4.5,), actions=(1,))
    frame = episode[0][0]["frame"]
    values = [[-1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [.5, .5, .5]]
    frame["forecast_events"][0]["values"] = values
    frame["policy_observation"][106:115] = [value for row in values for value in row]
    result = analysis.summarize_episode(*episode)
    forecasts = result["forecasts"]
    slower = forecasts["by_action"]["SLOWER"]
    idle = forecasts["by_action"]["IDLE"]
    assert slower["minimum_margin_gap"]["mean"] == -1
    assert slower["minimum_margin_gap"]["at_lower_bound"] == 1
    assert slower["first_margin_conflict"]["at_lower_bound"] == 1
    assert idle["minimum_margin_gap"]["mean"] == 1
    assert idle["first_margin_conflict"]["mean"] == 1
    assert idle["first_margin_conflict"]["at_upper_bound"] == 1
    assert forecasts["chosen_action"] == idle
    assert forecasts["by_action"]["FASTER"]["minimum_margin_gap"]["at_upper_bound"] == 0


@pytest.mark.parametrize("mutation", [
    "missing_decision", "missing_transition", "reordered_transition", "wrong_seed",
    "decision_index", "transition_frame_index", "terminal_frame_index", "wrong_executed_action",
    "probability_sum", "probability_argmax", "probability_nonfinite", "boolean_action",
    "early_terminal", "missing_terminal_flag", "terminal_flag_mismatch", "terminal_time",
    "non_grid_target", "wrong_reward", "wrong_length", "wrong_travel_time", "wrong_ttc",
    "wrong_unsafe_count", "wrong_collision", "lying_clip_mask", "lying_boundary_mask",
    "forecast_prefix_mismatch", "native_prefix_mismatch", "target_prefix_mismatch",
])
def test_episode_rejects_bad_action_frame_time_metric_or_observation_joins(mutation):
    episode = synthetic_episode()
    decisions, transitions, terminal, metrics = episode
    if mutation == "missing_decision":
        decisions.pop()
    elif mutation == "missing_transition":
        transitions.pop()
    elif mutation == "reordered_transition":
        transitions[0], transitions[1] = transitions[1], transitions[0]
    elif mutation == "wrong_seed":
        transitions[0]["seed"] = 8
    elif mutation == "decision_index":
        decisions[0]["decision_index"] = 1
    elif mutation == "transition_frame_index":
        transitions[0]["observation_frame_index"] = 0
    elif mutation == "terminal_frame_index":
        terminal["observation_frame_index"] -= 1
    elif mutation == "wrong_executed_action":
        transitions[0]["executed_action"] = 1
    elif mutation == "probability_sum":
        decisions[0]["action_probabilities"] = [1.0, .2, .2]
    elif mutation == "probability_argmax":
        decisions[0]["action_probabilities"] = [.1, .8, .1]
    elif mutation == "probability_nonfinite":
        decisions[0]["action_probabilities"][0] = math.nan
    elif mutation == "boolean_action":
        decisions[0]["proposed_action"] = False
    elif mutation == "early_terminal":
        transitions[0]["terminated"] = True
    elif mutation == "missing_terminal_flag":
        terminal["truncated"] = False
    elif mutation == "terminal_flag_mismatch":
        terminal["terminated"] = True
    elif mutation == "terminal_time":
        terminal["frame"]["state"]["time"] += .1
    elif mutation == "non_grid_target":
        decisions[0]["frame"]["state"]["target_speed"] = 3.0
    elif mutation == "wrong_reward":
        metrics["reward"] += 1
    elif mutation == "wrong_length":
        metrics["length"] += 1
    elif mutation == "wrong_travel_time":
        metrics["travel_time"] += .1
    elif mutation == "wrong_ttc":
        metrics["min_ttc"] = 2.9
    elif mutation == "wrong_unsafe_count":
        metrics["unsafe_ttc_events"] = 1
    elif mutation == "wrong_collision":
        metrics["collision"] = True
    elif mutation == "lying_clip_mask":
        decisions[0]["frame"]["native_events"][-1]["clipped_mask"][0][1] = True
    elif mutation == "lying_boundary_mask":
        decisions[0]["frame"]["native_events"][-1]["boundary_mask"][0][0] = False
    elif mutation == "forecast_prefix_mismatch":
        decisions[0]["frame"]["policy_observation"][106] = .1
    elif mutation == "native_prefix_mismatch":
        decisions[0]["frame"]["policy_observation"][0] = 0
    else:
        decisions[0]["frame"]["policy_observation"][105] = 0
    with pytest.raises(ValueError):
        analysis.summarize_episode(*episode)


def serialized_records(episode):
    decisions, transitions, terminal, _ = episode
    records = [record for pair in zip(decisions, transitions) for record in pair]
    return [*records, terminal]


def write_trace(path, records, trailing_newline=True):
    text = "\n".join(json.dumps(record, allow_nan=False) for record in records)
    if trailing_newline:
        text += "\n"
    path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))


@pytest.mark.parametrize("arm", ["geometry", "padding", "sector"])
def test_streaming_parser_preserves_complete_synthetic_episode_records(tmp_path, arm):
    episode = synthetic_episode(post_targets=(4.5,), actions=(1,), arm=arm)
    path = tmp_path / "synthetic.trace.jsonl.gz"
    write_trace(path, serialized_records(episode))
    parsed = list(analysis.read_trace_episodes(path, arm, seeds=(7,)))
    assert parsed == [(7, *episode[:3])]
    assert analysis.summarize_episode(*parsed[0][1:], episode[-1])["arm"] == arm


@pytest.mark.parametrize("mutation", [
    "empty", "no_terminal", "no_transition", "terminal_first", "transition_first",
    "decision_twice", "wrong_arm", "wrong_seed", "boolean_seed", "wrong_transition_seed",
    "wrong_transition_arm", "duplicate_terminal", "extra_episode", "missing_newline",
    "truncated_gzip", "bad_crc", "too_many_steps",
])
def test_streaming_parser_rejects_truncation_order_identity_and_extra_records(tmp_path, mutation):
    records = serialized_records(synthetic_episode(post_targets=(4.5,), actions=(1,)))
    if mutation == "empty":
        records = []
    elif mutation == "no_terminal":
        records.pop()
    elif mutation == "no_transition":
        records = records[:1]
    elif mutation == "terminal_first":
        records = records[-1:]
    elif mutation == "transition_first":
        records = records[1:]
    elif mutation == "decision_twice":
        records.insert(1, copy.deepcopy(records[0]))
    elif mutation == "wrong_arm":
        records[0]["arm"] = "padding"
    elif mutation == "wrong_seed":
        records[0]["seed"] = 8
    elif mutation == "boolean_seed":
        records[0]["seed"] = True
    elif mutation == "wrong_transition_seed":
        records[1]["seed"] = 8
    elif mutation == "wrong_transition_arm":
        records[1]["arm"] = "padding"
    elif mutation == "duplicate_terminal":
        records.append(copy.deepcopy(records[-1]))
    elif mutation == "extra_episode":
        records += copy.deepcopy(records)
    elif mutation == "too_many_steps":
        records = records[:2] * 152 + records[-1:]
    path = tmp_path / "synthetic.trace.jsonl.gz"
    write_trace(path, records, trailing_newline=mutation != "missing_newline")
    if mutation == "truncated_gzip":
        path.write_bytes(path.read_bytes()[:-5])
    elif mutation == "bad_crc":
        content = bytearray(path.read_bytes())
        content[-8] ^= 1
        path.write_bytes(content)
    with pytest.raises((ValueError, EOFError, gzip.BadGzipFile)):
        list(analysis.read_trace_episodes(path, "sector", seeds=(7,)))


@pytest.mark.parametrize("text", [
    '{"kind":"decision","kind":"terminal","arm":"sector","seed":7}\n',
    '{"kind":"decision","arm":"sector","seed":7,"value":NaN}\n',
    '{"kind":"decision","arm":"sector","seed":7,"value":Infinity}\n',
    '{"kind":"decision","arm":"sector","seed":7,"value":-Infinity}\n',
    '[]\n',
    '{invalid json}\n',
])
def test_trace_parser_rejects_ambiguous_or_nonstandard_json(tmp_path, text):
    path = tmp_path / "synthetic.trace.jsonl.gz"
    path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    with pytest.raises(ValueError):
        list(analysis.read_trace_episodes(path, "sector", seeds=(7,)))


def test_groups_weight_each_episode_once_and_keep_undefined_rates_explicit():
    short = analysis.summarize_episode(*synthetic_episode(post_targets=(0,), actions=(1,)))
    long = analysis.summarize_episode(*synthetic_episode())
    short["speed"]["decision_samples"]["mean"] = 10
    long["speed"]["decision_samples"]["mean"] = 2
    short["seed"], long["seed"] = 7, 8
    group = analysis._group([short, long])
    assert group["episode_count"] == 2 and group["seeds"] == [7, 8]
    distributions = group["episode_distributions"]
    assert distributions["decision_sample_mean_speed"]["mean"] == 6
    assert distributions["decision_sample_mean_speed"]["count"] == 2
    assert distributions["elapsed_seconds"]["mean"] == pytest.approx(.7)
    rate = distributions["positive_target_progress_per_second"]
    assert rate["count"] == 1 and rate["undefined_episodes"] == 1
    assert rate["mean"] == pytest.approx(.5)


def test_empty_outcome_group_is_retained_without_invented_zero_metrics():
    result = analysis._group([])
    assert result["episode_count"] == 0 and result["seeds"] == []
    for distribution in result["episode_distributions"].values():
        assert distribution == {"count": 0, "min": None, "median": None,
                                "mean": None, "max": None, "undefined_episodes": 0}
    assert result["target_seconds"]["0"]["mean"] is None
    assert result["action_counts"]["IDLE"]["mean"] is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1"])
def test_statistics_never_silently_drop_nonfinite_or_nonnumeric_values(value):
    with pytest.raises(ValueError):
        analysis.stats([1.0, value])


@pytest.fixture
def source_tree(tmp_path):
    names = {"scripts/historical.py", *analysis.ALLOWED_NEW_SOURCES}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic bytes\n", encoding="utf-8")
    dependency = tmp_path / "dependency.py"
    dependency.write_text("# external dependency fixture\n", encoding="utf-8")
    hashes = {name: analysis.sha(tmp_path / name) for name in names}
    hashes[str(dependency)] = analysis.sha(dependency)
    return tmp_path, hashes


def test_pinned_input_accepts_exact_bytes_and_rejects_wrong_hash(source_tree):
    root, hashes = source_tree
    relative = "scripts/historical.py"
    assert analysis._pinned(root, relative, hashes[relative]) == root / relative
    with pytest.raises(ValueError, match="Frozen input hash mismatch"):
        analysis._pinned(root, relative, "0" * 64)


def test_post_read_checks_accept_unchanged_relative_and_absolute_inputs(source_tree):
    root, hashes = source_tree
    before = copy.deepcopy(hashes)
    analysis._verify_unchanged(root, hashes)
    assert hashes == before


@pytest.mark.parametrize("mutation", ["source", "dependency", "new_python", "missing_source"])
def test_post_read_checks_reject_changed_bytes_or_source_inventory(source_tree, mutation):
    root, hashes = source_tree
    if mutation == "source":
        (root / "scripts/historical.py").write_text("changed", encoding="utf-8")
    elif mutation == "dependency":
        (root / "dependency.py").write_text("changed", encoding="utf-8")
    elif mutation == "new_python":
        (root / "scripts/unreleased.py").write_text("extra", encoding="utf-8")
    else:
        (root / "scripts/historical.py").unlink()
    with pytest.raises(ValueError):
        analysis._verify_unchanged(root, hashes)


def synthetic_ledger(tmp_path, monkeypatch):
    # 111 cheap, fabricated records; these are not historical research seed identities or data.
    seeds = tuple(range(7, 44))
    monkeypatch.setattr(analysis.protocol, "SEEDS", seeds)
    monkeypatch.setattr(analysis.protocol, "FIRST_SEED", 7)
    metrics = synthetic_episode(post_targets=(4.5,), actions=(1,))[-1]
    references = {arm: [copy.deepcopy(metrics) for _ in seeds] for arm in analysis.ARM_ORDER}
    records, checks = [], []
    for index, (arm, seed) in enumerate((a, s) for a in analysis.ARM_ORDER for s in seeds):
        start = {"kind": "start", "arm": arm, "seed": seed, "attempt_index": index,
                 "utc": "2026-01-01T00:00:00+00:00"}
        finish = {"kind": "complete", "arm": arm, "seed": seed, "attempt_index": index,
                  "actual": copy.deepcopy(metrics), "reference": copy.deepcopy(metrics),
                  "deltas": analysis.protocol.compare_episode(metrics, metrics)}
        records += [start, finish]
        checks.append(copy.deepcopy(finish))
    path = tmp_path / analysis.RAW_DIRECTORY / "episodes.jsonl"
    path.parent.mkdir(parents=True)
    return path, records, {"episode_checks": checks}, references


def test_ledger_verifies_each_reference_actual_metric_and_zero_delta(tmp_path, monkeypatch):
    path, records, summary, references = synthetic_ledger(tmp_path, monkeypatch)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    actual = analysis._read_ledger(tmp_path, summary, references)
    assert len(actual) == 111
    assert actual[("sector", 43)] == references["sector"][-1]


@pytest.mark.parametrize("mutation", [
    "missing_line", "wrong_start_index", "boolean_seed", "naive_time", "summary_disagrees",
    "actual_disagrees", "reference_disagrees", "nonzero_delta", "missing_delta",
])
def test_ledger_rejects_incomplete_or_unreconciled_provenance(tmp_path, monkeypatch, mutation):
    path, records, summary, references = synthetic_ledger(tmp_path, monkeypatch)
    if mutation == "missing_line":
        records.pop()
    elif mutation == "wrong_start_index":
        records[0]["attempt_index"] = 1
    elif mutation == "boolean_seed":
        records[0]["seed"] = True
    elif mutation == "naive_time":
        records[0]["utc"] = "2026-01-01T00:00:00"
    elif mutation == "summary_disagrees":
        summary["episode_checks"][0]["actual"]["length"] += 1
    elif mutation == "actual_disagrees":
        records[1]["actual"]["reward"] += 1
    elif mutation == "reference_disagrees":
        records[1]["reference"]["reward"] += 1
    elif mutation == "nonzero_delta":
        records[1]["deltas"]["reward"] = 1
    else:
        records[1]["deltas"].pop("reward")
    if mutation != "summary_disagrees":
        summary["episode_checks"][0] = copy.deepcopy(records[1])
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError):
        analysis._read_ledger(tmp_path, summary, references)


def synthetic_analysis_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis.protocol, "SEEDS", (7,))
    release = {"source_sha256": {}, "dependency_sha256": {}}
    summary = {"finished_utc": "2026-01-01T00:00:00+00:00"}
    hashes = {name: "synthetic" for name in analysis.ALLOWED_NEW_SOURCES}
    episodes = {arm: synthetic_episode(post_targets=(4.5,), actions=(1,), arm=arm)
                for arm in analysis.ARM_ORDER}
    ledger = {(arm, 7): episode[-1] for arm, episode in episodes.items()}
    calls = []
    monkeypatch.setattr(analysis, "_verify_inputs", lambda root: (release, summary, {}, hashes))
    monkeypatch.setattr(analysis, "_read_ledger", lambda *args: ledger)
    monkeypatch.setattr(analysis, "_file", lambda root, name: root / name)

    def read_synthetic(path, arm):
        calls.append(("read", arm))
        return iter([(7, *episodes[arm][:3])])

    monkeypatch.setattr(analysis, "read_trace_episodes", read_synthetic)
    monkeypatch.setattr(analysis, "_verify_unchanged", lambda *args: calls.append(("postflight",)))
    return calls


def test_analysis_joins_synthetic_cases_groups_and_postflight_without_writing(
        tmp_path, monkeypatch):
    calls = synthetic_analysis_setup(tmp_path, monkeypatch)
    result = analysis.analyze(tmp_path)
    assert result["status"] == "complete" and result["episode_count"] == 3
    assert result["research_episodes_executed_by_analysis"] == 0
    assert result["selected_seeds"] == [7]
    assert [e["arm"] for e in result["episodes"]] == list(analysis.ARM_ORDER)
    assert result["verification"]["ledger_source_bound_columns"] == [
        "success", "safety_interventions",
    ]
    for arm in analysis.ARM_ORDER:
        outcomes = result["groups"][arm]["by_observed_outcome"]
        assert outcomes["incomplete"]["episode_count"] == 1
        assert outcomes["success"]["episode_count"] == outcomes["collision"]["episode_count"] == 0
    assert calls == [("read", arm) for arm in analysis.ARM_ORDER] + [("postflight",)]
    assert list(tmp_path.iterdir()) == []


def test_analysis_cannot_return_success_after_postflight_hash_failure(tmp_path, monkeypatch):
    synthetic_analysis_setup(tmp_path, monkeypatch)

    def fail_postflight(*args):
        raise ValueError("synthetic input changed during read")

    monkeypatch.setattr(analysis, "_verify_unchanged", fail_postflight)
    with pytest.raises(ValueError, match="input changed during read"):
        analysis.analyze(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cli_requires_preservation_flag_and_disallows_output_override(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Invalid CLI must not start analysis")

    monkeypatch.setattr(analysis, "analyze", forbidden)
    for extras in ([], ["--refuse-overwrite", "--output", "other.json"]):
        monkeypatch.setattr(sys, "argv", ["analyzer", "--root", str(tmp_path), *extras])
        with pytest.raises(SystemExit) as error:
            analysis.main()
        assert error.value.code == 2


def cli_setup(tmp_path, monkeypatch):
    destination = tmp_path / analysis.OUTPUT
    destination.parent.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", [
        "analyzer", "--root", str(tmp_path), "--refuse-overwrite",
    ])
    return destination


def test_cli_refuses_existing_output_before_analysis(tmp_path, monkeypatch):
    destination = cli_setup(tmp_path, monkeypatch)
    destination.write_bytes(b"preserved prior result")
    monkeypatch.setattr(analysis, "analyze", lambda *args: pytest.fail("Must refuse first"))
    with pytest.raises(ValueError, match="Preserve existing analysis output"):
        analysis.main()
    assert destination.read_bytes() == b"preserved prior result"


def test_cli_exclusive_create_preserves_output_that_appears_during_analysis(tmp_path, monkeypatch):
    destination = cli_setup(tmp_path, monkeypatch)

    def raced_analysis(root):
        destination.write_bytes(b"concurrent artifact")
        return {"status": "complete"}

    monkeypatch.setattr(analysis, "analyze", raced_analysis)
    with pytest.raises(FileExistsError):
        analysis.main()
    assert destination.read_bytes() == b"concurrent artifact"


def test_cli_analysis_failure_creates_no_report(tmp_path, monkeypatch):
    destination = cli_setup(tmp_path, monkeypatch)

    def failed_analysis(root):
        raise ValueError("synthetic verification failed")

    monkeypatch.setattr(analysis, "analyze", failed_analysis)
    with pytest.raises(ValueError, match="synthetic verification failed"):
        analysis.main()
    assert not destination.exists()


def test_cli_writes_only_fixed_analysis_artifact_after_success(tmp_path, monkeypatch, capsys):
    destination = cli_setup(tmp_path, monkeypatch)
    result = {"status": "complete", "research_episodes_executed_by_analysis": 0}
    roots = []

    def verified_analysis(root):
        roots.append(root)
        return result

    monkeypatch.setattr(analysis, "analyze", verified_analysis)
    analysis.main()
    assert roots == [tmp_path.resolve()]
    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert destination.read_bytes().endswith(b"\n")
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [destination]
    assert "Offline analysis saved" in capsys.readouterr().out
