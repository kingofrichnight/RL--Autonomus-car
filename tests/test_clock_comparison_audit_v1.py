import copy
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_clock_comparison_v1 as audit

COLUMNS = ("reward", "length", "success", "collision", "travel_time", "min_ttc",
           "unsafe_ttc_events", "safety_interventions")
RELEASE_HASH = "a" * 64
ORIGINAL_INSPECT_MODEL = audit.inspect_model


def forbidden(*args, **kwargs):
    pytest.fail("Synthetic audit tests must not load or execute any real policy or environment")


@pytest.fixture(autouse=True)
def forbid_real_model_inspection(monkeypatch):
    import gymnasium as gym
    from stable_baselines3 import PPO

    from safeintent_rl.envs import intersection

    monkeypatch.setattr(audit, "inspect_model", forbidden)
    monkeypatch.setattr(gym, "make", forbidden)
    monkeypatch.setattr(intersection, "make_intersection_env", forbidden)
    for method in ("__init__", "load", "save", "predict", "learn", "train"):
        monkeypatch.setattr(PPO, method, forbidden)


def episode(**overrides):
    row = {"reward": 2.0, "length": 10, "success": True, "collision": False,
           "travel_time": 2.0, "min_ttc": 3.0, "unsafe_ttc_events": 0,
           "safety_interventions": 0}
    row.update(overrides)
    return row


def rows500():
    return [episode() for _ in range(500)]


def write_rows(path, rows, columns=COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_reader_preserves_exact_500_synthetic_episode_rows_and_boolean_types(tmp_path):
    rows = rows500()
    rows[1] = episode(success=False, collision=True, reward=-10.0)
    rows[2] = episode(success=False, collision=False, length=151, travel_time=30.2)
    path = tmp_path / "synthetic.csv"
    write_rows(path, rows)
    parsed = audit.read_rows(path)
    assert parsed == rows and len(parsed) == 500
    assert type(parsed[0]["success"]) is bool and type(parsed[0]["collision"]) is bool
    assert list(parsed[0]) == list(COLUMNS)


@pytest.mark.parametrize("count", [0, 1, 499, 501])
def test_reader_requires_exactly_500_rows(tmp_path, count):
    path = tmp_path / "synthetic.csv"
    write_rows(path, [episode() for _ in range(count)])
    with pytest.raises(ValueError):
        audit.read_rows(path)


@pytest.mark.parametrize("field,value", [
    ("reward", "NaN"), ("reward", "Infinity"), ("reward", "-Infinity"),
    ("length", "1.5"), ("length", "0"), ("length", "152"), ("length", "True"),
    ("success", "1"), ("success", "true"), ("collision", "0"),
    ("travel_time", "NaN"), ("travel_time", "2.1"), ("travel_time", "True"),
    ("unsafe_ttc_events", "-1"), ("unsafe_ttc_events", "11"),
    ("safety_interventions", "1"), ("safety_interventions", "False"),
    ("min_ttc", "-1"), ("min_ttc", "NaN"), ("min_ttc", "-Infinity"),
    ("min_ttc", ""),
])
def test_reader_rejects_invalid_types_ranges_and_nonfinite_non_ttc_metrics(tmp_path, field, value):
    rows = rows500()
    rows[0][field] = value
    path = tmp_path / "synthetic.csv"
    write_rows(path, rows)
    with pytest.raises(ValueError):
        audit.read_rows(path)


def test_reader_rejects_success_collision_overlap(tmp_path):
    rows = rows500()
    rows[0]["collision"] = True
    path = tmp_path / "synthetic.csv"
    write_rows(path, rows)
    with pytest.raises(ValueError):
        audit.read_rows(path)


@pytest.mark.parametrize("mutation", ["header_order", "extra_column", "missing_column",
                                      "short_row", "extra_cell", "duplicate_header"])
def test_reader_rejects_changed_eight_column_schema_or_malformed_rows(tmp_path, mutation):
    rows = rows500()
    columns = list(COLUMNS)
    if mutation == "header_order":
        columns[0], columns[1] = columns[1], columns[0]
    elif mutation == "extra_column":
        columns.append("seed")
        for row in rows:
            row["seed"] = 7
    elif mutation == "missing_column":
        columns.remove("min_ttc")
        for row in rows:
            row.pop("min_ttc")
    path = tmp_path / "synthetic.csv"
    write_rows(path, rows, columns)
    if mutation in ("short_row", "extra_cell", "duplicate_header"):
        lines = path.read_text().splitlines()
        if mutation == "short_row":
            lines[1] = ",".join(lines[1].split(",")[:-1])
        elif mutation == "extra_cell":
            lines[1] += ",unexpected"
        else:
            lines[0] = lines[0].replace("reward", "length", 1)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit.read_rows(path)


def test_positive_infinite_ttc_is_preserved_and_excluded_from_episode_mean(tmp_path):
    rows = rows500()
    for row in rows:
        row["min_ttc"] = math.inf
    rows[0].update(min_ttc=1.0, unsafe_ttc_events=1, length=1, travel_time=.2)
    rows[1].update(min_ttc=3.0, length=100, travel_time=20.0)
    path = tmp_path / "synthetic.csv"
    write_rows(path, rows)
    parsed = audit.read_rows(path)
    assert math.isinf(parsed[2]["min_ttc"])
    metrics, counts = audit.metrics(parsed)
    assert metrics["mean_min_ttc"] == 2.0
    assert counts["excluded_nonfinite_ttc"] == 498
    assert metrics["mean_min_ttc"] != pytest.approx((1 + 100 * 3) / 101)


def paired_rows(rescues, regressions):
    reference, candidate = rows500(), rows500()
    for index in range(rescues):
        reference[index]["success"] = False
        reference[index]["collision"] = True
    for index in range(rescues, rescues + regressions):
        candidate[index]["success"] = False
        candidate[index]["collision"] = True
    return reference, candidate


@pytest.mark.parametrize("rescues,regressions,expected_p,favorable", [
    (0, 0, 1.0, False), (6, 0, .03125, True), (5, 0, .0625, False),
    (0, 6, .03125, False), (3, 3, 1.0, False),
])
def test_paired_exact_two_sided_success_comparison(rescues, regressions, expected_p, favorable):
    reference, candidate = paired_rows(rescues, regressions)
    before = copy.deepcopy((reference, candidate))
    result = audit.paired(reference, candidate)
    assert result["rescues"] == rescues and result["regressions"] == regressions
    assert result["p_exact_two_sided"] == expected_p and result["favorable"] is favorable
    assert (reference, candidate) == before


def test_paired_comparison_is_positional_and_detects_swapped_success_identities():
    reference, candidate = paired_rows(0, 0)
    reference[0]["success"], reference[0]["collision"] = False, True
    candidate[1]["success"], candidate[1]["collision"] = False, True
    result = audit.paired(reference, candidate)
    assert result["rescues"] == result["regressions"] == 1
    assert result["p_exact_two_sided"] == 1 and not result["favorable"]


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_paired_comparison_rejects_unpaired_population_lengths(side):
    reference, candidate = paired_rows(0, 0)
    (reference if side == "reference" else candidate).pop()
    with pytest.raises(ValueError):
        audit.paired(reference, candidate)


@pytest.mark.parametrize("p_value,expected", [
    (math.nextafter(.05, 0), True), (.05, False), (math.nextafter(.05, 1), False),
])
def test_favorable_predicate_requires_strict_p_less_than_point_zero_five(p_value, expected):
    # Test the predicate directly; no fabricated claim that a .05 exact table exists.
    assert audit.favorable(10, 1, p_value) is expected
    assert audit.favorable(1, 10, p_value) is False
    assert audit.favorable(10, 10, p_value) is False


HISTORICAL_KEYS = {
    "absolute_success", "absolute_collision", "incomplete", "original_ttc",
    "original_paired_success", "control_success", "control_collision", "control_ttc",
    "control_paired_success", "v1_success", "v1_collision", "v1_ttc", "v1_paired_success",
    "geometry_success", "geometry_collision", "geometry_incomplete",
}


def counts(success=400, collision=92, incomplete=8):
    return {"success": success, "collision": collision, "incomplete": incomplete,
            "finite_ttc": 500, "excluded_nonfinite_ttc": 0}


def comparisons():
    return {name: {"favorable": True} for name in ("original", "control", "v1", "geometry")}


def test_historical_gates_are_exactly_sixteen_with_no_extra_geometry_significance_gate():
    paired = comparisons()
    paired["geometry"]["favorable"] = False
    gates = audit.historical_gates(counts(), 1.0, paired)
    assert set(gates) == HISTORICAL_KEYS and len(gates) == 16
    assert all(value is True for value in gates.values())


@pytest.mark.parametrize("key,field,threshold,adverse_direction", [
    ("absolute_success", "success", 315, -1),
    ("control_success", "success", 304, -1),
    ("v1_success", "success", 364, -1),
    ("geometry_success", "success", 394, -1),
    ("absolute_collision", "collision", 174, 1),
    ("control_collision", "collision", 195, 1),
    ("v1_collision", "collision", 101, 1),
    ("geometry_collision", "collision", 98, 1),
    ("incomplete", "incomplete", 10, 1),
    ("geometry_incomplete", "incomplete", 8, 1),
])
def test_each_historical_count_gate_keeps_exact_inclusive_boundary(
        key, field, threshold, adverse_direction):
    for value, expected in ((threshold, True), (threshold + adverse_direction, False)):
        actual = counts()
        actual[field] = value
        if field == "success":
            actual["collision"] = 500 - actual["success"] - actual["incomplete"]
        else:
            actual["success"] = 500 - actual["collision"] - actual["incomplete"]
        assert audit.historical_gates(actual, 1.0, comparisons())[key] is expected


@pytest.mark.parametrize("key,threshold", [
    ("original_ttc", .6003656548142169), ("control_ttc", .6065401801078688),
    ("v1_ttc", .6728939419963959),
])
def test_each_historical_ttc_gate_keeps_exact_float_boundary(key, threshold):
    assert audit.historical_gates(counts(), threshold, comparisons())[key] is True
    below = audit.historical_gates(counts(), math.nextafter(threshold, 0), comparisons())
    assert below[key] is False


@pytest.mark.parametrize("name", ["original", "control", "v1"])
def test_each_registered_historical_paired_success_gate_remains_required(name):
    paired = comparisons()
    paired[name]["favorable"] = False
    gates = audit.historical_gates(counts(), 1.0, paired)
    assert [key for key, passed in gates.items() if not passed] == [f"{name}_paired_success"]


@pytest.mark.parametrize("unavailable", [None, math.inf, -math.inf, math.nan])
def test_unavailable_ttc_cannot_pass_any_historical_ttc_gate(unavailable):
    gates = audit.historical_gates(counts(), unavailable, comparisons())
    assert all(gates[key] is False for key in ("original_ttc", "control_ttc", "v1_ttc"))


def test_metrics_report_all_nonfinite_ttc_as_unavailable_without_fabricating_zero():
    rows = [episode(min_ttc=math.inf) for _ in range(500)]
    metrics, actual_counts = audit.metrics(rows)
    assert metrics["mean_min_ttc"] is None
    assert actual_counts["finite_ttc"] == 0 and actual_counts["excluded_nonfinite_ttc"] == 500


def test_metrics_compute_exclusive_outcome_counts_and_episode_means():
    rows = ([episode(reward=1.0) for _ in range(400)]
            + [episode(success=False, collision=True, reward=-10.0) for _ in range(92)]
            + [episode(success=False, length=151, travel_time=30.2, reward=0.0) for _ in range(8)])
    metrics, actual_counts = audit.metrics(rows)
    assert actual_counts == counts()
    assert metrics["episodes"] == 500
    assert metrics["success_rate"] == .8 and metrics["collision_rate"] == .184
    assert metrics["mean_reward"] == pytest.approx((400 - 920) / 500)
    assert metrics["mean_length"] == pytest.approx((492 * 10 + 8 * 151) / 500)
    assert metrics["mean_travel_time"] == pytest.approx(metrics["mean_length"] / 5)


BENEFIT_KEYS = {"all_historical_gates", "paired_success_vs_zero", "collision_vs_zero",
                "incomplete_vs_zero", "ttc_vs_zero"}


def evidence(success=400, collision=92, incomplete=8, mean_ttc=1.0):
    return {"counts": counts(success, collision, incomplete),
            "metrics": {"mean_min_ttc": mean_ttc}}


def test_clock_benefit_adds_exactly_five_rules_and_equal_safety_counts_are_non_regression():
    old = {key: True for key in HISTORICAL_KEYS}
    zero, clock = evidence(), evidence()
    gates = audit.clock_benefit_gates(old, zero, clock, {"favorable": True})
    assert set(gates) == BENEFIT_KEYS and all(value is True for value in gates.values())


@pytest.mark.parametrize("mutation,failed_key", [
    ("historical", "all_historical_gates"), ("paired", "paired_success_vs_zero"),
    ("collision", "collision_vs_zero"), ("incomplete", "incomplete_vs_zero"),
    ("ttc", "ttc_vs_zero"), ("clock_unavailable", "ttc_vs_zero"),
    ("zero_unavailable", "ttc_vs_zero"),
])
def test_clock_benefit_cannot_replace_failed_historical_or_additional_gates(mutation, failed_key):
    old = {key: True for key in HISTORICAL_KEYS}
    zero, clock, paired = evidence(), evidence(), {"favorable": True}
    if mutation == "historical":
        old["geometry_incomplete"] = False
    elif mutation == "paired":
        paired["favorable"] = False
    elif mutation in ("collision", "incomplete"):
        clock["counts"][mutation] += 1
        clock["counts"]["success"] -= 1
    elif mutation == "ttc":
        clock["metrics"]["mean_min_ttc"] = math.nextafter(1.0, 0)
    else:
        (clock if mutation == "clock_unavailable" else zero)["metrics"]["mean_min_ttc"] = None
    gates = audit.clock_benefit_gates(old, zero, clock, paired)
    assert [key for key, passed in gates.items() if not passed] == [failed_key]


@pytest.mark.parametrize("field,value", [
    ("reward", True), ("length", True), ("success", 1), ("collision", 0),
    ("travel_time", False), ("min_ttc", True), ("unsafe_ttc_events", False),
    ("safety_interventions", False),
])
def test_direct_metrics_rejects_boolean_numeric_coercion(field, value):
    rows = rows500()
    rows[0][field] = value
    with pytest.raises(ValueError):
        audit.metrics(rows)


@pytest.mark.parametrize("text,summary", [
    ('{"status":"complete","status":"failed"}', False),
    ('{"mean_min_ttc":Infinity}', False),
    ('{"mean_min_ttc":NaN}', True),
    ('{"mean_min_ttc":-Infinity}', True),
    ('{"mean_reward":Infinity}', True),
    ('{"nested":{"mean_min_ttc":Infinity}}', True),
    ('[]', False),
])
def test_json_rejects_duplicate_keys_and_unregistered_nonfinite_values(tmp_path, text, summary):
    path = tmp_path / "synthetic.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        audit.read_json(path, summary=summary)


def test_summary_json_retains_only_historical_positive_infinite_minimum_ttc(tmp_path):
    path = tmp_path / "synthetic.summary.json"
    path.write_text('{"mean_min_ttc":Infinity,"mean_reward":1.0}', encoding="utf-8")
    parsed = audit.read_json(path, summary=True)
    assert parsed["mean_min_ttc"] == math.inf and parsed["mean_reward"] == 1.0


@pytest.mark.parametrize("actual,expected", [
    (False, 0), (0, False), (True, 1.0), (True, 1), ({}, {"absent": None}),
    ({"safety_shield": 0}, {"safety_shield": False}),
])
def test_structural_equality_rejects_bool_number_aliases_and_missing_nulls(actual, expected):
    assert audit.same(actual, expected) is False


def fabricated_provenance():
    return {"release_sha256": RELEASE_HASH, "input_sha256": {"old.py": "b" * 64},
            "runtime": {"synthetic": True}, "config_sha256": audit.CONFIG_SHA256}


def fabricated_summary(arm="zero", mode="evaluate", model_hash="b" * 64, provenance=None,
                       config_hash=None):
    provenance = fabricated_provenance() if provenance is None else provenance
    suffix = "zero" if arm == "zero" else "remaining"
    summary = {
        "config_path": "configs/intersection_v3_predictive_geometry_v2.yaml",
        "config_sha256": (config_hash or
                          "a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7"),
        "model_path": f"models/ppo_v3_clock_{suffix}_v1_seed42.zip", "model_sha256": model_hash,
        "observation_protocol": ("predictive_post_spawn_clock_zero_v1" if arm == "zero"
                                 else "predictive_post_spawn_clock_remaining_v1"),
        "clock_comparison_arm": arm, "clock_duration": 30,
        "initial_policy": {
            "parameter_count": 192516,
            "sha256": "d1c314ef787fa038568eeaff5dfc6c1a3851aba5f39a4dd8ba767634943a57ec",
        },
        "clock_provenance": copy.deepcopy(provenance),
        "safety_shield": False, "risk_fusion": False, "target_speed_observation": False,
        "target_speed_scale": None, "intent_model_path": None, "intent_model_sha256": None,
        "intent_neighbors": 0, "intent_history_length": None,
        "intent_history_tracking_neighbors": None, "intent_device": None,
        "collision_first_reward": True, "fusion_neighbors": 0, "fusion_features_per_neighbor": 0,
        "fusion_range_scale": None, "fusion_relative_speed_scale": None, "fusion_ttc_scale": None,
        "fusion_cpa_horizon": None, "fusion_cpa_distance_scale": None,
        "predictive_safety_observation": {"horizon": 3.0, "margin": .5, "uncertainty_growth": .25,
                                          "clearance_scale": 10.0, "speed_scale": 9.0,
                                          "max_neighbors": 14},
    }
    if mode == "train":
        summary.update(
            algorithm="PPO", training_seed=42, internal_evaluation_seed_offset=70000,
            timesteps_requested=200000, timesteps_collected=200704, learning_rate=.0003,
            n_steps=1024, batch_size=64, n_envs=1, env_seed_stride=1000,
            training_seed_offsets=[0], training_initial_seeds=[42], rollout_size=1024,
            eval_episodes=50, evaluation_freq_timesteps=10000,
            evaluation_callback_frequency=10000, checkpoint_freq_timesteps=25000,
            checkpoint_callback_frequency=25000, gamma=.99, gae_lambda=.95, ent_coef=.01,
            policy_network=[256, 256], observation_shape=[116], ttc_threshold=2.0,
        )
    else:
        summary.update(
            episodes=500.0, first_seed=40042, last_seed=40541, unsafe_ttc_threshold=2.0,
            safety_shield_type=None, cpa_time_threshold=None, cpa_distance_threshold=None,
            cpa_horizon=None, cpa_max_range=None, cpa_override_action=None,
            reference_csv_path="results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv",
            reference_csv_sha256="aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4",
            mean_reward=2.0, mean_length=10.0, success_rate=1.0, collision_rate=0.0,
            mean_travel_time=2.0, mean_min_ttc=3.0, mean_unsafe_ttc_events=0.0,
            mean_safety_interventions=0.0,
        )
    return summary


@pytest.mark.parametrize("arm", ["zero", "clock"])
@pytest.mark.parametrize("mode", ["train", "evaluate"])
def test_independent_full_summary_fixture_matches_contract_and_normalizes_only_paths(arm, mode):
    summary = fabricated_summary(arm, mode)
    for key in ("config_path", "model_path", "reference_csv_path"):
        if key in summary:
            summary[key] = summary[key].replace("/", "\\")
    before = copy.deepcopy(summary)
    audit.checked_summary(summary, arm, mode, fabricated_provenance(), "b" * 64)
    assert summary == before


@pytest.mark.parametrize("mode", ["train", "evaluate"])
@pytest.mark.parametrize("mutation", ["missing_null", "extra", "arm", "duration", "config",
                                      "seed", "predictive", "bool_as_count", "number_as_bool"])
def test_checked_summary_rejects_scientific_and_structural_changes(mode, mutation):
    summary = fabricated_summary(mode=mode)
    if mutation == "missing_null":
        summary.pop("fusion_range_scale")
    elif mutation == "extra":
        summary["unregistered_sensor"] = False
    elif mutation == "arm":
        summary["clock_comparison_arm"] = "clock"
    elif mutation == "duration":
        summary["clock_duration"] = 29
    elif mutation == "config":
        summary["config_sha256"] = "0" * 64
    elif mutation == "seed":
        summary["training_seed" if mode == "train" else "last_seed"] += 1
    elif mutation == "predictive":
        summary["predictive_safety_observation"]["max_neighbors"] = 13
    elif mutation == "bool_as_count":
        summary["intent_neighbors"] = False
    else:
        summary["safety_shield"] = 0
    with pytest.raises(ValueError):
        audit.checked_summary(summary, "zero", mode, fabricated_provenance(), "b" * 64)


@pytest.mark.parametrize("mutation", ["reward", "mean_ttc", "bool_metric", "first_seed",
                                      "last_seed", "shield", "ttc_threshold"])
def test_reconcile_rejects_metric_or_reset_metadata_mismatch(mutation):
    summary = fabricated_summary()
    if mutation == "reward":
        summary["mean_reward"] += .01
    elif mutation == "mean_ttc":
        summary["mean_min_ttc"] += .01
    elif mutation == "bool_metric":
        summary["mean_safety_interventions"] = False
    elif mutation in ("first_seed", "last_seed"):
        summary[mutation] += 1
    elif mutation == "shield":
        summary["safety_shield"] = 0
    else:
        summary["unsafe_ttc_threshold"] = 1.5
    with pytest.raises(ValueError):
        audit.reconcile(summary, rows500())


def test_reconcile_preserves_raw_infinite_ttc_but_reports_unavailable():
    summary = fabricated_summary()
    rows = [episode(min_ttc=math.inf) for _ in range(500)]
    summary["mean_min_ttc"] = math.inf
    calculated, actual_counts = audit.reconcile(summary, rows)
    assert calculated["mean_min_ttc"] is None and actual_counts["finite_ttc"] == 0
    assert summary["mean_min_ttc"] == math.inf
    summary["mean_min_ttc"] = None
    with pytest.raises(ValueError):
        audit.reconcile(summary, rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def released(tmp_path, monkeypatch):
    current = {"input_sha256": {"old.py": "b" * 64}, "runtime": {"frozen": True},
               "dependency_sha256": {"synthetic-dependency": "c" * 64},
               "snapshot_archive_sha256": "d" * 64}
    validation = {"ruff": "passed", "pytest": "passed", "independent_review": "passed",
                  "tests_passed": 1129, "completed_utc": "2026-01-01T00:00:00+00:00",
                  "source_head": "f" * 40}
    proof = {"protocol": audit.protocol_description(), **copy.deepcopy(current),
             "validation": copy.deepcopy(validation)}
    write_json(tmp_path / audit.VERIFICATION_PATH, proof)
    validation["verification_sha256"] = audit.sha(tmp_path / audit.VERIFICATION_PATH)
    release = {"schema_version": 1, "status": "released",
               "created_utc": "2026-01-01T01:00:00+00:00",
               "protocol": audit.protocol_description(), "validation": validation,
               **copy.deepcopy(current)}
    write_json(tmp_path / audit.RELEASE_PATH, release)
    monkeypatch.setattr(audit, "current_fingerprints", lambda root: copy.deepcopy(current))
    return SimpleNamespace(root=tmp_path, release=release, proof=proof, current=current)


def test_independent_release_and_proof_binding_accepts_exact_fixture(released):
    digest = audit.sha(released.root / audit.RELEASE_PATH)
    assert audit.verify_release(released.root, digest) == released.release


@pytest.mark.parametrize("mutation", ["schema_bool", "extra", "not_released", "protocol",
                                      "old_source", "dependency", "runtime", "snapshot",
                                      "too_few_tests", "bool_tests", "review", "head",
                                      "chronology", "naive_date", "proof_hash", "proof_contents"])
def test_release_rejects_missing_or_changed_frozen_metadata(released, mutation):
    release = copy.deepcopy(released.release)
    if mutation == "schema_bool":
        release["schema_version"] = True
    elif mutation == "extra":
        release["extra"] = "not registered"
    elif mutation == "not_released":
        release["status"] = "draft"
    elif mutation == "protocol":
        release["protocol"]["order"].reverse()
    elif mutation == "old_source":
        released.current["input_sha256"]["old.py"] = "0" * 64
    elif mutation == "dependency":
        released.current["dependency_sha256"]["synthetic-dependency"] = "0" * 64
    elif mutation == "runtime":
        released.current["runtime"]["frozen"] = False
    elif mutation == "snapshot":
        released.current["snapshot_archive_sha256"] = "0" * 64
    elif mutation == "too_few_tests":
        release["validation"]["tests_passed"] = 1128
    elif mutation == "bool_tests":
        release["validation"]["tests_passed"] = True
    elif mutation == "review":
        release["validation"]["independent_review"] = "pending"
    elif mutation == "head":
        release["validation"]["source_head"] = "not-a-git-hash"
    elif mutation == "chronology":
        release["validation"]["completed_utc"] = "2026-01-01T02:00:00+00:00"
    elif mutation == "naive_date":
        release["created_utc"] = "2026-01-01T01:00:00"
    elif mutation == "proof_hash":
        release["validation"]["verification_sha256"] = "0" * 64
    else:
        proof = copy.deepcopy(released.proof)
        proof["validation"]["tests_passed"] += 1
        write_json(released.root / audit.VERIFICATION_PATH, proof)
        release["validation"]["verification_sha256"] = audit.sha(
            released.root / audit.VERIFICATION_PATH,
        )
    write_json(released.root / audit.RELEASE_PATH, release)
    with pytest.raises(ValueError):
        audit.verify_release(released.root, audit.sha(released.root / audit.RELEASE_PATH))


def test_missing_release_prevents_completed_chain_or_model_inspection(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "completed_chain", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.audit(tmp_path, RELEASE_HASH)
    assert list(tmp_path.iterdir()) == []


def fabricated_test_evidence(started):
    return {"status": "passed", "tests_passed": 1129,
            "completed_utc": (started + timedelta(seconds=1)).isoformat(),
            "commands": [
                {"command": [sys.executable, *arguments], "exit_code": 0,
                 "stdout_sha256": "b" * 64, "stderr_sha256": "c" * 64}
                for arguments in (["-m", "ruff", "check", "."],
                                  ["-m", "pytest", "-p", "no:cacheprovider"])
            ]}


def build_completed(tmp_path, monkeypatch, release=None, digest=RELEASE_HASH):
    if release is None:
        release = {"input_sha256": {"old.py": "b" * 64}, "runtime": {"frozen": True},
                   "validation": {"tests_passed": 1129},
                   "created_utc": "2026-01-01T00:00:00+00:00"}
    provenance = audit.provenance_for(release, digest)
    dependencies = {}
    records = {}
    calls = []
    for index, (arm, mode) in enumerate(audit.ORDER):
        paths = audit.artifact_paths(arm)
        model = tmp_path / paths["model"]
        model.parent.mkdir(parents=True, exist_ok=True)
        if not model.exists():
            model.write_bytes(f"synthetic {arm} model bytes".encode())
        model_hash = audit.sha(model)
        summary_key = "training_summary" if mode == "train" else "evaluation_summary"
        summary_path = tmp_path / paths[summary_key]
        write_json(summary_path, fabricated_summary(
            arm, mode, model_hash, provenance, config_hash=audit.CONFIG_SHA256,
        ))
        started = datetime(2026, 1, 2, tzinfo=timezone.utc) + timedelta(minutes=index)
        record = {
            "status": "complete", "arm": arm, "mode": mode, "release_sha256": digest,
            "provenance": copy.deepcopy(provenance),
            "arguments": (audit.training_arguments(arm) if mode == "train"
                          else audit.evaluation_arguments(arm, model_hash)),
            "expected_initial_policy": copy.deepcopy(audit.INITIAL_POLICY),
            "initial_policy": copy.deepcopy(audit.INITIAL_POLICY),
            "predecessor_sha256": dict(dependencies), "model_sha256": model_hash,
            "summary_sha256": audit.sha(summary_path), "started_utc": started.isoformat(),
            "finished_utc": (started + timedelta(seconds=2)).isoformat(),
            "tests": fabricated_test_evidence(started),
        }
        keys = ("train_record", "model", "training_summary")
        if mode == "evaluate":
            write_rows(tmp_path / paths["evaluation_csv"], rows500())
            record["csv_sha256"] = audit.sha(tmp_path / paths["evaluation_csv"])
            keys = ("evaluate_record", "evaluation_csv", "evaluation_summary")
        else:
            (tmp_path / paths["logdir"]).mkdir(parents=True)
        write_json(tmp_path / paths[f"{mode}_record"], record)
        dependencies.update({paths[key].as_posix(): audit.sha(tmp_path / paths[key])
                             for key in keys})
        for log in audit.stage_outputs(arm, mode)[-2:]:
            target = tmp_path / log
            target.parent.mkdir(parents=True, exist_ok=True)
            if str(log).endswith(".stdout.log"):
                marker = (f"Clock comparison {arm} {mode} complete: "
                          f"{tmp_path / paths[f'{mode}_record']}\n")
                target.write_text("synthetic console evidence\n" + marker, encoding="utf-8")
            else:
                target.write_bytes(b"")
        records[f"{arm}_{mode}"] = record
    monkeypatch.setattr(audit, "inspect_model", lambda *args: calls.append(args))
    return SimpleNamespace(root=tmp_path, release=release, records=records,
                           dependencies=dependencies, calls=calls, digest=digest)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    return build_completed(tmp_path, monkeypatch)


def test_completed_chain_independently_checks_all_four_stages_before_two_inspections(completed):
    records, fingerprints = audit.completed_chain(completed.root, completed.release, RELEASE_HASH)
    assert records == completed.records
    assert [len(record["predecessor_sha256"]) for record in records.values()] == [0, 3, 6, 9]
    assert len(fingerprints) == 20
    assert len(completed.calls) == 2 and [call[1] for call in completed.calls] == ["zero", "clock"]
    assert all(fingerprints[name] == digest for name, digest in completed.dependencies.items())


@pytest.mark.parametrize("mutation", ["status", "error_field", "release", "cumulative",
                                      "initial", "chronology", "test_time", "test_count",
                                      "test_command", "test_exit_bool", "summary", "model",
                                      "csv", "log_missing", "record_missing", "batch_lock",
                                      "truncated_completion", "wrong_completion"])
def test_completed_chain_rejects_partial_changed_or_misordered_evidence_before_models(
        completed, mutation):
    root = completed.root
    paths = audit.artifact_paths("clock")
    record_path = root / paths["evaluate_record"]
    record = json.loads(record_path.read_text())
    if mutation == "status":
        record["status"] = "failed"
    elif mutation == "error_field":
        record["error"] = "previous failure"
    elif mutation == "release":
        record["release_sha256"] = "0" * 64
    elif mutation == "cumulative":
        record["predecessor_sha256"].pop(next(iter(record["predecessor_sha256"])))
    elif mutation == "initial":
        record["initial_policy"]["sha256"] = "0" * 64
    elif mutation == "chronology":
        record["started_utc"] = "2026-01-02T00:00:00+00:00"
    elif mutation == "test_time":
        record["tests"]["completed_utc"] = "2026-01-03T00:00:00+00:00"
    elif mutation == "test_count":
        record["tests"]["tests_passed"] = 1128
    elif mutation == "test_command":
        record["tests"]["commands"][1]["command"].append("only_one_test.py")
    elif mutation == "test_exit_bool":
        record["tests"]["commands"][0]["exit_code"] = False
    elif mutation == "summary":
        summary = json.loads((root / paths["evaluation_summary"]).read_text())
        summary["first_seed"] += 1
        write_json(root / paths["evaluation_summary"], summary)
        record["summary_sha256"] = audit.sha(root / paths["evaluation_summary"])
    elif mutation == "model":
        (root / paths["model"]).write_bytes(b"model changed")
    elif mutation == "csv":
        (root / paths["evaluation_csv"]).write_bytes(b"CSV changed")
    elif mutation == "log_missing":
        (root / audit.stage_outputs("clock", "evaluate")[-1]).unlink()
    elif mutation == "batch_lock":
        write_json(root / audit.LOCK_PATH, {"status": "failed"})
    elif mutation in ("truncated_completion", "wrong_completion"):
        stdout = root / audit.stage_outputs("clock", "evaluate")[-2]
        text = stdout.read_text()
        text = (text[:-15] if mutation == "truncated_completion"
                else text.replace("complete:", "failed:"))
        stdout.write_text(text, encoding="utf-8")
    write_json(record_path, record)
    if mutation == "record_missing":
        record_path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.completed_chain(root, completed.release, RELEASE_HASH)
    assert completed.calls == []


@pytest.mark.parametrize("ttc,unsafe", [(2.0, 0), (0.0, 0), (3.0, 1), (math.inf, 1)])
def test_episode_ttc_and_unsafe_count_must_describe_the_same_step_measurements(ttc, unsafe):
    rows = rows500()
    rows[0].update(min_ttc=ttc, unsafe_ttc_events=unsafe)
    with pytest.raises(ValueError):
        audit.metrics(rows)


def test_checkpoint_inspection_cannot_mutate_already_verified_evidence(completed, monkeypatch):
    def tampering_inspection(path, arm, provenance, expected_hash):
        path.write_bytes(b"synthetic checkpoint changed during inspection")

    monkeypatch.setattr(audit, "inspect_model", tampering_inspection)
    with pytest.raises(ValueError, match="Evidence bytes changed"):
        audit.completed_chain(completed.root, completed.release, RELEASE_HASH)


def fabricated_model(arm="zero"):
    import gymnasium as gym
    import numpy as np
    import torch

    state = {}
    for branch in ("policy_net", "value_net"):
        for layer, inputs in ((0, 116), (2, 256)):
            state[f"mlp_extractor.{branch}.{layer}.weight"] = torch.zeros((256, inputs))
            state[f"mlp_extractor.{branch}.{layer}.bias"] = torch.zeros(256)
    for head, outputs in (("action_net", 3), ("value_net", 1)):
        state[f"{head}.weight"] = torch.zeros((outputs, 256))
        state[f"{head}.bias"] = torch.zeros(outputs)
    model = SimpleNamespace(
        num_timesteps=200704, seed=42, n_steps=1024, batch_size=64, n_envs=1, n_epochs=10,
        learning_rate=.0003, gamma=.99, gae_lambda=.95, ent_coef=.01, vf_coef=.5,
        max_grad_norm=.5, target_kl=None, normalize_advantage=True, clip_range_vf=None,
        policy_kwargs={"net_arch": [256, 256]}, _n_updates=1960,
        safeintent_clock_arm=arm, safeintent_clock_duration=30,
        safeintent_observation_protocol=audit.ARMS[arm]["protocol"],
        safeintent_initial_policy=copy.deepcopy(audit.INITIAL_POLICY),
        safeintent_clock_provenance=fabricated_provenance(), device="cpu", clip_range=lambda x: .2,
        observation_space=gym.spaces.Box(
            np.asarray([-np.inf] * 105 + [0] + [-1, 0, 0] * 3 + [0], dtype=np.float32),
            np.asarray([np.inf] * 105 + [1] * 11, dtype=np.float32), dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(3),
        policy=SimpleNamespace(state_dict=lambda: state, parameters=lambda: list(state.values())),
    )
    return model, state


@pytest.mark.parametrize("arm", ["zero", "clock"])
def test_independent_checkpoint_structure_accepts_fabricated_zero_tensors_without_ppo(arm):
    model, state = fabricated_model(arm)
    assert len(state) == 12 and sum(t.numel() for t in state.values()) == 192516
    audit.verify_model(model, arm, fabricated_provenance())


@pytest.mark.parametrize("mutation", ["missing_tensor", "wrong_width", "dtype", "nonfinite",
                                      "bounds", "seed_bool", "initial_hash", "provenance",
                                      "timesteps", "device"])
def test_independent_checkpoint_structure_rejects_fabricated_invalid_models(mutation):
    import torch

    model, state = fabricated_model()
    name = "mlp_extractor.policy_net.0.weight"
    if mutation == "missing_tensor":
        state.pop(name)
    elif mutation == "wrong_width":
        state[name] = torch.zeros((256, 115))
    elif mutation == "dtype":
        state[name] = state[name].double()
    elif mutation == "nonfinite":
        state[name][0, 0] = math.nan
    elif mutation == "bounds":
        model.observation_space.high[-1] = 2
    elif mutation == "seed_bool":
        model.n_envs = True
    elif mutation == "initial_hash":
        model.safeintent_initial_policy["sha256"] = "0" * 64
    elif mutation == "provenance":
        model.safeintent_clock_provenance["release_sha256"] = "0" * 64
    elif mutation == "timesteps":
        model.num_timesteps = 200000
    else:
        model.device = "cuda"
    with pytest.raises(ValueError):
        audit.verify_model(model, "zero", fabricated_provenance())


@pytest.mark.parametrize("mutation", [None, "before_load", "during_load"])
def test_inspection_loads_only_preverified_bound_payload_and_rechecks_live_file(
        tmp_path, monkeypatch, mutation):
    import io

    from stable_baselines3 import PPO

    path = tmp_path / "synthetic-model.zip"
    payload = b"opaque synthetic bytes; never a valid model archive"
    path.write_bytes(payload)
    expected_hash = audit.sha(path)
    loaded, verified = [], []
    model = SimpleNamespace()

    def load(stream, **kwargs):
        assert isinstance(stream, io.BytesIO) and stream.getvalue() == payload
        assert kwargs == {"device": "cpu"}
        loaded.append(True)
        if mutation == "during_load":
            path.write_bytes(b"changed while mocked loader runs")
        return model

    monkeypatch.setattr(PPO, "load", load)
    monkeypatch.setattr(audit, "verify_model", lambda *args: verified.append(args))
    if mutation == "before_load":
        path.write_bytes(b"unbound replacement before inspection")
    if mutation is None:
        ORIGINAL_INSPECT_MODEL(path, "zero", fabricated_provenance(), expected_hash)
        assert verified == [(model, "zero", fabricated_provenance())]
    else:
        with pytest.raises(ValueError):
            ORIGINAL_INSPECT_MODEL(path, "zero", fabricated_provenance(), expected_hash)
    assert len(loaded) == int(mutation != "before_load")


@pytest.fixture
def full_audit(tmp_path, monkeypatch):
    config = tmp_path / audit.CONFIG
    config.parent.mkdir(parents=True)
    config.write_bytes(b"synthetic configuration bytes, not an executable environment")
    config_hash = audit.sha(config)
    # The release is mocked; bind this deliberately fabricated configuration's actual bytes.
    # Separate direct summary tests use the unchanged real preregistered config digest.
    monkeypatch.setattr(audit, "CONFIG_SHA256", config_hash)
    pins = {audit.CONFIG: config_hash}
    recorded_only = {
        "original": "models/ppo_reward_v3_seed42.zip",
        "control": "models/ppo_v3_predictive_control_seed42.zip",
        "v1": "models/ppo_v3_predictive_safety_v1_seed42.zip",
    }
    summaries = {}
    for name, stem in audit.REFERENCES.items():
        csv_path = tmp_path / f"results/{stem}.csv"
        summary_path = csv_path.with_suffix(".summary.json")
        model_path = recorded_only.get(name, f"models/synthetic_{name}.zip")
        model_hash = "c" * 64
        if name not in recorded_only:
            target = tmp_path / model_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"synthetic retained {name} checkpoint bytes".encode())
            model_hash = audit.sha(target)
            pins[model_path] = model_hash
        summary = fabricated_summary(model_hash=model_hash, config_hash=config_hash)
        summary["model_path"] = model_path
        write_rows(csv_path, rows500())
        write_json(summary_path, summary)
        pins[csv_path.relative_to(tmp_path).as_posix()] = audit.sha(csv_path)
        pins[summary_path.relative_to(tmp_path).as_posix()] = audit.sha(summary_path)
        summaries[name] = summary_path
    write_json(tmp_path / audit.RELEASE_PATH, {"synthetic": "release binding stand-in"})
    write_json(tmp_path / audit.VERIFICATION_PATH, {"synthetic": "proof binding stand-in"})
    digest = audit.sha(tmp_path / audit.RELEASE_PATH)
    release = {"input_sha256": pins, "runtime": {"frozen": True},
               "created_utc": "2026-01-01T00:00:00+00:00",
               "validation": {"tests_passed": 1129,
                              "verification_sha256": audit.sha(tmp_path / audit.VERIFICATION_PATH)}}
    fixture = build_completed(tmp_path, monkeypatch, release=release, digest=digest)
    monkeypatch.setattr(audit, "verify_release", lambda *args: copy.deepcopy(release))
    fixture.summaries = summaries
    fixture.recorded_only = recorded_only
    return fixture


def test_full_synthetic_audit_checks_real_chain_statistics_and_recorded_only_old_models(full_audit):
    fixture = full_audit
    before = {str(path): path.read_bytes() for path in fixture.root.rglob("*") if path.is_file()}
    result = audit.audit(fixture.root, fixture.digest)
    assert result["status"] == "verified" and result["automatic_promotion"] is False
    assert result["clock_benefit_established_under_frozen_gates"] is False
    assert len(result["evidence"]) == 9 and len(result["completed_stages"]) == 4
    assert len(fixture.calls) == 2
    for name, recorded_path in fixture.recorded_only.items():
        identity = result["evidence"][name]["artifact_identity"]["model"]
        assert identity["path"] == recorded_path and identity["live_byte_verified"] is False
        assert not (fixture.root / recorded_path).exists()
    for name in ("geometry", "synchronized_retry", "padding", "sector", "zero", "clock"):
        assert result["evidence"][name]["artifact_identity"]["model"]["live_byte_verified"] is True
    assert all(len(gates) == 16 for gates in result["historical_gates"].values())
    assert result["failed_clock_benefit_gates"] == [
        "all_historical_gates", "paired_success_vs_zero",
    ]
    after = {str(path): path.read_bytes() for path in fixture.root.rglob("*") if path.is_file()}
    assert after == before
    assert not (fixture.root / audit.OUTPUT).exists()


@pytest.mark.parametrize("mutation", ["missing_retained_model_pin", "wrong_recorded_path",
                                      "unbound_csv", "summary_seed", "config_changed"])
def test_full_audit_rejects_unbound_or_inconsistent_evidence_without_writing(full_audit, mutation):
    fixture = full_audit
    if mutation == "missing_retained_model_pin":
        fixture.release["input_sha256"].pop("models/synthetic_geometry.zip")
    elif mutation == "unbound_csv":
        fixture.release["input_sha256"].pop(f"results/{audit.REFERENCES['original']}.csv")
    elif mutation == "config_changed":
        (fixture.root / audit.CONFIG).write_bytes(b"changed configuration bytes")
    else:
        path = fixture.summaries["original"]
        summary = json.loads(path.read_text())
        if mutation == "wrong_recorded_path":
            summary["model_path"] = "models/unregistered_old_model.zip"
        else:
            summary["first_seed"] += 1
        write_json(path, summary)
        fixture.release["input_sha256"][path.relative_to(fixture.root).as_posix()] = audit.sha(path)
    with pytest.raises(ValueError):
        audit.audit(fixture.root, fixture.digest)
    assert not (fixture.root / audit.OUTPUT).exists()


def test_cli_requires_hash_and_preservation_guard_without_output_or_protocol_overrides():
    for arguments in ([], ["--release-sha256", RELEASE_HASH], ["--refuse-overwrite"],
                      ["--release-sha256", "A" * 64, "--refuse-overwrite"],
                      ["--release-sha256", RELEASE_HASH, "--refuse-overwrite",
                       "--output", "elsewhere"],
                      ["--release-sha256", RELEASE_HASH, "--refuse-overwrite", "--seed", "7"]):
        with pytest.raises(SystemExit) as caught:
            audit.parse_args(arguments)
        assert caught.value.code == 2
    args = audit.parse_args(["--release-sha256", RELEASE_HASH, "--refuse-overwrite"])
    assert args.release_sha256 == RELEASE_HASH and args.refuse_overwrite is True


def cli_setup(tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(tmp_path)
    output = tmp_path / audit.OUTPUT
    output.parent.mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["clock-audit", "--release-sha256", RELEASE_HASH,
                                      "--refuse-overwrite"])
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    return output


def test_cli_refuses_existing_report_before_audit_or_model_work(tmp_path, monkeypatch):
    output = cli_setup(tmp_path, monkeypatch)
    output.write_bytes(b"preserve existing audit")
    monkeypatch.setattr(audit, "audit", forbidden)
    with pytest.raises(FileExistsError):
        audit.main()
    assert output.read_bytes() == b"preserve existing audit"


def test_cli_exclusive_creation_preserves_late_created_report(tmp_path, monkeypatch):
    output = cli_setup(tmp_path, monkeypatch)

    def raced_audit(*args):
        output.write_bytes(b"late external audit report")
        return {"status": "verified"}

    monkeypatch.setattr(audit, "audit", raced_audit)
    with pytest.raises(FileExistsError):
        audit.main()
    assert output.read_bytes() == b"late external audit report"


@pytest.mark.parametrize("failure", ["audit", "nonfinite_report"])
def test_cli_failure_creates_no_partial_report(tmp_path, monkeypatch, failure):
    output = cli_setup(tmp_path, monkeypatch)

    def failed_audit(*args):
        if failure == "audit":
            raise ValueError("synthetic audit verification failed")
        return {"mean_reward": math.nan}

    monkeypatch.setattr(audit, "audit", failed_audit)
    with pytest.raises(ValueError):
        audit.main()
    assert not output.exists()


def test_cli_writes_only_reserved_audit_after_success(tmp_path, monkeypatch, capsys):
    output = cli_setup(tmp_path, monkeypatch)
    calls = []

    def verified(root, digest):
        calls.append((root, digest))
        return {"status": "verified", "automatic_promotion": False}

    monkeypatch.setattr(audit, "audit", verified)
    audit.main()
    assert calls == [(tmp_path.resolve(), RELEASE_HASH)]
    assert json.loads(output.read_text()) == {"status": "verified", "automatic_promotion": False}
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [output]
    assert "Saved independent clock audit" in capsys.readouterr().out


@pytest.fixture
def fabricated_inventory(tmp_path, monkeypatch):
    import torch

    from scripts import sector_timeout_protocol_v1 as retained

    digest = "b" * 64
    pins = {f"scripts/historical_{index}.py": digest for index in range(95)}
    pins.update({f"results/opaque_{index}.json": digest for index in range(436)})
    for name in [*list(pins)[:95], *audit.NEW_SOURCES]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic inventory file\n", encoding="utf-8")
    runtime = {"package_origins": "synthetic frozen origins"}
    dependencies = {f"synthetic_dependency_{index}": digest for index in range(227)}
    old_release = {"runtime": copy.deepcopy(runtime),
                   "dependency_sha256": copy.deepcopy(dependencies)}

    def read(path, **kwargs):
        if Path(path).as_posix().endswith(audit.ANALYSIS):
            return {"input_sha256": pins}
        return old_release

    monkeypatch.setattr(audit, "PINNED", {audit.ANALYSIS: digest})
    monkeypatch.setattr(audit, "sha", lambda path: digest)
    monkeypatch.setattr(audit, "read_json", read)
    monkeypatch.setattr(retained, "verify_snapshot",
                        lambda root: {"archive_fingerprint": {"sha256": digest}})
    monkeypatch.setattr(retained, "verify_runtime", lambda root: runtime)
    monkeypatch.setattr(retained, "dependency_fingerprints", lambda: dependencies)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 8)
    for name, module in tuple(sys.modules.items()):
        if module is not None and (name == "safeintent_rl" or
                                   name.startswith(("safeintent_rl.", "scripts."))):
            suffix = "/__init__.py" if hasattr(module, "__path__") else ".py"
            source = tmp_path / (name.replace(".", "/") + suffix)
            monkeypatch.setattr(module, "__file__", str(source))
    return SimpleNamespace(root=tmp_path, pins=pins, runtime=runtime,
                           dependencies=dependencies, retained=retained)


def test_direct_inventory_accepts_exact_synthetic_95_plus_6_sources_and_227_dependencies(
        fabricated_inventory):
    fixture = fabricated_inventory
    result = audit.current_fingerprints(fixture.root)
    assert result["runtime"] == fixture.runtime
    assert result["dependency_sha256"] == fixture.dependencies
    assert len(result["dependency_sha256"]) == 227
    assert audit.NEW_SOURCES <= result["input_sha256"].keys()
    assert all(result["input_sha256"][name] == value for name, value in fixture.pins.items())


@pytest.mark.parametrize("mutation", ["missing_source", "extra_source", "source_drift",
                                      "origin", "dependency_count", "dependency_hash", "runtime",
                                      "threads", "interop"])
def test_direct_inventory_rejects_source_origin_dependency_or_runtime_changes(
        fabricated_inventory, monkeypatch, mutation):
    import torch

    fixture = fabricated_inventory
    if mutation == "missing_source":
        (fixture.root / "scripts/audit_clock_comparison_v1.py").unlink()
    elif mutation == "extra_source":
        (fixture.root / "scripts/unregistered.py").write_text("# extra\n", encoding="utf-8")
    elif mutation == "source_drift":
        fixture.pins["scripts/historical_0.py"] = "0" * 64
    elif mutation == "origin":
        monkeypatch.setattr(audit, "__file__", str(fixture.root / "wrong/auditor.py"))
    elif mutation == "dependency_count":
        fixture.dependencies.pop("synthetic_dependency_0")
    elif mutation == "dependency_hash":
        fixture.dependencies["synthetic_dependency_0"] = "0" * 64
    elif mutation == "runtime":
        fixture.runtime["package_origins"] = "changed origin"
    elif mutation == "threads":
        monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    else:
        monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    with pytest.raises(ValueError):
        audit.current_fingerprints(fixture.root)
