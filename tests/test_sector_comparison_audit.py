import copy
import json
import math
import sys

import pytest

from scripts import audit_sector_comparison_v1 as audit
from scripts.audit_geometry_development import paired
from scripts.audit_synchronized_retry import retention_gates


def historical_gates():
    return retention_gates(
        {"success": 394, "collision": 98, "incomplete": 8}, 0.6728939419963959,
        {name: {"favorable": True} for name in ("original", "control", "v1")},
    )


def evidence():
    return {"counts": {"success": 394, "collision": 98, "incomplete": 8,
                       "excluded_nonfinite_ttc": 0},
            "metrics": {"mean_min_ttc": 0.7}}


def favorable_comparison():
    return paired([{"success": False}] * 6, [{"success": True}] * 6)


def test_feature_benefit_uses_exactly_five_additional_registered_conditions():
    padding, sector, gates = evidence(), evidence(), historical_gates()
    before = copy.deepcopy((padding, sector, gates))
    actual = audit.feature_benefit_gates(gates, padding, sector, favorable_comparison())
    assert actual == {
        "all_historical_gates": True, "paired_success_vs_padding": True,
        "collision_vs_padding": True, "incomplete_vs_padding": True, "ttc_vs_padding": True,
    }
    assert (padding, sector, gates) == before


@pytest.mark.parametrize("failed", list(historical_gates()))
def test_each_historical_gate_is_required_even_when_sector_beats_padding(failed):
    gates = historical_gates()
    gates[failed] = False
    actual = audit.feature_benefit_gates(gates, evidence(), evidence(), favorable_comparison())
    assert [key for key, value in actual.items() if not value] == ["all_historical_gates"]


@pytest.mark.parametrize("field,key,change", [
    ("collision", "collision_vs_padding", 1),
    ("incomplete", "incomplete_vs_padding", 1),
    ("mean_min_ttc", "ttc_vs_padding", -1),
])
def test_feature_nonregression_limits_accept_equality_but_reject_smallest_regression(
        field, key, change):
    padding, sector = evidence(), evidence()
    assert audit.feature_benefit_gates(
        historical_gates(), padding, sector, favorable_comparison(),
    )[key]
    if change < 0:
        sector["metrics"][field] = math.nextafter(padding["metrics"][field], -math.inf)
    else:
        sector["counts"][field] += change
    actual = audit.feature_benefit_gates(
        historical_gates(), padding, sector, favorable_comparison(),
    )
    assert [name for name, value in actual.items() if not value] == [key]


@pytest.mark.parametrize("rescues,regressions,expected", [
    (6, 0, True), (5, 0, False), (0, 6, False), (6, 6, False), (0, 0, False),
])
def test_sector_benefit_requires_favorable_exact_paired_success(rescues, regressions, expected):
    reference = [{"success": False}] * rescues + [{"success": True}] * regressions
    candidate = [{"success": True}] * rescues + [{"success": False}] * regressions
    comparison = paired(reference, candidate)
    actual = audit.feature_benefit_gates(historical_gates(), evidence(), evidence(), comparison)
    assert actual["paired_success_vs_padding"] is expected


@pytest.mark.parametrize("incomplete", [8, 9, 10, 11])
def test_historical_incomplete_ten_and_geometry_eight_limits_remain_distinct(incomplete):
    counts = {"success": 394, "collision": 106 - incomplete, "incomplete": incomplete}
    gates = retention_gates(counts, 0.7,
                            {name: {"favorable": True} for name in ("original", "control", "v1")})
    assert gates["incomplete"] is (incomplete <= 10)
    assert gates["geometry_incomplete"] is (incomplete <= 8)
    actual = audit.feature_benefit_gates(gates, evidence(), evidence(), favorable_comparison())
    assert actual["all_historical_gates"] is (incomplete == 8)


@pytest.fixture
def reconciliation_fixture(monkeypatch, tmp_path):
    stems = {"padding": "padding_fixture", "sector": "sector_fixture"}
    rows, summaries, source_bytes, reads = {}, {}, {}, []
    (tmp_path / "results").mkdir()
    for name, stem in stems.items():
        row = {"reward": 1.0, "length": 2, "success": True, "collision": False,
               "travel_time": .4, "min_ttc": 1.0, "unsafe_ttc_events": 0,
               "safety_interventions": 0}
        rows[name] = [dict(row) for _ in range(500)]
        computed, _ = audit.metrics(rows[name])
        summaries[name] = {
            **computed, "first_seed": 40042, "last_seed": 40541,
            "unsafe_ttc_threshold": 2.0, "safety_shield": False,
        }
        csv_path = tmp_path / "results" / f"{stem}.csv"
        csv_path.write_bytes(b"Immutable engineering CSV placeholder; reader is mocked")
        summary_path = csv_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summaries[name]), encoding="utf-8")
        for path in (csv_path, summary_path):
            source_bytes[path] = path.read_bytes()

    def read_rows(path):
        reads.append(path)
        name = next(key for key, value in stems.items() if path.stem == value)
        return copy.deepcopy(rows[name])

    monkeypatch.setattr(audit, "read_rows", read_rows)
    return {"root": tmp_path, "stems": stems, "rows": rows, "summaries": summaries,
            "source_bytes": source_bytes, "reads": reads}


def rewrite_summary(fixture, name, field, value):
    fixture["summaries"][name][field] = value
    path = fixture["root"] / "results" / f"{fixture['stems'][name]}.summary.json"
    path.write_text(json.dumps(fixture["summaries"][name]), encoding="utf-8")


def test_reconciled_evidence_returns_independent_metrics_counts_and_unchanged_hashes(
        reconciliation_fixture):
    fixture = reconciliation_fixture
    rows, summaries, measured = audit.reconciled_evidence(fixture["root"], fixture["stems"])
    assert rows == fixture["rows"]
    assert summaries == fixture["summaries"]
    assert len(fixture["reads"]) == 2
    for name, stem in fixture["stems"].items():
        assert measured[name]["metrics"] == audit.metrics(fixture["rows"][name])[0]
        assert measured[name]["counts"] == {
            "success": 500, "collision": 0, "incomplete": 0, "excluded_nonfinite_ttc": 0,
        }
        csv_path = fixture["root"] / "results" / f"{stem}.csv"
        assert measured[name]["csv_sha256"] == audit.sha(csv_path)
        assert measured[name]["summary_sha256"] == audit.sha(csv_path.with_suffix(".summary.json"))
    assert all(path.read_bytes() == contents for path, contents in fixture["source_bytes"].items())


@pytest.mark.parametrize("field,value", [
    ("episodes", 499), ("mean_reward", 1.1), ("success_rate", .999),
    ("collision_rate", .01), ("mean_min_ttc", 1.01), ("mean_length", 2.1),
    ("mean_travel_time", .5), ("mean_safety_interventions", 1),
    ("first_seed", 40043), ("last_seed", 40542),
    ("unsafe_ttc_threshold", 3), ("safety_shield", True),
])
def test_reconciliation_rejects_wrong_metrics_or_evaluation_metadata(
        reconciliation_fixture, field, value):
    fixture = reconciliation_fixture
    rewrite_summary(fixture, "sector", field, value)
    before = {path: path.read_bytes() for path in fixture["source_bytes"]}
    with pytest.raises(ValueError):
        audit.reconciled_evidence(fixture["root"], fixture["stems"])
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("difference,accepted", [(5e-13, True), (1e-9, False)])
def test_reconciliation_uses_declared_float_tolerance(reconciliation_fixture, difference, accepted):
    fixture = reconciliation_fixture
    rewrite_summary(fixture, "sector", "mean_reward", 1.0 + difference)
    if accepted:
        _, summaries, measured = audit.reconciled_evidence(fixture["root"], fixture["stems"])
        assert summaries["sector"]["mean_reward"] == 1.0 + difference
        assert measured["sector"]["metrics"]["mean_reward"] == 1.0
    else:
        with pytest.raises(ValueError):
            audit.reconciled_evidence(fixture["root"], fixture["stems"])


def test_reconciliation_reports_nonfinite_ttc_exclusions_without_removing_rows(
        reconciliation_fixture):
    fixture = reconciliation_fixture
    fixture["rows"]["sector"][0]["min_ttc"] = math.inf
    rows, _, measured = audit.reconciled_evidence(fixture["root"], fixture["stems"])
    assert len(rows["sector"]) == 500
    assert math.isinf(rows["sector"][0]["min_ttc"])
    assert measured["sector"]["counts"]["excluded_nonfinite_ttc"] == 1
    assert measured["sector"]["metrics"]["mean_min_ttc"] == 1.0


def test_reconciliation_propagates_strict_raw_reader_failure(monkeypatch, reconciliation_fixture):
    def fail(path):
        raise ValueError("CSV columns differ")

    monkeypatch.setattr(audit, "read_rows", fail)
    fixture = reconciliation_fixture
    with pytest.raises(ValueError, match="CSV columns differ"):
        audit.reconciled_evidence(fixture["root"], fixture["stems"])


def geometry_summary():
    return {
        "episodes": 500.0, "mean_reward": 1.0, "mean_length": 2.0,
        "success_rate": .788, "collision_rate": .196, "mean_travel_time": .4,
        "mean_min_ttc": .6666153731858254, "mean_unsafe_ttc_events": 0.0,
        "mean_safety_interventions": 0.0, "first_seed": 40042, "last_seed": 40541,
        "unsafe_ttc_threshold": 2.0, "safety_shield": False,
        "model_path": "models/geometry.zip", "model_sha256": "geometry-model-sha",
        "config_path": "configs/geometry.yaml", "config_sha256": "geometry-config-sha",
        "reference_csv_path": "results/original.csv", "reference_csv_sha256": "reference-sha",
        "risk_fusion": False, "intent_model_path": None,
        "predictive_safety_observation": {"horizon": 3.0, "max_neighbors": 14},
    }


def arm_summary(arm):
    return {
        **geometry_summary(), "mean_reward": 2.0,
        "model_path": f"models/{audit.ARMS[arm]}.zip", "model_sha256": "candidate-model-sha",
        "observation_protocol": audit.PROTOCOLS[arm], "sector_comparison_arm": arm,
        "initial_policy": dict(audit.INITIAL),
    }


@pytest.mark.parametrize("arm", ["padding", "sector"])
def test_checked_summary_permits_only_registered_arm_and_metric_changes_without_mutation(arm):
    geometry, candidate = geometry_summary(), arm_summary(arm)
    before = copy.deepcopy((geometry, candidate))
    audit.checked_summary(candidate, geometry, arm, "candidate-model-sha")
    assert (geometry, candidate) == before


@pytest.mark.parametrize("arm", ["padding", "sector"])
def test_checked_summary_normalizes_only_three_known_path_fields(arm):
    geometry, candidate = geometry_summary(), arm_summary(arm)
    for key in ("model_path", "config_path", "reference_csv_path"):
        candidate[key] = candidate[key].replace("/", "\\")
    before = copy.deepcopy(candidate)
    audit.checked_summary(candidate, geometry, arm, "candidate-model-sha")
    assert candidate == before
    candidate["config_sha256"] = "geometry/config/sha"
    geometry["config_sha256"] = "geometry\\config\\sha"
    with pytest.raises(ValueError, match="Evaluation settings differ"):
        audit.checked_summary(candidate, geometry, arm, "candidate-model-sha")


@pytest.mark.parametrize("field,value", [
    ("first_seed", 40043), ("last_seed", 40540), ("unsafe_ttc_threshold", 1.5),
    ("safety_shield", True), ("risk_fusion", True), ("intent_model_path", "models/intent.pt"),
    ("config_path", "configs/changed.yaml"), ("config_sha256", "changed"),
    ("reference_csv_path", "results/changed.csv"), ("reference_csv_sha256", "changed"),
    ("model_path", "models/callback_best.zip"), ("model_sha256", "wrong-model-sha"),
    ("observation_protocol", "predictive_post_spawn_sync_v1"),
    ("sector_comparison_arm", "padding"),
    ("initial_policy", {"sha256": "0" * 64, "parameter_count": 216580}),
    ("initial_policy", {**audit.INITIAL, "parameter_count": 1}),
    ("predictive_safety_observation", {"horizon": 5.0, "max_neighbors": 14}),
])
def test_checked_summary_rejects_wrong_fixed_metadata(field, value):
    candidate = arm_summary("sector")
    candidate[field] = value
    with pytest.raises(ValueError, match="Evaluation settings differ"):
        audit.checked_summary(candidate, geometry_summary(), "sector", "candidate-model-sha")


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_checked_summary_rejects_missing_or_extra_metadata_keys(change):
    candidate = arm_summary("padding")
    if change == "missing":
        candidate.pop("intent_model_path")
    else:
        candidate["unregistered_override"] = False
    with pytest.raises(ValueError, match="Evaluation settings differ"):
        audit.checked_summary(candidate, geometry_summary(), "padding", "candidate-model-sha")


def test_checked_summary_rejects_unknown_arm():
    with pytest.raises(ValueError, match="Unknown comparison arm"):
        audit.checked_summary(arm_summary("sector"), geometry_summary(), "new-arm", "hash")


@pytest.fixture
def completed_chain_fixture(monkeypatch, tmp_path):
    sources, dependencies, record_paths, records = {}, {}, [], []
    for index in range(56):
        path = tmp_path / "sources" / f"frozen_{index}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source {index}", encoding="utf-8")
        sources[path.relative_to(tmp_path).as_posix()] = audit.sha(path)
    for index, (arm, mode) in enumerate(audit.ORDER):
        model_path = f"models/{audit.ARMS[arm]}.zip"
        model = tmp_path / model_path
        model.parent.mkdir(parents=True, exist_ok=True)
        if not model.exists():
            model.write_bytes(f"engineering {arm} model fixture".encode())
        stem = audit.ARMS[arm] if mode == "train" else audit.EVALUATIONS[arm]
        suffix = "training" if mode == "train" else "summary"
        summary_path = f"results/{stem}.{suffix}.json"
        summary = tmp_path / summary_path
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"model_sha256": audit.sha(model),
                                       "initial_policy": audit.INITIAL}), encoding="utf-8")
        record = {
            "status": "complete", "arm": arm, "mode": mode,
            "observation_protocol": audit.PROTOCOLS[arm], "initial_policy": dict(audit.INITIAL),
            "source_sha256": sources.copy(), "predecessor_sha256": dependencies.copy(),
            "started_utc": f"2026-09-12T0{index}:00:00+00:00",
            "finished_utc": f"2026-09-12T0{index}:01:00+00:00",
            "model_sha256": audit.sha(model), "summary_sha256": audit.sha(summary),
        }
        if mode == "evaluate":
            csv_path = f"results/{stem}.csv"
            (tmp_path / csv_path).write_bytes(b"engineering CSV fixture")
            record["csv_sha256"] = audit.sha(tmp_path / csv_path)
        record_path = f"results/{audit.ARMS[arm]}.{mode}.run.json"
        (tmp_path / record_path).write_text(json.dumps(record), encoding="utf-8")
        dependencies[record_path] = audit.sha(tmp_path / record_path)
        dependencies[summary_path] = audit.sha(summary)
        dependencies[model_path if mode == "train" else csv_path] = (
            record["model_sha256"] if mode == "train" else record["csv_sha256"]
        )
        record_paths.append(tmp_path / record_path)
        records.append(record)
    monkeypatch.setattr(audit, "RECORD_HASHES", tuple(audit.sha(path) for path in record_paths))
    monkeypatch.setattr(audit, "AUDIT_DEPENDENCIES", {})
    return {"root": tmp_path, "sources": sources, "dependencies": dependencies,
            "record_paths": record_paths, "records": records}


def test_completed_chain_verifies_all_four_stages_sources_and_artifacts(completed_chain_fixture):
    fixture = completed_chain_fixture
    records, fingerprints = audit.completed_chain(fixture["root"])
    assert list(records) == [f"{arm}_{mode}" for arm, mode in audit.ORDER]
    assert len(records) == 4
    assert fingerprints == {**fixture["sources"], **fixture["dependencies"]}


@pytest.mark.parametrize("index,field,value", [
    (0, "status", "running"), (0, "arm", "sector"), (0, "mode", "evaluate"),
    (0, "initial_policy", {"sha256": "0" * 64, "parameter_count": 216580}),
    (0, "predecessor_sha256", {"unregistered": "sha"}),
    (0, "finished_utc", "2026-09-11T23:59:59+00:00"),
    (1, "started_utc", "2026-09-12T00:00:00+00:00"),
    (1, "source_sha256", {}),
])
def test_completed_chain_rejects_incomplete_wrong_or_overlapping_stage_records(
        monkeypatch, completed_chain_fixture, index, field, value):
    fixture = completed_chain_fixture
    fixture["records"][index][field] = value
    fixture["record_paths"][index].write_text(
        json.dumps(fixture["records"][index]), encoding="utf-8",
    )
    monkeypatch.setattr(audit, "RECORD_HASHES",
                        tuple(audit.sha(path) for path in fixture["record_paths"]))
    with pytest.raises(ValueError):
        audit.completed_chain(fixture["root"])


def test_completed_chain_rejects_changed_frozen_file(completed_chain_fixture):
    fixture = completed_chain_fixture
    path = fixture["root"] / next(iter(fixture["sources"]))
    path.write_bytes(b"changed frozen input")
    with pytest.raises(ValueError, match="Frozen hash mismatch"):
        audit.completed_chain(fixture["root"])
    assert path.read_bytes() == b"changed frozen input"


def test_completed_chain_rejects_added_package_source(completed_chain_fixture):
    path = completed_chain_fixture["root"] / "safeintent_rl" / "unregistered.py"
    path.parent.mkdir()
    path.write_bytes(b"# unregistered source")
    with pytest.raises(ValueError, match="Package source inventory changed"):
        audit.completed_chain(completed_chain_fixture["root"])


def printed_report():
    return {key: {} for key in (
        "evidence", "paired_success", "failed_historical_gates", "sector_vs_padding",
        "failed_feature_benefit_gates", "arm_decisions",
    )}


def test_main_refuses_existing_report_before_audit(monkeypatch, tmp_path):
    output = tmp_path / "existing.audit.json"
    output.write_bytes(b"preserved audit")

    def forbidden(root):
        pytest.fail("Existing report must be rejected before reading experiment artifacts")

    monkeypatch.setattr(audit, "audit", forbidden)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(ValueError, match="Preserve existing audit report"):
        audit.main()
    assert output.read_bytes() == b"preserved audit"


def test_main_writes_only_requested_new_report(monkeypatch, tmp_path, capsys):
    output = tmp_path / "new.audit.json"
    report = {**printed_report(), "status": "verified", "automatic_promotion": False}
    calls = []

    def fake_audit(root):
        calls.append(root)
        return report

    monkeypatch.setattr(audit, "audit", fake_audit)
    monkeypatch.setattr(sys, "argv", ["audit", "--root", str(tmp_path), "--output", str(output)])
    audit.main()
    assert calls == [tmp_path]
    assert json.loads(output.read_text()) == report
    assert output.read_bytes().endswith(b"\n")
    assert list(tmp_path.iterdir()) == [output]
    assert json.loads(capsys.readouterr().out) == printed_report()


def test_main_exclusive_create_preserves_a_report_created_during_audit(monkeypatch, tmp_path):
    output = tmp_path / "raced.audit.json"

    def fake_audit(root):
        output.write_bytes(b"preserved concurrent audit")
        return printed_report()

    monkeypatch.setattr(audit, "audit", fake_audit)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(FileExistsError):
        audit.main()
    assert output.read_bytes() == b"preserved concurrent audit"
