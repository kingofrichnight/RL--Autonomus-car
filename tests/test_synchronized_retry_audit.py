import json
import math
import sys

import pytest

from scripts import audit_synchronized_retry as audit
from scripts.audit_geometry_development import COLUMNS, gates


def favorable_comparisons():
    return {name: {"favorable": True} for name in ("original", "control", "v1")}


def baseline_counts():
    return {"success": 394, "collision": 98, "incomplete": 8}


def test_retention_adds_only_registered_geometry_count_gates():
    counts = baseline_counts()
    comparisons = favorable_comparisons()
    comparisons["geometry"] = {"favorable": False, "p_exact_two_sided": 1.0}
    ttc = 0.6728939419963959
    actual = audit.retention_gates(counts, ttc, comparisons)
    assert actual == {
        **gates(counts, ttc, comparisons), "geometry_success": True,
        "geometry_collision": True, "geometry_incomplete": True,
    }
    assert len(actual) == 16
    assert all(actual.values())
    assert counts == baseline_counts()
    assert not comparisons["geometry"]["favorable"]
    del comparisons["geometry"]
    assert audit.retention_gates(counts, ttc, comparisons) == actual


@pytest.mark.parametrize("field,key,limit,direction", [
    ("success", "absolute_success", 315, -1),
    ("collision", "absolute_collision", 174, 1),
    ("incomplete", "incomplete", 10, 1),
    ("success", "control_success", 304, -1),
    ("collision", "control_collision", 195, 1),
    ("success", "v1_success", 364, -1),
    ("collision", "v1_collision", 101, 1),
    ("success", "geometry_success", 394, -1),
    ("collision", "geometry_collision", 98, 1),
    ("incomplete", "geometry_incomplete", 8, 1),
])
def test_count_gate_limits_are_inclusive_and_not_relaxed(field, key, limit, direction):
    counts = baseline_counts()
    counts[field] = limit
    assert audit.retention_gates(counts, 1.0, favorable_comparisons())[key]
    counts[field] = limit + direction
    assert not audit.retention_gates(counts, 1.0, favorable_comparisons())[key]


@pytest.mark.parametrize("key,limit", [
    ("original_ttc", 0.6003656548142169),
    ("control_ttc", 0.6065401801078688),
    ("v1_ttc", 0.6728939419963959),
])
def test_ttc_gate_preserves_exact_unrounded_threshold(key, limit):
    counts = baseline_counts()
    comparisons = favorable_comparisons()
    assert audit.retention_gates(counts, limit, comparisons)[key]
    assert not audit.retention_gates(counts, math.nextafter(limit, -math.inf), comparisons)[key]


@pytest.mark.parametrize("name", ["original", "control", "v1"])
def test_historical_favorable_paired_success_gates_remain_required(name):
    comparisons = favorable_comparisons()
    comparisons[name]["favorable"] = False
    actual = audit.retention_gates(baseline_counts(), 1.0, comparisons)
    assert [key for key, passed in actual.items() if not passed] == [f"{name}_paired_success"]


def test_better_outcomes_do_not_override_failed_v1_ttc_gate():
    actual = audit.retention_gates(
        {"success": 450, "collision": 45, "incomplete": 5},
        0.6666153731858254, favorable_comparisons(),
    )
    assert [key for key, passed in actual.items() if not passed] == ["v1_ttc"]


def test_check_hashes_verifies_each_source_and_does_not_modify_files(tmp_path):
    paths = [tmp_path / "first.txt", tmp_path / "nested" / "second.txt"]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen research input {index}".encode())
    hashes = {path.relative_to(tmp_path).as_posix(): audit.sha(path) for path in paths}
    before = {path: path.read_bytes() for path in paths}
    audit.check_hashes(tmp_path, hashes)
    assert {path: path.read_bytes() for path in paths} == before
    paths[1].write_bytes(b"changed input")
    with pytest.raises(ValueError, match="Frozen hash mismatch: nested/second.txt"):
        audit.check_hashes(tmp_path, hashes)
    assert paths[0].read_bytes() == before[paths[0]]
    assert paths[1].read_bytes() == b"changed input"


def test_audit_stops_at_pinned_hash_mismatch_before_reading_records(monkeypatch, tmp_path):
    artifact = tmp_path / "source.txt"
    artifact.write_bytes(b"existing source, not JSON")
    monkeypatch.setattr(audit, "PINNED", {artifact.name: "0" * 64})
    with pytest.raises(ValueError, match="Frozen hash mismatch: source.txt"):
        audit.audit(tmp_path)
    assert artifact.read_bytes() == b"existing source, not JSON"
    assert list(tmp_path.iterdir()) == [artifact]


def report_fixture():
    return {"status": "verified", "evidence": {}, "paired_success": {},
            "failed_gates": ["v1_ttc"], "decision": "not_retained_under_frozen_gates"}


def test_main_preserves_existing_report_without_starting_audit(monkeypatch, tmp_path):
    output = tmp_path / "existing.audit.json"
    output.write_bytes(b"preserved prior audit")

    def forbidden_audit(root):
        pytest.fail("Existing report must be rejected before reading source artifacts")

    monkeypatch.setattr(audit, "audit", forbidden_audit)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(ValueError, match="Preserve existing audit report"):
        audit.main()
    assert output.read_bytes() == b"preserved prior audit"


def test_main_writes_only_requested_new_report(monkeypatch, tmp_path, capsys):
    output = tmp_path / "new.audit.json"
    source_root = tmp_path / "source"
    calls = []
    report = report_fixture()

    def fake_audit(root):
        calls.append(root)
        return report

    monkeypatch.setattr(audit, "audit", fake_audit)
    monkeypatch.setattr(sys, "argv", [
        "audit", "--root", str(source_root), "--output", str(output),
    ])
    audit.main()
    assert calls == [source_root]
    assert json.loads(output.read_text()) == report
    assert output.read_bytes().endswith(b"\n")
    assert list(tmp_path.iterdir()) == [output]
    printed = json.loads(capsys.readouterr().out)
    assert printed["decision"] == "not_retained_under_frozen_gates"
    assert printed["failed_gates"] == ["v1_ttc"]


def test_main_exclusive_create_preserves_report_created_during_audit(monkeypatch, tmp_path):
    output = tmp_path / "raced.audit.json"

    def fake_audit(root):
        output.write_bytes(b"report created by another process")
        return report_fixture()

    monkeypatch.setattr(audit, "audit", fake_audit)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(FileExistsError):
        audit.main()
    assert output.read_bytes() == b"report created by another process"


def test_main_does_not_create_report_when_audit_fails(monkeypatch, tmp_path):
    output = tmp_path / "failed.audit.json"

    def fail(root):
        raise ValueError("deliberate source discrepancy")

    monkeypatch.setattr(audit, "audit", fail)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(output)])
    with pytest.raises(ValueError, match="deliberate source discrepancy"):
        audit.main()
    assert not output.exists()


def test_inherited_metrics_reject_mean_ttc_without_any_finite_observations():
    row = dict(zip(COLUMNS, [1.0, 1, False, False, 0.2, math.inf, 0, 0]))
    with pytest.raises(ValueError, match="no finite observations"):
        audit.metrics([row])
