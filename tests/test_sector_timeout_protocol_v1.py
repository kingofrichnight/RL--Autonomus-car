"""Synthetic, read-only protocol checks. No simulator or research policy execution."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import sector_timeout_protocol_v1 as protocol


def episode(**changes):
    row = {"reward": 1.0, "length": 10, "success": True, "collision": False,
           "travel_time": 2.0, "min_ttc": 1.5, "unsafe_ttc_events": 2,
           "safety_interventions": 0}
    row.update(changes)
    return row


def references():
    rows = {arm: [episode() for _ in range(500)] for arm in protocol.ARMS}
    padding = protocol.SEEDS[:24]
    sector = protocol.SEEDS[:10] + protocol.SEEDS[24:]
    for arm, seeds in (("padding", padding), ("sector", sector)):
        for seed in seeds:
            rows[arm][seed - protocol.FIRST_SEED]["success"] = False
    return rows


def test_frozen_constants():
    assert tuple(protocol.ARMS) == ("geometry", "padding", "sector")
    assert [spec["shape"] for spec in protocol.ARMS.values()] == [115, 163, 163]
    assert protocol.ARMS["geometry"]["protocol"] is None
    assert len(protocol.SEEDS) == len(set(protocol.SEEDS)) == 37
    assert tuple(sorted(protocol.SEEDS)) == protocol.SEEDS
    assert protocol.SEEDS[0] == 40084 and protocol.SEEDS[-1] == 40541
    assert protocol.TOLERANCES == {
        "reward": {"abs_tol": 1e-9, "rel_tol": 1e-12},
        "min_ttc": {"abs_tol": 1e-9, "rel_tol": 1e-12},
        "travel_time": {"abs_tol": 1e-12, "rel_tol": 0.0},
    }


def test_exact_match_and_numeric_deltas():
    assert protocol.compare_episode(episode(), episode()) == {
        "length": 0, "unsafe_ttc_events": 0, "safety_interventions": 0,
        "reward": 0.0, "travel_time": 0.0, "min_ttc": 0.0,
    }
    actual = episode(reward=1 + 5e-10, min_ttc=1.5 - 5e-10, travel_time=2 + 5e-13)
    delta = protocol.compare_episode(actual, episode())
    assert delta["reward"] > 0 and delta["min_ttc"] < 0


@pytest.mark.parametrize("key,value", [
    ("success", 1), ("collision", 0), ("length", True), ("length", 10.0),
    ("unsafe_ttc_events", 2.0), ("safety_interventions", False),
    ("length", 0), ("length", 152), ("unsafe_ttc_events", -1),
    ("unsafe_ttc_events", 11), ("safety_interventions", 1),
    ("reward", True), ("reward", "1"), ("min_ttc", -0.1),
    ("collision", True), ("travel_time", 2 + 1e-8),
])
def test_invalid_metric_types_ranges_and_exclusivity(key, value):
    with pytest.raises(ValueError):
        protocol.compare_episode(episode(**{key: value}), episode())


@pytest.mark.parametrize("key", protocol.FLOAT_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_no_nonfinite_values_even_when_both_match(key, value):
    row = episode(**{key: value})
    with pytest.raises(ValueError, match="finite"):
        protocol.compare_episode(row, row)


@pytest.mark.parametrize("row", [dict(episode(), extra=0), {"reward": 1.0}, None])
def test_missing_extra_keys_rejected(row):
    with pytest.raises(ValueError, match="keys"):
        protocol.compare_episode(row, episode())


@pytest.mark.parametrize("key,value", [("reward", 1 + 2e-9), ("min_ttc", 1.5 + 2e-9),
                                      ("unsafe_ttc_events", 3), ("success", False)])
def test_valid_metrics_still_must_reproduce(key, value):
    with pytest.raises(ValueError, match="mismatch"):
        protocol.compare_episode(episode(**{key: value}), episode())


def test_count_changes_are_exact_and_native_151_steps_allowed():
    row = episode(length=151, travel_time=30.2, success=False)
    protocol.compare_episode(row, row)
    with pytest.raises(ValueError, match="length"):
        protocol.compare_episode(episode(length=11, travel_time=2.2), episode())


def test_relative_reward_tolerance_but_never_relative_travel_tolerance():
    protocol.compare_episode(episode(reward=1e6 + 5e-7), episode(reward=1e6))
    with pytest.raises(ValueError, match="Travel"):
        protocol.compare_episode(episode(length=150, travel_time=30 + 1e-11),
                                 episode(length=150, travel_time=30))


def test_selection_exact_500_rows_24_23_10_union_37():
    rows = references()
    before = copy.deepcopy(rows)
    assert protocol.selected_seeds(rows) == protocol.SEEDS
    assert rows == before


@pytest.mark.parametrize("change", ["missing_arm", "extra_arm", "short", "count", "union",
                                    "overlap", "nonfinite", "float_count"])
def test_selection_fail_closed(change):
    rows = references()
    if change == "missing_arm":
        del rows["geometry"]
    elif change == "extra_arm":
        rows["new"] = copy.deepcopy(rows["geometry"])
    elif change == "short":
        rows["geometry"].pop()
    elif change == "count":
        rows["padding"][0]["success"] = False
    elif change == "union":
        rows["padding"][protocol.SEEDS[10] - protocol.FIRST_SEED]["success"] = True
        rows["padding"][0]["success"] = False
    elif change == "overlap":
        rows["sector"][protocol.SEEDS[0] - protocol.FIRST_SEED]["success"] = True
        rows["sector"][0]["success"] = False
    elif change == "nonfinite":
        rows["geometry"][0]["min_ttc"] = math.inf
    else:
        rows["geometry"][0]["length"] = 10.0
    with pytest.raises(ValueError):
        protocol.selected_seeds(rows)


def test_load_references_converts_only_validated_counts(tmp_path, monkeypatch):
    from scripts import audit_geometry_development as historical

    expected = references()
    raw = copy.deepcopy(expected)
    for arm_rows in raw.values():
        for row in arm_rows:
            for field in protocol.COUNT_FIELDS:
                row[field] = float(row[field])
    lookup = {spec["csv"]: arm for arm, spec in protocol.ARMS.items()}
    calls = []

    def read(path, count):
        assert count == 500
        return copy.deepcopy(raw[lookup[path.relative_to(tmp_path).as_posix()]])

    monkeypatch.setattr(historical, "read_rows", read)
    monkeypatch.setattr(protocol, "_pinned", lambda root, name, sha: calls.append(name))
    monkeypatch.setattr(protocol, "source_path", lambda root, name: root / name)
    monkeypatch.setattr(protocol, "_json", lambda path: {})
    monkeypatch.setattr(protocol, "_summary_checks", lambda *args: None)
    result = protocol.load_references(tmp_path)
    assert result == expected and len(calls) == 10
    assert all(type(row["length"]) is int for rows in result.values() for row in rows)
    raw["geometry"][0]["min_ttc"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        protocol.load_references(tmp_path)


def summary(arm):
    spec = protocol.ARMS[arm]
    rows = references()[arm]
    result = {
        "episodes": 500.0, "first_seed": 40042, "last_seed": 40541,
        "config_sha256": protocol.CONFIG_SHA256, "model_sha256": spec["model_sha256"],
        "model_path": spec["model"].replace("/", "\\"), "config_path": protocol.CONFIG,
        "safety_shield": False, "safety_shield_type": None, "unsafe_ttc_threshold": 2.0,
        "risk_fusion": False, "target_speed_observation": False, "collision_first_reward": True,
        "intent_model_path": None, "intent_model_sha256": None, "intent_neighbors": 0,
        "fusion_neighbors": 0,
    }
    mapping = {"mean_reward": "reward", "mean_length": "length", "success_rate": "success",
               "collision_rate": "collision", "mean_travel_time": "travel_time",
               "mean_min_ttc": "min_ttc", "mean_unsafe_ttc_events": "unsafe_ttc_events",
               "mean_safety_interventions": "safety_interventions"}
    result.update({key: statistics.fmean(row[field] for row in rows)
                   for key, field in mapping.items()})
    if arm != "geometry":
        result.update(observation_protocol=spec["protocol"], sector_comparison_arm=arm,
                      initial_policy=protocol.INITIAL_POLICY)
    return result


@pytest.mark.parametrize("arm", protocol.ARMS)
def test_summary_metadata_protocol_and_numeric_reconciliation(arm):
    saved = summary(arm)
    rows = references()[arm]
    protocol._summary_checks(arm, protocol.ARMS[arm], saved, rows)
    saved["mean_reward"] += 1e-6
    with pytest.raises(ValueError, match="summary metric"):
        protocol._summary_checks(arm, protocol.ARMS[arm], saved, rows)


@pytest.mark.parametrize("key,value", [("first_seed", 10042), ("last_seed", 40540),
                                      ("model_path", "models/callback_best.zip"),
                                      ("safety_shield", 0), ("safety_shield_type", "ttc"),
                                      ("unsafe_ttc_threshold", 3.0),
                                      ("observation_protocol", "different"),
                                      ("initial_policy", {}), ("sector_comparison_arm", "sector")])
def test_summary_rejects_metadata_changes(key, value):
    saved = summary("padding")
    saved[key] = value
    with pytest.raises(ValueError):
        protocol._summary_checks("padding", protocol.ARMS["padding"], saved,
                                 references()["padding"])


def test_geometry_not_relabelled_and_null_metadata_must_be_present():
    saved = summary("geometry")
    saved["observation_protocol"] = None
    with pytest.raises(ValueError, match="relabeled"):
        protocol._summary_checks("geometry", protocol.ARMS["geometry"], saved,
                                 references()["geometry"])
    saved = summary("geometry")
    del saved["safety_shield_type"]
    with pytest.raises(ValueError, match="metadata"):
        protocol._summary_checks("geometry", protocol.ARMS["geometry"], saved,
                                 references()["geometry"])


@pytest.mark.parametrize("name", ["../outside", "/root", "C:/file", "a\\b", "a//b",
                                   "a/./b", ".git/config", "", "a/../b"])
def test_unsafe_source_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        protocol.source_path(tmp_path, name)


def test_pinned_bytes_and_duplicate_json(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert protocol._pinned(tmp_path, "sample.json", digest) == path
    with pytest.raises(ValueError, match="hash"):
        protocol._pinned(tmp_path, "sample.json", "0" * 64)
    with pytest.raises(ValueError, match="Duplicate"):
        protocol._json(path)


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    folder = tmp_path / protocol.SNAPSHOT
    folder.mkdir(parents=True)
    content = {"README.md": b"pending user readme\r\n", "MILESTONES.md": b"baseline\r\n"}
    inventory = {}
    with zipfile.ZipFile(folder / "snapshot.zip", "w") as archive:
        for name, payload in content.items():
            (tmp_path / name).write_bytes(payload)
            archive.writestr(name, payload)
            inventory[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    archive = folder / "snapshot.zip"
    archive_hash = protocol.sha(archive)
    manifest = {"status": "complete", "files": inventory, "member_count": 2,
                "uncompressed_bytes": sum(map(len, content.values())),
                "archive_fingerprint": {"bytes": archive.stat().st_size, "sha256": archive_hash}}
    target = folder / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(protocol, "SNAPSHOT_MANIFEST_SHA256", protocol.sha(target))
    monkeypatch.setattr(protocol, "SNAPSHOT_ARCHIVE_SHA256", archive_hash)
    monkeypatch.setattr(protocol, "SNAPSHOT_MEMBERS", 2)
    monkeypatch.setattr(protocol, "SNAPSHOT_BYTES", manifest["uncompressed_bytes"])
    return tmp_path, manifest


def test_snapshot_readback_all_bytes_and_append_only_milestones(snapshot):
    root, manifest = snapshot
    assert protocol.verify_snapshot(root) == manifest
    (root / "MILESTONES.md").write_text("baseline\nnew append\n", encoding="utf-8")
    assert protocol.verify_snapshot(root) == manifest


@pytest.mark.parametrize("file,payload", [("README.md", b"changed"),
                                         ("MILESTONES.md", b"rewritten baseline")])
def test_snapshot_refuses_live_baseline_modification(snapshot, file, payload):
    root, _ = snapshot
    (root / file).write_bytes(payload)
    with pytest.raises(ValueError):
        protocol.verify_snapshot(root)


def test_snapshot_refuses_archive_or_manifest_corruption(snapshot):
    root, _ = snapshot
    archive = root / protocol.SNAPSHOT / "snapshot.zip"
    archive.write_bytes(archive.read_bytes() + b"unexpected tail")
    with pytest.raises(ValueError, match="hash"):
        protocol.verify_snapshot(root)


def test_runtime_inventory_exact_packages_and_python(monkeypatch):
    packages = [{"name": f"package{i:02d}", "version": "1"} for i in range(46)]
    expected = {"python": "3.12.9", "implementation": "CPython", "packages": packages}
    monkeypatch.setattr(protocol, "_snapshot_manifest", lambda root: {"runtime": expected})
    monkeypatch.setattr(protocol.platform, "python_version", lambda: "3.12.9")
    monkeypatch.setattr(protocol.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(protocol, "_discover_runtime_packages", lambda: [
        dict(row, location="installed", metadata_sha256={"METADATA": "fakehash"})
        for row in packages])
    assert protocol.verify_runtime(Path("."))["packages"] == packages
    packages.append({"name": "newpackage", "version": "2"})
    with pytest.raises(ValueError, match="inventory"):
        protocol.verify_runtime(Path("."))
    monkeypatch.setattr(protocol.platform, "python_version", lambda: "3.13.0")
    with pytest.raises(ValueError, match="version"):
        protocol.verify_runtime(Path("."))


@pytest.fixture
def runtime_tree(tmp_path, monkeypatch):
    site = tmp_path / ".venv/Lib/site-packages"
    names = [f"package{i:02d}" for i in range(45)] + ["safeintent-rl"]
    expected = [{"name": name, "version": "0.1.0" if name == "safeintent-rl" else "1"}
                for name in names]
    discovered = [dict(item, location=site.as_posix(), metadata_sha256={
        (site / ("safeintent_rl-0.1.0.dist-info/METADATA" if item["name"] == "safeintent-rl"
                 else f"{item['name']}-1.dist-info/METADATA")).as_posix(): "metadata_hash"})
        for item in expected]
    monkeypatch.setattr(protocol, "_snapshot_manifest", lambda root: {"runtime": {
        "python": "3.12.9", "implementation": "CPython", "packages": expected}})
    monkeypatch.setattr(protocol.platform, "python_version", lambda: "3.12.9")
    monkeypatch.setattr(protocol.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(protocol, "_discover_runtime_packages", lambda: copy.deepcopy(discovered))
    alias = {"name": "safeintent-rl", "version": "0.1.0", "location": tmp_path.as_posix(),
             "metadata_sha256": {(tmp_path / "safeintent_rl.egg-info/PKG-INFO").as_posix():
                                 "metadata_hash"}}
    return tmp_path, expected, discovered, alias


def test_runtime_unique_baseline_and_exact_local_duplicate_evidence(runtime_tree):
    root, expected, discovered, alias = runtime_tree
    baseline = protocol.verify_runtime(root)
    assert baseline["packages"] == expected
    assert baseline["discovered_packages"] == discovered
    assert baseline["excluded_local_project_metadata"] is None
    discovered.append(alias)
    result = protocol.verify_runtime(root)
    assert result["packages"] == expected and len(result["packages"]) == 46
    assert result["discovered_packages"] == discovered and len(discovered) == 47
    assert result["excluded_local_project_metadata"] == alias


@pytest.mark.parametrize("change", ["elsewhere", "new_local_version", "other_duplicate",
                                    "missing_counterpart", "double_local", "wrong_metadata",
                                    "wrong_counterpart_version", "wrong_installed_origin",
                                    "different_metadata_bytes"])
def test_runtime_never_generally_deduplicates_or_ignores_versions(runtime_tree, change):
    root, _, discovered, alias = runtime_tree
    discovered.append(alias)
    if change == "elsewhere":
        alias["location"] = (root / "unrelated").as_posix()
    elif change == "new_local_version":
        alias["version"] = "0.2.0"
    elif change == "other_duplicate":
        discovered.pop()
        discovered.append(copy.deepcopy(discovered[0]))
    elif change == "missing_counterpart":
        discovered.pop(-2)
    elif change == "double_local":
        discovered.append(copy.deepcopy(alias))
    elif change == "wrong_metadata":
        alias["metadata_sha256"] = {(root / "other.egg-info/PKG-INFO").as_posix(): "hash"}
    elif change == "wrong_counterpart_version":
        discovered[-2]["version"] = "0.2.0"
    elif change == "wrong_installed_origin":
        discovered[-2]["location"] = (root / "other_installed").as_posix()
    else:
        alias["metadata_sha256"][next(iter(alias["metadata_sha256"]))] = "different_hash"
    with pytest.raises(ValueError):
        protocol.verify_runtime(root)


def test_raw_runtime_discovery_hashes_exact_metadata_files(tmp_path, monkeypatch):
    local = tmp_path / "safeintent_rl.egg-info/PKG-INFO"
    local.parent.mkdir()
    local.write_bytes(b"Name: safeintent-rl\nVersion: 0.1.0\n")
    dist = SimpleNamespace(metadata={"Name": "safeintent-rl"}, version="0.1.0",
                           files=[Path("safeintent_rl.egg-info/PKG-INFO")],
                           locate_file=lambda file: tmp_path / file)
    monkeypatch.setattr(protocol.importlib.metadata, "distributions", lambda: [dist])
    result = protocol._discover_runtime_packages()
    assert result == [{"name": "safeintent-rl", "version": "0.1.0",
                       "location": tmp_path.as_posix(),
                       "metadata_sha256": {local.as_posix(): protocol.sha(local)}}]
    dist.files = []
    with pytest.raises(ValueError, match="metadata"):
        protocol._discover_runtime_packages()


def test_runtime_metadata_discovery_excludes_nested_vendored_distributions(tmp_path, monkeypatch):
    own = tmp_path / "setuptools-84.0.0.dist-info/METADATA"
    own.parent.mkdir()
    own.write_bytes(b"Name: setuptools\nVersion: 84.0.0\n")
    nested = tmp_path / "setuptools/_vendor/packaging-26.0.dist-info/METADATA"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"Name: packaging\nVersion: 26.0\n")
    dist = SimpleNamespace(metadata={"Name": "setuptools"}, version="84.0.0",
                           files=[own.relative_to(tmp_path), nested.relative_to(tmp_path)],
                           locate_file=lambda file: tmp_path / file)
    monkeypatch.setattr(protocol.importlib.metadata, "distributions", lambda: [dist])
    result = protocol._discover_runtime_packages()
    assert result[0]["metadata_sha256"] == {own.as_posix(): protocol.sha(own)}


def test_dependency_fingerprints_stream_every_selected_file(tmp_path, monkeypatch):
    paths = {tmp_path / name for name in ("policy.py", "torch_cpu.dll", "python312.dll")}
    for path in paths:
        path.write_bytes((path.name.encode() + b"x") * 100000)
    monkeypatch.setattr(protocol, "_dependency_paths", lambda: paths)
    assert protocol.dependency_fingerprints() == {
        p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


@pytest.fixture
def fingerprint_tree(tmp_path, monkeypatch):
    def write(name, payload=b"source"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return protocol.sha(path)

    sources = {"safeintent_rl/__init__.py": write("safeintent_rl/__init__.py")}
    sources.update({f"scripts/old{i:02d}.py": write(f"scripts/old{i:02d}.py") for i in range(55)})
    predecessors = {f"results/old{i}.json": write(f"results/old{i}.json") for i in range(9)}
    old = {name: {"sha256": digest} for name, digest in sources.items()}
    old["README.md"] = {"sha256": write("README.md", b"pending user doc")}
    monkeypatch.setattr(protocol, "verify_snapshot", lambda root: {"files": old})
    record = {"status": "complete", "source_sha256": sources,
              "predecessor_sha256": predecessors}
    monkeypatch.setattr(protocol, "BATCH_RECORD_SHA256",
                        write(protocol.BATCH_RECORD, json.dumps(record).encode()))
    for name, key in ((protocol.CONFIG, "CONFIG_SHA256"), (protocol.DESIGN, "DESIGN_SHA256"),
                      (protocol.TRACE, "TRACE_SHA256"),
                      (protocol.PREPARATION, "PREPARATION_SHA256"),
                      (f"{protocol.SNAPSHOT}/snapshot.zip", "SNAPSHOT_ARCHIVE_SHA256"),
                      (f"{protocol.SNAPSHOT}/manifest.json", "SNAPSHOT_MANIFEST_SHA256")):
        monkeypatch.setattr(protocol, key, write(name))
    arms = copy.deepcopy(protocol.ARMS)
    for spec in arms.values():
        for key in ("model", "csv", "summary"):
            spec[f"{key}_sha256"] = write(spec[key])
    monkeypatch.setattr(protocol, "ARMS", arms)
    for name in (protocol.PROTOCOL, protocol.RUNNER, "tests/new_test.py", "scripts/new_helper.py"):
        write(name)
    return tmp_path


def test_input_fingerprints_include_entire_script_and_test_inventory(fingerprint_tree):
    result = protocol.fingerprint_inputs(fingerprint_tree)
    assert result["tests/new_test.py"] == protocol.sha(fingerprint_tree / "tests/new_test.py")
    assert protocol.PROTOCOL in result and protocol.RUNNER in result
    assert "scripts/new_helper.py" in result and "README.md" in result
    assert "MILESTONES.md" not in result
    assert len([name for name in result if name.startswith("models/")]) == 3


@pytest.mark.parametrize("change", ["package_added", "source_modified", "predecessor_modified",
                                    "runner_missing", "tests_missing"])
def test_input_fingerprints_reject_drift(fingerprint_tree, change):
    root = fingerprint_tree
    if change == "package_added":
        (root / "safeintent_rl/extra.py").write_bytes(b"new unapproved package source")
    elif change == "source_modified":
        (root / "scripts/old00.py").write_bytes(b"edited")
    elif change == "predecessor_modified":
        (root / "results/old0.json").write_bytes(b"edited")
    elif change == "runner_missing":
        (root / protocol.RUNNER).unlink()
    else:
        (root / "tests/new_test.py").unlink()
    with pytest.raises(ValueError):
        protocol.fingerprint_inputs(root)


def test_import_does_not_import_simulator_or_torch():
    # The test runner may already import them; the module itself exposes only lazy imports.
    assert "torch" not in protocol.__dict__ and "highway_env" not in protocol.__dict__
    assert protocol.sys is sys
