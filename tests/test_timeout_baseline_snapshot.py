from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import preserve_timeout_baseline_v1 as snapshot


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    tracked = {"README.md": b"pending README\r\n",
               "MILESTONES.md": b"history\r\npending append\r\n",
               "source.py": b"original = True\n",
               "config.yaml": b"duration: 30\n", "design.md": b"frozen design\n"}
    models = {f"models/{arm}.zip": f"checkpoint {arm}".encode()
              for arm in ("geometry", "padding", "sector")}
    content = {**tracked, **models}
    for name, payload in content.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / "untracked.txt").write_bytes(b"not in authorized inventory")
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in content.items()}
    monkeypatch.setattr(snapshot, "MODEL_HASHES", {name: hashes[name] for name in models})
    monkeypatch.setattr(snapshot, "PINNED", {name: hashes[name]
                                           for name in (*models, "config.yaml", "design.md")})
    state = {"head": "a" * 40, "status_porcelain_v1_z": " M README.md\0 M MILESTONES.md\0",
             "index_entries_z": "index evidence\0", "tracked_paths": sorted(tracked)}
    monkeypatch.setattr(snapshot, "git_state", lambda unused: copy.deepcopy(state))
    monkeypatch.setattr(snapshot, "_require_ignored", lambda *unused: None)
    monkeypatch.setattr(snapshot, "_runtime_inventory", lambda: {"python": "test"})
    return root, root / snapshot.DEFAULT_OUTPUT, content, state


def test_complete_snapshot_preserves_pending_bytes_and_records_hashes(baseline):
    root, destination, content, state = baseline
    result = snapshot.create_snapshot(root)
    assert result["status"] == "complete" and result["git"] == state
    assert result["member_count"] == len(content)
    assert result["uncompressed_bytes"] == sum(map(len, content.values()))
    assert result == json.loads((destination / "manifest.json").read_text())
    archive_path = destination / "snapshot.zip"
    assert result["archive_fingerprint"] == snapshot.fingerprint(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == sorted(content)
        for name, payload in content.items():
            assert archive.read(name) == payload == (root / name).read_bytes()
            assert result["files"][name] == snapshot.fingerprint(root / name)
    assert not (destination / "manifest.update.json").exists()


def test_existing_output_is_preserved(baseline):
    root, destination, _, _ = baseline
    destination.mkdir(parents=True)
    sentinel = destination / "sentinel"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        snapshot.create_snapshot(root)
    assert sentinel.read_bytes() == b"preserve"
    assert list(destination.iterdir()) == [sentinel]


def test_racing_directory_is_not_reused(baseline, monkeypatch):
    root, destination, _, _ = baseline

    def race(*unused):
        destination.mkdir(parents=True)
        (destination / "sentinel").write_bytes(b"other process")

    monkeypatch.setattr(snapshot, "_require_ignored", race)
    with pytest.raises(FileExistsError):
        snapshot.create_snapshot(root)
    assert (destination / "sentinel").read_bytes() == b"other process"
    assert not (destination / "manifest.json").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/absolute", "a\\b",
                                  "a//b", "a/./b", ".git/config", "a/../../outside"])
def test_unsafe_member_paths_are_rejected(baseline, name):
    root, _, _, _ = baseline
    with pytest.raises(ValueError, match="Unsafe archive member"):
        snapshot.source_path(root, name)


@pytest.mark.parametrize("destination", ["../outside", "logs", "snapshot", "logs/unrelated"])
def test_destination_must_be_scoped_under_ignored_logs(baseline, destination):
    root, _, _, _ = baseline
    with pytest.raises(ValueError):
        snapshot.create_snapshot(root, Path(destination))


def test_changed_frozen_model_fails_before_creating_output(baseline):
    root, destination, _, _ = baseline
    (root / "models/geometry.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        snapshot.create_snapshot(root)
    assert not destination.exists()


def test_missing_tracked_file_fails_before_creating_output(baseline):
    root, destination, _, _ = baseline
    (root / "source.py").unlink()
    with pytest.raises(ValueError, match="Missing/nonregular"):
        snapshot.create_snapshot(root)
    assert not destination.exists()


def test_write_failure_preserves_partial_archive_and_failed_manifest(baseline, monkeypatch):
    root, destination, _, _ = baseline

    def fail(path, *unused):
        path.write_bytes(b"partial ZIP evidence")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(snapshot, "_write_archive", fail)
    with pytest.raises(OSError, match="simulated disk failure"):
        snapshot.create_snapshot(root)
    assert (destination / "snapshot.zip").read_bytes() == b"partial ZIP evidence"
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "simulated disk failure" in manifest["error"]
    with pytest.raises(FileExistsError):
        snapshot.create_snapshot(root)


def test_readback_failure_does_not_claim_completion(baseline, monkeypatch):
    root, destination, _, _ = baseline

    def fail(*unused):
        raise ValueError("simulated readback mismatch")

    monkeypatch.setattr(snapshot, "verify_archive", fail)
    with pytest.raises(ValueError, match="simulated readback mismatch"):
        snapshot.create_snapshot(root)
    assert (destination / "snapshot.zip").is_file()
    assert json.loads((destination / "manifest.json").read_text())["status"] == "failed"


def test_input_change_after_copy_preserves_original_snapshot_and_fails(baseline, monkeypatch):
    root, destination, content, _ = baseline
    original = snapshot._write_archive

    def changed(path, repo, inventory):
        original(path, repo, inventory)
        (repo / "README.md").write_bytes(b"concurrent user edit")

    monkeypatch.setattr(snapshot, "_write_archive", changed)
    with pytest.raises(ValueError, match="Input changed during snapshot: README.md"):
        snapshot.create_snapshot(root)
    with zipfile.ZipFile(destination / "snapshot.zip") as archive:
        assert archive.read("README.md") == content["README.md"]
    assert (root / "README.md").read_bytes() == b"concurrent user edit"
    assert json.loads((destination / "manifest.json").read_text())["status"] == "failed"


def test_git_head_change_is_detected(baseline, monkeypatch):
    root, destination, _, state = baseline
    original = snapshot._write_archive

    def changed(*args):
        original(*args)
        state["head"] = "b" * 40

    monkeypatch.setattr(snapshot, "_write_archive", changed)
    with pytest.raises(ValueError, match="Git HEAD"):
        snapshot.create_snapshot(root)
    assert json.loads((destination / "manifest.json").read_text())["status"] == "failed"


def test_archive_member_hash_validation(baseline):
    root, destination, _, _ = baseline
    result = snapshot.create_snapshot(root)
    corrupted = destination / "corrupt-test.zip"
    with zipfile.ZipFile(corrupted, "w") as archive:
        for name in result["files"]:
            archive.writestr(name, b"wrong bytes")
    with pytest.raises(ValueError, match="byte/hash verification"):
        snapshot.verify_archive(corrupted, result["files"])


def test_case_collision_is_rejected(baseline):
    root, destination, _, state = baseline
    state["tracked_paths"].append("readme.md")
    with pytest.raises(ValueError, match="Case-colliding"):
        snapshot.create_snapshot(root)
    assert not destination.exists()
