"""Preserve exact working-tree bytes and three final models before diagnostic edits.

This is a local content backup, not a Git-object/index or installed-runtime backup.
Preflight failures create nothing. Failures after reserving the output directory
retain the archive (including partial bytes) and a failed/running manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path, PurePosixPath

DEFAULT_OUTPUT = "logs/v3_sector_timeout_diagnostic_v1_snapshot_20260912T0949Z"
MODEL_HASHES = {
    "models/ppo_v3_predictive_geometry_v2_seed42.zip":
        "f3bfeaa9826f989ab1934b71a46a92a97cef159c8a800c56b23b4530c17636d2",
    "models/ppo_v3_sector_padding_v1_seed42.zip":
        "82da273ff74b39776b7486a6523c96102428b5e84d72499e130d9c7c22fddbd6",
    "models/ppo_v3_sector_features_v1_seed42.zip":
        "0695d09d098ecdae60ae63d33b919c691af8844f782af5a66649c33e497bfaef",
}
PINNED = {
    **MODEL_HASHES,
    "configs/intersection_v3_predictive_geometry_v2.yaml":
        "a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7",
    "V3_SECTOR_TIMEOUT_DIAGNOSTIC_V1.md":
        "4846e927192fed546e569d90a98b410d7d665c97b3742d08f97d0363a5abb087",
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True)
    return result.stdout.decode("utf-8")


def git_state(root: Path) -> dict:
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root:
        raise ValueError("Snapshot root must be the Git repository root")
    tracked = _git(root, "ls-files", "--cached", "-z").split("\0")[:-1]
    if not tracked or len(tracked) != len(set(tracked)):
        raise ValueError("Empty or ambiguous/unmerged tracked-file inventory")
    return {"head": _git(root, "rev-parse", "HEAD").strip(),
            "status_porcelain_v1_z": _git(root, "status", "--porcelain=v1", "-z",
                                          "--untracked-files=all"),
            "index_entries_z": _git(root, "ls-files", "--stage", "-z"),
            "tracked_paths": sorted(tracked)}


def source_path(root: Path, name: str) -> Path:
    parts = PurePosixPath(name).parts
    if (not parts or "\\" in name or ":" in name or name.startswith("/")
            or any(part in ("", ".", "..") for part in name.split("/"))
            or parts[0].casefold() == ".git"):
        raise ValueError(f"Unsafe archive member: {name}")
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing/nonregular snapshot input: {name}")
    path.resolve(strict=True).relative_to(root)
    return path


def fingerprint(path: Path) -> dict:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _require_ignored(root: Path, destination: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", "--no-index", "--",
         (destination / "snapshot.zip").relative_to(root).as_posix()],
        check=False, capture_output=True)
    if result.returncode != 0:
        raise ValueError("Snapshot destination must be ignored by Git")


def _runtime_inventory() -> dict:
    packages = sorted(
        ({"name": dist.metadata.get("Name", ""), "version": dist.version}
         for dist in distributions()), key=lambda item: (item["name"], item["version"]))
    return {"python": platform.python_version(), "executable": sys.executable,
            "implementation": platform.python_implementation(), "platform": platform.platform(),
            "packages": packages}


def _save_manifest(path: Path, manifest: dict, *, first: bool = False) -> None:
    target = path if first else path.with_name("manifest.update.json")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not first:
        os.replace(target, path)  # Only this invocation's exclusively owned manifest.


def _write_archive(path: Path, root: Path, inventory: dict[str, dict]) -> None:
    with path.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for name, expected in inventory.items():
                payload = source_path(root, name).read_bytes()
                observed = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                if observed != expected:
                    raise ValueError(f"Input changed before archive write: {name}")
                archive.writestr(name, payload)
        stream.flush()
        os.fsync(stream.fileno())


def verify_archive(path: Path, inventory: dict[str, dict]) -> None:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != list(inventory) or archive.testzip() is not None:
            raise ValueError("Archive inventory or CRC verification failed")
        for name, expected in inventory.items():
            payload = archive.read(name)
            if {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} != expected:
                raise ValueError(f"Archive member failed byte/hash verification: {name}")


def create_snapshot(root: Path, output_dir: Path | None = None) -> dict:
    root = root.resolve(strict=True)
    destination = output_dir if output_dir is not None else Path(DEFAULT_OUTPUT)
    destination = (destination if destination.is_absolute() else root / destination).resolve()
    relative = destination.relative_to(root)
    if (len(relative.parts) != 2 or relative.parts[0] != "logs"
            or not relative.parts[1].startswith("v3_sector_timeout_diagnostic_v1_snapshot")):
        raise ValueError("Use a unique timeout snapshot directory directly inside repository logs")
    if destination.exists():
        raise FileExistsError(f"Preserve existing snapshot directory: {destination}")
    _require_ignored(root, destination)
    before = git_state(root)
    if not {"README.md", "MILESTONES.md"}.issubset(before["tracked_paths"]):
        raise ValueError("Both pending research documents must be tracked and preserved")
    if not (set(PINNED) - set(MODEL_HASHES)).issubset(before["tracked_paths"]):
        raise ValueError("Frozen configuration and diagnostic design must be tracked")
    names = sorted(set(before["tracked_paths"]) | set(MODEL_HASHES))
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Case-colliding member names cannot be restored safely on Windows")
    inventory = {name: fingerprint(source_path(root, name)) for name in names}
    for name, digest in PINNED.items():
        if inventory[name]["sha256"] != digest:
            raise ValueError(f"Frozen input changed: {name}")
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(root), "git": before, "runtime": _runtime_inventory(),
                "archive": "snapshot.zip", "files": inventory,
                "limits": "Working-tree file bytes only. No .git objects/index content, installed "
                          "runtime, other ignored files, or untracked files are backed up. "
                          "Runtime inventory is provenance, not an environment backup.",
                "restore": "Verify archive and every member SHA-256, then extract into a NEW "
                            "empty directory. Do not overwrite the live repository. Pending "
                            "README/MILESTONES edits are preserved as current file bytes."}
    destination.mkdir(parents=True, exist_ok=False)  # Atomic directory reservation; reject races.
    manifest_path, archive_path = destination / "manifest.json", destination / "snapshot.zip"
    _save_manifest(manifest_path, manifest, first=True)
    try:
        _write_archive(archive_path, root, inventory)
        verify_archive(archive_path, inventory)
        for name, expected in inventory.items():
            if fingerprint(source_path(root, name)) != expected:
                raise ValueError(f"Input changed during snapshot: {name}")
        if git_state(root) != before:
            raise ValueError("Git HEAD, tracked inventory, index or working status changed")
        manifest.update(status="complete", archive_fingerprint=fingerprint(archive_path),
                        member_count=len(inventory),
                        uncompressed_bytes=sum(item["bytes"] for item in inventory.values()))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _save_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--refuse-overwrite", action="store_true", required=True)
    args = parser.parse_args()
    manifest = create_snapshot(args.root, args.output_dir)
    print(json.dumps({key: manifest[key] for key in
                      ("status", "member_count", "uncompressed_bytes", "archive_fingerprint")},
                     indent=2))


if __name__ == "__main__":
    main()
