"""Frozen inputs and fail-closed checks for the selected timeout diagnostic.

Read-only: this module neither creates release files nor simulates episodes.
CSV count fields are converted to Python ints only after the historical strict
reader validates them. Raw files are never changed. Dependency hashes cover all
Python sources in the three simulator/RL packages and selected critical runtime
binaries; they are byte provenance, not a certification of every loaded binary.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import statistics
import sys
import zipfile
from numbers import Real
from pathlib import Path, PurePosixPath

FIRST_SEED = 40042
EPISODES = 500
SEEDS = (
    40084, 40106, 40125, 40143, 40160, 40168, 40173, 40175, 40181, 40193,
    40203, 40207, 40222, 40223, 40243, 40256, 40271, 40276, 40286, 40295,
    40296, 40339, 40341, 40349, 40390, 40396, 40403, 40429, 40440, 40449,
    40476, 40495, 40506, 40514, 40530, 40539, 40541,
)
CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"
CONFIG_SHA256 = "a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7"
DESIGN = "V3_SECTOR_TIMEOUT_DIAGNOSTIC_V1.md"
DESIGN_SHA256 = "4846e927192fed546e569d90a98b410d7d665c97b3742d08f97d0363a5abb087"
RUNNER = "scripts/run_sector_timeout_diagnostic_v1.py"
PROTOCOL = "scripts/sector_timeout_protocol_v1.py"
TRACE = "scripts/sector_timeout_trace_v1.py"
TRACE_SHA256 = "be4e23b57faf3593136b910b5f54b8ca73a206be2e1359f81c37a0211b67f94d"
PREPARATION = "results/v3_sector_timeout_trace_v1.preparation.json"
PREPARATION_SHA256 = "8cddf592bc42c52467322f830f1576f5b5f313e9d4acc4199cd9c2718f5efcf3"
BATCH_RECORD = "results/ppo_v3_sector_features_v1_seed42.evaluate.run.json"
BATCH_RECORD_SHA256 = "144b8df605a1edb7c52a5d2c3ba47d7d2c76965268448a9aa6b481b5916674c1"
HISTORICAL_READER_SHA256 = "80d98ac72d453b172a729f6eac2b5eef9f962172bdb7d1d3a66af75833c274d0"
SNAPSHOT = "logs/v3_sector_timeout_diagnostic_v1_snapshot_20260912T0949Z"
SNAPSHOT_ARCHIVE_SHA256 = "68b9c311587cf9fe801d0a879f23f6251be209e6946a72037012385543173406"
SNAPSHOT_MANIFEST_SHA256 = "1f5f638d020b464c88cee753546fffe83a8e7d7a0e155e29b9d63e86a26b29f2"
SNAPSHOT_MEMBERS = 235
SNAPSHOT_BYTES = 15698716
INITIAL_POLICY = {
    "sha256": "ca35bda9cfe2f2afce2eb0cf71d2abd74fb6808d2f935b7e20a5b31c1d054c19",
    "parameter_count": 216580,
}

ARMS = {
    "geometry": {
        "model": "models/ppo_v3_predictive_geometry_v2_seed42.zip",
        "model_sha256": "f3bfeaa9826f989ab1934b71a46a92a97cef159c8a800c56b23b4530c17636d2",
        "csv": "results/ppo_v3_predictive_geometry_v2_development_seed40042.csv",
        "csv_sha256": "e145084f4ccce3d419c34886bd9434c0b5a817fb6fa3c1ceab616a89e1929030",
        "summary": "results/ppo_v3_predictive_geometry_v2_development_seed40042.summary.json",
        "summary_sha256": "227675bacf3f113209d970ec28763ccb57a86e352cfcd546f4ee8f6b1b5ecc61",
        "protocol": None,
        "shape": 115,
    },
    "padding": {
        "model": "models/ppo_v3_sector_padding_v1_seed42.zip",
        "model_sha256": "82da273ff74b39776b7486a6523c96102428b5e84d72499e130d9c7c22fddbd6",
        "csv": "results/ppo_v3_sector_padding_v1_development_seed40042.csv",
        "csv_sha256": "b35aa0e97aee4417870a40433e8f0d7a36235771f7abf2ea4981c3a087c54e96",
        "summary": "results/ppo_v3_sector_padding_v1_development_seed40042.summary.json",
        "summary_sha256": "53db2479022478371f6555f4a7eecd521a0a58481d274182c61d207b0c3aa53b",
        "protocol": "predictive_post_spawn_padding_v1",
        "shape": 163,
    },
    "sector": {
        "model": "models/ppo_v3_sector_features_v1_seed42.zip",
        "model_sha256": "0695d09d098ecdae60ae63d33b919c691af8844f782af5a66649c33e497bfaef",
        "csv": "results/ppo_v3_sector_features_v1_development_seed40042.csv",
        "csv_sha256": "68e7259e0c08519978b9a07bc48b812878b4f1b6481afd12d20e30abe5199cbb",
        "summary": "results/ppo_v3_sector_features_v1_development_seed40042.summary.json",
        "summary_sha256": "5b5c01f105b3ef018f08ac2aeccd2384c0351ce5eecec15e65fa380f7984453d",
        "protocol": "predictive_post_spawn_sector_v1",
        "shape": 163,
    },
}
BOOL_FIELDS = ("success", "collision")
COUNT_FIELDS = ("length", "unsafe_ttc_events", "safety_interventions")
FLOAT_FIELDS = ("reward", "travel_time", "min_ttc")
COLUMNS = ("reward", "length", "success", "collision", "travel_time", "min_ttc",
           "unsafe_ttc_events", "safety_interventions")
TOLERANCES = {
    "reward": {"abs_tol": 1e-9, "rel_tol": 1e-12},
    "min_ttc": {"abs_tol": 1e-9, "rel_tol": 1e-12},
    "travel_time": {"abs_tol": 1e-12, "rel_tol": 0.0},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Stream even large runtime DLLs rather than allocating their full bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(root: Path, name: str) -> Path:
    require(isinstance(name, str) and bool(name), "Invalid relative input name")
    parts = PurePosixPath(name).parts
    require(not name.startswith("/") and "\\" not in name and ":" not in name
            and not any(p in ("", ".", "..") for p in name.split("/"))
            and parts[0].casefold() != ".git", f"Unsafe input path: {name}")
    path = root.joinpath(*parts)
    require(path.is_file() and not path.is_symlink(), f"Missing/nonregular input: {name}")
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    return path


def _pinned(root: Path, name: str, expected: str) -> Path:
    path = source_path(root, name)
    require(sha(path) == expected, f"Frozen hash mismatch: {name}")
    return path


def _json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def validate_episode(row: dict) -> None:
    """Require precisely eight metrics with genuine bools and Python int counts."""
    require(isinstance(row, dict) and set(row) == set(COLUMNS), "Episode keys differ")
    for key in BOOL_FIELDS:
        require(type(row[key]) is bool, f"Not a boolean: {key}")
    for key in COUNT_FIELDS:
        require(type(row[key]) is int, f"Not an integer count: {key}")
    for key in FLOAT_FIELDS:
        value = row[key]
        require(isinstance(value, Real) and not isinstance(value, bool)
                and math.isfinite(value), f"Not a finite numeric metric: {key}")
    require(not (row["success"] and row["collision"]), "Overlapping outcomes")
    require(1 <= row["length"] <= 151, "Invalid episode length")
    require(0 <= row["unsafe_ttc_events"] <= row["length"], "Invalid unsafe count")
    require(row["safety_interventions"] == 0, "Unexpected intervention")
    require(row["min_ttc"] >= 0, "Negative TTC")
    require(math.isclose(row["travel_time"], row["length"] / 5,
                         **TOLERANCES["travel_time"]), "Travel time/length mismatch")


def compare_episode(actual: dict, reference: dict) -> dict[str, float | int]:
    """Raise on any mismatch; otherwise return six signed numeric deltas."""
    validate_episode(actual)
    validate_episode(reference)
    for key in BOOL_FIELDS + COUNT_FIELDS:
        require(actual[key] == reference[key], f"Episode mismatch: {key}")
    for key, tolerance in TOLERANCES.items():
        require(math.isclose(actual[key], reference[key], **tolerance),
                f"Episode mismatch: {key}")
    return {key: actual[key] - reference[key] for key in COUNT_FIELDS + FLOAT_FIELDS}


def _summary_checks(arm: str, spec: dict, summary: dict, rows: list[dict]) -> None:
    expected = {
        "first_seed": FIRST_SEED, "last_seed": FIRST_SEED + EPISODES - 1,
        "config_sha256": CONFIG_SHA256, "model_sha256": spec["model_sha256"],
        "safety_shield": False, "safety_shield_type": None,
        "unsafe_ttc_threshold": 2.0, "risk_fusion": False,
        "target_speed_observation": False, "collision_first_reward": True,
        "intent_model_path": None, "intent_model_sha256": None,
        "intent_neighbors": 0, "fusion_neighbors": 0,
    }
    for key, value in expected.items():
        require(key in summary and summary[key] == value, f"{arm} metadata differs: {key}")
        if isinstance(value, bool):
            require(type(summary[key]) is bool, f"{arm} invalid metadata boolean: {key}")
    for key, path in (("model_path", spec["model"]), ("config_path", CONFIG)):
        require(isinstance(summary.get(key), str)
                and summary[key].replace("\\", "/") == path, f"{arm} path differs: {key}")
    if arm == "geometry":
        require("observation_protocol" not in summary
                and "sector_comparison_arm" not in summary, "Geometry protocol was relabeled")
    else:
        require(summary.get("observation_protocol") == spec["protocol"]
                and summary.get("sector_comparison_arm") == arm
                and summary.get("initial_policy") == INITIAL_POLICY,
                f"{arm} protocol/initial policy differs")
    mapping = {
        "mean_reward": "reward", "mean_length": "length", "success_rate": "success",
        "collision_rate": "collision", "mean_travel_time": "travel_time",
        "mean_min_ttc": "min_ttc", "mean_unsafe_ttc_events": "unsafe_ttc_events",
        "mean_safety_interventions": "safety_interventions",
    }
    require(summary.get("episodes") == EPISODES, f"{arm} summary episode count differs")
    for key, field in mapping.items():
        observed = summary.get(key)
        require(isinstance(observed, Real) and not isinstance(observed, bool)
                and math.isfinite(observed)
                and math.isclose(observed, statistics.fmean(r[field] for r in rows),
                                 abs_tol=1e-12, rel_tol=1e-12),
                f"{arm} summary metric mismatch: {key}")


def load_references(root: Path) -> dict[str, list[dict]]:
    """Read all 500 records per frozen arm; source hashes bind the row/seed order."""
    from scripts import audit_geometry_development as historical

    require(sha(Path(historical.__file__)) == HISTORICAL_READER_SHA256,
            "Executing historical CSV reader differs")
    _pinned(root, CONFIG, CONFIG_SHA256)
    references = {}
    for arm, spec in ARMS.items():
        for key in ("model", "csv", "summary"):
            _pinned(root, spec[key], spec[f"{key}_sha256"])
        rows = historical.read_rows(source_path(root, spec["csv"]), count=EPISODES)
        for row in rows:
            for key in COUNT_FIELDS:
                value = row[key]
                require(math.isfinite(value) and float(value).is_integer(),
                        f"Nonintegral historical count: {key}")
                row[key] = int(value)
            validate_episode(row)
        _summary_checks(arm, spec, _json(source_path(root, spec["summary"])), rows)
        references[arm] = rows
    selected_seeds(references)
    return references


def selected_seeds(rows_by_arm: dict[str, list[dict]]) -> tuple[int, ...]:
    require(set(rows_by_arm) == set(ARMS), "Reference arms differ")
    incomplete = {}
    for arm, rows in rows_by_arm.items():
        require(len(rows) == EPISODES, f"{arm} reference count differs")
        for row in rows:
            validate_episode(row)
        incomplete[arm] = {FIRST_SEED + i for i, row in enumerate(rows)
                           if not row["success"] and not row["collision"]}
    require(len(incomplete["padding"]) == 24 and len(incomplete["sector"]) == 23,
            "Frozen incomplete counts differ")
    require(len(incomplete["padding"] & incomplete["sector"]) == 10,
            "Frozen incomplete overlap differs")
    result = tuple(sorted(incomplete["padding"] | incomplete["sector"]))
    require(result == SEEDS, "Frozen selected seed union differs")
    return result


def _snapshot_manifest(root: Path) -> dict:
    manifest = _json(_pinned(root, f"{SNAPSHOT}/manifest.json", SNAPSHOT_MANIFEST_SHA256))
    require(manifest.get("status") == "complete", "Snapshot is not complete")
    return manifest


def verify_snapshot(root: Path) -> dict:
    """Verify all archived bytes and live baseline bytes, allowing only MILE append."""
    manifest = _snapshot_manifest(root)
    archive_path = _pinned(root, f"{SNAPSHOT}/snapshot.zip", SNAPSHOT_ARCHIVE_SHA256)
    inventory = manifest["files"]
    require(len(inventory) == SNAPSHOT_MEMBERS == manifest["member_count"],
            "Snapshot member count differs")
    require(len({name.casefold() for name in inventory}) == len(inventory),
            "Snapshot member case collision")
    require(sum(item["bytes"] for item in inventory.values()) == SNAPSHOT_BYTES
            == manifest["uncompressed_bytes"], "Snapshot total bytes differ")
    require(manifest["archive_fingerprint"] == {
        "bytes": archive_path.stat().st_size, "sha256": SNAPSHOT_ARCHIVE_SHA256},
        "Snapshot archive fingerprint differs")
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.namelist() == list(inventory) and archive.testzip() is None,
                "Snapshot member inventory/CRC differs")
        for name, expected in inventory.items():
            path = source_path(root, name)
            payload = archive.read(name)
            require({"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                    == expected, f"Snapshot archived member differs: {name}")
            if name == "MILESTONES.md":
                archived_text = payload.decode("utf-8").replace("\r\n", "\n")
                require(path.read_text(encoding="utf-8").startswith(archived_text),
                        "MILESTONES baseline was edited instead of appended")
            else:
                require(path.stat().st_size == expected["bytes"]
                        and sha(path) == expected["sha256"], f"Live baseline changed: {name}")
    return manifest


def fingerprint_inputs(root: Path, *, require_runner: bool = True) -> dict[str, str]:
    """Verify historical lineage and inventory all live repo execution sources."""
    manifest = verify_snapshot(root)
    record = _json(_pinned(root, BATCH_RECORD, BATCH_RECORD_SHA256))
    require(record.get("status") == "complete" and len(record["source_sha256"]) == 56
            and len(record["predecessor_sha256"]) == 9, "Frozen batch record differs")
    expected = {name: item["sha256"] for name, item in manifest["files"].items()
                if name != "MILESTONES.md"}
    for lineage in (record["source_sha256"], record["predecessor_sha256"]):
        for name, digest in lineage.items():
            require(name not in expected or expected[name] == digest,
                    f"Conflicting historical fingerprint: {name}")
            expected[name] = digest
    expected.update({BATCH_RECORD: BATCH_RECORD_SHA256, CONFIG: CONFIG_SHA256,
                     DESIGN: DESIGN_SHA256, TRACE: TRACE_SHA256,
                     PREPARATION: PREPARATION_SHA256,
                     f"{SNAPSHOT}/snapshot.zip": SNAPSHOT_ARCHIVE_SHA256,
                     f"{SNAPSHOT}/manifest.json": SNAPSHOT_MANIFEST_SHA256})
    for spec in ARMS.values():
        expected.update({spec[k]: spec[f"{k}_sha256"] for k in ("model", "csv", "summary")})
    for name, digest in expected.items():
        _pinned(root, name, digest)
    package = {p.relative_to(root).as_posix() for p in (root / "safeintent_rl").rglob("*.py")}
    require(package == {name for name in record["source_sha256"]
                        if name.startswith("safeintent_rl/") and name.endswith(".py")},
            "SafeIntent source inventory changed")
    scripts = {p.relative_to(root).as_posix() for p in (root / "scripts").rglob("*.py")}
    tests = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("*.py")}
    require(PROTOCOL in scripts and TRACE in scripts, "Diagnostic protocol/trace source missing")
    require(not require_runner or RUNNER in scripts, "Diagnostic runner source missing")
    require(bool(tests), "Test source inventory missing")
    for name in package | scripts | tests:
        observed = sha(source_path(root, name))
        require(name not in expected or observed == expected[name],
                f"Historical source changed during fingerprinting: {name}")
        expected[name] = observed
    return dict(sorted(expected.items()))


def _discover_runtime_packages() -> list[dict]:
    """Preserve discovery multiplicity, resolved origins, and metadata byte hashes."""
    discovered = []
    for dist in importlib.metadata.distributions():
        # A wheel may vendor other distributions. Only its own top-level metadata
        # identifies this discovery entry; nested vendor metadata is not an alias.
        metadata_paths = [Path(dist.locate_file(file)).resolve(strict=True)
                          for file in (dist.files or [])
                          if len(Path(file).parts) == 2
                          and ((Path(file).parts[0].endswith(".dist-info")
                                and Path(file).name == "METADATA")
                               or (Path(file).parts[0].endswith(".egg-info")
                                   and Path(file).name == "PKG-INFO"))]
        require(len(metadata_paths) == 1 and metadata_paths[0].is_file(),
                f"Ambiguous/missing distribution metadata: {dist.metadata.get('Name', '')}")
        discovered.append({
            "name": dist.metadata.get("Name", ""), "version": dist.version,
            "location": Path(dist.locate_file("")).resolve(strict=True).as_posix(),
            "metadata_sha256": {path.as_posix(): sha(path) for path in metadata_paths},
        })
    return sorted(discovered, key=lambda item: (item["name"], item["version"], item["location"],
                                               tuple(item["metadata_sha256"])))


def verify_runtime(root: Path) -> dict:
    """Require the frozen installed 46; expose the one permitted local discovery alias.

    The original snapshot script did not put the repository root on sys.path.
    A -m/-c launch there also discovers the existing local project egg-info.
    Only that exact additional alias is allowed, never general deduplication.
    The returned raw inventory binds even this launch-context difference.
    """
    root = root.resolve(strict=True)
    expected = _snapshot_manifest(root)["runtime"]
    require(platform.python_version() == expected["python"] == "3.12.9",
            "Python version differs from frozen runtime")
    require(platform.python_implementation() == expected["implementation"] == "CPython",
            "Python implementation differs")
    discovered = _discover_runtime_packages()
    installed = list(discovered)
    local_aliases = [item for item in discovered
                     if item["name"] == "safeintent-rl" and item["location"] == root.as_posix()]
    require(len(local_aliases) <= 1, "Multiple local project metadata aliases")
    alias = local_aliases[0] if local_aliases else None
    if alias is not None:
        site_packages = root / ".venv" / "Lib" / "site-packages"
        local_metadata = (root / "safeintent_rl.egg-info" / "PKG-INFO").as_posix()
        installed_metadata = (
            site_packages / "safeintent_rl-0.1.0.dist-info" / "METADATA").as_posix()
        counterpart = [item for item in discovered
                       if item["name"] == "safeintent-rl" and item["version"] == "0.1.0"
                       and item["location"] == site_packages.as_posix()
                       and set(item["metadata_sha256"]) == {installed_metadata}]
        require(alias["version"] == "0.1.0"
                and set(alias["metadata_sha256"]) == {local_metadata}
                and len(counterpart) == 1 and len(discovered) == 47,
                "Unapproved local project metadata alias or missing installed counterpart")
        require(alias["metadata_sha256"][local_metadata]
                == counterpart[0]["metadata_sha256"][installed_metadata],
                "Local project metadata bytes differ from installed counterpart")
        installed.remove(alias)
    packages = sorted(({"name": item["name"], "version": item["version"]}
                       for item in installed), key=lambda item: (item["name"], item["version"]))
    require(len(expected["packages"]) == 46 and packages == expected["packages"],
            "Installed package inventory differs from frozen runtime")
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "executable": sys.executable, "platform": platform.platform(), "packages": packages,
            "discovered_packages": discovered, "excluded_local_project_metadata": alias}


def _dependency_paths() -> set[Path]:
    paths = set()
    for package in ("highway_env", "gymnasium", "stable_baselines3"):
        spec = importlib.util.find_spec(package)
        require(spec is not None and spec.submodule_search_locations is not None,
                f"Installed package cannot be located: {package}")
        for directory in spec.submodule_search_locations:
            paths.update(Path(directory).resolve(strict=True).rglob("*.py"))
    for module in ("numpy._core._multiarray_umath", "torch._C"):
        paths.add(Path(importlib.import_module(module).__file__).resolve(strict=True))
    torch_root = Path(importlib.import_module("torch").__file__).resolve(strict=True).parent
    paths.update((torch_root / "lib" / "torch_cpu.dll", torch_root / "lib" / "torch_python.dll",
                  Path(sys.executable).resolve(strict=True)))
    # This diagnostic is explicitly the frozen Windows runtime, not a portable release.
    require(sys.platform == "win32", "Diagnostic runtime must be Windows")
    paths.add(Path(sys.base_prefix) / f"python{sys.version_info.major}{sys.version_info.minor}.dll")
    require(all(path.is_file() for path in paths), "Critical installed runtime file missing")
    return paths


def dependency_fingerprints() -> dict[str, str]:
    """Return absolute paths: all 3 package sources and critical NumPy/Torch/Python bytes."""
    return {path.resolve(strict=True).as_posix(): sha(path)
            for path in sorted(_dependency_paths())}
