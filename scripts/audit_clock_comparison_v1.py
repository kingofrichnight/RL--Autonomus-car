"""Independent audit of the preregistered, completed two-arm clock comparison.

No runner acceptance helper is used. Imports are stdlib-only until verified
runtime/checkpoint inspection; no environment, prediction or learning is run.
Only main may write, exclusively to the reserved new audit report. A successful
audit is development evidence, never automatic policy promotion or a release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

NAME = "v3_clock_comparison_v1"
CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"
CONFIG_SHA256 = "a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7"
DESIGN = "V3_CLOCK_OBSERVATION_V1.md"
PREPARATION = "results/v3_clock_observation_v1.preparation.json"
ANALYSIS = "results/v3_sector_timeout_diagnostic_v1.analysis.json"
RELEASE_PATH = Path(f"configs/{NAME}.release.json")
VERIFICATION_PATH = Path(f"results/{NAME}.release_verification.json")
LOCK_PATH = Path(f"logs/{NAME}.active.lock")
OUTPUT = Path("results/ppo_v3_clock_comparison_v1_development_seed40042.audit.json")
INITIAL_POLICY = {
    "parameter_count": 192516,
    "sha256": "d1c314ef787fa038568eeaff5dfc6c1a3851aba5f39a4dd8ba767634943a57ec",
}
ARMS = {
    "zero": {"stem": "ppo_v3_clock_zero_v1_seed42",
             "evaluation_stem": "ppo_v3_clock_zero_v1_development_seed40042",
             "protocol": "predictive_post_spawn_clock_zero_v1"},
    "clock": {"stem": "ppo_v3_clock_remaining_v1_seed42",
              "evaluation_stem": "ppo_v3_clock_remaining_v1_development_seed40042",
              "protocol": "predictive_post_spawn_clock_remaining_v1"},
}
ORDER = (("zero", "train"), ("clock", "train"), ("zero", "evaluate"), ("clock", "evaluate"))
REFERENCES = {
    "original": "ppo_reward_v3_cpa_baseline_holdout_seed40042",
    "control": "ppo_v3_predictive_control_development_seed40042",
    "v1": "ppo_v3_predictive_safety_v1_development_seed40042",
    "geometry": "ppo_v3_predictive_geometry_v2_development_seed40042",
    "synchronized_retry": "ppo_v3_predictive_sync_v1_retry01_development_seed40042",
    "padding": "ppo_v3_sector_padding_v1_development_seed40042",
    "sector": "ppo_v3_sector_features_v1_development_seed40042",
}
RECORDED_ONLY_MODELS = {
    "original": "models/ppo_reward_v3_seed42.zip",
    "control": "models/ppo_v3_predictive_control_seed42.zip",
    "v1": "models/ppo_v3_predictive_safety_v1_seed42.zip",
}
ORIGINAL_CSV_SHA = "aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4"
PINNED = {
    "V3_CLOCK_LAUNCHER_GUARD_V1.md":
        "86d8ba8d8f447ec825407aecb9456c3b47336fe6493cce3bbc20ff1482979508",
    DESIGN: "3266b72e0687e309816375a63377e7f4a7d3fe9d9c4e8461840b8fde7153973c",
    PREPARATION: "f25260c79a2a8bbb535a8130b52d6526c2100093f5e6bb399dc028a7fa37f2a9",
    ANALYSIS: "ac193ba5925672de86149547865e977d1d94d74824c0d45107c50be3f2d03c32",
    "scripts/clock_observation_v1.py":
        "c441bcbbcdac563a29675010430260f9bb3f52810ca73dce5e9558995d46c136",
    "tests/test_clock_observation_v1.py":
        "a59a6fc03593a60f1dc725dd208262c8a5331123cc598044142ae75c7525f958",
    "scripts/run_clock_comparison_v1.py":
        "175c4a12d82f1e8f8cdf209924bc44ba506bd3759ebff0ab5689e0213b653bdd",
    "tests/test_clock_comparison_v1.py":
        "5cdb87b4164e7f91e5486a2e96a7f272417ab7938fa8d6b020320fbee088b7d7",
}
NEW_SOURCES = {name for name in PINNED if name.endswith(".py")} | {
    "scripts/audit_clock_comparison_v1.py", "tests/test_clock_comparison_audit_v1.py",
}
COLUMNS = ["reward", "length", "success", "collision", "travel_time", "min_ttc",
           "unsafe_ttc_events", "safety_interventions"]
METRIC_FIELDS = {
    "mean_reward": "reward", "mean_length": "length", "success_rate": "success",
    "collision_rate": "collision", "mean_travel_time": "travel_time",
    "mean_unsafe_ttc_events": "unsafe_ttc_events",
    "mean_safety_interventions": "safety_interventions",
}
PREDICTIVE = {"horizon": 3.0, "margin": .5, "uncertainty_growth": .25,
              "clearance_scale": 10.0, "speed_scale": 9.0, "max_neighbors": 14}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def artifact_path(root, name):
    name = str(name).replace("\\", "/")
    parts = PurePosixPath(name).parts
    require(parts and not name.startswith("/") and ":" not in name
            and not any(part in ("..", ".git") for part in parts), "Unsafe artifact path")
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*parts)
    require(not path.is_symlink() and path.resolve().is_relative_to(root),
            "Artifact link replaces or escapes the named path")
    return path


def check_hashes(root, hashes):
    for name, expected in hashes.items():
        require(is_digest(expected), f"Invalid fingerprint: {name}")
        path = Path(name) if Path(name).is_absolute() else artifact_path(root, name)
        require(sha(path) == expected, f"Evidence bytes changed: {name}")


def read_json(path, *, summary=False):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    require(type(result) is dict, "Expected JSON object")

    def check(value, location=()):
        if isinstance(value, dict):
            for key, item in value.items():
                check(item, (*location, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check(item, (*location, index))
        elif isinstance(value, float) and not math.isfinite(value):
            require(summary and location == ("mean_min_ttc",) and value == math.inf,
                    "Unexpected nonfinite JSON value")

    check(result)
    return result


def same(actual, expected):
    """Structural equality that never accepts a boolean as a number or a missing null."""
    if isinstance(expected, dict):
        return (type(actual) is dict and actual.keys() == expected.keys()
                and all(same(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (type(actual) is list and len(actual) == len(expected)
                and all(same(a, b) for a, b in zip(actual, expected)))
    if type(expected) is float:
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


def dated(value):
    require(type(value) is str, "Missing timestamp")
    stamp = datetime.fromisoformat(value)
    require(stamp.utcoffset() is not None, "Timestamp must be timezone-aware")
    return stamp


def artifact_paths(arm):
    require(arm in ARMS, "Unknown clock arm")
    stem, evaluation = ARMS[arm]["stem"], ARMS[arm]["evaluation_stem"]
    return {"model": Path(f"models/{stem}.zip"),
            "training_summary": Path(f"results/{stem}.training.json"),
            "evaluation_csv": Path(f"results/{evaluation}.csv"),
            "evaluation_summary": Path(f"results/{evaluation}.summary.json"),
            "train_record": Path(f"results/{stem}.train.run.json"),
            "evaluate_record": Path(f"results/{stem}.evaluate.run.json"),
            "logdir": Path(f"logs/{stem}")}


def training_arguments(arm):
    paths = artifact_paths(arm)
    return ["--config", CONFIG, "--config-sha256", CONFIG_SHA256,
            "--timesteps", "200000", "--seed", "42", "--learning-rate", "0.0003",
            "--n-steps", "1024", "--batch-size", "64", "--n-envs", "1",
            "--env-seed-stride", "1000", "--eval-seed-offset", "70000",
            "--eval-episodes", "50", "--evaluation-freq", "10000",
            "--checkpoint-freq", "25000", "--summary-output",
            paths["training_summary"].as_posix(), "--output",
            paths["model"].with_suffix("").as_posix(), "--refuse-overwrite"]


def evaluation_arguments(arm, model_hash):
    require(is_digest(model_hash), "Invalid model fingerprint")
    paths = artifact_paths(arm)
    return ["--model", paths["model"].as_posix(), "--model-sha256", model_hash,
            "--config", CONFIG, "--config-sha256", CONFIG_SHA256,
            "--episodes", "500", "--seed", "40042", "--unsafe-ttc", "2.0",
            "--reference-csv", f"results/{REFERENCES['original']}.csv",
            "--reference-csv-sha256", ORIGINAL_CSV_SHA,
            "--output", paths["evaluation_csv"].as_posix(), "--refuse-overwrite"]


def stage_outputs(arm, mode):
    require((arm, mode) in ORDER, "Unknown clock stage")
    paths = artifact_paths(arm)
    keys = (["train_record", "model", "training_summary", "logdir"] if mode == "train"
            else ["evaluate_record", "evaluation_csv", "evaluation_summary"])
    return [paths[key] for key in keys] + [
        Path(f"logs/{ARMS[arm]['stem']}.{mode}.{stream}.log") for stream in ("stdout", "stderr")]


def protocol_description():
    # Intentionally reconstructed here, not imported from the execution runner.
    return json.loads(json.dumps({
        "name": NAME, "order": [list(stage) for stage in ORDER], "arms": ARMS,
        "design_sha256": PINNED[DESIGN], "preparation_sha256": PINNED[PREPARATION],
        "config": CONFIG, "config_sha256": CONFIG_SHA256, "initial_policy": INITIAL_POLICY,
        "observation_shape": [116], "duration": 30, "torch_threads": [8, 8],
        "training_arguments": {arm: training_arguments(arm) for arm in ARMS},
        "evaluation_arguments": {arm: evaluation_arguments(arm, "0" * 64) for arm in ARMS},
        "evaluation_model_hash_placeholder": "Bind to that arm's verified final model SHA256",
        "collected_timesteps_per_arm": 200704, "epoch_updates_per_arm": 1960,
        "internal_validation": {"initial_seed": 70042, "callbacks": 20,
                                "episodes_per_callback": 50, "explicit_reseeding": False},
        "development": {"first_seed": 40042, "last_seed": 40541, "episodes_per_arm": 500},
        "checkpoint_callbacks_per_arm": 8, "final_root_models_only": True,
        "no_resume_retry_or_extra_episodes": True, "native_bootstrap_and_cutoff": "unchanged",
        "decision_rules": "All16 historical gates plus all5 clock-benefit conditions in design",
        "outputs": {f"{a}.{m}": [p.as_posix() for p in stage_outputs(a, m)] for a, m in ORDER},
        "lock": LOCK_PATH.as_posix(),
    }))


def current_fingerprints(root):
    """Independently bind inventory; reuse only frozen low-level runtime/snapshot checks."""
    root = Path(root).resolve(strict=True)
    check_hashes(root, PINNED)
    pins = read_json(artifact_path(root, ANALYSIS))["input_sha256"]
    require(type(pins) is dict and len(pins) == 531, "Historical pin inventory changed")
    check_hashes(root, pins)
    baseline = {p for p in pins if not Path(p).is_absolute() and p.endswith(".py")}
    actual = {p.relative_to(root).as_posix() for folder in ("scripts", "tests", "safeintent_rl")
              for p in (root / folder).rglob("*.py")}
    require(len(baseline) == 95 and actual == baseline | NEW_SOURCES,
            "Require exact 101-source historical/clock inventory")
    fingerprints = {**pins, **PINNED,
                    **{p: sha(artifact_path(root, p)) for p in NEW_SOURCES}}
    # Their source bytes were checked above before these lazy imports.
    import torch

    from scripts import sector_timeout_protocol_v1 as retained

    snapshot = retained.verify_snapshot(root)
    runtime = retained.verify_runtime(root)
    dependencies = retained.dependency_fingerprints()
    old = read_json(artifact_path(root, "configs/v3_sector_timeout_diagnostic_v1.release.json"))
    require(same(runtime, old["runtime"]) and same(dependencies, old["dependency_sha256"])
            and len(dependencies) == 227, "Frozen runtime/dependency identity differs")
    require([torch.get_num_threads(), torch.get_num_interop_threads()] == [8, 8],
            "Frozen Torch threads differ")
    for name, module in tuple(sys.modules.items()):
        if module is not None and (name == "safeintent_rl" or
                                   name.startswith(("safeintent_rl.", "scripts."))):
            suffix = "/__init__.py" if hasattr(module, "__path__") else ".py"
            expected = artifact_path(root, name.replace(".", "/") + suffix)
            require(Path(module.__file__).resolve() == expected, f"Import origin differs: {name}")
    require(Path(__file__).resolve() == artifact_path(root, "scripts/audit_clock_comparison_v1.py"),
            "Auditor imported from a different repository")
    return {"input_sha256": dict(sorted(fingerprints.items())), "runtime": runtime,
            "dependency_sha256": dependencies,
            "snapshot_archive_sha256": snapshot["archive_fingerprint"]["sha256"]}


def verify_release(root, digest):
    require(is_digest(digest), "Require exact lowercase release SHA256")
    path = artifact_path(root, RELEASE_PATH)
    require(sha(path) == digest, "Release hash differs")
    release = read_json(path)
    current = current_fingerprints(root)
    require(set(release) == {"schema_version", "status", "created_utc", "protocol",
                             "validation", *current}, "Release schema differs")
    require(same(release["schema_version"], 1) and release["status"] == "released"
            and same(release["protocol"], protocol_description())
            and all(same(release[k], v) for k, v in current.items()), "Released protocol differs")
    validation = release["validation"]
    require(set(validation) == {"ruff", "pytest", "independent_review", "tests_passed",
                                "completed_utc", "source_head", "verification_sha256"},
            "Release validation schema differs")
    require(all(validation[k] == "passed" for k in ("ruff", "pytest", "independent_review"))
            and type(validation["tests_passed"]) is int and validation["tests_passed"] >= 1129
            and isinstance(validation["source_head"], str)
            and re.fullmatch(r"[0-9a-f]{40}", validation["source_head"]) is not None,
            "Missing full-test/review evidence")
    require(dated(validation["completed_utc"]) <= dated(release["created_utc"]),
            "Release predates verification")
    proof_path = artifact_path(root, VERIFICATION_PATH)
    require(is_digest(validation["verification_sha256"])
            and sha(proof_path) == validation["verification_sha256"], "Proof hash differs")
    proof = {"protocol": protocol_description(), **current,
             "validation": {k: v for k, v in validation.items() if k != "verification_sha256"}}
    require(same(read_json(proof_path), proof), "Proof does not bind exact release evidence")
    require(sha(path) == digest, "Release changed during verification")
    return release


def provenance_for(release, digest):
    return {"release_sha256": digest, "input_sha256": release["input_sha256"],
            "runtime": release["runtime"], "config_sha256": CONFIG_SHA256}


def validate_rows(rows):
    require(len(rows) == 500, "Require exactly 500 episode rows")
    for row in rows:
        require(type(row) is dict and set(row) == set(COLUMNS), "Episode schema differs")
        for key in ("success", "collision"):
            require(type(row[key]) is bool, "Invalid outcome boolean")
        require(not (row["success"] and row["collision"]), "Overlapping outcomes")
        for key in set(COLUMNS) - {"success", "collision"}:
            value = row[key]
            require(type(value) in (int, float), "Invalid numeric metric")
            require(math.isfinite(value) or (key == "min_ttc" and value == math.inf),
                    "Invalid nonfinite episode metric")
        length = row["length"]
        require(float(length).is_integer() and 1 <= length <= 151, "Invalid length")
        require(math.isclose(row["travel_time"], length / 5, rel_tol=0, abs_tol=1e-12),
                "Travel time differs from length/5")
        require(row["min_ttc"] >= 0, "Negative TTC")
        unsafe = row["unsafe_ttc_events"]
        require(float(unsafe).is_integer() and 0 <= unsafe <= length, "Invalid unsafe-event count")
        require((unsafe > 0) == (row["min_ttc"] <= 2.0), "TTC/unsafe-event evidence conflicts")
        require(row["safety_interventions"] == 0, "Unexpected safety intervention")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == COLUMNS, "CSV columns differ")
        raw = list(reader)
    require(len(raw) == 500, "Require exactly 500 episode rows")
    rows = []
    for source in raw:
        require(set(source) == set(COLUMNS) and all(type(v) is str and v for v in source.values()),
                "Malformed or missing CSV value")
        row = {}
        for key, value in source.items():
            if key in ("success", "collision"):
                require(value in ("True", "False"), "Invalid CSV boolean")
                row[key] = value == "True"
            else:
                row[key] = float(value)
        rows.append(row)
    validate_rows(rows)
    return rows


def metrics(rows):
    validate_rows(rows)
    finite = [r["min_ttc"] for r in rows if math.isfinite(r["min_ttc"])]
    computed = {key: statistics.fmean(r[field] for r in rows)
                for key, field in METRIC_FIELDS.items()}
    # None is report-only unavailable, never a replacement observation or estimator.
    computed.update(episodes=500.0, mean_min_ttc=statistics.fmean(finite) if finite else None)
    counts = {key: sum(r[key] for r in rows) for key in ("success", "collision")}
    counts.update(incomplete=500 - counts["success"] - counts["collision"],
                  finite_ttc=len(finite), excluded_nonfinite_ttc=500 - len(finite))
    return computed, counts


def favorable(rescues, regressions, p_value):
    require(type(rescues) is int and type(regressions) is int
            and rescues >= 0 and regressions >= 0 and rescues + regressions <= 500,
            "Invalid discordance counts")
    require(type(p_value) in (int, float) and math.isfinite(p_value) and 0 <= p_value <= 1,
            "Invalid exact p-value")
    return rescues > regressions and p_value < .05


def paired(reference, candidate):
    validate_rows(reference)
    validate_rows(candidate)
    rescues = sum(not r["success"] and c["success"] for r, c in zip(reference, candidate))
    regressions = sum(r["success"] and not c["success"] for r, c in zip(reference, candidate))
    n = rescues + regressions
    p_value = (min(1.0, 2 * sum(math.comb(n, k) for k in range(min(rescues, regressions) + 1))
                   / 2**n) if n else 1.0)
    return {"rescues": rescues, "regressions": regressions, "p_exact_two_sided": p_value,
            "favorable": favorable(rescues, regressions, p_value)}


def finite_ttc(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def historical_gates(counts, mean_ttc, comparisons):
    def ttc_at_least(threshold):
        return finite_ttc(mean_ttc) and mean_ttc >= threshold

    return {
        "absolute_success": counts["success"] >= 315,
        "absolute_collision": counts["collision"] <= 174,
        "incomplete": counts["incomplete"] <= 10,
        "original_ttc": ttc_at_least(.6003656548142169),
        "original_paired_success": comparisons["original"]["favorable"],
        "control_success": counts["success"] >= 304,
        "control_collision": counts["collision"] <= 195,
        "control_ttc": ttc_at_least(.6065401801078688),
        "control_paired_success": comparisons["control"]["favorable"],
        "v1_success": counts["success"] >= 364,
        "v1_collision": counts["collision"] <= 101,
        "v1_ttc": ttc_at_least(.6728939419963959),
        "v1_paired_success": comparisons["v1"]["favorable"],
        "geometry_success": counts["success"] >= 394,
        "geometry_collision": counts["collision"] <= 98,
        "geometry_incomplete": counts["incomplete"] <= 8,
    }


def clock_benefit_gates(clock_gates, zero_evidence, clock_evidence, comparison):
    zero_ttc = zero_evidence["metrics"]["mean_min_ttc"]
    clock_ttc = clock_evidence["metrics"]["mean_min_ttc"]
    require(len(clock_gates) == 16 and all(type(v) is bool for v in clock_gates.values()),
            "Require all 16 historical boolean gates")
    return {
        "all_historical_gates": all(clock_gates.values()),
        "paired_success_vs_zero": comparison["favorable"],
        "collision_vs_zero": clock_evidence["counts"]["collision"] <=
        zero_evidence["counts"]["collision"],
        "incomplete_vs_zero": clock_evidence["counts"]["incomplete"] <=
        zero_evidence["counts"]["incomplete"],
        "ttc_vs_zero": finite_ttc(clock_ttc) and finite_ttc(zero_ttc) and clock_ttc >= zero_ttc,
    }


def summary_contract(arm, mode, provenance, model_hash):
    require((arm, mode) in ORDER and is_digest(model_hash), "Invalid summary identity")
    paths = artifact_paths(arm)
    expected = {
        "config_path": CONFIG, "config_sha256": CONFIG_SHA256,
        "model_path": paths["model"].as_posix(), "model_sha256": model_hash,
        "observation_protocol": ARMS[arm]["protocol"], "clock_comparison_arm": arm,
        "clock_duration": 30, "initial_policy": INITIAL_POLICY, "clock_provenance": provenance,
        "safety_shield": False, "risk_fusion": False, "target_speed_observation": False,
        "target_speed_scale": None, "intent_model_path": None, "intent_model_sha256": None,
        "intent_neighbors": 0, "intent_history_length": None,
        "intent_history_tracking_neighbors": None, "intent_device": None,
        "collision_first_reward": True, "fusion_neighbors": 0,
        "fusion_features_per_neighbor": 0, "fusion_range_scale": None,
        "fusion_relative_speed_scale": None, "fusion_ttc_scale": None,
        "fusion_cpa_horizon": None, "fusion_cpa_distance_scale": None,
        "predictive_safety_observation": PREDICTIVE,
    }
    if mode == "train":
        expected.update(algorithm="PPO", training_seed=42, internal_evaluation_seed_offset=70000,
                        timesteps_requested=200000, timesteps_collected=200704, learning_rate=.0003,
                        n_steps=1024, batch_size=64, n_envs=1, env_seed_stride=1000,
                        training_seed_offsets=[0], training_initial_seeds=[42], rollout_size=1024,
                        eval_episodes=50, evaluation_freq_timesteps=10000,
                        evaluation_callback_frequency=10000, checkpoint_freq_timesteps=25000,
                        checkpoint_callback_frequency=25000, gamma=.99, gae_lambda=.95,
                        ent_coef=.01, policy_network=[256, 256], observation_shape=[116],
                        ttc_threshold=2.0)
    else:
        expected.update(episodes=500.0, first_seed=40042, last_seed=40541,
                        unsafe_ttc_threshold=2.0, safety_shield_type=None, cpa_time_threshold=None,
                        cpa_distance_threshold=None, cpa_horizon=None, cpa_max_range=None,
                        cpa_override_action=None, mean_safety_interventions=0.0,
                        reference_csv_path=f"results/{REFERENCES['original']}.csv",
                        reference_csv_sha256=ORIGINAL_CSV_SHA)
    return expected


def checked_summary(summary, arm, mode, provenance, model_hash):
    expected = summary_contract(arm, mode, provenance, model_hash)
    actual = dict(summary)
    for key in ("config_path", "model_path", "reference_csv_path"):
        if key in actual:
            require(type(actual[key]) is str, "Invalid summary path")
            actual[key] = actual[key].replace("\\", "/")
    metric_keys = set(METRIC_FIELDS) | {"episodes", "mean_min_ttc"}
    if mode == "evaluate":
        require(metric_keys <= actual.keys(), "Missing evaluation metric")
        for key in metric_keys:
            value = actual[key]
            require(type(value) in (int, float)
                    and (math.isfinite(value) or (key == "mean_min_ttc" and value == math.inf)),
                    "Invalid summary metric type/value")
            if key not in expected:
                del actual[key]
    require(same(actual, expected), "Summary scientific schema/protocol differs")


def verify_model(model, arm, provenance):
    """Independent checkpoint structure/tensor checks; never infer an action or learn."""
    import gymnasium as gym
    import numpy as np
    import torch

    require(arm in ARMS, "Unknown checkpoint arm")
    expected = {
        "num_timesteps": 200704, "seed": 42, "n_steps": 1024, "batch_size": 64,
        "n_envs": 1, "n_epochs": 10, "learning_rate": .0003, "gamma": .99,
        "gae_lambda": .95, "ent_coef": .01, "vf_coef": .5, "max_grad_norm": .5,
        "target_kl": None, "normalize_advantage": True, "clip_range_vf": None,
        "policy_kwargs": {"net_arch": [256, 256]}, "_n_updates": 1960,
        "safeintent_clock_arm": arm, "safeintent_clock_duration": 30,
        "safeintent_observation_protocol": ARMS[arm]["protocol"],
        "safeintent_initial_policy": INITIAL_POLICY, "safeintent_clock_provenance": provenance,
    }
    require(all(same(getattr(model, k, object()), v) for k, v in expected.items()),
            "Checkpoint PPO/provenance differs")
    require(str(model.device) == "cpu" and model.clip_range(1) == .2,
            "Checkpoint device/clip differs")
    obs, action = model.observation_space, model.action_space
    require(isinstance(obs, gym.spaces.Box) and obs.shape == (116,) and obs.dtype == np.float32
            and isinstance(action, gym.spaces.Discrete) and action.n == 3 and action.start == 0,
            "Checkpoint observation/action space differs")
    low = [-math.inf] * 105 + [0] + [-1, 0, 0] * 3 + [0]
    high = [math.inf] * 105 + [1] * 11
    require(np.array_equal(obs.low, low) and np.array_equal(obs.high, high),
            "Checkpoint observation bounds differ")
    shapes = {
        "mlp_extractor.policy_net.0.weight": (256, 116),
        "mlp_extractor.policy_net.0.bias": (256,),
        "mlp_extractor.policy_net.2.weight": (256, 256),
        "mlp_extractor.policy_net.2.bias": (256,),
        "mlp_extractor.value_net.0.weight": (256, 116),
        "mlp_extractor.value_net.0.bias": (256,),
        "mlp_extractor.value_net.2.weight": (256, 256),
        "mlp_extractor.value_net.2.bias": (256,),
        "action_net.weight": (3, 256), "action_net.bias": (3,),
        "value_net.weight": (1, 256), "value_net.bias": (1,),
    }
    state = model.policy.state_dict()
    require(set(state) == set(shapes), "Checkpoint tensor inventory differs")
    for name, shape in shapes.items():
        tensor = state[name]
        require(tuple(tensor.shape) == shape and tensor.dtype == torch.float32
                and str(tensor.device) == "cpu" and bool(torch.isfinite(tensor).all()),
                f"Invalid checkpoint tensor: {name}")
    require(sum(p.numel() for p in model.policy.parameters()) == 192516,
            "Checkpoint parameter count differs")


def inspect_model(path, arm, provenance, expected_hash):
    # Call only after the externally bound release and cumulative artifacts pass.
    # Loading a trusted local SB3 checkpoint deserializes it; it is not a sandbox
    # for untrusted downloaded models. Tests substitute a fabricated loader.
    from stable_baselines3 import PPO

    require(is_digest(expected_hash), "Require already bound model fingerprint")
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_hash,
            "Checkpoint changed before deserialization")
    # Deserialize only the exact bytes just checked, not a mutable pathname.
    with io.BytesIO(payload) as stream:
        model = PPO.load(stream, device="cpu")
    verify_model(model, arm, provenance)
    require(sha(path) == expected_hash, "Checkpoint changed while loading/inspecting")


def check_test_evidence(tests, release, started, finished):
    require(type(tests) is dict and set(tests) == {"status", "tests_passed", "commands",
                                                 "completed_utc"}, "Test evidence schema differs")
    require(tests["status"] == "passed" and type(tests["tests_passed"]) is int
            and tests["tests_passed"] >= release["validation"]["tests_passed"],
            "Fresh full-test gate missing/regressed")
    require(started <= dated(tests["completed_utc"]) <= finished, "Test gate chronology differs")
    commands = tests["commands"]
    require(type(commands) is list and len(commands) == 2, "Require both full test commands")
    for record, arguments in zip(commands, (["-m", "ruff", "check", "."],
                                           ["-m", "pytest", "-p", "no:cacheprovider"])):
        require(type(record) is dict and set(record) == {"command", "exit_code", "stdout_sha256",
                                                        "stderr_sha256"},
                "Test command evidence schema differs")
        require(same(record["command"], [sys.executable, *arguments])
                and same(record["exit_code"], 0) and is_digest(record["stdout_sha256"])
                and is_digest(record["stderr_sha256"]), "Test command/result differs")


def completed_chain(root, release, digest):
    """Verify every artifact and exact cumulative chain before checkpoint inspection."""
    require(not artifact_path(root, LOCK_PATH).exists(), "Active/failed batch lock remains")
    provenance = provenance_for(release, digest)
    dependencies, fingerprints, records = {}, {}, {}
    previous_finish = dated(release["created_utc"])
    for arm, mode in ORDER:
        paths = artifact_paths(arm)
        summary_key = "training_summary" if mode == "train" else "evaluation_summary"
        record_name = paths[f"{mode}_record"].as_posix()
        record_path = artifact_path(root, record_name)
        record_hash = sha(record_path)
        record = read_json(record_path)
        model_hash = sha(artifact_path(root, paths["model"]))
        summary_path = artifact_path(root, paths[summary_key])
        summary_hash = sha(summary_path)
        summary = read_json(summary_path, summary=mode == "evaluate")
        require(sha(record_path) == record_hash and sha(summary_path) == summary_hash,
                "Stage evidence changed while reading")
        arguments = (training_arguments(arm) if mode == "train"
                     else evaluation_arguments(arm, model_hash))
        expected = {
            "status": "complete", "arm": arm, "mode": mode, "release_sha256": digest,
            "provenance": provenance, "arguments": arguments,
            "expected_initial_policy": INITIAL_POLICY, "initial_policy": INITIAL_POLICY,
            "predecessor_sha256": dict(dependencies), "model_sha256": model_hash,
            "summary_sha256": summary_hash,
        }
        if mode == "evaluate":
            expected["csv_sha256"] = sha(artifact_path(root, paths["evaluation_csv"]))
        require(set(record) == {*expected, "started_utc", "finished_utc", "tests"}
                and all(same(record[k], v) for k, v in expected.items()),
                "Incomplete/changed stage or cumulative predecessor chain")
        started, finished = dated(record["started_utc"]), dated(record["finished_utc"])
        require(previous_finish <= started <= finished, "Four-stage chronology differs")
        previous_finish = finished
        check_test_evidence(record["tests"], release, started, finished)
        checked_summary(summary, arm, mode, provenance, model_hash)
        keys = (("train_record", "model", "training_summary") if mode == "train"
                else ("evaluate_record", "evaluation_csv", "evaluation_summary"))
        stage_hashes = {record_name: record_hash, paths[summary_key].as_posix(): summary_hash}
        if mode == "train":
            stage_hashes[paths["model"].as_posix()] = model_hash
        else:
            stage_hashes[paths["evaluation_csv"].as_posix()] = expected["csv_sha256"]
        require(set(stage_hashes) == {paths[k].as_posix() for k in keys},
                "Stage artifact inventory differs")
        dependencies.update(stage_hashes)
        for log in stage_outputs(arm, mode)[-2:]:
            fingerprints[log.as_posix()] = sha(artifact_path(root, log))
        stdout = artifact_path(root, stage_outputs(arm, mode)[-2])
        lines = [line.strip() for line in stdout.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        require(lines and lines[-1] == f"Clock comparison {arm} {mode} complete: {record_path}",
                "Missing final stage stdout marker; record alone is not completion")
        if mode == "train":
            require(artifact_path(root, paths["logdir"]).is_dir(), "Missing training log directory")
        records[f"{arm}_{mode}"] = record
    fingerprints.update(dependencies)
    require(len(dependencies) == 12, "Incomplete final artifact inventory")
    check_hashes(root, fingerprints)
    for arm in ARMS:
        model_path = artifact_paths(arm)["model"]
        inspect_model(artifact_path(root, model_path), arm, provenance,
                      fingerprints[model_path.as_posix()])
        check_hashes(root, fingerprints)
    return records, fingerprints


def reconcile(summary, rows):
    computed, counts = metrics(rows)
    for key, value in computed.items():
        actual = summary.get(key)
        require(type(actual) in (int, float), f"Invalid summary metric: {key}")
        if value is None:
            require(actual == math.inf, "Unavailable TTC must retain legacy +Infinity summary")
        else:
            require(math.isfinite(actual)
                    and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12),
                    f"Summary metric differs: {key}")
    require(same(summary.get("first_seed"), 40042) and same(summary.get("last_seed"), 40541)
            and same(summary.get("unsafe_ttc_threshold"), 2.0)
            and same(summary.get("safety_shield"), False),
            "Development reset/metric metadata differs")
    return computed, counts


def audit(root, release_sha256):
    root = Path(root).resolve(strict=True)
    release = verify_release(root, release_sha256)
    records, fingerprints = completed_chain(root, release, release_sha256)
    fingerprints.update(release["input_sha256"])
    fingerprints.update({
        RELEASE_PATH.as_posix(): release_sha256,
        VERIFICATION_PATH.as_posix(): release["validation"]["verification_sha256"],
    })
    rows, evidence = {}, {}
    stems = {**REFERENCES, **{arm: spec["evaluation_stem"] for arm, spec in ARMS.items()}}
    for name, stem in stems.items():
        csv_name, summary_name = f"results/{stem}.csv", f"results/{stem}.summary.json"
        # All seven historical CSVs/summaries must be bound by the immutable release,
        # not simply hashed for the first time after seeing their outcomes.
        require(csv_name in fingerprints and summary_name in fingerprints,
                "Unbound comparison evidence")
        check_hashes(root, {p: fingerprints[p] for p in (csv_name, summary_name)})
        rows[name] = read_rows(artifact_path(root, csv_name))
        summary = read_json(artifact_path(root, summary_name), summary=True)
        computed, counts = reconcile(summary, rows[name])
        identities = {}
        for kind in ("model", "config"):
            path = summary[f"{kind}_path"].replace("\\", "/")
            artifact_path(root, path)  # Validate path without requiring an old checkpoint to exist.
            digest = summary[f"{kind}_sha256"]
            require(is_digest(digest), f"Invalid {name} {kind} recorded identity")
            live = path in fingerprints
            if live:
                require(digest == fingerprints[path], f"Conflicting {name} {kind} identity")
            else:
                require(kind == "model" and RECORDED_ONLY_MODELS.get(name) == path,
                        f"Unbound {name} {kind} identity")
            identities[kind] = {"path": path, "sha256": digest, "live_byte_verified": live,
                                "identity_source": "hash_pinned_summary"}
        evidence[name] = {"metrics": computed, "counts": counts,
                          "csv_sha256": fingerprints[csv_name],
                          "summary_sha256": fingerprints[summary_name],
                          "artifact_identity": identities}
    comparisons, historical, decisions = {}, {}, {}
    for arm in ARMS:
        comparisons[arm] = {name: paired(rows[name], rows[arm]) for name in REFERENCES}
        historical[arm] = historical_gates(evidence[arm]["counts"],
                                           evidence[arm]["metrics"]["mean_min_ttc"],
                                           comparisons[arm])
        decisions[arm] = ("eligible_for_separately_preregistered_validation"
                          if all(historical[arm].values()) else "not_retained_under_frozen_gates")
    comparison = paired(rows["zero"], rows["clock"])
    benefit = clock_benefit_gates(historical["clock"], evidence["zero"], evidence["clock"],
                                 comparison)
    require(same(verify_release(root, release_sha256), release), "Release changed during audit")
    require(not artifact_path(root, LOCK_PATH).exists(), "Batch lock appeared during audit")
    check_hashes(root, fingerprints)
    return {
        "status": "verified", "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_sha256": release_sha256, "first_seed": 40042, "last_seed": 40541,
        "rows_per_arm": 500, "reference_arms": list(REFERENCES),
        "pairing": "positional under recorded contiguous reset protocol; no row seed IDs",
        "evidence": evidence, "paired_success": comparisons, "historical_gates": historical,
        "failed_historical_gates": {a: [k for k, v in g.items() if not v]
                                    for a, g in historical.items()},
        "arm_decisions": decisions, "clock_vs_zero": comparison,
        "clock_benefit_gates": benefit,
        "failed_clock_benefit_gates": [k for k, v in benefit.items() if not v],
        "clock_benefit_established_under_frozen_gates": all(benefit.values()),
        "automatic_promotion": False, "completed_stages": list(records),
        "limitations": [
            "Consumed development seeds; one training seed, not independent generalization",
            "No new significance filters or multiple-search correction",
            "Finite episode-minimum TTC mean unchanged; null means unavailable, never a pass",
            "Equal widths and measured initialization do not imply equal effective capacity",
            "Clock information changes no reward, truncation bootstrap, sensor or policy override",
            "Reset identity is inferred from frozen code/arguments; CSVs have no per-row seed IDs",
            "Test exit codes are recorded; the supervisor must separately check OS worker exit",
            "Original/control/V1 model identities come from pinned summaries, not live model bytes",
            "No change to Geometry V2 designation, Sector archival or original V3 preservation",
        ],
        "frozen_sha256": dict(sorted(fingerprints.items())), "audit_source_sha256": sha(__file__),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--refuse-overwrite", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not is_digest(args.release_sha256):
        parser.error("--release-sha256 requires an exact lowercase SHA256")
    return args


def main():
    args = parse_args()
    root = Path.cwd().resolve(strict=True)
    output = artifact_path(root, OUTPUT)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve existing clock audit report")
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    report = audit(root, args.release_sha256)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # Re-resolve after audit; exclusive creation preserves late-created evidence.
    with artifact_path(root, OUTPUT).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    print(f"Saved independent clock audit to {output}")


if __name__ == "__main__":
    main()
