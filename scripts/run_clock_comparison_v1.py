"""Release-gated, dedicated-process adapter for the frozen two-arm clock study.

One invocation executes exactly one stage. No release/proof exists yet: importing
this module or implementing it does not release training. Historical mains,
observations, rewards, resets and metrics are not edited. Read the complete
V3_CLOCK_OBSERVATION_V1.md and append-only MILESTONES.md before future use.
"""

from __future__ import annotations

import argparse
import builtins
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from safeintent_rl.config import load_config
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
from scripts import evaluate_policy, sector_timeout_protocol_v1, train_ppo
from scripts.audit_geometry_development import REFERENCE_HASHES, STEMS
from scripts.clock_observation_v1 import ClockPredictiveObservation
from scripts.run_sector_comparison_v1 import initial_policy_fingerprint

NAME = "v3_clock_comparison_v1"
CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"
CONFIG_SHA256 = "a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7"
DESIGN = "V3_CLOCK_OBSERVATION_V1.md"
PREPARATION = "results/v3_clock_observation_v1.preparation.json"
ANALYSIS = "results/v3_sector_timeout_diagnostic_v1.analysis.json"
RELEASE_PATH = Path(f"configs/{NAME}.release.json")
VERIFICATION_PATH = Path(f"results/{NAME}.release_verification.json")
LOCK_PATH = Path(f"logs/{NAME}.active.lock")
INITIAL_POLICY = {"parameter_count": 192516,
                  "sha256": "d1c314ef787fa038568eeaff5dfc6c1a3851aba5f39a4dd8ba767634943a57ec"}
ARMS = {
    "zero": {"stem": "ppo_v3_clock_zero_v1_seed42",
             "evaluation_stem": "ppo_v3_clock_zero_v1_development_seed40042",
             "protocol": "predictive_post_spawn_clock_zero_v1"},
    "clock": {"stem": "ppo_v3_clock_remaining_v1_seed42",
              "evaluation_stem": "ppo_v3_clock_remaining_v1_development_seed40042",
              "protocol": "predictive_post_spawn_clock_remaining_v1"},
}
ORDER = (("zero", "train"), ("clock", "train"), ("zero", "evaluate"), ("clock", "evaluate"))
PINNED = {
    DESIGN: "3266b72e0687e309816375a63377e7f4a7d3fe9d9c4e8461840b8fde7153973c",
    PREPARATION: "f25260c79a2a8bbb535a8130b52d6526c2100093f5e6bb399dc028a7fa37f2a9",
    ANALYSIS: "ac193ba5925672de86149547865e977d1d94d74824c0d45107c50be3f2d03c32",
    "scripts/clock_observation_v1.py":
        "c441bcbbcdac563a29675010430260f9bb3f52810ca73dce5e9558995d46c136",
    "tests/test_clock_observation_v1.py":
        "a59a6fc03593a60f1dc725dd208262c8a5331123cc598044142ae75c7525f958",
}
# The future auditor/test must exist and be independently reviewed before release.
NEW_SOURCES = {
    "scripts/clock_observation_v1.py", "tests/test_clock_observation_v1.py",
    "scripts/run_clock_comparison_v1.py", "tests/test_clock_comparison_v1.py",
    "scripts/audit_clock_comparison_v1.py", "tests/test_clock_comparison_audit_v1.py",
}


def sha(path):
    return sector_timeout_protocol_v1.sha(Path(path))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(root, name):
    """Only named repository-relative artifacts; never follow an escaping link."""
    name = Path(name).as_posix()
    parts = PurePosixPath(name).parts
    _require(parts and not name.startswith("/") and ":" not in name
             and not any(part in ("..", ".git") for part in parts), "Unsafe artifact path")
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*parts)
    _require(not path.is_symlink() and path.resolve().is_relative_to(root),
             "Artifact link escapes or replaces the named path")
    return path


def _read(path):
    return sector_timeout_protocol_v1._json(Path(path))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _write_record(path, record, *, exclusive=False, allow_nonfinite=False):
    # Existing writes are only to this invocation's exclusively created record
    # or the summary just produced by its unchanged historical main.
    with Path(path).open("x" if exclusive else "r+", encoding="utf-8", newline="\n") as out:
        json.dump(record, out, indent=2, allow_nan=allow_nonfinite)
        out.write("\n")
        out.truncate()
        out.flush()
        os.fsync(out.fileno())


def artifact_paths(arm):
    _require(arm in ARMS, "Unknown clock arm")
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
    _require(_digest(model_hash), "Invalid final model fingerprint")
    paths = artifact_paths(arm)
    return ["--model", paths["model"].as_posix(), "--model-sha256", model_hash,
            "--config", CONFIG, "--config-sha256", CONFIG_SHA256,
            "--episodes", "500", "--seed", "40042", "--unsafe-ttc", "2.0",
            "--reference-csv", f"results/{STEMS['original']}.csv",
            "--reference-csv-sha256", REFERENCE_HASHES["original"][0],
            "--output", paths["evaluation_csv"].as_posix(), "--refuse-overwrite"]


def stage_outputs(arm, mode):
    _require((arm, mode) in ORDER, "Unknown clock stage")
    paths = artifact_paths(arm)
    keys = (["train_record", "model", "training_summary", "logdir"] if mode == "train"
            else ["evaluate_record", "evaluation_csv", "evaluation_summary"])
    return [paths[key] for key in keys] + [
        Path(f"logs/{ARMS[arm]['stem']}.{mode}.{stream}.log") for stream in ("stdout", "stderr")]


def protocol_description():
    return copy.deepcopy({"name": NAME, "order": [list(stage) for stage in ORDER], "arms": ARMS,
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
            "lock": LOCK_PATH.as_posix()})


def current_fingerprints(root):
    """Hash-only old evidence check, no CSV analysis or replay. Fail closed on new sources."""
    root = Path(root).resolve(strict=True)
    for name, expected in PINNED.items():
        _require(sha(_path(root, name)) == expected, f"Frozen clock input changed: {name}")
    pins = dict(_read(_path(root, ANALYSIS))["input_sha256"])
    _require(len(pins) == 531, "Completed analysis pin inventory changed")
    for name, digest in pins.items():
        path = Path(name) if Path(name).is_absolute() else _path(root, name)
        _require(sha(path) == digest, f"Historical input changed: {name}")
    baseline = {name for name in pins if not Path(name).is_absolute() and name.endswith(".py")}
    actual = {p.relative_to(root).as_posix() for folder in ("scripts", "tests", "safeintent_rl")
              for p in (root / folder).rglob("*.py")}
    _require(len(baseline) == 95 and actual == baseline | NEW_SOURCES,
             "Require exact historical/clock inventory including reviewed auditor/tests")
    pins.update(PINNED)
    pins.update({name: sha(_path(root, name)) for name in NEW_SOURCES})
    snapshot = sector_timeout_protocol_v1.verify_snapshot(root)
    runtime = sector_timeout_protocol_v1.verify_runtime(root)
    dependencies = sector_timeout_protocol_v1.dependency_fingerprints()
    old_release = _read(_path(root, "configs/v3_sector_timeout_diagnostic_v1.release.json"))
    _require(dependencies == old_release["dependency_sha256"], "Frozen dependency bytes changed")
    _require(runtime == old_release["runtime"], "Frozen package origins/runtime changed")
    _require([torch.get_num_threads(), torch.get_num_interop_threads()] == [8, 8],
             "Frozen Torch threads changed")
    _require(Path(__file__).resolve() == _path(root, "scripts/run_clock_comparison_v1.py"),
             "Runner imported from a different repository")
    for name, module in tuple(sys.modules.items()):
        if module is not None and (name == "safeintent_rl" or
                                   name.startswith(("safeintent_rl.", "scripts."))):
            suffix = "/__init__.py" if hasattr(module, "__path__") else ".py"
            expected_origin = _path(root, name.replace(".", "/") + suffix)
            _require(Path(module.__file__).resolve() == expected_origin,
                     f"Imported execution origin changed: {name}")
    return {"input_sha256": dict(sorted(pins.items())), "runtime": runtime,
            "dependency_sha256": dependencies,
            "snapshot_archive_sha256": snapshot["archive_fingerprint"]["sha256"]}


def verify_release(root, digest):
    _require(_digest(digest), "An exact lowercase release SHA256 is required")
    release_path = _path(root, RELEASE_PATH)
    _require(sha(release_path) == digest, "Release hash differs")
    release = _read(release_path)
    current = current_fingerprints(root)
    _require(set(release) == {"schema_version", "status", "created_utc", "protocol",
                             "validation", *current}, "Release schema differs")
    _require(type(release["schema_version"]) is int and release["schema_version"] == 1
             and release["status"] == "released" and release["protocol"] == protocol_description(),
             "No valid release for this exact clock protocol")
    _require(datetime.fromisoformat(release["created_utc"]).utcoffset() is not None,
             "Require dated release evidence")
    _require(all(release[key] == value for key, value in current.items()),
             "Released inputs changed")
    validation = release["validation"]
    _require(all(validation.get(k) == "passed" for k in ("ruff", "pytest", "independent_review"))
             and type(validation.get("tests_passed")) is int and validation["tests_passed"] >= 927
             and isinstance(validation.get("source_head"), str) and bool(validation["source_head"])
             and datetime.fromisoformat(validation["completed_utc"]).utcoffset() is not None,
             "Require full recorded tests and independent release review")
    proof_path = _path(root, VERIFICATION_PATH)
    _require(_digest(validation.get("verification_sha256"))
             and sha(proof_path) == validation["verification_sha256"], "Proof hash differs")
    expected_proof = {"protocol": protocol_description(), **current,
                      "validation": {k: v for k, v in validation.items()
                                     if k != "verification_sha256"}}
    _require(_read(proof_path) == expected_proof, "Proof does not bind the exact inputs and tests")
    return release


def _provenance(release, digest):
    return copy.deepcopy({"release_sha256": digest, "input_sha256": release["input_sha256"],
                          "runtime": release["runtime"], "config_sha256": CONFIG_SHA256})


def _verify_parameters(model, arm, timesteps):
    expected = {"num_timesteps": timesteps, "seed": 42, "n_steps": 1024, "batch_size": 64,
                "n_envs": 1, "n_epochs": 10, "learning_rate": .0003, "gamma": .99,
                "gae_lambda": .95, "ent_coef": .01, "vf_coef": .5, "max_grad_norm": .5,
                "target_kl": None, "normalize_advantage": True, "clip_range_vf": None,
                "policy_kwargs": {"net_arch": [256, 256]}, "_n_updates": 1960 if timesteps else 0}
    _require(timesteps in (0, 200704) and arm in ARMS, "Unregistered checkpoint stage")
    _require(all(getattr(model, key, object()) == value for key, value in expected.items()),
             "Checkpoint PPO parameters differ")
    _require(isinstance(model.observation_space, gym.spaces.Box)
             and isinstance(model.action_space, gym.spaces.Discrete)
             and model.observation_space.shape == (116,)
             and model.observation_space.dtype == np.float32
             and model.action_space.n == 3 and model.action_space.start == 0
             and model.clip_range(1) == .2 and str(model.device) == "cpu"
             and getattr(model, "safeintent_clock_arm", None) == arm
             and getattr(model, "safeintent_observation_protocol", None) == ARMS[arm]["protocol"]
             and getattr(model, "safeintent_clock_duration", None) == 30,
             "Checkpoint clock identity, dimensions or device differs")
    low = np.concatenate([np.full(105, -np.inf), [0], np.tile([-1, 0, 0], 3), [0]])
    high = np.concatenate([np.full(105, np.inf), np.ones(11)])
    _require(np.array_equal(model.observation_space.low, low)
             and np.array_equal(model.observation_space.high, high), "Checkpoint Box bounds differ")


def verify_model(model, arm, provenance):
    _verify_parameters(model, arm, 200704)
    _require(getattr(model, "safeintent_initial_policy", None) == INITIAL_POLICY
             and getattr(model, "safeintent_clock_provenance", None) == provenance
             and initial_policy_fingerprint(model)["parameter_count"] == 192516,
             "Checkpoint initialization/source/runtime provenance differs")


def _verify_summary(summary, arm, mode, provenance):
    paths = artifact_paths(arm)
    expected = {"config_path": str(Path(CONFIG)), "config_sha256": CONFIG_SHA256,
                "model_path": str(paths["model"]), "observation_protocol": ARMS[arm]["protocol"],
                "clock_comparison_arm": arm, "clock_duration": 30,
                "initial_policy": INITIAL_POLICY, "clock_provenance": provenance,
                "safety_shield": False, "risk_fusion": False, "target_speed_observation": False,
                "target_speed_scale": None, "intent_model_path": None, "intent_model_sha256": None,
                "intent_neighbors": 0, "intent_history_length": None,
                "intent_history_tracking_neighbors": None, "intent_device": None,
                "collision_first_reward": True, "fusion_neighbors": 0,
                "fusion_features_per_neighbor": 0, "fusion_range_scale": None,
                "fusion_relative_speed_scale": None, "fusion_ttc_scale": None,
                "fusion_cpa_horizon": None, "fusion_cpa_distance_scale": None,
                "predictive_safety_observation": load_config(CONFIG)[
                    "predictive_safety_observation"]}
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
                        reference_csv_path=str(Path(f"results/{STEMS['original']}.csv")),
                        reference_csv_sha256=REFERENCE_HASHES["original"][0])
    _require(all(key in summary and summary[key] == value for key, value in expected.items()),
             "Summary scientific protocol differs")


def _absent(root, paths):
    for relative in paths:
        path = _path(root, relative)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Preserve existing clock artifact: {path}")


def check_predecessors(root, arm, mode, release, digest):
    index = ORDER.index((arm, mode))
    for future_arm, future_mode in ORDER[index + 1:]:
        _absent(root, stage_outputs(future_arm, future_mode))
    dependencies = {}
    provenance = _provenance(release, digest)
    for prior_arm, prior_mode in ORDER[:index]:
        paths = artifact_paths(prior_arm)
        record = _read(_path(root, paths[f"{prior_mode}_record"]))
        key = "training_summary" if prior_mode == "train" else "evaluation_summary"
        summary = _read(_path(root, paths[key]))
        model_hash = sha(_path(root, paths["model"]))
        arguments = (training_arguments(prior_arm) if prior_mode == "train"
                     else evaluation_arguments(prior_arm, model_hash))
        expected = {"status": "complete", "arm": prior_arm, "mode": prior_mode,
                    "release_sha256": digest, "provenance": provenance,
                    "arguments": arguments, "initial_policy": INITIAL_POLICY,
                    "predecessor_sha256": dependencies, "model_sha256": model_hash,
                    "summary_sha256": sha(_path(root, paths[key]))}
        _require(all(record.get(k) == v for k, v in expected.items())
                 and "error" not in record and bool(record.get("finished_utc"))
                 and record.get("tests", {}).get("status") == "passed"
                 and summary.get("model_sha256") == model_hash, "Incomplete/changed predecessor")
        _verify_summary(summary, prior_arm, prior_mode, provenance)
        if prior_mode == "train":
            model_path = _path(root, paths["model"])
            model = PPO.load(model_path, device="cpu")
            _require(sha(model_path) == model_hash, "Predecessor changed while loading")
            verify_model(model, prior_arm, provenance)
        else:
            _require(record.get("csv_sha256") == sha(_path(root, paths["evaluation_csv"])),
                     "Predecessor CSV changed")
        keys = (("train_record", "model", "training_summary") if prior_mode == "train"
                else ("evaluate_record", "evaluation_csv", "evaluation_summary"))
        dependencies.update({paths[k].as_posix(): sha(_path(root, paths[k])) for k in keys})
    return dependencies


def no_active_research(root):
    """Fail closed if Windows cannot enumerate competing research workers."""
    script = (
        "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
        "Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine) | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            cwd=root, check=True, capture_output=True, text=True)
    rows = json.loads(result.stdout or "[]")
    if isinstance(rows, dict):
        rows = [rows]
    own_pids = {os.getpid()}
    own_rows = [r for r in rows if int(r["ProcessId"]) == os.getpid()]
    _require(len(own_rows) == 1, "Process enumeration omitted or duplicated the current worker")
    current = own_rows[0]
    parent = next((r for r in rows if int(r["ProcessId"]) == os.getppid()), None)
    # Windows venv redirector can wait as a parent with the exact same command.
    # Exempt only that proven launcher, never arbitrary Python ancestors.
    launcher = Path(root) / ".venv" / "Scripts" / "python.exe"
    if (current and parent and int(current.get("ParentProcessId", -1)) == os.getppid()
            and current.get("CommandLine") == parent.get("CommandLine")
            and parent.get("ExecutablePath")
            and Path(parent["ExecutablePath"]).resolve() == launcher.resolve()):
        own_pids.add(os.getppid())
    pattern = r"(?:train_ppo|evaluate_policy|run_\w*(?:comparison|diagnostic|predictive|retry))"
    for row in rows:
        _require(isinstance(row.get("CommandLine"), str), "Cannot inspect a Python process")
        if int(row["ProcessId"]) not in own_pids and re.search(pattern, row["CommandLine"]):
            raise RuntimeError(f"Research worker already active: PID {row['ProcessId']}")


def fresh_test_gate(root):
    records = []
    for arguments in (["-m", "ruff", "check", "."], ["-m", "pytest", "-p", "no:cacheprovider"]):
        result = subprocess.run([sys.executable, *arguments], cwd=root,
                                capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        records.append({"command": [sys.executable, *arguments], "exit_code": result.returncode,
                        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest()})
        _require(result.returncode == 0, "Fresh full test gate failed; preserve logs and lock")
    matches = re.findall(r"\b(\d+) passed\b", result.stdout)
    _require(bool(matches), "Pytest did not report a passing test count")
    return {"status": "passed", "tests_passed": int(matches[-1]), "commands": records,
            "completed_utc": _utc()}


@contextmanager
def experiment_lock(root, arm, mode):
    path = _path(root, LOCK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "arm": arm, "mode": mode, "started_utc": _utc()}
    _write_record(path, owner, exclusive=True)
    yield
    # An exception deliberately skips this block. Never remove a failed/replaced lock.
    _require(_read(path) == owner, "Lock ownership changed; preserve lock")
    path.unlink()


@contextmanager
def exclusive_stage_artifact_writes(root, arm, mode):
    """Atomically refuse late-created final outputs without changing callback writes.

    SB3's frozen ZIP writer uses Path.open; summary writing uses Path.write_text,
    and pandas CSV writing uses builtins.open. Guard only their named final paths
    while the historical main runs. No closure is attached to a serialized model.
    """
    paths = artifact_paths(arm)
    keys = ("model", "training_summary") if mode == "train" else (
        "evaluation_csv", "evaluation_summary")
    protected = {(Path(root) / paths[key]).absolute(): paths[key] for key in keys}
    old_path_open, old_open = Path.open, builtins.open

    def guarded_mode(file, requested):
        if isinstance(file, (str, os.PathLike)):
            lexical = Path(file).absolute()
            if lexical in protected:
                path = _path(root, protected[lexical])
                if "+" in requested or requested.startswith("a"):
                    raise ValueError("Final outputs cannot be appended or updated by the main")
                if requested.startswith("w"):
                    if path.exists() or path.is_symlink():
                        raise FileExistsError(f"Preserve late-created output: {path}")
                    return requested.replace("w", "x", 1)
        return requested

    def path_open(path, mode="r", *args, **kwargs):
        return old_path_open(path, guarded_mode(path, mode), *args, **kwargs)

    def builtin_open(file, mode="r", *args, **kwargs):
        return old_open(file, guarded_mode(file, mode), *args, **kwargs)

    try:
        Path.open, builtins.open = path_open, builtin_open
        yield
    finally:
        Path.open, builtins.open = old_path_open, old_open


@contextmanager
def adapted(module, arguments, arm, record, record_path, provenance):
    old_factory, old_ppo, old_argv = module.make_intersection_env, module.PPO, sys.argv
    created, constructors = [], []
    seeds = [42, 70042] if record["mode"] == "train" else [40042]

    def factory(*args, **kwargs):
        _require(not args and len(created) < len(seeds)
                 and kwargs.get("seed") == seeds[len(created)]
                 and kwargs.get("config_path") == CONFIG and not kwargs.get("safety_shield")
                 and not kwargs.get("cpa_safety_shield")
                 and not kwargs.get("risk_fusion") and not kwargs.get("target_speed_observation")
                 and kwargs.get("intent_model") is None, "Factory seed/config/modifier differs")
        inner = old_factory(*args, **kwargs)  # Retain its existing seeded bootstrap exactly once.
        created.append(inner)  # Own cleanup even if either outer constructor fails.
        wrapper = ClockPredictiveObservation(SynchronizedPredictiveObservation(inner), arm=arm)
        created[-1] = wrapper
        return wrapper

    def stamped_ppo(*args, **kwargs):
        _require(not constructors and kwargs.get("device", "cpu") == "cpu",
                 "Require exactly one fresh CPU policy")
        kwargs["device"] = "cpu"
        model = old_ppo(*args, **kwargs)
        constructors.append(model)
        model.safeintent_clock_arm = arm
        model.safeintent_observation_protocol = ARMS[arm]["protocol"]
        model.safeintent_clock_duration = 30
        _verify_parameters(model, arm, 0)
        initial = initial_policy_fingerprint(model)
        record["initial_policy"] = initial
        _write_record(record_path, record)
        _require(initial == INITIAL_POLICY, "Initial tensors differ from measured preparation")
        model.safeintent_initial_policy = dict(initial)
        model.safeintent_clock_provenance = copy.deepcopy(provenance)
        return model  # Only now can the unchanged historical main call learn().

    class CpuEvaluationPPO:
        @staticmethod
        def load(path, **kwargs):
            _require(str(Path(path)) == str(artifact_paths(arm)["model"])
                     and kwargs.get("device", "cpu") == "cpu", "Only this arm's final CPU model")
            expected_hash = arguments[arguments.index("--model-sha256") + 1]
            _require(sha(path) == expected_hash, "Evaluation model bytes changed before loading")
            kwargs["device"] = "cpu"
            model = old_ppo.load(path, **kwargs)
            _require(sha(path) == expected_hash, "Evaluation model bytes changed while loading")
            verify_model(model, arm, provenance)
            return model

    try:
        module.make_intersection_env = factory
        module.PPO = stamped_ppo if record["mode"] == "train" else CpuEvaluationPPO
        sys.argv = [module.__name__, *arguments]
        yield
        _require(len(created) == len(seeds), "Factory bootstrap count differs")
        _require(record["mode"] != "train" or len(constructors) == 1, "Fresh PPO count differs")
    except BaseException as error:
        for env in reversed(created):
            try:
                env.close()
            except BaseException as close_error:
                error.add_note(f"Cleanup failed: {type(close_error).__name__}: {close_error}")
        raise
    finally:
        module.make_intersection_env, module.PPO, sys.argv = old_factory, old_ppo, old_argv


def run(arm, mode, release_sha256, root=None):
    _require((arm, mode) in ORDER, "Only the four preregistered stages are supported")
    root = Path.cwd() if root is None else Path(root)
    root = root.resolve(strict=True)
    _require(root == Path.cwd().resolve(), "Run from the verified repository root")
    release = verify_release(root, release_sha256)  # No outputs/model/env work on missing release.
    no_active_research(root)
    _absent(root, [LOCK_PATH, *stage_outputs(arm, mode)])
    predecessors = check_predecessors(root, arm, mode, release, release_sha256)
    paths = artifact_paths(arm)
    record_path = _path(root, paths[f"{mode}_record"])
    arguments = (training_arguments(arm) if mode == "train"
                 else evaluation_arguments(arm, sha(_path(root, paths["model"]))))
    provenance = _provenance(release, release_sha256)
    record = {"status": "running", "arm": arm, "mode": mode, "started_utc": _utc(),
              "release_sha256": release_sha256, "provenance": provenance, "arguments": arguments,
              "expected_initial_policy": dict(INITIAL_POLICY),
              "initial_policy": dict(INITIAL_POLICY) if mode == "evaluate" else None,
              "predecessor_sha256": predecessors}
    reserved = False
    try:
        with experiment_lock(root, arm, mode):
            _absent(root, stage_outputs(arm, mode))
            record_path.parent.mkdir(parents=True, exist_ok=True)
            _write_record(record_path, record, exclusive=True)
            reserved = True
            stdout_path, stderr_path = [_path(root, p) for p in stage_outputs(arm, mode)[-2:]]
            with stdout_path.open("x", encoding="utf-8", buffering=1) as stdout:
                with stderr_path.open("x", encoding="utf-8", buffering=1) as stderr:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        try:
                            record["tests"] = fresh_test_gate(root)
                            _require(record["tests"]["tests_passed"] >=
                                     release["validation"]["tests_passed"], "Test count regressed")
                            _write_record(record_path, record)
                            _require(verify_release(root, release_sha256) == release,
                                     "Release changed during fresh tests")
                            _require(check_predecessors(root, arm, mode, release, release_sha256)
                                     == predecessors, "Predecessors changed before execution")
                            no_active_research(root)
                            module = train_ppo if mode == "train" else evaluate_policy
                            with adapted(module, arguments, arm, record, record_path, provenance):
                                with exclusive_stage_artifact_writes(root, arm, mode):
                                    module.main()
                            summary_path = _path(root, paths["training_summary" if mode == "train"
                                                             else "evaluation_summary"])
                            summary = _read(summary_path)
                            summary.update(observation_protocol=ARMS[arm]["protocol"],
                                           clock_comparison_arm=arm, clock_duration=30,
                                           initial_policy=record["initial_policy"],
                                           clock_provenance=provenance)
                            _verify_summary(summary, arm, mode, provenance)
                            model_hash = sha(_path(root, paths["model"]))
                            _require(summary["model_sha256"] == model_hash, "Model hash differs")
                            if mode == "train":
                                verify_model(PPO.load(_path(root, paths["model"]), device="cpu"),
                                             arm, provenance)
                                _require(sha(_path(root, paths["model"])) == model_hash,
                                         "Final model bytes changed while loading")
                            else:
                                record["csv_sha256"] = sha(_path(root, paths["evaluation_csv"]))
                            _require(verify_release(root, release_sha256) == release,
                                     "Released inputs changed during execution")
                            _require(all(sha(_path(root, p)) == h for p, h in predecessors.items()),
                                     "Predecessor artifacts changed during execution")
                            _require(sha(_path(root, paths["model"])) == model_hash,
                                     "Current model changed during final checks")
                            if mode == "evaluate":
                                _require(sha(_path(root, paths["evaluation_csv"])) ==
                                         record["csv_sha256"], "Current CSV changed during checks")
                            # Keep the historical numeric Infinity encoding for wholly
                            # unavailable TTC; the independent audit cannot pass that gate.
                            _write_record(summary_path, summary, allow_nonfinite=True)
                            _require(_read(summary_path) == summary, "Summary readback differs")
                            record.update(status="complete", model_sha256=model_hash,
                                          summary_sha256=sha(summary_path), finished_utc=_utc())
                            _write_record(record_path, record)
                            _require(_read(record_path) == record, "Completion readback differs")
                            print(f"Clock comparison {arm} {mode} complete: {record_path}",
                                  flush=True)
                        except BaseException:
                            traceback.print_exc(file=stderr)
                            raise
    except BaseException as error:
        if reserved:
            record.update(status="failed", finished_utc=_utc(),
                          error=f"{type(error).__name__}: {error}")
            try:
                _write_record(record_path, record)
            except BaseException as record_error:
                error.add_note(f"Failed record finalization: {record_error}")
        raise
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("arm", choices=tuple(ARMS))
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--refuse-overwrite", action="store_true", required=True)
    args = parser.parse_args(argv)
    if not _digest(args.release_sha256):
        parser.error("--release-sha256 must be an exact lowercase SHA256")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    run(args.arm, args.mode, args.release_sha256)


if __name__ == "__main__":
    main()
