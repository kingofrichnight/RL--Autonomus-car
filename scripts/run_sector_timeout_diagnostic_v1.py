"""One hash-bound, non-interventional replay of the 111 fixed timeout cases.

No training, resume, retry, alternate seed list or output override. A separately
recorded release with fresh test/review evidence is required before execution.
Read V3_SECTOR_TIMEOUT_DIAGNOSTIC_V1.md and the append-only research record.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from safeintent_rl.evaluation import EpisodeMetrics, detect_collision, detect_success
from safeintent_rl.safety.ttc import minimum_ttc
from scripts import sector_timeout_protocol_v1 as protocol
from scripts.run_sector_comparison_v1 import _verify_parameters
from scripts.sector_timeout_trace_v1 import (
    ObservationCapture,
    captured_predict,
    make_diagnostic_env,
)

NAME = "v3_sector_timeout_diagnostic_v1"
ARM_ORDER = ("geometry", "padding", "sector")
SEEDS = protocol.SEEDS
FIRST_SEED = 40042
CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"
RELEASE_PATH = Path("configs/v3_sector_timeout_diagnostic_v1.release.json")
VERIFICATION_PATH = Path("results/v3_sector_timeout_diagnostic_v1.release_verification.json")
OUTPUT_DIR = Path("logs/v3_sector_timeout_diagnostic_v1")
SUMMARY_PATH = Path("results/v3_sector_timeout_diagnostic_v1.summary.json")
LOCK_PATH = Path("logs/v3_sector_timeout_diagnostic_v1.active.lock")
CONSOLE_PATHS = tuple(Path(f"logs/{NAME}.{kind}.log") for kind in ("stdout", "stderr"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    """Preserve valid infinite step TTC explicitly; never substitute it in metrics."""
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_line(stream, value, *, durable=False):
    stream.write(json.dumps(json_safe(value), allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()
    if durable:
        os.fsync(stream.fileno())


def write_json_x(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def replay_episode(env, model, arm, seed, trace_stream):
    """Retain evaluate_policy's exact action/TTC/reward/termination ordering."""
    with ObservationCapture(env) as capture:
        capture.reset_frame(new_episode=True)
        observation, info = env.reset(seed=seed)
        terminated = truncated = False
        reward_sum, steps, min_ttc, unsafe, interventions = 0.0, 0, math.inf, 0, 0
        while not (terminated or truncated):
            base = env.unwrapped
            pre_step_ttc = minimum_ttc(base.vehicle, list(base.road.vehicles))
            result, probabilities = captured_predict(model, observation)
            action = result[0]
            write_line(trace_stream, {
                "kind": "decision", "arm": arm, "seed": seed,
                "decision_index": steps, "observation_frame_index": steps,
                "proposed_action": int(np.asarray(action).item()),
                "action_probabilities": probabilities, "pre_step_ttc": pre_step_ttc,
                "frame": capture.frame(observation, info),
            })
            capture.reset_frame()
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            steps += 1
            step_ttc = float(info.get("min_ttc", pre_step_ttc))
            min_ttc = min(min_ttc, step_ttc)
            unsafe += int(step_ttc <= 2.0)
            interventions += int(info.get("safety_intervened", False))
            # The action is called executed only after the delegated step returned.
            write_line(trace_stream, {
                "kind": "transition", "arm": arm, "seed": seed,
                "decision_index": steps - 1, "observation_frame_index": steps,
                "executed_action": int(np.asarray(action).item()), "reward": float(reward),
                "metric_ttc": step_ttc, "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
        write_line(trace_stream, {
            "kind": "terminal", "arm": arm, "seed": seed,
            "observation_frame_index": steps,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "frame": capture.frame(observation, info),
        })
        collision = detect_collision(env, info)
        success = detect_success(env, info)
        frequency = float(getattr(env.unwrapped, "config", {}).get("policy_frequency", 1))
        return EpisodeMetrics(reward_sum, steps, success, collision, steps / frequency,
                              min_ttc, unsafe, interventions).as_dict()


def load_model(root, arm):
    model = PPO.load(root / protocol.ARMS[arm]["model"], device="cpu")
    if arm != "geometry":
        _verify_parameters(model, arm, 200704)
    else:
        expected = {"num_timesteps": 200704, "seed": 42, "n_steps": 1024,
                    "batch_size": 64, "n_envs": 1, "_n_updates": 1960,
                    "policy_kwargs": {"net_arch": [256, 256]}}
        if any(getattr(model, key) != value for key, value in expected.items()):
            raise ValueError("Geometry final checkpoint protocol differs")
        if getattr(model, "safeintent_observation_protocol", None) is not None:
            raise ValueError("Historical Geometry must remain unsynchronized")
    size = 115 if arm == "geometry" else 163
    if (str(model.device) != "cpu" or model.observation_space.shape != (size,)
            or model.observation_space.dtype != np.float32 or model.action_space.n != 3
            or any(not torch.isfinite(value).all()
                   for value in model.policy.state_dict().values())):
        raise ValueError("Model device, dimensions or finite-parameter contract differs")
    return model


@contextmanager
def exclusive_trace(path):
    """Retain partial bytes on failure; close gzip before its final hash is computed."""
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                yield stream
        raw.flush()
        os.fsync(raw.fileno())


def execute_cases(root, references, directory, ledger, report):
    for arm in ARM_ORDER:
        report.update(current_arm=arm, current_phase="model_loading")
        model = load_model(root, arm)
        report["current_phase"] = "environment_bootstrap"
        env = make_diagnostic_env(arm, root / CONFIG, bootstrap_seed=40042)
        pending_error = None
        try:
            with exclusive_trace(directory / f"{arm}.trace.jsonl.gz") as traces:
                for seed in SEEDS:
                    reference = references[arm][seed - FIRST_SEED]
                    attempt = {"arm": arm, "seed": seed,
                               "attempt_index": report["attempted_episodes"]}
                    report.update(last_attempt=attempt, current_phase="episode_start")
                    # An attempt is conservatively consumed before its reset, even if reset fails.
                    write_line(ledger, {"kind": "start", **attempt, "utc": utc_now()}, durable=True)
                    report["attempted_episodes"] += 1
                    try:
                        report["current_phase"] = "episode_replay"
                        actual = replay_episode(env, model, arm, seed, traces)
                        report["completed_episodes"] += 1
                        try:
                            deltas = protocol.compare_episode(actual, reference)
                        except ValueError as error:
                            check = {"kind": "mismatch", **attempt, "actual": actual,
                                     "reference": reference, "error": str(error)}
                            report["episode_checks"].append(check)
                            write_line(ledger, check, durable=True)
                            raise
                        check = {"kind": "complete", **attempt, "actual": actual,
                                 "reference": reference, "deltas": deltas}
                        report["episode_checks"].append(check)
                        write_line(ledger, check, durable=True)
                        report["matched_episodes"] += 1
                        print(f"{arm} seed={seed} matched; {report['matched_episodes']}/111",
                              flush=True)
                    except BaseException as error:
                        write_line(ledger, {"kind": "failed", **attempt, "utc": utc_now(),
                                            "error": f"{type(error).__name__}: {error}"},
                                   durable=True)
                        raise
        except BaseException as error:
            pending_error = error
            raise
        finally:
            prior_phase = report["current_phase"]
            try:
                report["current_phase"] = "environment_close"
                env.close()
            except BaseException as error:
                report["close_error"] = f"{type(error).__name__}: {error}"
                if pending_error is None:
                    raise
                report["current_phase"] = prior_phase
            else:
                report["current_phase"] = prior_phase


def protocol_description():
    return {
        "name": NAME, "arm_order": list(ARM_ORDER), "selected_seeds": list(SEEDS),
        "episodes": 111, "inner_bootstrap_seed": 40042, "deterministic": True,
        "config": CONFIG, "cpu": True, "torch_threads": [8, 8],
        "unsafe_ttc_threshold": 2.0, "no_new_step_cap": True,
        "no_shield": True, "no_training": True, "no_resume_or_retry": True,
        "outcome_and_integer_metrics": "exact typed equality",
        "float_tolerances": {"reward": {"abs": 1e-9, "rel": 1e-12},
                             "min_ttc": {"abs": 1e-9, "rel": 1e-12},
                             "travel_time": {"abs": 1e-12, "rel": 0.0}},
        "output_directory": OUTPUT_DIR.as_posix(), "summary": SUMMARY_PATH.as_posix(),
        "lock": LOCK_PATH.as_posix(),
        "console_logs": [path.as_posix() for path in CONSOLE_PATHS],
        "directory_artifacts": ["started.json", "episodes.jsonl", "complete.json",
                                *[f"{arm}.trace.jsonl.gz" for arm in ARM_ORDER]],
        "nonfinite_trace_encoding": "Infinity, -Infinity and NaN strings; metric comparison "
                                    "rejects nonfinite episode minima",
        "interpretation": "Outcome-selected consumed development replays, not extra benchmark "
                          "samples or proven counterfactual safety",
    }


def current_fingerprints(root):
    snapshot = protocol.verify_snapshot(root)
    return {"source_sha256": protocol.fingerprint_inputs(root),
            "dependency_sha256": protocol.dependency_fingerprints(),
            "runtime": protocol.verify_runtime(root),
            "snapshot_archive_sha256": snapshot["archive_fingerprint"]["sha256"]}


def build_release(root, validation):
    """Return prospective release content only; does not create output or run episodes."""
    references = protocol.load_references(root)
    if protocol.selected_seeds(references) != SEEDS:
        raise ValueError("Frozen case selection differs")
    return {"schema_version": 1, "status": "released", "created_utc": utc_now(),
            "protocol": protocol_description(), "validation": validation,
            **current_fingerprints(root)}


def validate_release(root, release_sha256):
    path = root / RELEASE_PATH
    if (len(release_sha256) != 64 or sha(path) != release_sha256
            or Path(__file__).resolve() != root / "scripts/run_sector_timeout_diagnostic_v1.py"
            or Path(protocol.__file__).resolve() != root / "scripts/sector_timeout_protocol_v1.py"):
        raise ValueError("Release fingerprint or executing source location differs")
    release = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {"schema_version", "status", "created_utc", "protocol", "validation",
                     "source_sha256", "dependency_sha256", "runtime", "snapshot_archive_sha256"}
    if (set(release) != expected_keys or release["schema_version"] != 1
            or release["status"] != "released" or release["protocol"] != protocol_description()):
        raise ValueError("Release scientific/output protocol differs")
    validation = release["validation"]
    if (validation.get("ruff") != "passed" or validation.get("pytest") != "passed"
            or validation.get("independent_review") != "passed"
            or type(validation.get("tests_passed")) is not int or validation["tests_passed"] < 570
            or not validation.get("completed_utc") or not validation.get("source_head")):
        raise ValueError("Require recorded fresh full tests and independent review")
    current = current_fingerprints(root)
    if any(release[key] != value for key, value in current.items()):
        raise ValueError("Released source, dependencies, runtime or snapshot changed")
    proof_path = root / VERIFICATION_PATH
    if sha(proof_path) != validation.get("verification_sha256"):
        raise ValueError("Release verification evidence fingerprint differs")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    recorded_validation = {key: value for key, value in validation.items()
                           if key != "verification_sha256"}
    if (set(proof) != {"validation", *current}
            or proof["validation"] != recorded_validation
            or any(proof[key] != value for key, value in current.items())):
        raise ValueError("Tests/review were not recorded for these exact inputs")
    if [torch.get_num_threads(), torch.get_num_interop_threads()] != [8, 8]:
        raise ValueError("Frozen Torch thread settings differ")
    return release


def verify_final_inputs(root, release, release_sha256):
    if sha(root / RELEASE_PATH) != release_sha256:
        raise ValueError("Release changed during replay")
    current = current_fingerprints(root)
    if any(release[key] != value for key, value in current.items()):
        raise ValueError("Source/dependency/runtime/snapshot changed during replay")
    if sha(root / VERIFICATION_PATH) != release["validation"]["verification_sha256"]:
        raise ValueError("Test/review evidence changed during replay")


def verify_ledger(directory, report):
    """Read durable evidence back, not just the in-memory completion list."""
    expected_paths = {"started.json", "episodes.jsonl",
                      *[f"{arm}.trace.jsonl.gz" for arm in ARM_ORDER]}
    paths = list(directory.iterdir())
    if ({path.name for path in paths} != expected_paths
            or any(path.is_symlink() or not path.is_file() for path in paths)):
        raise ValueError("Final diagnostic artifact inventory differs")
    with (directory / "episodes.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    expected = [(arm, seed) for arm in ARM_ORDER for seed in SEEDS]
    if len(records) != 2 * len(expected) or len(report["episode_checks"]) != len(expected):
        raise ValueError("Durable diagnostic ledger length differs")
    for index, (arm, seed) in enumerate(expected):
        start, complete = records[2 * index:2 * index + 2]
        if (set(start) != {"kind", "arm", "seed", "attempt_index", "utc"}
                or start["kind"] != "start" or not isinstance(start["utc"], str)
                or datetime.fromisoformat(start["utc"]).utcoffset() is None
                or set(complete) != {"kind", "arm", "seed", "attempt_index", "actual",
                                     "reference", "deltas"}
                or complete["kind"] != "complete"
                or complete != report["episode_checks"][index]):
            raise ValueError("Durable diagnostic start/completion differs")
        for record in (start, complete):
            if (record["arm"] != arm or type(record["seed"]) is not int
                    or record["seed"] != seed or type(record["attempt_index"]) is not int
                    or record["attempt_index"] != index):
                raise ValueError("Durable diagnostic arm/seed/index differs")
        deltas = protocol.compare_episode(complete["actual"], complete["reference"])
        if complete["deltas"] != deltas:
            raise ValueError("Durable diagnostic metric deltas differ")


def finalize_report(directory, report, summary, primary_error):
    """Keep primary failures if optional artifact hashing also fails."""
    report["finished_utc"] = utc_now()
    report["artifact_sha256"] = {}
    errors = []
    try:
        paths = sorted(directory.iterdir())
    except BaseException as error:
        paths = []
        errors.append(f"inventory: {type(error).__name__}: {error}")
    for path in paths:
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Expected a regular non-symlink artifact")
            report["artifact_sha256"][path.name] = sha(path)
        except BaseException as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    expected_paths = {"started.json", "episodes.jsonl",
                      *[f"{arm}.trace.jsonl.gz" for arm in ARM_ORDER]}
    if report["status"] == "complete" and set(report["artifact_sha256"]) != expected_paths:
        errors.append("Complete report requires the exact final artifact fingerprint inventory")
    if errors:
        report.update(status="failed", finalization_errors=errors)
        if primary_error is None:
            primary_error = ValueError("Final artifact fingerprinting failed")
            report.update(error=str(primary_error), failure_phase="artifact_hashing")
    json.dump(json_safe(report), summary, indent=2, allow_nan=False)
    summary.write("\n")
    summary.flush()
    os.fsync(summary.fileno())
    return primary_error


def run(root, release_sha256):
    root = root.resolve(strict=True)
    directory, summary_path, lock_path = root / OUTPUT_DIR, root / SUMMARY_PATH, root / LOCK_PATH
    for path in (directory, summary_path, lock_path,
                 root / "logs/v3_sector_comparison_v1.active.lock"):
        if path.exists():
            raise FileExistsError(f"Preserve existing run/lock evidence: {path}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # A hard exit deliberately leaves the lock; never delete a stale lock automatically.
    with lock_path.open("x", encoding="utf-8") as lock:
        write_line(lock, {"pid": os.getpid(), "parent_pid": os.getppid(), "utc": utc_now()},
                   durable=True)
    normal_exit = False
    try:
        directory.mkdir(exist_ok=False)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the permanent summary BEFORE any model/environment work.
        with summary_path.open("x", encoding="utf-8", newline="\n") as summary:
            report = {"name": NAME, "status": "running", "started_utc": utc_now(),
                      "release_sha256": release_sha256, "release": None,
                      "attempted_episodes": 0, "completed_episodes": 0, "matched_episodes": 0,
                      "last_attempt": None, "episode_checks": [], "current_phase": "reserved",
                      "limitations": protocol_description()["interpretation"]}
            write_json_x(directory / "started.json", report)
            primary_error = None
            try:
                report["current_phase"] = "release_preflight"
                release = validate_release(root, release_sha256)
                report["release"] = release
                references = protocol.load_references(root)
                if protocol.selected_seeds(references) != SEEDS:
                    raise ValueError("Frozen case selection differs")
                ledger_path = directory / "episodes.jsonl"
                with ledger_path.open("x", encoding="utf-8", newline="\n") as log:
                    execute_cases(root, references, directory, log, report)
                expected = [(arm, seed) for arm in ARM_ORDER for seed in SEEDS]
                actual = [(item["arm"], item["seed"]) for item in report["episode_checks"]
                          if item["kind"] == "complete"]
                if (actual != expected or len(expected) != 111
                        or any(report[key] != 111 for key in
                               ("attempted_episodes", "completed_episodes", "matched_episodes"))):
                    raise ValueError("Incomplete or out-of-order diagnostic ledger")
                report["current_phase"] = "final_integrity_check"
                verify_ledger(directory, report)
                verify_final_inputs(root, release, release_sha256)
                report.update(status="complete", current_phase="finished")
            except BaseException as error:
                primary_error = error
                report.update(status="failed", error=f"{type(error).__name__}: {error}",
                              failure_phase=report["current_phase"],
                              exception_traceback=traceback.format_exc())
            finally:
                primary_error = finalize_report(directory, report, summary, primary_error)
        if primary_error is not None:
            raise primary_error
        # Summary is fully flushed AND closed before its final fingerprint/marker.
        write_json_x(directory / "complete.json", {
            "status": "complete", "matched_episodes": 111,
            "summary_sha256": sha(summary_path), "finished_utc": utc_now()})
        normal_exit = True
        print(f"Diagnostic complete: {summary_path}", flush=True)
    finally:
        if normal_exit:
            lock_path.unlink()  # Only this completed invocation's live lock.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--refuse-overwrite", action="store_true", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    # Own exclusive console streams; no background launcher may redirect to these names.
    paths = [args.root / path for path in CONSOLE_PATHS]
    if any(path.exists() for path in paths):
        raise FileExistsError("Preserve existing diagnostic console logs")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8", buffering=1) as stdout:
        with paths[1].open("x", encoding="utf-8", buffering=1) as stderr:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = stdout, stderr
                run(args.root, args.release_sha256)
            except BaseException:
                traceback.print_exc(file=stderr)
                stderr.flush()
                raise
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
