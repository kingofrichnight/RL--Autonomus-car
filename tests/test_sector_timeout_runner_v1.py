import gzip
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_sector_timeout_diagnostic_v1 as runner


def metric_row(**changes):
    result = {"reward": 3.0, "length": 2, "success": True, "collision": False,
              "travel_time": .4, "min_ttc": 1.5, "unsafe_ttc_events": 2,
              "safety_interventions": 0}
    result.update(changes)
    return result


class EpisodeEnv:
    def __init__(self, events, transitions=None):
        self.events = events
        self.transitions = transitions or [
            (1.0, False, False, {"min_ttc": 2.0}),
            (2.0, True, False, {"arrived": True}),
        ]
        self.unwrapped = SimpleNamespace(
            vehicle=SimpleNamespace(crashed=False), road=SimpleNamespace(vehicles=[object()]),
            config={"policy_frequency": 5}, time=0.0,
        )
        self.steps = 0
        self.closed = False

    def reset(self, *, seed):
        self.events.append(("reset", seed))
        self.steps = 0
        self.unwrapped.time = 0.0
        return np.zeros(115, dtype=np.float32), {"route_progress": 0.0}

    def step(self, action):
        self.events.append(("step", self.steps, int(action)))
        reward, terminated, truncated, info = self.transitions[self.steps]
        self.steps += 1
        self.unwrapped.time = self.steps / 5
        return (np.full(115, self.steps / 10, dtype=np.float32), reward,
                terminated, truncated, info.copy())

    def close(self):
        self.events.append(("close",))
        self.closed = True


@pytest.fixture
def fake_episode(monkeypatch):
    events = []
    env = EpisodeEnv(events)
    model = object()

    class Capture:
        def __init__(self, selected_env):
            assert selected_env is env

        def __enter__(self):
            events.append(("capture_enter",))
            return self

        def __exit__(self, *args):
            events.append(("capture_exit",))

        def reset_frame(self, *, new_episode=False):
            events.append(("reset_frame", new_episode))

        def frame(self, observation, info=None):
            events.append(("frame", env.steps))
            return {"policy_observation": observation.tolist(),
                    "state": {"time": env.unwrapped.time,
                              "route_progress": (info or {}).get("route_progress")}}

    def predict(selected_model, observation):
        assert selected_model is model
        events.append(("predict", env.steps))
        return (np.array(1), None), [.2, .5, .3]

    def ttc(vehicle, vehicles):
        assert vehicle is env.unwrapped.vehicle
        assert vehicles == env.unwrapped.road.vehicles and isinstance(vehicles, list)
        events.append(("ttc", env.steps))
        return 5.0 if env.steps == 0 else 1.5

    monkeypatch.setattr(runner, "ObservationCapture", Capture)
    monkeypatch.setattr(runner, "captured_predict", predict)
    monkeypatch.setattr(runner, "minimum_ttc", ttc)
    return {"env": env, "model": model, "events": events}


def test_replay_reproduces_eight_metrics_and_original_ttc_prediction_step_order(
        fake_episode, tmp_path):
    fixture = fake_episode
    output = tmp_path / "trace.jsonl"
    with output.open("x", encoding="utf-8") as stream:
        actual = runner.replay_episode(fixture["env"], fixture["model"], "geometry", 7, stream)
    assert actual == metric_row()
    sequence = [event for event in fixture["events"] if event[0] in ("ttc", "predict", "step")]
    assert sequence == [
        ("ttc", 0), ("predict", 0), ("step", 0, 1),
        ("ttc", 1), ("predict", 1), ("step", 1, 1),
    ]
    assert ("frame", 2) in fixture["events"]
    assert ("predict", 2) not in fixture["events"]
    assert fixture["events"][-1] == ("capture_exit",)
    assert not fixture["env"].closed
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [item["kind"] for item in records] == [
        "decision", "transition", "decision", "transition", "terminal",
    ]
    assert records[0]["pre_step_ttc"] == 5.0
    assert records[1]["metric_ttc"] == 2.0
    assert records[2]["pre_step_ttc"] == records[3]["metric_ttc"] == 1.5
    assert records[-1]["observation_frame_index"] == 2
    assert "proposed_action" not in records[-1] and "executed_action" not in records[-1]


@pytest.mark.parametrize("info,terminated,truncated,expected", [
    ({"crashed": True, "arrived": True}, True, False,
     {"collision": True, "success": False}),
    ({"arrived": False}, False, True, {"collision": False, "success": False}),
    ({"arrived": True}, True, False, {"collision": False, "success": True}),
])
def test_replay_preserves_terminal_outcome_semantics_without_another_prediction(
        fake_episode, tmp_path, info, terminated, truncated, expected):
    fixture = fake_episode
    fixture["env"].transitions = [(1.0, terminated, truncated, info)]
    with (tmp_path / "terminal.jsonl").open("x", encoding="utf-8") as stream:
        actual = runner.replay_episode(fixture["env"], fixture["model"], "geometry", 7, stream)
    assert actual == metric_row(reward=1.0, length=1, travel_time=.2, min_ttc=5.0,
                                unsafe_ttc_events=0, **expected)
    assert len([item for item in fixture["events"] if item[0] == "predict"]) == 1
    assert ("frame", 1) in fixture["events"]


def test_replay_counts_explicit_intervention_and_inclusive_ttc_threshold(fake_episode, tmp_path):
    fixture = fake_episode
    fixture["env"].transitions = [
        (0.0, True, False, {"min_ttc": 2.0, "safety_intervened": True}),
    ]
    with (tmp_path / "intervention.jsonl").open("x", encoding="utf-8") as stream:
        actual = runner.replay_episode(fixture["env"], fixture["model"], "geometry", 7, stream)
    assert actual == metric_row(reward=0.0, length=1, success=False, travel_time=.2,
                                min_ttc=2.0, unsafe_ttc_events=1, safety_interventions=1)


def test_replay_closes_capture_on_predict_failure_without_a_step(
        monkeypatch, fake_episode, tmp_path):
    fixture = fake_episode

    def fail(*args):
        raise RuntimeError("deliberate predict failure")

    monkeypatch.setattr(runner, "captured_predict", fail)
    with (tmp_path / "failed.jsonl").open("x", encoding="utf-8") as stream:
        with pytest.raises(RuntimeError, match="deliberate predict failure"):
            runner.replay_episode(fixture["env"], fixture["model"], "geometry", 7, stream)
    assert fixture["events"][-1] == ("capture_exit",)
    assert not any(item[0] == "step" for item in fixture["events"])


def empty_report():
    return {"attempted_episodes": 0, "completed_episodes": 0, "matched_episodes": 0,
            "episode_checks": [], "last_attempt": None, "current_phase": "before_cases"}


@pytest.fixture
def fake_cases(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ARM_ORDER", ("geometry", "padding"))
    monkeypatch.setattr(runner, "SEEDS", (7, 8))
    monkeypatch.setattr(runner, "FIRST_SEED", 7)
    events, environments = [], []
    references = {arm: [metric_row(), metric_row()] for arm in runner.ARM_ORDER}
    directory = tmp_path / "traces"
    directory.mkdir()
    ledger_path = tmp_path / "episodes.jsonl"
    report = empty_report()

    def model(root, arm):
        events.append(("model", arm))
        return arm

    def factory(arm, config, *, bootstrap_seed):
        assert bootstrap_seed == 40042
        events.append(("factory", arm))
        env = EpisodeEnv(events)
        env.arm = arm
        environments.append(env)
        return env

    def replay(env, model, arm, seed, stream):
        assert env.arm == model == arm
        ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        assert ledger[-1]["kind"] == "start"
        assert ledger[-1]["arm"] == arm and ledger[-1]["seed"] == seed
        assert report["attempted_episodes"] == sum(item["kind"] == "start" for item in ledger)
        events.append(("replay", arm, seed))
        env.reset(seed=seed)
        runner.write_line(stream, {"kind": "engineering_fixture", "arm": arm, "seed": seed})
        return metric_row()

    def compare(actual, reference):
        assert actual == reference
        return {key: 0 for key in ("reward", "min_ttc", "travel_time")}

    monkeypatch.setattr(runner, "load_model", model)
    monkeypatch.setattr(runner, "make_diagnostic_env", factory)
    monkeypatch.setattr(runner, "replay_episode", replay)
    monkeypatch.setattr(runner.protocol, "compare_episode", compare)
    return {"root": tmp_path, "references": references, "directory": directory,
            "ledger_path": ledger_path, "report": report, "events": events,
            "environments": environments, "replay": replay}


def execute_fixture(fixture):
    with fixture["ledger_path"].open("x", encoding="utf-8") as ledger:
        runner.execute_cases(fixture["root"], fixture["references"], fixture["directory"],
                             ledger, fixture["report"])


def test_execute_writes_durable_start_before_reset_and_completes_each_case_once(fake_cases):
    fixture = fake_cases
    execute_fixture(fixture)
    report = fixture["report"]
    assert all(report[key] == 4 for key in (
        "attempted_episodes", "completed_episodes", "matched_episodes",
    ))
    expected = [(arm, seed) for arm in runner.ARM_ORDER for seed in runner.SEEDS]
    assert [(item["arm"], item["seed"]) for item in report["episode_checks"]] == expected
    assert [(item[1], item[2]) for item in fixture["events"] if item[0] == "replay"] == expected
    ledger = [json.loads(line) for line in fixture["ledger_path"].read_text().splitlines()]
    assert [item["kind"] for item in ledger] == ["start", "complete"] * 4
    assert all(env.closed for env in fixture["environments"])
    for arm in runner.ARM_ORDER:
        with gzip.open(fixture["directory"] / f"{arm}.trace.jsonl.gz", "rt") as stream:
            records = [json.loads(line) for line in stream]
        assert [item["seed"] for item in records] == [7, 8]


def test_execute_stops_at_first_mismatch_and_preserves_actual_reference_and_partial_trace(
        monkeypatch, fake_cases):
    fixture = fake_cases

    def mismatch(actual, reference):
        raise ValueError("deliberate metric mismatch")

    monkeypatch.setattr(runner.protocol, "compare_episode", mismatch)
    with pytest.raises(ValueError, match="deliberate metric mismatch"):
        execute_fixture(fixture)
    report = fixture["report"]
    assert report["attempted_episodes"] == report["completed_episodes"] == 1
    assert report["matched_episodes"] == 0
    assert len(report["episode_checks"]) == 1
    assert report["episode_checks"][0]["kind"] == "mismatch"
    assert report["episode_checks"][0]["actual"] == metric_row()
    assert report["episode_checks"][0]["reference"] == metric_row()
    ledger = [json.loads(line) for line in fixture["ledger_path"].read_text().splitlines()]
    assert [item["kind"] for item in ledger] == ["start", "mismatch", "failed"]
    assert fixture["environments"][0].closed
    assert not (fixture["directory"] / "padding.trace.jsonl.gz").exists()
    with gzip.open(fixture["directory"] / "geometry.trace.jsonl.gz", "rt") as stream:
        assert len(list(stream)) == 1


def test_execute_records_consumed_attempt_before_reset_failure_and_stops(monkeypatch, fake_cases):
    fixture = fake_cases

    def reset_failure(env, model, arm, seed, stream):
        ledger = [json.loads(line) for line in fixture["ledger_path"].read_text().splitlines()]
        assert ledger[-1]["kind"] == "start"
        assert fixture["report"]["attempted_episodes"] == 1
        raise RuntimeError("deliberate reset failure")

    monkeypatch.setattr(runner, "replay_episode", reset_failure)
    with pytest.raises(RuntimeError, match="deliberate reset failure"):
        execute_fixture(fixture)
    assert fixture["report"]["attempted_episodes"] == 1
    assert fixture["report"]["completed_episodes"] == fixture["report"]["matched_episodes"] == 0
    ledger = [json.loads(line) for line in fixture["ledger_path"].read_text().splitlines()]
    assert [item["kind"] for item in ledger] == ["start", "failed"]
    assert fixture["environments"][0].closed
    assert not any(item == ("model", "padding") for item in fixture["events"])


def test_exclusive_trace_refuses_existing_file_and_preserves_bytes(tmp_path):
    path = tmp_path / "existing.gz"
    path.write_bytes(b"preserved partial trace")
    with pytest.raises(FileExistsError):
        with runner.exclusive_trace(path):
            pytest.fail("Existing trace was opened")
    assert path.read_bytes() == b"preserved partial trace"


def test_json_trace_nonfinite_encoding_and_durable_journal_flush(monkeypatch, tmp_path):
    output = tmp_path / "safe.jsonl"
    synced = []
    monkeypatch.setattr(runner.os, "fsync", synced.append)
    with output.open("x", encoding="utf-8") as stream:
        descriptor = stream.fileno()
        runner.write_line(stream, {"ttc": math.inf, "negative": -math.inf,
                                   "nested": [np.float32(1), np.nan]}, durable=True)
        assert json.loads(output.read_text()) == {
            "ttc": "Infinity", "negative": "-Infinity", "nested": [1.0, "NaN"],
        }
    assert synced == [descriptor]


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    engineering_seeds = tuple(range(7, 44))
    monkeypatch.setattr(runner, "SEEDS", engineering_seeds)
    monkeypatch.setattr(runner, "FIRST_SEED", 7)
    release = {"protocol": {"interpretation": "engineering fixture only"}}
    calls = []

    def validate(root, digest):
        assert root == tmp_path
        assert (root / runner.SUMMARY_PATH).exists()
        assert (root / runner.OUTPUT_DIR / "started.json").exists()
        calls.append("preflight")
        return release

    def execute(root, references, directory, ledger, report):
        assert (root / runner.SUMMARY_PATH).exists()
        assert (directory / "started.json").exists()
        assert (root / runner.LOCK_PATH).exists()
        calls.append("execute")
        checks = []
        for arm in runner.ARM_ORDER:
            with runner.exclusive_trace(directory / f"{arm}.trace.jsonl.gz") as traces:
                runner.write_line(traces, {"kind": "engineering_fixture"})
            for seed in engineering_seeds:
                attempt = {"arm": arm, "seed": seed, "attempt_index": len(checks)}
                check = {"kind": "complete", **attempt, "actual": metric_row(),
                         "reference": metric_row(),
                         "deltas": runner.protocol.compare_episode(metric_row(), metric_row())}
                runner.write_line(ledger, {"kind": "start", **attempt, "utc": runner.utc_now()})
                runner.write_line(ledger, check)
                checks.append(check)
        report.update(attempted_episodes=111, completed_episodes=111, matched_episodes=111,
                      episode_checks=checks)

    monkeypatch.setattr(runner, "validate_release", validate)
    monkeypatch.setattr(runner.protocol, "load_references", lambda root: {})
    monkeypatch.setattr(runner.protocol, "selected_seeds", lambda refs: engineering_seeds)
    monkeypatch.setattr(runner, "execute_cases", execute)
    monkeypatch.setattr(runner, "verify_final_inputs", lambda *args: calls.append("postflight"))
    return {"root": tmp_path, "calls": calls, "execute": execute, "release": release}


def test_run_reserves_artifacts_then_verifies_final_inputs_and_completes_once(fake_run):
    fixture = fake_run
    root = fixture["root"]
    runner.run(root, "a" * 64)
    assert fixture["calls"] == ["preflight", "execute", "postflight"]
    summary_path = root / runner.SUMMARY_PATH
    report = json.loads(summary_path.read_text())
    assert report["status"] == "complete" and report["matched_episodes"] == 111
    complete = json.loads((root / runner.OUTPUT_DIR / "complete.json").read_text())
    assert complete["summary_sha256"] == runner.sha(summary_path)
    assert not (root / runner.LOCK_PATH).exists()
    before = summary_path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(root, "a" * 64)
    assert fixture["calls"] == ["preflight", "execute", "postflight"]
    assert summary_path.read_bytes() == before


@pytest.mark.parametrize("failure", ["episode", "postflight", "incomplete_ledger"])
def test_run_preserves_failure_summary_lock_and_ledger_without_completion_marker(
        monkeypatch, fake_run, failure):
    fixture = fake_run

    def fail_execute(root, references, directory, ledger, report):
        fixture["execute"](root, references, directory, ledger, report)
        if failure == "episode":
            raise ValueError("deliberate episode mismatch")
        report["episode_checks"].pop()

    def fail_postflight(*args):
        raise ValueError("deliberate source change")

    if failure == "postflight":
        monkeypatch.setattr(runner, "verify_final_inputs", fail_postflight)
    else:
        monkeypatch.setattr(runner, "execute_cases", fail_execute)
    with pytest.raises(ValueError):
        runner.run(fixture["root"], "a" * 64)
    root = fixture["root"]
    report = json.loads((root / runner.SUMMARY_PATH).read_text())
    assert report["status"] == "failed" and report["error"] and report["finished_utc"]
    assert (root / runner.LOCK_PATH).exists()
    assert (root / runner.OUTPUT_DIR / "episodes.jsonl").exists()
    assert not (root / runner.OUTPUT_DIR / "complete.json").exists()
    before = (root / runner.SUMMARY_PATH).read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(root, "a" * 64)
    assert (root / runner.SUMMARY_PATH).read_bytes() == before


@pytest.mark.parametrize("target", ["directory", "summary", "lock", "training_lock"])
def test_run_refuses_each_existing_artifact_before_release_or_model_access(fake_run, target):
    root = fake_run["root"]
    paths = {"directory": runner.OUTPUT_DIR, "summary": runner.SUMMARY_PATH,
             "lock": runner.LOCK_PATH,
             "training_lock": Path("logs/v3_sector_comparison_v1.active.lock")}
    path = root / paths[target]
    path.parent.mkdir(parents=True, exist_ok=True)
    if target == "directory":
        path.mkdir()
        preserved = path / "sentinel"
    else:
        preserved = path
    preserved.write_bytes(b"preserved existing evidence")
    with pytest.raises(FileExistsError):
        runner.run(root, "a" * 64)
    assert not fake_run["calls"]
    assert preserved.read_bytes() == b"preserved existing evidence"


def test_failed_release_preflight_is_recorded_permanently_without_starting_an_attempt(
        monkeypatch, fake_run):
    def fail(*args):
        raise ValueError("changed frozen release")

    monkeypatch.setattr(runner, "validate_release", fail)
    with pytest.raises(ValueError, match="changed frozen release"):
        runner.run(fake_run["root"], "a" * 64)
    root = fake_run["root"]
    for path in (runner.OUTPUT_DIR, runner.SUMMARY_PATH, runner.LOCK_PATH):
        assert (root / path).exists()
    report = json.loads((root / runner.SUMMARY_PATH).read_text())
    assert report["status"] == "failed" and report["failure_phase"] == "release_preflight"
    assert report["attempted_episodes"] == report["completed_episodes"] == 0
    assert not (root / runner.OUTPUT_DIR / "complete.json").exists()
    before = (root / runner.SUMMARY_PATH).read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(root, "a" * 64)
    assert (root / runner.SUMMARY_PATH).read_bytes() == before
    assert not fake_run["calls"]


@pytest.fixture
def released_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "SEEDS", (7,))
    monkeypatch.setattr(runner, "__file__", str(
        tmp_path / "scripts/run_sector_timeout_diagnostic_v1.py",
    ))
    monkeypatch.setattr(runner.protocol, "__file__", str(
        tmp_path / "scripts/sector_timeout_protocol_v1.py",
    ))
    current = {"source_sha256": {"source.py": "source-sha"},
               "dependency_sha256": {"runtime.bin": "dependency-sha"},
               "runtime": {"python": "3.12.9"}, "snapshot_archive_sha256": "snapshot-sha"}
    release = {
        "schema_version": 1, "status": "released", "created_utc": "engineering-created",
        "protocol": runner.protocol_description(),
        "validation": {"ruff": "passed", "pytest": "passed", "independent_review": "passed",
                       "tests_passed": 600, "completed_utc": "engineering-tested",
                       "source_head": "source-commit"},
        **current,
    }
    path = tmp_path / runner.RELEASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    proof_path = tmp_path / runner.VERIFICATION_PATH
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps({"validation": dict(release["validation"]), **current}),
                          encoding="utf-8")
    release["validation"]["verification_sha256"] = runner.sha(proof_path)
    path.write_text(json.dumps(release), encoding="utf-8")
    monkeypatch.setattr(runner, "current_fingerprints", lambda root: current)
    monkeypatch.setattr(runner.torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(runner.torch, "get_num_interop_threads", lambda: 8)
    return {"root": tmp_path, "release": release, "current": current, "path": path,
            "digest": runner.sha(path), "proof_path": proof_path}


def test_release_validation_and_final_integrity_accept_exact_same_evidence(released_fixture):
    fixture = released_fixture
    assert runner.validate_release(fixture["root"], fixture["digest"]) == fixture["release"]
    runner.verify_final_inputs(fixture["root"], fixture["release"], fixture["digest"])


@pytest.mark.parametrize("field", [
    "source_sha256", "dependency_sha256", "runtime", "snapshot_archive_sha256",
])
def test_release_and_postflight_reject_changed_inputs_or_runtime(released_fixture, field):
    fixture = released_fixture
    fixture["current"][field] = "changed frozen evidence"
    with pytest.raises(ValueError, match="changed"):
        runner.validate_release(fixture["root"], fixture["digest"])
    with pytest.raises(ValueError, match="changed"):
        runner.verify_final_inputs(fixture["root"], fixture["release"], fixture["digest"])


@pytest.mark.parametrize("field,value", [
    ("ruff", "failed"), ("pytest", "failed"), ("independent_review", "pending"),
    ("tests_passed", 569), ("tests_passed", "600"), ("completed_utc", ""), ("source_head", ""),
])
def test_release_rejects_missing_or_failed_fresh_verification(released_fixture, field, value):
    fixture = released_fixture
    fixture["release"]["validation"][field] = value
    fixture["path"].write_text(json.dumps(fixture["release"]), encoding="utf-8")
    with pytest.raises(ValueError, match="fresh full tests"):
        runner.validate_release(fixture["root"], runner.sha(fixture["path"]))


def test_final_integrity_rejects_changed_release_bytes(released_fixture):
    fixture = released_fixture
    fixture["path"].write_bytes(fixture["path"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="Release changed"):
        runner.verify_final_inputs(fixture["root"], fixture["release"], fixture["digest"])


@pytest.mark.parametrize("arguments", [
    [], ["--release-sha256", "a" * 64], ["--refuse-overwrite"],
    ["--release-sha256", "a" * 64, "--refuse-overwrite", "--resume"],
])
def test_cli_rejects_missing_release_guard_and_resume(monkeypatch, arguments):
    calls = []
    monkeypatch.setattr(runner, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["diagnostic", *arguments])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not calls


def test_cli_preserves_exclusive_console_logs_and_restores_standard_streams(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", lambda value: None)
    before = sys.stdout, sys.stderr

    def fail(root, digest):
        calls.append((root, digest))
        print("engineering run begins")
        raise RuntimeError("deliberate CLI failure")

    monkeypatch.setattr(runner, "run", fail)
    monkeypatch.setattr(sys, "argv", [
        "diagnostic", "--root", str(tmp_path), "--release-sha256", "a" * 64,
        "--refuse-overwrite",
    ])
    with pytest.raises(RuntimeError, match="deliberate CLI failure"):
        runner.main()
    assert (sys.stdout, sys.stderr) == before
    stdout, stderr = [tmp_path / path for path in runner.CONSOLE_PATHS]
    assert "engineering run begins" in stdout.read_text()
    assert "deliberate CLI failure" in stderr.read_text()
    saved = stdout.read_bytes(), stderr.read_bytes()
    with pytest.raises(FileExistsError):
        runner.main()
    assert len(calls) == 1
    assert (stdout.read_bytes(), stderr.read_bytes()) == saved
    assert (sys.stdout, sys.stderr) == before


@pytest.fixture
def durable_ledger_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ARM_ORDER", ("geometry",))
    monkeypatch.setattr(runner, "SEEDS", (7, 8))
    (tmp_path / "started.json").write_text("{}", encoding="utf-8")
    with runner.exclusive_trace(tmp_path / "geometry.trace.jsonl.gz"):
        pass
    records, checks = [], []
    for index, seed in enumerate((7, 8)):
        attempt = {"arm": "geometry", "seed": seed, "attempt_index": index}
        complete = {"kind": "complete", **attempt, "actual": metric_row(),
                    "reference": metric_row(),
                    "deltas": runner.protocol.compare_episode(metric_row(), metric_row())}
        records.extend([{"kind": "start", **attempt, "utc": runner.utc_now()}, complete])
        checks.append(complete)
    with (tmp_path / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            runner.write_line(stream, record, durable=True)
    return {"directory": tmp_path, "records": records, "report": {"episode_checks": checks}}


def test_verify_ledger_reads_exact_durable_pairs_and_metrics(durable_ledger_fixture):
    fixture = durable_ledger_fixture
    runner.verify_ledger(fixture["directory"], fixture["report"])


@pytest.mark.parametrize("mutation", [
    "seed_type", "index_type", "naive_utc", "extra_start_field", "delta", "actual_metric",
    "truncated", "reordered", "unknown_artifact",
])
def test_verify_ledger_rejects_tampered_disk_evidence_even_when_memory_agrees(
        durable_ledger_fixture, mutation):
    fixture = durable_ledger_fixture
    records = fixture["records"]
    if mutation == "seed_type":
        records[0]["seed"] = records[1]["seed"] = 7.0
    elif mutation == "index_type":
        records[0]["attempt_index"] = records[1]["attempt_index"] = False
    elif mutation == "naive_utc":
        records[0]["utc"] = "2026-09-12T00:00:00"
    elif mutation == "extra_start_field":
        records[0]["unregistered"] = True
    elif mutation == "delta":
        records[1]["deltas"]["reward"] = 1
    elif mutation == "actual_metric":
        records[1]["actual"]["reward"] += 1
    elif mutation == "truncated":
        records.pop()
    elif mutation == "reordered":
        records.reverse()
    else:
        (fixture["directory"] / "unregistered.txt").write_bytes(b"extra file")
    with (fixture["directory"] / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            runner.write_line(stream, record)
    with pytest.raises(ValueError):
        runner.verify_ledger(fixture["directory"], fixture["report"])


@pytest.mark.parametrize("has_primary", [True, False])
def test_finalization_hash_errors_preserve_primary_failure_and_writable_summary(
        monkeypatch, tmp_path, has_primary):
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "partial.trace").write_bytes(b"preserved partial bytes")
    primary = ValueError("original replay mismatch") if has_primary else None
    report = {"status": "failed" if has_primary else "complete"}
    if has_primary:
        report.update(error="ValueError: original replay mismatch", failure_phase="episode_replay")

    def fail_hash(path):
        raise OSError("deliberate secondary hash failure")

    monkeypatch.setattr(runner, "sha", fail_hash)
    summary = tmp_path / "summary.json"
    with summary.open("x", encoding="utf-8") as stream:
        returned = runner.finalize_report(directory, report, stream, primary)
    saved = json.loads(summary.read_text())
    assert saved["status"] == "failed" and saved["finished_utc"]
    assert any("deliberate secondary hash failure" in item for item in saved["finalization_errors"])
    if has_primary:
        assert returned is primary
        assert saved["error"] == "ValueError: original replay mismatch"
        assert saved["failure_phase"] == "episode_replay"
    else:
        assert isinstance(returned, ValueError)
        assert saved["failure_phase"] == "artifact_hashing"
    assert (directory / "partial.trace").read_bytes() == b"preserved partial bytes"


def test_release_and_postflight_reject_changed_test_review_proof(released_fixture):
    fixture = released_fixture
    fixture["proof_path"].write_bytes(fixture["proof_path"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="verification evidence fingerprint"):
        runner.validate_release(fixture["root"], fixture["digest"])
    with pytest.raises(ValueError, match="evidence changed"):
        runner.verify_final_inputs(fixture["root"], fixture["release"], fixture["digest"])


def test_release_rejects_proof_for_different_sources_even_with_updated_proof_hash(released_fixture):
    fixture = released_fixture
    proof = json.loads(fixture["proof_path"].read_text())
    proof["source_sha256"] = {"different.py": "different-input"}
    fixture["proof_path"].write_text(json.dumps(proof), encoding="utf-8")
    fixture["release"]["validation"]["verification_sha256"] = runner.sha(fixture["proof_path"])
    fixture["path"].write_text(json.dumps(fixture["release"]), encoding="utf-8")
    with pytest.raises(ValueError, match="these exact inputs"):
        runner.validate_release(fixture["root"], runner.sha(fixture["path"]))


@pytest.mark.parametrize("failure", ["replay", "mismatch"])
def test_replay_or_mismatch_primary_and_phase_survive_additional_environment_close_failure(
        monkeypatch, fake_cases, failure):
    fixture = fake_cases
    primary = ValueError("original episode failure")

    def fail_primary(*args):
        raise primary

    def fail_close(env):
        env.closed = True
        raise RuntimeError("secondary close failure")

    monkeypatch.setattr(EpisodeEnv, "close", fail_close)
    if failure == "replay":
        monkeypatch.setattr(runner, "replay_episode", fail_primary)
    else:
        monkeypatch.setattr(runner.protocol, "compare_episode", fail_primary)
    with pytest.raises(ValueError) as caught:
        execute_fixture(fixture)
    assert caught.value is primary
    assert fixture["report"]["current_phase"] == "episode_replay"
    assert "secondary close failure" in fixture["report"]["close_error"]
    assert fixture["report"]["attempted_episodes"] == 1
    assert fixture["report"]["completed_episodes"] == (1 if failure == "mismatch" else 0)
    assert fixture["report"]["matched_episodes"] == 0
    assert not any(item == ("model", "padding") for item in fixture["events"])
    ledger = [json.loads(line) for line in fixture["ledger_path"].read_text().splitlines()]
    assert ledger[-1]["kind"] == "failed"
    assert "original episode failure" in ledger[-1]["error"]


def test_close_only_failure_remains_primary_and_stops_before_next_arm(monkeypatch, fake_cases):
    fixture = fake_cases
    primary = RuntimeError("close alone failed")

    def fail_close(env):
        raise primary

    monkeypatch.setattr(EpisodeEnv, "close", fail_close)
    with pytest.raises(RuntimeError) as caught:
        execute_fixture(fixture)
    assert caught.value is primary
    assert fixture["report"]["current_phase"] == "environment_close"
    assert fixture["report"]["close_error"] == "RuntimeError: close alone failed"
    assert fixture["report"]["matched_episodes"] == 2
    assert not any(item == ("model", "padding") for item in fixture["events"])


def test_close_only_failure_is_not_confused_with_a_callers_handled_exception(
        monkeypatch, fake_cases):
    primary = RuntimeError("close alone failed inside caller exception handler")

    def fail_close(env):
        raise primary

    monkeypatch.setattr(EpisodeEnv, "close", fail_close)
    try:
        raise LookupError("unrelated exception already handled by caller")
    except LookupError:
        with pytest.raises(RuntimeError) as caught:
            execute_fixture(fake_cases)
    assert caught.value is primary
    assert fake_cases["report"]["current_phase"] == "environment_close"


@pytest.mark.parametrize("filename", ["started.json", "episodes.jsonl", "geometry.trace.jsonl.gz"])
@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_verify_ledger_rejects_each_nonregular_or_symlink_artifact(
        monkeypatch, durable_ledger_fixture, filename, kind):
    fixture = durable_ledger_fixture
    target = fixture["directory"] / filename
    if kind == "directory":
        target.unlink()
        target.mkdir()
    else:
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    with pytest.raises(ValueError):
        runner.verify_ledger(fixture["directory"], fixture["report"])


@pytest.fixture
def final_artifacts(tmp_path):
    directory = tmp_path / "artifacts"
    directory.mkdir()
    names = {"started.json", "episodes.jsonl",
             *[f"{arm}.trace.jsonl.gz" for arm in runner.ARM_ORDER]}
    for name in names:
        (directory / name).write_bytes(f"engineering artifact {name}".encode())
    return {"directory": directory, "names": names, "summary": tmp_path / "summary.json"}


def test_successful_finalization_hashes_exactly_all_five_regular_artifacts(final_artifacts):
    fixture = final_artifacts
    report = {"status": "complete"}
    with fixture["summary"].open("x", encoding="utf-8") as stream:
        error = runner.finalize_report(fixture["directory"], report, stream, None)
    assert error is None
    saved = json.loads(fixture["summary"].read_text())
    assert saved["status"] == "complete"
    assert set(saved["artifact_sha256"]) == fixture["names"]
    assert saved["artifact_sha256"] == {
        name: runner.sha(fixture["directory"] / name) for name in fixture["names"]
    }


@pytest.mark.parametrize("mutation", ["missing", "extra", "directory", "symlink"])
def test_completed_finalization_rejects_missing_extra_or_nonregular_artifacts(
        monkeypatch, final_artifacts, mutation):
    fixture = final_artifacts
    target = fixture["directory"] / "geometry.trace.jsonl.gz"
    if mutation in ("missing", "directory"):
        target.unlink()
        if mutation == "directory":
            target.mkdir()
    elif mutation == "extra":
        (fixture["directory"] / "unregistered.json").write_bytes(b"unexpected artifact")
    else:
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    report = {"status": "complete"}
    with fixture["summary"].open("x", encoding="utf-8") as stream:
        error = runner.finalize_report(fixture["directory"], report, stream, None)
    assert isinstance(error, ValueError)
    saved = json.loads(fixture["summary"].read_text())
    assert saved["status"] == "failed" and saved["finalization_errors"]
    assert saved["failure_phase"] == "artifact_hashing"
    if mutation in ("missing", "directory", "symlink"):
        assert target.name not in saved["artifact_sha256"]


def test_failed_finalization_records_nonregular_artifact_without_replacing_primary(
        final_artifacts):
    fixture = final_artifacts
    target = fixture["directory"] / "geometry.trace.jsonl.gz"
    target.unlink()
    target.mkdir()
    primary = RuntimeError("original replay error")
    report = {"status": "failed", "error": "RuntimeError: original replay error",
              "failure_phase": "episode_replay"}
    with fixture["summary"].open("x", encoding="utf-8") as stream:
        error = runner.finalize_report(fixture["directory"], report, stream, primary)
    assert error is primary
    saved = json.loads(fixture["summary"].read_text())
    assert saved["status"] == "failed" and saved["finalization_errors"]
    assert saved["error"] == "RuntimeError: original replay error"
    assert saved["failure_phase"] == "episode_replay"
    assert target.name not in saved["artifact_sha256"]
