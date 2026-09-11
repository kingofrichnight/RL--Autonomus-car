import json
import sys
from pathlib import Path

import pytest

from scripts import run_synchronized_retry as retry


def snapshot_original():
    return tuple(getattr(retry.original, name) for name in (
        "STEM", "MODEL", "TRAIN_SUMMARY", "EVAL_OUTPUT", "fingerprint_inputs", "PROTOCOL",
    ))


def test_retry_changes_only_output_paths_in_frozen_arguments():
    original = retry.original
    before = snapshot_original()
    train = original.training_arguments()
    evaluate = original.evaluation_arguments("checkpoint-fingerprint")
    expected_train = train.copy()
    expected_train[train.index("--summary-output") + 1] = retry.TRAIN_SUMMARY
    expected_train[train.index("--output") + 1] = f"models/{retry.STEM}"
    expected_evaluate = evaluate.copy()
    expected_evaluate[evaluate.index("--model") + 1] = retry.MODEL
    expected_evaluate[evaluate.index("--output") + 1] = retry.EVAL_OUTPUT

    with retry.retry_paths():
        assert original.training_arguments() == expected_train
        assert original.evaluation_arguments("checkpoint-fingerprint") == expected_evaluate
        assert original.PROTOCOL == "predictive_post_spawn_sync_v1"
        assert original.fingerprint_inputs is retry.retry_inputs
    assert snapshot_original() == before
    assert retry.STEM == "ppo_v3_predictive_sync_v1_retry01_seed42"
    assert retry.MODEL == f"models/{retry.STEM}.zip"
    assert retry.TRAIN_SUMMARY == f"results/{retry.STEM}.training.json"
    assert retry.EVAL_OUTPUT == (
        "results/ppo_v3_predictive_sync_v1_retry01_development_seed40042.csv"
    )


def test_retry_paths_restore_original_contract_on_exception():
    before = snapshot_original()
    with pytest.raises(RuntimeError, match="deliberate"):
        with retry.retry_paths():
            assert retry.original.STEM == retry.STEM
            raise RuntimeError("deliberate context failure")
    assert snapshot_original() == before


@pytest.mark.parametrize("mode", ["train", "evaluate"])
@pytest.mark.parametrize("failure", [False, True])
def test_run_delegates_once_and_restores_paths(monkeypatch, mode, failure):
    calls = []
    before = snapshot_original()

    def delegate(selected):
        calls.append(selected)
        assert retry.original.STEM == retry.STEM
        assert retry.original.MODEL == retry.MODEL
        assert retry.original.TRAIN_SUMMARY == retry.TRAIN_SUMMARY
        assert retry.original.EVAL_OUTPUT == retry.EVAL_OUTPUT
        assert retry.original.fingerprint_inputs is retry.retry_inputs
        if failure:
            raise RuntimeError("deliberate delegated failure")

    def forbidden_main():
        pytest.fail("Retry must delegate to the original run, not parse its CLI")

    monkeypatch.setattr(retry.original, "run", delegate)
    monkeypatch.setattr(retry.original, "main", forbidden_main)
    if failure:
        with pytest.raises(RuntimeError, match="deliberate delegated failure"):
            retry.run(mode)
    else:
        retry.run(mode)
    assert calls == [mode]
    assert snapshot_original() == before


def test_retry_keeps_original_artifacts_and_records_failure(monkeypatch, tmp_path):
    original = retry.original
    before = snapshot_original()
    monkeypatch.chdir(tmp_path)
    old_record = Path(f"results/{original.STEM}.train.run.json")
    old_record.parent.mkdir()
    old_record.write_text('{"status": "running", "preserve": true}', encoding="utf-8")
    old_model = Path(original.MODEL)
    old_model.parent.mkdir()
    old_model.write_bytes(b"preserve interrupted checkpoint")
    originals = {path: path.read_bytes() for path in (old_record, old_model)}
    monkeypatch.setattr(retry, "retry_inputs", lambda: {})
    monkeypatch.setattr(original, "version", lambda name: original.VERSIONS[name])
    monkeypatch.setattr(original.platform, "python_version", lambda: "3.12.9")
    calls = []

    def fail_training():
        calls.append(sys.argv.copy())
        raise RuntimeError("deliberate retry execution failure")

    factory = original.train_ppo.make_intersection_env
    monkeypatch.setattr(original.train_ppo, "main", fail_training)
    with pytest.raises(RuntimeError, match="deliberate retry execution failure"):
        retry.run("train")
    report_path = Path(f"results/{retry.STEM}.train.run.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert "deliberate retry execution failure" in report["error"]
    assert report["observation_protocol"] == original.PROTOCOL
    assert report["arguments"] == calls[0][1:]
    assert len(calls) == 1
    assert original.train_ppo.make_intersection_env is factory
    assert snapshot_original() == before
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    failed_record = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        retry.run("train")
    assert len(calls) == 1
    assert report_path.read_bytes() == failed_record
    assert all(path.read_bytes() == contents for path, contents in originals.items())


@pytest.mark.parametrize("mode", ["train", "evaluate"])
def test_cli_delegates_explicit_mode_with_refuse_overwrite(monkeypatch, mode):
    calls = []
    monkeypatch.setattr(retry, "run", calls.append)
    monkeypatch.setattr(sys, "argv", ["retry", mode, "--refuse-overwrite"])
    retry.main()
    assert calls == [mode]


@pytest.mark.parametrize("arguments", [["train"], ["evaluate"], ["resume", "--refuse-overwrite"]])
def test_cli_rejects_missing_overwrite_guard_and_resume(monkeypatch, arguments):
    calls = []
    monkeypatch.setattr(retry, "run", calls.append)
    monkeypatch.setattr(sys, "argv", ["retry", *arguments])
    with pytest.raises(SystemExit) as error:
        retry.main()
    assert error.value.code == 2
    assert not calls


@pytest.fixture
def fingerprint_fixture(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    original_script = Path("scripts/run_synchronized_predictive.py")
    original_script.parent.mkdir()
    original_script.write_text("# Historical runner fixture\n", encoding="utf-8")
    sources = {
        "safeintent_rl/frozen.py": "historical-source-fingerprint",
        original_script.as_posix(): retry.sha(original_script),
    }
    prior = Path("results/original.train.run.json")
    prior.parent.mkdir()
    prior.write_text(json.dumps({
        "status": "running", "observation_protocol": retry.original.PROTOCOL,
        "source_sha256": sources.copy(),
    }), encoding="utf-8")
    interruption = Path("results/original.interruption.json")
    interruption.write_text(json.dumps({
        "status": "interrupted", "training_stem": "ppo_v3_predictive_sync_v1_seed42",
    }), encoding="utf-8")
    new_script = Path("scripts/run_synchronized_retry.py")
    new_script.write_text("# Fresh retry runner fixture\n", encoding="utf-8")
    lineage = {
        path.as_posix(): retry.sha(path) for path in (original_script, prior, interruption)
    }
    monkeypatch.setattr(retry, "LINEAGE_HASHES", lineage)
    monkeypatch.setattr(retry, "ORIGINAL_RECORD", prior.as_posix())
    monkeypatch.setattr(retry, "INTERRUPTION_RECORD", interruption.as_posix())
    monkeypatch.setattr(retry, "ORIGINAL_FINGERPRINT", lambda: sources)
    monkeypatch.setattr(retry, "__file__", str(new_script.resolve()))
    return {"sources": sources, "prior": prior, "interruption": interruption,
            "runner": new_script, "original_script": original_script}


def test_retry_inputs_add_provenance_without_mutating_original_source_map(fingerprint_fixture):
    fixture = fingerprint_fixture
    before = fixture["sources"].copy()
    expected = {**before, **retry.LINEAGE_HASHES,
                "scripts/run_synchronized_retry.py": retry.sha(fixture["runner"])}
    actual = retry.retry_inputs()
    assert actual == expected
    assert actual is not fixture["sources"]
    assert fixture["sources"] == before
    assert retry.retry_inputs() == actual


@pytest.mark.parametrize("target", ["original_script", "prior", "interruption"])
def test_retry_inputs_reject_changed_lineage(fingerprint_fixture, target):
    fixture = fingerprint_fixture
    path = fixture[target]
    path.write_bytes(path.read_bytes() + b"changed")
    before = fixture["sources"].copy()
    with pytest.raises(ValueError, match="Retry lineage changed"):
        retry.retry_inputs()
    assert fixture["sources"] == before


@pytest.mark.parametrize("change", ["modified", "added", "removed"])
def test_retry_inputs_require_exact_historical_source_snapshot(fingerprint_fixture, change):
    sources = fingerprint_fixture["sources"]
    if change == "modified":
        sources["safeintent_rl/frozen.py"] = "different-source-fingerprint"
    elif change == "added":
        sources["safeintent_rl/unregistered_sensor.py"] = "additional-source-fingerprint"
    else:
        sources.pop("safeintent_rl/frozen.py")
    before = sources.copy()
    with pytest.raises(ValueError, match="Original training input snapshot differs"):
        retry.retry_inputs()
    assert sources == before


@pytest.mark.parametrize("target,field,value", [
    ("prior", "observation_protocol", "historical-unsynchronized"),
    ("interruption", "status", "complete"),
    ("interruption", "training_stem", "ppo_v3_predictive_sync_v1_retry01_seed42"),
])
def test_retry_inputs_require_matching_protocol_and_interrupted_predecessor(
        fingerprint_fixture, target, field, value):
    path = fingerprint_fixture[target]
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    retry.LINEAGE_HASHES[path.as_posix()] = retry.sha(path)
    with pytest.raises(ValueError, match="recorded interrupted predecessor"):
        retry.retry_inputs()


def test_retry_inputs_fingerprint_new_runner(fingerprint_fixture):
    path = fingerprint_fixture["runner"]
    previous = retry.retry_inputs()
    path.write_text("# Deliberately changed retry fixture\n", encoding="utf-8")
    current = retry.retry_inputs()
    key = "scripts/run_synchronized_retry.py"
    assert current[key] == retry.sha(path)
    assert current[key] != previous[key]
    assert {name: digest for name, digest in current.items() if name != key} == {
        name: digest for name, digest in previous.items() if name != key
    }


@pytest.mark.parametrize("mode,output_kind", [
    ("train", "model"), ("train", "training_summary"), ("train", "logdir"),
    ("train", "run_record"), ("evaluate", "csv"), ("evaluate", "evaluation_summary"),
    ("evaluate", "run_record"),
])
def test_retry_refuses_existing_outputs_before_any_input_or_training_access(
        monkeypatch, tmp_path, mode, output_kind):
    monkeypatch.chdir(tmp_path)
    before = snapshot_original()
    paths = {
        "model": Path(retry.MODEL), "training_summary": Path(retry.TRAIN_SUMMARY),
        "logdir": Path("logs") / retry.STEM, "csv": Path(retry.EVAL_OUTPUT),
        "evaluation_summary": Path(retry.EVAL_OUTPUT).with_suffix(".summary.json"),
        "run_record": Path(f"results/{retry.STEM}.{mode}.run.json"),
    }
    target = paths[output_kind]
    target.parent.mkdir(parents=True, exist_ok=True)
    if output_kind == "logdir":
        target.mkdir()
        preserved = target / "sentinel.txt"
    else:
        preserved = target
    preserved.write_bytes(b"preserve existing research output")

    def forbidden_access(*args, **kwargs):
        pytest.fail("Existing artifacts must be rejected before fingerprinting or execution")

    monkeypatch.setattr(retry, "retry_inputs", forbidden_access)
    monkeypatch.setattr(retry.original.train_ppo, "main", forbidden_access)
    monkeypatch.setattr(retry.original.evaluate_policy, "main", forbidden_access)
    with pytest.raises(FileExistsError, match="Preserve existing artifact"):
        retry.run(mode)
    assert preserved.read_bytes() == b"preserve existing research output"
    assert snapshot_original() == before
    assert not Path(retry.original.MODEL).exists()
    assert not Path(retry.original.TRAIN_SUMMARY).exists()
    assert not Path(retry.original.EVAL_OUTPUT).exists()
