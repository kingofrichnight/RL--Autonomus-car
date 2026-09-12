import builtins
import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from scripts import run_clock_comparison_v1 as runner
from scripts import run_sector_comparison_v1 as historical

RELEASE_HASH = "a" * 64


def forbidden(*args, **kwargs):
    pytest.fail("Runner unit tests must not execute real policies, environments, or research mains")


@pytest.fixture(autouse=True)
def no_research(monkeypatch):
    monkeypatch.setattr(runner.train_ppo, "main", forbidden)
    monkeypatch.setattr(runner.evaluate_policy, "main", forbidden)
    monkeypatch.setattr(runner.train_ppo, "make_intersection_env", forbidden)
    monkeypatch.setattr(runner.evaluate_policy, "make_intersection_env", forbidden)
    for method in ("__init__", "predict", "load", "save", "learn", "train"):
        monkeypatch.setattr(runner.PPO, method, forbidden)


def argument_values(arguments):
    assert arguments[-1] == "--refuse-overwrite"
    assert len(arguments[:-1]) % 2 == 0
    return dict(zip(arguments[:-1:2], arguments[1:-1:2], strict=True))


def test_exact_four_stage_order_and_measured_initial_tensor_contract():
    assert runner.ORDER == (("zero", "train"), ("clock", "train"),
                            ("zero", "evaluate"), ("clock", "evaluate"))
    assert tuple(runner.ARMS) == ("zero", "clock")
    assert runner.INITIAL_POLICY["parameter_count"] == 192516
    assert runner.INITIAL_POLICY["sha256"] == (
        "d1c314ef787fa038568eeaff5dfc6c1a3851aba5f39a4dd8ba767634943a57ec"
    )
    assert runner.ARMS["zero"]["protocol"] == "predictive_post_spawn_clock_zero_v1"
    assert runner.ARMS["clock"]["protocol"] == "predictive_post_spawn_clock_remaining_v1"


@pytest.mark.parametrize("arm,suffix", [("zero", "zero"), ("clock", "remaining")])
def test_training_arguments_preserve_every_old_setting_except_new_artifact_names(arm, suffix):
    expected = argument_values(historical.training_arguments("padding"))
    actual = argument_values(runner.training_arguments(arm))
    assert actual.pop("--output") == f"models/ppo_v3_clock_{suffix}_v1_seed42"
    assert actual.pop("--summary-output") == (
        f"results/ppo_v3_clock_{suffix}_v1_seed42.training.json"
    )
    expected.pop("--output")
    expected.pop("--summary-output")
    assert actual == expected
    assert int(actual["--timesteps"]) == 200000
    rollout_size = int(actual["--n-steps"]) * int(actual["--n-envs"])
    rollout_count = (int(actual["--timesteps"]) + rollout_size - 1) // rollout_size
    assert rollout_size == 1024 and rollout_count == 196
    assert rollout_count * rollout_size == 200704 and rollout_count * 10 == 1960
    assert 200704 // int(actual["--evaluation-freq"]) == 20
    assert 20 * int(actual["--eval-episodes"]) == 1000
    assert 200704 // int(actual["--checkpoint-freq"]) == 8
    assert actual["--seed"] == "42" and actual["--eval-seed-offset"] == "70000"
    assert actual["--env-seed-stride"] == "1000"


@pytest.mark.parametrize("arm,suffix", [("zero", "zero"), ("clock", "remaining")])
def test_evaluation_arguments_preserve_fixed_final_model_and_development_protocol(arm, suffix):
    model_hash = "b" * 64
    expected = argument_values(historical.evaluation_arguments("padding", model_hash))
    actual = argument_values(runner.evaluation_arguments(arm, model_hash))
    assert actual.pop("--model") == f"models/ppo_v3_clock_{suffix}_v1_seed42.zip"
    assert actual.pop("--output") == (
        f"results/ppo_v3_clock_{suffix}_v1_development_seed40042.csv"
    )
    expected.pop("--model")
    expected.pop("--output")
    assert actual == expected
    assert actual["--episodes"] == "500" and actual["--seed"] == "40042"
    assert actual["--unsafe-ttc"] == "2.0" and actual["--model-sha256"] == model_hash
    assert not any("shield" in name or "intent" in name or "fusion" in name for name in actual)


@pytest.mark.parametrize("arm,suffix", [("zero", "zero"), ("clock", "remaining")])
@pytest.mark.parametrize("mode", ["train", "evaluate"])
def test_stage_outputs_cover_each_reserved_artifact_and_both_console_streams(arm, suffix, mode):
    paths = runner.artifact_paths(arm)
    stem = f"ppo_v3_clock_{suffix}_v1_seed42"
    expected = {
        Path(f"results/{stem}.{mode}.run.json"),
        Path(f"logs/{stem}.{mode}.stdout.log"),
        Path(f"logs/{stem}.{mode}.stderr.log"),
    }
    if mode == "train":
        expected.update({Path(f"models/{stem}.zip"), Path(f"results/{stem}.training.json"),
                         Path(f"logs/{stem}")})
    else:
        evaluation = f"ppo_v3_clock_{suffix}_v1_development_seed40042"
        expected.update({Path(f"results/{evaluation}.csv"),
                         Path(f"results/{evaluation}.summary.json")})
    assert set(runner.stage_outputs(arm, mode)) == expected
    assert paths["model"] == Path(f"models/{stem}.zip")


@pytest.mark.parametrize("arm", ["zero", "clock"])
@pytest.mark.parametrize("mode", ["train", "evaluate"])
def test_cli_accepts_only_explicit_released_stage_and_preservation_guard(arm, mode):
    args = runner.parse_args([arm, mode, "--release-sha256", RELEASE_HASH, "--refuse-overwrite"])
    assert args.arm == arm and args.mode == mode
    assert args.release_sha256 == RELEASE_HASH and args.refuse_overwrite is True


@pytest.mark.parametrize("arguments", [
    ["zero", "train"],
    ["clock", "train", "--refuse-overwrite"],
    ["zero", "train", "--release-sha256", RELEASE_HASH],
    ["padding", "train", "--release-sha256", RELEASE_HASH, "--refuse-overwrite"],
    ["clock", "resume", "--release-sha256", RELEASE_HASH, "--refuse-overwrite"],
    ["zero", "train", "--release-sha256", RELEASE_HASH, "--refuse-overwrite", "--seed", "43"],
    ["zero", "train", "--release-sha256", RELEASE_HASH, "--refuse-overwrite", "--timesteps", "1"],
    ["clock", "evaluate", "--release-sha256", RELEASE_HASH, "--refuse-overwrite", "--best"],
])
def test_cli_rejects_missing_release_guard_resume_and_scientific_overrides(arguments):
    with pytest.raises(SystemExit) as error:
        runner.parse_args(arguments)
    assert error.value.code == 2


def test_main_delegates_exactly_one_requested_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", [
        "clock-comparison", "clock", "train", "--release-sha256", RELEASE_HASH,
        "--refuse-overwrite",
    ])
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", lambda value: None)
    runner.main()
    assert calls == [(("clock", "train", RELEASE_HASH), {})]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def released_fixture(tmp_path, monkeypatch):
    current = {
        "input_sha256": {"scripts/historical.py": "b" * 64},
        "runtime": {"python": "3.12.9", "package_origin": "synthetic-frozen-origin"},
        "dependency_sha256": {"synthetic-dependency": "c" * 64},
        "snapshot_archive_sha256": "d" * 64,
    }
    validation = {"ruff": "passed", "pytest": "passed", "independent_review": "passed",
                  "tests_passed": 927, "completed_utc": "2026-01-01T00:00:00+00:00",
                  "source_head": "f" * 40}
    proof = {"protocol": copy.deepcopy(runner.protocol_description()), **copy.deepcopy(current),
             "validation": copy.deepcopy(validation)}
    write_json(tmp_path / runner.VERIFICATION_PATH, proof)
    validation["verification_sha256"] = runner.sha(tmp_path / runner.VERIFICATION_PATH)
    release = {"schema_version": 1, "status": "released",
               "created_utc": "2026-01-01T00:00:00+00:00",
               "protocol": copy.deepcopy(runner.protocol_description()),
               "validation": validation, **copy.deepcopy(current)}
    write_json(tmp_path / runner.RELEASE_PATH, release)
    monkeypatch.setattr(runner, "current_fingerprints", lambda root: copy.deepcopy(current))
    return SimpleNamespace(root=tmp_path, release=release, proof=proof, current=current)


def test_release_requires_exact_hash_and_all_bound_current_inputs(released_fixture):
    fixture = released_fixture
    digest = runner.sha(fixture.root / runner.RELEASE_PATH)
    assert runner.verify_release(fixture.root, digest) == fixture.release
    with pytest.raises(ValueError, match="Release hash differs"):
        runner.verify_release(fixture.root, "0" * 64)


@pytest.mark.parametrize("digest", ["", "A" * 64, "a" * 63, "not-a-hash", None])
def test_release_digest_is_mandatory_and_strict_before_any_input_check(
        tmp_path, monkeypatch, digest):
    monkeypatch.setattr(runner, "current_fingerprints", forbidden)
    with pytest.raises(ValueError):
        runner.verify_release(tmp_path, digest)


def test_missing_release_fails_without_model_factory_outputs_or_fingerprint_work(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "current_fingerprints", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.run("zero", "train", RELEASE_HASH)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", [
    "status", "schema_bool", "schema_extra", "created_naive", "protocol_order",
    "protocol_seed", "old_source", "dependency", "runtime_origin", "snapshot",
    "ruff_failed", "pytest_failed", "review_missing", "too_few_tests", "boolean_tests",
    "naive_validation", "empty_head", "proof_hash", "proof_contents",
])
def test_release_rejects_changed_protocol_old_pins_runtime_or_validation(
        released_fixture, mutation):
    fixture = released_fixture
    release = copy.deepcopy(fixture.release)
    if mutation == "status":
        release["status"] = "draft"
    elif mutation == "schema_bool":
        release["schema_version"] = True
    elif mutation == "schema_extra":
        release["unregistered_override"] = True
    elif mutation == "created_naive":
        release["created_utc"] = "2026-01-01T00:00:00"
    elif mutation == "protocol_order":
        release["protocol"]["order"].reverse()
    elif mutation == "protocol_seed":
        release["protocol"]["development"]["first_seed"] = 40043
    elif mutation == "old_source":
        fixture.current["input_sha256"]["scripts/historical.py"] = "0" * 64
    elif mutation == "dependency":
        fixture.current["dependency_sha256"]["synthetic-dependency"] = "0" * 64
    elif mutation == "runtime_origin":
        fixture.current["runtime"]["package_origin"] = "other-origin"
    elif mutation == "snapshot":
        fixture.current["snapshot_archive_sha256"] = "0" * 64
    elif mutation in ("ruff_failed", "pytest_failed"):
        release["validation"][mutation.split("_")[0]] = "failed"
    elif mutation == "review_missing":
        release["validation"].pop("independent_review")
    elif mutation == "too_few_tests":
        release["validation"]["tests_passed"] = 926
    elif mutation == "boolean_tests":
        release["validation"]["tests_passed"] = True
    elif mutation == "naive_validation":
        release["validation"]["completed_utc"] = "2026-01-01T00:00:00"
    elif mutation == "empty_head":
        release["validation"]["source_head"] = ""
    elif mutation == "proof_hash":
        release["validation"]["verification_sha256"] = "0" * 64
    else:
        proof = copy.deepcopy(fixture.proof)
        proof["validation"]["tests_passed"] += 1
        write_json(fixture.root / runner.VERIFICATION_PATH, proof)
        release["validation"]["verification_sha256"] = runner.sha(
            fixture.root / runner.VERIFICATION_PATH,
        )
    write_json(fixture.root / runner.RELEASE_PATH, release)
    with pytest.raises(ValueError):
        runner.verify_release(fixture.root, runner.sha(fixture.root / runner.RELEASE_PATH))


def valid_model(arm="zero", timesteps=200704):
    return SimpleNamespace(
        num_timesteps=timesteps, seed=42, n_steps=1024, batch_size=64, n_envs=1, n_epochs=10,
        learning_rate=.0003, gamma=.99, gae_lambda=.95, ent_coef=.01, vf_coef=.5,
        max_grad_norm=.5, target_kl=None, normalize_advantage=True, clip_range_vf=None,
        policy_kwargs={"net_arch": [256, 256]}, _n_updates=1960 if timesteps else 0,
        observation_space=gym.spaces.Box(
            np.asarray([-np.inf] * 105 + [0] + [-1, 0, 0] * 3 + [0], dtype=np.float32),
            np.asarray([np.inf] * 105 + [1] * 11, dtype=np.float32), dtype=np.float32,
        ),
        action_space=gym.spaces.Discrete(3), clip_range=lambda fraction: .2, device="cpu",
        safeintent_observation_protocol=runner.ARMS[arm]["protocol"], safeintent_clock_arm=arm,
        safeintent_clock_duration=30,
    )


@pytest.mark.parametrize("arm", ["zero", "clock"])
@pytest.mark.parametrize("timesteps", [0, 200704])
def test_fake_model_exact_initial_and_final_parameters_are_accepted(arm, timesteps):
    runner._verify_parameters(valid_model(arm, timesteps), arm, timesteps)


@pytest.mark.parametrize("attribute,value", [
    ("num_timesteps", 200000), ("_n_updates", 1950), ("seed", 43), ("n_envs", 2),
    ("n_steps", 2048), ("batch_size", 128), ("learning_rate", .0001), ("n_epochs", 11),
    ("gamma", .95), ("gae_lambda", .9), ("ent_coef", 0), ("normalize_advantage", False),
    ("policy_kwargs", {"net_arch": [128, 128]}),
    ("observation_space", gym.spaces.Box(-1, 1, (115,), dtype=np.float32)),
    ("observation_space", gym.spaces.Box(-1, 1, (163,), dtype=np.float32)),
    ("observation_space", gym.spaces.Box(-1, 1, (116,), dtype=np.float64)),
    ("action_space", gym.spaces.Discrete(5)), ("action_space", gym.spaces.Discrete(3, start=1)),
    ("device", "cuda"), ("safeintent_clock_arm", "clock"), ("safeintent_clock_duration", 29),
    ("safeintent_observation_protocol", "predictive_post_spawn_sync_v1"),
])
def test_fake_model_rejects_wrong_checkpoint_protocol(attribute, value):
    model = valid_model()
    setattr(model, attribute, value)
    with pytest.raises(ValueError):
        runner._verify_parameters(model, "zero", 200704)


@pytest.mark.parametrize("mutation", [None, "initial", "source", "parameter_count"])
def test_checkpoint_initialization_and_release_provenance_guards(monkeypatch, mutation):
    model = valid_model()
    provenance = {"release_sha256": RELEASE_HASH, "input_sha256": {"old.py": "b" * 64}}
    model.safeintent_initial_policy = copy.deepcopy(runner.INITIAL_POLICY)
    model.safeintent_clock_provenance = copy.deepcopy(provenance)
    count = 192516
    if mutation == "initial":
        model.safeintent_initial_policy["sha256"] = "0" * 64
    elif mutation == "source":
        model.safeintent_clock_provenance["input_sha256"]["old.py"] = "0" * 64
    elif mutation == "parameter_count":
        count = 216580
    monkeypatch.setattr(runner, "initial_policy_fingerprint",
                        lambda model: {"parameter_count": count})
    if mutation is None:
        runner.verify_model(model, "zero", provenance)
    else:
        with pytest.raises(ValueError):
            runner.verify_model(model, "zero", provenance)


@pytest.fixture
def adapter_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    model = SimpleNamespace()

    def factory(**kwargs):
        calls.append(("factory", copy.deepcopy(kwargs)))
        return SimpleNamespace(close=lambda: calls.append(("close", kwargs["seed"])))

    def constructor(*args, **kwargs):
        calls.append(("PPO", args, kwargs))
        return model

    module = SimpleNamespace(__name__="fake_historical_main", make_intersection_env=factory,
                             PPO=constructor)
    monkeypatch.setattr(runner, "SynchronizedPredictiveObservation", lambda env: env)
    monkeypatch.setattr(runner, "ClockPredictiveObservation", lambda env, arm: env)
    monkeypatch.setattr(runner, "_verify_parameters", lambda *args: None)
    monkeypatch.setattr(runner, "initial_policy_fingerprint",
                        lambda model: copy.deepcopy(runner.INITIAL_POLICY))
    path = tmp_path / "owned.run.json"
    write_json(path, {})
    return SimpleNamespace(module=module, calls=calls, model=model, path=path,
                           provenance={"release_sha256": RELEASE_HASH,
                                       "input_sha256": {"old.py": "b" * 64}})


def create_factory_pair(module):
    for seed in (42, 70042):
        module.make_intersection_env(config_path=runner.CONFIG, seed=seed)


@pytest.mark.parametrize("arm", ["zero", "clock"])
def test_scoped_training_adapter_stamps_exact_measured_initialization_and_restores(
        adapter_fixture, arm):
    fixture = adapter_fixture
    module = fixture.module
    before = module.make_intersection_env, module.PPO, sys.argv
    record = {"mode": "train"}
    with runner.adapted(module, ["--fixed"], arm, record, fixture.path, fixture.provenance):
        assert sys.argv == [module.__name__, "--fixed"]
        create_factory_pair(module)
        assert module.PPO("MlpPolicy") is fixture.model
        assert fixture.model.safeintent_clock_arm == arm
        assert fixture.model.safeintent_observation_protocol == runner.ARMS[arm]["protocol"]
        assert fixture.model.safeintent_clock_duration == 30
        assert fixture.model.safeintent_initial_policy == runner.INITIAL_POLICY
        assert fixture.model.safeintent_clock_provenance == fixture.provenance
        assert fixture.model.safeintent_clock_provenance is not fixture.provenance
    assert (module.make_intersection_env, module.PPO, sys.argv) == before
    assert [entry[1]["seed"] for entry in fixture.calls if entry[0] == "factory"] == [42, 70042]
    assert fixture.calls[-1][2]["device"] == "cpu"
    assert json.loads(fixture.path.read_text())["initial_policy"] == runner.INITIAL_POLICY


@pytest.mark.parametrize("mutation", ["weights", "parameter_count"])
def test_initial_policy_mismatch_is_preserved_before_return_or_learning(
        adapter_fixture, monkeypatch, mutation):
    fixture = adapter_fixture
    measured = copy.deepcopy(runner.INITIAL_POLICY)
    measured["sha256" if mutation == "weights" else "parameter_count"] = (
        "0" * 64 if mutation == "weights" else 216580
    )
    monkeypatch.setattr(runner, "initial_policy_fingerprint", lambda model: measured)
    old = fixture.module.make_intersection_env, fixture.module.PPO, sys.argv
    returned = []
    with pytest.raises(ValueError, match="Initial tensors"):
        with runner.adapted(fixture.module, [], "zero", {"mode": "train"},
                            fixture.path, fixture.provenance):
            create_factory_pair(fixture.module)
            returned.append(fixture.module.PPO())
    assert not returned
    assert json.loads(fixture.path.read_text())["initial_policy"] == measured
    assert [row for row in fixture.calls if row[0] == "close"] == [("close", 70042), ("close", 42)]
    assert (fixture.module.make_intersection_env, fixture.module.PPO, sys.argv) == old


@pytest.mark.parametrize("kwargs", [
    {"config_path": runner.CONFIG, "seed": 43},
    {"config_path": "changed.yaml", "seed": 42},
    {"config_path": runner.CONFIG, "seed": 42, "safety_shield": True},
    {"config_path": runner.CONFIG, "seed": 42, "cpa_safety_shield": True},
    {"config_path": runner.CONFIG, "seed": 42, "risk_fusion": True},
    {"config_path": runner.CONFIG, "seed": 42, "target_speed_observation": True},
    {"config_path": runner.CONFIG, "seed": 42, "intent_model": "old-model"},
])
def test_adapter_rejects_wrong_factory_seed_config_or_intervention_before_delegate(
        adapter_fixture, kwargs):
    fixture = adapter_fixture
    with pytest.raises(ValueError, match="Factory"):
        with runner.adapted(fixture.module, [], "zero", {"mode": "train"},
                            fixture.path, fixture.provenance):
            fixture.module.make_intersection_env(**kwargs)
    assert fixture.calls == []


@pytest.mark.parametrize("count", [0, 2])
def test_adapter_requires_exactly_one_fresh_constructor(adapter_fixture, count):
    fixture = adapter_fixture
    with pytest.raises(ValueError):
        with runner.adapted(fixture.module, [], "clock", {"mode": "train"},
                            fixture.path, fixture.provenance):
            create_factory_pair(fixture.module)
            for _ in range(count):
                fixture.module.PPO()
    assert sum(row[0] == "PPO" for row in fixture.calls) == min(count, 1)


def test_adapter_preserves_primary_exception_if_owned_environment_cleanup_also_fails(
        adapter_fixture, monkeypatch):
    fixture = adapter_fixture
    primary = RuntimeError("synthetic construction failure")
    old = fixture.module.make_intersection_env, fixture.module.PPO, sys.argv

    def factory(**kwargs):
        def close():
            raise OSError("synthetic close failure")
        return SimpleNamespace(close=close)

    def wrapping_failure(*args, **kwargs):
        raise primary

    fixture.module.make_intersection_env = factory
    monkeypatch.setattr(runner, "ClockPredictiveObservation", wrapping_failure)
    with pytest.raises(RuntimeError) as caught:
        with runner.adapted(fixture.module, [], "zero", {"mode": "train"},
                            fixture.path, fixture.provenance):
            fixture.module.make_intersection_env(config_path=runner.CONFIG, seed=42)
    assert caught.value is primary
    assert any("synthetic close failure" in note for note in primary.__notes__)
    assert fixture.module.make_intersection_env is factory
    assert (fixture.module.PPO, sys.argv) == old[1:]


@pytest.mark.parametrize("arm", ["zero", "clock"])
def test_evaluation_adapter_loads_only_named_final_cpu_model_with_provenance(
        adapter_fixture, monkeypatch, arm):
    fixture = adapter_fixture
    loaded, verified = [], []

    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return fixture.model

    fixture.module.PPO = SimpleNamespace(load=load)
    original = fixture.module.PPO
    monkeypatch.setattr(runner, "verify_model", lambda *args: verified.append(args))
    path = runner.artifact_paths(arm)["model"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fabricated model bytes, never a real policy")
    arguments = runner.evaluation_arguments(arm, runner.sha(path))
    with runner.adapted(fixture.module, arguments, arm, {"mode": "evaluate"},
                        fixture.path, fixture.provenance):
        fixture.module.make_intersection_env(config_path=runner.CONFIG, seed=40042)
        assert fixture.module.PPO.load(path) is fixture.model
    assert loaded == [(path, {"device": "cpu"})]
    assert verified == [(fixture.model, arm, fixture.provenance)]
    assert fixture.module.PPO is original


@pytest.mark.parametrize("mutation", ["other_arm", "best", "cuda"])
def test_evaluation_adapter_refuses_alternate_checkpoint_or_device_before_load(
        adapter_fixture, mutation):
    fixture = adapter_fixture
    fixture.module.PPO = SimpleNamespace(load=forbidden)
    path = runner.artifact_paths("zero")["model"]
    kwargs = {}
    if mutation == "other_arm":
        path = runner.artifact_paths("clock")["model"]
    elif mutation == "best":
        path = Path("logs/best/best_model.zip")
    else:
        kwargs["device"] = "cuda"
    with pytest.raises(ValueError):
        with runner.adapted(fixture.module, [], "zero", {"mode": "evaluate"},
                            fixture.path, fixture.provenance):
            fixture.module.PPO.load(path, **kwargs)


@pytest.mark.parametrize("when", ["before", "during"])
def test_evaluation_adapter_rejects_changed_final_bytes_before_returning_model(
        adapter_fixture, monkeypatch, when):
    fixture = adapter_fixture
    path = runner.artifact_paths("zero")["model"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic original")
    arguments = runner.evaluation_arguments("zero", runner.sha(path))
    loads = []

    def load(path, **kwargs):
        loads.append(True)
        path.write_bytes(b"synthetic changed")
        return fixture.model

    fixture.module.PPO = SimpleNamespace(load=load)
    monkeypatch.setattr(runner, "verify_model", forbidden)
    if when == "before":
        path.write_bytes(b"synthetic changed")
    with pytest.raises(ValueError, match="model bytes changed"):
        with runner.adapted(fixture.module, arguments, "zero", {"mode": "evaluate"},
                            fixture.path, fixture.provenance):
            fixture.module.PPO.load(path)
    assert len(loads) == int(when == "during")


def test_exact_parameter_guard_rejects_wrong_bounds_and_fake_discrete_objects():
    model = valid_model()
    model.observation_space.low[-1] = -1
    with pytest.raises(ValueError):
        runner._verify_parameters(model, "zero", 200704)
    model = valid_model()
    model.action_space = SimpleNamespace(n=3, start=0)
    with pytest.raises(ValueError):
        runner._verify_parameters(model, "zero", 200704)


@pytest.mark.parametrize("outcome", ["success", "failure", "replaced"])
def test_lock_removal_requires_success_and_unchanged_ownership(tmp_path, outcome):
    lock = tmp_path / runner.LOCK_PATH
    if outcome == "success":
        with runner.experiment_lock(tmp_path, "zero", "train"):
            assert json.loads(lock.read_text())["arm"] == "zero"
            before = lock.read_bytes()
            with pytest.raises(FileExistsError):
                with runner.experiment_lock(tmp_path, "clock", "train"):
                    pytest.fail("Concurrent lock acquisition succeeded")
            assert lock.read_bytes() == before
        assert not lock.exists()
    else:
        with pytest.raises((RuntimeError, ValueError)):
            with runner.experiment_lock(tmp_path, "zero", "train"):
                if outcome == "failure":
                    raise RuntimeError("synthetic stage failed")
                write_json(lock, {"other_owner": True})
        assert lock.exists()
        if outcome == "replaced":
            assert json.loads(lock.read_text()) == {"other_owner": True}


def predecessor_fixture(tmp_path, monkeypatch, index):
    release = {"input_sha256": {"old.py": "a" * 64}, "runtime": {"frozen": True}}
    provenance = runner._provenance(release, RELEASE_HASH)
    dependencies = {}
    verified = []
    monkeypatch.setattr(runner, "_verify_summary", lambda *args: None)
    monkeypatch.setattr(runner.PPO, "load", lambda path, **kwargs: (path, kwargs))
    monkeypatch.setattr(runner, "verify_model", lambda *args: verified.append(args))
    for arm, mode in runner.ORDER[:index]:
        paths = runner.artifact_paths(arm)
        model = tmp_path / paths["model"]
        model.parent.mkdir(parents=True, exist_ok=True)
        if not model.exists():
            model.write_bytes(f"synthetic {arm} model".encode())
        key = "training_summary" if mode == "train" else "evaluation_summary"
        model_hash = runner.sha(model)
        write_json(tmp_path / paths[key], {"model_sha256": model_hash})
        record = {"status": "complete", "arm": arm, "mode": mode,
                  "release_sha256": RELEASE_HASH, "provenance": copy.deepcopy(provenance),
                  "arguments": (runner.training_arguments(arm) if mode == "train"
                                else runner.evaluation_arguments(arm, model_hash)),
                  "initial_policy": copy.deepcopy(runner.INITIAL_POLICY),
                  "predecessor_sha256": dependencies.copy(), "model_sha256": model_hash,
                  "summary_sha256": runner.sha(tmp_path / paths[key]),
                  "finished_utc": "2026-01-01T00:00:00+00:00", "tests": {"status": "passed"}}
        keys = ("train_record", "model", "training_summary")
        if mode == "evaluate":
            (tmp_path / paths["evaluation_csv"]).write_bytes(b"synthetic opaque result bytes")
            record["csv_sha256"] = runner.sha(tmp_path / paths["evaluation_csv"])
            keys = ("evaluate_record", "evaluation_csv", "evaluation_summary")
        write_json(tmp_path / paths[f"{mode}_record"], record)
        dependencies.update({paths[k].as_posix(): runner.sha(tmp_path / paths[k]) for k in keys})
    return release, dependencies, verified


@pytest.mark.parametrize("index", range(4))
def test_predecessors_require_all_earlier_stages_and_their_cumulative_release_bound_hashes(
        tmp_path, monkeypatch, index):
    release, dependencies, verified = predecessor_fixture(tmp_path, monkeypatch, index)
    arm, mode = runner.ORDER[index]
    assert runner.check_predecessors(tmp_path, arm, mode, release, RELEASE_HASH) == dependencies
    assert len(verified) == min(index, 2)
    for model, prior_arm, provenance in verified:
        assert model[1] == {"device": "cpu"}
        assert prior_arm in ("zero", "clock")
        assert provenance == runner._provenance(release, RELEASE_HASH)


@pytest.mark.parametrize("mutation", ["failed", "release", "source", "initial", "ancestor",
                                      "tests", "model_bytes", "summary_bytes"])
def test_predecessors_reject_changed_or_incomplete_evidence(tmp_path, monkeypatch, mutation):
    release, _, _ = predecessor_fixture(tmp_path, monkeypatch, 1)
    paths = runner.artifact_paths("zero")
    path = tmp_path / paths["train_record"]
    record = json.loads(path.read_text())
    if mutation == "failed":
        record["status"] = "failed"
    elif mutation == "release":
        record["release_sha256"] = "0" * 64
    elif mutation == "source":
        record["provenance"]["input_sha256"]["old.py"] = "0" * 64
    elif mutation == "initial":
        record["initial_policy"]["sha256"] = "0" * 64
    elif mutation == "ancestor":
        record["predecessor_sha256"] = {"unregistered": "0" * 64}
    elif mutation == "tests":
        record["tests"]["status"] = "failed"
    elif mutation == "model_bytes":
        (tmp_path / paths["model"]).write_bytes(b"changed")
    else:
        write_json(tmp_path / paths["training_summary"], {"model_sha256": "0" * 64})
    write_json(path, record)
    with pytest.raises(ValueError):
        runner.check_predecessors(tmp_path, "clock", "train", release, RELEASE_HASH)


def test_predecessors_reject_missing_stage_and_future_artifacts(tmp_path, monkeypatch):
    release, _, _ = predecessor_fixture(tmp_path, monkeypatch, 0)
    with pytest.raises(FileNotFoundError):
        runner.check_predecessors(tmp_path, "clock", "train", release, RELEASE_HASH)
    future = tmp_path / runner.stage_outputs("clock", "evaluate")[-1]
    future.parent.mkdir(parents=True)
    future.write_bytes(b"preserve future stage")
    with pytest.raises(FileExistsError):
        runner.check_predecessors(tmp_path, "zero", "train", release, RELEASE_HASH)
    assert future.read_bytes() == b"preserve future stage"


@pytest.mark.parametrize("failure", [None, "ruff", "pytest", "missing_count"])
def test_full_gate_runs_only_mocked_ruff_then_pytest_and_fails_closed(
        tmp_path, monkeypatch, failure):
    calls = []

    def command(arguments, **kwargs):
        calls.append((arguments, kwargs))
        is_pytest = "pytest" in arguments
        fails = failure == ("pytest" if is_pytest else "ruff")
        stdout = "927 passed in 1s" if is_pytest and failure != "missing_count" else "checks passed"
        return SimpleNamespace(returncode=int(fails), stdout=stdout, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", command)
    if failure is None:
        result = runner.fresh_test_gate(tmp_path)
        assert result["status"] == "passed" and result["tests_passed"] == 927
    else:
        with pytest.raises(ValueError):
            runner.fresh_test_gate(tmp_path)
    assert calls[0][0] == [sys.executable, "-m", "ruff", "check", "."]
    assert all(kwargs["cwd"] == tmp_path for _, kwargs in calls)
    if failure != "ruff":
        assert calls[1][0] == [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]


@pytest.mark.parametrize("prefix,executable", [
    (r"C:\Python312\python.exe", r"C:\Python312\python.exe"),
    ('"C:\\Program Files\\Python312\\python.exe"', r"C:\Program Files\Python312\python.exe"),
])
@pytest.mark.parametrize("separator", [" ", "\t", " \t  \t"])
@pytest.mark.parametrize("tail", [
    "-m scripts.run_clock_comparison_v1 zero train",
    '-c "print(\'quoted words\')" ""',
    '-c "print(\'x\')" "C:\\folder with spaces\\\\"',
    '-c "print(\\\"escaped\\\")"',
    '-c "first line\r\nsecond line"  \t\r\n',
])
def test_windows_command_parser_preserves_complete_opaque_tail(prefix, executable, separator, tail):
    assert runner._windows_command_parts(prefix + separator + tail) == (executable, tail)


@pytest.mark.parametrize("command", [
    None, 42, True, b"python.exe -m module", "", " ", "\t",
    " python.exe -m module", "\tpython.exe -m module",
    "python.exe", '"python.exe"', "python.exe ", '"python.exe" \t',
    '"" -m module', '"python.exe -m module', '"python.exe"junk -m module',
    'py"thon.exe -m module', "python.exe\r-m module", "python.exe\n-m module",
    '"py\rthon.exe" -m module', '"py\nthon.exe" -m module',
    "py\0thon.exe -m module", "python.exe -m mod\0ule",
])
def test_windows_command_parser_rejects_missing_or_malformed_executable_prefix(command):
    with pytest.raises(ValueError):
        runner._windows_command_parts(command)


def test_process_path_requires_existing_absolute_file_and_resolves_lexical_parent(tmp_path):
    directory = tmp_path / "Program Files" / "unused"
    directory.mkdir(parents=True)
    executable = directory.parent / "python.exe"
    executable.write_bytes(b"non-executable synthetic identity file")
    assert runner._process_path(str(directory / ".." / executable.name)) == executable.resolve()
    assert executable.read_bytes() == b"non-executable synthetic identity file"


@pytest.mark.parametrize("value", [None, 42, True, "", "python.exe", "folder/python.exe",
                                   "C:python.exe", "\0", '"C:\\Python312\\python.exe"'])
def test_process_path_rejects_uninspectable_or_relative_executables(value):
    with pytest.raises((ValueError, OSError)):
        runner._process_path(value)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_process_path_rejects_absent_executable_or_directory(tmp_path, kind):
    path = tmp_path / "python.exe"
    if kind == "directory":
        path.mkdir()
    with pytest.raises((ValueError, OSError)):
        runner._process_path(str(path))


@pytest.fixture
def process_census(tmp_path, monkeypatch):
    root = tmp_path / "repository with spaces"
    launcher = root / ".venv/Scripts/python.exe"
    base = tmp_path / "Program Files/Python312/python.exe"
    other = tmp_path / "Other Python/python.exe"
    for path in (launcher, base, other):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic file identity only; never execute")
    tail = ("-B -m scripts.run_clock_comparison_v1 zero train "
            f"--release-sha256 {RELEASE_HASH} --refuse-overwrite")
    current = {"ProcessId": 10, "ParentProcessId": 5, "ExecutablePath": str(base),
               "CommandLine": f'"{base}" {tail}'}
    parent = {"ProcessId": 5, "ParentProcessId": 1, "ExecutablePath": str(launcher),
              "CommandLine": f'"{launcher}" {tail}'}
    fixture = SimpleNamespace(root=root, launcher=launcher, base=base, other=other,
                              tail=tail, current=current, parent=parent,
                              rows=[current, parent], calls=[])

    def command(arguments, **kwargs):
        fixture.calls.append((arguments, kwargs))
        return SimpleNamespace(stdout=json.dumps(fixture.rows), returncode=0, stderr="")

    monkeypatch.setattr(runner.os, "getpid", lambda: 10)
    monkeypatch.setattr(runner.os, "getppid", lambda: 5)
    monkeypatch.setattr(sys, "executable", str(launcher))
    monkeypatch.setattr(sys, "_base_executable", str(base))
    monkeypatch.setattr(sys, "base_prefix", str(base.parent))
    monkeypatch.setattr(runner.subprocess, "run", command)
    return fixture


@pytest.mark.parametrize("tail", [
    None,
    '-c "print(\'quoted words\')" ""',
    '-c "print(\'x\')" "C:\\folder with spaces\\\\"',
    '-c "print(\\\"escaped\\\")"',
    '-c "line one\r\nline two"  \t\r\n',
])
def test_process_census_accepts_only_realistic_venv_prefix_rewrite(process_census, tail):
    fixture = process_census
    tail = fixture.tail if tail is None else tail
    fixture.current["CommandLine"] = f'"{fixture.base}"\t{tail}'
    fixture.parent["CommandLine"] = f'"{fixture.launcher}" \t  {tail}'
    before = copy.deepcopy(fixture.rows)
    assert fixture.current["CommandLine"] != fixture.parent["CommandLine"]
    runner.no_active_research(fixture.root)
    assert fixture.rows == before and len(fixture.calls) == 1
    command, kwargs = fixture.calls[0]
    assert command[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    assert "Get-CimInstance Win32_Process" in command[4]
    assert kwargs == {"cwd": fixture.root, "check": True, "capture_output": True, "text": True}
    assert all(path.read_bytes() == b"synthetic file identity only; never execute"
               for path in (fixture.launcher, fixture.base, fixture.other))


@pytest.mark.parametrize("tail", [
    "", "-B -m scripts.run_clock_comparison_v1 clock train",
    "-B -m scripts.run_clock_comparison_v1 zero evaluate",
    "-B  -m scripts.run_clock_comparison_v1 zero train",
    '-B -m scripts.run_clock_comparison_v1 "zero" train',
])
def test_process_census_rejects_changed_or_missing_argument_tail(process_census, tail):
    fixture = process_census
    fixture.parent["CommandLine"] = f'"{fixture.launcher}" {tail}'
    with pytest.raises(ValueError):
        runner.no_active_research(fixture.root)


@pytest.mark.parametrize("source", ["current", "parent"])
def test_process_census_rejects_old_identical_whole_command_mock(process_census, source):
    fixture = process_census
    command = getattr(fixture, source)["CommandLine"]
    fixture.current["CommandLine"] = fixture.parent["CommandLine"] = command
    with pytest.raises(ValueError, match="executable identity differs"):
        runner.no_active_research(fixture.root)


def test_process_census_preserves_multiline_tail_instead_of_comparing_first_line(process_census):
    fixture = process_census
    fixture.current["CommandLine"] = f'"{fixture.base}" -c "line one\r\nline two"'
    fixture.parent["CommandLine"] = f'"{fixture.launcher}" -c "line one\r\nchanged line"'
    with pytest.raises(ValueError, match="tails differ"):
        runner.no_active_research(fixture.root)


@pytest.mark.parametrize("mutation", ["extra_argument", "missing_argument", "changed_digest",
                                      "trailing_space", "trailing_tab", "trailing_crlf",
                                      "empty_argument", "changed_interior_space"])
def test_process_census_does_not_normalize_away_any_argument_tail_difference(
        process_census, mutation):
    fixture = process_census
    tail = fixture.tail
    if mutation == "missing_argument":
        tail = tail.removesuffix(" --refuse-overwrite")
    elif mutation == "changed_digest":
        tail = tail.replace(RELEASE_HASH, "b" * 64)
    elif mutation == "changed_interior_space":
        tail = tail.replace("zero train", "zero\ttrain")
    else:
        tail += {"extra_argument": " --extra", "trailing_space": " ",
                 "trailing_tab": "\t", "trailing_crlf": "\r\n", "empty_argument": ' ""'}[mutation]
    fixture.parent["CommandLine"] = f'"{fixture.launcher}" {tail}'
    with pytest.raises(ValueError, match="tails differ"):
        runner.no_active_research(fixture.root)


@pytest.mark.parametrize("row_name", ["current", "parent"])
@pytest.mark.parametrize("mutation", ["wrong_executable", "missing_executable", "null_executable",
                                      "relative_executable", "absent_executable", "wrong_token",
                                      "bare_token", "unquoted_space_path", "empty_command",
                                      "null_command", "nul_command", "malformed_prefix"])
def test_process_census_rejects_unproven_executable_and_command_identity(
        process_census, row_name, mutation):
    fixture = process_census
    row = getattr(fixture, row_name)
    if mutation == "missing_executable":
        row.pop("ExecutablePath")
    elif mutation in ("wrong_executable", "null_executable", "relative_executable",
                       "absent_executable"):
        row["ExecutablePath"] = {
            "wrong_executable": str(fixture.other), "null_executable": None,
            "relative_executable": "python.exe", "absent_executable": str(fixture.root / "gone"),
        }[mutation]
    elif mutation == "wrong_token":
        row["CommandLine"] = f'"{fixture.other}" {fixture.tail}'
    elif mutation == "bare_token":
        row["CommandLine"] = "python.exe " + fixture.tail
    elif mutation == "unquoted_space_path":
        row["CommandLine"] = row["ExecutablePath"] + " " + fixture.tail
    elif mutation == "malformed_prefix":
        row["CommandLine"] = f'"{row["ExecutablePath"]}"extra {fixture.tail}'
    else:
        row["CommandLine"] = {"empty_command": "", "null_command": None,
                              "nul_command": row["CommandLine"] + "\0"}[mutation]
    with pytest.raises((ValueError, OSError)):
        runner.no_active_research(fixture.root)


@pytest.mark.parametrize("mutation", ["sys_executable", "sys_base_executable", "base_prefix",
                                      "base_equals_venv", "missing_launcher", "missing_base",
                                      "parent_missing", "current_missing", "same_pid",
                                      "wrong_child_parent", "parent_is_child", "parent_is_itself"])
def test_process_census_requires_direct_ancestry_and_frozen_interpreter_origins(
        process_census, monkeypatch, mutation):
    fixture = process_census
    if mutation == "sys_executable":
        monkeypatch.setattr(sys, "executable", str(fixture.other))
    elif mutation == "sys_base_executable":
        monkeypatch.setattr(sys, "_base_executable", str(fixture.other))
    elif mutation == "base_prefix":
        monkeypatch.setattr(sys, "base_prefix", str(fixture.other.parent))
    elif mutation == "base_equals_venv":
        monkeypatch.setattr(sys, "_base_executable", str(fixture.launcher))
        monkeypatch.setattr(sys, "base_prefix", str(fixture.launcher.parent))
    elif mutation == "missing_launcher":
        fixture.launcher.unlink()
    elif mutation == "missing_base":
        fixture.base.unlink()
    elif mutation == "parent_missing":
        fixture.rows = [fixture.current]
    elif mutation == "current_missing":
        fixture.rows = [fixture.parent]
    elif mutation == "same_pid":
        monkeypatch.setattr(runner.os, "getppid", lambda: 10)
    elif mutation == "wrong_child_parent":
        fixture.current["ParentProcessId"] = 9
    else:
        fixture.parent["ParentProcessId"] = 10 if mutation == "parent_is_child" else 5
    with pytest.raises((ValueError, OSError)):
        runner.no_active_research(fixture.root)


@pytest.mark.parametrize("field,value", [
    ("ProcessId", True), ("ProcessId", 10.0), ("ProcessId", "10"),
    ("ProcessId", 0), ("ProcessId", -1), ("ProcessId", None),
    ("ParentProcessId", True), ("ParentProcessId", 5.0), ("ParentProcessId", "5"),
    ("ParentProcessId", -1), ("ParentProcessId", None),
])
def test_process_census_rejects_boolean_coerced_or_invalid_identifiers(
        process_census, field, value):
    process_census.current[field] = value
    with pytest.raises(ValueError):
        runner.no_active_research(process_census.root)


@pytest.mark.parametrize("case", ["duplicate_self", "duplicate_parent", "duplicate_other",
                                 "missing_pid", "missing_ppid", "not_a_row", "empty",
                                 "null", "string", "number", "boolean", "single_object"])
def test_process_census_rejects_ambiguous_or_incomplete_enumeration(process_census, case):
    fixture = process_census
    if case.startswith("duplicate"):
        source = fixture.current if case == "duplicate_self" else fixture.parent
        if case == "duplicate_other":
            source = {"ProcessId": 20, "ParentProcessId": 0, "CommandLine": "python notes.py"}
            fixture.rows.append(source)
        fixture.rows.append(copy.deepcopy(source))
    elif case in ("missing_pid", "missing_ppid"):
        fixture.current.pop("ProcessId" if case == "missing_pid" else "ParentProcessId")
    elif case == "not_a_row":
        fixture.rows.append("not a process record")
    else:
        fixture.rows = {"empty": [], "null": None, "string": "bad census", "number": 1,
                        "boolean": True, "single_object": fixture.current}[case]
    with pytest.raises(ValueError):
        runner.no_active_research(fixture.root)


@pytest.mark.parametrize("command", [None, "", " \t\r\n", "python notes.py\0", 42])
def test_process_census_rejects_uninspectable_other_python_even_when_not_research(
        process_census, command):
    fixture = process_census
    fixture.rows.append({"ProcessId": 20, "ParentProcessId": 0, "CommandLine": command})
    with pytest.raises(ValueError):
        runner.no_active_research(fixture.root)


@pytest.mark.parametrize("program", ["train_ppo.py", "evaluate_policy.py",
                                     "run_clock_comparison_v1", "run_sector_timeout_diagnostic_v1",
                                     "run_synchronized_predictive", "run_synchronized_retry"])
def test_process_census_still_rejects_each_competing_research_worker(process_census, program):
    fixture = process_census
    fixture.rows.append({"ProcessId": 20, "ParentProcessId": 0,
                         "ExecutablePath": str(fixture.other),
                         "CommandLine": f'"{fixture.other}" -m scripts.{program}'})
    with pytest.raises(RuntimeError, match="PID 20"):
        runner.no_active_research(fixture.root)


def test_process_census_does_not_exempt_an_arbitrary_python_ancestor(process_census):
    fixture = process_census
    fixture.rows.append({"ProcessId": 1, "ParentProcessId": 0,
                         "ExecutablePath": str(fixture.launcher),
                         "CommandLine": fixture.parent["CommandLine"]})
    with pytest.raises(RuntimeError, match="PID 1"):
        runner.no_active_research(fixture.root)


def test_process_census_allows_inspectable_nonresearch_python_without_identity_exemption(
        process_census):
    fixture = process_census
    fixture.rows.append({"ProcessId": 20, "ParentProcessId": 0,
                         "ExecutablePath": str(fixture.other), "CommandLine": "python notes.py"})
    runner.no_active_research(fixture.root)


@pytest.mark.parametrize("failure", ["subprocess", "invalid_json", "empty_output"])
def test_process_census_fails_closed_when_enumeration_fails(process_census, monkeypatch, failure):
    primary = runner.subprocess.CalledProcessError(1, ["fabricated census"])

    def command(*args, **kwargs):
        if failure == "subprocess":
            raise primary
        return SimpleNamespace(stdout="{" if failure == "invalid_json" else "")

    monkeypatch.setattr(runner.subprocess, "run", command)
    with pytest.raises((ValueError, runner.subprocess.CalledProcessError)) as caught:
        runner.no_active_research(process_census.root)
    if failure == "subprocess":
        assert caught.value is primary


def test_runner_and_independent_auditor_bind_the_exact_launcher_amendment():
    from scripts import audit_clock_comparison_v1 as auditor

    amendment = "V3_CLOCK_LAUNCHER_GUARD_V1.md"
    root = Path(__file__).resolve().parents[1]
    assert runner.PINNED[amendment] == auditor.PINNED[amendment] == runner.sha(root / amendment)
    assert runner.PINNED[runner.DESIGN] == (
        "3266b72e0687e309816375a63377e7f4a7d3fe9d9c4e8461840b8fde7153973c"
    )


@pytest.fixture
def stage_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    release = {"input_sha256": {"old.py": "a" * 64}, "runtime": {"frozen": True},
               "validation": {"tests_passed": 927}}
    calls = []
    paths = runner.artifact_paths("zero")
    monkeypatch.setattr(runner, "verify_release", lambda *args: copy.deepcopy(release))
    monkeypatch.setattr(runner, "check_predecessors", lambda *args: {})
    monkeypatch.setattr(runner, "no_active_research", lambda *args: calls.append("process_check"))

    def gate(root):
        assert (root / runner.LOCK_PATH).exists()
        assert (root / paths["train_record"]).exists()
        calls.append("full_tests")
        return {"status": "passed", "tests_passed": 927}

    @contextmanager
    def adapted(*args):
        calls.append("adapter_enter")
        args[3]["initial_policy"] = copy.deepcopy(runner.INITIAL_POLICY)
        try:
            yield
        finally:
            calls.append("adapter_exit")

    def train():
        calls.append("main")
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        paths["model"].write_bytes(b"synthetic checkpoint stand-in")
        write_json(paths["training_summary"], {"model_sha256": runner.sha(paths["model"])})

    monkeypatch.setattr(runner, "fresh_test_gate", gate)
    monkeypatch.setattr(runner, "adapted", adapted)
    monkeypatch.setattr(runner.train_ppo, "main", train)
    monkeypatch.setattr(runner, "_verify_summary", lambda *args: None)
    monkeypatch.setattr(runner.PPO, "load", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(runner, "verify_model", lambda *args: calls.append("verify_model"))
    return SimpleNamespace(root=tmp_path, release=release, paths=paths, calls=calls, train=train)


def test_successful_stage_runs_once_records_hashes_and_removes_only_owned_lock(stage_fixture):
    fixture = stage_fixture
    result = runner.run("zero", "train", RELEASE_HASH)
    record = json.loads(fixture.paths["train_record"].read_text())
    assert result == record and record["status"] == "complete"
    assert record["tests"]["status"] == "passed" and record["finished_utc"]
    assert record["model_sha256"] == runner.sha(fixture.paths["model"])
    assert record["summary_sha256"] == runner.sha(fixture.paths["training_summary"])
    assert record["initial_policy"] == runner.INITIAL_POLICY
    assert not runner.LOCK_PATH.exists()
    assert fixture.calls.index("full_tests") < fixture.calls.index("main")
    assert fixture.calls.count("main") == 1 and fixture.calls.count("verify_model") == 1
    before = fixture.paths["train_record"].read_bytes()
    with pytest.raises(FileExistsError):
        runner.run("zero", "train", RELEASE_HASH)
    assert fixture.paths["train_record"].read_bytes() == before
    assert fixture.calls.count("main") == 1


@pytest.mark.parametrize("failure", ["main", "test_gate", "test_count", "release_postflight",
                                    "model_hash", "checkpoint_contract"])
def test_failed_stage_keeps_record_console_logs_lock_and_partial_outputs(
        stage_fixture, monkeypatch, failure):
    fixture = stage_fixture
    primary = RuntimeError("synthetic main failure")
    if failure == "main":
        def fail_main():
            fixture.train()
            raise primary
        monkeypatch.setattr(runner.train_ppo, "main", fail_main)
    elif failure == "test_gate":
        def fail_gate(root):
            raise ValueError("synthetic full gate failed")
        monkeypatch.setattr(runner, "fresh_test_gate", fail_gate)
    elif failure == "test_count":
        monkeypatch.setattr(runner, "fresh_test_gate",
                            lambda root: {"status": "passed", "tests_passed": 926})
    elif failure == "release_postflight":
        calls = []
        def release(*args):
            calls.append(True)
            if len(calls) == 3:
                raise ValueError("synthetic source changed")
            return copy.deepcopy(fixture.release)
        monkeypatch.setattr(runner, "verify_release", release)
    elif failure == "model_hash":
        external_open = builtins.open
        def changed_model():
            fixture.train()
            with external_open(fixture.paths["model"], "wb") as stream:
                stream.write(b"external change after summary")
        monkeypatch.setattr(runner.train_ppo, "main", changed_model)
    else:
        def reject_model(*args):
            raise ValueError("synthetic checkpoint contract differs")
        monkeypatch.setattr(runner, "verify_model", reject_model)
    with pytest.raises((RuntimeError, ValueError)) as caught:
        runner.run("zero", "train", RELEASE_HASH)
    if failure == "main":
        assert caught.value is primary
    record = json.loads(fixture.paths["train_record"].read_text())
    assert record["status"] == "failed" and record["error"] and record["finished_utc"]
    assert runner.LOCK_PATH.exists()
    assert all(path.exists() for path in runner.stage_outputs("zero", "train")[-2:])
    before = fixture.paths["train_record"].read_bytes()
    with pytest.raises(FileExistsError):
        runner.run("zero", "train", RELEASE_HASH)
    assert fixture.paths["train_record"].read_bytes() == before
    if failure in ("test_gate", "test_count"):
        assert "main" not in fixture.calls and not fixture.paths["model"].exists()
    else:
        assert fixture.paths["model"].exists() and fixture.paths["training_summary"].exists()


@pytest.mark.parametrize("index", range(6))
def test_existing_stage_artifacts_are_preserved_without_reservation_or_main(stage_fixture, index):
    fixture = stage_fixture
    target = runner.stage_outputs("zero", "train")[index]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target == fixture.paths["logdir"]:
        target.mkdir()
        target = target / "sentinel"
    target.write_bytes(b"existing material")
    with pytest.raises(FileExistsError):
        runner.run("zero", "train", RELEASE_HASH)
    assert target.read_bytes() == b"existing material"
    assert not runner.LOCK_PATH.exists() and "main" not in fixture.calls


@pytest.mark.parametrize("index", range(5))
def test_existing_evaluation_artifacts_are_preserved_before_checkpoint_access(stage_fixture, index):
    target = runner.stage_outputs("zero", "evaluate")[index]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"existing evaluation evidence")
    with pytest.raises(FileExistsError):
        runner.run("zero", "evaluate", RELEASE_HASH)
    assert target.read_bytes() == b"existing evaluation evidence"
    assert not runner.LOCK_PATH.exists() and "main" not in stage_fixture.calls


@pytest.fixture
def fabricated_fingerprint_inventory(tmp_path, monkeypatch):
    digest = "b" * 64
    pins = {f"scripts/historical_{index}.py": digest for index in range(95)}
    pins.update({f"results/opaque_{index}.json": digest for index in range(436)})
    for name in [*list(pins)[:95], *runner.NEW_SOURCES]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fabricated inventory bytes\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PINNED", {runner.ANALYSIS: digest})
    runtime = {"package_origins": "synthetic frozen origins"}
    dependencies = {"synthetic dependency": digest}
    old_release = {"runtime": runtime, "dependency_sha256": dependencies}

    def read(path):
        if Path(path).as_posix().endswith(runner.ANALYSIS):
            return {"input_sha256": pins}
        return old_release

    monkeypatch.setattr(runner, "_read", read)
    monkeypatch.setattr(runner, "sha", lambda path: digest)
    monkeypatch.setattr(runner.sector_timeout_protocol_v1, "verify_snapshot",
                        lambda root: {"archive_fingerprint": {"sha256": digest}})
    monkeypatch.setattr(runner.sector_timeout_protocol_v1, "verify_runtime", lambda root: runtime)
    monkeypatch.setattr(runner.sector_timeout_protocol_v1, "dependency_fingerprints",
                        lambda: dependencies)
    monkeypatch.setattr(runner.torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(runner.torch, "get_num_interop_threads", lambda: 8)
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "scripts/run_clock_comparison_v1.py"))
    for name, module in tuple(sys.modules.items()):
        if module is not None and (name == "safeintent_rl" or
                                   name.startswith(("safeintent_rl.", "scripts."))):
            suffix = "/__init__.py" if hasattr(module, "__path__") else ".py"
            source = tmp_path / (name.replace(".", "/") + suffix)
            monkeypatch.setattr(module, "__file__", str(source))
    return SimpleNamespace(root=tmp_path, pins=pins, runtime=runtime, dependencies=dependencies)


def test_current_fingerprints_accepts_only_complete_mocked_old_and_new_inventory(
        fabricated_fingerprint_inventory):
    fixture = fabricated_fingerprint_inventory
    result = runner.current_fingerprints(fixture.root)
    assert result["runtime"] == fixture.runtime
    assert result["dependency_sha256"] == fixture.dependencies
    assert all(result["input_sha256"][name] == digest for name, digest in fixture.pins.items())
    assert runner.NEW_SOURCES <= result["input_sha256"].keys()


@pytest.mark.parametrize("mutation", ["missing_auditor", "new_source", "old_pin",
                                      "import_origin", "dependency", "runtime", "threads"])
def test_current_fingerprints_rejects_missing_auditor_drift_or_wrong_execution_origin(
        fabricated_fingerprint_inventory, monkeypatch, mutation):
    fixture = fabricated_fingerprint_inventory
    if mutation == "missing_auditor":
        (fixture.root / "scripts/audit_clock_comparison_v1.py").unlink()
    elif mutation == "new_source":
        (fixture.root / "scripts/unreleased.py").write_text("# extra\n", encoding="utf-8")
    elif mutation == "old_pin":
        fixture.pins["scripts/historical_0.py"] = "0" * 64
    elif mutation == "import_origin":
        monkeypatch.setattr(runner.train_ppo, "__file__", str(fixture.root / "wrong/train_ppo.py"))
    elif mutation == "dependency":
        monkeypatch.setattr(runner.sector_timeout_protocol_v1, "dependency_fingerprints",
                            lambda: {"changed": "0" * 64})
    elif mutation == "runtime":
        monkeypatch.setattr(runner.sector_timeout_protocol_v1, "verify_runtime",
                            lambda root: {"package_origins": "different"})
    else:
        monkeypatch.setattr(runner.torch, "get_num_threads", lambda: 1)
    with pytest.raises(ValueError):
        runner.current_fingerprints(fixture.root)


@pytest.mark.parametrize("mode,key", [
    ("train", "model"), ("train", "training_summary"),
    ("evaluate", "evaluation_csv"), ("evaluate", "evaluation_summary"),
])
@pytest.mark.parametrize("route", ["path", "builtin"])
def test_exact_final_write_guard_preserves_late_created_artifacts(tmp_path, mode, key, route):
    target = tmp_path / runner.artifact_paths("zero")[key]
    target.parent.mkdir(parents=True)
    original_path_open, original_open = Path.open, builtins.open
    with runner.exclusive_stage_artifact_writes(tmp_path, "zero", mode):
        # Simulate another writer not running through this process's scoped hooks.
        with original_open(target, "wb") as stream:
            stream.write(b"late external sentinel")
        with pytest.raises(FileExistsError):
            if route == "path":
                target.open("wb")
            else:
                builtins.open(str(target), mode="w", encoding="utf-8")
        assert target.read_bytes() == b"late external sentinel"
    assert (Path.open, builtins.open) == (original_path_open, original_open)


@pytest.mark.parametrize("route", ["path", "builtin"])
def test_write_guard_uses_atomic_exclusive_open_even_if_absence_check_is_stale(
        tmp_path, monkeypatch, route):
    target = tmp_path / runner.artifact_paths("zero")["model"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"external file appeared after absence check")
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path == target else exists(path))
    with runner.exclusive_stage_artifact_writes(tmp_path, "zero", "train"):
        with pytest.raises(FileExistsError):
            if route == "path":
                target.open("wb")
            else:
                builtins.open(target, "wb")
    assert target.read_bytes() == b"external file appeared after absence check"


@pytest.mark.parametrize("mode", ["a", "ab", "r+", "w+", "a+"])
def test_exact_final_write_guard_rejects_append_and_update_modes(tmp_path, mode):
    target = tmp_path / runner.artifact_paths("zero")["training_summary"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve existing")
    with runner.exclusive_stage_artifact_writes(tmp_path, "zero", "train"):
        with pytest.raises(ValueError, match="appended or updated"):
            target.open(mode)
    assert target.read_bytes() == b"preserve existing"


@pytest.mark.parametrize("route", ["path", "builtin"])
def test_write_guard_preserves_partial_failure_and_restores_both_hooks(tmp_path, route):
    target = tmp_path / runner.artifact_paths("zero")["model"]
    target.parent.mkdir(parents=True)
    before = Path.open, builtins.open
    primary = RuntimeError("synthetic serializer failed halfway")
    with pytest.raises(RuntimeError) as caught:
        with runner.exclusive_stage_artifact_writes(tmp_path, "zero", "train"):
            stream = target.open("wb") if route == "path" else builtins.open(target, "wb")
            with stream:
                stream.write(b"partial checkpoint bytes")
                raise primary
    assert caught.value is primary and (Path.open, builtins.open) == before
    assert target.read_bytes() == b"partial checkpoint bytes"


def test_final_write_guard_leaves_callback_best_and_unrelated_io_unchanged(tmp_path):
    callback = tmp_path / runner.artifact_paths("zero")["logdir"] / "best/best_model.zip"
    callback.parent.mkdir(parents=True)
    callback.write_bytes(b"old callback best")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("old", encoding="utf-8")
    protected = tmp_path / runner.artifact_paths("zero")["training_summary"]
    protected.parent.mkdir(parents=True)
    with runner.exclusive_stage_artifact_writes(tmp_path, "zero", "train"):
        callback.write_bytes(b"new callback best")
        with builtins.open(unrelated, "a", encoding="utf-8") as stream:
            stream.write(" appended")
        protected.write_text('{"status":"synthetic"}', encoding="utf-8")
        assert json.loads(protected.read_text())["status"] == "synthetic"
    assert callback.read_bytes() == b"new callback best"
    assert unrelated.read_text() == "old appended"


def synthetic_summary(monkeypatch, arm, mode):
    predictive = {"horizon": 3.0, "margin": .5, "uncertainty_growth": .25,
                  "clearance_scale": 10.0, "speed_scale": 9.0, "max_neighbors": 14}
    monkeypatch.setattr(runner, "load_config",
                        lambda path: {"predictive_safety_observation": copy.deepcopy(predictive)})
    provenance = {"release_sha256": RELEASE_HASH, "input_sha256": {"old.py": "b" * 64},
                  "runtime": {"synthetic": True}, "config_sha256": runner.CONFIG_SHA256}
    paths = runner.artifact_paths(arm)
    summary = {
        "config_path": str(Path(runner.CONFIG)), "config_sha256": runner.CONFIG_SHA256,
        "model_path": str(paths["model"]), "observation_protocol": runner.ARMS[arm]["protocol"],
        "clock_comparison_arm": arm, "clock_duration": 30,
        "initial_policy": copy.deepcopy(runner.INITIAL_POLICY),
        "clock_provenance": copy.deepcopy(provenance),
        "safety_shield": False, "risk_fusion": False, "target_speed_observation": False,
        "target_speed_scale": None, "intent_model_path": None, "intent_model_sha256": None,
        "intent_neighbors": 0, "intent_history_length": None,
        "intent_history_tracking_neighbors": None, "intent_device": None,
        "collision_first_reward": True, "fusion_neighbors": 0, "fusion_features_per_neighbor": 0,
        "fusion_range_scale": None, "fusion_relative_speed_scale": None, "fusion_ttc_scale": None,
        "fusion_cpa_horizon": None, "fusion_cpa_distance_scale": None,
        "predictive_safety_observation": copy.deepcopy(predictive),
    }
    if mode == "train":
        summary.update(
            algorithm="PPO", training_seed=42, internal_evaluation_seed_offset=70000,
            timesteps_requested=200000, timesteps_collected=200704, learning_rate=.0003,
            n_steps=1024, batch_size=64, n_envs=1, env_seed_stride=1000,
            training_seed_offsets=[0], training_initial_seeds=[42], rollout_size=1024,
            eval_episodes=50, evaluation_freq_timesteps=10000,
            evaluation_callback_frequency=10000, checkpoint_freq_timesteps=25000,
            checkpoint_callback_frequency=25000, gamma=.99, gae_lambda=.95, ent_coef=.01,
            policy_network=[256, 256], observation_shape=[116], ttc_threshold=2.0,
        )
    else:
        summary.update(
            episodes=500.0, first_seed=40042, last_seed=40541, unsafe_ttc_threshold=2.0,
            safety_shield_type=None, cpa_time_threshold=None, cpa_distance_threshold=None,
            cpa_horizon=None, cpa_max_range=None, cpa_override_action=None,
            mean_safety_interventions=0.0,
            reference_csv_path=str(Path(f"results/{runner.STEMS['original']}.csv")),
            reference_csv_sha256=runner.REFERENCE_HASHES["original"][0],
        )
    return summary, provenance


@pytest.mark.parametrize("arm", ["zero", "clock"])
@pytest.mark.parametrize("mode", ["train", "evaluate"])
def test_direct_summary_validation_accepts_complete_frozen_synthetic_metadata(
        monkeypatch, arm, mode):
    summary, provenance = synthetic_summary(monkeypatch, arm, mode)
    before = copy.deepcopy(summary)
    runner._verify_summary(summary, arm, mode, provenance)
    assert summary == before


@pytest.mark.parametrize("mode", ["train", "evaluate"])
@pytest.mark.parametrize("mutation", [
    "arm", "protocol", "duration", "config_hash", "config_path", "model_path", "seed",
    "predictive", "missing_null", "wrong_null", "initial_hash", "provenance", "budget",
])
def test_direct_summary_validation_rejects_wrong_or_missing_scientific_fields(
        monkeypatch, mode, mutation):
    summary, provenance = synthetic_summary(monkeypatch, "zero", mode)
    if mutation == "arm":
        summary["clock_comparison_arm"] = "clock"
    elif mutation == "protocol":
        summary["observation_protocol"] = runner.ARMS["clock"]["protocol"]
    elif mutation == "duration":
        summary["clock_duration"] = 29
    elif mutation == "config_hash":
        summary["config_sha256"] = "0" * 64
    elif mutation == "config_path":
        summary["config_path"] = "configs/changed.yaml"
    elif mutation == "model_path":
        summary["model_path"] = str(runner.artifact_paths("clock")["model"])
    elif mutation == "seed":
        summary["training_seed" if mode == "train" else "first_seed"] += 1
    elif mutation == "predictive":
        summary["predictive_safety_observation"]["horizon"] = 4
    elif mutation == "missing_null":
        summary.pop("fusion_range_scale")
    elif mutation == "wrong_null":
        summary["intent_model_path"] = "models/old_intent.pt"
    elif mutation == "initial_hash":
        summary["initial_policy"]["sha256"] = "0" * 64
    elif mutation == "provenance":
        summary["clock_provenance"]["release_sha256"] = "0" * 64
    else:
        summary["timesteps_collected" if mode == "train" else "episodes"] -= 1
    with pytest.raises(ValueError, match="Summary scientific protocol"):
        runner._verify_summary(summary, "zero", mode, provenance)


def test_post_training_fake_cpu_reload_mutation_fails_and_retains_owned_evidence(
        stage_fixture, monkeypatch):
    fixture = stage_fixture

    def mutating_load(path, **kwargs):
        assert kwargs == {"device": "cpu"}
        Path(path).write_bytes(b"synthetic model changed during CPU verification reload")
        return SimpleNamespace()

    monkeypatch.setattr(runner.PPO, "load", mutating_load)
    with pytest.raises(ValueError):
        runner.run("zero", "train", RELEASE_HASH)
    record = json.loads(fixture.paths["train_record"].read_text())
    assert record["status"] == "failed" and record["error"]
    assert runner.LOCK_PATH.exists()
    assert fixture.paths["model"].read_bytes() == (
        b"synthetic model changed during CPU verification reload"
    )
    assert fixture.calls.count("main") == 1
