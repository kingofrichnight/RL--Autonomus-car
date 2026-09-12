import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
from scripts import run_sector_comparison_v1 as runner
from scripts import run_synchronized_predictive as historical
from scripts.sector_observation_v1 import SectorPredictiveObservation, sector_features


def argument_values(arguments):
    assert arguments[-1] == "--refuse-overwrite"
    return dict(zip(arguments[:-1:2], arguments[1:-1:2], strict=True))


@pytest.mark.parametrize("arm", ["padding", "sector"])
def test_training_arguments_preserve_frozen_protocol_except_artifact_names(arm):
    expected = argument_values(historical.training_arguments())
    actual = argument_values(runner.training_arguments(arm))
    for key in ("--output", "--summary-output"):
        assert actual.pop(key) != expected.pop(key)
    assert actual == expected
    assert actual["--timesteps"] == "200000"
    assert actual["--seed"] == "42"
    assert actual["--eval-seed-offset"] == "70000"
    assert actual["--n-envs"] == "1"


@pytest.mark.parametrize("arm", ["padding", "sector"])
def test_evaluation_arguments_preserve_fixed_500_development_protocol(arm):
    expected = argument_values(historical.evaluation_arguments("model-fingerprint"))
    actual = argument_values(runner.evaluation_arguments(arm, "model-fingerprint"))
    for key in ("--model", "--output"):
        assert actual.pop(key) != expected.pop(key)
    assert actual == expected
    assert actual["--episodes"] == "500"
    assert actual["--seed"] == "40042"
    assert actual["--unsafe-ttc"] == "2.0"
    assert "--safety-shield" not in actual


def test_padding_and_sector_keep_prefix_action_space_and_short_dynamics_identical():
    padding = runner.PaddingPredictiveObservation(
        SynchronizedPredictiveObservation(make_intersection_env(historical.CONFIG))
    )
    sector = SectorPredictiveObservation(
        SynchronizedPredictiveObservation(make_intersection_env(historical.CONFIG))
    )
    try:
        left, left_info = padding.reset(seed=42)
        right, right_info = sector.reset(seed=42)
        left_info.pop("action")
        right_info.pop("action")
        assert left_info == right_info
        assert padding.action_space == sector.action_space == gym.spaces.Discrete(3)
        assert padding.observation_space == sector.observation_space
        assert padding.observation_space.shape == (163,)
        for action in [2, 1, 0]:
            np.testing.assert_array_equal(left[:115], right[:115])
            np.testing.assert_array_equal(left[115:], np.zeros(48, dtype=np.float32))
            np.testing.assert_array_equal(right[115:], sector_features(right[:115]).ravel())
            assert padding.observation_space.contains(left)
            assert sector.observation_space.contains(right)
            assert padding.unwrapped.time == sector.unwrapped.time
            assert padding.unwrapped.steps == sector.unwrapped.steps
            assert (padding.unwrapped.np_random.bit_generator.state
                    == sector.unwrapped.np_random.bit_generator.state)
            assert len(padding.unwrapped.road.vehicles) == len(sector.unwrapped.road.vehicles)
            for first, second in zip(padding.unwrapped.road.vehicles,
                                     sector.unwrapped.road.vehicles, strict=True):
                np.testing.assert_array_equal(first.position, second.position)
                np.testing.assert_array_equal(first.velocity, second.velocity)
            left, *left_outcome = padding.step(action)
            right, *right_outcome = sector.step(action)
            assert left_outcome == right_outcome
            if left_outcome[1] or left_outcome[2]:
                break
        np.testing.assert_array_equal(left[:115], right[:115])
    finally:
        padding.close()
        sector.close()


def test_padding_augmentation_is_read_only_without_observation_physics_or_rng(monkeypatch):
    sync = SynchronizedPredictiveObservation(make_intersection_env(historical.CONFIG))
    env = runner.PaddingPredictiveObservation(sync)
    try:
        observation, _ = sync.reset(seed=7)
        snapshot = observation.copy()
        observation.flags.writeable = False
        state = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)

        def forbidden(*args, **kwargs):
            pytest.fail("Padding cannot observe, forecast, step, or sample again")

        monkeypatch.setattr(env.unwrapped.observation_type, "observe", forbidden)
        monkeypatch.setattr(sync.env, "forecast", forbidden)
        monkeypatch.setattr(sync, "step", forbidden)
        monkeypatch.setattr(env.unwrapped.road, "step", forbidden)
        first = env._augment(observation)
        second = env._augment(observation)
        assert first.dtype == np.float32
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:115], snapshot)
        np.testing.assert_array_equal(first[115:], np.zeros(48, dtype=np.float32))
        np.testing.assert_array_equal(observation, snapshot)
        assert env.unwrapped.np_random.bit_generator.state == state
    finally:
        env.close()


def test_initial_policy_hash_matches_equal_initializations_and_detects_weight_changes():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        first = SimpleNamespace(policy=torch.nn.Linear(163, 3))
        torch.manual_seed(42)
        second = SimpleNamespace(policy=torch.nn.Linear(163, 3))
    expected = runner.initial_policy_fingerprint(first)
    assert expected == runner.initial_policy_fingerprint(second)
    assert expected["parameter_count"] == 492
    assert len(expected["sha256"]) == 64
    with torch.no_grad():
        second.policy.weight[0, 0] += 1
    changed = runner.initial_policy_fingerprint(second)
    assert changed["sha256"] != expected["sha256"]
    assert changed["parameter_count"] == expected["parameter_count"]


def test_real_cpu_ppo_initialization_matches_across_both_163_wrappers_without_learning():
    environments = []
    fingerprints = []
    try:
        for arm, wrapper in (("padding", runner.PaddingPredictiveObservation),
                             ("sector", SectorPredictiveObservation)):
            env = wrapper(SynchronizedPredictiveObservation(
                make_intersection_env(historical.CONFIG)
            ))
            environments.append(env)
            model = runner.PPO(
                "MlpPolicy", env, learning_rate=.0003, n_steps=1024, batch_size=64,
                gamma=.99, gae_lambda=.95, ent_coef=.01, seed=42,
                policy_kwargs={"net_arch": [256, 256]}, device="cpu", verbose=0,
            )
            model.safeintent_observation_protocol = runner.ARMS[arm]["protocol"]
            model.safeintent_sector_arm = arm
            runner._verify_parameters(model, arm, 0)
            assert model.num_timesteps == 0
            fingerprints.append(runner.initial_policy_fingerprint(model))
        assert fingerprints[0] == fingerprints[1]
        assert fingerprints[0]["parameter_count"] == runner.PARAMETER_COUNT == 216580
    finally:
        for env in environments:
            env.close()


@pytest.mark.parametrize("arm", ["padding", "sector"])
@pytest.mark.parametrize("mode", ["train", "evaluate"])
def test_cli_delegates_explicit_arm_mode_and_overwrite_guard(monkeypatch, arm, mode):
    calls = []
    monkeypatch.setattr(runner, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["sector-comparison", arm, mode, "--refuse-overwrite"])
    runner.main()
    assert calls == [(arm, mode)]


@pytest.mark.parametrize("arguments", [
    ["padding", "train"],
    ["sector", "evaluate"],
    ["unknown", "train", "--refuse-overwrite"],
    ["padding", "resume", "--refuse-overwrite"],
    ["sector", "train", "--refuse-overwrite", "--seed", "43"],
    ["sector", "train", "--refuse-overwrite", "--timesteps", "1000"],
])
def test_cli_rejects_missing_guard_resume_and_protocol_overrides(monkeypatch, arguments):
    calls = []
    monkeypatch.setattr(runner, "run", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["sector-comparison", *arguments])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not calls


def initial_fingerprint():
    return {"sha256": "a" * 64, "parameter_count": runner.PARAMETER_COUNT}


def fake_module(model):
    return SimpleNamespace(__name__="fake_train", make_intersection_env=lambda: None,
                           PPO=lambda *args, **kwargs: model)


@pytest.mark.parametrize("arm", ["padding", "sector"])
def test_training_adapter_stamps_and_records_initial_weights_then_restores(
        monkeypatch, tmp_path, arm):
    model = SimpleNamespace()
    module = fake_module(model)
    original_factory, original_ppo, original_argv = (
        module.make_intersection_env, module.PPO, sys.argv,
    )
    record = {"mode": "train"}
    target = tmp_path / "initial.json"
    checks = []
    monkeypatch.setattr(runner, "_verify_parameters", lambda *args: checks.append(args))
    monkeypatch.setattr(runner, "initial_policy_fingerprint", lambda model: initial_fingerprint())
    with runner.adapted(module, ["--fixed"], arm, record, target, initial_fingerprint()):
        assert sys.argv == ["fake_train", "--fixed"]
        actual = module.PPO()
        assert actual is model
        assert model.safeintent_sector_arm == arm
        assert model.safeintent_observation_protocol == runner.ARMS[arm]["protocol"]
        assert model.safeintent_initial_policy == initial_fingerprint()
        assert model.safeintent_initial_policy is not record["initial_policy"]
        assert json.loads(target.read_text())["initial_policy"] == initial_fingerprint()
    assert checks == [(model, arm, 0)]
    assert (module.make_intersection_env, module.PPO, sys.argv) == (
        original_factory, original_ppo, original_argv,
    )


@pytest.mark.parametrize("mismatch", ["weights", "parameter_count"])
def test_initial_mismatch_is_rejected_and_recorded_before_learn(monkeypatch, tmp_path, mismatch):
    learned = []
    model = SimpleNamespace(learn=lambda: learned.append(True))
    module = fake_module(model)
    original_factory, original_ppo, original_argv = (
        module.make_intersection_env, module.PPO, sys.argv,
    )
    measured = initial_fingerprint()
    measured["sha256" if mismatch == "weights" else "parameter_count"] = (
        "b" * 64 if mismatch == "weights" else 1
    )
    monkeypatch.setattr(runner, "_verify_parameters", lambda *args: None)
    monkeypatch.setattr(runner, "initial_policy_fingerprint", lambda model: measured)
    record = {"mode": "train"}
    target = tmp_path / "mismatch.json"
    with pytest.raises(ValueError, match="Initial policy"):
        with runner.adapted(module, [], "sector", record, target, initial_fingerprint()):
            module.PPO().learn()
    assert not learned
    assert json.loads(target.read_text())["initial_policy"] == measured
    assert (module.make_intersection_env, module.PPO, sys.argv) == (
        original_factory, original_ppo, original_argv,
    )


@pytest.mark.parametrize("constructions", [0, 2])
def test_training_adapter_requires_exactly_one_fresh_policy(monkeypatch, tmp_path, constructions):
    calls = []
    module = fake_module(SimpleNamespace())
    original_constructor = module.PPO

    def build(*args, **kwargs):
        calls.append(True)
        assert kwargs["device"] == "cpu"
        return original_constructor(*args, **kwargs)

    module.PPO = build
    monkeypatch.setattr(runner, "_verify_parameters", lambda *args: None)
    monkeypatch.setattr(runner, "initial_policy_fingerprint", lambda model: initial_fingerprint())
    with pytest.raises(ValueError, match="[Ee]xactly one"):
        with runner.adapted(module, [], "padding", {"mode": "train"}, tmp_path / "record.json"):
            for _ in range(constructions):
                module.PPO()
    assert len(calls) == min(constructions, 1)
    assert module.PPO is build


def test_evaluation_adapter_uses_real_cpu_loader_and_restores_on_failure(monkeypatch, tmp_path):
    model = SimpleNamespace(safeintent_initial_policy=initial_fingerprint())
    calls = []
    module = fake_module(model)

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    module.PPO = SimpleNamespace(load=load)
    monkeypatch.setattr(runner, "_verify_parameters", lambda *args: None)
    before = module.make_intersection_env, module.PPO, sys.argv
    with pytest.raises(RuntimeError, match="deliberate"):
        with runner.adapted(module, [], "sector", {"mode": "evaluate"}, tmp_path / "unused",
                            initial_fingerprint()):
            assert module.PPO.load("frozen-model.zip") is model
            raise RuntimeError("deliberate evaluation failure")
    assert calls == [(("frozen-model.zip",), {"device": "cpu"})]
    assert (module.make_intersection_env, module.PPO, sys.argv) == before
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("mode", ["train", "evaluate"])
def test_adapters_reject_non_cpu_backend_before_model_access(tmp_path, mode):
    module = fake_module(SimpleNamespace())
    with pytest.raises(ValueError, match="frozen CPU"):
        with runner.adapted(module, [], "padding", {"mode": mode}, tmp_path / "unused"):
            if mode == "train":
                module.PPO(device="cuda")
            else:
                module.PPO.load("model.zip", device="cuda")
    assert not (tmp_path / "unused").exists()


def valid_model(arm="padding", timesteps=200704):
    return SimpleNamespace(
        num_timesteps=timesteps, seed=42, n_steps=1024, batch_size=64, n_envs=1, n_epochs=10,
        learning_rate=.0003, gamma=.99, gae_lambda=.95, ent_coef=.01, vf_coef=.5,
        max_grad_norm=.5, target_kl=None, normalize_advantage=True, clip_range_vf=None,
        policy_kwargs={"net_arch": [256, 256]}, _n_updates=1960 if timesteps else 0,
        observation_space=gym.spaces.Box(-1, 1, (163,), dtype=np.float32),
        action_space=gym.spaces.Discrete(3), clip_range=lambda fraction: .2, device="cpu",
        safeintent_observation_protocol=runner.ARMS[arm]["protocol"], safeintent_sector_arm=arm,
    )


@pytest.mark.parametrize("arm", ["padding", "sector"])
@pytest.mark.parametrize("timesteps", [0, 200704])
def test_verify_parameters_accepts_only_registered_initial_and_final_contract(arm, timesteps):
    runner._verify_parameters(valid_model(arm, timesteps), arm, timesteps)


@pytest.mark.parametrize("attribute,value", [
    ("num_timesteps", 200000), ("_n_updates", 1950), ("seed", 43), ("n_envs", 2),
    ("n_steps", 2048), ("batch_size", 128), ("learning_rate", .0001),
    ("observation_space", gym.spaces.Box(-1, 1, (115,), dtype=np.float32)),
    ("observation_space", gym.spaces.Box(-1, 1, (163,), dtype=np.float64)),
    ("action_space", gym.spaces.Discrete(5)), ("device", "cuda"),
    ("safeintent_sector_arm", "sector"),
    ("safeintent_observation_protocol", "predictive_post_spawn_sync_v1"),
])
def test_verify_parameters_rejects_changed_training_or_space_contract(attribute, value):
    model = valid_model()
    setattr(model, attribute, value)
    with pytest.raises(ValueError, match="Checkpoint"):
        runner._verify_parameters(model, "padding", 200704)


@pytest.mark.parametrize("changed", ["versions", "python", "threads", "interop"])
def test_runtime_gate_rejects_changed_environment(monkeypatch, changed):
    monkeypatch.setattr(runner, "version", lambda name: runner.VERSIONS[name])
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.12.9")
    monkeypatch.setattr(runner.torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(runner.torch, "get_num_interop_threads", lambda: 8)
    assert runner.runtime_settings() == {
        "versions": runner.VERSIONS, "python": "3.12.9", "torch_threads": [8, 8],
    }
    if changed == "versions":
        monkeypatch.setattr(runner, "version", lambda name: "unregistered")
    elif changed == "python":
        monkeypatch.setattr(runner.platform, "python_version", lambda: "3.13.0")
    elif changed == "threads":
        monkeypatch.setattr(runner.torch, "get_num_threads", lambda: 1)
    else:
        monkeypatch.setattr(runner.torch, "get_num_interop_threads", lambda: 1)
    with pytest.raises(ValueError, match="Frozen runtime"):
        runner.runtime_settings()


@pytest.mark.parametrize("arm", ["padding", "sector"])
@pytest.mark.parametrize("mode,index", [
    ("train", 0), ("train", 1), ("train", 2), ("train", 3),
    ("evaluate", 0), ("evaluate", 1), ("evaluate", 2),
])
def test_every_existing_stage_output_is_preserved_before_lock_or_execution(
        monkeypatch, tmp_path, arm, mode, index):
    monkeypatch.chdir(tmp_path)
    output = runner.stage_outputs(arm, mode)[index]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == runner.artifact_paths(arm)["logdir"]:
        output.mkdir()
        preserved = output / "sentinel"
    else:
        preserved = output
    preserved.write_bytes(b"preserve existing artifact")

    def forbidden():
        pytest.fail("Existing stage outputs must be rejected before fingerprinting")

    monkeypatch.setattr(runner, "fingerprint_inputs", forbidden)
    with pytest.raises(FileExistsError):
        runner.run(arm, mode)
    assert preserved.read_bytes() == b"preserve existing artifact"
    assert not runner.LOCK_PATH.exists()


def test_experiment_lock_excludes_parallel_stages_and_preserves_stale_lock(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with runner.experiment_lock("padding", "train"):
        contents = runner.LOCK_PATH.read_bytes()
        assert json.loads(contents)["arm"] == "padding"
        with pytest.raises(FileExistsError):
            with runner.experiment_lock("sector", "train"):
                pytest.fail("Parallel stage acquired an existing lock")
        assert runner.LOCK_PATH.read_bytes() == contents
    assert not runner.LOCK_PATH.exists()
    runner.LOCK_PATH.write_bytes(b"preserve stale interruption evidence")
    with pytest.raises(FileExistsError):
        runner.run("padding", "train")
    assert runner.LOCK_PATH.read_bytes() == b"preserve stale interruption evidence"


def test_experiment_lock_is_released_after_caught_stage_exception(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="deliberate"):
        with runner.experiment_lock("padding", "train"):
            raise RuntimeError("deliberate failure")
    assert not runner.LOCK_PATH.exists()


@pytest.mark.parametrize("stage", range(4))
def test_predecessors_require_exact_registered_stage_order(monkeypatch, tmp_path, stage):
    monkeypatch.chdir(tmp_path)
    sources = {"frozen.py": "fingerprint"}
    dependencies = {}
    expected_calls = []
    for arm, mode in runner.ORDER[:stage]:
        paths = runner.artifact_paths(arm)
        keys = ("train_record", "model", "training_summary") if mode == "train" else (
            "evaluate_record", "evaluation_csv", "evaluation_summary",
        )
        record = {"initial_policy": initial_fingerprint(),
                  "predecessor_sha256": dependencies.copy()}
        for key in keys:
            paths[key].parent.mkdir(parents=True, exist_ok=True)
            paths[key].write_text(json.dumps(record) if key.endswith("record") else key,
                                  encoding="utf-8")
        dependencies.update({paths[key].as_posix(): runner.sha(paths[key]) for key in keys})
        expected_calls.append((arm, mode))
    calls = []

    def training(arm, actual_sources):
        assert actual_sources == sources
        calls.append((arm, "train"))
        return json.loads(runner.artifact_paths(arm)["train_record"].read_text())

    def complete(arm, mode, actual_sources):
        assert actual_sources == sources
        calls.append((arm, mode))
        return json.loads(runner.artifact_paths(arm)[f"{mode}_record"].read_text()), {}

    monkeypatch.setattr(runner, "verify_training", training)
    monkeypatch.setattr(runner, "_read_complete_record", complete)
    arm, mode = runner.ORDER[stage]
    assert runner.check_predecessors(arm, mode, sources) == dependencies
    assert calls == expected_calls


@pytest.mark.parametrize("target", ["model", "train_record", "stdout", "stderr"])
def test_future_stage_artifacts_are_refused_before_predecessor_loading(
        monkeypatch, tmp_path, target):
    monkeypatch.chdir(tmp_path)
    if target in ("stdout", "stderr"):
        path = Path(f"logs/{runner.ARMS['sector']['stem']}.train.{target}.log")
    else:
        path = runner.artifact_paths("sector")[target]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preserved future-stage evidence")
    with pytest.raises(FileExistsError):
        runner.check_predecessors("padding", "train", {})
    assert path.read_bytes() == b"preserved future-stage evidence"


def test_predecessors_reject_mismatched_arm_initial_weights(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for arm in runner.ARMS:
        paths = runner.artifact_paths(arm)
        for key in ("train_record", "model", "training_summary"):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
            paths[key].write_bytes(b"fixed ancestor")
    padding_paths = runner.artifact_paths("padding")
    dependencies = {padding_paths[key].as_posix(): runner.sha(padding_paths[key])
                    for key in ("train_record", "model", "training_summary")}
    records = {
        "padding": {"initial_policy": initial_fingerprint(), "predecessor_sha256": {}},
        "sector": {"initial_policy": {**initial_fingerprint(), "sha256": "b" * 64},
                   "predecessor_sha256": dependencies},
    }
    monkeypatch.setattr(runner, "verify_training", lambda arm, sources: records[arm])
    with pytest.raises(ValueError, match="identical policy tensors"):
        runner.check_predecessors("padding", "evaluate", {})


def test_predecessors_reject_unregistered_ancestor_dependencies(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "verify_training", lambda arm, sources: {
        "initial_policy": initial_fingerprint(), "predecessor_sha256": {"extra.json": "changed"},
    })
    with pytest.raises(ValueError, match="predecessor evidence differs"):
        runner.check_predecessors("sector", "train", {})


@pytest.fixture
def fake_training_stage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths = runner.artifact_paths("padding")
    sources = {"frozen.py": "unchanged"}
    calls = []
    monkeypatch.setattr(runner, "fingerprint_inputs", lambda: sources.copy())
    monkeypatch.setattr(runner, "runtime_settings", lambda: {
        "versions": runner.VERSIONS, "python": "3.12.9", "torch_threads": [8, 8],
    })
    monkeypatch.setattr(runner, "check_predecessors", lambda *args: {})
    monkeypatch.setattr(runner, "_verify_summary", lambda *args: None)
    monkeypatch.setattr(runner, "verify_model", lambda *args: calls.append("verified_model"))
    monkeypatch.setattr(runner.PPO, "load", lambda *args, **kwargs: object())

    @contextmanager
    def adapted(module, arguments, arm, record, record_path, expected_initial=None):
        record["initial_policy"] = initial_fingerprint()
        yield

    def train():
        calls.append("main")
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        paths["model"].write_bytes(b"engineering fixture, not a PPO checkpoint")
        paths["training_summary"].write_text(json.dumps({
            "model_sha256": runner.sha(paths["model"]),
        }), encoding="utf-8")

    monkeypatch.setattr(runner, "adapted", adapted)
    monkeypatch.setattr(runner.train_ppo, "main", train)
    return {"paths": paths, "sources": sources, "calls": calls, "train": train}


def test_run_records_completion_hashes_and_never_repeats_finished_stage(fake_training_stage):
    fixture = fake_training_stage
    paths = fixture["paths"]
    runner.run("padding", "train")
    record = json.loads(paths["train_record"].read_text())
    assert record["status"] == "complete"
    assert record["finished_utc"]
    assert record["source_sha256"] == fixture["sources"]
    assert record["initial_policy"] == initial_fingerprint()
    assert record["model_sha256"] == runner.sha(paths["model"])
    assert record["summary_sha256"] == runner.sha(paths["training_summary"])
    assert fixture["calls"] == ["main", "verified_model"]
    assert not runner.LOCK_PATH.exists()
    before = paths["train_record"].read_bytes()
    with pytest.raises(FileExistsError):
        runner.run("padding", "train")
    assert paths["train_record"].read_bytes() == before
    assert fixture["calls"] == ["main", "verified_model"]


@pytest.mark.parametrize("failure", ["main", "changed_sources", "model_hash"])
def test_run_preserves_failed_record_and_partial_outputs(monkeypatch, fake_training_stage, failure):
    fixture = fake_training_stage

    def fail():
        fixture["train"]()
        if failure == "main":
            raise RuntimeError("deliberate main failure")
        if failure == "changed_sources":
            fixture["sources"]["frozen.py"] = "changed after training began"
        else:
            fixture["paths"]["model"].write_bytes(b"model changed after summary")

    monkeypatch.setattr(runner.train_ppo, "main", fail)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run("padding", "train")
    record = json.loads(fixture["paths"]["train_record"].read_text())
    assert record["status"] == "failed"
    assert record["finished_utc"] and record["error"]
    assert fixture["paths"]["model"].exists()
    assert fixture["paths"]["training_summary"].exists()
    assert not runner.LOCK_PATH.exists()
    with pytest.raises(FileExistsError):
        runner.run("padding", "train")
    assert fixture["calls"].count("main") == 1


def test_preflight_fingerprint_failure_creates_no_run_record_and_releases_lock(
        monkeypatch, fake_training_stage):
    def fail():
        raise ValueError("frozen input changed")

    monkeypatch.setattr(runner, "fingerprint_inputs", fail)
    with pytest.raises(ValueError, match="frozen input changed"):
        runner.run("padding", "train")
    assert not fake_training_stage["paths"]["train_record"].exists()
    assert not runner.LOCK_PATH.exists()
    assert not fake_training_stage["calls"]
