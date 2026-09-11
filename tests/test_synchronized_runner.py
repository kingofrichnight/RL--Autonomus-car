import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_synchronized_predictive as runner


def test_frozen_arguments_and_separate_names():
    args = runner.training_arguments()
    values = dict(zip(args[:-1:2], args[1:-1:2]))
    assert values["--timesteps"] == "200000"
    assert values["--seed"] == "42" and values["--eval-seed-offset"] == "70000"
    assert values["--n-envs"] == "1" and values["--eval-episodes"] == "50"
    assert args[-1] == "--refuse-overwrite"
    assert runner.STEM in values["--output"]
    evaluation = runner.evaluation_arguments("fingerprint")
    assert evaluation[evaluation.index("--seed") + 1] == "40042"
    assert evaluation[evaluation.index("--episodes") + 1] == "500"
    assert "--safety-shield" not in args + evaluation


def test_adapters_restore_state_on_failure_and_stamp_model(monkeypatch):
    def original_factory():
        return "inner"

    def original_model():
        return SimpleNamespace()
    module = SimpleNamespace(__name__="fake", make_intersection_env=original_factory,
                             PPO=original_model)
    old_argv = sys.argv
    monkeypatch.setattr(runner, "SynchronizedPredictiveObservation", lambda x: ("sync", x))
    with pytest.raises(RuntimeError):
        with runner.adapted(module, ["--test"], True):
            assert module.make_intersection_env() == ("sync", "inner")
            assert module.PPO().safeintent_observation_protocol == runner.PROTOCOL
            assert sys.argv == ["fake", "--test"]
            raise RuntimeError("deliberate failure")
    assert module.make_intersection_env is original_factory
    assert module.PPO is original_model and sys.argv is old_argv


def test_evaluation_keeps_real_model_loader(monkeypatch):
    model = object()
    module = SimpleNamespace(__name__="fake", make_intersection_env=lambda: None, PPO=model)
    monkeypatch.setattr(runner, "SynchronizedPredictiveObservation", lambda x: x)
    with runner.adapted(module, [], False):
        assert module.PPO is model


def test_refuses_existing_artifacts_and_unstamped_model(tmp_path):
    with pytest.raises(FileExistsError):
        runner.refuse_existing([tmp_path])
    runner.refuse_existing([tmp_path / "absent"])
    with pytest.raises(ValueError, match="stamp"):
        runner.verify_model(SimpleNamespace(), {})


def test_summary_path_matches_historical_writer():
    assert Path(runner.TRAIN_SUMMARY).with_suffix(".json") == Path(runner.TRAIN_SUMMARY)
    assert str(Path(runner.EVAL_OUTPUT).with_suffix(".summary.json")).endswith(".summary.json")


def test_real_training_adapter_and_saved_stamp(tmp_path):
    with runner.adapted(runner.train_ppo, [], True):
        env = runner.train_ppo.make_intersection_env(runner.CONFIG)
        try:
            assert isinstance(env, runner.SynchronizedPredictiveObservation)
            model = runner.train_ppo.PPO(
                "MlpPolicy", env, seed=7, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs={"net_arch": [8]}, device="cpu",
            )
            model.learn(total_timesteps=8)
            model.save(tmp_path / "engineering_only")
        finally:
            env.close()
    loaded = runner.PPO.load(tmp_path / "engineering_only.zip", device="cpu")
    assert loaded.safeintent_observation_protocol == runner.PROTOCOL


def test_failed_run_preserves_report_and_restores_adapter(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "fingerprint_inputs", lambda: {})
    monkeypatch.setattr(runner, "version", lambda name: runner.VERSIONS[name])
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.12.9")

    def fail():
        raise RuntimeError("deliberate execution failure")

    original_factory = runner.train_ppo.make_intersection_env
    monkeypatch.setattr(runner.train_ppo, "main", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        runner.run("train")
    report = json.loads(Path(f"results/{runner.STEM}.train.run.json").read_text())
    assert report["status"] == "failed" and "deliberate" in report["error"]
    assert runner.train_ppo.make_intersection_env is original_factory
    with pytest.raises(FileExistsError):
        runner.run("train")
