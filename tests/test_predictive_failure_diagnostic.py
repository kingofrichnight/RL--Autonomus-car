import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from safeintent_rl.envs.intersection import make_intersection_env
from scripts import diagnose_predictive_failures as diagnostic
from scripts.evaluate_predictive_stress import episode
from scripts.recover_predictive_stress import rows_match


class FixedPolicy:
    def obs_to_tensor(self, observation):
        return torch.as_tensor(observation).unsqueeze(0), False

    def get_distribution(self, observation):
        return SimpleNamespace(distribution=torch.distributions.Categorical(
            probs=torch.tensor([[0.0, 1.0, 0.0]])))


class FixedModel:
    policy = FixedPolicy()

    def predict(self, observation, deterministic=True):
        assert deterministic
        return np.array(1), None


def historical_rows(scenario):
    outcome = "incomplete" if scenario == "dense" else "collision"
    return [{"seed": seed, "success": seed not in diagnostic.SELECTED_SEEDS[scenario],
             "collision": outcome == "collision" and seed in diagnostic.SELECTED_SEEDS[scenario],
             "incomplete": outcome == "incomplete" and seed in diagnostic.SELECTED_SEEDS[scenario],
             "steps": 1, "travel_time": 0.2, "stopped_time": 0.0, "zero_target_time": 0.0,
             "reward": 0.0, "min_ttc": None}
            for seed in range(80042, 80142)]


def test_selection_is_exactly_nine_plus_thirty_seven_and_rejects_changes():
    rows = {name: historical_rows(name) for name in diagnostic.SELECTED_SEEDS}
    protocol = {"first_seed": 80042, "episodes_per_scenario": 100}
    selected = diagnostic.select_cases(rows, protocol)
    assert {name: len(seeds) for name, seeds in selected.items()} == {"dense": 9, "aggressive": 37}
    rows["dense"][0].update(success=False, incomplete=True)
    with pytest.raises(ValueError, match="selection differs"):
        diagnostic.select_cases(rows, protocol)


def test_selection_rejects_duplicate_or_missing_historical_seed():
    rows = {name: historical_rows(name) for name in diagnostic.SELECTED_SEEDS}
    rows["aggressive"][0]["seed"] = rows["aggressive"][1]["seed"]
    with pytest.raises(ValueError, match="seed order"):
        diagnostic.select_cases(rows, {"first_seed": 80042, "episodes_per_scenario": 100})


@pytest.mark.parametrize("arm,config,target", [
    ("candidate", "configs/intersection_v3_predictive_safety_v1.yaml", False),
    ("control", "configs/intersection_reward_v3_collision_first.yaml", True),
])
def test_tracing_preserves_complete_episode_actions_metrics_and_rng(arm, config, target):
    plain = make_intersection_env(config, target_speed_observation=target)
    traced = make_intersection_env(config, target_speed_observation=target)
    try:
        expected = episode(plain, FixedModel(), 7)
        expected_rng = copy.deepcopy(plain.unwrapped.np_random.bit_generator.state)
        stream = io.StringIO()
        torch_rng = torch.random.get_rng_state().clone()
        result = diagnostic.replay_episode(traced, FixedModel(), expected, stream,
                                           arm=arm, scenario="engineering")
        assert result["matched"]
        assert rows_match(expected, result["actual"])
        assert traced.unwrapped.np_random.bit_generator.state == expected_rng
        assert torch.equal(torch.random.get_rng_state(), torch_rng)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert len(records) == expected["steps"] == result["decisions"]
        assert records[0]["live_matching_native_rows"] == [True] * 15
        assert records[0]["policy_step"] == 0
        for record in records:
            assert record["action"] == 1
            assert record["action_probabilities"] == [0, 1, 0]
            assert (record["forecast_features"] is None) == (arm == "control")
            assert "driver_profile_counts" not in record
            assert "intent_label" not in record
    finally:
        plain.close()
        traced.close()


def test_snapshot_read_only_and_birth_ids_persist():
    env = make_intersection_env("configs/intersection_v3_predictive_safety_v1.yaml")
    try:
        obs, _ = env.reset(seed=7)
        trace = diagnostic.TracePolicy(FixedModel(), env, io.StringIO(),
                                       arm="candidate", scenario="engineering", seed=7)
        state = copy.deepcopy(env.unwrapped.np_random.bit_generator.state)
        first = trace.snapshot(obs)
        assert first == trace.snapshot(obs)
        assert state == env.unwrapped.np_random.bit_generator.state
        assert trace.birth_ids[env.unwrapped.vehicle] == 0
        assert all(type(value) is int for value in first["live_selected_birth_ids"])
        assert first["native_y_boundary_slots"] > 0
        with pytest.raises(ValueError, match="deterministic"):
            trace.predict(obs, deterministic=False)
        with pytest.raises(ValueError, match="shape"):
            trace.snapshot(obs[:-1])
    finally:
        env.close()


def test_real_ppo_trace_preserves_episode_and_torch_rng():
    config = "configs/intersection_reward_v3_collision_first.yaml"
    plain = make_intersection_env(config, target_speed_observation=True)
    traced = make_intersection_env(config, target_speed_observation=True)
    try:
        model = diagnostic.PPO(
            "MlpPolicy", plain, seed=7, n_steps=8, batch_size=8,
            policy_kwargs={"net_arch": [8]}, device="cpu",
        )
        expected = episode(plain, model, 7)
        expected_rng = copy.deepcopy(plain.unwrapped.np_random.bit_generator.state)
        torch_rng = torch.random.get_rng_state().clone()
        stream = io.StringIO()
        result = diagnostic.replay_episode(
            traced, model, expected, stream, arm="control", scenario="engineering",
        )
        assert result["matched"]
        assert traced.unwrapped.np_random.bit_generator.state == expected_rng
        assert torch.equal(torch.random.get_rng_state(), torch_rng)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert len(records) == expected["steps"]
        assert all(record["action"] == np.argmax(record["action_probabilities"])
                   for record in records)
    finally:
        plain.close()
        traced.close()


def test_reproduction_mismatch_is_returned_with_both_rows():
    env = make_intersection_env("configs/intersection_reward_v3_collision_first.yaml",
                                target_speed_observation=True)
    try:
        reference = episode(env, FixedModel(), 7)
        reference["reward"] += 1
        result = diagnostic.replay_episode(env, FixedModel(), reference, io.StringIO(),
                                           arm="control", scenario="engineering")
        assert not result["matched"]
        assert result["reference"]["reward"] == pytest.approx(result["actual"]["reward"] + 1)
    finally:
        env.close()


def test_run_refuses_overwrite_and_wrong_protocol_before_reading_references(tmp_path):
    with pytest.raises(FileExistsError, match="Preserve"):
        diagnostic.run(tmp_path / "missing", tmp_path / "missing", tmp_path,
                       diagnostic.PROTOCOL_SHA256)
    with pytest.raises(ValueError, match="protocol hash"):
        diagnostic.run(tmp_path, tmp_path, tmp_path / "output", "wrong")
    assert not (tmp_path / "output").exists()


def test_failed_model_load_retains_failed_report(monkeypatch, tmp_path):
    protocol = json.loads(Path("configs/v3_predictive_stress_v1.json").read_text())
    root = tmp_path / "reference"
    candidate = root / "candidate_recovery_01"
    for folder in (root / "control", candidate):
        folder.mkdir(parents=True)
        for name in diagnostic.SELECTED_SEEDS:
            (folder / f"{name}.episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in historical_rows(name)), encoding="utf-8")
    (root / "control" / "report.json").write_text(json.dumps({
        "protocol": protocol, "versions": {}, "source_sha256": {}}), encoding="utf-8")
    monkeypatch.setattr(diagnostic, "analyze", lambda *_: {
        "protocol_sha256": diagnostic.PROTOCOL_SHA256, "episodes": 1000,
        "arm_report_sha256": {"control": "test", "candidate": "test"}})

    def fail_load(*args, **kwargs):
        raise RuntimeError("deliberate load failure")

    monkeypatch.setattr(diagnostic.PPO, "load", fail_load)
    output = tmp_path / "diagnostic"
    with pytest.raises(RuntimeError, match="deliberate load failure"):
        diagnostic.run(root, candidate, output, diagnostic.PROTOCOL_SHA256)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["completed_replays"] == report["matched_replays"] == 0
    assert "deliberate load failure" in report["error"]
    assert "episode_checks.jsonl" in report["artifacts_sha256"]
