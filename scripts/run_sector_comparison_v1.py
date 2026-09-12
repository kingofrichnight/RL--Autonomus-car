"""Fixed, sequential null-feature versus sector-feature PPO development batch.

Scientific settings are unchanged; only the 48 appended observation values
differ. Historical scripts remain untouched. Read V3_SECTOR_COMPARISON_V1.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from safeintent_rl.config import load_config
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
from scripts import evaluate_policy, run_synchronized_predictive, run_synchronized_retry, train_ppo
from scripts.audit_geometry_development import FROZEN, REFERENCE_HASHES, STEMS, sha
from scripts.sector_observation_v1 import FEATURE_RANGES, SectorPredictiveObservation

ARMS = {
    "padding": {"stem": "ppo_v3_sector_padding_v1_seed42",
                "evaluation_stem": "ppo_v3_sector_padding_v1_development_seed40042",
                "protocol": "predictive_post_spawn_padding_v1"},
    "sector": {"stem": "ppo_v3_sector_features_v1_seed42",
               "evaluation_stem": "ppo_v3_sector_features_v1_development_seed40042",
               "protocol": "predictive_post_spawn_sector_v1"},
}
ORDER = (("padding", "train"), ("sector", "train"),
         ("padding", "evaluate"), ("sector", "evaluate"))
CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"
PREREGISTRATION = "V3_SECTOR_COMPARISON_V1.md"
LOCK_PATH = Path("logs/v3_sector_comparison_v1.active.lock")
PARAMETER_COUNT = 216580
VERSIONS = dict(run_synchronized_predictive.VERSIONS)
RETRY_TRAIN_RECORD = "results/ppo_v3_predictive_sync_v1_retry01_seed42.train.run.json"
RETRY_EVALUATE_RECORD = "results/ppo_v3_predictive_sync_v1_retry01_seed42.evaluate.run.json"
RETRY_AUDIT = "results/ppo_v3_predictive_sync_v1_retry01_development_seed40042.audit.json"
PINNED = {
    PREREGISTRATION:
        "fbc0a4383a0d6224a78f6e4cbd7686cb6fa9593f471daa67a78686b99f74d56d",
    "scripts/run_synchronized_retry.py":
        "bb9dfa47723325176b2727ef74fbbed0b7157dc343479e9712e7f7e003def577",
    "scripts/sector_observation_v1.py":
        "f0a78c5befe3c8e0edcc101ee49e7b2bb18c80be3afc6840e359653f99762919",
    "scripts/audit_synchronized_retry.py":
        "0c7ebea4537425f81edf0a9abed0cd2a1cae13e825403a764c0292127bfc2fd7",
    "models/ppo_v3_predictive_sync_v1_retry01_seed42.zip":
        "8a363a7b5e7caef621d6b616890175c10d457531b4d4a213d05fc2011244f395",
    RETRY_TRAIN_RECORD:
        "de38b7ab02ceb7378ec07de76b3b2c95b8a25306a8ba437e42a06c9f58b97dba",
    "results/ppo_v3_predictive_sync_v1_retry01_seed42.training.json":
        "e858bc356e2f430e409ae96d53c62f164204376ed0799fa75f43fe3524d0edfa",
    RETRY_EVALUATE_RECORD:
        "da3e4e4b98d1f4d754d521d687e300dfa3fee0afb7ed243020d5adef209b5d2d",
    "results/ppo_v3_predictive_sync_v1_retry01_development_seed40042.csv":
        "b2865f8d45db051a450744eb94f95850fd742089f6f12445820d6067a3514dfc",
    "results/ppo_v3_predictive_sync_v1_retry01_development_seed40042.summary.json":
        "763cfef6e8c4a9364f69896c5ede4a6ece7e3f6dcf0e54ea33b0df140da2338d",
    RETRY_AUDIT: "0292a40705889a3b6e75874c4955ecfb5e5ac19420b1909c88d1c104476bd30a",
}


class PaddingPredictiveObservation(gym.Wrapper):
    """Preserve the synchronized 115-vector and append 48 exact float32 zeros."""

    def __init__(self, env: SynchronizedPredictiveObservation) -> None:
        if not isinstance(env, SynchronizedPredictiveObservation):
            raise TypeError("Wrap SynchronizedPredictiveObservation directly")
        super().__init__(env)
        self._check_contract()
        # Exactly the same bounds and dimensions as the sector arm, not a
        # 115-input model with different initialization dimensions.
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low, np.tile([0, 0, -1], 16)])
            .astype(np.float32),
            high=np.concatenate([env.observation_space.high, np.ones(48)]).astype(np.float32),
            dtype=np.float32,
        )

    def _check_contract(self) -> None:
        self.env._check_contract()
        if (self.unwrapped.observation_type.features_range != FEATURE_RANGES
                or self.env.observation_space.shape != (115,)
                or self.env.observation_space.dtype != np.float32):
            raise ValueError("Requires the frozen synchronized geometry contract")

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        self._check_contract()
        original = np.asarray(observation)
        if (original.dtype != np.float32 or original.shape != (115,)
                or not np.isfinite(original).all()
                or not self.env.observation_space.contains(original)):
            raise ValueError("Requires a valid finite float32 synchronized 115-vector")
        native = original[:105].reshape(15, 7)
        if native[0, 0] != 1 or not np.isin(native[:, 0], [0, 1]).all():
            raise ValueError("Requires binary presence and a present ego row")
        if not math.isclose(math.hypot(*native[0, 5:7]), 1, rel_tol=0, abs_tol=1e-5):
            raise ValueError("Requires a unit ego heading")
        return np.concatenate([original, np.zeros(48, dtype=np.float32)])

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info


def artifact_paths(arm: str) -> dict[str, Path]:
    if arm not in ARMS:
        raise ValueError("Unknown sector-comparison arm")
    stem, evaluation = ARMS[arm]["stem"], ARMS[arm]["evaluation_stem"]
    return {"model": Path(f"models/{stem}.zip"),
            "training_summary": Path(f"results/{stem}.training.json"),
            "evaluation_csv": Path(f"results/{evaluation}.csv"),
            "evaluation_summary": Path(f"results/{evaluation}.summary.json"),
            "train_record": Path(f"results/{stem}.train.run.json"),
            "evaluate_record": Path(f"results/{stem}.evaluate.run.json"),
            "logdir": Path("logs") / stem}


def training_arguments(arm: str) -> list[str]:
    paths = artifact_paths(arm)
    return ["--config", CONFIG, "--config-sha256", FROZEN[CONFIG],
            "--timesteps", "200000", "--seed", "42", "--learning-rate", "0.0003",
            "--n-steps", "1024", "--batch-size", "64", "--n-envs", "1",
            "--env-seed-stride", "1000", "--eval-seed-offset", "70000",
            "--eval-episodes", "50", "--evaluation-freq", "10000",
            "--checkpoint-freq", "25000", "--summary-output",
            paths["training_summary"].as_posix(), "--output",
            paths["model"].with_suffix("").as_posix(), "--refuse-overwrite"]


def evaluation_arguments(arm: str, model_hash: str) -> list[str]:
    paths = artifact_paths(arm)
    return ["--model", paths["model"].as_posix(), "--model-sha256", model_hash,
            "--config", CONFIG, "--config-sha256", FROZEN[CONFIG],
            "--episodes", "500", "--seed", "40042", "--unsafe-ttc", "2.0",
            "--reference-csv", f"results/{STEMS['original']}.csv",
            "--reference-csv-sha256", REFERENCE_HASHES["original"][0],
            "--output", paths["evaluation_csv"].as_posix(), "--refuse-overwrite"]


def fingerprint_inputs() -> dict[str, str]:
    sources = dict(run_synchronized_retry.retry_inputs())
    for path, expected in PINNED.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Frozen sector-comparison input changed: {path}")
    for path in (RETRY_TRAIN_RECORD, RETRY_EVALUATE_RECORD):
        previous = json.loads(Path(path).read_text())
        if previous["status"] != "complete" or previous["source_sha256"] != sources:
            raise ValueError("Completed retry source lineage differs")
    if json.loads(Path(RETRY_AUDIT).read_text())["status"] != "verified":
        raise ValueError("An independently verified retry audit is required")
    sources.update(PINNED)
    sources[PREREGISTRATION] = sha(Path(PREREGISTRATION))
    sources["scripts/run_sector_comparison_v1.py"] = sha(Path(__file__))
    return sources


def runtime_settings() -> dict:
    versions = {name: version(name) for name in VERSIONS}
    if (versions != VERSIONS or platform.python_version() != "3.12.9"
            or [torch.get_num_threads(), torch.get_num_interop_threads()] != [8, 8]):
        raise ValueError("Frozen runtime or Torch thread settings changed")
    return {"versions": versions, "python": "3.12.9", "torch_threads": [8, 8]}


def initial_policy_fingerprint(model) -> dict:
    """Hash ordered tensor names, dtypes, shapes and bytes without consuming RNG."""
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("Policy contains nonfinite tensors")
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                              separators=(",", ":")).encode("utf-8")
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return {"sha256": digest.hexdigest(),
            "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters())}


def _verify_parameters(model, arm: str, timesteps: int) -> None:
    expected = {"num_timesteps": timesteps, "seed": 42, "n_steps": 1024,
                "batch_size": 64, "n_envs": 1, "n_epochs": 10, "learning_rate": .0003,
                "gamma": .99, "gae_lambda": .95, "ent_coef": .01, "vf_coef": .5,
                "max_grad_norm": .5, "target_kl": None, "normalize_advantage": True,
                "clip_range_vf": None, "policy_kwargs": {"net_arch": [256, 256]},
                "_n_updates": 1960 if timesteps else 0}
    if any(getattr(model, key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint training parameters differ")
    if (model.observation_space.shape != (163,) or model.action_space.n != 3
            or model.observation_space.dtype != np.float32 or model.clip_range(1) != .2
            or str(model.device) != "cpu"
            or getattr(model, "safeintent_observation_protocol", None) != ARMS[arm]["protocol"]
            or getattr(model, "safeintent_sector_arm", None) != arm):
        raise ValueError("Checkpoint arm, space, device or clipping differs")


def _verify_summary(summary: dict, arm: str, mode: str) -> None:
    paths = artifact_paths(arm)
    expected = {"config_path": str(Path(CONFIG)), "config_sha256": FROZEN[CONFIG],
                "model_path": str(paths["model"]), "observation_protocol": ARMS[arm]["protocol"],
                "sector_comparison_arm": arm, "safety_shield": False, "risk_fusion": False,
                "target_speed_observation": False, "target_speed_scale": None,
                "intent_model_path": None, "intent_model_sha256": None, "intent_neighbors": 0,
                "intent_history_length": None, "intent_history_tracking_neighbors": None,
                "intent_device": None, "collision_first_reward": True,
                "fusion_neighbors": 0, "fusion_features_per_neighbor": 0,
                "fusion_range_scale": None, "fusion_relative_speed_scale": None,
                "fusion_ttc_scale": None, "fusion_cpa_horizon": None,
                "fusion_cpa_distance_scale": None,
                "predictive_safety_observation": load_config(CONFIG)[
                    "predictive_safety_observation"]}
    if mode == "train":
        expected.update(algorithm="PPO", training_seed=42, internal_evaluation_seed_offset=70000,
                        timesteps_requested=200000, timesteps_collected=200704,
                        learning_rate=.0003, n_steps=1024, batch_size=64, n_envs=1,
                        env_seed_stride=1000, training_seed_offsets=[0],
                        training_initial_seeds=[42],
                        rollout_size=1024, eval_episodes=50, evaluation_freq_timesteps=10000,
                        evaluation_callback_frequency=10000, checkpoint_freq_timesteps=25000,
                        checkpoint_callback_frequency=25000, gamma=.99, gae_lambda=.95,
                        ent_coef=.01, policy_network=[256, 256], observation_shape=[163],
                        ttc_threshold=2.0)
    else:
        expected.update(episodes=500.0, first_seed=40042, last_seed=40541,
                        unsafe_ttc_threshold=2.0, safety_shield_type=None,
                        cpa_time_threshold=None, cpa_distance_threshold=None, cpa_horizon=None,
                        cpa_max_range=None, cpa_override_action=None,
                        reference_csv_path=str(Path(f"results/{STEMS['original']}.csv")),
                        reference_csv_sha256=REFERENCE_HASHES["original"][0])
    if any(key not in summary or summary[key] != value for key, value in expected.items()):
        raise ValueError("Summary scientific protocol differs")


def verify_model(model, summary: dict, arm: str) -> None:
    _verify_parameters(model, arm, 200704)
    _verify_summary(summary, arm, "train")
    initial = summary["initial_policy"]
    if (initial["parameter_count"] != PARAMETER_COUNT
            or len(initial["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in initial["sha256"])
            or getattr(model, "safeintent_initial_policy", None) != initial
            or initial_policy_fingerprint(model)["parameter_count"] != PARAMETER_COUNT):
        raise ValueError("Initial policy provenance or parameter count differs")


def _read_complete_record(arm: str, mode: str, sources: dict) -> tuple[dict, dict]:
    paths = artifact_paths(arm)
    record = json.loads(paths[f"{mode}_record"].read_text())
    summary_key = "training_summary" if mode == "train" else "evaluation_summary"
    summary = json.loads(paths[summary_key].read_text())
    model_hash = sha(paths["model"])
    arguments = (training_arguments(arm) if mode == "train"
                 else evaluation_arguments(arm, model_hash))
    expected = {"status": "complete", "arm": arm, "mode": mode,
                "observation_protocol": ARMS[arm]["protocol"], "source_sha256": sources,
                "arguments": arguments, "model_sha256": model_hash,
                "summary_sha256": sha(paths[summary_key]), **runtime_settings()}
    if (any(record.get(key) != value for key, value in expected.items())
            or not record.get("finished_utc") or summary["model_sha256"] != model_hash
            or summary["initial_policy"] != record["initial_policy"]):
        raise ValueError(f"Completed {arm} {mode} evidence differs")
    _verify_summary(summary, arm, mode)
    if mode == "evaluate" and record.get("csv_sha256") != sha(paths["evaluation_csv"]):
        raise ValueError("Completed evaluation CSV fingerprint differs")
    return record, summary


def verify_training(arm: str, sources: dict) -> dict:
    record, summary = _read_complete_record(arm, "train", sources)
    verify_model(PPO.load(artifact_paths(arm)["model"], device="cpu"), summary, arm)
    return record


def stage_outputs(arm: str, mode: str) -> list[Path]:
    paths = artifact_paths(arm)
    keys = (["train_record", "model", "training_summary", "logdir"] if mode == "train"
            else ["evaluate_record", "evaluation_csv", "evaluation_summary"])
    return [paths[key] for key in keys]


def check_predecessors(arm: str, mode: str, sources: dict) -> dict[str, str]:
    """Verify all completed earlier stages and reject any out-of-order future artifact."""
    index = ORDER.index((arm, mode))
    for future_arm, future_mode in ORDER[index + 1:]:
        run_synchronized_predictive.refuse_existing(stage_outputs(future_arm, future_mode))
        stem = ARMS[future_arm]["stem"]
        run_synchronized_predictive.refuse_existing([
            Path(f"logs/{stem}.{future_mode}.{stream}.log") for stream in ("stdout", "stderr")
        ])
    dependencies = {}
    initial = None
    for previous_arm, previous_mode in ORDER[:index]:
        if previous_mode == "train":
            record = verify_training(previous_arm, sources)
            if initial is not None and record["initial_policy"] != initial:
                raise ValueError("The two arms did not start from identical policy tensors")
            initial = record["initial_policy"]
            keys = ("train_record", "model", "training_summary")
        else:
            record, _ = _read_complete_record(previous_arm, previous_mode, sources)
            trained = json.loads(artifact_paths(previous_arm)["train_record"].read_text())
            if record["initial_policy"] != trained["initial_policy"]:
                raise ValueError("Evaluation initial policy lineage differs")
            keys = ("evaluate_record", "evaluation_csv", "evaluation_summary")
        if record["predecessor_sha256"] != dependencies:
            raise ValueError("Completed stage predecessor evidence differs")
        paths = artifact_paths(previous_arm)
        dependencies.update({paths[key].as_posix(): sha(paths[key]) for key in keys})
    return dependencies


def _write_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


@contextmanager
def experiment_lock(arm: str, mode: str):
    """Exclusive cross-process lock; abrupt exits leave evidence requiring diagnosis."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "arm": arm, "mode": mode,
                   "started_utc": datetime.now(timezone.utc).isoformat()}, stream)
    try:
        yield
    finally:
        LOCK_PATH.unlink()


@contextmanager
def adapted(module, arguments: list[str], arm: str, record: dict, record_path: Path,
            expected_initial: dict | None = None):
    """Adapt a dedicated process only; preserve the unmodified train/evaluate mains."""
    old_factory, old_ppo, old_argv = module.make_intersection_env, module.PPO, sys.argv
    constructors = []

    def factory(*args, **kwargs):
        inner = old_factory(*args, **kwargs)
        try:
            synchronized = SynchronizedPredictiveObservation(inner)
            wrapper = (PaddingPredictiveObservation if arm == "padding"
                       else SectorPredictiveObservation)
            return wrapper(synchronized)
        except BaseException:
            inner.close()
            raise

    def stamped_ppo(*args, **kwargs):
        if constructors:
            raise ValueError("Exactly one fresh PPO construction is allowed per training stage")
        if kwargs.get("device", "cpu") != "cpu":
            raise ValueError("Only the frozen CPU backend is allowed")
        kwargs["device"] = "cpu"
        model = old_ppo(*args, **kwargs)
        constructors.append(model)
        model.safeintent_observation_protocol = ARMS[arm]["protocol"]
        model.safeintent_sector_arm = arm
        _verify_parameters(model, arm, 0)
        initial = initial_policy_fingerprint(model)
        record["initial_policy"] = initial
        # Preserve measured initial evidence even if comparison fails. Returning
        # the model is what permits the historical main to call model.learn().
        _write_record(record_path, record)
        if initial["parameter_count"] != PARAMETER_COUNT:
            raise ValueError("Initial policy parameter count differs")
        if expected_initial is not None and initial != expected_initial:
            raise ValueError("Initial policy differs from the padding control")
        model.safeintent_initial_policy = dict(initial)
        return model

    class CpuEvaluationPPO:
        @staticmethod
        def load(*args, **kwargs):
            if kwargs.get("device", "cpu") != "cpu":
                raise ValueError("Only the frozen CPU backend is allowed")
            kwargs["device"] = "cpu"
            model = old_ppo.load(*args, **kwargs)
            _verify_parameters(model, arm, 200704)
            if getattr(model, "safeintent_initial_policy", None) != expected_initial:
                raise ValueError("Evaluation model initialization provenance differs")
            return model

    try:
        module.make_intersection_env = factory
        if record["mode"] == "train":
            module.PPO = stamped_ppo
        else:
            module.PPO = CpuEvaluationPPO
        sys.argv = [module.__name__, *arguments]
        yield
        if record["mode"] == "train" and len(constructors) != 1:
            raise ValueError("Training did not construct exactly one fresh PPO policy")
    finally:
        module.make_intersection_env, module.PPO, sys.argv = old_factory, old_ppo, old_argv


def run(arm: str, mode: str) -> None:
    if (arm, mode) not in ORDER:
        raise ValueError("Only the four preregistered arm/stage combinations are supported")
    paths = artifact_paths(arm)
    run_synchronized_predictive.refuse_existing(stage_outputs(arm, mode))
    with experiment_lock(arm, mode):
        # Recheck after locking, including race with a just-completed process.
        run_synchronized_predictive.refuse_existing(stage_outputs(arm, mode))
        sources, runtime = fingerprint_inputs(), runtime_settings()
        predecessors = check_predecessors(arm, mode, sources)
        expected_initial = None
        if arm == "sector" or mode == "evaluate":
            expected_initial = json.loads(artifact_paths("padding")["train_record"].read_text())[
                "initial_policy"]
        if mode == "train":
            arguments = training_arguments(arm)
        else:
            arguments = evaluation_arguments(arm, sha(paths["model"]))
        record = {"status": "running", "arm": arm, "mode": mode,
                  "observation_protocol": ARMS[arm]["protocol"],
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "arguments": arguments, "source_sha256": sources,
                  "predecessor_sha256": predecessors, **runtime}
        if mode == "evaluate":
            record["initial_policy"] = expected_initial
        record_path = paths[f"{mode}_record"]
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        try:
            module = train_ppo if mode == "train" else evaluate_policy
            with adapted(module, arguments, arm, record, record_path, expected_initial):
                module.main()
            summary_path = paths["training_summary" if mode == "train" else "evaluation_summary"]
            summary = json.loads(summary_path.read_text())
            summary.update(observation_protocol=ARMS[arm]["protocol"],
                           sector_comparison_arm=arm, initial_policy=record["initial_policy"])
            _write_record(summary_path, summary)
            _verify_summary(summary, arm, mode)
            model_hash = sha(paths["model"])
            if summary["model_sha256"] != model_hash:
                raise ValueError("Final model fingerprint differs from summary")
            if mode == "train":
                verify_model(PPO.load(paths["model"], device="cpu"), summary, arm)
            if fingerprint_inputs() != sources:
                raise ValueError("Sources changed during execution")
            if any(sha(Path(path)) != digest for path, digest in predecessors.items()):
                raise ValueError("Predecessor artifacts changed during execution")
            record.update(status="complete", model_sha256=model_hash,
                          summary_sha256=sha(summary_path))
            if mode == "evaluate":
                record["csv_sha256"] = sha(paths["evaluation_csv"])
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _write_record(record_path, record)
    print(f"Sector comparison {arm} {mode} complete: {record_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=tuple(ARMS))
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--refuse-overwrite", action="store_true", required=True)
    args = parser.parse_args()
    run(args.arm, args.mode)


if __name__ == "__main__":
    main()
