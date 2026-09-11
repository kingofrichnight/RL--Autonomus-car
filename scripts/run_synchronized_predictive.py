"""Fixed-protocol entry point for the opt-in synchronized PPO ablation.

Adapters are process-local and restored in finally; historical scripts remain
byte-identical. Both training and evaluation explicitly use the new observation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from stable_baselines3 import PPO

from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
from scripts import evaluate_policy, train_ppo
from scripts.audit_geometry_development import FROZEN, REFERENCE_HASHES, STEMS, sha

PROTOCOL = "predictive_post_spawn_sync_v1"
STEM = "ppo_v3_predictive_sync_v1_seed42"
CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"
MODEL = f"models/{STEM}.zip"
TRAIN_SUMMARY = f"results/{STEM}.training.json"
EVAL_OUTPUT = "results/ppo_v3_predictive_sync_v1_development_seed40042.csv"
SYNC_SOURCE = "safeintent_rl/sensors/synchronized_predictive.py"
SYNC_SHA = "758d665d2c635d88693b1c71845b547d735580221830392c3f328724f68eb200"
VERSIONS = {"stable-baselines3": "2.9.0", "torch": "2.13.0", "numpy": "2.5.2",
            "gymnasium": "1.3.0", "highway-env": "1.12.1"}


def training_arguments() -> list[str]:
    return ["--config", CONFIG, "--config-sha256", FROZEN[CONFIG],
            "--timesteps", "200000", "--seed", "42", "--learning-rate", "0.0003",
            "--n-steps", "1024", "--batch-size", "64", "--n-envs", "1",
            "--env-seed-stride", "1000", "--eval-seed-offset", "70000",
            "--eval-episodes", "50", "--evaluation-freq", "10000",
            "--checkpoint-freq", "25000", "--summary-output", TRAIN_SUMMARY,
            "--output", f"models/{STEM}", "--refuse-overwrite"]


def evaluation_arguments(model_hash: str) -> list[str]:
    return ["--model", MODEL, "--model-sha256", model_hash,
            "--config", CONFIG, "--config-sha256", FROZEN[CONFIG],
            "--episodes", "500", "--seed", "40042", "--unsafe-ttc", "2.0",
            "--reference-csv", f"results/{STEMS['original']}.csv",
            "--reference-csv-sha256", REFERENCE_HASHES["original"][0],
            "--output", EVAL_OUTPUT, "--refuse-overwrite"]


def fingerprint_inputs() -> dict[str, str]:
    expected = {p: digest for p, digest in FROZEN.items()
                if p.startswith(("scripts/", "safeintent_rl/", "configs/"))}
    expected[SYNC_SOURCE] = SYNC_SHA
    for name, hashes in REFERENCE_HASHES.items():
        for suffix, digest in zip((".csv", ".summary.json"), hashes):
            expected[f"results/{STEMS[name]}{suffix}"] = digest
    expected[f"results/{STEMS['geometry']}.csv"] = (
        "e145084f4ccce3d419c34886bd9434c0b5a817fb6fa3c1ceab616a89e1929030")
    expected[f"results/{STEMS['geometry']}.summary.json"] = (
        "227675bacf3f113209d970ec28763ccb57a86e352cfcd546f4ee8f6b1b5ecc61")
    for path, digest in expected.items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    for path in sorted(Path("safeintent_rl").rglob("*.py")):
        expected[path.as_posix()] = sha(path)
    expected["scripts/audit_geometry_development.py"] = sha(
        Path("scripts/audit_geometry_development.py"))
    expected["scripts/run_synchronized_predictive.py"] = sha(Path(__file__))
    return expected


def refuse_existing(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Preserve existing artifact: {path}")


@contextmanager
def adapted(module, arguments: list[str], training: bool):
    """Explicitly adapt only this invocation, not saved source or other processes."""
    old_factory, old_ppo, old_argv = module.make_intersection_env, module.PPO, sys.argv

    def factory(*args, **kwargs):
        inner = old_factory(*args, **kwargs)
        try:
            return SynchronizedPredictiveObservation(inner)
        except BaseException:
            inner.close()
            raise

    def stamped_ppo(*args, **kwargs):
        model = old_ppo(*args, **kwargs)
        model.safeintent_observation_protocol = PROTOCOL
        return model

    try:
        module.make_intersection_env = factory
        if training:
            module.PPO = stamped_ppo
        sys.argv = [module.__name__, *arguments]
        yield
    finally:
        module.make_intersection_env, module.PPO, sys.argv = old_factory, old_ppo, old_argv


def verify_model(model, summary: dict) -> None:
    if getattr(model, "safeintent_observation_protocol", None) != PROTOCOL:
        raise ValueError("Checkpoint lacks the synchronized observation protocol stamp")
    expected = {"num_timesteps": 200704, "seed": 42, "n_steps": 1024,
                "batch_size": 64, "n_envs": 1, "n_epochs": 10, "learning_rate": .0003,
                "gamma": .99, "gae_lambda": .95, "ent_coef": .01, "vf_coef": .5,
                "max_grad_norm": .5, "target_kl": None, "normalize_advantage": True,
                "clip_range_vf": None, "policy_kwargs": {"net_arch": [256, 256]}}
    if any(getattr(model, key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint training parameters differ")
    if (model.observation_space.shape != (115,) or model.action_space.n != 3
            or model.clip_range(1) != .2):
        raise ValueError("Checkpoint space or clipping differs")
    if (summary["timesteps_collected"] != 200704
            or summary["observation_protocol"] != PROTOCOL
            or summary["config_sha256"] != FROZEN[CONFIG]):
        raise ValueError("Training summary protocol differs")


def run(mode: str) -> None:
    record_path = Path(f"results/{STEM}.{mode}.run.json")
    summary_path = Path(TRAIN_SUMMARY if mode == "train" else EVAL_OUTPUT).with_suffix(
        ".json" if mode == "train" else ".summary.json")
    outputs = ([Path(MODEL), Path(TRAIN_SUMMARY), Path("logs") / STEM]
               if mode == "train" else [Path(EVAL_OUTPUT), summary_path])
    refuse_existing([record_path, *outputs])
    sources = fingerprint_inputs()
    runtime = {name: version(name) for name in VERSIONS}
    if runtime != VERSIONS or platform.python_version() != "3.12.9":
        raise ValueError("Frozen runtime changed")
    import torch
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (8, 8):
        raise ValueError("Frozen Torch thread settings changed")
    if mode == "evaluate":
        trained = json.loads(Path(f"results/{STEM}.train.run.json").read_text())
        summary = json.loads(Path(TRAIN_SUMMARY).read_text())
        model_hash = sha(Path(MODEL))
        if (trained["status"] != "complete" or trained["source_sha256"] != sources
                or trained["model_sha256"] != model_hash
                or summary["model_sha256"] != model_hash
                or trained["summary_sha256"] != sha(Path(TRAIN_SUMMARY))):
            raise ValueError("Completed training evidence differs")
        verify_model(PPO.load(MODEL, device="cpu"), summary)
        arguments = evaluation_arguments(model_hash)
    else:
        arguments = training_arguments()
    record = {"status": "running", "mode": mode, "observation_protocol": PROTOCOL,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "arguments": arguments, "source_sha256": sources, "versions": runtime,
              "python": platform.python_version(), "torch_threads": [8, 8]}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    try:
        module = train_ppo if mode == "train" else evaluate_policy
        with adapted(module, arguments, mode == "train"):
            module.main()
        summary = json.loads(summary_path.read_text())
        summary["observation_protocol"] = PROTOCOL
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if fingerprint_inputs() != sources:
            raise ValueError("Inputs changed during execution")
        if mode == "train":
            verify_model(PPO.load(MODEL, device="cpu"), summary)
        record.update(status="complete", model_sha256=sha(Path(MODEL)),
                      summary_sha256=sha(summary_path))
        if mode == "evaluate":
            record["csv_sha256"] = sha(Path(EVAL_OUTPUT))
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Synchronized {mode} complete: {record_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--refuse-overwrite", action="store_true", required=True)
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
