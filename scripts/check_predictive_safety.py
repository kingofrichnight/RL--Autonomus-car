"""Small engineering check only; never a driving-performance evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.intent.inference import file_sha256

CONFIG = "configs/intersection_v3_predictive_safety_v1.yaml"
CONFIG_SHA256 = "e355448e7ab438840ab34b3c88d3883063dbbfba6f5327e16a2dda9adb4fca14"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Preserve existing engineering evidence: {output}")
    if file_sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError("Engineering config differs from the recorded fingerprint")
    report = {"kind": "engineering_only_not_policy_performance", "config": CONFIG,
              "config_sha256": CONFIG_SHA256, "seed": 7, "status": "running"}
    env = None
    try:
        env = make_intersection_env(CONFIG)
        obs, _ = env.reset(seed=7)
        start = time.perf_counter()
        for _ in range(20):
            forecast = env.forecast()
            assert np.isfinite(forecast).all()
        report["mean_forecast_seconds_initial_state"] = (time.perf_counter() - start) / 20
        model = PPO("MlpPolicy", env, seed=7, n_steps=16, batch_size=16, n_epochs=1,
                    policy_kwargs={"net_arch": [32, 32]}, device="cpu", verbose=0)
        start = time.perf_counter()
        model.learn(total_timesteps=32)
        report["training_seconds"] = time.perf_counter() - start
        assert all(torch.isfinite(value).all() for value in model.policy.state_dict().values())
        with tempfile.TemporaryDirectory(prefix="v3_predictive_engineering_") as temporary:
            checkpoint = Path(temporary) / "smoke.zip"
            model.save(checkpoint)
            restored = PPO.load(checkpoint, env=env, device="cpu")
            assert restored.num_timesteps == 32
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                          restored.predict(obs, deterministic=True)[0])
        report.update({"status": "passed", "observation_shape": list(obs.shape),
                       "training_steps": 32, "finite_parameters": True,
                       "save_reload_action_match": True, "temporary_checkpoint_removed": True})
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if env is not None:
            env.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
