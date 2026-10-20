# Quick start

Use Python 3.11 or 3.12. Run commands from the repository root.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS.

```sh
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python tools/check_repository.py
python -m pytest
python scripts/random_agent.py --episodes 1
```

These checks do not require a saved checkpoint. For optional MP4 recording install
`python -m pip install -r requirements-video.txt`; a trained compatible model is required.

## New PPO smoke run

```sh
python scripts/train_ppo.py --timesteps 128 --n-steps 128 --batch-size 64 --eval-episodes 1 --output models/quickstart_smoke --summary-output results/quickstart_smoke.training.json --refuse-overwrite
python scripts/evaluate_policy.py --model models/quickstart_smoke.zip --episodes 2 --output results/quickstart_smoke.csv
```

This short run tests the pipeline; it is not a research result. Use unique output names
for further runs. See the research workflows for full experiments and seed protocols.

Historical checksum-locked runners may need the original code, artifacts and runtime.
Do not rewrite their manifests to make a new installation appear identical.
