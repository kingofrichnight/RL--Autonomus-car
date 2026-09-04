# SafeIntent-RL

**Intent-Aware and Safety-Constrained Reinforcement Learning for Interactive Autonomous Driving**

SafeIntent-RL is a master's-level research project for studying whether inferred driver behavior and a time-to-collision (TTC) safety shield improve PPO decision-making at unsignalized intersections.

> **Research record:** See [MILESTONES.md](MILESTONES.md) for the complete chronological engineering log, mathematical formulation, experiment results, design decisions, failures, corrections, and planned milestones.

The repository currently provides a complete, runnable research foundation:

- Gymnasium + HighwayEnv intersection setup
- PPO training and evaluation
- cautious, normal, and aggressive NPC behavior profiles
- trajectory collection for intent learning
- a PyTorch GRU intent classifier
- TTC and closing-speed safety metrics
- a safety wrapper that can override unsafe high-level actions
- rule-based and PPO baselines
- video recording, CSV results, tests, and reproducible seeds

## Research comparison

| Method | Purpose |
|---|---|
| Rule-based | Non-learning reference baseline |
| PPO | Standard reinforcement-learning baseline |
| PPO + intent | PPO using learned behavior probabilities |
| PPO + safety | PPO protected by a TTC shield |
| SafeIntent-PPO | Intent-aware PPO with the TTC shield |

## Installation

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/kingofrichnight/RL--Autonomus-car.git
cd RL--Autonomus-car
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Quick start

Watch an untrained/random agent:

```bash
python scripts/random_agent.py --episodes 3 --render
```

Train a PPO baseline:

```bash
python scripts/train_ppo.py --timesteps 200000 --seed 42
```

Evaluate it:

```bash
python scripts/evaluate_policy.py --model models/ppo_intersection.zip --episodes 100
```

Evaluate the TTC rule-based reference with the same seeds:

```bash
python scripts/evaluate_rule_based.py --episodes 500 --seed 42 \
  --output results/rule_based_seed42.csv
```

Train PPO Reward V2 using the isolated reward configuration:

```bash
python scripts/train_ppo.py --config configs/intersection_reward_v2.yaml \
  --timesteps 200000 --seed 42 --output models/ppo_reward_v2_seed42
```

Evaluate Reward V2 with the same configuration and seeds:

```bash
python scripts/evaluate_policy.py --model models/ppo_reward_v2_seed42.zip \
  --config configs/intersection_reward_v2.yaml --episodes 500 --seed 42 \
  --output results/ppo_reward_v2_seed42.csv
```

Train PPO Reward V3 with dense route-progress feedback and a small time cost:

```bash
python scripts/train_ppo.py --config configs/intersection_reward_v3.yaml \
  --timesteps 200000 --seed 42 --output models/ppo_reward_v3_seed42
```

Evaluate Reward V3 over the same 500 environment seeds:

```bash
python scripts/evaluate_policy.py --model models/ppo_reward_v3_seed42.zip \
  --config configs/intersection_reward_v3.yaml --episodes 500 --seed 42 \
  --output results/ppo_reward_v3_seed42.csv
```

Before comparing Reward V4, evaluate the existing V3 model on the untouched holdout seeds 10042–10541:

```bash
python scripts/evaluate_policy.py --model models/ppo_reward_v3_seed42.zip \
  --config configs/intersection_reward_v3.yaml --episodes 500 --seed 10042 \
  --output results/ppo_reward_v3_holdout_seed10042.csv
```

Train risk-aware PPO Reward V4:

```bash
python scripts/train_ppo.py --config configs/intersection_reward_v4.yaml \
  --timesteps 200000 --seed 42 --output models/ppo_reward_v4_seed42
```

Evaluate V4 on the identical holdout seeds:

```bash
python scripts/evaluate_policy.py --model models/ppo_reward_v4_seed42.zip \
  --config configs/intersection_reward_v4.yaml --episodes 500 --seed 10042 \
  --output results/ppo_reward_v4_holdout_seed10042.csv
```

Watch a trained policy drive live:

```bash
python scripts/watch_policy.py --model models/ppo_intersection.zip --episodes 3
```

`watch_policy.py` is autonomous mode: PPO controls the ego vehicle and arrow-key driving is disabled to prevent keyboard actions from conflicting with the learned policy. Close the animation window or press `Ctrl+C` in the terminal to stop it.

Train with the TTC safety shield:

```bash
python scripts/train_ppo.py --timesteps 200000 --safety-shield --seed 42
```

Train the intent-aware PPO variant after training the GRU:

```bash
python scripts/train_ppo.py --timesteps 200000 \
  --intent-model models/intent_gru.pt --output models/ppo_intent
```

Train the complete SafeIntent-PPO variant:

```bash
python scripts/train_ppo.py --timesteps 200000 --intent-model models/intent_gru.pt \
  --safety-shield --output models/safeintent_ppo
```

Record a trained episode:

```bash
python scripts/record_episode.py --model models/ppo_intersection.zip
```

Plot one or more evaluation CSV files:

```bash
python scripts/plot_results.py results/ppo.csv results/safeintent.csv \
  --labels PPO SafeIntent-PPO
```

## Intent prediction pipeline

1. Collect labeled NPC trajectory histories:

```bash
python scripts/collect_intent_data.py --episodes 300 --output data/intent_trajectories.npz
```

2. Train the GRU:

```bash
python scripts/train_intent.py --data data/intent_trajectories.npz --output models/intent_gru.pt
```

3. Inspect test-set performance:

```bash
python scripts/evaluate_intent.py --data data/intent_trajectories.npz --model models/intent_gru.pt
```

The dataset stores a short history of `[relative_x, relative_y, relative_vx, relative_vy, acceleration, distance]`, the hidden behavior label used by the simulator, and the source episode. Train/validation/test splitting is episode-based to prevent overlapping trajectory windows from leaking between splits.

## Repository layout

```text
SafeIntent-RL/
├── configs/                 Experiment configuration
├── safeintent_rl/
│   ├── agents/              Rule-based baseline
│   ├── envs/                Environment and behavior wrappers
│   ├── evaluation/          Episode metrics and summaries
│   ├── intent/              Dataset, GRU model, inference
│   └── safety/              TTC and safety shield
├── scripts/                 Runnable entry points
├── tests/                   Unit tests
├── data/                    Generated datasets (ignored)
├── models/                  Trained checkpoints (ignored)
└── results/                 Versioned evaluation CSV/JSON outputs
```

## Reproducible experiments

Use at least five seeds for final comparisons:

```bash
for seed in 11 22 33 44 55; do
  python scripts/train_ppo.py --seed $seed --timesteps 500000 \
    --output "models/ppo_seed_${seed}"
done
```

Report mean and standard deviation for success rate, collision rate, travel time, minimum TTC, unsafe-TTC events, and safety intervention rate.

## Current scope

This repository intentionally focuses on PPO + intent prediction + safety in HighwayEnv. CARLA, camera/LiDAR perception, world models, GNNs, multi-agent PPO, V2X, and adversarial RL are later extensions rather than part of the initial master's milestone.
