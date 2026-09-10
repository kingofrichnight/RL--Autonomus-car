# SafeIntent-RL

**Intent-Aware and Safety-Constrained Reinforcement Learning for Interactive Autonomous Driving**

SafeIntent-RL is a master's-level research project for studying whether inferred driver behavior and a time-to-collision (TTC) safety shield improve PPO decision-making at unsignalized intersections.

> **Research record:** See [MILESTONES.md](MILESTONES.md) for the complete chronological engineering log, mathematical formulation, experiment results, design decisions, failures, corrections, and planned milestones.

## Reference-driven extension (separate from the intersection benchmark)

### New V3 successor: V3-PredictiveSafety v1

The separately named [V3-PredictiveSafety](V3_PREDICTIVE_SAFETY.md) engineering
candidate preserves V3's intersection dynamics and reward coefficients, enables
the documented collision-first correction, and appends per-action geometric
forecasts to the original kinematics. PPO still chooses all actions; there is
no automatic shield or labeled-action training. The new 115-input policy needs
its own training and must not load the original 105-input V3 checkpoint.
This is not yet a measured safety improvement or a released long training run.

### Separate obstacle and pedestrian scenario track

The selected [REFERENCE.md](REFERENCE.md) motivates a staged hybrid architecture:
deterministic routing, tactical control, and a separately evaluated safety layer.
The first implementation, `obstacle_route_v1`, has two straight lanes with one
static blockage and either an open alternate lane or a second blockage.
It provides graph-based route selection, lateral actions, controller/route
observations, and a rule-based feasibility evaluator. A blocked-route safe stop
is explicitly **not** route success. It is not a pedestrian implementation or
a predictive safety shield, and its easier scenario rates must not be compared
to V3 intersection rates.

See MILESTONES.md section 70 for the exact protocol, engineering checks, and
remaining stages. Existing intersection factories, PPO commands, checkpoints,
and the section 69 corrected-reward control experiment are unchanged. Do not
load a V3 checkpoint into the new 27-input/five-action environment.

The 20-case B0 engineering check passed: ten open-lane route completions and
ten safe blocked stops, with no collisions. These are simple static-scenario
checks, not learned-policy performance or evidence of pedestrian safety.

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

Train the first controlled intent-aware PPO variant with the accepted GRU and V3 reward:

```bash
python scripts/train_ppo.py --config configs/intersection_reward_v3.yaml \
  --timesteps 200000 --seed 42 --intent-model models/intent_gru_seed42.pt \
  --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 \
  --intent-neighbors 5 --intent-device cpu --eval-seed-offset 1000 \
  --summary-output results/ppo_intent_v1_seed42.training.json \
  --output models/ppo_intent_v1_seed42
```

Evaluate it on the paired V3 holdout only after recording the training summary:

```bash
python scripts/evaluate_policy.py --model models/ppo_intent_v1_seed42.zip \
  --config configs/intersection_reward_v3.yaml --episodes 500 --seed 10042 \
  --intent-model models/intent_gru_seed42.pt \
  --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 \
  --intent-neighbors 5 --intent-device cpu \
  --output results/ppo_intent_v1_holdout_seed10042.csv
```

PPO + intent V1 was rejected after that holdout: it produced 59.0% success and
41.0% collision versus V3's 59.6% and 40.4%. Before any retraining, run the
frozen non-interventional diagnostic on the same policy and seeds:

```bash
python scripts/diagnose_intent_rollout.py \
  --model models/ppo_intent_v1_seed42.zip \
  --model-sha256 954a1d4ef9431ca451de367213d6b65c087d11f277802f9d7bc1ac38c47471e8 \
  --config configs/intersection_reward_v3.yaml \
  --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 \
  --intent-model models/intent_gru_seed42.pt \
  --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 \
  --intent-neighbors 5 --intent-device cpu --episodes 500 --seed 10042 \
  --unsafe-ttc 2.0 \
  --reference-csv results/ppo_intent_v1_holdout_seed10042.csv \
  --reference-csv-sha256 72fe15fb876e5ad16007c97e01a8811aa62c5935468c21dac49df8edcdebb498 \
  --output results/ppo_intent_v1_online_diagnostics_seed10042.json
```

The diagnostic reads simulator intent labels only after inference for aggregate
analysis; labels never enter the policy observation or action path. It also
requires exact reproduction of the committed 500 episode rows. Commit only its
small JSON output, not either model checkpoint.

The frozen 500-episode diagnostic completed with exact reference reproduction.
The GRU remained effective online (64.83% accuracy and 64.05% macro F1), but only
43.75% of visible-vehicle decision slots had the required history. The next step
is therefore this frozen non-interventional history-coverage feasibility study:

```bash
python scripts/diagnose_intent_rollout.py \
  --model models/ppo_intent_v1_seed42.zip \
  --model-sha256 954a1d4ef9431ca451de367213d6b65c087d11f277802f9d7bc1ac38c47471e8 \
  --config configs/intersection_reward_v3.yaml \
  --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 \
  --intent-model models/intent_gru_seed42.pt \
  --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 \
  --intent-neighbors 5 --shadow-history-neighbors 14 --intent-device cpu \
  --episodes 500 --seed 10042 --unsafe-ttc 2.0 \
  --reference-csv results/ppo_intent_v1_holdout_seed10042.csv \
  --reference-csv-sha256 72fe15fb876e5ad16007c97e01a8811aa62c5935468c21dac49df8edcdebb498 \
  --reference-diagnostic-json results/ppo_intent_v1_online_diagnostics_seed10042.json \
  --reference-diagnostic-json-sha256 c39ffdc93c6f46ab75b4624b8b48ec04caab877ef310bbf14148e1d9504e4045 \
  --output results/ppo_intent_v1_shadow_history_diagnostics_seed10042.json
```

The shadow store tracks all 14 traffic rows already available in the base
observation but only measures counterfactual readiness for the same five output
slots. It never changes the probabilities received by PPO. Do not retrain PPO or
the GRU yet. PPO V3 remains the best driving policy.

The 500-episode shadow diagnostic reproduced both references and improved
coverage from 43.75% to 63.88%, but it failed the frozen 70% feasibility gate.
Shadow accuracy (62.53%) and macro F1 (62.04%) still passed. Wider tracking alone
was therefore insufficient.

The subsequent M9A coverage-only selection run reproduced the complete M8B,
M8C, and M8D reference chain. Under the frozen rule, an eight-observation history
was the longest candidate to pass: it reached 71.23% coverage, while length nine
reached only 67.52%. This was a non-interventional development result; driving
remained 59.0% success and 41.0% collision. The next step is to implement and
freeze the leakage-safe eight-step GRU training/evaluation protocol.

That protocol is now implemented. The single eight-step training run passed its
validation gate: 60.98% accuracy, 59.95% macro F1, and 47.47% minimum class
recall. Its local checkpoint SHA-256 is
`74a72cf2b99b115bb5b4d55fdb350b55e20551876f6ba41be5266d8953b7fc05`,
and independent inspection confirmed that held-out test metrics remain sealed.

The one permitted held-out evaluation also passed: 60.18% accuracy, 58.46%
macro F1, 43.91% minimum class recall, and 71.23% frozen online coverage. Its
accuracy is 3.18 percentage points below the original ten-step GRU, within the
predefined five-point limit. The eight-step checkpoint is accepted for one
controlled PPO experiment.

Production eight-step inference with 14-slot history retention is now
implemented and preserves the same five intent outputs and 120-value policy
observation. The controlled PPO + intent V2 training run completed at 200,704
steps and passed its checkpoint audit. The final policy SHA-256 is
`4fc855e064c366b0d0b94b807066b235a46f8cf3c7bfb06edceb6a56fa9b1773`.

The paired M10 holdout is complete. On seeds 20042–20541, V3 recorded 60.8%
success and 39.4% collision; PPO + intent V2 recorded 61.8% success and 38.2%
collision. The changes were only +1.0 and -1.2 percentage points, and the exact
paired McNemar test was not significant (`p = 0.6029`). Intent V2 failed three
of the five precommitted gates and is rejected as an improvement. V3 remains
the current best accepted driving policy.

One V3 terminal row (seed 20122) reported both arrival and collision. The raw
result is preserved in full, and a collision-first outcome rule now prevents
future evaluations from counting a crashed arrival as a success. See
`MILESTONES.md` for the raw and collision-first sensitivity analyses. Do not
rerun M10 or compare later evaluations that use the revised outcome rule
directly against the earlier summaries.

M11 tested V3's automatically selected 120K validation checkpoint against the
frozen 200K final checkpoint. The 120K policy was worse: 57.6% success and 42.4%
collision versus 60.8% and 39.2%. It produced 23 paired rescues but 39 paired
regressions (`p = 0.0559`, unfavorable), so it is rejected. Validation reward
over only 20 episodes was not a reliable success-selection rule. The 200K V3
checkpoint remains current best. The next safety direction is a
non-interventional conflict diagnostic followed by selectively targeted
caution, not another global waiting penalty or blanket TTC shield.

The M12A non-interventional diagnostic completed and exactly reproduced V3 on
the already-consumed seeds 10042–10541. The mechanically selected mild rule
vetoes only `FASTER` when constant-velocity closest approach is at most 2.0 s
with miss distance at most 3.0 m. In the final two seconds it appeared in
159/202 collision episodes and only 1/298 success episodes. This is association,
not a prevented-collision claim. M12B will test that one-step `FASTER`-to-`IDLE`
veto against an independently rerun V3 baseline on untouched paired seeds;
the stronger emergency-braking rule remains deferred.

The current 2.0-second TTC shield was rejected as too conservative. Do not combine it with
the intent-aware policy until a new safety experiment is explicitly designed and recorded.

The separate M12B mild CPA shield was also rejected. Its selective
`FASTER`-to-`IDLE` veto produced 58.6% success and 41.4% collision versus the
paired V3 baseline's 58.8% and 41.2%. It avoided incomplete episodes but changed
only one paired terminal outcome despite 1,489 recorded vetoes. V3 therefore
remains current best. The next development step must test causal action potency
on consumed seeds before spending another untouched holdout.

The development-only M13 selective-braking screen also failed. Changing
matching `FASTER` proposals to `SLOWER` reduced success to 52.2% and increased
collision to 47.8% on the consumed paired seeds. All tested hand-written
shields are rejected; higher mean minimum TTC did not imply fewer collisions.
Future work should improve the learned base policy rather than tune another CPA
or TTC override on consumed outcomes.

M14 adds physics-informed feature-level sensor fusion for the next learned
policy. It preserves all 105 normalized kinematic values and appends radar-like
range/range-rate plus radial-TTC and closest-approach features for the 14
traffic slots, producing a 175-value observation. These are deterministic
features from HighwayEnv state, not real camera/LiDAR measurements. The first
controlled training keeps V3 reward, traffic, actions, PPO hyperparameters,
seed, and 200K budget fixed so only the observation changes.

Fusion V1 produced the first strong development improvement: 63.0% success and
34.8% collision versus V3's paired 58.8% and 41.2%, with paired `p=0.00380`.
However, 11/500 episodes were incomplete (2.2%), one episode above the frozen
2.0% ceiling, so Fusion V1 is rejected as-is. Fusion V2 keeps the same 175
features and reward but trains longer across four deterministic environment
seed streams to test whether greater experience removes the observed stalls.

Fusion V2 training completed at 500,736 steps with checkpoint SHA-256
`7253cf4a...38b9`. The four-stream checkpoint passed all structural and
finite-parameter checks. Its development evaluation must now preserve Fusion
V1's 63.0% success / 34.8% collision while reducing incomplete episodes to at
most 2.0%; it is not accepted before that comparison.

Fusion V2 subsequently failed development: 58.2% success, 41.8% collision,
and 0% incomplete on the same 500 seeds. It failed three of five frozen gates
and is rejected. Of Fusion V1's 11 incomplete episodes, only two became
successes and nine became collisions. V3 remains the best accepted policy;
the next work is a frozen diagnostic of action timing in these regressions
before another training experiment. See MILESTONES.md section 66.

The M15A action-timing diagnostic is now implemented. It replays 89 frozen
paired cases (178 episodes), checks each against its original CSV row, and
records action, target-speed, motion, lane, and conflict traces. Run the
hash-bound command in MILESTONES.md section 67 after the test gate; commit its
single JSON report, including a failed report if reproduction fails.

M15A completed with all 178 episode replays and 89 observation-prefix checks
passing. All eleven V1 incomplete cases ended with IDLE maintaining a zero
speed target; they spent an average 25.53 seconds at low speed. In V2's 51
added-collision cases, 157 of 202 CPA-flagged terminal-window decisions were
IDLE with a positive target. Current observations omit that controller target.
The next proposed experiment is to expose it explicitly to PPO; its benefit
is untested. V3 remains the best accepted policy. See MILESTONES.md section 68.

The optional target-speed input and a separate collision-first reward
configuration are now implemented. Legacy V3 settings remain unchanged.
The next matched experiment trains a 175-input corrected-reward control first,
then a 176-input candidate with the same reward, seed, and 200K budget. Run
only the control training command in MILESTONES.md section 69 after tests pass;
commit its training JSON and keep the checkpoint local. Neither policy has
been trained for the research comparison yet.

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
python scripts/collect_intent_data.py --episodes 300 --history-length 10 \
  --sample-stride 2 --seed 42 --output data/intent_trajectories_seed42.npz \
  --summary-output results/intent_dataset_seed42.summary.json
```

2. Train the GRU:

```bash
python scripts/train_intent.py --data data/intent_trajectories_seed42.npz \
  --output models/intent_gru_seed42.pt --epochs 30 --batch-size 128 \
  --learning-rate 0.001 --seed 42
```

3. Inspect test-set performance:

```bash
python scripts/evaluate_intent.py --data data/intent_trajectories_seed42.npz \
  --model models/intent_gru_seed42.pt \
  --output results/intent_gru_seed42.metrics.json
```

The dataset stores a short history of `[relative_x, relative_y, relative_vx, relative_vy, acceleration, distance]`, the hidden behavior label used by the simulator, and the source episode. Train/validation/test splitting is episode-based to prevent overlapping trajectory windows from leaking between splits.

The collection script seeds both the environment and random ego action sampler. The training
checkpoint stores the dataset SHA-256 fingerprint and exact split indices; evaluation refuses a
different dataset. The large `.npz` dataset and `.pt` checkpoint remain local and ignored by Git.
Commit only the small dataset-summary and held-out-metrics JSON files after they are verified.

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
