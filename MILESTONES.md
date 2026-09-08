# SafeIntent-RL Research Milestones and Engineering Log

**Project:** Intent-Aware and Safety-Constrained Reinforcement Learning for Interactive Autonomous Driving  
**Repository:** [kingofrichnight/RL--Autonomus-car](https://github.com/kingofrichnight/RL--Autonomus-car)  
**Environment:** Gymnasium + HighwayEnv `intersection-v2`  
**Primary algorithm:** Proximal Policy Optimization (PPO)  
**Document type:** Living technical record  
**Created:** 2026-09-03  
**Last updated:** 2026-09-04  

---

## 1. Purpose of this document

This file is the authoritative chronological record of the SafeIntent-RL project. It is intended to support meetings with the project supervisor, the final report, reproducibility, and later thesis or publication writing.

It records:

- the research question and frozen project scope;
- every important method selected and why it was selected;
- environment, observation, action, reward, and control definitions;
- mathematical formulas used by the learning and safety systems;
- experiments, fixed variables, changed variables, and random seeds;
- actual results, including unsuccessful results;
- implementation failures, diagnoses, and corrections;
- optimization decisions and the evidence used to make them;
- the current state of each milestone and the next planned work.

### Evidence rule

The following terms are used deliberately:

- **Planned:** agreed as future work but not yet implemented.
- **Implemented:** code exists, but a complete experiment may not have been performed.
- **Smoke-tested:** verified on a small run to detect implementation failures.
- **Evaluated:** measured with a stated protocol and saved metrics.
- **Final result:** evaluated with multiple training seeds and a sufficiently large test set.

Smoke-test numbers must not be presented as final research results.

---

## 2. Research definition

### 2.1 Working title

**SafeIntent-RL: Intent-Aware and Safety-Constrained Reinforcement Learning for Interactive Autonomous Driving**

### 2.2 Central research question

> Can an autonomous vehicle use learned driver-intent information and a risk-based safety mechanism to make safer and more efficient decisions than standard PPO in uncertain intersection traffic?

### 2.3 Main hypothesis

Adding probabilistic estimates of surrounding-driver behavior to the PPO observation should improve decision quality under mixed traffic. Adding a TTC-based safety shield should further reduce collisions and unsafe interactions, although excessive intervention may reduce efficiency or show that the underlying policy has not learned safe behavior.

### 2.4 Required method comparison

| Method ID | Method | Purpose | Current status |
|---|---|---|---|
| B0 | Rule-based controller | Non-learning reference | Implemented; full evaluation pending |
| B1 | Standard PPO | Learning baseline | Smoke-trained and evaluated |
| B2 | PPO + intent | Test the value of intent estimates | Implemented framework; training pending |
| B3 | PPO + safety shield | Isolate safety-shield effect | Implemented framework; training pending |
| P1 | SafeIntent-PPO | PPO + intent + safety | Implemented framework; training pending |

### 2.5 Frozen master's-level scope

The initial project includes:

- unsignalized intersection decision-making;
- one learning ego vehicle and simulated non-learning traffic;
- PPO with discrete longitudinal control;
- hidden cautious, normal, and aggressive traffic behavior;
- trajectory-history collection;
- GRU-based probabilistic intent classification;
- TTC-based risk measurement and action shielding;
- reproducible evaluation, ablations, plots, and videos.

The following are deliberately postponed to later research extensions:

- CARLA validation;
- camera, LiDAR, or YOLO perception;
- world models;
- graph neural networks;
- multi-agent PPO;
- V2X communication;
- adversarial reinforcement learning;
- sim-to-real transfer;
- end-to-end control from images.

**Reason for limiting scope:** the master's project must answer one research question rigorously rather than include many advanced components without sufficient evaluation.

---

## 3. System architecture

```text
HighwayEnv intersection
        |
        v
Kinematic observations + short vehicle histories
        |                         |
        |                         v
        |                  GRU intent predictor
        |                         |
        +----------+--------------+
                   v
          Augmented observation
                   |
                   v
              PPO policy
                   |
                   v
            Proposed action
                   |
                   v
        TTC-based safety shield
                   |
                   v
            Executed action
                   |
                   v
              Environment
```

The driver profile is available internally only as a supervised-learning label. It is not directly supplied to PPO. The ego agent must infer behavior from observable motion.

---

## 4. Current milestone summary

| Milestone | Description | Status | Evidence |
|---:|---|---|---|
| M0 | Define research question and 3–4 month scope | Completed | Scope and comparison frozen |
| M1 | Create repository and reproducible Python setup | Completed | Public GitHub repository; Python 3.12 environment |
| M2 | Run and inspect `intersection-v2` | Completed | Three rendered random-agent episodes |
| M3 | Implement configurable environment and driver profiles | Implemented and smoke-tested | Environment completed episodes successfully |
| M4 | Train first standard PPO smoke model | Completed | `ppo_smoke.zip`, 10,240 collected timesteps |
| M5 | Correct evaluation and create live PPO viewer | Completed | 9 tests; 20-episode corrected evaluation |
| M6 | Train publication-quality PPO baseline | Seed 42 evaluated; multi-seed study pending | 200,704-step checkpoint; 500 evaluation episodes |
| M6A | Evaluate rule-based reference and diagnose PPO V1 | Evaluated | 500 episodes over seeds 42–541 |
| M6B | Design and test PPO Baseline V2 reward | Evaluated; rejected as improvement | Success 54.0%, collision 42.6%; predefined gate failed |
| M6C | Add progress/stall shaping for PPO V3 | Evaluated; screening gate passed narrowly | Success 56.6%, collision 43.4%, no incomplete episodes; paired gain not significant |
| M6D | Add TTC-risk shaping for PPO V4 | Implemented; holdout evaluation pending | Preserve V3 completion while reducing collision risk on untouched seeds |
| M7 | Collect intent dataset and train GRU | Planned after M6C reward experiment | Data and training scripts available |
| M8 | Train PPO + intent | Planned | Requires final GRU checkpoint |
| M9 | Evaluate TTC shield and PPO + safety | Planned | Safety code implemented |
| M10 | Train and evaluate SafeIntent-PPO | Planned | Requires M8 and M9 |
| M11 | Ablations and generalization tests | Planned | Protocol defined below |
| M12 | Final plots, video, report, and presentation | Planned | Depends on final experimental table |

---

## 5. Milestone M0 — Project formulation

**Status:** Completed  
**Date:** 2026-09-01

The project was reduced from a broad PhD-scale architecture to a focused master's contribution:

```text
PPO baseline + driver-intent prediction + safety mechanism
```

### Decision

HighwayEnv is used for learning and controlled experimentation. CARLA is not used in the initial phase.

### Reason

HighwayEnv supports fast interaction, configurable traffic, reproducible seeds, and many training steps without the computational cost of high-fidelity rendering. CARLA can later be used for validation after the learning method is established.

### Selected scenario

`intersection-v2` was selected because interactive right-of-way and gap-acceptance decisions expose the difference between efficient and unsafe behavior more clearly than simple lane keeping.

---

## 6. Milestone M1 — Repository and reproducible setup

**Status:** Completed  
**Date:** 2026-09-01

### Repository

The public repository `kingofrichnight/RL--Autonomus-car` was created and initialized. The first complete code upload added 37 source, configuration, test, and documentation files.

### Python version decision

An initial virtual environment was accidentally created with Python 3.14.6. Installation stopped because the project declares:

```text
Python >= 3.11 and < 3.13
```

Python 3.12 was selected and a new virtual environment was created.

### Reason for the change

Python 3.12 provides compatible builds for the selected Gymnasium, HighwayEnv, PyTorch, and Stable-Baselines3 stack. The environment was recreated rather than weakening the version constraint without dependency evidence.

### Reproducible Windows setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 7. Milestone M2 — Environment definition and random baseline

**Status:** Completed  
**Date:** 2026-09-01

### 7.1 Environment configuration

| Parameter | Selected value | Meaning |
|---|---:|---|
| Environment | `intersection-v2` | Connected-lane intersection scenario |
| Episode duration | 30 s | Maximum simulated episode time |
| Simulation frequency | 15 Hz | Vehicle dynamics updates per second |
| Policy frequency | 5 Hz | Agent decisions per second |
| Simulation steps/action | 3 | `15 / 5` dynamics steps per policy decision |
| Initial vehicle count | 10 | Initial traffic population |
| Spawn probability | 0.6 | Probability used for traffic spawning |
| Reward normalization | `false` | Preserve configured reward scale |

At 5 Hz, one policy action is selected every:

$$
\Delta t_{policy}=\frac{1}{5}=0.2\ \text{s}
$$

### 7.2 Observation space

The kinematic observation is:

$$
O_t\in\mathbb{R}^{15\times7}
$$

Each row describes the ego vehicle or another observed vehicle:

$$
o_t^{(i)}=
[p,\ x,\ y,\ v_x,\ v_y,\ \cos(\psi),\ \sin(\psi)]
$$

where:

- $p$ is the vehicle-presence indicator;
- $(x,y)$ is position;
- $(v_x,v_y)$ is velocity;
- $\psi$ is heading.

The selected settings use relative, normalized, sorted observations:

```yaml
absolute: false
normalize: true
order: sorted
```

The standard PPO multilayer perceptron receives the flattened 105-dimensional vector:

$$
15\times7=105
$$

When intent probabilities for five neighboring vehicles are appended, the planned augmented input is:

$$
105+(5\times3)=120
$$

### 7.3 Action space

The environment uses a three-action discrete longitudinal controller:

| Action index | Symbolic action | Effect |
|---:|---|---|
| 0 | `SLOWER` | Move toward a lower target-speed index |
| 1 | `IDLE` | Maintain the current high-level target |
| 2 | `FASTER` | Move toward a higher target-speed index |

The configured target speeds are:

$$
V_{target}=\{0,\ 4.5,\ 9.0\}\ \text{m/s}
$$

Lateral control is disabled. The vehicle follows its planned route while PPO makes high-level longitudinal decisions.

### 7.4 Implemented environment reward

Define the clipped speed term:

$$
r_{speed}=\operatorname{clip}\left(\frac{v-7}{9-7},0,1\right)
$$

Before arrival, the current configured reward is:

$$
R_t=I_{road}\left[-1.0I_{collision}+0.2r_{speed}\right]
$$

where $I_{road}=1$ when the vehicle remains on the road. When the vehicle satisfies HighwayEnv's arrival condition, the reward is replaced by:

$$
R_t=1.0
$$

Current weights:

| Component | Weight |
|---|---:|
| Collision | -1.0 |
| High speed | +0.2 |
| Arrival | +1.0 |

**Important:** this is the initial baseline reward. Risk, unsafe TTC, jerk, and unnecessary waiting are planned experimental additions; they are not yet part of the baseline reward.

### 7.5 Random-agent validation

Command:

```powershell
python scripts/random_agent.py --episodes 3 --render
```

Observed episode returns:

| Episode | Total reward |
|---:|---:|
| 1 | 2.775 |
| 2 | 0.921 |
| 3 | 2.719 |

### Interpretation

This test confirmed environment reset, stepping, reward production, termination, traffic motion, and graphical rendering. These rewards are not a learned-policy baseline because the actions were random.

---

## 8. Milestone M3 — Hidden driver-behavior model

**Status:** Implemented and smoke-tested  
**Full experimental validation:** Pending

Each non-ego vehicle is assigned a hidden behavioral profile. The PPO observation does not contain the profile label.

| Property | Cautious | Normal | Aggressive |
|---|---:|---:|---:|
| Sampling probability | 0.30 | 0.45 | 0.25 |
| Desired-speed scale | 0.80 | 1.00 | 1.20 |
| Acceleration scale | 0.75 | 1.00 | 1.30 |
| Following-distance scale | 1.35 | 1.00 | 0.70 |
| Time-gap scale | 1.30 | 1.00 | 0.70 |
| Yield-probability label | 0.85 | 0.55 | 0.20 |

If $q$ is a nominal vehicle parameter and $k_c$ is the profile scale, the profile-specific parameter is:

$$
q_{profile}=k_cq_{nominal}
$$

### Implementation correction

The first implementation assigned profiles only to vehicles present at reset. HighwayEnv can spawn new vehicles during an episode, so those vehicles would have remained unlabeled. The wrapper was changed to assign a profile to every newly observed, unlabeled NPC after each environment step.

### Research significance

Because the true class is hidden from the controller, the problem becomes partially observable. The intent model must estimate latent behavior from motion history instead of receiving an unrealistic ground-truth driver-type field.

---

## 9. Milestone M4 — PPO baseline and optimization method

**Status:** Smoke training completed  
**Date:** 2026-09-01  
**Checkpoint:** `models/ppo_smoke.zip`

### 9.1 Policy architecture

The baseline uses Stable-Baselines3 PPO with an `MlpPolicy`:

```text
105-dimensional flattened observation
              ->
       hidden layer: 256
              ->
       hidden layer: 256
              ->
 policy logits and value estimate
```

### 9.2 Discounted return

The discounted return is:

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kr_{t+k}
$$

with discount factor:

$$
\gamma=0.99
$$

### 9.3 Temporal-difference residual and GAE

The one-step temporal-difference residual is:

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

Generalized Advantage Estimation is:

$$
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}
$$

with:

$$
\lambda=0.95
$$

### 9.4 PPO probability ratio

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)}
$$

### 9.5 Clipped PPO objective

$$
L^{CLIP}(\theta)=
\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

with:

$$
\epsilon=0.2
$$

Clipping limits excessively large policy updates, improving training stability.

### 9.6 Combined optimization loss

Stable-Baselines3 minimizes:

$$
L_{total}=L_{policy}+c_vL_{value}+c_eL_{entropy}
$$

where the implementation represents entropy loss with a negative sign. Current coefficients are:

| Coefficient | Value |
|---|---:|
| Value coefficient $c_v$ | 0.5 |
| Entropy coefficient $c_e$ | 0.01 |
| Maximum gradient norm | 0.5 |

The entropy term encourages exploration; gradient clipping reduces the risk of unstable updates.

### 9.7 Baseline hyperparameters

| Hyperparameter | Value |
|---|---:|
| Requested timesteps | 10,000 |
| Actual collected timesteps | 10,240 |
| Rollout length `n_steps` | 1,024 |
| Batch size | 64 |
| Optimization epochs/rollout | 10 |
| Learning rate | $3\times10^{-4}$ |
| Discount factor $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| PPO clip range $\epsilon$ | 0.2 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Network | `[256, 256]` |
| Training seed | 42 |

The actual count is 10,240 because PPO completes whole 1,024-step rollout buffers:

$$
\left\lceil\frac{10000}{1024}\right\rceil\times1024=10240
$$

### 9.8 Training snapshot at 8,192 timesteps

| Logged quantity | Value |
|---|---:|
| Mean episode length | 38.5 decisions |
| Mean episode reward | 5.48 |
| Approximate KL | 0.01238 |
| Clip fraction | 0.153 |
| Entropy loss | -0.814 |
| Explained variance | 0.228 |
| Policy-gradient loss | -0.0181 |
| Value loss | 1.13 |
| Training speed | 37 FPS |

### Interpretation

The model was learning without numerical failure. The positive but low explained variance indicates that the value function had begun learning but was still inaccurate. Ten thousand steps is intentionally too short for a final autonomous-driving policy.

---

## 10. Milestone M5 — Evaluation protocol and corrected results

**Status:** Completed for smoke model  
**Final multi-seed evaluation:** Pending

### 10.1 Evaluation protocol

| Setting | Value |
|---|---:|
| Checkpoint | `ppo_smoke.zip` |
| Evaluation episodes | 20 |
| First environment seed | 42 |
| Episode seeds | 42–61 |
| Action selection | Deterministic |
| Safety shield | Disabled |

### 10.2 Metric definitions

Success rate:

$$
Success\ Rate=\frac{N_{arrived}}{N_{episodes}}
$$

Collision rate:

$$
Collision\ Rate=\frac{N_{collision}}{N_{episodes}}
$$

Mean travel time:

$$
\bar T=\frac{1}{N}\sum_{i=1}^{N}\frac{steps_i}{f_{policy}}
$$

Mean episode reward:

$$
\bar R=\frac{1}{N}\sum_{i=1}^{N}\sum_t r_t^{(i)}
$$

### 10.3 Corrected smoke-model results

| Metric | Result |
|---|---:|
| Episodes | 20 |
| Mean reward | 6.6722 |
| Mean episode length | 33.95 decisions |
| Mean travel time | 6.79 s |
| Success rate | 50.0% |
| Collision rate | 50.0% |
| Mean minimum finite TTC | 0.5717 s |
| Mean unsafe-TTC events/episode | 16.2 |
| Safety interventions | 0 |

### Interpretation

The smoke-trained policy learned enough to arrive in half of the evaluation episodes, but a 50% collision rate and a mean minimum TTC below one second are unacceptable for the final project. The result validates the pipeline; it does not validate safety.

### Evaluation bug and correction

**Observed problem:** the first evaluation script reported a 0% success rate.

**Cause:** HighwayEnv's intersection environment does not always return an `arrived` or `is_success` key in the final `info` dictionary.

**Correction:** success detection now uses the explicit info field when available and otherwise calls:

```python
env.unwrapped.has_arrived(env.unwrapped.vehicle)
```

**Verified outcome:** corrected evaluation returned a 50% success rate over the same 20-episode protocol. Two regression tests were added, increasing the unit-test total from 7 to 9.

---

## 11. TTC safety formulation

**Implementation status:** Implemented  
**Full experimental evaluation:** Pending

For ego vehicle $e$ and another vehicle $o$, define relative position and velocity:

$$
\mathbf p_r=\mathbf p_o-\mathbf p_e
$$

$$
\mathbf v_r=\mathbf v_o-\mathbf v_e
$$

Distance and line-of-sight direction are:

$$
d=\lVert\mathbf p_r\rVert_2
$$

$$
\hat{\mathbf p}_r=\frac{\mathbf p_r}{d}
$$

The radial closing speed is:

$$
v_{closing}=-\mathbf v_r^T\hat{\mathbf p}_r
$$

The implemented TTC estimate is:

$$
TTC=
\begin{cases}
\frac{d}{v_{closing}}, & v_{closing}>0\\
\infty, & v_{closing}\le0
\end{cases}
$$

The minimum TTC is evaluated across nearby vehicles within 60 m:

$$
TTC_{min}=\min_i TTC(e,i)
$$

### Initial engineering threshold

$$
TTC_{critical}=2.0\ \text{s}
$$

This is an experimental engineering threshold, not a universal real-world safety claim.

### Current safety-shield rule

```text
IF minimum TTC <= 2.0 s
AND PPO proposes FASTER or IDLE
THEN execute SLOWER
ELSE execute PPO action
```

Intervention rate is:

$$
Intervention\ Rate=
\frac{N_{overrides}}{N_{decisions}}
$$

A low collision rate with an extremely high intervention rate would show that the shield, rather than the PPO policy, is providing most of the safety. Both metrics must therefore be reported together.

### Limitation of current TTC

The current TTC uses radial closing speed and constant current velocity. It does not yet model future steering, acceleration uncertainty, lane-conflict geometry, or multi-modal trajectories. This limitation must be considered when interpreting shield performance.

---

## 12. Intent-prediction formulation

**Framework status:** Implemented and smoke-tested  
**Full dataset/model result:** Pending

### 12.1 Per-vehicle history

For each neighboring vehicle, the default history contains 10 observations:

$$
H_i=[x_{t-9}^{(i)},\ldots,x_t^{(i)}]
$$

Each history element is:

$$
x_t^{(i)}=
[\Delta x,\Delta y,\Delta v_x,\Delta v_y,\Delta v,d]
$$

where:

- $(\Delta x,\Delta y)$ is relative position to the ego;
- $(\Delta v_x,\Delta v_y)$ is relative velocity;
- $\Delta v=\lVert\mathbf v_t-\mathbf v_{t-1}\rVert_2$ is the implemented velocity-change magnitude;
- $d$ is Euclidean distance.

**Implementation note:** $\Delta v$ is currently a velocity-change proxy per observation, not a physical acceleration in $\text{m/s}^2$, because it is not divided by elapsed time. This may be changed during feature-engineering experiments, but any change must be recorded.

### 12.2 Standardization

Training features are standardized using training-set statistics only:

$$
\tilde x=\frac{x-\mu_{train}}{\sigma_{train}+10^{-6}}
$$

### 12.3 GRU model

The model uses a 64-unit GRU. A standard GRU update can be represented as:

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z)
$$

$$
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r)
$$

$$
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})+b_h)
$$

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t
$$

The final hidden state passes through:

```text
LayerNorm(64) -> Linear(64,32) -> ReLU -> Linear(32,3)
```

The output probabilities are:

$$
P(c\mid H_i)=\operatorname{softmax}(g(h_t))
$$

for classes:

```text
[cautious, normal, aggressive]
```

Probabilities are retained instead of using only the argmax class so PPO can receive information about prediction uncertainty.

### 12.4 Data-splitting correction

Overlapping windows from one episode are highly correlated. A random sample-level split could place nearly identical histories in training and testing, inflating reported accuracy.

The pipeline therefore uses episode-grouped splitting:

| Split | Approximate proportion |
|---|---:|
| Training episodes | 70% |
| Validation episodes | 15% |
| Test episodes | 15% |

No episode should appear in more than one split.

### 12.5 Intent metrics

For each class:

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

$$
F1=2\frac{Precision\times Recall}{Precision+Recall}
$$

The final report will include accuracy, macro precision, macro recall, macro F1, and a confusion matrix.

---

## 13. Viewer/control conflict and correction

**Status:** Corrected  
**Date:** 2026-09-03

### Observed behavior

The trained-policy viewer worked normally until an arrow key was pressed. Arrow-key input could modify the controlled vehicle or stop the program.

### Root cause

HighwayEnv's renderer forwards keyboard events to the action controller even while `watch_policy.py` is supplying PPO actions. In the current configuration:

- Right requests `FASTER`;
- Left requests `SLOWER`;
- Up requests a lane-left action;
- lateral actions are disabled.

The keyboard and PPO were therefore competing for control. The Up key could request an unavailable action.

### Correction

`watch_policy.py` now blocks Pygame `KEYDOWN` and `KEYUP` events after initializing the viewer. Window-close events remain available.

### Design decision

Autonomous evaluation and manual driving must remain separate operating modes:

- `watch_policy.py`: PPO controls the vehicle; arrow keys are disabled.
- A future manual-control tool, if required, must not run the PPO policy simultaneously.

This separation prevents human inputs from invalidating autonomous-policy evaluation.

---

## 14. Decision register

| ID | Decision | Alternatives considered | Reason | Status |
|---|---|---|---|---|
| D-001 | Use HighwayEnv first | CARLA from the beginning | Faster training and controlled experiments | Retained |
| D-002 | Use `intersection-v2` | Highway/lane-keeping only | Strong interactive decision scenario | Retained |
| D-003 | Use PPO baseline | Immediate complex model-based RL | Stable, well-supported baseline | Retained |
| D-004 | Use discrete longitudinal actions | Continuous steering/acceleration | Easier first decision problem and interpretation | Retained for baseline |
| D-005 | Hide driver labels from PPO | Give PPO ground-truth type | Preserve partial observability and realism | Retained |
| D-006 | Use three behavior classes | One homogeneous NPC type | Enables intent-learning question | Retained |
| D-007 | Use GRU first | LSTM or Transformer | Compact temporal baseline for short histories | Pending full comparison |
| D-008 | Keep intent probabilities | Use only predicted class | Preserve confidence/uncertainty signal | Retained |
| D-009 | Split intent data by episode | Random sample split | Prevent temporal-window leakage | Retained |
| D-010 | Begin with radial TTC shield | Collision penalty alone | Add interpretable risk constraint | Pending evaluation |
| D-011 | Use Python 3.12 | Python 3.14.6 | Dependency compatibility | Retained |
| D-012 | Disable keyboard in PPO viewer | Allow simultaneous manual input | Prevent control conflict and invalid evaluation | Retained |
| D-013 | Use environment arrival method | Depend only on `info` keys | Correct success measurement across versions | Retained |
| D-014 | Preserve an append-only research history | Replace old values with only the latest result | Maintain an auditable record for supervision and thesis writing | Retained |
| D-015 | Evaluate the rule-based controller before GRU work | Proceed directly to intent learning | PPO V1 success and collision rates require a fair non-learning reference and reward diagnosis | Completed and retained |
| D-016 | Develop PPO Reward V2 before GRU work | Add more PPO timesteps with the original reward | Both evaluated baselines expose a safety-efficiency tradeoff caused partly by reward alignment | Evaluated; V2 rejected |
| D-017 | Preserve V2 as a negative result and design V3 | Accept V2 because collision fell slightly | V2 failed the predefined success-and-collision gate and increased timeouts | Retained |
| D-018 | Add normalized route progress and a small time cost in V3 | Tune PPO hyperparameters or add intent immediately | V2 lacked dense goal feedback and produced 17 timeouts | Evaluated; retained as screening candidate |
| D-019 | Add a bounded TTC-risk penalty in V4 and use untouched holdout seeds | Continue reward tuning on seeds 42–541 | V3 eliminated timeouts but retained 43.4% collision and showed worse TTC-risk indicators | Active; evaluation pending |

---

## 15. Change log

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-01 | Created repository and uploaded research framework | Establish reproducible project structure | 37 project files published | `96dd29b` |
| 2026-09-01 | Added driver profiles for vehicles spawned after reset | Newly spawned vehicles initially lacked labels | Environment smoke test | Included in initial codebase |
| 2026-09-01 | Changed intent split to episode-grouped split | Prevent overlapping-window leakage | Intent pipeline smoke test | Included in initial codebase |
| 2026-09-01 | Trained PPO smoke model | Verify end-to-end RL training | 10,240 steps; checkpoint loaded | Local checkpoint |
| 2026-09-01 | Fixed success-rate detection | `info` omitted arrival field | Corrected 20-episode evaluation; 9 tests | `5e14d3f` |
| 2026-09-01 | Added trained-policy live viewer | Visually inspect PPO decisions | Viewer completed an arrival episode | `5e14d3f` |
| 2026-09-03 | Disabled keyboard input in autonomous viewer | Arrow keys conflicted with PPO | Syntax check and source-level diagnosis | `d988e7e` |
| 2026-09-03 | Added this living milestone record | Supervisor requested complete method/change record | Markdown review and repository link | `f1a396a` |
| 2026-09-03 | Adopted an append-only record policy | Preserve old and new evidence throughout the project | Policy added to the update procedure | This documentation update |
| 2026-09-03 | Completed and evaluated PPO baseline seed 42 | Establish the long-training B1 baseline | 200,704 training steps and 500 deterministic evaluation episodes | Training/results: `764a971`; documentation: this update |
| 2026-09-03 | Corrected the M6 summary filename | Initial record omitted the `.summary` portion | Verified exact file list in commit `764a971` | This documentation update |
| 2026-09-03 | Added rule-based evaluation script | Compare B0 and B1 over identical seeds and metrics | Syntax and TTC threshold decisions verified | `b9f95d6`, `dcffeb3` |
| 2026-09-03 | Revised the immediate research sequence | PPO V1 achieved only 55.8% success and 43.6% collision | Baseline evidence review documented in Section 25 | `bde2b69` plus this update |
| 2026-09-03 | Evaluated TTC rule-based reference over 500 episodes | Establish a non-learning comparison using identical seeds | Raw CSV and summary verified | Results: `97b6867`; documentation: this update |
| 2026-09-03 | Selected PPO Reward V2 as the next experiment | Neither PPO V1 nor the conservative TTC rule was satisfactory | Quantitative comparison documented in Section 26 | This documentation update |
| 2026-09-03 | Added isolated PPO Reward V2 configuration | Increase successful arrival while making collision-return negative | Coefficient-difference and reward-bound checks passed | `5f099ad`, `3395a81` |
| 2026-09-03 | Added `--config` to PPO training and evaluation | Ensure V2 uses the same reward during both phases | GitHub source verification | `74b9833`, `d7c6324` |
| 2026-09-03 | Documented V2 run commands | Make the controlled experiment reproducible | README reviewed | `cddfa85` |
| 2026-09-04 | Evaluated PPO Reward V2 over 500 episodes | Test whether reward rebalance improves completion and safety | Raw CSV, JSON, and paired-seed analysis verified | Results: `81b9fd7`; documentation: this update |
| 2026-09-04 | Rejected V2 as an improvement | Success fell to 54.0% despite collision falling to 42.6% | Predefined Section 27.4 gate applied | This documentation update |
| 2026-09-04 | Implemented PPO Reward V3 route-progress shaping | Provide dense goal feedback and penalize waiting while retaining V2 collision/arrival priorities | Source, isolated YAML, integration, and two unit tests added; syntax checks passed | `5303b63`, `9860c91`, `cd74532`, `1562432`, `2f8411b` |
| 2026-09-04 | Documented reproducible V3 commands | Ensure training and evaluation use the identical V3 configuration | README command review | `ca59c64` |
| 2026-09-04 | Evaluated PPO Reward V3 over 500 episodes | Test whether progress/time shaping restores completion | Raw CSV/JSON, acceptance gate, paired transitions, and exact McNemar test verified | Results: `31b3af1`; documentation: this update |
| 2026-09-04 | Implemented bounded TTC-risk shaping for V4 | Reduce collision-dominated failures while preserving V3 progress pressure | Syntax checks passed; dedicated risk test added; full pytest pending in project environment | `9c14fa1`, `e3133e9`, `a296317` |
| 2026-09-04 | Created untouched V3/V4 holdout protocol | Avoid further selection on evaluation seeds 42–541 | Seeds 10042–10541 and acceptance rules fixed before V4 training | `7f114fb` plus this update |

---

## 16. Planned optimization protocol

Hyperparameters must be changed through controlled experiments, not by changing many values simultaneously.

### 16.1 Initial PPO parameters to investigate

| Parameter | Baseline | Candidate values | Primary effect to monitor |
|---|---:|---|---|
| Learning rate | $3\times10^{-4}$ | $10^{-4}$, $3\times10^{-4}$, $5\times10^{-4}$ | Stability and convergence speed |
| Rollout steps | 1,024 | 512, 1,024, 2,048 | Advantage quality and update frequency |
| Batch size | 64 | 32, 64, 128 | Gradient noise and training speed |
| Entropy coefficient | 0.01 | 0, 0.005, 0.01 | Exploration versus convergence |
| Network | `[256,256]` | `[128,128]`, `[256,256]` | Capacity and computation |
| TTC threshold | 2.0 s | 1.5, 2.0, 2.5, 3.0 s | Safety-efficiency tradeoff |
| History length | 10 | 5, 10, 20 | Intent accuracy versus latency |

### 16.2 Controlled experiment requirements

For each experiment:

1. State one hypothesis.
2. Change one primary factor or define a justified factorial design.
3. Record all fixed variables.
4. Use the same training and evaluation seeds for paired comparisons.
5. Save model, configuration, logs, and raw episode results.
6. Report mean and standard deviation across training seeds.
7. Record unsuccessful or inconclusive outcomes.
8. State the decision taken from the evidence.

### 16.3 Final statistical protocol

Planned final training seeds:

```text
11, 22, 33, 44, 55
```

Planned evaluation size:

```text
500–1,000 episodes per trained method, subject to runtime
```

For metric $m$ across $n$ independently trained seeds:

$$
\bar m=\frac{1}{n}\sum_{i=1}^{n}m_i
$$

$$
s_m=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(m_i-\bar m)^2}
$$

Results will be reported as:

$$
\bar m\pm s_m
$$

---

## 17. Planned experiment matrix

| Experiment | Comparison | Question | Status |
|---:|---|---|---|
| E1 | Rule-based vs PPO | Does learned control improve performance? | Pending |
| E2 | PPO vs PPO + intent | Does explicit intent information help? | Pending |
| E3 | PPO vs PPO + safety | How much risk reduction comes from the shield? | Pending |
| E4 | All baselines vs SafeIntent-PPO | Does the complete method offer the best tradeoff? | Pending |
| E5 | Complete model minus intent | What is the contribution of intent? | Pending |
| E6 | Complete model minus safety | What is the contribution of the shield? | Pending |
| E7 | Intent prediction noise sweep | How accurate must intent prediction be to help PPO? | Pending |
| E8 | Aggressive-driver proportion shift | Does the policy generalize to unseen behavior mixtures? | Pending |
| E9 | TTC threshold sweep | How does conservatism affect safety and efficiency? | Pending |

---

## 18. Final evaluation metrics

### Safety

- collision rate;
- near-collision/unsafe-TTC event rate;
- mean and minimum TTC;
- emergency braking events;
- safety interventions and intervention rate.

### Task performance

- arrival/success rate;
- episode completion;
- travel time;
- average speed;
- waiting time.

### Comfort

- acceleration;
- deceleration;
- jerk.

Jerk is defined as:

$$
j(t)=\frac{da(t)}{dt}
$$

### Intent prediction

- accuracy;
- class-wise and macro precision;
- class-wise and macro recall;
- class-wise and macro F1;
- confusion matrix;
- confidence distribution.

### Learning behavior

- mean episodic return;
- return variance;
- convergence rate;
- sample efficiency;
- approximate KL divergence;
- entropy;
- explained variance.

---

## 19. Known limitations and open technical questions

1. The PPO smoke model has been trained for only 10,240 steps.
2. The current 20-episode evaluation is too small for final conclusions.
3. The baseline reward favors speed but has no explicit TTC or comfort penalty.
4. Driver profiles are simulator-defined behavioral categories, not labels learned from real driving data.
5. The intent feature called acceleration is currently a velocity-change magnitude.
6. The radial TTC approximation assumes constant instantaneous velocity.
7. The safety threshold of 2.0 s requires sensitivity analysis.
8. The safety shield has been implemented but not yet compared across multiple seeds.
9. Intent-PPO and complete SafeIntent-PPO require full training and evaluation.
10. Generalization to a changed aggressive-driver distribution remains untested.

---

## 20. Recorded M6 plan before execution

**Status:** Executed for seed 42 and documented in Section 24. The original plan is retained below as the pre-experiment record.

### M6 — Publication-quality PPO baseline

The next objective is to train standard PPO beyond the smoke-test stage before adding intent and safety components.

Proposed first baseline command:

```powershell
python scripts/train_ppo.py --timesteps 200000 --seed 42 --output models/ppo_baseline_seed42
```

After training:

```powershell
python scripts/evaluate_policy.py `
  --model models/ppo_baseline_seed42.zip `
  --episodes 500 `
  --seed 42 `
  --output results/ppo_baseline_seed42.csv
```

This first long run is still not the final multi-seed result. Its purpose is to select a reasonable training budget and determine whether learning has converged sufficiently for the five-seed study.

---

## 21. Template for every future experiment

Copy this section for each new run.

```markdown
### Experiment ID and title

- Date:
- Research question:
- Hypothesis:
- Git commit:
- Environment/config file:
- Method:
- Training seed(s):
- Evaluation seed(s):
- Changed variable:
- Baseline value:
- New value:
- Fixed variables:
- Training timesteps:
- Model output path:
- Raw-results path:

#### Result

| Metric | Baseline | New method | Change |
|---|---:|---:|---:|
| Success rate | | | |
| Collision rate | | | |
| Mean minimum TTC | | | |
| Travel time | | | |
| Intervention rate | | | |

#### Observation

Describe what happened without interpreting beyond the evidence.

#### Interpretation

Explain why the result may have occurred, including uncertainty and limitations.

#### Decision

State whether the change is retained, rejected, or requires another experiment.
```

---

## 22. Update procedure for this record

### Record-preservation policy

This document is an append-only research history. Previous experiments, results, decisions, failures, and observations must not be deleted merely because a newer method performs better or a correction is made.

- Every experiment receives a unique ID and date.
- New results are added alongside earlier results, not substituted without explanation.
- An outdated result is marked **Superseded**, **Rejected**, or **Corrected** and linked to its replacement.
- A correction records the original value, corrected value, cause, verification method, and relevant Git commit.
- Current technical sections may describe the latest implementation, but their earlier state remains recoverable through the dated change log and Git history.
- Failed and inconclusive experiments remain because they are part of the research evidence.

Whenever project code, equations, configurations, datasets, training procedures, or evaluation logic change:

1. update the relevant technical section;
2. add a dated entry to the change log;
3. add or update the related decision-register row;
4. record the test or experiment used for verification;
5. identify results as smoke, evaluated, or final;
6. link the Git commit and saved result path;
7. never silently replace an unsuccessful result.

This procedure will be followed for all future SafeIntent-RL work.

---

## 23. Technical references

1. J. Schulman et al., “Proximal Policy Optimization Algorithms,” 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
2. [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
3. [HighwayEnv intersection documentation](https://highway-env.farama.org/environments/intersection/)
4. [Gymnasium custom environment documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
5. K. Cho et al., “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” 2014. [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)



---

## 24. Milestone M6 — Long PPO baseline training and evaluation

**Experiment ID:** E-B1-S42-200K  
**Status:** Evaluated for one training seed; final multi-seed evaluation pending  
**Date:** 2026-09-03  
**Method:** Standard PPO without intent input or safety shield  
**Training seed:** 42  
**Raw-result commit:** [`764a971`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/764a97130806f990810b3005caa5192064c7d5dc)

### 24.1 Training artifact verification

The supplied checkpoint `ppo_baseline_seed42.zip` passed ZIP integrity checks and contained the expected Stable-Baselines3 policy, optimizer, variables, metadata, and system-information files.

| Item | Verified value |
|---|---:|
| Requested training timesteps | 200,000 |
| Actual collected timesteps | 200,704 |
| PPO updates | 1,960 |
| Seed | 42 |
| Rollout steps | 1,024 |
| Batch size | 64 |
| Epochs per rollout | 10 |
| Learning rate | 0.0003 |
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Network | [256, 256] |
| Checkpoint SHA-256 | `bf6cdb5d258c795aba37e3ab8a1b93fb1bfc6f88b5fca2b77b7808e6f3b34901` |

The checkpoint was produced using Python 3.12.9, Stable-Baselines3 2.9.0, PyTorch 2.13.0 CPU, Gymnasium 1.3.0, and NumPy 2.5.2.

The final checkpoint's rolling 100-episode training buffer reported:

| Training statistic | Value |
|---|---:|
| Mean reward | 6.7702 |
| Reward standard deviation | 3.0222 |
| Reward range | 1.1245–10.8000 |
| Mean episode length | 36.77 decisions |
| Episode-length range | 18–81 decisions |

These are on-policy training-buffer statistics, not deterministic test results.

### 24.2 Deterministic evaluation protocol

| Setting | Value |
|---|---:|
| Evaluation episodes | 500 |
| Environment seeds | 42–541 |
| Action selection | Deterministic |
| Intent model | Disabled |
| Safety shield | Disabled |
| Raw episode file | `results/ppo_baseline_seed42.csv` |
| Summary file | `results/ppo_baseline_seed42.summary.json` |

**Filename correction (2026-09-03):** the first documentation pass referred to `ppo_baseline_seed42.json`. The committed evaluator output is actually `ppo_baseline_seed42.summary.json`, as verified from commit `764a971`. The original naming error is recorded here for traceability.

### 24.3 Evaluation result

| Metric | Result |
|---|---:|
| Successful episodes | 279/500 |
| Collision episodes | 218/500 |
| Incomplete non-collision episodes | 3/500 |
| Mean reward | 6.9434 |
| Mean episode length | 37.35 decisions |
| Success rate | 55.8% |
| Collision rate | 43.6% |
| Mean travel time | 7.47 s |
| Mean minimum finite TTC | 0.6332 s |
| Mean unsafe-TTC events/episode | 15.142 |
| Mean safety interventions | 0 |

### 24.4 Comparison with the earlier smoke model

The earlier smoke evaluation is retained in Section 10. It used only 20 episodes, so its changes are descriptive rather than a statistically controlled conclusion.

| Metric | Smoke: 10,240 steps, 20 episodes | M6: 200,704 steps, 500 episodes | Observed change |
|---|---:|---:|---:|
| Mean reward | 6.6722 | 6.9434 | +0.2712 |
| Mean episode length | 33.95 | 37.35 | +3.40 |
| Success rate | 50.0% | 55.8% | +5.8 percentage points |
| Collision rate | 50.0% | 43.6% | -6.4 percentage points |
| Mean travel time | 6.79 s | 7.47 s | +0.68 s |
| Mean minimum TTC | 0.5717 s | 0.6332 s | +0.0615 s |
| Unsafe-TTC events/episode | 16.2 | 15.142 | -1.058 |
| Safety interventions | 0 | 0 | No change; shield disabled |

### 24.5 Interpretation and decision

Longer training produced a modest descriptive improvement in success rate, collision rate, mean reward, and TTC relative to the smoke run. However, a 43.6% collision rate remains unacceptable for a safety-oriented controller, and the mean minimum TTC remains well below the initial 2.0-second risk threshold.

**Decision:** retain this checkpoint as the standard-PPO seed-42 baseline. Do not describe it as safe. Proceed to the intent-data milestone while keeping multi-seed PPO training as a requirement for final statistical conclusions. Later experiments must compare intent and safety variants against this same evaluation protocol.

**Subsequent revision:** the instruction to proceed immediately to intent learning was superseded after reviewing the low success rate and dominant collision failure mode. Section 25 preserves the evidence and records the revised sequence.


---

## 25. Milestone M6A — Rule-based reference and PPO V1 diagnosis

**Status:** In progress  
**Date:** 2026-09-03  
**Trigger:** Review of the M6 PPO V1 success and collision rates

### 25.1 Evidence that triggered the revision

PPO V1 succeeded in 279 of 500 episodes, collided in 218, and ended incomplete without a collision in only 3. Therefore:

$$
\frac{218}{500-279}\times100=98.64\%
$$

of unsuccessful episodes were collision failures.

For the observed success proportion $\hat p=0.558$ over $n=500$ episodes, the approximate standard error is:

$$
SE=\sqrt{\frac{\hat p(1-\hat p)}{n}}\approx0.0222
$$

An approximate 95% interval is 51.4%–60.2%. Even the upper end is not a satisfactory final success level. The longer run improved success by only 5.8 percentage points and reduced collision by 6.4 percentage points relative to the small smoke evaluation, so simply increasing training steps may not address the underlying objective.

### 25.2 Reward-design concern

The current policy can receive up to approximately +0.2 speed reward at each decision while a collision contributes -1.0 when it occurs. Consequently, accumulated positive speed rewards can outweigh the terminal collision cost. The raw evaluation data confirms that many collision episodes still finish with positive total rewards.

This means the policy may be optimizing the configured reward correctly while the configured reward does not sufficiently represent the research objective.

### 25.3 Revised decision

Before training the GRU intent model:

1. preserve PPO Baseline V1 and its 500-episode result;
2. evaluate the existing TTC rule-based controller over the same seeds 42–541;
3. compare success, collision, travel time, TTC, and unsafe-event counts;
4. design PPO Baseline V2 using the evidence from both controllers;
5. keep TTC reward/shield effects separate from the standard-PPO baseline comparison.

This revision supersedes only the immediate sequence stated in Section 24.5. It does not delete or alter the earlier evidence.

### 25.4 Rule-based controller

The reference controller applies:

$$
a_t=
\begin{cases}
\text{SLOWER}, & TTC_{min}\le2.0\text{ s}\\
\text{FASTER}, & TTC_{min}\ge4.0\text{ s}\\
\text{IDLE}, & \text{otherwise}
\end{cases}
$$

The new `scripts/evaluate_rule_based.py` uses the same environment seeding and episode metrics as `evaluate_policy.py`. Its JSON summary also records the threshold settings and action counts.

Verification completed before publication:

- Python syntax compilation passed;
- TTC = 2.0 s selected `SLOWER`;
- TTC = 3.0 s selected `IDLE`;
- TTC = 4.0 s selected `FASTER`;
- three unit tests were added for these boundary decisions;
- the full test runner was unavailable in the temporary maintenance workspace, so the project test suite must be run in the configured Python 3.12 environment.

### 25.5 Next command

After pulling the latest repository changes, run:

```powershell
python scripts/evaluate_rule_based.py --episodes 500 --seed 42 --output results/rule_based_seed42.csv
```

Expected outputs:

```text
results/rule_based_seed42.csv
results/rule_based_seed42.summary.json
```

**Execution status:** completed. Results are recorded in Section 26; the original command remains above as the reproducibility record.


---

## 26. Milestone M6A result — Rule-based versus PPO V1

**Experiment ID:** E-B0-S42-500E  
**Status:** Evaluated for 500 episodes  
**Date:** 2026-09-03  
**Environment seeds:** 42–541  
**Raw-result commit:** [`97b6867`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/97b68679f2a5586491e5fb77c4c6567674390cc4)

### 26.1 Rule-based result

| Metric | Result |
|---|---:|
| Successful episodes | 201/500 |
| Collision episodes | 175/500 |
| Incomplete non-collision episodes | 124/500 |
| Mean reward | 3.8233 |
| Mean episode length | 101.432 decisions |
| Success rate | 40.2% |
| Collision rate | 35.0% |
| Mean travel time | 20.2864 s |
| Mean minimum finite TTC | 1.1357 s |
| Mean unsafe-TTC events/episode | 30.448 |
| Safety interventions | 0 |

The approximate 95% interval for rule-based success is 35.9%–44.5%; for collision rate it is 30.8%–39.2%.

### 26.2 Controller action distribution

The controller made 50,716 decisions:

| Action | Count | Share |
|---|---:|---:|
| `SLOWER` | 15,224 | 30.02% |
| `IDLE` | 30,130 | 59.41% |
| `FASTER` | 5,362 | 10.57% |

The high `IDLE` share and long episodes indicate conservative waiting behavior.

### 26.3 Direct comparison

Both methods were evaluated for 500 episodes over seeds 42–541.

| Metric | PPO V1 | TTC rule-based | Rule minus PPO |
|---|---:|---:|---:|
| Success rate | 55.8% | 40.2% | -15.6 percentage points |
| Collision rate | 43.6% | 35.0% | -8.6 percentage points |
| Mean reward | 6.9434 | 3.8233 | -3.1201 |
| Mean travel time | 7.47 s | 20.2864 s | +12.8164 s |
| Mean minimum TTC | 0.6332 s | 1.1357 s | +0.5024 s |
| Unsafe events/episode | 15.142 | 30.448 | +15.306 |

The raw unsafe-event count is larger for the rule controller mainly because its episodes are 171.6% longer. Normalizing by episode decisions gives:

$$
Rate_{unsafe,PPO}=\frac{15.142}{37.35}=0.4054
$$

$$
Rate_{unsafe,rule}=\frac{30.448}{101.432}=0.3002
$$

Thus the rule controller has approximately 26.0% fewer unsafe-TTC decisions proportionally, despite more unsafe events per episode.

### 26.4 Interpretation

The TTC rule reduces collision rate and increases minimum TTC, but it does not solve the driving task efficiently. It produces many timeouts and lowers success by 15.6 percentage points. Of its 299 unsuccessful episodes, 175 collided and 124 ended without either collision or arrival.

PPO V1 arrives more often and much faster, but its collision rate is too high. The rule-based method is more conservative but frequently waits or reacts to radial closing risks without enough route-conflict context. Neither baseline is suitable as the proposed final controller.

### 26.5 Decision and next experiment

Both results are retained as reference baselines:

- **B0:** TTC rule-based — safer but overly conservative;
- **B1-V1:** standard PPO with original reward — more efficient but unsafe.

The next controlled experiment is **B1-V2**, a revised standard-PPO reward. It must not include intent probabilities or the TTC safety shield, because those remain separate experimental factors.

The proposed reward family is:

$$
R_t^{V2}=
w_a I_{arrival}
-w_c I_{collision}
+w_p\Delta progress
+w_v R_{speed}
-w_w I_{stall}
$$

The exact coefficients will be documented before training. The collision cost must dominate the maximum plausible accumulated speed reward. PPO V1 remains unchanged so V2 can be compared against it over the same seeds and 500-episode protocol.


---

## 27. Milestone M6B — PPO Reward V2 design

**Experiment ID:** E-B1-V2-S42-200K  
**Status:** Configuration implemented and verified; training pending  
**Date:** 2026-09-03  
**Primary objective:** increase successful arrivals while reducing collisions relative to PPO V1

### 27.1 Controlled-variable design

Only three environment reward coefficients change. PPO architecture, hyperparameters, observation space, action space, traffic configuration, seed, and training budget remain fixed.

| Reward coefficient | PPO V1 | PPO V2 | Reason |
|---|---:|---:|---|
| Collision | -1.0 | -10.0 | Make collision return strongly negative |
| Arrival | +1.0 | +5.0 | Directly strengthen the incentive to complete the route |
| High speed | +0.2/decision | +0.05/decision | Prevent repeated speed reward from dominating safety and arrival |

The implementation is stored in `configs/intersection_reward_v2.yaml`.

### 27.2 Implemented reward

Let:

$$
r_{speed}=\operatorname{clip}\left(\frac{v-7}{9-7},0,1\right)
$$

Before arrival:

$$
R_t^{V2}=I_{road}\left[-10I_{collision}+0.05r_{speed}\right]
$$

When HighwayEnv reports arrival, the terminal reward is replaced by:

$$
R_t^{V2}=+5
$$

At the nominal maximum of 150 policy decisions in 30 seconds, the largest possible accumulated speed component is:

$$
30\times5\times0.05=7.5
$$

Because:

$$
|R_{collision}|=10>7.5
$$

a collision cannot be made attractive solely by accumulating the maximum nominal speed reward.

### 27.3 Why this may increase success

The original reward paid repeated speed bonuses but only a small arrival bonus. V2 raises the explicit arrival outcome by five times while reducing the incentive for fast, risky motion. This gives PPO a clearer distinction among:

- arrive safely: strongly positive;
- wait for the whole episode: approximately zero;
- collide: negative.

A possible failure mode is excessive waiting because a zero-reward timeout may appear safer than exploration. That outcome will be measured rather than hidden.

### 27.4 Predefined decision rule

V2 is retained as an improvement only if the 500-episode seed-42 screening run satisfies both:

$$
Success_{V2}>55.8\%
$$

and:

$$
Collision_{V2}<43.6\%
$$

Reward magnitude alone is not an acceptance metric because V1 and V2 use different reward scales. Incomplete/time-out rate and travel time will also be reported. If collision falls but success does not rise, V2 is classified as overly conservative and the next controlled version will add progress or stall/time shaping.

### 27.5 Verification

- The V2 YAML differs from V1 only in the three documented coefficients.
- The nominal maximum accumulated speed reward is 7.5.
- The collision magnitude is 10.0 and therefore exceeds that bound.
- Python syntax checks passed in the maintenance workspace.
- Dedicated configuration tests were added.
- Training and evaluation both accept the same explicit `--config` path.
- The full test suite must be run in the configured project environment before training.

### 27.6 Reproducible commands

Pull and test:

```powershell
git pull origin main
python -m pytest
```

Train:

```powershell
python scripts/train_ppo.py --config configs/intersection_reward_v2.yaml --timesteps 200000 --seed 42 --output models/ppo_reward_v2_seed42
```

Evaluate after training:

```powershell
python scripts/evaluate_policy.py --model models/ppo_reward_v2_seed42.zip --config configs/intersection_reward_v2.yaml --episodes 500 --seed 42 --output results/ppo_reward_v2_seed42.csv
```

Expected evaluation outputs:

```text
results/ppo_reward_v2_seed42.csv
results/ppo_reward_v2_seed42.summary.json
```

**Execution status:** completed; outcome recorded in Section 28.


---

## 28. Milestone M6B result — PPO Reward V2

**Experiment ID:** E-B1-V2-S42-200K  
**Status:** Evaluated and rejected as an improvement  
**Date recorded:** 2026-09-04  
**Training seed:** 42  
**Evaluation seeds:** 42–541  
**Raw-result commit:** [`81b9fd7`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/81b9fd77164fb9a2d44eaf03df2804ad36348589)

### 28.1 Evaluation result

| Metric | PPO V2 result |
|---|---:|
| Successful episodes | 270/500 |
| Collision episodes | 213/500 |
| Incomplete non-collision episodes | 17/500 |
| Mean reward | -0.0236 |
| Mean episode length | 47.672 decisions |
| Success rate | 54.0% |
| Collision rate | 42.6% |
| Mean travel time | 9.5344 s |
| Mean minimum finite TTC | 0.6549 s |
| Mean unsafe-TTC events/episode | 16.414 |
| Mean safety interventions | 0 |

Mean reward is not compared directly with V1 because the reward scale changed.

### 28.2 Predefined acceptance test

The decision rule was fixed before evaluation:

$$
Success_{V2}>55.8\%
$$

and:

$$
Collision_{V2}<43.6\%
$$

Observed:

$$
54.0\%\not>55.8\%
$$

$$
42.6\%<43.6\%
$$

V2 passed the collision condition but failed the success condition. It is therefore rejected as an overall improvement under the predefined rule.

### 28.3 Comparison with PPO V1

| Metric | PPO V1 | PPO V2 | V2 minus V1 |
|---|---:|---:|---:|
| Success rate | 55.8% | 54.0% | -1.8 percentage points |
| Collision rate | 43.6% | 42.6% | -1.0 percentage point |
| Incomplete rate | 0.6% | 3.4% | +2.8 percentage points |
| Mean travel time | 7.47 s | 9.5344 s | +2.0644 s |
| Mean minimum TTC | 0.6332 s | 0.6549 s | +0.0216 s |
| Unsafe events/episode | 15.142 | 16.414 | +1.272 |

V2 episodes were 27.6% longer. Normalized unsafe-TTC frequency was:

$$
Rate_{unsafe,V1}=\frac{15.142}{37.35}=0.4054
$$

$$
Rate_{unsafe,V2}=\frac{16.414}{47.672}=0.3443
$$

This is a 15.1% reduction in unsafe-TTC decisions proportionally, but it did not translate into higher success.

### 28.4 Paired-seed outcome transitions

Because both policies used the same 500 environment seeds, episode outcomes were paired directly.

| PPO V1 outcome | PPO V2 outcome | Episodes |
|---|---|---:|
| Success | Success | 249 |
| Success | Collision | 29 |
| Success | Incomplete | 1 |
| Collision | Success | 21 |
| Collision | Collision | 183 |
| Collision | Incomplete | 14 |
| Incomplete | Collision | 1 |
| Incomplete | Incomplete | 2 |

V2 converted 21 former collisions into successes, but 30 former successes were lost. Fourteen former collisions became timeouts. The net changes were nine fewer successes, five fewer collisions, and fourteen more incomplete episodes.

### 28.5 Outcome-specific V2 behavior

| V2 outcome | Episodes | Mean reward | Mean length | Mean minimum TTC |
|---|---:|---:|---:|---:|
| Success | 270 | 6.9711 | 51.25 | 0.6333 s |
| Collision | 213 | -8.9318 | 34.89 | 0.6378 s |
| Incomplete | 17 | 0.4976 | 151.00 | 1.2109 s |

The revised reward correctly made collision episodes strongly negative. However, some policies avoided collision by waiting until truncation, and overall successful completion did not improve.

### 28.6 Interpretation and next decision

V2 improved reward alignment: collisions became negative and the normalized unsafe-event rate decreased. It did not solve task completion. Raising the arrival reward is insufficient when the agent receives little dense feedback about approaching the goal, while a zero-reward or slightly positive timeout can remain preferable to risky exploration.

**Decision:** preserve V2 as a scientifically useful negative result. Do not replace V1 with V2. The next controlled experiment, PPO V3, will retain the strong collision and arrival terms while adding a dense route-progress signal and a small stall/time cost. The V3 coefficients and route-progress calculation must be fixed and documented before training.

---

## 29. Milestone M6C design — PPO Reward V3

**Experiment ID:** E-B1-V3-S42-200K  
**Status:** Evaluated; result recorded in Section 30  
**Date fixed:** 2026-09-04  
**Primary changed factor:** dense route-progress shaping plus a per-decision time cost  
**Configuration:** `configs/intersection_reward_v3.yaml`

### 29.1 Motivation from V2 evidence

PPO V2 reduced the proportion of unsafe-TTC decisions but lowered success from 55.8% to 54.0% and increased incomplete episodes from 0.6% to 3.4%. Its reward strongly distinguished arrival from collision, but it supplied little intermediate information about whether an action moved the ego vehicle toward successful route completion.

V3 targets that failure directly. It retains the V2 collision, arrival, and speed coefficients and adds:

1. normalized forward route-progress reward;
2. a small cost for every policy decision, including waiting.

No PPO hyperparameter, observation, action, traffic, training-budget, or seed change is introduced in this experiment.

### 29.2 Route-progress calculation

At reset, the wrapper captures the planned route and the length (L_k) of every route lane. The final exit lane is capped at the configured arrival distance:

$$
L_m^*=\min(L_m,25\ \text{m})
$$

For route-lane index (j) and longitudinal lane coordinate (s_t), absolute progress is:

$$
P_t=\sum_{k<j}L_k+\min(\max(s_t,0),L_j^*)
$$

where only the final lane uses the capped length. Let (P_0) be progress at reset and (P_g) the route goal. The non-negative normalized progress increment is:

$$
\Delta p_t=
\frac{\max(0,P_t-P_{t-1})}
{\max(P_g-P_0,1)}
$$

Backward motion cannot earn a negative progress term, and repeated forward distance cannot be rewarded twice because the wrapper stores the greatest achieved progress.

### 29.3 V3 reward

The fixed V2 base coefficients remain:

| Component | V3 value |
|---|---:|
| Collision reward | -10.0 |
| Arrival reward | +5.0 |
| High-speed reward | +0.05 |

The wrapper applies:

$$
R_t^{V3}=R_t^{V2}+2.0\Delta p_t-0.005
$$

At the 5 Hz policy frequency and 30-second duration, a full timeout contains at most 150 decisions, so its maximum cumulative time cost is:

$$
150\times0.005=0.75
$$

The cumulative normalized progress bonus is bounded by 2.0 for a monotonic route completion. This gives PPO useful intermediate feedback while keeping collision as the dominant single terminal penalty.

### 29.4 Hypothesis and acceptance rule

**Hypothesis:** dense progress feedback will reduce waiting/timeouts and increase successful completion relative to both V1 and V2, without reversing V2's collision improvement.

V3 is retained for the next research phase only if the 500-episode seed-42 screening run satisfies all three predefined conditions:

$$
Success_{V3}>55.8\%
$$

$$
Collision_{V3}<43.6\%
$$

$$
Incomplete_{V3}<3.4\%
$$

Reward magnitude is not an acceptance metric because V3 uses a different reward definition.

### 29.5 Implementation and verification

Added:

- `safeintent_rl/envs/reward.py`: `RouteProgressRewardWrapper`;
- `configs/intersection_reward_v3.yaml`: isolated V3 coefficients;
- `tests/test_progress_reward.py`: progress and stall test cases;
- environment-factory integration and package export;
- per-step diagnostic fields for base reward, route progress, progress delta, progress reward, time cost, and shaped reward.

Python syntax checks passed in the maintenance workspace. The complete dependency-based test suite must be run in the user's Python 3.12 project environment before training.

### 29.6 Reproducible execution

Pull and test:

```powershell
git pull origin main
python -m pytest
```

Train:

```powershell
python scripts/train_ppo.py --config configs/intersection_reward_v3.yaml --timesteps 200000 --seed 42 --output models/ppo_reward_v3_seed42
```

Evaluate after training:

```powershell
python scripts/evaluate_policy.py --model models/ppo_reward_v3_seed42.zip --config configs/intersection_reward_v3.yaml --episodes 500 --seed 42 --output results/ppo_reward_v3_seed42.csv
```

Expected evaluation outputs:

```text
results/ppo_reward_v3_seed42.csv
results/ppo_reward_v3_seed42.summary.json
```

After evaluation, both files and the observed acceptance-test decision must be appended to this record even if V3 performs worse.

---

## 30. Milestone M6C result — PPO Reward V3

**Experiment ID:** E-B1-V3-S42-200K  
**Status:** Evaluated; screening gate passed, improvement not statistically confirmed  
**Date recorded:** 2026-09-04  
**Training seed:** 42  
**Evaluation seeds:** 42–541  
**Raw-result commit:** [`31b3af1`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/31b3af13f2d5d6a3247a788d2f0c75f15eadeacb)

### 30.1 Verified result

| Metric | PPO V3 result |
|---|---:|
| Successful episodes | 283/500 |
| Collision episodes | 217/500 |
| Incomplete non-collision episodes | 0/500 |
| Mean reward | 1.5800 |
| Mean episode length | 37.256 decisions |
| Success rate | 56.6% |
| Collision rate | 43.4% |
| Mean travel time | 7.4512 s |
| Mean minimum finite TTC | 0.6212 s |
| Mean unsafe-TTC events/episode | 15.346 |
| Mean unsafe-TTC events/decision | 0.4119 |
| Mean safety interventions | 0 |

The raw CSV and summary JSON were verified in the repository. Reward magnitude is not compared with V1 or V2 because each version uses a different reward definition.

### 30.2 Predefined screening gate

The V3 conditions were fixed before training:

$$
Success_{V3}>55.8\%
$$

$$
Collision_{V3}<43.6\%
$$

$$
Incomplete_{V3}<3.4\%
$$

Observed:

$$
56.6\%>55.8\%,\qquad43.4\%<43.6\%,\qquad0.0\%<3.4\%
$$

V3 passed all three screening conditions. The margins over V1 were small, so passing this engineering gate is not treated as proof of a general improvement.

### 30.3 Comparison across reward versions

| Metric | PPO V1 | PPO V2 | PPO V3 | V3 minus V1 |
|---|---:|---:|---:|---:|
| Success rate | 55.8% | 54.0% | 56.6% | +0.8 pp |
| Collision rate | 43.6% | 42.6% | 43.4% | -0.2 pp |
| Incomplete rate | 0.6% | 3.4% | 0.0% | -0.6 pp |
| Mean travel time | 7.4700 s | 9.5344 s | 7.4512 s | -0.0188 s |
| Mean minimum TTC | 0.6332 s | 0.6549 s | 0.6212 s | -0.0120 s |
| Unsafe events/episode | 15.142 | 16.414 | 15.346 | +0.204 |
| Unsafe events/decision | 0.4054 | 0.3443 | 0.4119 | +0.0065 |

V3 eliminated the V2 timeout behavior and restored efficient completion. However, minimum TTC decreased and unsafe-event frequency per decision increased relative to both earlier policies. The progress/time terms therefore produced a more decisive policy, not a clearly safer one.

### 30.4 Paired V1-to-V3 transitions

The 500 rows correspond to identical environment seeds.

| V1 outcome | V3 outcome | Episodes |
|---|---|---:|
| Success | Success | 266 |
| Success | Collision | 13 |
| Collision | Success | 16 |
| Collision | Collision | 202 |
| Incomplete | Success | 1 |
| Incomplete | Collision | 2 |

Relative to V1, V3 gained 17 successful episodes and lost 13. An exact paired McNemar test on success disagreement gave:

$$
p=0.5847
$$

The observed +0.8 percentage-point success change is not statistically significant at the 0.05 level.

### 30.5 Paired V2-to-V3 transitions

| V2 outcome | V3 outcome | Episodes |
|---|---|---:|
| Success | Success | 249 |
| Success | Collision | 21 |
| Collision | Success | 32 |
| Collision | Collision | 181 |
| Incomplete | Success | 2 |
| Incomplete | Collision | 15 |

V3 gained 34 successes and lost 21 relative to V2. The exact paired McNemar result was (p=0.1048), which is also not significant at the 0.05 level.

### 30.6 Decision

V3 is retained as a better completion-oriented screening candidate because it passed the predefined gate and produced no incomplete episodes. It is not declared the final baseline or a statistically confirmed improvement.

All V3 failures were collisions. The next controlled change must therefore target interaction risk while preserving the progress and time terms that eliminated waiting.

---

## 31. Milestone M6D design — PPO Reward V4 risk-aware experiment

**Experiment ID:** E-B1-V4-S42-200K-H10042  
**Status:** Implemented; execution pending  
**Date fixed:** 2026-09-04  
**Configuration:** `configs/intersection_reward_v4.yaml`  
**Primary changed factor:** bounded TTC-risk penalty

### 31.1 Hypothesis

Adding a moderate dense penalty as TTC approaches zero will teach PPO to avoid collision trajectories earlier. Retaining V3's progress reward and time cost should prevent the conservative timeout failure observed in V2.

### 31.2 Risk term

For finite minimum TTC (	au_t) and threshold (	au_c=2.0\text{ s}), define:

$$
q_t=\max\left(0,1-\frac{\tau_t}{\tau_c}\right)
$$

For no finite closing interaction, (q_t=0). The V4 reward is:

$$
R_t^{V4}=R_t^{V3}-w_q q_t
$$

with:

$$
w_q=0.2
$$

Therefore:

$$
0\leq w_q q_t\leq0.2
$$

per policy decision. The term is largest when collision is imminent and becomes zero at or above 2.0 seconds TTC. This TTC is a radial constant-velocity risk surrogate, not a guaranteed physical collision probability.

### 31.3 Controlled variables

Unchanged from V3:

- environment and traffic configuration;
- observation and three-action control space;
- V2 base reward coefficients;
- progress weight 2.0;
- time cost 0.005;
- PPO hyperparameters;
- 200,000 requested training timesteps;
- training seed 42.

Only the TTC-risk term is activated.

### 31.4 Untouched holdout protocol

Seeds 42–541 have already influenced V4 design and will not be used for V4 selection. Before V4 training is interpreted, the existing V3 checkpoint is evaluated on the untouched seeds:

$$
10042,10043,\ldots,10541
$$

V4 is then evaluated on exactly the same 500 holdout episodes. Paired rows permit direct transition analysis and an exact McNemar test.

A meaningful screening improvement requires:

1. V4 success at least 3 percentage points above V3 holdout;
2. V4 collision at least 3 percentage points below V3 holdout;
3. V4 incomplete rate no greater than 2%;
4. V4 mean minimum TTC no lower than V3 holdout;
5. paired success improvement with (p<0.05).

A promising but non-significant result may advance only to multi-training-seed confirmation; it will not be called an improvement.

### 31.5 Verification status

- V4 YAML differs from V3 only by `risk_weight: 0.2` and `risk_ttc_threshold: 2.0`.
- The risk term is bounded and zero for TTC at or above the threshold.
- A unit test checks TTC 1.5 s, normalized risk 0.25, and penalty 0.05.
- Python syntax checks passed.
- The maintenance environment lacked pytest, so the complete suite must pass in the user's Python 3.12 environment before training.

### 31.6 Reproducible commands

Pull and test:

```powershell
git pull origin main
python -m pytest
```

Evaluate the existing V3 model on the untouched holdout first:

```powershell
python scripts/evaluate_policy.py --model models/ppo_reward_v3_seed42.zip --config configs/intersection_reward_v3.yaml --episodes 500 --seed 10042 --output results/ppo_reward_v3_holdout_seed10042.csv
```

Train V4:

```powershell
python scripts/train_ppo.py --config configs/intersection_reward_v4.yaml --timesteps 200000 --seed 42 --output models/ppo_reward_v4_seed42
```

Evaluate V4 on the identical holdout:

```powershell
python scripts/evaluate_policy.py --model models/ppo_reward_v4_seed42.zip --config configs/intersection_reward_v4.yaml --episodes 500 --seed 10042 --output results/ppo_reward_v4_holdout_seed10042.csv
```

Expected comparison artifacts:

```text
results/ppo_reward_v3_holdout_seed10042.csv
results/ppo_reward_v3_holdout_seed10042.summary.json
results/ppo_reward_v4_holdout_seed10042.csv
results/ppo_reward_v4_holdout_seed10042.summary.json
```

The result and decision must be appended even if V4 performs worse.


---

## 32. Milestone M6D result — PPO Reward V4 holdout

**Experiment ID:** E-B1-V4-S42-200K-H10042

**Status:** Evaluated and rejected as an improvement

**Date recorded:** 2026-09-04

**Training seed:** 42

**Evaluation seeds:** 10042–10541

**Configuration:** `configs/intersection_reward_v4.yaml`

**Raw-result commit:** [`a29e705`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/a29e705415f5f5b7a88c74c598d5d5e7babef2ae)

**Raw result paths:** `results/ppo_reward_v4_holdout_seed10042.csv` and `results/ppo_reward_v4_holdout_seed10042.summary.json`

### 32.1 Verified holdout result

| Metric | V3 holdout | V4 holdout | V4 minus V3 |
|---|---:|---:|---:|
| Successful episodes | 298/500 | 271/500 | -27 |
| Collision episodes | 202/500 | 205/500 | +3 |
| Incomplete non-collision episodes | 0/500 | 24/500 | +24 |
| Success rate | 59.6% | 54.2% | -5.4 pp |
| Collision rate | 40.4% | 41.0% | +0.6 pp |
| Incomplete rate | 0.0% | 4.8% | +4.8 pp |
| Mean reward | 2.0889 | 0.0787 | -2.0102 |
| Mean episode length | 38.106 | 49.200 | +11.094 decisions |
| Mean travel time | 7.6212 s | 9.8400 s | +2.2188 s |
| Mean minimum finite TTC | 0.6164 s | 0.6316 s | +0.0152 s |
| Mean unsafe-TTC events/episode | 15.816 | 17.186 | +1.370 |
| Unsafe-TTC events/decision | 0.4151 | 0.3493 | -15.8% relative |

V4 improved mean minimum TTC and reduced unsafe-TTC frequency after normalizing for its longer episodes. Those safety indicators did not translate into fewer collisions or more completed routes.

### 32.2 Predefined acceptance test

The Section 31.4 gate was fixed before V4 training. Relative to the V3 holdout, V4 required at least 62.6% success, at most 37.4% collision, at most 2% incomplete episodes, non-worsening mean minimum TTC, and a paired success improvement with $p<0.05$.

Observed:

| Requirement | Observed V4 result | Decision |
|---|---:|---|
| Success at least 62.6% | 54.2% | Failed |
| Collision at most 37.4% | 41.0% | Failed |
| Incomplete at most 2.0% | 4.8% | Failed |
| Mean minimum TTC at least 0.6164 s | 0.6316 s | Passed |
| Significant paired success improvement | Significant decrease, $p=0.00182$ | Failed |

V4 failed four of the five predefined requirements.

### 32.3 Paired V3-to-V4 outcome transitions

The 500 rows correspond to the identical environment seeds.

| V3 holdout outcome | V4 holdout outcome | Episodes |
|---|---|---:|
| Success | Success | 249 |
| Success | Collision | 36 |
| Success | Incomplete | 13 |
| Collision | Success | 22 |
| Collision | Collision | 169 |
| Collision | Incomplete | 11 |

V4 gained 22 successes from V3 collision episodes but lost 49 V3 successes. An exact two-sided McNemar test on success disagreement gave:

$$
p=0.0018204
$$

The significant direction is harmful: V4 reduced success. For collision disagreement, V4 removed 33 V3 collisions and introduced 36 new collisions, giving:

$$
p=0.80995
$$

There is no paired evidence that V4 changed collision probability.

### 32.4 Interpretation and decision

The bounded TTC-risk reward made the policy more cautious, as shown by higher mean TTC and fewer unsafe decisions proportionally. It also increased travel time by 29.1% and produced 24 incomplete episodes without reducing collision rate. This reproduces the conservative-waiting failure previously observed in V2.

**Decision:** reject V4 as an improvement and preserve it as a negative result. Do not replace V3. V3 remains the current best completion-oriented policy for the next controlled comparisons.


---

## 33. Milestone M9 result — PPO V3 with TTC safety shield

**Experiment ID:** E-B3-V3-S42-H10042-TTC2

**Status:** Evaluated for one training seed; rejected as a replacement for unshielded V3

**Date recorded:** 2026-09-04

**Base checkpoint:** `models/ppo_reward_v3_seed42.zip`

**Training seed:** 42

**Evaluation seeds:** 10042–10541

**Configuration:** `configs/intersection_reward_v3.yaml`

**Changed factor:** inference-time TTC safety shield enabled

**TTC threshold:** 2.0 s

**Raw-result commit:** [`3e5dcfb`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/3e5dcfb5e3b830f996f18a591d11d85c19d607ac)

**Raw result paths:** `results/ppo_reward_v3_shield_holdout_seed10042.csv` and `results/ppo_reward_v3_shield_holdout_seed10042.summary.json`

No model retraining, reward coefficient, environment configuration, seed, episode count, or evaluation rule changed. The only experimental difference from the V3 holdout was `--safety-shield`.

### 33.1 Pre-experiment verification

The complete project test suite was run before accepting the experiment result:

```text
17 passed in 2.07 s
```

Command:

```powershell
python -m pytest -p no:cacheprovider
```

The shield rule remained the Section 11 rule:

```text
IF minimum TTC <= 2.0 s
AND PPO proposes FASTER or IDLE
THEN execute SLOWER
ELSE execute PPO action
```

Evaluation command:

```powershell
python scripts/evaluate_policy.py --model models/ppo_reward_v3_seed42.zip --config configs/intersection_reward_v3.yaml --episodes 500 --seed 10042 --safety-shield --output results/ppo_reward_v3_shield_holdout_seed10042.csv
```

### 33.2 Artifact validation

The CSV contains exactly 500 rows and the expected eight metric columns. No row is simultaneously marked as success and collision. Every value in the summary JSON was recomputed from the CSV and matched.

SHA-256 checksums:

```text
ppo_reward_v3_shield_holdout_seed10042.csv
397a3c897302773c1af697395cdd1f74252f5ace490e261e6adffff2db793215

ppo_reward_v3_shield_holdout_seed10042.summary.json
e94fe9ea13a1ef0c4e25dd5f869936f9be929927555fded7643da7056106e2fd
```

### 33.3 Verified result

| Metric | V3 holdout | V3 + shield | Shield minus V3 |
|---|---:|---:|---:|
| Successful episodes | 298/500 | 201/500 | -97 |
| Collision episodes | 202/500 | 156/500 | -46 |
| Incomplete non-collision episodes | 0/500 | 143/500 | +143 |
| Success rate | 59.6% | 40.2% | -19.4 pp |
| Collision rate | 40.4% | 31.2% | -9.2 pp |
| Incomplete rate | 0.0% | 28.6% | +28.6 pp |
| Mean reward | 2.0889 | 1.0006 | -1.0883 |
| Mean episode length | 38.106 | 96.474 | +58.368 decisions |
| Mean travel time | 7.6212 s | 19.2948 s | +11.6736 s |
| Mean minimum finite TTC | 0.6164 s | 1.1315 s | +0.5152 s |
| Mean unsafe-TTC events/episode | 15.816 | 34.962 | +19.146 |
| Unsafe-TTC events/decision | 0.4151 | 0.3624 | -12.7% relative |
| Mean safety interventions/episode | 0 | 34.72 | +34.72 |
| Aggregate intervention rate | 0.0% | 35.99% | +35.99 pp |

The shield executed 17,360 overrides across 48,237 policy decisions. Raw unsafe-event counts rose because shielded episodes were much longer; the normalized unsafe-event frequency fell.

### 33.4 Paired V3-to-shield outcome transitions

| V3 holdout outcome | V3 + shield outcome | Episodes |
|---|---|---:|
| Success | Success | 143 |
| Success | Collision | 63 |
| Success | Incomplete | 92 |
| Collision | Success | 58 |
| Collision | Collision | 93 |
| Collision | Incomplete | 51 |

The shield converted 58 V3 collisions into successes, but it lost 155 V3 successes: 63 became collisions and 92 became incomplete episodes. An exact two-sided McNemar test on success disagreement gave:

$$
p=2.1493\times10^{-11}
$$

The success reduction is statistically significant. For collision disagreement, the shield removed 109 V3 collisions and introduced 63 new collisions. The exact paired result was:

$$
p=0.00056145
$$

The collision reduction is also statistically significant, but 51 of the removed collisions became incomplete episodes rather than successes.

### 33.5 Interpretation and decision

The 2.0-second radial-TTC shield produced a real collision reduction and substantially increased minimum TTC. However, it intervened on approximately 36% of all decisions, more than doubled mean travel time, reduced success by 19.4 percentage points, and caused 143 timeouts. It is therefore too conservative under the current rule.

This result resembles the rule-based and V2 waiting failures. Radial closing TTC reacts to nearby closing motion without modeling route-conflict geometry, future steering, or intent. Repeatedly replacing both `IDLE` and `FASTER` with `SLOWER` can stop the ego vehicle even when proceeding would complete the route safely.

**Decision:** retain this run as the evaluated B3 safety-shield ablation and as a scientifically useful negative result. Reject the current 2.0-second shield as a replacement for V3. Do not silently change its threshold or action rule. Any shield redesign or threshold sweep must be recorded as a new experiment. V3 remains the current best policy.


---

## 34. Append-only status, workflow, and decision update

**Date:** 2026-09-04

This section updates current status without deleting or rewriting the earlier planned-state tables.

### 34.1 Milestone status updates

| Milestone | Earlier recorded state | Current state after Sections 32–33 |
|---:|---|---|
| M6D | Implemented; holdout evaluation pending | Evaluated; rejected because success fell, collision did not improve, and incomplete episodes increased |
| M9 | Planned; safety code implemented | Single-seed V3 + shield holdout evaluated; current 2.0 s shield rejected as a V3 replacement |

### 34.2 Decision-register additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-020 | Reject PPO Reward V4 and retain PPO V3 | Retain V4 because minimum TTC improved | V4 success 54.2%, collision 41.0%, incomplete 4.8%; predefined gate failed | Retained |
| D-021 | Preserve the 2.0 s V3 shield as a negative ablation result | Replace V3 with shielded V3 because collision fell | Collision fell to 31.2%, but success fell to 40.2%, incomplete rose to 28.6%, and intervention rate was 35.99% | Retained |
| D-022 | Separate long runs from repository maintenance | Run long PPO jobs inside maintenance tasks or commit model ZIPs | Local computer is better suited to long training/evaluation/animation; GitHub should store code, small raw results, and documentation | Retained |

### 34.3 Change-log additions

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-04 | Evaluated and rejected PPO Reward V4 on holdout seeds 10042–10541 | Test whether bounded TTC-risk shaping reduces collision while preserving V3 completion | 500-row V3/V4 paired analysis and exact McNemar tests | Raw results: `a29e705`; documentation: this update |
| 2026-09-04 | Evaluated V3 with the 2.0 s TTC safety shield | Isolate the effect of inference-time shielding without retraining | 17 tests passed; 500-row CSV/JSON consistency, checksums, paired transitions, and exact tests verified | Raw results: `3e5dcfb`; documentation: this update |
| 2026-09-04 | Formalized local/Codex/GitHub workflow | Keep long simulation jobs local and version only appropriate artifacts | No model ZIP added; only code, small CSV/JSON results, and documentation are eligible for GitHub | This documentation update |

### 34.4 Artifact-storage rule

Large model checkpoints remain intentionally excluded from Git. They will be committed only if a dedicated artifact-storage mechanism is explicitly selected and configured. The current repository commit includes only source code, configurations, tests, small CSV/JSON results, and documentation.

### 34.5 Next planned milestone

With V4 and the initial shield rule both rejected, the next planned project phase remains M7: verify the intent-data pipeline, freeze its dataset protocol, collect the trajectory dataset on the local computer, and train/evaluate the GRU before proceeding to PPO + intent. No intent-data command, dataset size, or GRU result is claimed by this status update.


---

## 35. Milestone M7 design and intent-pipeline hardening

**Experiment ID:** E-I1-DATA-S42-300E

**Status:** Implementation corrected and verified; full dataset collection pending

**Date fixed:** 2026-09-04

**Primary objective:** create a reproducible, leakage-resistant trajectory dataset and a verifiable held-out GRU evaluation before PPO + intent training

### 35.1 Pre-run audit findings

The intent pipeline existed and had been smoke-tested, but the audit found four issues that had to be corrected before a long dataset or GRU run:

1. `action_space.sample()` had its own random-number generator. Seeding only the environment did not make the random ego trajectory reproducible.
2. The stored best GRU state used `detach().cpu()` without `clone()`. On CPU, those tensors can share storage with the live model, allowing later optimization epochs to overwrite the state selected at the best validation epoch.
3. A grouped split could contain no samples from one driver class, making macro metrics and conclusions invalid.
4. Intent evaluation printed metrics to the terminal but did not save a small versionable result artifact or verify that the supplied dataset was the exact archive used for training.

### 35.2 Implemented corrections

The following changes were made without changing the six intent features, GRU architecture, driver-profile probabilities, or environment configuration:

- seed the action space after every environment reset, because HighwayEnv may recreate the action space during reset;
- validate positive history length and sample stride;
- validate dataset shape, label range, sample/label length, and episode-ID length;
- save collection parameters in the dataset archive and a separate summary JSON;
- create deterministic 70%/15%/15% splits by complete episode;
- require the training, validation, and test splits each to contain cautious, normal, and aggressive samples;
- clone the selected best model state so it cannot share CPU tensor storage with later epochs;
- store the exact train, validation, and test indices in the checkpoint;
- store training seed, hyperparameters, class counts, best epoch, and dataset SHA-256 in the checkpoint;
- require evaluation to match the checkpoint's dataset fingerprint and held-out indices;
- save accuracy, majority-class accuracy, balanced accuracy, macro precision/recall/F1, per-class metrics, and the confusion matrix to JSON;
- add focused tests for validation, grouped splitting, checkpoint isolation, reproducibility metadata, and classification metrics.

The large dataset archive and model checkpoint remain ignored by Git. Only the small collection-summary and held-out-metrics JSON files are intended for GitHub.

### 35.3 Reproducibility smoke tests and corrections

An initial three-episode engineering smoke run produced a held-out split with no aggressive samples. Its one-epoch accuracy and macro-F1 values are invalid as research evidence and must not be reported as model performance. This failure motivated the new all-class split gate.

During the same audit, repeated collection over seeds 42–44 produced different sample totals when the action space was seeded before reset. After moving the action-space seed after reset, two independent collections produced exactly 365 samples each with identical feature arrays, labels, episode IDs, label names, and metadata.

The corrected three-episode smoke distribution was:

| Class | Samples |
|---|---:|
| Cautious | 108 |
| Normal | 201 |
| Aggressive | 56 |
| Total | 365 |

These are engineering smoke counts only. They are not the M7 dataset result.

### 35.4 Verification

```text
ruff check: passed
pytest: 25 passed in 2.86 s
repeated seeded collection: arrays and metadata identical
```

The pre-existing 101-character line in `scripts/evaluate_rule_based.py` was also wrapped during lint cleanup; its behavior did not change.

### 35.5 Frozen dataset protocol

| Setting | Fixed value |
|---|---:|
| Environment configuration | `configs/intersection.yaml` |
| Driver behaviors | Enabled |
| Ego collection policy | Seeded random discrete actions |
| Episodes | 300 |
| Environment/action seeds | 42–341 |
| History length | 10 observations |
| Sample stride | 2 policy decisions |
| Features/history step | 6 |
| Dataset path | `data/intent_trajectories_seed42.npz` |
| Summary path | `results/intent_dataset_seed42.summary.json` |

The feature vector remains:

$$
x_t=[\Delta x,\Delta y,\Delta v_x,\Delta v_y,\Delta v,d]
$$

The fifth feature remains the recorded velocity-change magnitude rather than physical acceleration. Changing it now would define a new dataset experiment.

The collection result is accepted for GRU training only if:

1. all 300 requested episodes produce samples;
2. cautious, normal, and aggressive samples are all present;
3. each class represents at least 10% of all samples;
4. the archive and summary report matching parameters and SHA-256 fingerprint.

If any condition fails, the dataset remains recorded as an unsuccessful collection and the training step does not begin until a new protocol is documented.

### 35.6 Frozen GRU training and evaluation protocol

The existing model architecture remains unchanged: one 64-unit GRU followed by LayerNorm, a 32-unit ReLU layer, and a three-class output layer.

| Setting | Fixed value |
|---|---:|
| Split unit | Complete episode |
| Train/validation/test proportions | 70%/15%/15% |
| Split seed | 42 |
| Epochs | 30 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Model-selection metric | Validation accuracy |
| Checkpoint path | `models/intent_gru_seed42.pt` |
| Held-out metrics path | `results/intent_gru_seed42.metrics.json` |

The GRU passes its initial screening only if all of the following hold on the untouched test episodes:

1. accuracy exceeds the test-set majority-class baseline by at least 5 percentage points;
2. macro F1 is at least 0.50;
3. recall for every driver class is at least 0.40;
4. recomputed accuracy exactly matches the accuracy stored in the checkpoint.

These are screening criteria for one dataset/training seed, not a final multi-seed intent result. A failed or inconclusive result must be appended and preserved before feature or model changes are proposed.

### 35.7 Reproducible local-compute commands

Pull and verify before collection:

```powershell
git pull origin main
python -m pytest
```

Collect the frozen dataset:

```powershell
python scripts/collect_intent_data.py --episodes 300 --history-length 10 --sample-stride 2 --seed 42 --output data/intent_trajectories_seed42.npz --summary-output results/intent_dataset_seed42.summary.json
```

After the collection summary is reviewed and recorded, train the GRU:

```powershell
python scripts/train_intent.py --data data/intent_trajectories_seed42.npz --output models/intent_gru_seed42.pt --epochs 30 --batch-size 128 --learning-rate 0.001 --seed 42
```

Evaluate only the checkpoint's untouched test split:

```powershell
python scripts/evaluate_intent.py --data data/intent_trajectories_seed42.npz --model models/intent_gru_seed42.pt --output results/intent_gru_seed42.metrics.json
```

### 35.8 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Reason | Status |
|---|---|---|---|---|
| D-023 | Seed the random ego action space after every reset | Seed only the environment or seed before reset | Repeated collections were not reproducible until post-reset action seeding | Retained |
| D-024 | Require dataset fingerprints, exact split indices, and all-class split coverage | Trust filenames and silently evaluate any supplied archive | Prevent dataset mismatch, sample leakage, and invalid class metrics | Retained |
| D-025 | Clone the best CPU model state | Keep `detach().cpu()` references | Prevent later epochs from overwriting the validation-selected checkpoint | Retained |
| D-026 | Version only intent summary/metrics JSON artifacts | Commit `.npz` datasets and `.pt` checkpoints | Follow the local-compute/GitHub artifact-storage boundary | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-04 | Hardened and froze the M7 intent pipeline | Correct reproducibility, checkpoint-selection, split-validity, and result-storage defects before long runs | Ruff passed; 25 tests passed; repeated seeded collection matched | This implementation update |

**Next action:** run only the 300-episode collection command first. Review and append its summary before starting GRU training.


---

## 36. Milestone M7A result — frozen intent dataset collection

**Experiment ID:** E-M7-DATA-S42-E300-H10-K2

**Status:** Collected, validated, and accepted for the initial GRU screening

**Date recorded:** 2026-09-04

**Environment configuration:** `configs/intersection.yaml`

**Collector:** seeded random ego policy with driver behaviors enabled

**Episode seeds:** 42–341

**Changed factor:** none; this run used the frozen Section 35.5 protocol

**Summary commit:** [`1ec8e03`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/1ec8e03a6e420baf4a361e61e9695495a5caedc6)

**Local dataset path:** `data/intent_trajectories_seed42.npz`

**Versioned summary path:** `results/intent_dataset_seed42.summary.json`

The dataset archive remains local and ignored by Git. Only its small summary JSON was committed.

### 36.1 Pre-collection verification

The M7 implementation was verified immediately before the collection phase as recorded in Section 35.4:

```text
ruff check: passed
pytest: 25 passed in 2.86 s
repeated seeded collection: arrays and metadata identical
```

Collection command:

```powershell
python scripts/collect_intent_data.py --episodes 300 --history-length 10 --sample-stride 2 --seed 42 --output data/intent_trajectories_seed42.npz --summary-output results/intent_dataset_seed42.summary.json
```

### 36.2 Artifact validation

The local archive and committed summary were inspected after collection.

| Property | Verified value |
|---|---:|
| Dataset file size | 5,415,856 bytes |
| Feature-array shape | `(109596, 10, 6)` |
| Label-array shape | `(109596,)` |
| Episode-ID shape | `(109596,)` |
| Unique episode IDs | 300 |
| Episode-ID range | 0–299 |
| Samples per episode | 18–1,421 |
| Label names | cautious, normal, aggressive |
| Metadata collector | `seeded_random_ego_policy` |
| Metadata episode count | 300 |
| Metadata seed range | 42–341 |
| Metadata history length | 10 |
| Metadata sample stride | 2 |

The label counts sum exactly to 109,596 samples. The dataset SHA-256 recomputed from the local archive matches the fingerprint stored in the committed summary:

```text
56433621bdcc5fe9a635f57f068e096a9cb3d47036179a64ab390311fab302b0
```

### 36.3 Class distribution

| Driver class | Samples | Fraction |
|---|---:|---:|
| Cautious | 32,631 | 29.774% |
| Normal | 49,213 | 44.904% |
| Aggressive | 27,752 | 25.322% |
| **Total** | **109,596** | **100.000%** |

The distribution is not uniform, but no class is rare enough to fail the predefined collection gate. Episode-grouped splitting and the training-only class weights remain necessary; no balancing or resampling was introduced after seeing these counts.

### 36.4 Predefined collection gate

| Requirement | Observed result | Decision |
|---|---:|---|
| All 300 episodes produce samples | 300/300 episodes | Passed |
| All three classes are present | 3/3 classes | Passed |
| Every class is at least 10% | Minimum 25.322% | Passed |
| Archive and summary parameters/fingerprint match | Exact match | Passed |

All four frozen acceptance conditions passed.

### 36.5 Interpretation and decision

The collection produced enough samples for the planned episode-grouped split while retaining substantial representation of all three simulated driver classes. The wide range in samples per episode reflects differing episode durations, so samples must not be randomly split across episodes. The frozen group-split rule prevents histories from the same episode leaking into multiple partitions.

**Decision:** accept this archive as the single frozen dataset for the initial seed-42 GRU screening. Proceed with the Section 35.6 hyperparameters only after pulling this record and rerunning the test suite. Do not recollect, rebalance, alter features, change the split seed, or substitute another dataset without recording a new experiment.

### 36.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-027 | Accept `intent_trajectories_seed42.npz` for the initial GRU screening | Recollect or rebalance after observing the class distribution | All four predefined dataset gates passed; 109,596 samples across all 300 episodes; minimum class fraction 25.322%; fingerprint verified | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-04 | Collected and accepted the frozen M7 intent dataset | Establish the leakage-resistant input artifact for the initial GRU screening | Archive structure, metadata, counts, episode coverage, class fractions, and SHA-256 matched the committed summary | Summary: `1ec8e03`; documentation: this update |

**Next action:** pull this documentation update, rerun the complete test suite, and train only the frozen seed-42 GRU described in Section 35.6. Do not evaluate or tune it until the training result is preserved.


---

## 37. Milestone M7B result — frozen seed-42 GRU training

**Experiment ID:** E-M7-GRU-S42-DATA56433621

**Status:** Training completed; checkpoint validated and retained locally; frozen evaluator pending

**Date recorded:** 2026-09-04

**Dataset:** `data/intent_trajectories_seed42.npz`

**Dataset SHA-256:** `56433621bdcc5fe9a635f57f068e096a9cb3d47036179a64ab390311fab302b0`

**Changed factor:** none; this run used the frozen Section 35.6 training protocol

### 37.1 Pre-training verification

The complete checks were rerun after accepting the dataset and immediately before authorizing training:

```text
ruff check: passed
pytest: 25 passed in 2.79 s
```

Training command:

```powershell
python scripts/train_intent.py --data data/intent_trajectories_seed42.npz --output models/intent_gru_seed42.pt --epochs 30 --batch-size 128 --learning-rate 0.001 --seed 42
```

### 37.2 Frozen training result

| Property | Verified value |
|---|---:|
| Training seed | 42 |
| Epochs | 30 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Model-selection metric | Validation accuracy |
| Best epoch | 28 |
| Best validation accuracy | 63.4638% |
| Stored test accuracy | 63.3549% |
| Split mode | Complete episode |
| Train/validation/test episodes | 210/45/45 |
| Train/validation/test samples | 76,761/16,912/15,923 |

The stored test accuracy is a provisional checkpoint field produced by the frozen training script. It is not the complete M7 screening result. The separate evaluator must reproduce it and compute macro F1 and per-class recall before an acceptance decision is made.

### 37.3 Split and checkpoint validation

| Check | Result |
|---|---|
| Checkpoint dataset fingerprint matches the frozen archive | Passed |
| Train, validation, and test indices cover all 109,596 samples exactly once | Passed |
| Sample-index sets are pairwise disjoint | Passed |
| Episode-ID sets are pairwise disjoint | Passed |
| Every split contains all three classes | Passed |
| Model tensors are finite | Passed |
| Normalization statistics are finite with positive standard deviations | Passed |

Class counts within each split:

| Split | Cautious | Normal | Aggressive | Total |
|---|---:|---:|---:|---:|
| Train | 22,587 | 33,779 | 20,395 | 76,761 |
| Validation | 5,483 | 7,846 | 3,583 | 16,912 |
| Test | 4,561 | 7,588 | 3,774 | 15,923 |

The test-set majority-class baseline is 47.6543%. The checkpoint's stored test accuracy is 15.7006 percentage points higher, so the accuracy portion of the screening gate is provisionally satisfied. Macro F1 and per-class recall remain unknown until the frozen evaluator is run.

Checkpoint artifact details:

| Property | Value |
|---|---|
| Local path | `models/intent_gru_seed42.pt` |
| File size | 1,045,349 bytes |
| SHA-256 | `10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05` |
| Git status | Ignored by `models/*.pt`; not versioned |

### 37.4 Documentation correction

Section 36.3 stated that "training-only class weights remain necessary." That sentence was inaccurate: the frozen implementation uses unweighted `CrossEntropyLoss`, and no class weights, balancing, or resampling were added before this run. This correction is appended here rather than rewriting the earlier record. The Section 35.6 hyperparameters and executed training protocol remain unchanged.

### 37.5 Interpretation and decision

The checkpoint is structurally valid, matches the accepted dataset, and preserves a leakage-resistant episode split. Its accuracy is materially better than the test majority baseline, but accuracy alone cannot show whether cautious and aggressive drivers are recognized adequately.

**Decision:** accept the local checkpoint as the sole input to the frozen held-out evaluator. Do not commit the `.pt`, retrain, tune, or inspect alternative epochs before evaluation. Run the evaluator once against the exact dataset and checkpoint, then version only `results/intent_gru_seed42.metrics.json` and append the result whether it passes or fails.

### 37.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-028 | Retain the trained GRU checkpoint locally and proceed to one frozen held-out evaluation | Commit the `.pt`, retrain immediately, or select a different epoch after seeing results | Dataset/split integrity passed; best epoch 28; validation accuracy 63.4638%; checkpoint hash recorded | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-04 | Trained and validated the frozen seed-42 intent GRU | Test whether the fixed trajectory representation supports intent classification before PPO integration | Fingerprint, exact split coverage, episode isolation, class coverage, finite tensors, normalization statistics, and checkpoint hash verified | Documentation: this update |

**Next action:** run the frozen evaluator once and commit only `results/intent_gru_seed42.metrics.json`; keep `models/intent_gru_seed42.pt` local.


---

## 38. Milestone M7C result — frozen GRU held-out evaluation

**Experiment ID:** E-M7-GRU-EVAL-S42-DATA56433621

**Status:** Evaluated; all predefined initial screening gates passed

**Date recorded:** 2026-09-04

**Dataset:** `data/intent_trajectories_seed42.npz`

**Dataset SHA-256:** `56433621bdcc5fe9a635f57f068e096a9cb3d47036179a64ab390311fab302b0`

**Checkpoint:** local `models/intent_gru_seed42.pt`

**Checkpoint SHA-256:** `10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05`

**Metrics commit:** [`d478c04`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/d478c04278bcdf5fd52f6cab37510314119ab059)

**Changed factor:** none; this was the single frozen evaluation specified in Sections 35.6 and 37.5

The `.pt` checkpoint remains local and ignored. Only `results/intent_gru_seed42.metrics.json` was committed.

### 38.1 Verification context and command

The 25-test project suite passed immediately before the frozen training phase, and no source code or evaluation protocol changed between training and evaluation. The user subsequently reported completion of the prescribed pull, test, and evaluation sequence.

Evaluation command:

```powershell
python scripts/evaluate_intent.py --data data/intent_trajectories_seed42.npz --model models/intent_gru_seed42.pt --output results/intent_gru_seed42.metrics.json
```

### 38.2 Verified held-out metrics

| Metric | Result |
|---|---:|
| Test samples | 15,923 |
| Accuracy | 63.3549% |
| Majority-class accuracy | 47.6543% |
| Accuracy margin over majority | +15.7006 pp |
| Balanced accuracy | 60.7874% |
| Macro precision | 63.5768% |
| Macro recall | 60.7874% |
| Macro F1 | 61.8289% |

Per-class results:

| Class | Precision | Recall | F1 | Support | Predicted |
|---|---:|---:|---:|---:|---:|
| Cautious | 64.9438% | 59.5045% | 62.1053% | 4,561 | 4,179 |
| Normal | 62.6769% | 71.7712% | 66.9165% | 7,588 | 8,689 |
| Aggressive | 63.1097% | 51.0864% | 56.4651% | 3,774 | 3,055 |

Confusion matrix, with true classes in rows and predicted classes in columns:

| True / predicted | Cautious | Normal | Aggressive |
|---|---:|---:|---:|
| Cautious | 2,714 | 1,668 | 179 |
| Normal | 1,194 | 5,446 | 948 |
| Aggressive | 271 | 1,575 | 1,928 |

### 38.3 Independent artifact validation

The metrics JSON was independently recomputed from its confusion matrix:

- row totals exactly match the three reported class supports;
- column totals exactly match the three reported prediction counts;
- the diagonal contains 10,088 correct predictions out of 15,923;
- recomputed accuracy is exactly `0.6335489543427746`;
- recomputed macro F1 is exactly `0.6182894884252775`;
- the reported accuracy exactly matches the checkpoint's stored test accuracy;
- the metrics dataset fingerprint matches the accepted dataset.

Metrics JSON SHA-256:

```text
7e4fcf349049dcc969c449aa00b38bbaac777b3ecf4e4608ec11555cd71a73f2
```

### 38.4 Predefined screening decision

| Requirement | Threshold | Observed | Decision |
|---|---:|---:|---|
| Accuracy exceeds majority baseline by at least 5 pp | At least 52.6543% | 63.3549% | Passed |
| Macro F1 | At least 50.0000% | 61.8289% | Passed |
| Recall for every class | At least 40.0000% | Minimum 51.0864% | Passed |
| Accuracy reproduces checkpoint value exactly | Exact equality | Exact equality | Passed |

All four predefined gates passed.

### 38.5 Interpretation and decision

The fixed six-feature, ten-step history contains usable predictive information for all three driver classes. Normal drivers are identified most reliably. Aggressive-driver recall is the weakest at 51.0864%; 1,575 of 3,774 aggressive samples were predicted as normal. The classifier therefore passes the initial screening but remains imperfect, and downstream PPO must receive predictions rather than privileged ground-truth labels.

This is a single dataset and training seed. It establishes readiness for the controlled PPO + intent experiment, not final classifier generalization or a claim that intent will improve driving outcomes.

**Decision:** accept the seed-42 GRU as the fixed intent model for the first PPO + intent experiment. Do not tune or retrain it based on this test result. Before M8 training, audit and freeze the PPO + intent configuration, observation interface, training seed/timesteps, evaluation seeds, and acceptance criteria. V3 remains the driving-policy baseline until a paired holdout evaluation proves improvement.

### 38.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-029 | Accept the frozen seed-42 GRU for the first controlled PPO + intent experiment | Reject the classifier, tune on the test split, or retrain before integration | All four predefined screening gates passed; macro F1 61.8289%; minimum class recall 51.0864% | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-04 | Evaluated and accepted the frozen seed-42 intent GRU | Determine whether the fixed trajectory representation supports all three intent classes before PPO integration | Metrics arithmetic, confusion matrix totals, dataset fingerprint, checkpoint accuracy, and all four gates independently verified | Metrics: `d478c04`; documentation: this update |

**Next action:** audit and freeze M8 before starting any long PPO + intent training run.


---

## 39. Milestone M8 design and PPO + intent integration hardening

**Experiment ID:** E-M8-PPO-INTENT-V1-S42-200K

**Status:** Implemented, audited, and frozen; long training not yet started

**Date fixed:** 2026-09-04

**Primary hypothesis:** appending inferred driver-class probabilities to the V3 observation will improve completion and collision outcomes without causing conservative waiting

**Primary changed factor:** 15 inferred intent probabilities for the first five visible traffic slots

### 39.1 Pre-run audit findings

The existing PPO + intent framework had not been validated for a long run. The audit found five issues:

1. the wrapper ordered NPC probabilities by Euclidean distance, while HighwayEnv's configured kinematics observation orders traffic slots by absolute lane distance;
2. the base observation can include road objects, so independently selecting only vehicles could shift probabilities away from the rows they were intended to describe;
3. the wrapper invoked the GRU separately for each neighbor at every policy step, causing unnecessary inference overhead;
4. checkpoint loading used unrestricted pickle deserialization and did not pin the accepted GRU fingerprint;
5. PPO training/evaluation did not save complete model, configuration, intent-checkpoint, observation, and seed metadata in small JSON artifacts.

The existing training callback also used a seed offset of 10,000, which overlaps the fixed holdout starting at seed 10042 for a training seed of 42. Although the callback does not update PPO or select the final saved checkpoint, M8 uses a distinct internal-evaluation offset to keep monitoring traffic separate from the paired holdout.

### 39.2 Implemented corrections

The following changes were made before M8 training:

- query the exact sorted traffic objects used by the base kinematics observation;
- keep obstacle slots aligned and append zero intent probabilities for those non-vehicle rows;
- assign vehicle predictions to the corresponding visible traffic slots;
- batch all ready neighbor histories into one GRU inference call per environment step;
- require sorted kinematics observations and reject incompatible slot counts;
- load checkpoints with `weights_only=True` and an explicit NumPy allowlist;
- validate input size, label order, normalization shapes, finite values, and positive standard deviations;
- pin the accepted intent checkpoint by SHA-256 for training and evaluation;
- freeze CPU intent inference for the first controlled run;
- expose the neighbor count, device, fingerprint, and internal-evaluation seed offset explicitly on the command line;
- save a versionable PPO training-summary JSON containing configuration, model, intent-model, seed, hyperparameter, timestep, and observation metadata;
- extend evaluation summaries with model/configuration fingerprints, seed range, intent settings, and shield state;
- add focused tests for visible-slot alignment, batched inference, obstacle-slot preservation, incompatible ordering, safe checkpoint loading, and fingerprint rejection.

No driver profile, intent feature, GRU weight, V3 reward coefficient, action, traffic setting, PPO optimization hyperparameter, training seed, or holdout seed changed.

### 39.3 Verification and engineering smoke results

Final static and unit verification:

```text
ruff check: passed
pytest: 28 passed in 2.70 s
```

A real-environment paired smoke using the accepted GRU and identical seed/action sequence verified:

```text
augmented observation shape: (120,)
all observation values finite: yes
observation contained by declared space: yes
two independent seeded environments identical: yes
12 policy steps completed with batched intent predictions: yes
```

The first 32-step PPO smoke launch failed before environment creation because the maintenance workspace used an editable installation pointing to the user's separate checkout. It raised an import error for the newly added fingerprint helper; no PPO steps were collected. Rerunning the same smoke as a module resolved the maintenance-only import context and completed successfully:

```text
requested/collected timesteps: 32/32
reported throughput: 58 fps
training summary written: yes
model/configuration/intent hashes written: yes
observation shape recorded as [120]: yes
```

These are engineering smoke results only and are not M8 policy-performance evidence.

### 39.4 Frozen observation interface

The original V3 kinematics observation remains 15 rows by 7 features, flattened to 105 values by the intent wrapper. The appended block contains three probabilities for each of the first five visible traffic slots:

$$
o_t^{M8}=[o_t^{V3},\hat p_1(C,N,A),\ldots,\hat p_5(C,N,A)]
$$

Therefore:

$$
\dim(o_t^{M8})=105+5\times3=120
$$

For a visible vehicle with fewer than ten consecutive history steps, its probability triplet is `(1/3, 1/3, 1/3)`. Missing traffic slots and observed obstacle slots receive `(0, 0, 0)`. Once ten steps are available, the frozen GRU supplies cautious, normal, and aggressive softmax probabilities in that order. Ground-truth driver labels are never appended or read by the intent wrapper.

### 39.5 Frozen M8 training protocol

| Setting | Fixed value |
|---|---:|
| Environment/reward configuration | `configs/intersection_reward_v3.yaml` |
| Configuration SHA-256 | `433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69` |
| Training seed | 42 |
| Requested timesteps | 200,000 |
| Expected collected timesteps | 200,704 |
| Learning rate | 0.0003 |
| PPO rollout steps | 1,024 |
| Batch size | 64 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |
| Policy network | `[256, 256]` |
| Intent checkpoint | `models/intent_gru_seed42.pt` |
| Intent checkpoint SHA-256 | `10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05` |
| Intent neighbors | 5 visible slots |
| Intent history | 10 policy observations |
| Intent inference device | CPU |
| Augmented observation shape | 120 |
| Internal-evaluation seed offset | 1,000 |
| Internal evaluation | 20 episodes every 10,000 calls |
| Safety shield | Disabled |
| Final checkpoint | `models/ppo_intent_v1_seed42.zip` |
| Training summary | `results/ppo_intent_v1_seed42.training.json` |

The final post-training PPO state is the experiment checkpoint, matching the V3 procedure. The callback's best model remains diagnostic and is not substituted after observing evaluation results.

Training command:

```powershell
python scripts/train_ppo.py --config configs/intersection_reward_v3.yaml --timesteps 200000 --seed 42 --intent-model models/intent_gru_seed42.pt --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 --intent-neighbors 5 --intent-device cpu --eval-seed-offset 1000 --summary-output results/ppo_intent_v1_seed42.training.json --output models/ppo_intent_v1_seed42
```

After training, stop and validate the JSON before evaluation. It must report the fixed configuration and intent hashes, seed 42, 200,704 collected timesteps, observation shape `[120]`, five neighbors, CPU intent inference, and no safety shield. The ZIP remains local; only the training summary is versioned.

### 39.6 Frozen paired holdout protocol

After the training artifact is recorded, evaluate exactly 500 episodes on seeds 10042–10541:

```powershell
python scripts/evaluate_policy.py --model models/ppo_intent_v1_seed42.zip --config configs/intersection_reward_v3.yaml --episodes 500 --seed 10042 --intent-model models/intent_gru_seed42.pt --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 --intent-neighbors 5 --intent-device cpu --output results/ppo_intent_v1_holdout_seed10042.csv
```

Expected versionable outputs:

```text
results/ppo_intent_v1_holdout_seed10042.csv
results/ppo_intent_v1_holdout_seed10042.summary.json
```

The comparator is the existing V3 holdout over the identical seeds:

| V3 metric | Fixed baseline |
|---|---:|
| Success | 59.6% |
| Collision | 40.4% |
| Incomplete | 0.0% |
| Mean minimum TTC | 0.6164 s |

M8 passes its initial driving-policy screen only if all predefined conditions hold:

1. success is at least 62.6%;
2. collision is at most 37.4%;
3. incomplete non-collision episodes are at most 2.0%;
4. mean minimum TTC is at least 0.6164 seconds;
5. the paired success change is favorable with an exact two-sided McNemar test at `p < 0.05`.

These thresholds reuse the previously defined minimum three-percentage-point effect and completion/safety constraints. Reward magnitude is not an acceptance metric. A failed or mixed result must be recorded before any classifier, neighbor-count, history, or PPO change is considered.

### 39.7 Interpretation and decisions

This design isolates the value of inferred intent as closely as possible: the V3 dynamics, reward, PPO budget, optimization settings, training seed, and holdout episodes remain fixed. Only the augmented observation changes. The internal monitoring seed offset and result metadata are research-hygiene corrections that do not supply PPO gradients or alter the final-state selection rule.

The rejected 2.0-second safety shield is not enabled in M8. Combining safety and intent remains a later experiment requiring a separately recorded safety design.

| ID | Decision | Alternatives considered | Reason | Status |
|---|---|---|---|---|
| D-030 | Align intent triplets to the exact visible traffic slots and preserve obstacle positions | Sort an independent NPC list by Euclidean distance or disable obstacles | Prevent semantic row mismatch and avoid changing the V3 base observation | Retained |
| D-031 | Batch inference and pin the accepted GRU fingerprint on CPU | Per-neighbor calls, automatic device selection, or an unverified checkpoint path | Reduce long-run overhead and guarantee the evaluated classifier is used | Retained |
| D-032 | Compare PPO + intent against V3 under the fixed 200K/seed-42/holdout protocol | Change reward, add the rejected shield, or tune intent after seeing holdout results | Isolate the contribution of inferred intent | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-04 | Hardened and froze the M8 PPO + intent experiment | Correct observation alignment, obstacle handling, inference overhead, checkpoint loading, fingerprinting, monitoring seeds, and artifact metadata before a long run | Ruff passed; 28 tests passed; deterministic real-environment smoke passed; 32-step PPO smoke passed after one recorded import-context failure | This implementation update |

**Next action:** pull this implementation, rerun the complete test suite, and execute only the frozen M8 training command. Do not start holdout evaluation until the training summary is validated and appended.


---

## 40. Milestone M8A result — PPO + intent training

**Experiment ID:** E-M8-PPO-INTENT-V1-S42-200K

**Status:** Training completed and checkpoint validated; paired holdout evaluation pending

**Date recorded:** 2026-09-05

**Training-summary commit:** [`b11fc66`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/b11fc667cd524a21528caae45f63a4524b9f7867)

**Changed factor relative to V3:** append the frozen GRU probabilities for five aligned visible traffic slots

No safety shield, reward, traffic, action, PPO optimization, training-budget, or training-seed change was introduced.

### 40.1 Pre-training verification and command

The user reported completion of the prescribed pull, Ruff, pytest, and training sequence. The frozen M8 implementation had passed 28 tests and its real-environment and 32-step PPO engineering smokes before this long run.

Executed command:

```powershell
python scripts/train_ppo.py --config configs/intersection_reward_v3.yaml --timesteps 200000 --seed 42 --intent-model models/intent_gru_seed42.pt --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 --intent-neighbors 5 --intent-device cpu --eval-seed-offset 1000 --summary-output results/ppo_intent_v1_seed42.training.json --output models/ppo_intent_v1_seed42
```

### 40.2 Training-summary validation

| Frozen property | Expected | Recorded | Decision |
|---|---:|---:|---|
| Algorithm | PPO | PPO | Passed |
| Configuration SHA-256 | `433e6972...fae69` | `433e6972...fae69` | Passed |
| Training seed | 42 | 42 | Passed |
| Internal-evaluation seed offset | 1,000 | 1,000 | Passed |
| Requested timesteps | 200,000 | 200,000 | Passed |
| Collected timesteps | 200,704 | 200,704 | Passed |
| Learning rate | 0.0003 | 0.0003 | Passed |
| Rollout steps | 1,024 | 1,024 | Passed |
| Batch size | 64 | 64 | Passed |
| Gamma | 0.99 | 0.99 | Passed |
| GAE lambda | 0.95 | 0.95 | Passed |
| Entropy coefficient | 0.01 | 0.01 | Passed |
| Policy network | `[256, 256]` | `[256, 256]` | Passed |
| Observation shape | `[120]` | `[120]` | Passed |
| Safety shield | Disabled | Disabled | Passed |
| Intent checkpoint SHA-256 | `10483649...abb05` | `10483649...abb05` | Passed |
| Intent neighbors | 5 | 5 | Passed |
| Intent device | CPU | CPU | Passed |

The configuration and accepted intent-checkpoint hashes were independently recomputed from the local files and matched the training summary.

Training-summary JSON SHA-256:

```text
408619b0274972b735f7bef40fc1779a086c08620e318367a83351e3f1c69263
```

### 40.3 Checkpoint validation

| Property | Verified value |
|---|---|
| Local checkpoint | `models/ppo_intent_v1_seed42.zip` |
| File size | 2,370,824 bytes |
| SHA-256 | `954a1d4ef9431ca451de367213d6b65c087d11f277802f9d7bc1ac38c47471e8` |
| Stored timesteps | 200,704 |
| Observation space | 120 finite-or-unbounded float32 features; final 15 bounded to `[0,1]` |
| Action space | Discrete, 3 actions |
| Stored PPO settings | Exact match to Section 39.5 |
| Policy parameters | All finite |
| Git status | Ignored by `models/*.zip`; not versioned |

The local checkpoint SHA-256 exactly matches `model_sha256` in the committed training summary.

### 40.4 Interpretation and decision

The long run completed under the frozen protocol and produced a structurally valid final PPO state. This validates execution and provenance only. It does not establish whether inferred intent improves success, collision, TTC, or waiting behavior.

**Decision:** accept `ppo_intent_v1_seed42.zip` as the sole M8 checkpoint for the predefined paired holdout. Do not substitute the callback's best checkpoint, retrain, add the rejected shield, or change the GRU/neighbor interface. Pull this record, rerun the complete tests, and execute the Section 39.6 evaluation exactly once.

### 40.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-033 | Accept the final 200,704-step PPO + intent checkpoint for paired holdout evaluation | Retrain, select the callback best model, or alter intent/safety settings | All frozen training-summary fields and independent checkpoint integrity checks passed | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Completed and validated M8 PPO + intent training | Produce the controlled seed-42 policy checkpoint for the E2 comparison | Training metadata, file hashes, PPO spaces/hyperparameters, timestep count, and finite parameters verified | Training summary: `b11fc66`; documentation: this update |

**Next action:** run only the frozen 500-episode paired holdout evaluation on seeds 10042–10541 and commit its CSV/summary JSON; keep both model checkpoints local.


---

## 41. Milestone M8B result — PPO + intent paired holdout

**Experiment ID:** E-M8-PPO-INTENT-V1-S42-H10042

**Status:** Evaluated; rejected as an improvement over PPO V3

**Date recorded:** 2026-09-05

**Training seed:** 42

**Evaluation seeds:** 10042–10541

**Configuration:** `configs/intersection_reward_v3.yaml`

**Results commit:** [`dc20045`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/dc20045ea9a451aba8f9190d5d01067985731100)

**Changed factor relative to V3:** the 15-value inferred-intent observation block defined in Section 39.4

No reward, traffic, action, evaluation seed, episode count, safety shield, or success/collision rule changed.

### 41.1 Pre-evaluation verification and command

Immediately before authorizing the holdout evaluation, the complete checks passed:

```text
ruff check: passed
pytest: 28 passed in 4.10 s
```

Evaluation command:

```powershell
python scripts/evaluate_policy.py --model models/ppo_intent_v1_seed42.zip --config configs/intersection_reward_v3.yaml --episodes 500 --seed 10042 --intent-model models/intent_gru_seed42.pt --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 --intent-neighbors 5 --intent-device cpu --output results/ppo_intent_v1_holdout_seed10042.csv
```

The committed summary confirms the fixed configuration and model fingerprints, seeds 10042–10541, five intent neighbors, CPU inference, unsafe-TTC threshold 2.0 seconds, and no safety shield.

### 41.2 Artifact validation

The CSV contains exactly 500 rows with the expected eight episode-metric columns. No episode is simultaneously marked as successful and collided. Every numeric summary value was independently recomputed from the CSV and matched exactly.

SHA-256 checksums:

```text
ppo_intent_v1_holdout_seed10042.csv
72fe15fb876e5ad16007c97e01a8811aa62c5935468c21dac49df8edcdebb498

ppo_intent_v1_holdout_seed10042.summary.json
95ace2dc9cd3904e7b3e0119e56b3a039cdf7b57d0c7904614eacdcf896d6ccf
```

### 41.3 Verified result

| Metric | V3 holdout | PPO + intent | Intent minus V3 |
|---|---:|---:|---:|
| Successful episodes | 298/500 | 295/500 | -3 |
| Collision episodes | 202/500 | 205/500 | +3 |
| Incomplete non-collision episodes | 0/500 | 0/500 | 0 |
| Success rate | 59.6% | 59.0% | -0.6 pp |
| Collision rate | 40.4% | 41.0% | +0.6 pp |
| Incomplete rate | 0.0% | 0.0% | 0.0 pp |
| Mean reward | 2.0889 | 2.0107 | -0.0782 |
| Mean episode length | 38.106 | 38.356 | +0.250 decisions |
| Mean travel time | 7.6212 s | 7.6712 s | +0.0500 s |
| Mean minimum finite TTC | 0.6164 s | 0.6404 s | +0.0241 s |
| Mean unsafe-TTC events/episode | 15.816 | 15.786 | -0.030 |
| Unsafe-TTC events/decision | 0.4151 | 0.4116 | -0.84% relative |
| Mean safety interventions | 0 | 0 | 0 |

The intent-aware policy slightly improved TTC indicators, but it did not improve completion or collision outcomes.

### 41.4 Paired outcome transitions

Rows correspond to identical environment seeds.

| V3 outcome | PPO + intent outcome | Episodes |
|---|---|---:|
| Success | Success | 272 |
| Success | Collision | 26 |
| Success | Incomplete | 0 |
| Collision | Success | 23 |
| Collision | Collision | 179 |
| Collision | Incomplete | 0 |

The intent-aware policy gained 23 successes from V3 collision episodes but lost 26 V3 successes. The exact two-sided McNemar result for success disagreement was:

$$
p=0.77545
$$

Collision disagreement is the inverse of the same 23/26 transitions and therefore has the same p-value. There is no paired evidence that the intent block changed success or collision probability.

### 41.5 Predefined acceptance decision

| Requirement | Threshold | Observed | Decision |
|---|---:|---:|---|
| Success | At least 62.6% | 59.0% | Failed |
| Collision | At most 37.4% | 41.0% | Failed |
| Incomplete | At most 2.0% | 0.0% | Passed |
| Mean minimum TTC | At least 0.6164 s | 0.6404 s | Passed |
| Favorable paired success change | Exact two-sided `p < 0.05` | Unfavorable net -3; `p=0.77545` | Failed |

M8 failed three of the five predefined conditions.

### 41.6 Interpretation and decision

The GRU passed its held-out classification screen, but appending its probabilities for five visible slots did not improve the seed-42 PPO policy. The near-zero paired effect could reflect insufficient online prediction coverage, distribution shift from the random-ego collection policy to PPO trajectories, classifier uncertainty, limited use of the appended features by PPO, or a genuinely weak relationship between the current intent labels and the action decisions that determine collision.

**Decision:** reject PPO + intent V1 as an improvement and retain the result as a negative ablation. PPO V3 remains the current best driving policy. Do not combine this policy with the already rejected 2.0-second shield, tune on the holdout, increase training steps, or change the neighbor/history representation without a new recorded experiment.

Before any retraining, instrument the frozen M8 rollout to measure how often intent predictions are available, their confidence, and their online accuracy against hidden simulator labels used strictly for analysis. This diagnostic will distinguish classifier distribution/coverage failure from a control-learning failure without changing actions or retraining PPO.

### 41.7 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-034 | Reject PPO + intent V1 and retain V3 | Promote intent V1 because mean TTC improved | Success fell to 59.0%, collision rose to 41.0%, paired `p=0.77545`, and three predefined gates failed | Retained |
| D-035 | Diagnose online intent coverage and accuracy before another PPO run | Immediately retrain longer, tune on holdout, or combine two rejected components | M8 produced essentially zero paired task effect despite a passing offline classifier | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Evaluated and rejected PPO + intent V1 on the paired V3 holdout | Test whether inferred driver intent improves V3 under a single-factor comparison | 500-row CSV/summary validation, artifact hashes, paired transitions, exact McNemar test, and five frozen gates | Results: `dc20045`; documentation: this update |

**Next action:** implement and freeze a non-interventional online-intent diagnostic using the existing V1 checkpoint and the same holdout trajectories; do not start another PPO training run yet.


---

## 42. Milestone M8C design — frozen online-intent rollout diagnostic

**Experiment ID:** E-M8C-INTENT-DIAGNOSTIC-V1-H10042

**Status:** Diagnostic implemented and frozen; 500-episode run pending

**Date recorded:** 2026-09-05

**Policy checkpoint:** local `models/ppo_intent_v1_seed42.zip`

**Evaluation seeds:** 10042–10541

**Configuration:** `configs/intersection_reward_v3.yaml`

**Intervention:** none; the diagnostic neither retrains PPO nor changes its observations, actions, rewards, termination rules, or environment randomness

### 42.1 Question and fixed measurement unit

M8B produced essentially no paired task effect despite a GRU that passed the offline held-out screen. This diagnostic separates three explanations before another training run:

1. too few visible-vehicle slots acquire the required ten-step consecutive history;
2. the classifier degrades on PPO's online trajectory distribution;
3. predictions are sufficiently available and accurate, but the PPO control policy did not exploit them.

One diagnostic observation is one of the first five visible traffic slots at one **pre-action policy decision**. These slot observations repeat vehicles across time and therefore are correlated; classification metrics are decision-slot-weighted diagnostic evidence, not independent per-vehicle generalization estimates.

Missing slots and obstacle slots are counted separately. A visible vehicle is `warmup` until ten consecutive observations exist and `predicted` after the frozen GRU supplies probabilities. Classification includes only predicted vehicle slots with a valid hidden simulator label. The wrapper reads that label strictly after inference and only when diagnostics are enabled; it never appends the label to the observation or uses it to choose an action.

### 42.2 Frozen metrics and formulas

For five configured traffic slots and $D$ policy decisions:

$$
N_{slots}=5D
$$

$$
\text{vehicle-slot rate}=\frac{N_{vehicle}}{N_{slots}},\qquad
\text{prediction coverage}=\frac{N_{predicted}}{N_{vehicle}}
$$

$$
\text{labeled-prediction rate}=\frac{N_{labeled}}{N_{predicted}}
$$

For each predicted probability vector $p=(p_C,p_N,p_A)$, confidence is $\max_k p_k$ and normalized entropy is:

$$
H_{norm}(p)=\frac{-\sum_k p_k\log p_k}{\log 3}
$$

The output also records the three-class confusion matrix, accuracy, majority-class accuracy, balanced accuracy, macro precision/recall/F1, and per-class precision/recall/F1/support. Undefined rates remain JSON `null`; zero prediction coverage is preserved as valid diagnostic evidence rather than treated as a software crash.

### 42.3 Frozen protocol and integrity controls

| Property | Frozen value |
|---|---|
| PPO model SHA-256 | `954a1d4ef9431ca451de367213d6b65c087d11f277802f9d7bc1ac38c47471e8` |
| Intent model SHA-256 | `10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05` |
| Configuration SHA-256 | `433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69` |
| Reference CSV SHA-256 | `72fe15fb876e5ad16007c97e01a8811aa62c5935468c21dac49df8edcdebb498` |
| Episodes and seeds | 500; 10042–10541 |
| Policy inference | Deterministic PPO on CPU |
| Intent inference | Frozen GRU on CPU; five aligned slots |
| Safety shield | Disabled |
| Unsafe-TTC reporting threshold | 2.0 seconds |
| Required history | Ten consecutive observations |
| Reference tolerance | Exact booleans/integers; numeric absolute tolerance `1e-12`, relative tolerance zero |
| Output | `results/ppo_intent_v1_online_diagnostics_seed10042.json` |

CPU is explicitly frozen because the original M8 evaluation machine reported no available CUDA device. The new runner verifies every supplied SHA-256 before rollout, validates the reference schema and row count, rejects an existing output path, and requires its recreated episode metrics to match all 500 committed M8B CSV rows. A mismatch marks the run `failed`, saves the partial evidence and reason to the JSON, and raises an error instead of silently accepting a different trajectory.

### 42.4 Predefined interpretation gates

The following routing rules were fixed before the 500-episode diagnostic:

| Gate | Fixed rule | Interpretation if failed |
|---|---|---|
| Episode reproduction | All 500 reference rows match | Invalid diagnostic run; investigate reproducibility only |
| Prediction coverage | At least 70% of visible-vehicle slots | History continuity or slot churn is the primary limitation; do not retrain PPO |
| Label availability | At least 99% of predictions have a hidden label | Simulator/instrumentation mismatch; do not interpret classifier metrics |
| Class support | All three true classes observed | Otherwise classification comparison is inconclusive |
| Online accuracy | At least majority-class accuracy + 5 percentage points | Predictions provide insufficient improvement over the online class prior |
| Online macro F1 | At least 50% | Material on-policy classifier degradation if failed |
| Minimum class recall | At least 40% | One intent class is insufficiently recovered if failed |
| Offline-to-online accuracy drop | Less than 5 percentage points below 63.3549% | A larger drop flags distribution shift |

Routing is fixed as follows. Coverage below 70% directs the next experiment toward history/slot availability without PPO retraining. With adequate coverage, a failed classification or distribution-shift gate directs the next experiment toward collecting on-policy intent data. If coverage and classification both pass, the next experiment is an oracle-label upper-bound ablation to test whether the control problem can use intent before changing PPO architecture or budget.

These gates diagnose mechanism; they cannot retroactively promote the rejected M8 policy or establish statistical independence of repeated slot observations.

### 42.5 Implementation and verification

The implementation adds opt-in diagnostic snapshots to the existing intent wrapper and a standalone rollout accumulator/runner. Probability shape, finiteness, bounds, row sums, counter identities, class indices, confidence, entropy, and final decision/slot totals are validated before a result is accepted. The standard wrapper path remains unchanged when diagnostics are disabled.

Required pre-experiment checks completed:

```text
git diff --check: passed
ruff check: passed
pytest: 50 passed in 15.68 s
```

A paired three-episode engineering smoke used seeds 10042–10044 and both diagnostic-disabled and diagnostic-enabled environments:

| Engineering check | Result |
|---|---:|
| Episodes | 3 |
| Policy decisions | 129 |
| Slot observations | 645 |
| Observations/actions/rewards/termination identical at every step | Passed |
| Recreated episode metrics match the first three committed CSV rows | Passed |
| History-ready predictions | 268 |
| Smoke prediction coverage | 45.8904% |
| Smoke labeled-prediction rate | 100% |

The low three-episode coverage is useful engineering evidence that the full diagnostic is necessary, but it is not used as the 500-episode result and does not alter the frozen thresholds.

### 42.6 Frozen command

Run exactly once after pulling this implementation and rerunning the complete checks:

```powershell
python scripts/diagnose_intent_rollout.py --model models/ppo_intent_v1_seed42.zip --model-sha256 954a1d4ef9431ca451de367213d6b65c087d11f277802f9d7bc1ac38c47471e8 --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --intent-model models/intent_gru_seed42.pt --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 --intent-neighbors 5 --intent-device cpu --episodes 500 --seed 10042 --unsafe-ttc 2.0 --reference-csv results/ppo_intent_v1_holdout_seed10042.csv --reference-csv-sha256 72fe15fb876e5ad16007c97e01a8811aa62c5935468c21dac49df8edcdebb498 --output results/ppo_intent_v1_online_diagnostics_seed10042.json
```

Do not rerun, retrain, tune thresholds, or change inputs after observing the result. Commit only the small diagnostic JSON; keep both model checkpoints local.

### 42.7 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-036 | Freeze a non-interventional online diagnostic before any further training | Train longer immediately or tune the rejected M8 run | M8 had essentially zero paired task effect and several unresolved mechanisms | Retained |
| D-037 | Require exact reproduction of the committed M8B episode metrics | Accept aggregate-only similarity | Prevent diagnostic instrumentation or device changes from silently changing the evaluated trajectories | Retained |
| D-038 | Use decision-slot-weighted coverage and classification with explicit routing gates | Inspect ad hoc metrics and decide thresholds after the run | Preserve the diagnostic value of the holdout and separate coverage, shift, and control-use hypotheses | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Implemented and froze the M8C online-intent diagnostic | Identify the failure mechanism of rejected M8 without intervention or retraining | 50 tests passed; paired three-episode non-interference and reference-reproduction smoke passed | This implementation update |

**Next action:** run only the frozen 500-episode diagnostic command, commit its JSON whether its status is `complete` or `failed`, and append the result before designing another experiment.


---

## 43. Milestone M8C result — online intent is accurate but coverage-limited

**Experiment ID:** E-M8C-INTENT-DIAGNOSTIC-V1-H10042

**Status:** Completed; coverage gate failed and all online-classification gates passed

**Date recorded:** 2026-09-05

**Result commit:** [`cd1163e`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/cd1163ea4b8c847c34ddc3794f68817a0aab929b)

**Result artifact:** `results/ppo_intent_v1_online_diagnostics_seed10042.json`

**Artifact SHA-256:** `c39ffdc93c6f46ab75b4624b8b48ec04caab877ef310bbf14148e1d9504e4045`

### 43.1 Integrity and reference reproduction

The committed JSON reports `status: complete`, no error, all 500 prescribed episodes, and seeds 10042–10541. Its PPO, GRU, configuration, and reference-CSV SHA-256 values exactly match the frozen Section 42 protocol. CPU inference, five intent neighbors, no safety shield, and the 2.0-second unsafe-TTC reporting threshold are also unchanged.

The diagnostic reproduced every committed M8B episode row within the predefined exact/`1e-12` comparison. Its driving summary is therefore unchanged: 59.0% success, 41.0% collision, 0% incomplete, mean reward 2.0107, and 19,178 total decisions.

Independent count and confusion-matrix checks passed:

```text
5 * 19,178 decisions = 95,890 slot observations
9,647 missing + 0 obstacle + 86,243 vehicle = 95,890 slots
48,513 warmup + 37,730 predicted = 86,243 vehicle slots
37,730 confusion-matrix samples = 37,730 labeled predictions
24,461 diagonal predictions / 37,730 = 64.8317% accuracy
```

### 43.2 Observed online coverage and classifier behavior

| Metric | Observed |
|---|---:|
| Policy decisions | 19,178 |
| Slot observations | 95,890 |
| Vehicle slots | 86,243 (89.94% of all slots) |
| Warmup vehicle slots | 48,513 (56.25% of vehicle slots) |
| History-ready predicted slots | 37,730 |
| Prediction coverage | 43.7485% |
| Labeled-prediction rate | 100.0% |
| Mean confidence | 72.3280% |
| Mean normalized entropy | 55.4071% |
| Online accuracy | 64.8317% |
| Online majority-class accuracy | 46.6393% |
| Accuracy advantage over majority | +18.1924 pp |
| Offline held-out accuracy | 63.3549% |
| Online minus offline accuracy | +1.4768 pp |
| Balanced accuracy / macro recall | 63.7375% |
| Macro F1 | 64.0461% |

Per-class performance on the correlated, decision-slot-weighted online samples:

| True class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Cautious | 11,274 | 74.96% | 54.35% | 63.01% |
| Normal | 17,597 | 63.96% | 71.06% | 67.32% |
| Aggressive | 8,859 | 58.27% | 65.81% | 61.81% |

All three classes were observed. The minimum class recall was cautious at 54.35%.

### 43.3 Predefined gate decisions

| Gate | Frozen rule | Observed | Decision |
|---|---:|---:|---|
| Episode reproduction | All 500 rows match | All matched | Passed |
| Prediction coverage | At least 70% | 43.7485% | **Failed** |
| Label availability | At least 99% | 100.0% | Passed |
| Class support | All three classes | All three | Passed |
| Accuracy over majority | At least +5 pp | +18.1924 pp | Passed |
| Online macro F1 | At least 50% | 64.0461% | Passed |
| Minimum class recall | At least 40% | 54.3463% | Passed |
| Offline-to-online accuracy drop | Less than 5 pp | No drop; online was +1.4768 pp | Passed |

Only the coverage gate failed.

### 43.4 Interpretation and decision

The frozen classifier does not show evidence of harmful online distribution shift on the history-ready slots. Its online accuracy slightly exceeded the offline held-out result, beat the online majority baseline by 18.19 percentage points, and passed macro-F1 and every class-recall requirement. Because observations from the same vehicles repeat across decisions, these results support mechanism diagnosis rather than an independent-sample significance claim.

The dominant limitation is availability: 56.25% of visible-vehicle slots were still in warmup and received uniform probabilities, leaving only 43.75% with GRU predictions. This is consistent with the current wrapper retaining history only while a vehicle remains among the five selected neighbors; it purges history whenever the vehicle leaves that set. The V3 base observation contains more traffic rows than the five intent-output slots, so useful prior observations may be discarded before a vehicle enters the intent slots.

Per the frozen routing rule, do not retrain PPO or the GRU yet. First run a non-interventional shadow diagnostic that keeps histories for every traffic slot already present in the base kinematics observation while leaving the actual M8 policy observation and action path untouched. It should compare counterfactual history-ready coverage and classifier quality with the current five-slot tracker on the same decisions. This tests whether decoupling history tracking from the five appended intent slots can exceed 70% coverage without using information outside the existing observation interface.

Because seeds 10042–10541 have now informed design decisions, any later policy claiming improvement must use a newly frozen untouched holdout and a paired V3 evaluation on those new seeds.

**Decision:** accept M8C as a valid coverage-limited diagnostic result. Retain PPO V3 as the best policy and retain PPO + intent V1 as rejected. Do not change history length, retrain either model, add the rejected shield, or promote online accuracy alone as a driving improvement.

### 43.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-039 | Diagnose M8 as coverage-limited, not classifier-distribution-limited | Attribute the null driving effect to online GRU failure | Coverage was 43.75%, while every frozen classification and integrity gate passed | Retained |
| D-040 | Run a shadow all-observed-slot history feasibility diagnostic before retraining | Immediately shorten history, retrain the GRU/PPO, or track simulator vehicles outside the observation interface | The base observation already exposes additional traffic slots whose histories can be retained without affecting actions | Retained |
| D-041 | Reserve a new untouched holdout for any later policy claim | Continue tuning and testing new policies on seeds 10042–10541 | The current holdout has now directly informed representation design | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Completed and accepted the M8C online-intent diagnostic | Distinguish coverage, distribution-shift, and control-use explanations for rejected M8 | Exact 500-row reproduction, frozen hashes/settings, count identities, confusion matrix, and all predefined gates independently checked | Result: `cd1163e`; documentation: this update |

**Next action:** implement and freeze the non-interventional shadow history-coverage diagnostic described above; do not start PPO or GRU training yet.


---

## 44. Milestone M8D design — all-observed-slot shadow history feasibility

**Experiment ID:** E-M8D-SHADOW-HISTORY14-H10042

**Status:** Implemented and frozen; 500-episode diagnostic pending

**Date recorded:** 2026-09-05

**Primary diagnostic factor:** retain GRU histories for all 14 non-ego traffic slots already exposed by the V3 kinematics observation, while continuing to append the unchanged original M8 probabilities for only the first five slots

**Policy intervention:** none

### 44.1 Question and implementation boundary

M8C found 43.7485% prediction coverage but passing online classifier quality. The current wrapper uses `intent_neighbors=5` for both output and history retention, even though the frozen base observation has 15 rows: one ego row and 14 sorted traffic rows. When a vehicle leaves the first five, its history is deleted; if it later returns, its ten-step warmup restarts.

M8D asks whether decoupling **history tracking** from the five **intent output slots** can provide at least 70% prediction coverage. A separate shadow store tracks the same six intent features for up to 14 sorted traffic rows inside the existing perception/observation interface. At every pre-action decision it measures whether each of the original five target slots would be history-ready under that store.

The actual M8 store, probabilities appended to the 120-value PPO observation, deterministic PPO action, reward, environment state, and random-number flow remain unchanged. Shadow probabilities and hidden labels are written only to diagnostics and never enter the policy. Vehicles outside the 14 existing traffic rows are not tracked, so this is not an oracle or expanded sensor-range experiment.

### 44.2 Paired readiness accounting

Each visible-vehicle target slot is assigned to exactly one category:

| Category | Meaning |
|---|---|
| `both_ready` | Current five-slot and shadow 14-slot histories are ready |
| `current_only` | Current history is ready but shadow is not |
| `shadow_only` | Shadow history recovers a slot still in current warmup |
| `neither_ready` | Neither history is ready |

The implementation validates:

$$
N_{vehicle}=N_{both}+N_{current\ only}+N_{shadow\ only}+N_{neither}
$$

and reports:

$$
coverage_{current}=\frac{N_{both}+N_{current\ only}}{N_{vehicle}}
$$

$$
coverage_{shadow}=\frac{N_{both}+N_{shadow\ only}}{N_{vehicle}}
$$

$$
recovery_{warmup}=\frac{N_{shadow\ only}}{N_{shadow\ only}+N_{neither}}
$$

The shadow must be a readiness superset: `current_only` must equal zero. Slot identities, vehicle rows, predicted rows, and current/shadow count identities are checked before accumulation.

### 44.3 Frozen protocol and provenance

| Property | Frozen value |
|---|---|
| PPO checkpoint and SHA-256 | `models/ppo_intent_v1_seed42.zip`; `954a1d4e...71e8` |
| Intent checkpoint and SHA-256 | `models/intent_gru_seed42.pt`; `10483649...bb05` |
| Configuration and SHA-256 | `configs/intersection_reward_v3.yaml`; `433e6972...ae69` |
| Driving reference and SHA-256 | `results/ppo_intent_v1_holdout_seed10042.csv`; `72fe15fb...b498` |
| Intent reference and SHA-256 | `results/ppo_intent_v1_online_diagnostics_seed10042.json`; `c39ffdc9...4045` |
| Episodes and seeds | 500; 10042–10541 |
| Policy/intent device | CPU / CPU |
| Policy mode | Deterministic |
| Intent output slots | 5, unchanged |
| Shadow history slots | 14 |
| History length and features | 10; unchanged six-feature sequence |
| Safety shield | Disabled |
| Unsafe-TTC reporting threshold | 2.0 seconds |
| Output | `results/ppo_intent_v1_shadow_history_diagnostics_seed10042.json` |

The runner verifies every input hash, refuses to overwrite existing evidence, exactly reproduces all 500 driving rows, and reproduces the complete original `online_intent` block with numeric absolute tolerance `1e-12` and zero relative tolerance. A failure preserves its partial JSON and raises.

### 44.4 Frozen feasibility gates and routing

| Gate | Required result |
|---|---|
| Driving reference | All 500 M8B episode rows reproduced |
| Original diagnostic reference | Complete M8C `online_intent` block reproduced |
| Paired readiness | `current_only = 0` and every vehicle slot accounted for |
| Shadow prediction coverage | At least 70% |
| Shadow labeled-prediction rate | At least 99% |
| Shadow class support | All three true classes observed |
| Shadow accuracy advantage | At least 5 pp over its own majority-class accuracy |
| Shadow macro F1 | At least 50% |
| Shadow minimum class recall | At least 40% |
| Shadow accuracy versus offline | Less than a 5 pp drop from the frozen 63.3549% offline accuracy |

If every gate passes, wider history retention is feasible and the next step is to design a new PPO + intent experiment using 14-slot history tracking, the same five probability outputs, and a newly frozen untouched holdout with paired V3 evaluation. Passing M8D does not authorize training by itself.

If coverage remains below 70%, wider retention is insufficient; do not retrain PPO. The next development experiment should study a separately trained shorter-history classifier without reusing these seeds for a final policy claim. If coverage passes but classifier quality fails, do not train PPO; investigate the newly covered temporal population and on-policy intent data first.

### 44.5 Engineering verification

The complete implementation gate passed:

```text
git diff --check: passed
ruff check: passed
pytest: 63 passed in 3.84 s
```

The first Ruff attempt reported only an unsorted test import block. It was corrected before any real-environment smoke or experiment, and the subsequent complete gate passed.

A real three-episode engineering replay on seeds 10042–10044 produced:

| Check | Result |
|---|---:|
| Driving reference reproduced | Passed |
| Original diagnostic block reproduced | Passed |
| Policy decisions | 129 |
| Vehicle target slots | 584 |
| Current prediction coverage | 45.8904% |
| Shadow prediction coverage | 65.2397% |
| Absolute smoke coverage gain | +19.3493 pp |
| Current-ready/shadow-not-ready slots | 0 |
| Current warmup slots recovered by shadow | 113/316 (35.7595%) |

The smoke shows correct non-interference, paired accounting, and a plausible coverage gain. Its three episodes are not M8D research evidence and do not change the frozen 70% gate.

### 44.6 Frozen command

After pulling this implementation and rerunning the complete checks, execute exactly once:

```powershell
python scripts/diagnose_intent_rollout.py --model models/ppo_intent_v1_seed42.zip --model-sha256 954a1d4ef9431ca451de367213d6b65c087d11f277802f9d7bc1ac38c47471e8 --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --intent-model models/intent_gru_seed42.pt --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 --intent-neighbors 5 --shadow-history-neighbors 14 --intent-device cpu --episodes 500 --seed 10042 --unsafe-ttc 2.0 --reference-csv results/ppo_intent_v1_holdout_seed10042.csv --reference-csv-sha256 72fe15fb876e5ad16007c97e01a8811aa62c5935468c21dac49df8edcdebb498 --reference-diagnostic-json results/ppo_intent_v1_online_diagnostics_seed10042.json --reference-diagnostic-json-sha256 c39ffdc93c6f46ab75b4624b8b48ec04caab877ef310bbf14148e1d9504e4045 --output results/ppo_intent_v1_shadow_history_diagnostics_seed10042.json
```

Commit only the small JSON whether the run completes or fails. Do not commit either checkpoint, rerun after seeing the result, retrain, or change any threshold.

### 44.7 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-042 | Test 14-slot history retention only as a non-interventional shadow | Apply new probabilities directly to rejected M8 or immediately retrain | Isolate coverage feasibility while preserving every original action and result | Retained |
| D-043 | Require both driving and complete M8C diagnostic reproduction | Compare only aggregate coverage | Detect any policy-path, seed, or diagnostic drift before interpreting the shadow | Retained |
| D-044 | Require at least 70% coverage plus the existing classifier-quality gates | Select a threshold after seeing the 500-episode result | Preserve the predefined mechanism test | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Implemented and froze the M8D shadow history feasibility diagnostic | Test whether histories from all already-observed traffic rows resolve the M8 coverage bottleneck without intervention | 63 tests passed; paired three-episode real-environment smoke reproduced driving and original intent diagnostics | This implementation update |

**Next action:** run only the frozen M8D command and commit its JSON; do not start training.


---

## 45. Milestone M8D result — wider tracking improves but does not solve coverage

**Experiment ID:** E-M8D-SHADOW-HISTORY14-H10042

**Status:** Completed; shadow coverage feasibility rejected

**Date recorded:** 2026-09-05

**Result commit:** [`7d49dc2`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/7d49dc245614b356fc900a245890ebea503bbdb5)

**Result artifact:** `results/ppo_intent_v1_shadow_history_diagnostics_seed10042.json`

**Artifact SHA-256:** `d22140fa855aab867978ebd439d222a7622d3c97ed94fcf549cc89b623f2a844`

### 45.1 Integrity validation

The committed result reports `status: complete`, no error, 500 completed episodes, seeds 10042–10541, 14 shadow-history slots, five unchanged intent-output slots, CPU inference, no shield, and all frozen input fingerprints. Both mandatory reference checks passed:

- all 500 M8B driving rows reproduced;
- the complete M8C `online_intent` block reproduced.

The driving outcome consequently remains the rejected M8 result: 59.0% success, 41.0% collision, and 0% incomplete. M8D did not evaluate a changed driving policy.

Independent paired-count checks:

```text
37,730 both ready
     0 current only
17,364 shadow only
31,149 neither ready
------
86,243 vehicle target slots

37,730 + 17,364 = 55,094 shadow predictions
34,453 confusion-matrix diagonal / 55,094 = 62.5349% accuracy
```

### 45.2 Coverage comparison

| Metric | Current five-slot history | Shadow 14-slot history | Change |
|---|---:|---:|---:|
| History-ready predictions | 37,730 | 55,094 | +17,364 |
| Warmup vehicle slots | 48,513 | 31,149 | -17,364 |
| Prediction coverage | 43.7485% | 63.8823% | +20.1338 pp |
| Labeled-prediction rate | 100% | 100% | 0 pp |

The shadow was a strict readiness superset: `current_only=0`. It recovered 17,364 of the current tracker's 48,513 warmup slots, or 35.7925%. However, 31,149 visible-vehicle target slots still lacked ten observations even when histories were retained across all 14 traffic rows already present in the base observation.

### 45.3 Shadow classifier behavior

| Metric | Observed |
|---|---:|
| Samples | 55,094 |
| Accuracy | 62.5349% |
| Majority-class accuracy | 45.3062% |
| Accuracy advantage over majority | +17.2287 pp |
| Offline held-out accuracy | 63.3549% |
| Shadow online minus offline accuracy | -0.8200 pp |
| Balanced accuracy / macro recall | 61.7190% |
| Macro F1 | 62.0420% |
| Mean confidence | 70.8727% |
| Mean normalized entropy | 57.4545% |

Per-class shadow metrics:

| True class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Cautious | 16,922 | 75.70% | 52.42% | 61.94% |
| Normal | 24,961 | 60.46% | 68.48% | 64.22% |
| Aggressive | 13,211 | 56.20% | 64.26% | 59.96% |

All three classes were observed; cautious recall was the minimum at 52.42%.

### 45.4 Frozen gate decisions

| Gate | Requirement | Observed | Decision |
|---|---:|---:|---|
| Driving reference | All 500 rows | Reproduced | Passed |
| Original intent reference | Complete M8C block | Reproduced | Passed |
| Readiness superset | `current_only=0` | 0 | Passed |
| Shadow prediction coverage | At least 70% | 63.8823% | **Failed** |
| Label availability | At least 99% | 100% | Passed |
| Class support | All three | All three | Passed |
| Accuracy over majority | At least +5 pp | +17.2287 pp | Passed |
| Macro F1 | At least 50% | 62.0420% | Passed |
| Minimum class recall | At least 40% | 52.4170% | Passed |
| Accuracy drop from offline | Less than 5 pp | 0.8200 pp | Passed |

Only the 70% shadow-coverage gate failed, by 6.1177 percentage points.

### 45.5 Interpretation and decision

Retaining histories for all already-observed traffic rows materially improves availability and preserves useful classifier quality, so the five-slot purge behavior was a real contributor. It is not the complete cause: more than one third of vehicle target slots remain below the ten-step requirement even with wider tracking. This is consistent with vehicles entering the observable set late relative to the short 38.356-decision mean episode, but M8D does not by itself identify an optimal shorter history.

Per the predefined Section 44 routing, do not retrain PPO. Wider 14-slot retention alone failed its feasibility gate. The next controlled development experiment should derive shorter fixed-length histories from the accepted, episode-split intent dataset; select the length using training/validation data and a separately frozen coverage analysis; and evaluate the chosen classifier once on the existing held-out intent test split. No current policy holdout result may be used as a final claim.

A shorter-history model must pass the existing classifier standards and a predefined coverage target before a new PPO + intent experiment is authorized. Any eventual driving-policy comparison must evaluate both V3 and the new policy on a newly frozen untouched holdout, because seeds 10042–10541 have informed representation design.

**Decision:** accept M8D as a valid negative feasibility result. Retain its +20.13 pp coverage improvement as evidence, reject 14-slot retention alone as sufficient, keep PPO V3 as the best policy, and keep PPO + intent V1 rejected.

### 45.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-045 | Reject wider 14-slot tracking alone as the next PPO representation | Retrain PPO despite missing the frozen coverage gate | Coverage improved to 63.88% but remained below 70% | Retained |
| D-046 | Develop a shorter-history intent model before any new PPO run | Increase PPO budget, change reward, or accept uniform warmup values | Online classifier quality passed while ten-step availability remained the sole failed gate | Retained |
| D-047 | Select shorter history using grouped development evidence and reserve a new policy holdout | Choose a length directly from future final-policy outcomes | Prevent further test-guided policy selection | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Completed and rejected M8D as a sufficient coverage solution | Test wider history retention without changing M8 actions | Both references, artifact hashes, paired readiness counts, confusion matrix, and all frozen gates independently verified | Result: `7d49dc2`; documentation: this update |

**Next action:** audit and freeze the shorter-history classifier development protocol; do not train PPO or a new GRU until that protocol and its gates are recorded.


---

## 46. Milestone M9A design — shorter-history coverage selection

**Experiment ID:** E-M9A-HISTORY-CURVE-H10042

**Status:** Implemented and frozen; 500-episode coverage curve pending

**Date recorded:** 2026-09-05

**Research role:** development-only selection of one shorter history length before any new classifier training

**Policy intervention:** none

### 46.1 Leakage-control decision

Training multiple shorter-history GRUs and choosing among them by held-out test accuracy would reuse the intent test split for model selection. M9A instead selects a single history length using only a coverage curve on the already-designated development trajectories. The chosen GRU will later select its epoch on grouped validation episodes and receive one held-out intent test evaluation.

The M8 policy trajectories at seeds 10042–10541 are no longer considered untouched because M8C and M8D informed representation design. M9A may use them for development-only coverage selection, but no later policy success/collision claim may use them as its final holdout.

### 46.2 Frozen candidates and selection rule

Candidate history lengths are:

```text
4, 5, 6, 7, 8, 9, 10 policy observations
```

At the frozen 5 Hz policy frequency, four observations span three transitions, or 0.6 seconds. Lengths 1–3 are excluded because they provide at most two temporal transitions and the first velocity-difference feature is initialized to zero. Length 10 retains the current two-second collection convention and anchors the curve to M8D.

For each target vehicle slot, M9A records the current length of its 14-slot shadow history, capped at ten. For candidate $L$:

$$
coverage(L)=\frac{\#\{\text{vehicle target slots with shadow history length}\ge L\}}{N_{vehicle}}
$$

The selection is fixed before the run:

$$
L^*=\max\{L\in\{4,5,6,7,8,9,10\}: coverage(L)\ge0.70\}
$$

Choosing the longest passing window preserves the most temporal context while meeting the existing 70% availability requirement. If no candidate passes, M9A is inconclusive and no shorter-history GRU is authorized.

This rule uses coverage only. Hidden labels, classifier accuracy, PPO outcomes, and the three-episode engineering smoke cannot change the selected length.

### 46.3 Frozen reference chain and output

| Input | SHA-256 |
|---|---|
| PPO + intent V1 checkpoint | `954a1d4ef9431ca451de367213d6b65c087d11f277802f9d7bc1ac38c47471e8` |
| Intent GRU checkpoint | `10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05` |
| V3 configuration | `433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69` |
| M8B driving CSV | `72fe15fb876e5ad16007c97e01a8811aa62c5935468c21dac49df8edcdebb498` |
| M8C online diagnostic | `c39ffdc93c6f46ab75b4624b8b48ec04caab877ef310bbf14148e1d9504e4045` |
| M8D shadow diagnostic | `d22140fa855aab867978ebd439d222a7622d3c97ed94fcf549cc89b623f2a844` |

The output is `results/ppo_intent_v1_history_coverage_curve_seed10042.json`. It must report 500 episodes, seeds 10042–10541, five intent-output slots, 14 shadow-history slots, candidates 4–10, minimum coverage 0.70, CPU policy/intent inference, no shield, and the unchanged 2.0-second unsafe-TTC reporting threshold.

Before accepting a curve, the runner must reproduce:

1. all 500 M8B driving rows;
2. the complete M8C `online_intent` block;
3. the M8D `online_intent`, `shadow_online_intent`, and `coverage_comparison` blocks.

Numeric reference comparison uses absolute tolerance `1e-12` and zero relative tolerance. Every input hash is checked, the output refuses overwrite, and a failed run preserves its partial JSON and raises.

### 46.4 Dataset and future classifier boundary

M9A does not read or transform `data/intent_trajectories_seed42.npz` and does not train a GRU. After $L^*$ is recorded, a separate implementation must:

- verify the accepted dataset SHA-256 `56433621bdcc5fe9a635f57f068e096a9cb3d47036179a64ab390311fab302b0`;
- derive each candidate input causally as the last $L^*$ chronological rows of its existing ten-step history;
- preserve all 109,596 labels, episode IDs, and the exact seed-42 grouped train/validation/test episode assignment;
- calculate normalization only from the projected training split;
- use seed 42, 30 epochs, batch size 128, learning rate 0.001, the unchanged GRU architecture, and best-validation-epoch selection;
- keep the held-out test metrics unavailable until the selected checkpoint and its validation result are frozen;
- save projection length, source fingerprint, split indices, and hyperparameters in the checkpoint/summary;
- keep the derived dataset and checkpoint local and commit only small JSON provenance/metrics.

These requirements are recorded now but do not authorize training before the M9A curve result is validated and appended.

### 46.5 Implementation and engineering verification

The shadow snapshot now exposes validated per-target history lengths. A new accumulator checks their count and bounds, constructs the complete length histogram and monotone coverage curve, and applies the frozen longest-passing selection rule. The diagnostic runner additionally pins and reproduces the M8D JSON before accepting the curve.

The first quick Ruff pass found only two overlong lines and import ordering in new tests. They were corrected before the complete gate or real-environment smoke.

Final implementation checks:

```text
git diff --check: passed
ruff check: passed
pytest: 76 passed in 3.22 s
```

A chained three-episode real-environment smoke on seeds 10042–10044 reproduced the driving, M8C, and M8D blocks. Its curve was:

| History length | Smoke coverage |
|---:|---:|
| 4 | 88.36% |
| 5 | 84.76% |
| 6 | 80.82% |
| 7 | 76.88% |
| 8 | 72.95% |
| 9 | 69.01% |
| 10 | 65.24% |

The smoke selected length 8 and verified all 584 vehicle target slots were represented in the length histogram. This is engineering evidence only and cannot determine the 500-episode result.

### 46.6 Frozen command

After pulling the implementation and rerunning all tests, execute exactly once:

```powershell
python scripts/diagnose_intent_rollout.py --model models/ppo_intent_v1_seed42.zip --model-sha256 954a1d4ef9431ca451de367213d6b65c087d11f277802f9d7bc1ac38c47471e8 --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --intent-model models/intent_gru_seed42.pt --intent-model-sha256 10483649f77416b33a8c6dda8dffbb80655194781bd50630f1a2bc4bc36abb05 --intent-neighbors 5 --shadow-history-neighbors 14 --intent-device cpu --episodes 500 --seed 10042 --unsafe-ttc 2.0 --reference-csv results/ppo_intent_v1_holdout_seed10042.csv --reference-csv-sha256 72fe15fb876e5ad16007c97e01a8811aa62c5935468c21dac49df8edcdebb498 --reference-diagnostic-json results/ppo_intent_v1_online_diagnostics_seed10042.json --reference-diagnostic-json-sha256 c39ffdc93c6f46ab75b4624b8b48ec04caab877ef310bbf14148e1d9504e4045 --reference-shadow-diagnostic-json results/ppo_intent_v1_shadow_history_diagnostics_seed10042.json --reference-shadow-diagnostic-json-sha256 d22140fa855aab867978ebd439d222a7622d3c97ed94fcf549cc89b623f2a844 --history-coverage-lengths 4,5,6,7,8,9,10 --history-coverage-minimum 0.70 --output results/ppo_intent_v1_history_coverage_curve_seed10042.json
```

Commit only the output JSON, whether complete or failed. Do not train a GRU or PPO, change candidates, or rerun after observing the curve.

### 46.7 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-048 | Select one shorter history by a coverage-only curve before training | Train several GRUs and choose by held-out test performance | Prevent test-split model selection and reduce unnecessary training runs | Retained |
| D-049 | Restrict candidates to lengths 4–10 and select the longest reaching 70% | Include single-snapshot histories or choose the shortest/highest-coverage window | Preserve meaningful temporal context while satisfying the established availability gate | Retained |
| D-050 | Require the full M8B→M8C→M8D reference chain | Trust aggregate similarity or rerun without pinned prior diagnostics | Ensure the only new evidence is the history-length curve | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Implemented and froze the M9A history-length coverage selector | Choose one shorter temporal window without using intent test results | 76 tests passed; three-episode chained reference-reproduction smoke passed | This implementation update |

**Next action:** run only the frozen M9A curve command and commit its JSON; do not train a new GRU yet.


---

## 47. Milestone M9A result — eight-observation history selected

**Experiment ID:** E-M9A-HISTORY-CURVE-H10042

**Status:** Complete; history length 8 selected

**Date recorded:** 2026-09-05

**Result commit:** [`837c7b5`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/837c7b53412239adb5748c8bc73363af49f9a156)

**Result artifact:** `results/ppo_intent_v1_history_coverage_curve_seed10042.json`

**Artifact SHA-256:** `b364e39e104197b32670c9305806c74c972be5fa9e5497888ecd340d664be101`

### 47.1 Integrity validation

The committed result reports `status: complete`, no error, 500 completed episodes, and the exact frozen seed interval 10042–10541. It used the pinned PPO + intent V1 checkpoint, original ten-step intent GRU, V3 configuration, five policy-facing intent slots, 14-slot shadow tracker, CPU inference, no safety shield, and the unchanged 2.0-second unsafe-TTC reporting threshold.

All three mandatory reference checks passed:

- all 500 M8B driving rows reproduced;
- the complete M8C online-intent result reproduced;
- the M8D current-intent, shadow-intent, and coverage-comparison blocks reproduced.

Consequently, M9A did not change or reevaluate a driving policy. The reproduced outcome remains 59.0% success, 41.0% collision, and 0% incomplete for the already-rejected PPO + intent V1 policy.

Independent validation accounted for all observations:

```text
3,899 + 3,721 + 3,604 + 3,520 + 3,425
+ 3,364 + 3,279 + 3,202 + 3,135 + 55,094
= 86,243 vehicle target slots
```

For every candidate, independently summing histogram bins at or above that length exactly reproduced the reported ready count and coverage. Coverage was monotone decreasing as history length increased.

### 47.2 Frozen coverage curve

| History length | Ready slots | Warmup slots | Coverage | 70% gate |
|---:|---:|---:|---:|---|
| 4 | 75,019 | 11,224 | 86.9856% | Passed |
| 5 | 71,499 | 14,744 | 82.9041% | Passed |
| 6 | 68,074 | 18,169 | 78.9328% | Passed |
| 7 | 64,710 | 21,533 | 75.0322% | Passed |
| **8** | **61,431** | **24,812** | **71.2301%** | **Passed; selected** |
| 9 | 58,229 | 28,014 | 67.5174% | Failed |
| 10 | 55,094 | 31,149 | 63.8823% | Failed |

Eligible lengths were 4, 5, 6, 7, and 8. The independently recomputed longest passing candidate is 8, matching the artifact. Length 8 cleared the frozen gate by 1.2301 percentage points; length 9 missed it by 2.4826 percentage points. Relative to the original ten-observation requirement, length 8 makes 6,337 additional visible-vehicle slots ready, a 7.3478-point absolute coverage gain.

### 47.3 Interpretation and decision

The full run confirms the engineering-smoke indication without using classifier labels, accuracy, or driving outcomes to select the window. Eight observations are sufficient to cross the existing availability gate while preserving more temporal context than every other passing candidate. This result selects an input representation only; it is not evidence that an eight-step classifier will satisfy the frozen quality gates or improve PPO.

Exactly one shorter-history classifier is now authorized for development: use the causal suffix `x[:, -8:, :]` of every accepted ten-step trajectory. Preserve all 109,596 samples, labels, episode IDs, and the seed-42 grouped episode split. Recompute normalization from the projected training split only, choose the checkpoint epoch using validation episodes only, and keep held-out test metrics unavailable until the checkpoint and validation selection are frozen.

Do not train PPO. If the eight-step classifier later passes its predefined offline and coverage gates, a separate PPO experiment must use a newly frozen policy holdout; seeds 10042–10541 are development evidence and cannot support the final driving comparison.

**Decision:** accept M9A, freeze history length 8 for the next classifier experiment, keep PPO V3 as the best driving policy, and keep PPO + intent V1 rejected.

### 47.4 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-051 | Freeze an eight-observation input for the next intent classifier | Select a shorter passing length for higher availability or retain ten observations | Eight was the longest candidate meeting the predefined 70% gate | Retained |
| D-052 | Derive inputs causally as `x[:, -8:, :]` while preserving samples and grouped splits | Recollect data, change labels, or resplit episodes | Isolate history length as the only representation change and prevent split leakage | Retained |
| D-053 | Keep test evaluation sealed until validation selects and freezes the checkpoint | Inspect test metrics during training or compare multiple lengths on test | Preserve one valid held-out classifier evaluation after development selection | Retained |
| D-054 | Do not retrain PPO from the M9A result alone | Treat coverage feasibility as classifier or driving-policy evidence | M9A did not train an eight-step GRU or intervene on actions | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Completed M9A and selected history length 8 | Apply the frozen longest-window-at-70% rule before one classifier training run | All references reproduced; histogram, candidate arithmetic, monotonicity, and selection independently verified | Result: `837c7b5`; documentation: this update |

**Next action:** implement and freeze a leakage-safe eight-step GRU training and one-time evaluation protocol; do not train the GRU or PPO before that implementation is reviewed and recorded.


---

## 48. Milestone M9B design — sealed eight-observation GRU training

**Experiment ID:** E-M9B-GRU-H8-S42-DATA56433621

**Status:** Implemented and frozen; one local training run pending

**Date recorded:** 2026-09-05

**Research role:** train exactly one M9A-selected classifier while preserving a one-time held-out intent test

**Policy intervention:** none

### 48.1 Frozen inputs and single permitted projection

M9A selected history length 8 using coverage only. M9B therefore trains exactly one classifier using the accepted M7 archive; it does not compare history lengths, recollect trajectories, relabel examples, or change the grouped split.

| Item | Frozen value |
|---|---|
| Source dataset | `data/intent_trajectories_seed42.npz` |
| Dataset SHA-256 | `56433621bdcc5fe9a635f57f068e096a9cb3d47036179a64ab390311fab302b0` |
| Source shape | 109,596 × 10 × 6 |
| Label counts | 32,631 cautious; 49,213 normal; 27,752 aggressive |
| Projection | Causal suffix `x[:, -8:, :]` |
| Projected shape | 109,596 × 8 × 6 |
| Split seed and mode | 42; episode-grouped 70/15/15 |

The eight rows remain in their original chronological order. Labels and episode IDs are unchanged. No derived dataset is written, which avoids a second mutable data artifact.

The unchanged split implementation produces these frozen partitions:

| Split | Episodes | Samples | Index-array SHA-256 |
|---|---:|---:|---|
| Training | 210 | 76,761 | `85dc4bb43fcf7cc98f38bc8e1bc5affdb1d8d103f884e85ed7013bfe84a43d26` |
| Validation | 45 | 16,912 | `aaa8cd596d5cce5c323d2efb23f44c5207370492dbab152787705942daeff5bf` |
| Test | 45 | 15,923 | `eec6da2ec3820cd9a887b909e769349eb65295888c1860c66cd799e1ea08a6b6` |

The three partitions cover all 109,596 samples exactly once and contain disjoint episode IDs. Feature mean and standard deviation are recomputed after projection using only the 76,761 training samples.

### 48.2 Frozen optimization and checkpoint selection

| Setting | Value |
|---|---:|
| Input features | 6 |
| GRU hidden size / layers | 64 / 1 |
| Classifier head | LayerNorm → Linear(64,32) → ReLU → Linear(32,3) |
| Loss / optimizer | Cross entropy / Adam |
| Epochs | 30 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Seed | 42 |
| Device | CPU |
| Selection | Highest validation accuracy; earliest epoch wins ties |

The architecture, loss, optimizer, epoch budget, batch size, learning rate, and seed match the accepted original GRU. CPU is explicit for this run so device selection cannot change silently. The output paths are `models/intent_gru_h8_seed42.pt` and `results/intent_gru_h8_seed42.training.json`; both refuse overwrite.

After the best epoch is selected, M9B computes complete metrics on validation episodes only. The checkpoint and summary must pass all three development gates before the held-out evaluator is authorized:

| Validation gate | Requirement |
|---|---:|
| Accuracy advantage over validation majority class | At least +5 pp |
| Macro F1 | At least 50% |
| Minimum per-class recall | At least 40% |

A failed validation gate is retained as a negative result and ends M9B without evaluating the test split.

### 48.3 Held-out test remains sealed

With `--seal-test`, training stores the deterministic test indices for provenance but never constructs a test loader, predicts test labels, calculates test accuracy, or writes test metrics. The checkpoint contains `test_metrics_sealed: true` and omits `test_accuracy`; the JSON summary contains `test_accuracy: null`.

The summary also records the dataset and checkpoint SHA-256 values, source and projected history lengths, projection rule, all hyperparameters, split sample counts, split-index fingerprints, split episode IDs, best epoch, complete validation metrics, and validation-gate decision. The checkpoint remains local and ignored by Git; only the small summary is committed.

If the validation gate passes and the checkpoint/summary are verified, exactly one held-out test evaluation will be frozen using the generated checkpoint hash. Its acceptance rules are fixed now, before training:

| Final classifier gate | Requirement |
|---|---:|
| M9A coverage | At least 70% |
| All classes observed | All three |
| Accuracy advantage over test majority class | At least +5 pp |
| Macro F1 | At least 50% |
| Minimum per-class recall | At least 40% |
| Accuracy drop from original GRU | No more than 5 pp below 63.3548954% |

The last requirement gives a minimum accepted accuracy of 58.3548954%. The evaluator binds the M9A coverage JSON and the new checkpoint by SHA-256, requires a sealed checkpoint, applies the causal suffix recorded in that checkpoint, refuses output overwrite, and records every gate. These rules cannot be relaxed after seeing training or test results.

Passing the classifier gates would authorize design of a new PPO + intent experiment; it would not establish a driving improvement. PPO V3 remains the best policy, and a later policy comparison must use newly frozen untouched seeds.

### 48.4 Implementation and verification

The training pipeline now supports explicit causal history projection, source-dataset fingerprint enforcement, CPU selection, overwrite refusal, a test-sealed development mode, checkpoint fingerprinting, compact provenance summaries, and validation classification gates. Checkpoint loading validates history metadata and rejects the contradictory combination of a sealed test with stored test accuracy.

The intent evaluator remains backward-compatible with the original checkpoint and can now evaluate projected histories from a sealed checkpoint. It supports dataset/model/coverage hashes, expected history length, CPU selection, overwrite refusal, the frozen classifier gates, and recording whether test metrics were sealed before evaluation.

Unit tests verify causal suffix selection, training-only normalization after projection, source-hash rejection, test-metric omission, summary/checkpoint provenance, history-length enforcement during inference, sealed-checkpoint evaluation, coverage binding, and overwrite refusal. Final checks:

```text
git diff --check: passed
ruff check: passed
pytest: 79 passed in 3.49 s
```

The accepted source archive was inspected without training. Its shape, class counts, 300 episode IDs, split sample counts, and all three split-index hashes match the frozen values above.

### 48.5 Frozen local training command

After pulling this implementation, first run:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and neither output path exists, run exactly once:

```powershell
python scripts/train_intent.py --data data/intent_trajectories_seed42.npz --data-sha256 56433621bdcc5fe9a635f57f068e096a9cb3d47036179a64ab390311fab302b0 --history-length 8 --epochs 30 --batch-size 128 --learning-rate 0.001 --seed 42 --device cpu --seal-test --minimum-validation-accuracy-advantage 0.05 --minimum-validation-macro-f1 0.50 --minimum-validation-class-recall 0.40 --output models/intent_gru_h8_seed42.pt --summary-output results/intent_gru_h8_seed42.training.json --refuse-overwrite
```

Commit only `results/intent_gru_h8_seed42.training.json`, whether its validation gate passes or fails. Keep `models/intent_gru_h8_seed42.pt` local. Do not run `evaluate_intent.py`, train PPO, change the command, or rerun after observing validation results.

### 48.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-055 | Train exactly one eight-observation GRU from causal suffixes | Recollect data or compare multiple lengths | M9A selected length 8 without classifier/test feedback | Retained |
| D-056 | Keep test predictions and metrics sealed during training | Preserve the legacy train-and-test-in-one command | Protect the sole held-out intent evaluation from development feedback | Retained |
| D-057 | Require validation accuracy advantage, macro F1, and minimum recall gates before test evaluation | Advance on validation accuracy alone | Detect majority-class or per-class failure before consuming the test split | Retained |
| D-058 | Freeze final classifier gates before M9B training | Set thresholds after observing validation or test results | Prevent result-dependent acceptance criteria | Retained |
| D-059 | Keep the new checkpoint local and commit only its summary | Commit the `.pt` file | Follow the artifact policy while retaining a fingerprinted research record | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Implemented and froze M9B sealed eight-step training and evaluation support | Train the M9A-selected representation without leaking held-out intent-test metrics | Dataset/split audit, Ruff, and 79 tests passed | This implementation update |

**Next action:** run only the frozen M9B training command after the full local test gate; commit its JSON summary and keep its checkpoint local. Do not evaluate the test split yet.


---

## 49. Milestone M9B training result — validation gate passed with test sealed

**Experiment ID:** E-M9B-GRU-H8-S42-DATA56433621

**Status:** Training complete; validation gate passed; held-out test evaluation authorized once

**Date recorded:** 2026-09-05

**Result commit:** [`b7c1abb`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/b7c1abb70ac20989a2861cdc4afdef997a087a8d)

**Training summary:** `results/intent_gru_h8_seed42.training.json`

**Summary SHA-256:** `a368a1140b5539a7d992db827a9353128187b9eae01ad9ce106a84bc63b03caf`

**Local checkpoint:** `models/intent_gru_h8_seed42.pt`

**Checkpoint SHA-256:** `74a72cf2b99b115bb5b4d55fdb350b55e20551876f6ba41be5266d8953b7fc05`

### 49.1 Integrity and leakage audit

The committed result contains only the small training JSON; the `.pt` checkpoint remains local and ignored. The summary reports `status: complete` and exactly reproduces every frozen M9B input and setting:

- dataset SHA-256 `56433621...02b0`;
- 109,596 source samples with shape 10 × 6 and the unchanged three class counts;
- causal suffix history length 8;
- seed 42 episode-grouped split;
- 30 epochs, batch size 128, learning rate 0.001, and CPU;
- best-validation-accuracy checkpoint selection;
- the three predefined validation thresholds.

The recorded split is unchanged:

| Split | Episodes | Samples | Index-array SHA-256 verification |
|---|---:|---:|---|
| Training | 210 | 76,761 | Matched `85dc4bb4...3d26` |
| Validation | 45 | 16,912 | Matched `aaa8cd59...f5bf` |
| Test | 45 | 15,923 | Matched `eec6da2e...a6b6` |

Independent checkpoint inspection, without making test predictions, verified:

- the local checkpoint SHA-256 exactly matches the committed summary;
- `history_length=8`, `source_history_length=10`, and `history_projection=causal_suffix`;
- feature mean and standard deviation exactly reproduce computation from `x[train_indices, -8:, :]` only;
- all three checkpoint index arrays match the frozen summary hashes;
- the 210/45/45 episode-ID sets are pairwise disjoint and cover episodes 0–299 exactly;
- the validation confusion matrix contains exactly 16,912 samples and reproduces the reported validation accuracy;
- `test_metrics_sealed=true`, `test_accuracy` is absent from the checkpoint, and the summary records `test_accuracy: null`.

No held-out test label was predicted and no held-out classifier metric was observed during this audit.

### 49.2 Validation result

The best checkpoint occurred at epoch 25.

| Metric | Observed |
|---|---:|
| Validation samples | 16,912 |
| Accuracy | 60.9804% |
| Majority-class accuracy | 46.3931% |
| Accuracy advantage | +14.5873 pp |
| Balanced accuracy / macro recall | 58.5882% |
| Macro precision | 62.8823% |
| Macro F1 | 59.9526% |

Per-class validation metrics:

| True class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Cautious | 5,483 | 59.02% | 61.52% | 60.24% |
| Normal | 7,846 | 59.80% | 66.77% | 63.09% |
| Aggressive | 3,583 | 69.83% | 47.47% | 56.52% |

Validation gate decisions:

| Gate | Requirement | Observed | Margin | Decision |
|---|---:|---:|---:|---|
| Accuracy over majority | At least +5 pp | +14.5873 pp | +9.5873 pp | Passed |
| Macro F1 | At least 50% | 59.9526% | +9.9526 pp | Passed |
| Minimum class recall | At least 40% | 47.4742% | +7.4742 pp | Passed |

The shorter-history model loses 2.4834 percentage points of validation accuracy relative to the original ten-step model's 63.4638% on the same frozen validation partition. This is development context only; the predefined gates passed and the held-out test remains the final classifier screen.

### 49.3 Interpretation and decision

The eight-observation representation retains meaningful three-class performance on the validation episodes while raising frozen online availability from 63.8823% at length 10 to 71.2301% at length 8. Aggressive recall remains the weakest class, but its 47.47% validation recall clears the predefined 40% floor.

M9B training is accepted as a development result. The checkpoint is now frozen by SHA-256 and exactly one evaluation on the existing held-out intent-test partition is authorized. The evaluator must bind both the new checkpoint and the M9A coverage artifact, enforce all Section 48 final gates, and refuse overwrite. The evaluation must be committed and documented whether it passes or fails.

Do not retrain either classifier, inspect alternative epochs, change thresholds, or train PPO. A classifier pass will authorize a separate PPO experiment design only; PPO V3 remains the best driving policy.

### 49.4 Frozen held-out evaluation command

After pulling this documentation update, first run:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and `results/intent_gru_h8_seed42.metrics.json` does not exist, run exactly once:

```powershell
python scripts/evaluate_intent.py --data data/intent_trajectories_seed42.npz --data-sha256 56433621bdcc5fe9a635f57f068e096a9cb3d47036179a64ab390311fab302b0 --model models/intent_gru_h8_seed42.pt --model-sha256 74a72cf2b99b115bb5b4d55fdb350b55e20551876f6ba41be5266d8953b7fc05 --expected-history-length 8 --require-sealed-test --device cpu --coverage-json results/ppo_intent_v1_history_coverage_curve_seed10042.json --coverage-json-sha256 b364e39e104197b32670c9305806c74c972be5fa9e5497888ecd340d664be101 --minimum-coverage 0.70 --minimum-accuracy-advantage 0.05 --minimum-macro-f1 0.50 --minimum-class-recall 0.40 --reference-accuracy 0.6335489543427746 --maximum-reference-accuracy-drop 0.05 --output results/intent_gru_h8_seed42.metrics.json --refuse-overwrite
```

Commit only `results/intent_gru_h8_seed42.metrics.json`, whether accepted or rejected. Keep the checkpoint local. Do not rerun the evaluator after seeing its result and do not train PPO.

### 49.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-060 | Accept the M9B validation result | Reject before test or retrain for higher validation accuracy | All three frozen validation gates passed with at least 7.47 pp margin | Retained |
| D-061 | Freeze checkpoint SHA-256 `74a72cf2...fc05` | Inspect alternative epochs or repeat training | Epoch 25 was selected by the predefined validation rule and provenance checks passed | Retained |
| D-062 | Authorize exactly one held-out intent-test evaluation | Skip final screening or reuse validation as final evidence | Test metrics remained sealed through training and audit | Retained |
| D-063 | Continue to prohibit PPO training | Treat validation and coverage as proof of driving improvement | No eight-step classifier test or policy experiment has occurred | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Completed and accepted M9B training for one held-out classifier evaluation | Apply the frozen validation gates while preserving test integrity | Summary/checkpoint hash, split, normalization, validation confusion matrix, and sealed-test state independently verified | Result: `b7c1abb`; documentation: this update |

**Next action:** run the frozen held-out intent evaluation exactly once after the full local test gate; commit its JSON result and keep the checkpoint local. Do not train PPO.


---

## 50. Milestone M9C result — eight-observation intent classifier accepted

**Experiment ID:** E-M9C-GRU-H8-EVAL-S42-DATA56433621

**Status:** Complete; all frozen classifier gates passed

**Date recorded:** 2026-09-05

**Result commit:** [`c8319cb`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/c8319cbb33adf66f5b198ddf2a5fb0f238ee64f6)

**Result artifact:** `results/intent_gru_h8_seed42.metrics.json`

**Artifact SHA-256:** `33c33367cd40250d9e1b42600bcbf9bf87a4052e5c6cf7f4fa377b2f42636b1f`

**Evaluated checkpoint SHA-256:** `74a72cf2b99b115bb5b4d55fdb350b55e20551876f6ba41be5266d8953b7fc05`

### 50.1 Integrity validation

The result commit contains only the small metrics JSON. The evaluator reports the exact accepted dataset and checkpoint hashes, source history length 10, causal-suffix history length 8, episode-grouped seed-42 split, best epoch 25, CPU evaluation, 15,923 test samples, and the pinned M9A coverage artifact `b364e39e...e101`.

The checkpoint test metric was null before evaluation and `test_metrics_were_sealed=true`. This was the first and only prediction pass over the held-out intent-test partition for the M9B checkpoint.

Independent confusion-matrix checks:

```text
4,561 cautious + 7,588 normal + 3,774 aggressive = 15,923 samples
2,769 + 5,156 + 1,657 = 9,582 correct
9,582 / 15,923 = 60.1771% accuracy
```

All row supports, column prediction counts, per-class precision/recall/F1 values, aggregate metrics, coverage evidence, and gate decisions recomputed exactly from the committed artifact.

### 50.2 Held-out classifier result

| Metric | Observed |
|---|---:|
| Samples | 15,923 |
| Accuracy | 60.1771% |
| Majority-class accuracy | 47.6543% |
| Accuracy advantage | +12.5228 pp |
| Balanced accuracy / macro recall | 57.5218% |
| Macro precision | 60.8164% |
| Macro F1 | 58.4577% |

Per-class held-out metrics:

| True class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Cautious | 4,561 | 58.58% | 60.71% | 59.63% |
| Normal | 7,588 | 59.92% | 67.95% | 63.68% |
| Aggressive | 3,774 | 63.95% | 43.91% | 52.07% |

### 50.3 Frozen gate decisions

| Gate | Requirement | Observed | Margin | Decision |
|---|---:|---:|---:|---|
| M9A coverage | At least 70% | 71.2301% | +1.2301 pp | Passed |
| Class support | All three | All three | — | Passed |
| Accuracy over majority | At least +5 pp | +12.5228 pp | +7.5228 pp | Passed |
| Macro F1 | At least 50% | 58.4577% | +8.4577 pp | Passed |
| Minimum class recall | At least 40% | 43.9057% | +3.9057 pp | Passed |
| Accuracy versus original GRU | At least 58.3549% | 60.1771% | +1.8222 pp | Passed |

The eight-step classifier's accuracy is 3.1778 percentage points below the original ten-step GRU's 63.3549%, its macro F1 is 3.3712 points lower, and its minimum class recall is 7.1807 points lower. In exchange, the frozen 14-slot history analysis raises prediction coverage by 7.3478 points, from 63.8823% at ten observations to 71.2301% at eight. This is the predefined, accepted availability-versus-classification tradeoff; no threshold was changed after results were observed.

### 50.4 Interpretation and decision

The eight-observation GRU remains materially better than the held-out majority baseline and retains useful performance for all three behaviors. Aggressive recall is still the limiting classifier metric, but it clears the frozen 40% floor. Combined with the independently selected 71.23% availability, every Section 48 classifier gate passed.

Accept `models/intent_gru_h8_seed42.pt` at SHA-256 `74a72cf2...fc05` as the sole classifier input for the next controlled PPO + intent experiment. The checkpoint remains local and must not be retrained or committed.

This acceptance is not a driving-policy result. To realize the measured coverage, the production observation wrapper must retain histories across all 14 observable traffic rows while continuing to output three probabilities for only the nearest five traffic slots. It must use the checkpoint's eight-observation requirement and preserve the existing observation dimension, reward, policy hyperparameters, seed, and training budget.

Seeds 10042–10541 are development evidence and may not be used for the final PPO comparison. A new paired holdout for both V3 and the future PPO + intent policy must be frozen before PPO training. Until that design and implementation are recorded, do not train PPO. PPO V3 remains the best driving policy.

**Decision:** accept the eight-observation intent classifier, freeze its checkpoint hash, authorize implementation of one controlled PPO + intent experiment, and retain PPO V3 as current best.

### 50.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-064 | Accept the eight-observation classifier | Reject it for lower accuracy or relax a failed gate | All six frozen availability and classifier gates passed without modification | Retained |
| D-065 | Freeze checkpoint SHA-256 `74a72cf2...fc05` for one PPO experiment | Retrain, tune, or select another epoch | The one-time held-out test is now consumed and cannot support further classifier selection | Retained |
| D-066 | Combine eight-observation inference with 14-slot history retention and five output slots | Use five-slot retention or expand the policy observation | M9A measured 71.23% only with 14-slot retention; five outputs preserve the controlled PPO input shape | Retained |
| D-067 | Reserve a new paired policy holdout | Reuse seeds 10042–10541 | Those seeds informed history-length and tracker design | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-05 | Completed and accepted M9C held-out evaluation | Screen the M9A-selected classifier exactly once against predefined gates | Commit scope, artifact/checkpoint hashes, confusion matrix, aggregate metrics, coverage, and all gates independently verified | Result: `c8319cb`; documentation: this update |

**Next action:** implement and freeze production eight-step inference with 14-slot history retention, then define the controlled PPO + intent training command and a new untouched paired holdout before any PPO run.


---

## 51. Milestone M10 design — PPO + intent V2 with accepted availability

**Experiment ID:** E-M10-PPO-INTENT-V2-H8T14-S42-200K

**Status:** Implemented and frozen; PPO training pending

**Date recorded:** 2026-09-06

**Research role:** test whether the accepted eight-observation classifier and 14-slot retention improve driving outcomes under the unchanged V3 reward and PPO protocol

### 51.1 Controlled intervention

PPO + intent V2 appends exactly the same three probabilities for each of the nearest five traffic slots as rejected PPO + intent V1. The base V3 observation, five output slots, probability order, warmup values, missing/obstacle values, action space, reward, traffic, and augmented observation dimension remain unchanged.

The representation changes from V1 are limited to the jointly accepted M9 design:

| Property | PPO + intent V1 | PPO + intent V2 |
|---|---:|---:|
| Intent checkpoint | Original ten-step GRU | Accepted eight-step GRU |
| Intent checkpoint SHA-256 | `10483649...bb05` | `74a72cf2...fc05` |
| Required history | 10 observations | 8 observations |
| Retained traffic histories | Nearest 5 slots | All 14 observable traffic slots |
| Policy-facing intent slots | 5 | 5 |
| Augmented observation size | 120 | 120 |
| Safety shield | Disabled | Disabled |

At each step, the production wrapper queries the same sorted nearest-five list used for policy alignment and a sorted 14-slot list used only to update causal histories. It verifies that the first five tracked objects align with the policy-facing list. Histories outside the nearest five are never appended to the policy observation; they become eligible only if that vehicle later moves into a policy-facing slot. Vehicles leaving all 14 observable traffic slots are purged.

The wrapper now reads a checkpoint's stored history length. An explicit length must match that metadata; legacy checkpoints without the field retain the previous ten-observation default. Wider production tracking defaults to the output count unless explicitly enabled, preserving every prior experiment.

### 51.2 Frozen PPO training protocol

| Setting | Fixed value |
|---|---:|
| Environment/reward configuration | `configs/intersection_reward_v3.yaml` |
| Configuration SHA-256 | `433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69` |
| Training seed | 42 |
| Requested / expected collected timesteps | 200,000 / 200,704 |
| Learning rate | 0.0003 |
| PPO rollout steps | 1,024 |
| Batch size | 64 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Entropy coefficient | 0.01 |
| Policy network | `[256, 256]` |
| Intent checkpoint | `models/intent_gru_h8_seed42.pt` |
| Intent checkpoint SHA-256 | `74a72cf2b99b115bb5b4d55fdb350b55e20551876f6ba41be5266d8953b7fc05` |
| Intent output / tracking slots | 5 / 14 |
| Intent history / device | 8 / CPU |
| Augmented observation shape | `[120]` |
| Internal-evaluation seed offset | 1,000 |
| Internal evaluation | 20 episodes every 10,000 calls |
| Safety shield | Disabled |
| Final checkpoint | `models/ppo_intent_v2_seed42.zip` |
| Training summary | `results/ppo_intent_v2_seed42.training.json` |

All PPO values exactly match M8. The experiment checkpoint is the final post-training state; the callback's best model is diagnostic and cannot be substituted after observing internal evaluation. Model, configuration, and intent hashes are verified before environment creation. Checkpoint and summary outputs refuse overwrite.

Frozen command:

```powershell
python -m scripts.train_ppo --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --timesteps 200000 --seed 42 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --intent-model models/intent_gru_h8_seed42.pt --intent-model-sha256 74a72cf2b99b115bb5b4d55fdb350b55e20551876f6ba41be5266d8953b7fc05 --intent-neighbors 5 --intent-history-length 8 --intent-history-tracking-neighbors 14 --intent-device cpu --eval-seed-offset 1000 --summary-output results/ppo_intent_v2_seed42.training.json --output models/ppo_intent_v2_seed42 --refuse-overwrite
```

After training, stop. Commit only the JSON summary whether the run succeeds or fails; keep all `.zip` checkpoints local. Do not begin policy evaluation until the final checkpoint hash, collected timesteps, spaces, finite parameters, and every summary field are validated and appended.

### 51.3 Newly frozen paired holdout

Seeds 10042–10541 informed M8C, M8D, M9A, and the selected representation. They are excluded from final M10 policy claims. M10 reserves the previously unused contiguous block **20042–20541** for exactly 500 paired episodes of both policies.

The comparator is the accepted V3 checkpoint:

| V3 property | Frozen value |
|---|---|
| Checkpoint | `models/ppo_reward_v3_seed42.zip` |
| SHA-256 | `f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c` |
| File size | 2,278,271 bytes |
| Stored timesteps | 200,704 |
| Stored observation | Native `Box(15, 7)`; 105 scalar features |
| Action space | Discrete, 3 actions |

The V3 hash and metadata were independently verified before reserving the holdout. Neither V3 nor V2 may be evaluated on seeds 20042–20541 until the V2 final checkpoint is frozen. At that point, exact hash-pinned commands for both policies will be appended. Both evaluations will use V3 configuration `433e6972...fae69`, deterministic actions, no shield, 2.0-second unsafe-TTC reporting, and the unchanged success/collision definitions.

M10 passes the driving-policy screen only if all predefined conditions hold relative to the newly measured paired V3 baseline:

1. V2 success is at least 3.0 percentage points higher than V3;
2. V2 collision is at least 3.0 percentage points lower than V3;
3. V2 incomplete non-collision episodes are at most 2.0%;
4. V2 mean minimum TTC is at least the paired V3 value;
5. the paired success change is favorable with an exact two-sided McNemar test at `p < 0.05`.

These are the same minimum effect, completion, TTC, and significance rules frozen for M8, now applied to an untouched paired baseline rather than reused holdout outcomes. Reward magnitude remains descriptive, not an acceptance gate. A failed or mixed result is retained and PPO + intent V2 is rejected as an improvement.

### 51.4 Implementation and engineering verification

The environment factory, PPO trainer, and evaluator now carry explicit intent history length and production tracking count. Training/evaluation summaries record both. PPO and configuration fingerprints can be required explicitly, and both scripts support overwrite refusal. Existing defaults reproduce five-slot tracking and the legacy ten-step fallback.

Final automated checks:

```text
git diff --check: passed
ruff check: passed
pytest: 83 passed in 10.51 s
```

Engineering-only verification used seed 42, never the reserved holdout:

- an actual legacy checkpoint instantiated with history length 10 and five tracking slots;
- the accepted checkpoint instantiated with history length 8 and 14 tracking slots;
- both produced the expected policy input shape, and a 32-step in-memory PPO rollout completed;
- a 32-step module-form trainer smoke saved a structurally valid summary with configuration/intent hashes, `[120]` observation shape, and 8/14/5 metadata;
- a one-episode evaluator smoke recorded the correct model/configuration hashes and 8/14/5 metadata.

The first direct-file trainer smoke failed before environment creation because that invocation resolved the older editable installation in the user's separate checkout, which did not yet contain the new environment argument. It produced no policy result. Repeating the engineering smoke as `python -m scripts.train_ppo` bound the current repository source and passed; module form is therefore frozen for M10.

An initial read-only V3 metadata assertion incorrectly expected a flattened stored observation shape `(105,)`. Inspection showed the valid native stored `Box(15, 7)` shape, which the PPO feature extractor flattens to 105 scalars. The checkpoint otherwise matched all expected metadata; no file or protocol changed.

### 51.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-068 | Use eight-step inference with 14 retained histories and five policy outputs | Five-slot retention or 14 policy outputs | Realize M9A coverage while preserving the controlled 120-value observation | Retained |
| D-069 | Train PPO + intent V2 under the unchanged M8 PPO/V3 reward protocol | Change reward, budget, seed, architecture, or add the rejected shield | Isolate the accepted intent-availability representation | Retained |
| D-070 | Reserve seeds 20042–20541 for both V3 and V2 | Reuse development seeds 10042–10541 | Prevent representation-design feedback from contaminating the final policy claim | Retained |
| D-071 | Reuse M8's predefined 3 pp, completion, TTC, and McNemar gates | Set thresholds after seeing the new baseline or V2 result | Preserve comparable decision standards | Retained |
| D-072 | Use module-form entry points for M10 | Direct script execution from a checkout with a stale editable installation | Ensure the invoked package is the pulled repository source | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-06 | Implemented and froze M10 PPO + intent V2 training and untouched paired holdout | Test the accepted classifier/coverage representation without changing reward, PPO, shield, or policy input size | 83 tests, real-checkpoint environment smoke, 32-step PPO/trainer smoke, one-episode evaluator smoke, and V3 checkpoint audit passed; two engineering setup assumptions failed and were recorded | This implementation update |

**Next action:** pull this implementation, rerun Ruff and all tests, run only the frozen M10 training command, commit its JSON summary, and keep the final PPO checkpoint local. Do not evaluate seeds 20042–20541 yet.


---

## 52. Milestone M10A result — PPO + intent V2 training

**Experiment ID:** E-M10-PPO-INTENT-V2-H8T14-S42-200K

**Status:** Training complete; checkpoint validated; paired holdout authorized

**Date recorded:** 2026-09-06

**Training-summary commit:** [`0d71ab0`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/0d71ab074bff1c8410b426bdf42277aa568ad5bd)

**Training summary:** `results/ppo_intent_v2_seed42.training.json`

**Summary SHA-256:** `5d5b7138239670892744df958a2668a3828f14509f1a7d5eab5eb4680ae511d0`

**Local checkpoint:** `models/ppo_intent_v2_seed42.zip`

**Checkpoint SHA-256:** `4fc855e064c366b0d0b94b807066b235a46f8cf3c7bfb06edceb6a56fa9b1773`

### 52.1 Training-summary validation

The result commit contains only the small training JSON; all model checkpoints remain local. Every frozen Section 51 property matched:

| Property | Expected | Recorded | Decision |
|---|---:|---:|---|
| Algorithm | PPO | PPO | Passed |
| Configuration SHA-256 | `433e6972...fae69` | `433e6972...fae69` | Passed |
| Training seed | 42 | 42 | Passed |
| Internal-evaluation seed offset | 1,000 | 1,000 | Passed |
| Requested timesteps | 200,000 | 200,000 | Passed |
| Collected timesteps | 200,704 | 200,704 | Passed |
| Learning rate | 0.0003 | 0.0003 | Passed |
| Rollout steps | 1,024 | 1,024 | Passed |
| Batch size | 64 | 64 | Passed |
| Gamma / GAE lambda | 0.99 / 0.95 | 0.99 / 0.95 | Passed |
| Entropy coefficient | 0.01 | 0.01 | Passed |
| Policy network | `[256, 256]` | `[256, 256]` | Passed |
| Observation shape | `[120]` | `[120]` | Passed |
| Safety shield | Disabled | Disabled | Passed |
| Unsafe-TTC reporting threshold | 2.0 s | 2.0 s | Passed |
| Intent checkpoint SHA-256 | `74a72cf2...fc05` | `74a72cf2...fc05` | Passed |
| Intent output / tracking slots | 5 / 14 | 5 / 14 | Passed |
| Intent history / device | 8 / CPU | 8 / CPU | Passed |

The configuration, accepted intent checkpoint, and final PPO checkpoint hashes were independently recomputed from local files and match the summary.

### 52.2 Checkpoint validation

| Property | Verified value |
|---|---|
| File size | 2,370,824 bytes |
| Stored timesteps | 200,704 |
| Observation space | Flat float32 `Box(120,)`; final 15 probability features bounded to `[0,1]` |
| Action space | Discrete, 3 actions |
| PPO settings | Exact Section 51 match |
| Policy network | `[256, 256]` |
| Policy parameters | All finite |

The final post-training checkpoint—not the callback's best checkpoint—is frozen for evaluation. Validation loaded the ZIP on CPU and did not execute any episode from the reserved holdout.

### 52.3 Interpretation and decision

M10A confirms that the intended 8/14/5 observation pipeline completed the unchanged 200K PPO protocol and produced a structurally valid final policy. It provides no evidence yet about success, collision, incompleteness, TTC, or reward.

Accept `models/ppo_intent_v2_seed42.zip` at SHA-256 `4fc855e0...b1773` as the sole M10 policy checkpoint. No retraining, callback-best substitution, reward change, classifier change, tracking change, or safety shield is permitted.

Both V3 and V2 must now be evaluated exactly once on the previously reserved seeds 20042–20541. Running both policies is one paired experiment. Do not stop to redesign or tune after seeing the first policy's output. Commit all four small result artifacts and retain the outcome whether M10 passes or fails.

### 52.4 Frozen paired evaluation commands

After pulling this record, first run:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and none of the four output paths exists, run both commands:

```powershell
python -m scripts.evaluate_policy --model models/ppo_reward_v3_seed42.zip --model-sha256 f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 20042 --unsafe-ttc 2.0 --output results/ppo_reward_v3_holdout_seed20042.csv --refuse-overwrite

python -m scripts.evaluate_policy --model models/ppo_intent_v2_seed42.zip --model-sha256 4fc855e064c366b0d0b94b807066b235a46f8cf3c7bfb06edceb6a56fa9b1773 --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 20042 --unsafe-ttc 2.0 --intent-model models/intent_gru_h8_seed42.pt --intent-model-sha256 74a72cf2b99b115bb5b4d55fdb350b55e20551876f6ba41be5266d8953b7fc05 --intent-neighbors 5 --intent-history-length 8 --intent-history-tracking-neighbors 14 --intent-device cpu --output results/ppo_intent_v2_holdout_seed20042.csv --refuse-overwrite
```

Expected versionable outputs:

```text
results/ppo_reward_v3_holdout_seed20042.csv
results/ppo_reward_v3_holdout_seed20042.summary.json
results/ppo_intent_v2_holdout_seed20042.csv
results/ppo_intent_v2_holdout_seed20042.summary.json
```

Commit all four files together. Keep all `.zip` and `.pt` files local. Do not rerun, modify, or selectively omit a result after observing it.

### 52.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-073 | Accept final PPO + intent V2 checkpoint `4fc855e0...b1773` for paired evaluation | Retrain or substitute the callback-best checkpoint | All summary, ZIP, space, hyperparameter, and finiteness checks passed | Retained |
| D-074 | Run both V3 and V2 evaluations as one indivisible paired experiment | Inspect the new V3 baseline before deciding whether to run V2 | Preserve the precommitted design after either result becomes visible | Retained |
| D-075 | Commit all four holdout artifacts regardless of outcome | Commit only favorable or aggregate results | Preserve episode-level paired evidence and complete provenance | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-06 | Completed and validated M10A training | Freeze the final PPO + intent V2 checkpoint before touching the reserved holdout | Commit scope, four hashes, all summary fields, checkpoint spaces/hyperparameters/timesteps, probability bounds, and parameter finiteness verified | Result: `0d71ab0`; documentation: this update |

**Next action:** run both frozen 500-episode evaluations on seeds 20042–20541 after the full test gate; commit both CSVs and both summaries. Do not retrain or run only one policy.


---

## 53. Milestone M10B result — paired PPO V3 versus PPO + intent V2

**Experiment ID:** E-M10-PPO-INTENT-V2-H8T14-S42-200K

**Status:** Complete; PPO + intent V2 rejected as an improvement

**Date recorded:** 2026-09-07

**Result commit:** [`f652795`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/f652795453f320a63037b768e58d871b75347fac)

### 53.1 Result-artifact and protocol audit

The result commit is one commit directly after the frozen M10A documentation and contains exactly the four required files: both 500-row episode CSVs and both summaries. No checkpoint, configuration, source, or unrelated result changed.

| Artifact | SHA-256 |
|---|---|
| `results/ppo_reward_v3_holdout_seed20042.csv` | `f57d357da61ac309b8542e54ab8e74c5bcf456ee98c5ef459e25d83d40523f65` |
| `results/ppo_reward_v3_holdout_seed20042.summary.json` | `505bcae5d8d2c1ebd7fab0429d0d984e55adbe267f357d7f4bacd833ad59abd2` |
| `results/ppo_intent_v2_holdout_seed20042.csv` | `a54ddbdd564e9af28de7c2d2fe631c113bf3ce570d79a43d037ef69aa5f27c17` |
| `results/ppo_intent_v2_holdout_seed20042.summary.json` | `89450da5c36edce191ab181662c743c010722e5843eec07d96ae1623d42e0430` |

Both summaries record 500 episodes, seeds 20042–20541, deterministic policy evaluation, configuration SHA-256 `433e6972...fae69`, no safety shield, and a 2.0-second unsafe-TTC reporting threshold. V3 records the frozen checkpoint SHA-256 `f46964bf...706c` and no intent model. V2 records the frozen PPO SHA-256 `4fc855e0...b1773`, intent SHA-256 `74a72cf...fc05`, five policy-facing neighbors, 14 tracked histories, history length eight, and CPU inference. The artifacts therefore match the frozen paired protocol without a silent model or configuration change.

The pre-evaluation Ruff and pytest gate was user-reported as completed after the frozen commands were supplied; the result artifacts do not themselves encode the test output. The four output paths were newly added together and were not selectively omitted after either result became visible.

### 53.2 Raw aggregate result

| Metric | Paired V3 | PPO + intent V2 | V2 minus V3 |
|---|---:|---:|---:|
| Episodes | 500 | 500 | 0 |
| Mean reward | 2.284797 | 2.339301 | +0.054504 |
| Mean episode length | 38.314 | 39.678 | +1.364 |
| Success rate | 60.8% | 61.8% | +1.0 pp |
| Collision rate | 39.4% | 38.2% | -1.2 pp |
| Incomplete non-collision rate | 0.0% | 0.0% | 0.0 pp |
| Mean travel time | 7.6628 s | 7.9356 s | +0.2728 s |
| Mean minimum TTC | 0.584723 s | 0.604596 s | +0.019873 s |
| Mean unsafe-TTC events | 16.548 | 16.896 | +0.348 |
| Mean safety interventions | 0.0 | 0.0 | 0.0 |

V2 has small favorable raw changes in success, collision, reward, and mean minimum TTC, but also slightly longer travel time and more unsafe-TTC events. Reward was descriptive and cannot rescue a failed driving-policy gate.

### 53.3 Paired success test

Pairing the rows by their common seed gives:

| V3 success | V2 success | Episodes |
|---:|---:|---:|
| True | True | 277 |
| True | False | 27 |
| False | True | 32 |
| False | False | 164 |

There are 59 discordant pairs. The exact two-sided McNemar probability is

$$
p = 2\sum_{k=0}^{27}{59 \choose k}(0.5)^{59}=0.602923.
$$

The observed five-episode net gain is compatible with chance and does not satisfy `p < 0.05`.

### 53.4 Terminal-outcome anomaly and sensitivity analysis

The raw V3 CSV contains one internally overlapping terminal outcome: zero-based episode index 80, seed 20122, records `success=True` and `collision=True` with reward 8.975, length 45, travel time 9.0 seconds, minimum TTC 0.435576 seconds, and 31 unsafe-TTC events. V3 therefore has 304 success flags and 197 collision flags across 500 rows, totaling 501 flags. It has no row with neither flag. V2 has 309 success-only rows, 191 collision-only rows, no overlap, and no row with neither flag.

This was possible because the evaluator independently queried route arrival and crash state. The committed files and summaries are preserved unchanged. M10 is evaluated first under its frozen raw definitions; the anomaly was discovered only after the result and cannot justify selectively rewriting or rerunning the holdout.

A safety-first sensitivity analysis classifies the overlapping V3 row as collision, not success. Under that rule V3 success becomes 303/500 = 60.6%, so V2's success change becomes +1.2 percentage points. The paired discordances become 27 V2 regressions and 33 V2 gains, with exact two-sided McNemar `p = 0.518958`. Collision and incomplete rates are unchanged. Thus the M10 rejection is robust to the terminal-outcome correction.

### 53.5 Precommitted gate decision

| Gate | Required | Observed | Decision |
|---|---:|---:|---|
| Success improvement | at least +3.0 pp | +1.0 pp raw; +1.2 pp sensitivity | Failed |
| Collision reduction | at least -3.0 pp | -1.2 pp | Failed |
| V2 incomplete non-collision rate | at most 2.0% | 0.0% | Passed |
| Mean minimum TTC | V2 at least V3 | 0.604596 >= 0.584723 s | Passed |
| Favorable exact paired test | `p < 0.05` | `p = 0.602923`; sensitivity `p = 0.518958` | Failed |

Only two of five gates pass. **PPO + intent V2 is rejected as an improvement.** The eight-step classifier and wider history retention solved the predefined availability problem, but this single controlled PPO run did not convert that representation improvement into a sufficiently large or statistically supported driving improvement. The rejected V2 checkpoint remains research evidence and must not replace V3. PPO V3 remains the current best accepted policy.

### 53.6 Outcome-integrity correction for future work

After sealing the M10 decision, terminal outcome detection was centralized. `detect_collision` now applies the same HighwayEnv-compatible crash lookup to policy evaluation, rule-based evaluation, and intent-rollout diagnostics. `detect_success` first checks collision and returns false for a crashed arrival. A regression test covers an environment that reports both explicit success and crash. This implements the safety-first rule prospectively; it does not alter or reinterpret stored M10 artifacts.

Because this correction changes the success definition in the one overlapping edge case, every future paired experiment must establish its own comparator under the corrected code on newly reserved seeds. A later result must not be compared directly against a pre-correction aggregate as if outcome definitions were identical.

Verification after the correction:

```text
ruff check: passed
pytest attempt 1: 73 passed, 11 setup errors; sandbox denied access to the shared Windows pytest temp directory
pytest attempt 2: 73 passed, 11 setup errors; sandbox also denied access to a newly requested workspace temp directory
pytest outside the restricted temp sandbox: 84 passed in 3.97 s
```

Both failed attempts stopped at temporary-fixture setup and produced no experiment result. The unrestricted rerun exercised the complete suite and passed.

### 53.7 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-076 | Reject PPO + intent V2 as an improvement and retain V3 | Promote V2 based on favorable raw direction | V2 failed success-effect, collision-effect, and paired-significance gates | Retained |
| D-077 | Preserve the four raw M10 artifacts and do not rerun the consumed holdout | Rewrite the overlapping row or rerun after observing results | Append-only evidence and the rejection are robust under collision-first sensitivity | Retained |
| D-078 | Give collision precedence over arrival in all future success classification | Continue allowing simultaneous success and collision flags | Seed 20122 exposed an unsafe ambiguity; a crashed arrival is not a successful autonomous-driving outcome | Retained |
| D-079 | Treat seeds 20042–20541 as consumed development evidence | Reuse them for tuning or another final claim | Both policies and the anomaly have now been inspected | Retained |
| D-080 | Authorize no new training or evaluation yet | Immediately change reward, PPO settings, intent representation, or shield | M10 needs a separately frozen post-result diagnostic/design step before another intervention | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Completed and rejected M10B PPO + intent V2 | Apply all five gates to the untouched paired holdout | Four-file commit scope, artifact hashes, metadata, 1,000 episode rows, aggregate metrics, discordant counts, exact McNemar test, and collision-first sensitivity independently verified | Result: `f652795`; documentation: this update |
| 2026-09-07 | Centralized collision detection and made success collision-free prospectively | Prevent a crashed arrival from being counted as a policy success | Ruff passed; full suite passed with 84 tests after two recorded sandbox temp-directory failures | This implementation update |

**Next action:** design and append a non-interventional post-M10 diagnostic or a new controlled intervention before running any more training or evaluation. Do not tune on or reuse seeds 20042–20541, do not rerun M10, and do not combine the rejected shield or V2 policy without a newly frozen protocol.


---

## 54. Milestone M11 design — prospective V3 checkpoint selection

**Experiment ID:** E-M11-V3-BEST120K-VS-FINAL200K-S42

**Status:** Designed and frozen; paired evaluation pending

**Date recorded:** 2026-09-07

**Research role:** determine whether the automatically selected validation-reward checkpoint from the original V3 training run improves success relative to that run's final checkpoint

### 54.1 Motivation and controlled intervention

M10 showed that improved intent availability did not produce the required driving improvement. Before changing the reward, observation, PPO algorithm, traffic, or safety mechanism again, M11 tests a simpler hypothesis already suggested by the original V3 training trace: the final training state may not be the strongest state produced by the run.

The original V3 `EvalCallback` evaluated 20 deterministic episodes every 10,000 training steps and automatically saved a new checkpoint only when mean validation reward improved. Its maximum recorded mean reward was 4.655 at 120,000 steps; the final 200,000-step evaluation recorded 2.035. The callback therefore saved the 120K policy as `logs/ppo_reward_v3_seed42/best/best_model.zip`. This checkpoint existed before M10's results and was selected mechanically by the original callback, not by searching the new holdout.

M11 changes only the evaluated checkpoint within the same original training run:

| Property | Baseline | Candidate |
|---|---|---|
| Policy | V3 final checkpoint | V3 callback-best checkpoint |
| Stored timesteps | 200,704 | 120,000 |
| Training seed | 42 | 42 |
| Reward/configuration | V3 | V3 |
| Observation | Native `Box(15, 7)` | Native `Box(15, 7)` |
| Action space | Discrete, 3 actions | Discrete, 3 actions |
| Intent features | None | None |
| Safety shield | Disabled | Disabled |

No new PPO training occurs. The hypothesis is checkpoint-selection feasibility, not a claim that fewer training steps are generally superior.

### 54.2 Checkpoint audit

| Property | V3 final baseline | V3 120K candidate |
|---|---|---|
| Path | `models/ppo_reward_v3_seed42.zip` | `logs/ppo_reward_v3_seed42/best/best_model.zip` |
| SHA-256 | `f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c` | `3e7cff6c846ac75dc383bce45f2ab2d82234956423f8eb4f6ff7c4c47d05f0d9` |
| Stored timesteps | 200,704 | 120,000 |
| Learning rate | 0.0003 | 0.0003 |
| PPO rollout steps | 1,024 | 1,024 |
| Batch size | 64 | 64 |
| Gamma / GAE lambda | 0.99 / 0.95 | 0.99 / 0.95 |
| Entropy coefficient | 0.01 | 0.01 |
| Policy network | `[256, 256]` | `[256, 256]` |
| Parameter count | 186,884 | 186,884 |
| Parameters finite | Yes | Yes |

Both checkpoints and `configs/intersection_reward_v3.yaml` were independently loaded or hashed before this protocol was frozen. The configuration SHA-256 remains `433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69`.

The 120K candidate may not be replaced by a different periodic checkpoint after any M11 result is observed. Testing every stored checkpoint and reporting the maximum would create an unregistered multiple-comparison search.

### 54.3 Frozen paired holdout and gates

Seeds **30042–30541** are newly reserved for exactly 500 paired episodes per checkpoint. They do not appear in the preceding research record. Both policies use deterministic actions, the unchanged V3 configuration, no intent model, no safety shield, and the 2.0-second unsafe-TTC reporting threshold. The collision-first success definition introduced after M10 applies equally to both sides.

M11 accepts the 120K candidate only if all conditions hold relative to the newly measured 200K baseline:

1. candidate success is at least 3.0 percentage points higher;
2. candidate collision is at least 3.0 percentage points lower;
3. candidate incomplete non-collision episodes are at most 2.0%;
4. candidate mean minimum TTC is at least the paired baseline value;
5. the paired success change is favorable with exact two-sided McNemar `p < 0.05`.

These are the same effect-size, completion, TTC, and significance rules used for M10. Mean reward and travel time are descriptive. If any gate fails, the 120K candidate is rejected and the 200K V3 checkpoint remains current best. A result near 64% success would meet the practical success target only if the paired baseline remains near its historical 60–61% range; no success percentage is guaranteed before evaluation.

### 54.4 Pre-run verification and frozen commands

The repository passed `git diff --check`, Ruff, and all 84 tests immediately before this design was recorded. After pulling this protocol, rerun:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and none of the four new outputs exists, run both evaluations as one experiment:

```powershell
python -m scripts.evaluate_policy --model models/ppo_reward_v3_seed42.zip --model-sha256 f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 30042 --unsafe-ttc 2.0 --output results/ppo_reward_v3_final200k_seed30042.csv --refuse-overwrite

python -m scripts.evaluate_policy --model logs/ppo_reward_v3_seed42/best/best_model.zip --model-sha256 3e7cff6c846ac75dc383bce45f2ab2d82234956423f8eb4f6ff7c4c47d05f0d9 --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 30042 --unsafe-ttc 2.0 --output results/ppo_reward_v3_best120k_seed30042.csv --refuse-overwrite
```

Expected versionable outputs:

```text
results/ppo_reward_v3_final200k_seed30042.csv
results/ppo_reward_v3_final200k_seed30042.summary.json
results/ppo_reward_v3_best120k_seed30042.csv
results/ppo_reward_v3_best120k_seed30042.summary.json
```

Commit the four files together regardless of outcome. Keep both ZIP files local. Do not stop after the first result, rerun either output, test another stored checkpoint, or change the order, seeds, hashes, configuration, TTC reporting threshold, or evaluation code.

### 54.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-081 | Test the original automatically selected V3 120K checkpoint before new training | Immediately change reward, PPO budget, architecture, intent, or shield | Original validation reward peaked at 120K, while final-checkpoint reward was lower | Retained |
| D-082 | Compare only the automatic best checkpoint with the frozen final checkpoint | Scan all periodic checkpoints | Avoid a post-result multiple-comparison search | Retained |
| D-083 | Reserve seeds 30042–30541 for the corrected paired comparison | Reuse consumed seeds 10042–10541 or 20042–20541 | Preserve a clean evaluation after the prospective outcome-definition fix | Retained |
| D-084 | Reuse the five M10 acceptance gates | Lower the threshold to accept a small gain | Require a practically meaningful, safe, and statistically supported improvement | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Audited and froze M11 V3 checkpoint-selection experiment | Seek a higher-success policy before adding another algorithmic or reward confound | Candidate hash, timesteps, spaces, PPO settings, architecture, parameter count/finiteness, original validation trace, unused seeds, and 84-test baseline verified | This design update |

**Next action:** pull this design, rerun Ruff and all tests, execute both frozen M11 evaluations exactly once, and commit all four small result artifacts. Do not train a new model yet.


---

## 55. Milestone M11 result — prospective V3 checkpoint selection

**Experiment ID:** E-M11-V3-BEST120K-VS-FINAL200K-S42

**Status:** Complete; 120K checkpoint rejected

**Date recorded:** 2026-09-07

**Result commit:** [`992afdc`](https://github.com/kingofrichnight/RL--Autonomus-car/commit/992afdc67287ab1a1d1afb64cbf852a59e697e23)

### 55.1 Artifact and protocol audit

The result commit follows the frozen M11 design directly and contains exactly the two 500-row CSVs and their two summaries. No source, configuration, checkpoint, or unrelated result changed.

| Artifact | SHA-256 |
|---|---|
| `results/ppo_reward_v3_final200k_seed30042.csv` | `fa36efa8afb82745a3652c60504d07d3cf054496b7fea74933ee51e67eab60de` |
| `results/ppo_reward_v3_final200k_seed30042.summary.json` | `520bfcb14d5ba93b014e246db1c923674894d0c17f37d3972bb9f4909101a837` |
| `results/ppo_reward_v3_best120k_seed30042.csv` | `055a203af42d5f299ff9d757e92e8995ed943cb83796261f51d4712011c68951` |
| `results/ppo_reward_v3_best120k_seed30042.summary.json` | `afc390d237769e58c1dafbc3df2ac1c7d0e53cecf4408c5b96967ba554565ce8` |

Both summaries record seeds 30042–30541, the frozen V3 configuration SHA-256 `433e6972...fae69`, no intent model, no shield, and the 2.0-second unsafe-TTC reporting threshold. The baseline hash is `f46964bf...706c`; the sole candidate hash is `3e7cff6c...f0d9`. The new collision-first outcome definition was applied to both. All 1,000 rows are mutually exclusive: every episode is either success or collision, with no overlap and no incomplete row.

### 55.2 Aggregate result

| Metric | V3 final 200K | V3 automatic-best 120K | Candidate minus baseline |
|---|---:|---:|---:|
| Episodes | 500 | 500 | 0 |
| Mean reward | 2.272665 | 1.718763 | -0.553901 |
| Mean episode length | 38.404 | 39.084 | +0.680 |
| Success rate | 60.8% | 57.6% | -3.2 pp |
| Collision rate | 39.2% | 42.4% | +3.2 pp |
| Incomplete non-collision rate | 0.0% | 0.0% | 0.0 pp |
| Mean travel time | 7.6808 s | 7.8168 s | +0.1360 s |
| Mean minimum TTC | 0.612142 s | 0.611624 s | -0.000518 s |
| Mean unsafe-TTC events | 16.132 | 16.324 | +0.192 |
| Mean safety interventions | 0.0 | 0.0 | 0.0 |

The candidate is worse on success, collision, reward, travel time, minimum TTC, and unsafe-event count. Its only neutral result is completion.

### 55.3 Paired success test

| Final 200K success | Candidate 120K success | Episodes |
|---:|---:|---:|
| True | True | 265 |
| True | False | 39 |
| False | True | 23 |
| False | False | 173 |

There are 62 discordant pairs. The candidate rescues 23 baseline failures but regresses 39 baseline successes, a net loss of 16 successes. The exact two-sided McNemar probability is

$$
p = 2\sum_{k=0}^{23}{62 \choose k}(0.5)^{62}=0.055897.
$$

The result is close to 0.05 only in the unfavorable direction. It cannot satisfy the prospectively required favorable paired test.

### 55.4 Gate decision and interpretation

| Gate | Required | Observed | Decision |
|---|---:|---:|---|
| Success improvement | at least +3.0 pp | -3.2 pp | Failed |
| Collision reduction | at least -3.0 pp | +3.2 pp | Failed |
| Candidate incomplete rate | at most 2.0% | 0.0% | Passed |
| Mean minimum TTC | candidate at least baseline | 0.611624 < 0.612142 s | Failed |
| Favorable exact paired test | `p < 0.05` | unfavorable; `p = 0.055897` | Failed |

Only one of five gates passes. **The automatic-best 120K checkpoint is rejected.** The original 20-episode validation-reward maximum did not generalize to higher success on the independent paired holdout. The final 200K V3 checkpoint remains the current best accepted policy at SHA-256 `f46964bf...706c`.

No other periodic checkpoint may now be scanned or substituted under M11. Seeds 30042–30541 are consumed. This result does not show that 200K is universally optimal; it shows that the original automatic 120K reward-selected candidate is not a successful replacement.

The clean collision-first baseline also independently reproduced V3 at 60.8% success, close to its prior holdouts. This reinforces that the approximately 60–61% success level is stable enough to require an actual intervention rather than checkpoint cherry-picking.

### 55.5 Direction after M11

Global caution remains rejected by existing evidence: V4 introduced conservative waiting and the 2.0-second radial-TTC shield produced 28.6% incomplete episodes. M11 also rules out recovering a higher-success policy merely by substituting the original callback-best checkpoint.

The next safety work must therefore be **selective caution**. Before implementing an intervention, a non-interventional diagnostic should replay the accepted V3 policy on already consumed development seeds and measure action-conditioned intersection conflict geometry. Its purpose is to distinguish true projected crossing conflicts from low radial TTC caused by non-conflicting or clearing traffic. Thresholds and anti-deadlock behavior must be frozen from development diagnostics before any new paired holdout is reserved or evaluated.

No new training, safety shield evaluation, or untouched holdout is authorized by this result section.

### 55.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-085 | Reject the V3 automatic-best 120K checkpoint | Promote it from the original validation reward | 57.6% success and 42.4% collision; four of five gates failed | Retained |
| D-086 | Retain final 200K V3 as current best | Scan other stored checkpoints after seeing M11 | Final V3 beat the only prospectively selected candidate by 3.2 success points | Retained |
| D-087 | Consume seeds 30042–30541 and forbid checkpoint scanning on them | Use them to choose another training step | Prevent post-holdout multiple comparisons | Retained |
| D-088 | Develop action-conditioned selective caution next | Increase global caution or reuse the 2.0-second shield | Prior global caution caused waiting/incompletion; M11 shows checkpoint choice is insufficient | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Completed and rejected M11 automatic V3 checkpoint selection | Test whether the original validation-reward checkpoint improved success without retraining | Four-file scope, four hashes, frozen metadata, 1,000 exclusive episode outcomes, aggregate metrics, paired table, exact McNemar test, and all five gates independently verified | Result: `992afdc`; documentation: this update |

**Next action:** implement and freeze a non-interventional, action-conditioned conflict diagnostic on already consumed development seeds before designing a selective-caution shield. Do not train or evaluate a new safety intervention yet.


---

## 56. Milestone M12A design — action-conditioned conflict diagnostic

**Experiment ID:** E-M12A-V3-CPA-CONFLICT-DIAGNOSTIC-S10042

**Status:** Implemented and frozen; non-interventional diagnostic pending

**Date recorded:** 2026-09-07

**Research role:** identify selective conflict rules that cover collision episodes without imposing the broad intervention burden that caused TTC-shield deadlock

### 56.1 Scope and causal boundary

M12A is a development diagnostic, not a policy evaluation and not a safety result. It replays the accepted final V3 checkpoint on the already-consumed seeds 10042–10541. The PPO action is sent to the environment unchanged at every decision. No reward, observation, traffic, action, model, intent feature, or shield changes.

The run must reproduce `results/ppo_reward_v3_holdout_seed10042.csv` exactly before its conflict profiles are accepted. That reference predates the collision-first correction, but its 500 rows contain 298 success-only and 202 collision-only outcomes with no overlap or incompletion; the prospective correction therefore does not alter any reference row.

Frozen provenance:

| Property | Value |
|---|---|
| PPO checkpoint | `models/ppo_reward_v3_seed42.zip` |
| PPO SHA-256 | `f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c` |
| Configuration | `configs/intersection_reward_v3.yaml` |
| Configuration SHA-256 | `433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69` |
| Reference episode CSV | `results/ppo_reward_v3_holdout_seed10042.csv` |
| Reference SHA-256 | `642a0a34841f6bc351acd573b5c057a95c53ddcc2348553de1f10adee0286c09` |
| Episodes / seeds | 500 / 10042–10541 |
| Policy | Deterministic V3; no intent; no shield |
| Unsafe-TTC reporting threshold | 2.0 s |
| Output | `results/ppo_reward_v3_conflict_diagnostics_seed10042.json` |

Seeds 10042–10541 were already consumed by M5–M9 and remain development-only. The diagnostic cannot support a final performance claim.

### 56.2 Closest-approach geometry

For relative position $r=p_{other}-p_{ego}$ and relative velocity $v=v_{other}-v_{ego}$, constant-velocity time to closest approach is

$$
t_{CPA}=-\frac{r\cdot v}{v\cdot v}.
$$

For finite $t_{CPA}$ within the three-second horizon, predicted miss distance is

$$
d_{CPA}=\lVert r+v t_{CPA}\rVert_2.
$$

Stationary relative motion, a closest approach in the past, or a closest approach beyond the horizon is excluded. Traffic currently beyond 60 metres is excluded. Every finite nearby vehicle approach is retained. A decision matches a threshold pair if **any** retained vehicle has both $t_{CPA}$ and $d_{CPA}$ at or below that pair.

This is more selective than the rejected radial-TTC rule, which reacts to closing distance without requiring a small projected miss distance. It is still a constant-velocity diagnostic, not proof that an override would prevent a collision.

### 56.3 Frozen profile grid and windows

The diagnostic profiles two proposed-action scopes:

1. `FASTER_ONLY`, representing a mild caution veto that could replace acceleration with `IDLE`;
2. `FASTER_OR_IDLE`, representing an emergency scope that could replace acceleration or maintained speed with `SLOWER`.

Each scope is crossed with five CPA time thresholds and six miss-distance thresholds:

```text
time thresholds (s):     0.50, 0.75, 1.00, 1.50, 2.00
distance thresholds (m): 1.50, 2.00, 2.50, 3.00, 4.00, 5.00
```

This produces 60 fixed profiles. Every profile reports decision triggers and episode-level trigger coverage over the complete episode, final one second, and final two seconds. Reports separate collision-episode coverage from success-episode burden. The old 2.0-second `FASTER_OR_IDLE` radial-TTC rule is profiled as a fixed reference.

These labels measure association with final episode outcome, not counterfactual benefit. No reported trigger is called a prevented collision.

### 56.4 Development feasibility rule

For mild caution, inspect `FASTER_ONLY` in the final-two-second window. For emergency braking, inspect `FASTER_OR_IDLE` in the final-one-second window. A profile is development-feasible only if all hold:

1. at least 50% of collision episodes contain a trigger in the applicable window;
2. no more than 30% of success episodes contain a trigger in that window;
3. collision coverage minus success burden is at least 25 percentage points.

Within each action scope, rank feasible profiles by: largest coverage-minus-burden advantage, then largest collision coverage, then smallest success burden, then smallest all-episode trigger rate, then smaller time and distance thresholds. This rule is frozen before the 500-episode diagnostic. If neither scope has a feasible profile, no CPA shield is authorized and the failure must be recorded. If one or both scopes pass, their mechanically selected profiles inform a separately implemented M12B intervention with explicit release/anti-deadlock behavior; M12A itself does not authorize evaluation of that intervention.

### 56.5 Failure preservation and implementation verification

Model, configuration, and reference hashes are checked before environment creation. The output uses exclusive creation and cannot overwrite existing evidence. If episode reproduction fails, the JSON records `reference_reproduced=false`, the failure reason, summary, and observed episode rows before the command raises. A failed diagnostic output must be committed and analyzed rather than silently rerun.

The initial engineering implementation retained only the single smallest miss distance per decision. Review found that this could hide an earlier, slightly wider approach when evaluating a shorter time threshold. No research run used that implementation. It was corrected to retain every finite nearby closest approach, and a regression test constructs exactly this earlier-wider/later-closer case.

Final engineering verification:

```text
ruff check: passed
pytest: 89 passed in 3.31 s
two-episode seed-7 replay: exact reference reproduction
smoke decisions / fixed profiles: 89 / 60
```

The smoke used temporary files only; they were removed after verification. It did not touch any reserved or prior evaluation seed.

### 56.6 Frozen command

After pulling the implementation, run:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and the output path does not exist, run exactly:

```powershell
python -m scripts.diagnose_conflicts --model models/ppo_reward_v3_seed42.zip --model-sha256 f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 10042 --unsafe-ttc 2.0 --cpa-horizon 3.0 --max-range 60.0 --reference-csv results/ppo_reward_v3_holdout_seed10042.csv --reference-csv-sha256 642a0a34841f6bc351acd573b5c057a95c53ddcc2348553de1f10adee0286c09 --output results/ppo_reward_v3_conflict_diagnostics_seed10042.json
```

Commit only `results/ppo_reward_v3_conflict_diagnostics_seed10042.json`, whether it succeeds or preserves a failure. Do not commit the PPO ZIP, rerun with different thresholds, run a shield, train a model, or use a new holdout.

### 56.7 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-089 | Diagnose selective conflict geometry before implementing another shield | Immediately deploy a weaker radial-TTC threshold | The rejected shield had high intervention burden and no conflict-path filter | Retained |
| D-090 | Use consumed seeds 10042–10541 with exact reference reproduction | Spend a new untouched seed block | Threshold development is not a final policy claim | Retained |
| D-091 | Retain every finite nearby closest approach | Store only the globally smallest miss distance | The latter can mask an earlier threshold-matching conflict | Retained |
| D-092 | Freeze 60 profiles and mechanical feasibility/ranking rules | Choose thresholds after freely browsing arbitrary combinations | Constrain development selection and preserve failed feasibility evidence | Retained |
| D-093 | Keep M12A strictly non-interventional | Simulate overrides during threshold selection | Isolate conflict association before testing causal policy changes | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Implemented and froze M12A action-conditioned CPA diagnostic | Design selective caution while controlling success burden and deadlock risk | Geometry/profile tests, reference/failure guards, Ruff, 89 tests, and exact two-episode replay passed; the corrected single-approach design failure was recorded | This implementation update |

**Next action:** pull M12A, rerun Ruff and all tests, run only the frozen non-interventional diagnostic, and commit its JSON. Do not train or evaluate a shield yet.


---

## 57. Milestone M12A result and M12B design — selective mild CPA veto

**Experiment IDs:** E-M12A-V3-CPA-CONFLICT-DIAGNOSTIC-S10042 and E-M12B-V3-MILD-CPA-SHIELD-S40042

**Status:** M12A completed and accepted as development evidence; M12B design frozen, implementation pending

**Date recorded:** 2026-09-07

### 57.1 M12A artifact and integrity verification

The diagnostic result was added alone in commit `7dddf05`. Its SHA-256 is
`b0f052838d44ffd0256c43a0a7f134132bb86dfcaf240163fdda4be3656829ae`.
The committed metadata exactly matches the frozen Section 56 protocol:

| Property | Observed |
|---|---|
| Schema / diagnostic | 1 / `action_conditioned_closest_approach` |
| Non-interventional | `true` |
| PPO SHA-256 | `f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c` |
| Configuration SHA-256 | `433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69` |
| Reference CSV SHA-256 | `642a0a34841f6bc351acd573b5c057a95c53ddcc2348553de1f10adee0286c09` |
| Episodes / seeds | 500 / 10042–10541 |
| CPA horizon / range | 3.0 s / 60.0 m |
| Policy decisions / fixed profiles | 19,053 / 60 |
| Exact reference reproduction | `true` |

The reproduced aggregate is 59.6% success, 40.4% collision, 0% incomplete,
mean reward 2.088917, and mean minimum TTC 0.616385 s. This is an integrity
check of already-consumed development episodes, not a new performance estimate.

### 57.2 Frozen feasibility rule applied mechanically

Seventeen of the 30 `FASTER_ONLY` profiles pass the mild final-two-second
feasibility rule. The unique mechanically ranked winner is:

| Scope | CPA time | Miss distance | Collision coverage | Success burden | Advantage | All-decision trigger rate |
|---|---:|---:|---:|---:|---:|---:|
| `FASTER_ONLY` | 2.0 s | 3.0 m | 159/202 = 78.7129% | 1/298 = 0.3356% | 78.3773 pp | 1,370/19,053 = 7.1905% |

Fourteen of the 30 `FASTER_OR_IDLE` profiles pass the emergency
final-one-second rule. Its mechanically ranked winner uses the same geometric
thresholds and covers 172/202 collision episodes (85.1485%) with 0/298 success
episodes burdened in the final second, an 85.1485-point advantage. Its
all-decision trigger rate is 8.6286%.

For context, the rejected radial-TTC reference triggers on 40.5238% of all
decisions and appears in 297/298 successful episodes. The CPA profiles are much
more selective at decision level. Nevertheless, the selected mild profile
appears at least once anywhere in 177/298 successful episodes and 169/202
collision episodes. The strong terminal-window separation is associative and
does not imply that applying the rule throughout an episode will preserve those
successes.

### 57.3 Interpretation and intervention choice

M12A passes its development feasibility rule and authorizes one causal shield
experiment. It does not prove that either rule prevents collisions. To minimize
the risk of repeating V4 and the old TTC shield's conservative waiting, M12B
tests only the milder winner first:

1. compute every finite nearby CPA with the frozen 3.0-second horizon and
   60.0-metre range;
2. intervene only when the PPO proposes `FASTER` and any approach has
   $t_{CPA}\leq2.0$ s and $d_{CPA}\leq3.0$ m;
3. execute `IDLE` for that decision instead of `FASTER`;
4. do not latch, brake, or alter later actions: the veto releases immediately
   when the conflict clears or the policy proposes any action other than
   `FASTER`.

This one-decision neutral veto is the explicit anti-deadlock behavior. Repeated
independent vetoes remain possible while both the proposed action and geometry
continue to match, so incomplete rate and travel time remain required evidence.
The stronger `FASTER_OR_IDLE`-to-`SLOWER` emergency rule is deferred and must not
be combined with M12B after seeing its result.

### 57.4 Frozen untouched paired evaluation

Seeds **40042–40541** are newly reserved for exactly 500 episodes of the V3
baseline and 500 paired episodes of the mild CPA shield. They do not occur in
the preceding record. Both sides use deterministic final V3 checkpoint
`f46964bf...706c`, configuration `433e6972...fae69`, no intent model, unchanged
reward and traffic settings, collision-first success detection, and the 2.0 s
unsafe-TTC reporting threshold. The only policy-path difference is the frozen
one-step veto above.

The paired experiment passes only if all conditions hold:

1. shield success is at least 3.0 percentage points above the newly measured
   paired V3 baseline;
2. shield collision is at least 3.0 percentage points below baseline;
3. shield incomplete rate is at most 2.0%;
4. shield mean minimum TTC is at least the paired baseline value;
5. the paired success change is favorable with exact two-sided McNemar
   `p < 0.05`.

Mean reward, travel time, unsafe-TTC events, and interventions are descriptive.
If any gate fails, the mild CPA shield is rejected as an improvement and final
200K V3 remains current best. If the paired baseline is again near 60–61%, the
practical success gate requires approximately 63–64%; this is a threshold, not
a forecast.

The two evaluations are one indivisible experiment. Neither output may be used
to retune thresholds, add emergency braking, change the veto action, or decide
whether to run the other side. Both outputs must use exclusive creation and be
committed even if the first visible result is unfavorable.

### 57.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-094 | Accept M12A as valid development evidence | Reject it for no policy intervention | Exact 500-row reproduction, fixed hashes, 19,053 decisions, and all 60 profiles verified | Retained |
| D-095 | Mechanically select the 2.0 s / 3.0 m `FASTER_ONLY` profile | Choose a visually attractive profile after the run | It ranks first under the frozen coverage-minus-burden rule | Retained |
| D-096 | Test only a one-step `FASTER`-to-`IDLE` veto in M12B | Immediately combine mild and emergency rules | The old global shield and V4 both caused conservative failures | Retained |
| D-097 | Reserve paired seeds 40042–40541 and rerun V3 under collision-first detection | Reuse development seeds or compare against an older aggregate | M12A consumed 10042–10541 for design and earlier aggregates used different contexts | Retained |
| D-098 | Reuse the five 3 pp, completion, TTC, and paired-significance gates | Accept a small descriptive gain | Maintain the existing improvement standard | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Completed M12A and froze the M12B mild CPA experiment | Test selective caution after the diagnostic passed its prospective feasibility rule | Single-result-file commit, SHA-256 and metadata, exact reference flag, 60-profile count, mechanical feasibility and ranking, and legacy-reference metrics independently verified | Result: `7dddf05`; documentation: this update |

**Next action:** implement the mild CPA wrapper and evaluation metadata, add focused tests, then pass Ruff and the complete test suite. Do not evaluate seeds 40042–40541 until that implementation and the exact paired commands are recorded.


---

## 58. Milestone M12B implementation — mild CPA acceleration veto

**Experiment ID:** E-M12B-V3-MILD-CPA-SHIELD-S40042

**Status:** Implemented, verified, and frozen; paired evaluation pending

**Date recorded:** 2026-09-07

### 58.1 Implemented policy-path difference

`CPAAccelerationShield` uses the same every-approach geometry as M12A. At each
decision it computes all finite CPAs within 3.0 seconds for traffic initially
within 60.0 metres. If and only if PPO proposes `FASTER` and at least one
approach has CPA time at most 2.0 seconds and miss distance at most 3.0 metres,
the wrapper executes `IDLE`. It does not alter `IDLE`, `SLOWER`, or any other
action and carries no latch or hidden cautious state across decisions.

The existing rejected radial-TTC wrapper remains available under its old
`--safety-shield` flag for reproducibility. The evaluator exposes the new rule
only under the mutually exclusive `--cpa-shield` flag, so the two shields cannot
be combined. A CPA result summary records:

- `safety_shield=true`;
- `safety_shield_type="cpa_acceleration_veto"`;
- CPA time, distance, horizon, and range values;
- the unchanged PPO/configuration fingerprints and all existing protocol
  metadata.

Baseline summaries record `safety_shield=false`, no shield type, and null CPA
parameters. Per-step wrapper information distinguishes proposed and executed
actions, records whether a matching conflict existed, records its time and
distance, and maintains the existing intervention count used by episode CSVs.

### 58.2 Verification before holdout

Focused tests cover the exact boundary match, `FASTER`-to-`IDLE` execution,
non-intervention for `IDLE`, release on nonmatching geometry, intervention-rate
accounting, invalid nonfinite/nonpositive parameters, and a time threshold
beyond the CPA horizon. The environment factory rejects simultaneous TTC and
CPA shields.

The complete gate after implementation passed:

```text
ruff check: passed
pytest: 96 passed in 4.75 s
```

A one-episode real-environment engineering smoke used already-consumed seed 7,
the frozen V3/configuration hashes, and all frozen CPA values. It completed with
success, no collision, 44 decisions, and exactly one intervention. Its summary
correctly recorded `cpa_acceleration_veto` and all four parameters. This single
episode is only an integration check and makes no performance claim. Both
temporary smoke artifacts were deleted; no research result was overwritten and
seeds 40042–40541 were not touched.

### 58.3 Frozen execution protocol

After pulling this implementation, run the complete gate:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and none of the four output paths exists, run both commands as
one paired experiment. Do not stop or modify the second command after observing
the first result.

Baseline:

```powershell
python scripts/evaluate_policy.py --model models/ppo_reward_v3_seed42.zip --model-sha256 f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 40042 --unsafe-ttc 2.0 --output results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --refuse-overwrite
```

Mild CPA shield:

```powershell
python scripts/evaluate_policy.py --model models/ppo_reward_v3_seed42.zip --model-sha256 f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 40042 --unsafe-ttc 2.0 --cpa-shield --cpa-time-threshold 2.0 --cpa-distance-threshold 3.0 --cpa-horizon 3.0 --cpa-max-range 60.0 --output results/ppo_reward_v3_cpa_shield_holdout_seed40042.csv --refuse-overwrite
```

The required small result scope is exactly:

```text
results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv
results/ppo_reward_v3_cpa_baseline_holdout_seed40042.summary.json
results/ppo_reward_v3_cpa_shield_holdout_seed40042.csv
results/ppo_reward_v3_cpa_shield_holdout_seed40042.summary.json
```

Commit all four together regardless of outcome. Do not commit the PPO ZIP,
train anything, rerun either side, substitute seeds, add the emergency rule, or
change a threshold after seeing results.

### 58.4 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-099 | Implement the M12A-selected mild veto exactly | Add braking, a cooldown, intent, or another selected profile | Preserve a single interpretable intervention against V3 | Retained |
| D-100 | Make old TTC and new CPA flags mutually exclusive and record a shield type | Reuse the ambiguous old boolean alone | Prevent accidental shield combination and result misclassification | Retained |
| D-101 | Treat baseline and shield evaluations as one indivisible four-artifact experiment | Inspect baseline before deciding whether to run the shield | Preserve the precommitted paired design | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Implemented and froze the M12B mild CPA shield and paired commands | Causally test the selective M12A association without global braking | Ruff passed, 96 tests passed, and a seed-7 real-environment smoke verified one intervention and complete metadata | This implementation update |

**Next action:** pull this implementation, pass the complete test gate, run both frozen evaluations exactly once, and commit all four small artifacts together. Keep the model local.


---

## 59. Milestone M12B result — mild CPA veto rejected

**Experiment ID:** E-M12B-V3-MILD-CPA-SHIELD-S40042

**Status:** Completed; rejected as an improvement

**Date recorded:** 2026-09-07

### 59.1 Artifact scope and protocol integrity

Commit `15b7c8f` adds exactly the four required artifacts and no other file:

| Artifact | SHA-256 |
|---|---|
| `results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv` | `aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4` |
| `results/ppo_reward_v3_cpa_baseline_holdout_seed40042.summary.json` | `51fe8e4aa86f62b2bbb5b66a7e8979655305e3a03ad571ca631d40828c7556a1` |
| `results/ppo_reward_v3_cpa_shield_holdout_seed40042.csv` | `18a997fd0bca8bd3a724019240d755d56ab3b53a1718fd1812f2aebe890a41ff` |
| `results/ppo_reward_v3_cpa_shield_holdout_seed40042.summary.json` | `f234962ede4d871db94972c87cd6a932bdb326562f54ca68857803c3e9289bbd` |

Both CSVs contain exactly 500 exclusive collision-first outcomes for ordered
seeds 40042–40541. Both summaries match their CSV arithmetic and record the
same deterministic V3 checkpoint SHA-256 `f46964bf...706c`, configuration
SHA-256 `433e6972...fae69`, no intent model, and unsafe-TTC reporting threshold
2.0 seconds. The baseline records no shield. The candidate records
`cpa_acceleration_veto` with time 2.0 s, miss distance 3.0 m, horizon 3.0 s,
and range 60.0 m. No seed, checkpoint, configuration, or protocol drift was
found.

The user reported completing the required Ruff and pytest gate before the
paired run. The result artifacts do not independently encode that console
output.

### 59.2 Aggregate result

| Metric | Paired V3 baseline | Mild CPA shield | Shield minus baseline |
|---|---:|---:|---:|
| Success | 294/500 = 58.8% | 293/500 = 58.6% | -0.2 pp |
| Collision | 206/500 = 41.2% | 207/500 = 41.4% | +0.2 pp |
| Incomplete | 0/500 = 0.0% | 0/500 = 0.0% | 0.0 pp |
| Mean reward | 2.015837 | 1.978920 | -0.036917 |
| Mean length | 37.934 | 37.958 | +0.024 decisions |
| Mean travel time | 7.5868 s | 7.5916 s | +0.0048 s |
| Mean minimum TTC | 0.600366 s | 0.603559 s | +0.003193 s |
| Mean unsafe-TTC events | 16.116 | 16.120 | +0.004 |
| Mean interventions | 0.000 | 2.978 | +2.978 |

The shield made 1,489 recorded interventions across 18,979 decisions, a 7.8455%
decision intervention rate. It intervened in 361/500 episodes and at most 11
times in one episode. There were 705 interventions among shield-success
episodes and 784 among shield-collision episodes. Zero incompletions confirm
that the mild rule avoided the old shield's waiting failure, but it did not
convert that restraint into better terminal outcomes.

### 59.3 Paired outcome analysis

| V3 baseline success | Mild shield success | Episodes |
|---|---|---:|
| True | True | 293 |
| True | False | 1 |
| False | True | 0 |
| False | False | 206 |

Only one success-discordant pair exists. The shield rescued no V3 failures and
regressed one V3 success, so the exact two-sided McNemar probability is 1.0.
Collision transitions are the inverse: 206 episodes collide under both, zero
baseline-only collisions occur, and one shield-only collision occurs. Its exact
two-sided probability is also 1.0.

The sole terminal change is seed 40379. V3 succeeded in 9.0 s with reward
8.656246; the shield made six vetoes and collided after 5.8 s with reward
-8.115289. Its minimum TTC increased from 0.569823 to 1.063304 s, demonstrating
that a higher mean or episode minimum TTC cannot substitute for a collision
outcome gate.

Only 19/500 episodes changed any recorded reward, length, travel-time,
minimum-TTC, or unsafe-event metric even though 361 episodes recorded at least
one veto. HighwayEnv clips `FASTER` at the maximum discrete speed index, so a
`FASTER` proposal and `IDLE` can have identical low-level consequences when the
ego is already saturated. The present artifacts do not record speed index at
each veto, so saturation is a source-supported explanation rather than a
measured attribution. The key empirical result is still unambiguous: this veto
had almost no causal terminal effect.

### 59.4 Precommitted gate decision

| Gate | Required | Observed | Decision |
|---|---:|---:|---|
| Success improvement | at least +3.0 pp | -0.2 pp | Failed |
| Collision reduction | at least -3.0 pp | +0.2 pp | Failed |
| Candidate incomplete rate | at most 2.0% | 0.0% | Passed |
| Mean minimum TTC | candidate at least baseline | 0.603559 > 0.600366 s | Passed |
| Favorable exact paired test | `p < 0.05` | unfavorable; `p = 1.0` | Failed |

Only two of five gates pass. **The mild CPA acceleration veto is rejected as an
improvement.** It must not replace final 200K V3, and its favorable minimum-TTC
change must not be reported as a safety improvement while collision increased.
V3 remains the current best accepted policy.

Seeds 40042–40541 are now consumed. Neither the CPA thresholds nor the action
replacement may be tuned and then claimed on this block as final evidence.

### 59.5 Direction after M12B

M12A found a strong terminal association, but M12B showed that neutralizing a
`FASTER` proposal is usually not a causally potent control change in this
environment. Immediately deploying the M12A emergency rule would confound two
changes—expanding the action scope to `IDLE` and replacing actions with
`SLOWER`—while revisiting the conservative behavior already seen with the old
shield.

The next step should therefore be a single **development-only selective-braking
screen** on consumed seeds. Keep the selected CPA geometry and `FASTER_ONLY`
scope fixed, change only the executed override from `IDLE` to `SLOWER`, and bind
the run to the committed paired V3 baseline. This isolates action potency while
remaining narrower than the rejected `FASTER_OR_IDLE` radial-TTC shield. It may
decide whether another untouched holdout is worth spending, but cannot itself
establish a final improvement.

Before that run, the implementation, exact reference hash, output path, and
development feasibility gates must be recorded. No new holdout is reserved by
this result.

### 59.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-102 | Reject the M12B mild CPA veto and retain V3 | Promote it for slightly higher mean minimum TTC | Success fell, collision rose, and three of five gates failed | Retained |
| D-103 | Treat 1,489 vetoes as recorded interventions, not 1,489 effective control changes | Infer causal potency from wrapper counts | Only 19 episodes changed any reported trajectory metric and one changed terminal outcome | Retained |
| D-104 | Diagnose a `FASTER_ONLY` selective-braking override on consumed seeds next | Add the full emergency action scope or spend a new holdout immediately | Isolate stronger action potency without reintroducing broad idle-action braking | Retained |
| D-105 | Consume seeds 40042–40541 | Retune and reclaim them as untouched | Their paired outcomes directly determined the M12B rejection and next direction | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Completed and rejected M12B mild CPA veto | Test whether selective neutralization improves V3 without conservative waiting | Four-file scope and hashes, 1,000 rows, metadata, aggregate arithmetic, intervention counts, paired transitions, exact tests, and all five gates independently verified | Result: `15b7c8f`; documentation: this update |

**Next action:** implement and freeze one development-only `FASTER`-to-`SLOWER` CPA screen bound to the committed seed-40042 V3 baseline. Do not reserve or evaluate another untouched holdout yet.


---

## 60. Milestone M13 design and implementation — selective-braking development screen

**Experiment ID:** E-M13-V3-FASTER-ONLY-CPA-BRAKE-DEV-S40042

**Status:** Implemented, verified, and frozen; development run pending

**Date recorded:** 2026-09-07

### 60.1 Purpose and fixed single-factor change

M13 tests whether M12B failed because `FASTER`-to-`IDLE` was usually a weak or
clipped control change. It is a development screen, not a new holdout and not a
final safety claim. It preserves every selected conflict condition:

| Property | Frozen value |
|---|---|
| PPO / configuration | Final V3 / unchanged reward V3 |
| Proposed-action scope | `FASTER_ONLY` |
| CPA time threshold | 2.0 s |
| CPA miss-distance threshold | 3.0 m |
| CPA horizon / current range | 3.0 s / 60.0 m |
| Intent model | None |
| Unsafe-TTC reporting threshold | 2.0 s |
| Seeds / episodes | Consumed 40042–40541 / 500 |
| Executed override | `SLOWER` |

The only intervention change relative to M12B is the executed meta-action:
matching `FASTER` proposals become `SLOWER` rather than `IDLE`. The rule still
does not intervene on proposed `IDLE` actions and has no latch. This isolates a
stronger longitudinal response without adopting the broad
`FASTER_OR_IDLE` emergency scope.

The comparator is not rerun. M13 is bound to the committed M12B baseline:

```text
results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv
SHA-256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4
```

The evaluator verifies that fingerprint before loading the environment and
records the reference path and hash in the candidate summary. A mismatch ends
the run before an output is written.

### 60.2 Prospective development feasibility gates

The fixed reference contains 294/500 success, 206/500 collision, no incomplete
episodes, and mean minimum TTC 0.6003656548142169 s. The selective brake is
development-feasible only if all five conditions hold:

1. success is at least 60.8% (304/500), an improvement of at least 2.0 points;
2. collision is at most 39.2% (196/500), a reduction of at least 2.0 points;
3. incomplete rate is at most 2.0%;
4. mean minimum TTC is at least 0.6003656548142169 s;
5. the paired success direction is favorable with exact two-sided McNemar
   `p < 0.10`.

The 2-point and 0.10 levels are a development screen, not relaxed final
acceptance. If all pass, a separately designed untouched comparison must still
use the established final thresholds of at least +3 success points, at least
-3 collision points, at most 2% incomplete, non-worsening mean minimum TTC,
and favorable exact `p < 0.05`. If any development gate fails, reject selective
braking and do not spend another holdout on it.

Only this single candidate is authorized. Do not compare `IDLE` scope, another
override action, another CPA threshold, a cooldown, or a combined shield after
seeing the M13 output.

### 60.3 Implementation and engineering verification

`CPAAccelerationShield` now accepts only `IDLE` or `SLOWER` as its configured
override and retains `IDLE` as the default so M12B remains reproducible. The
evaluator exposes `--cpa-override-action`, verifies an optional reference CSV
fingerprint, and records both fields. `IDLE` outputs retain shield type
`cpa_acceleration_veto`; `SLOWER` outputs use `cpa_selective_brake`.

Focused tests verify `SLOWER` execution and reject unknown override names. The
complete post-implementation gate passed:

```text
ruff check: passed
pytest: 98 passed in 3.29 s
```

The first one-episode seed-7 engineering smoke correctly executed two braking
interventions and verified the reference hash, but exposed that the summary
still used the older `cpa_acceleration_veto` label. No research seed was used
and the temporary output was deleted. The label was corrected before M13; the
complete gate passed again, and a second seed-7 smoke reproduced the same
episode while recording `cpa_selective_brake`, `SLOWER`, and the exact reference
hash. Its two temporary files were also deleted. Neither smoke is performance
evidence.

### 60.4 Frozen command

After pulling the implementation, run:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and neither output exists, run exactly:

```powershell
python scripts/evaluate_policy.py --model models/ppo_reward_v3_seed42.zip --model-sha256 f46964bfac1a21ddc7356aabbaf916b12cb0584295206460d62d3787bd6a706c --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 40042 --unsafe-ttc 2.0 --cpa-shield --cpa-time-threshold 2.0 --cpa-distance-threshold 3.0 --cpa-horizon 3.0 --cpa-max-range 60.0 --cpa-override-action SLOWER --reference-csv results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --reference-csv-sha256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4 --output results/ppo_reward_v3_cpa_brake_development_seed40042.csv --refuse-overwrite
```

Commit exactly the new CSV and summary JSON regardless of outcome. Do not
rerun the V3 baseline, use a new seed, alter a threshold, run a second braking
candidate, or commit the PPO checkpoint.

### 60.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-106 | Screen `FASTER_ONLY`-to-`SLOWER` as the sole M13 candidate | Expand to `IDLE`, tune thresholds, or combine shields | Isolates action potency after the neutral veto changed almost no trajectories | Retained |
| D-107 | Reuse consumed seeds and the exact committed V3 baseline | Spend a new holdout or rerun the comparator | M13 is candidate triage, not a final policy claim | Retained |
| D-108 | Require all five prospective development gates | Advance on a descriptive success increase | Prevent another holdout for a weak or unsafe signal | Retained |
| D-109 | Give selective braking a distinct summary type | Reuse the mild-veto label | Preserve unambiguous artifact provenance | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Implemented and froze M13 selective-braking development screen | Test whether a causally stronger but still action-selective override merits another holdout | Ruff and 98 tests passed; two seed-7 smokes verified execution/reference binding and exposed then confirmed correction of the initial shield-type label | This implementation update |

**Next action:** pull this implementation, pass the complete test gate, run only the frozen M13 development command, and commit its two small outputs. Do not run a new holdout.


---

## 61. Milestone M13 result — selective braking rejected

**Experiment ID:** E-M13-V3-FASTER-ONLY-CPA-BRAKE-DEV-S40042

**Status:** Completed; failed development feasibility and rejected

**Date recorded:** 2026-09-07

### 61.1 Artifact and protocol verification

Commit `fd77a6d` adds exactly the required candidate CSV and summary:

| Artifact | SHA-256 |
|---|---|
| `results/ppo_reward_v3_cpa_brake_development_seed40042.csv` | `64d32187cb0a4407eba2d2e9bd8dc6896a913487143fed6b8df08c9511ac52a3` |
| `results/ppo_reward_v3_cpa_brake_development_seed40042.summary.json` | `20a09d44d7cb771e8d445e8fe3e78458930a362800618c5e6b65756d5c198f1f` |

The CSV contains 500 exclusive collision-first outcomes for seeds
40042–40541. The summary arithmetic matches the episode rows. It records final
V3 SHA-256 `f46964bf...706c`, configuration SHA-256
`433e6972...fae69`, no intent model, `cpa_selective_brake`, `SLOWER`, the
frozen 2.0 s / 3.0 m / 3.0 s / 60.0 m geometry, and unsafe-TTC reporting at
2.0 seconds. It also records the exact committed reference path and SHA-256
`aab91174...a47fb4`. No protocol drift was found.

The user reported completing Ruff and pytest before the run. As with M12B,
the CSV and summary do not independently contain that console output.

### 61.2 Aggregate development result

| Metric | Committed V3 reference | Selective brake | Brake minus reference |
|---|---:|---:|---:|
| Success | 294/500 = 58.8% | 261/500 = 52.2% | -6.6 pp |
| Collision | 206/500 = 41.2% | 239/500 = 47.8% | +6.6 pp |
| Incomplete | 0/500 = 0.0% | 0/500 = 0.0% | 0.0 pp |
| Mean reward | 2.015837 | 0.696949 | -1.318888 |
| Mean length | 37.934 | 39.542 | +1.608 decisions |
| Mean travel time | 7.5868 s | 7.9084 s | +0.3216 s |
| Mean minimum TTC | 0.600366 s | 0.682174 s | +0.081808 s |
| Mean unsafe-TTC events | 16.116 | 16.944 | +0.828 |
| Mean interventions | 0.000 | 3.532 | +3.532 |

The selective brake made 1,766 interventions over 19,771 policy decisions, an
8.9323% intervention rate. It intervened in 361/500 episodes and reached 16
interventions in one episode. Unlike the weak M12B veto, braking materially
changed paired trajectories, but its net effect was harmful.

### 61.3 Paired transitions and exact test

| V3 reference success | Selective-brake success | Episodes |
|---|---|---:|
| True | True | 240 |
| True | False | 54 |
| False | True | 21 |
| False | False | 185 |

The brake rescues 21 V3 collisions but destroys 54 V3 successes, a net loss of
33 successes across 75 discordant episodes. The exact two-sided McNemar
probability is `p = 0.000176309`. The difference is statistically clear in the
unfavorable direction. Collision transitions are exactly inverse and have the
same probability.

This result also reinforces the prior warning about TTC summaries. Mean minimum
TTC improves by 0.0818 seconds while collision rises by 6.6 points. Delaying or
changing intersection entry can increase the recorded closest pass before a
later collision; minimum TTC is not a surrogate acceptance outcome.

### 61.4 Frozen development-gate decision

| Gate | Required | Observed | Decision |
|---|---:|---:|---|
| Success | at least 60.8% | 52.2% | Failed |
| Collision | at most 39.2% | 47.8% | Failed |
| Incomplete | at most 2.0% | 0.0% | Passed |
| Mean minimum TTC | at least 0.600366 s | 0.682174 s | Passed |
| Favorable exact paired test | `p < 0.10` | unfavorable; `p = 0.000176309` | Failed |

Only two of five gates pass. **Selective CPA braking is rejected.** Per the
prospective routing rule, no untouched holdout may be spent on it. Do not tune
the brake action, CPA thresholds, action scope, persistence, or cooldown using
these results.

Final 200K PPO V3 remains the current best accepted policy. The radial-TTC
shield, mild CPA veto, and selective CPA brake are all rejected for different
failure modes: conservative incompletion, negligible causal effect, and
significant collision worsening, respectively.

### 61.5 Direction after hand-written shielding

The repeated shield failures show that constant-velocity conflict geometry is
useful for retrospective association but not sufficient to choose safe
intersection actions under reactive multi-vehicle dynamics. Further manual
threshold searches on consumed seeds would overfit the same outcomes and are
not authorized.

The next success-oriented direction is to improve the learned base policy's
robustness. Before another long run, audit and freeze a multi-environment,
multi-seed PPO V3 training protocol with more experience and a larger internal
development evaluation. Keep the V3 reward, observation, traffic, and action
spaces fixed so training diversity and budget are the controlled changes. The
new policy must first pass a paired development screen before any untouched
holdout is reserved. No training is authorized until that implementation,
seed schedule, budget, checkpoint rule, and gates are recorded.

### 61.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-110 | Reject M13 selective braking | Advance it for higher mean minimum TTC | Success fell 6.6 points, collision rose 6.6 points, and the paired change was significantly unfavorable | Retained |
| D-111 | End manual CPA/TTC shield tuning on consumed outcomes | Try another threshold, scope, or cooldown | Three shield designs failed through incompletion, non-effect, or collision worsening | Retained |
| D-112 | Improve the learned V3 base policy next | Reserve a holdout for another hand-written override | V3 remains stable near 59–61%, while post-processing has not improved it | Retained |
| D-113 | Require a multi-seed PPO candidate to pass development before a new holdout | Train and immediately claim on untouched seeds | Control compute and holdout expenditure after repeated negative experiments | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Completed and rejected M13 selective-braking development screen | Test whether stronger action potency could turn M12A association into improved outcomes | Two-file scope and hashes, reference binding, 500 rows, metadata, aggregate arithmetic, 75 paired transitions, exact tests, and all gates independently verified | Result: `fd77a6d`; documentation: this update |

**Next action:** design and implement a multi-environment, multi-seed PPO V3 training protocol with fixed reward/observation/action settings and prospective development gates. Do not train or evaluate another shield.


---

## 62. Milestone M14A design and implementation — physics-informed sensor fusion

**Experiment ID:** E-M14A-PPO-V3-KINEMATIC-RISK-FUSION-S42

**Status:** Implemented, verified, and frozen; controlled training pending

**Date recorded:** 2026-09-07

### 62.1 Revised direction and terminology

After M13, the user proposed improving sensors and using sensor fusion. This is
a stronger direction than another hand-written shield, but the simulator scope
must be stated accurately. HighwayEnv currently exposes exact normalized
kinematics rather than raw, noisy camera, LiDAR, or radar measurements. M14A is
therefore **feature-level kinematic risk fusion**, not a claim of real-world
multimodal perception.

The new observation combines three information groups:

1. the complete existing 15-by-7 kinematics observation;
2. radar-like range and radial closing speed for each observed traffic slot;
3. derived radial-TTC and constant-velocity closest-approach time/distance.

Unlike the rejected shields, these features never override an action. PPO can
learn when and how to use them jointly with the native state.

### 62.2 Frozen observation contract

The native 105 values remain first and in their original order. For each of the
14 sorted non-ego traffic slots, append this five-value block:

| Index | Feature | Normalization |
|---:|---|---|
| 0 | Euclidean range | `clip(range / 200 m, 0, 1)` |
| 1 | Radial closing speed | `clip(closing_speed / 20 m/s, -1, 1)` |
| 2 | Radial TTC | `clip(TTC / 10 s, 0, 1)`; receding/nonfinite = 1 |
| 3 | CPA time | `clip(t_CPA / 5 s, 0, 1)`; invalid/outside horizon = 1 |
| 4 | CPA miss distance | `clip(d_CPA / 20 m, 0, 1)`; invalid = 1 |

The relative geometry uses the same formulas recorded in Section 56, but the
network receives continuous normalized values rather than a threshold decision.
A missing traffic slot uses `[1, 0, 1, 1, 1]`, representing far/no closing/no
finite conflict; the corresponding native presence bit remains zero.

The fused observation dimension is fixed at

$$
15\times7 + 14\times5 = 175.
$$

Slot order exactly follows HighwayEnv's existing sorted kinematics query.
Observation values after the native block are float32 and bounded by the
declared space. The fusion wrapper requires sorted slots and refuses invalid
neighbor counts or nonfinite/nonpositive scales.

No sensor noise, occlusion, missed detection, calibration error, or learned
perception is modeled. Those require a separately scoped CARLA or synthetic
sensor experiment and cannot be inferred from M14A.

### 62.3 Controlled training protocol

To isolate the observation change, the first fusion policy retains the original
V3 training protocol rather than simultaneously increasing data or changing
PPO:

| Property | V3 reference | Fusion V1 |
|---|---:|---:|
| Configuration SHA-256 | `433e6972...fae69` | same |
| Reward / traffic / action | V3 | same |
| Training seed | 42 | 42 |
| Total requested steps | 200,000 | 200,000 |
| Environments | 1 | 1 |
| Learning rate | 0.0003 | 0.0003 |
| PPO `n_steps` / batch | 1024 / 64 | 1024 / 64 |
| Gamma / GAE / entropy | 0.99 / 0.95 / 0.01 | same |
| Network | `[256, 256]` | same |
| Observation | 105 kinematics | 175 fused |
| Intent / shield | none / none | none / none |

Internal monitoring uses offset 70,000, 50 episodes every 10,000 total steps,
and is descriptive. Only the final checkpoint is eligible; the callback-best
and periodic checkpoints must not be substituted after inspecting evaluation
curves. The different internal monitoring size/offset does not update PPO or
select the final artifact.

The trainer also now supports explicit independent environment seed streams and
converts total-timestep callback intervals to vector-step calls. M14A fixes
`n_envs=1`; the optional multi-environment path is infrastructure only and is
not part of this controlled fusion comparison.

Expected outputs:

```text
models/ppo_fusion_v1_seed42.zip
results/ppo_fusion_v1_seed42.training.json
```

The ZIP stays local and ignored. Commit only the small training JSON, whether
training succeeds or fails. Training completion is an integrity gate, not an
improvement result. Before a driving evaluation is authorized, the summary and
local ZIP must show 200,704 collected steps, observation shape `[175]`, the
frozen configuration and all fusion parameters, seed 42, one environment, no
intent, no shield, and matching model fingerprint.

### 62.4 Implementation verification

Unit tests cover feature values on an analytic crossing trajectory, native
observation preservation, missing-slot sentinels, observation-space bounds,
sorted-slot and neighbor-count requirements, invalid scaling, deterministic
training seed offsets, and vector callback-frequency accounting.

The complete gate passed:

```text
ruff check: passed
pytest: 114 passed in 3.81 s
```

Two real-environment checks used only engineering seed 7:

- native and fused environments matched exactly for the first 105 reset values
  and through ten identical `IDLE` steps, including reward, termination, and
  ego position; the added 70 values were finite and within `[0,1]` in that run;
- a 32-step PPO smoke produced a valid `[175]` model and summary with fusion
  enabled and 14 neighbors.

All temporary smoke models, summaries, callback outputs, and logs were deleted.
No research result, reserved seed, or existing artifact was touched.

### 62.5 Frozen training command

After pulling the implementation, run:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and neither output exists, run exactly:

```powershell
python -m scripts.train_ppo --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --timesteps 200000 --seed 42 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-envs 1 --env-seed-stride 1000 --eval-seed-offset 70000 --eval-episodes 50 --evaluation-freq 10000 --checkpoint-freq 25000 --risk-fusion --fusion-neighbors 14 --fusion-range-scale 200.0 --fusion-relative-speed-scale 20.0 --fusion-ttc-scale 10.0 --fusion-cpa-horizon 5.0 --fusion-cpa-distance-scale 20.0 --summary-output results/ppo_fusion_v1_seed42.training.json --output models/ppo_fusion_v1_seed42 --refuse-overwrite
```

After it finishes, stop and commit only
`results/ppo_fusion_v1_seed42.training.json`. Do not evaluate the policy, use a
different checkpoint, increase training, run a shield, add intent, or commit
the ZIP. A paired development evaluation will be frozen only after validating
the completed checkpoint.

### 62.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-114 | Replace manual shielding with learned kinematic-risk fusion | Tune another CPA/TTC override | Every tested shield failed while fused features leave control authority with PPO | Retained |
| D-115 | Describe M14A as feature-level fusion, not raw sensor fusion | Claim camera/LiDAR/radar fusion | HighwayEnv supplies simulator kinematics without a perception stack | Retained |
| D-116 | Preserve all 105 native values and append 70 normalized risk values | Replace kinematics or use threshold flags | Retain full state while making interaction geometry explicit and learnable | Retained |
| D-117 | Hold V3 training seed, budget, reward, PPO, and action settings fixed | Combine fusion with more training or multiple environments immediately | Isolate the observation representation as the causal change | Retained |
| D-118 | Accept only the final 200K checkpoint for later screening | Select callback-best from noisy internal reward | M11 showed the original reward-selected checkpoint did not generalize | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Implemented and froze M14A physics-informed feature fusion and reproducible trainer controls | Give PPO continuous interaction-risk features without another hand-written override | Ruff, 114 tests, ten-step paired non-interference check, and 32-step `[175]` PPO smoke passed; all temporary artifacts removed | This implementation update |

**Next action:** pull this implementation, pass the complete test gate, run only the frozen M14A training command, and commit its JSON summary while keeping the model local.


---

## 63. Milestone M14A training result and M14B development protocol

**Experiment IDs:** E-M14A-PPO-V3-KINEMATIC-RISK-FUSION-S42 and E-M14B-FUSION-V1-DEV-S40042

**Status:** Training completed and checkpoint validated; paired development evaluation pending

**Date recorded:** 2026-09-07

### 63.1 Training artifact verification

Commit `1d27c6c` adds only
`results/ppo_fusion_v1_seed42.training.json`. Its SHA-256 is
`62713bca9d6c9c2bf5df15b9d179e447c882c065f0ee8854fd0d9e9ea8404ec7`.
The local checkpoint remains ignored and has SHA-256
`690d90d738892c1428c6b1eb4f85769ef39f3827daaa96deb9078155a84644e3`,
exactly matching the committed summary.

| Training property | Required | Observed | Decision |
|---|---:|---:|---|
| Configuration SHA-256 | `433e6972...fae69` | exact | Passed |
| Training seed / environments | 42 / 1 | 42 / 1 | Passed |
| Requested / collected steps | 200,000 / 200,704 | 200,000 / 200,704 | Passed |
| Learning rate | 0.0003 | 0.0003 | Passed |
| `n_steps` / batch size | 1024 / 64 | 1024 / 64 | Passed |
| Gamma / GAE / entropy | 0.99 / 0.95 / 0.01 | exact | Passed |
| Policy network | `[256, 256]` | `[256, 256]` | Passed |
| Observation shape | `[175]` | `[175]` | Passed |
| Fusion slots/features | 14 / 5 | 14 / 5 | Passed |
| Fusion scales | 200 / 20 / 10 / 5 / 20 | exact | Passed |
| Intent / safety shield | none / false | none / false | Passed |
| Internal evaluation | offset 70,000; 50 every 10,000 | exact | Passed |

Independent loading confirmed 200,704 stored timesteps, a 175-value Box
observation, three discrete actions, the frozen PPO hyperparameters, and finite
policy parameters. A deterministic seed-7 reset, prediction, and step succeeded
with the production fusion environment. These are integrity checks, not driving
performance evidence.

The final 200K checkpoint is accepted as the sole M14B candidate. Callback-best
and periodic fusion checkpoints may not be inspected or substituted after any
development result is visible.

### 63.2 Frozen paired development comparison

M14B uses the already-consumed seeds 40042–40541 and the exact committed V3
baseline rather than spending a new holdout:

```text
results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv
SHA-256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4
```

That reference contains 58.8% success, 41.2% collision, 0% incomplete, and mean
minimum TTC 0.6003656548142169 s. The candidate uses the same configuration,
reward, traffic, action space, deterministic evaluation, collision-first outcome
rule, and unsafe-TTC reporting threshold. It adds only the frozen 175-value
fusion observation. No intent model or safety shield is used.

Because these seeds informed M12B/M13 and are development-only, passing M14B
would authorize a separately frozen untouched holdout but would not establish a
final improvement.

### 63.3 Prospective development gates

Fusion V1 is development-feasible only if all five conditions hold:

1. success is at least 60.8% (304/500), at least +2.0 points over reference;
2. collision is at most 39.2% (196/500), at least -2.0 points from reference;
3. incomplete rate is at most 2.0%;
4. mean minimum TTC is at least 0.6003656548142169 s;
5. the paired success direction is favorable with exact two-sided McNemar
   `p < 0.10`.

These are development gates only. A later untouched holdout would retain the
stricter final +3/-3-point, 2% incomplete, non-worsening TTC, and favorable
`p < 0.05` rules. Mean reward, travel time, and unsafe-TTC events are
descriptive and cannot rescue a failed gate.

If any M14B gate fails, Fusion V1 is rejected. Do not train longer, select a
stored checkpoint, change a feature scale, remove a channel, or reuse the
development outputs to claim final performance.

### 63.4 Frozen command

First rerun the complete test gate:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and neither output exists, run exactly:

```powershell
python scripts/evaluate_policy.py --model models/ppo_fusion_v1_seed42.zip --model-sha256 690d90d738892c1428c6b1eb4f85769ef39f3827daaa96deb9078155a84644e3 --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 40042 --unsafe-ttc 2.0 --risk-fusion --fusion-neighbors 14 --fusion-range-scale 200.0 --fusion-relative-speed-scale 20.0 --fusion-ttc-scale 10.0 --fusion-cpa-horizon 5.0 --fusion-cpa-distance-scale 20.0 --reference-csv results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --reference-csv-sha256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4 --output results/ppo_fusion_v1_development_seed40042.csv --refuse-overwrite
```

Commit exactly the new CSV and summary JSON regardless of outcome. Keep the
fusion and V3 ZIPs local. Do not rerun the baseline, evaluate a different
checkpoint, use new seeds, change fusion scales, or start another training run.

### 63.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-119 | Accept final Fusion V1 checkpoint for one development comparison | Retrain, train longer, or select callback-best | Summary and local ZIP match every frozen structural and provenance requirement | Retained |
| D-120 | Use the consumed seed-40042 V3 baseline for M14B | Spend a new holdout immediately | Screen representation value before consuming new final evidence | Retained |
| D-121 | Require all five prospective development gates | Advance based on reward or an unpaired aggregate | Preserve a meaningful completion/safety signal before final evaluation | Retained |
| D-122 | Forbid fusion checkpoint selection after M14B begins | Browse periodic or callback-best models | M11 demonstrated non-generalization from small reward-based checkpoint selection | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Completed and accepted M14A training for one paired development evaluation | Validate the controlled 175-feature fusion policy before spending a holdout | Single-summary commit, summary/ZIP hashes, all training fields, checkpoint spaces/timesteps/hyperparameters/finiteness, and seed-7 inference independently verified | Result: `1d27c6c`; documentation: this update |

**Next action:** pass the complete test gate, run only the frozen M14B command, and commit its two small outputs. Do not run a new holdout or another training job.


---

## 64. Milestone M14B result and M14C design — promising fusion with excess stalls

**Experiment IDs:** E-M14B-FUSION-V1-DEV-S40042 and E-M14C-FUSION-V2-MULTIENV-S42

**Status:** Fusion V1 failed one development gate and is rejected as-is; Fusion V2 training frozen

**Date recorded:** 2026-09-07

### 64.1 Artifact and protocol verification

Commit `380d517` adds exactly the two required development artifacts:

| Artifact | SHA-256 |
|---|---|
| `results/ppo_fusion_v1_development_seed40042.csv` | `1c54c4d36525bd84daf82c2773729aee3e4429c68db83d5ec67563b86605d632` |
| `results/ppo_fusion_v1_development_seed40042.summary.json` | `ef286f04a0a7bca4d40059c2bd7baadbe6f33ac541cc8c0a926579a2a783b7da` |

The CSV contains exactly 500 exclusive collision-first outcomes for consumed
seeds 40042–40541. Its arithmetic matches the summary. The summary records the
frozen Fusion V1 model SHA-256 `690d90d7...644e3`, configuration SHA-256
`433e6972...fae69`, exact V3 reference CSV SHA-256
`aab91174...a47fb4`, all five fixed fusion scales, 14 slots, five features per
slot, no intent, and no shield. No protocol drift was found.

The user reported completing Ruff and pytest before evaluation. The result
files do not independently encode the console output.

### 64.2 Aggregate result

| Metric | V3 development reference | Fusion V1 | Fusion minus V3 |
|---|---:|---:|---:|
| Success | 294/500 = 58.8% | 315/500 = 63.0% | +4.2 pp |
| Collision | 206/500 = 41.2% | 174/500 = 34.8% | -6.4 pp |
| Incomplete | 0/500 = 0.0% | 11/500 = 2.2% | +2.2 pp |
| Mean reward | 2.015837 | 2.816211 | +0.800374 |
| Mean length | 37.934 | 41.668 | +3.734 decisions |
| Mean travel time | 7.5868 s | 8.3336 s | +0.7468 s |
| Mean minimum TTC | 0.600366 s | 0.628087 s | +0.027721 s |
| Mean unsafe-TTC events | 16.116 | 16.486 | +0.370 |
| Mean interventions | 0.000 | 0.000 | 0.000 |

Fusion V1 is the first candidate in this record to exceed both the practical
success and collision effect sizes with a favorable paired result, but its
completion defect is real and prospectively disqualifying.

### 64.3 Paired transition analysis

The complete outcome transition matrix is:

| V3 outcome | Fusion success | Fusion collision | Fusion incomplete |
|---|---:|---:|---:|
| Success | 280 | 12 | 2 |
| Collision | 35 | 162 | 9 |
| Incomplete | 0 | 0 | 0 |

For binary success, Fusion V1 gains 35 successes and loses 14 across 49
discordant episodes, a net gain of 21. The exact two-sided McNemar probability
is `p = 0.003801654`. For collision, it removes 44 V3 collisions and introduces
12, with exact `p = 0.000020877`. Both effects are favorable and statistically
clear on this development set.

Nine of the 11 incomplete episodes replace V3 collisions, while two replace V3
successes. Every incomplete episode lasts 151 decisions / 30.2 seconds. Their
minimum TTC values range from 0.757 to 3.024 seconds, so they are not a single
near-collision edge case.

A read-only replay of sampled incomplete seeds under the frozen candidate
confirmed a stall mechanism. Seed 40181 chose 134 `IDLE`, 13 `SLOWER`, and four
`FASTER` actions; seed 40382 chose 131 `IDLE`, 13 `SLOWER`, and seven `FASTER`.
Both ended at speed 0 with zero motion during the last 20 decisions. These
already-consumed replays produced no artifact and did not change the policy.

### 64.4 Frozen development-gate decision

| Gate | Required | Observed | Decision |
|---|---:|---:|---|
| Success | at least 60.8% | 63.0% | Passed |
| Collision | at most 39.2% | 34.8% | Passed |
| Incomplete | at most 2.0% | 2.2% | Failed |
| Mean minimum TTC | at least 0.600366 s | 0.628087 s | Passed |
| Favorable exact paired test | `p < 0.10` | `p = 0.003801654` | Passed |

Four of five gates pass. The incomplete ceiling is missed by exactly one
episode, but the gate was frozen and cannot be rounded or relaxed after the
result. **Fusion V1 is rejected as-is and no untouched holdout is authorized.**
It must not replace V3 or be described as an accepted improvement.

The representation direction remains development-promising because both task
effect sizes and paired tests passed strongly. The next controlled attempt may
improve training robustness while keeping the fusion features and reward fixed;
it may not tune feature scales on these outcomes.

### 64.5 Frozen Fusion V2 training protocol

Fusion V2 retains all M14A feature definitions, configuration, reward, action
space, network, PPO coefficients, and seed-42 anchor. It changes the training
data protocol and budget together as one predefined robustness package:

| Property | Fusion V1 | Fusion V2 |
|---|---:|---:|
| Requested total steps | 200,000 | 500,000 |
| Environments | 1 | 4 |
| Initial environment seeds | `[42]` | `[42, 1042, 2042, 3042]` |
| Per-environment `n_steps` | 1024 | 256 |
| Total rollout size | 1024 | 1024 |
| Batch size | 64 | 64 |
| Learning rate | 0.0003 | 0.0003 |
| Expected collected steps | 200,704 | 500,736 |
| Internal evaluation | 50 every 10K; offset 70K | 100 every 25K; offset 80K |
| Eligible checkpoint | final only | final only |

Using four streams changes which experience enters a rollout while preserving
the total rollout size and optimizer batch. The larger budget supplies 2.5
times as much experience. These two training changes are intentionally bundled
as a single development candidate; M14C cannot distinguish their individual
effects.

The output paths are:

```text
models/ppo_fusion_v2_multienv_seed42.zip
results/ppo_fusion_v2_multienv_seed42.training.json
```

Keep the ZIP local and commit only the JSON. The final checkpoint is eligible
for a development comparison only if the summary and local ZIP show the exact
configuration/fusion settings, initial seeds, four environments, rollout 1024,
500,736 collected steps, no intent/shield, finite parameters, and matching
fingerprint. Callback-best and periodic checkpoints remain ineligible.

### 64.6 Engineering verification and frozen command

The four-environment trainer path passed a 32-total-step engineering smoke with
fusion observation `[175]`, initial seeds `[7,1007,2007,3007]`, rollout size 32,
and callback frequency correctly divided by four. All temporary checkpoint,
summary, callback, and log artifacts were removed. The prior complete gate was
Ruff clean with 114 tests passing.

After pulling this record, rerun:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and neither output exists, run exactly:

```powershell
python -m scripts.train_ppo --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --timesteps 500000 --seed 42 --learning-rate 0.0003 --n-steps 256 --batch-size 64 --n-envs 4 --env-seed-stride 1000 --eval-seed-offset 80000 --eval-episodes 100 --evaluation-freq 25000 --checkpoint-freq 50000 --risk-fusion --fusion-neighbors 14 --fusion-range-scale 200.0 --fusion-relative-speed-scale 20.0 --fusion-ttc-scale 10.0 --fusion-cpa-horizon 5.0 --fusion-cpa-distance-scale 20.0 --summary-output results/ppo_fusion_v2_multienv_seed42.training.json --output models/ppo_fusion_v2_multienv_seed42 --refuse-overwrite
```

After training, stop. Commit only the training JSON regardless of completion or
failure. Do not evaluate V2, substitute a checkpoint, change reward/features,
run a shield, or reserve a new holdout until the final ZIP is audited.

### 64.7 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-123 | Reject Fusion V1 as-is | Relax 2.0% incomplete ceiling to accept a strong aggregate | The precommitted completion gate failed at 2.2% | Retained |
| D-124 | Preserve fusion as a promising development direction | Abandon fusion because one gate failed | Success +4.2 pp, collision -6.4 pp, and paired `p=0.00380` all passed | Retained |
| D-125 | Keep feature scales and V3 reward fixed for Fusion V2 | Tune fusion thresholds or add a post-policy shield | Isolate training robustness without feedback-tuning the representation | Retained |
| D-126 | Use four seed streams, rollout 1024, and 500K steps as one frozen training package | Select seed count or duration after another outcome | Increase experience diversity and budget prospectively | Retained |
| D-127 | Continue to require the final checkpoint only | Inspect Fusion V1/V2 callback-best models | Avoid repeating M11 checkpoint-selection failure | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Completed M14B, rejected Fusion V1 as-is, and froze Fusion V2 multi-environment training | Preserve the one-episode gate failure while testing whether more diverse experience removes genuine stalls | Two-file scope/hashes, reference and fusion metadata, 500 rows, aggregate and 3-by-3 paired transitions, exact tests, all gates, sampled stall replays, and four-environment smoke verified | Result: `380d517`; documentation: this update |

**Next action:** pass the complete test gate, run only the frozen Fusion V2 training command, and commit its training JSON while keeping all checkpoints local.


---

## 65. Milestone M14C training result and M14D development protocol

**Experiment IDs:** E-M14C-FUSION-V2-MULTIENV-S42 and E-M14D-FUSION-V2-DEV-S40042

**Status:** Fusion V2 training completed and validated; paired development evaluation pending

**Date recorded:** 2026-09-07

### 65.1 Training artifact and checkpoint audit

Commit `86a0c9e` adds only
`results/ppo_fusion_v2_multienv_seed42.training.json`. Its SHA-256 is
`0f8f080f31f6026af2c02d41ffa68709939996ada9c9172022db97c2099e869f`.
The local final checkpoint remains ignored and has SHA-256
`7253cf4a64785bd9d851bf91dbe3c5c2b96d159910d238f0eb5640fea7cb38b9`,
matching the summary exactly.

| Training property | Frozen value | Observed | Decision |
|---|---:|---:|---|
| Configuration SHA-256 | `433e6972...fae69` | exact | Passed |
| Anchor seed | 42 | 42 | Passed |
| Environments | 4 | 4 | Passed |
| Initial seeds | 42, 1042, 2042, 3042 | exact | Passed |
| Requested / collected steps | 500,000 / 500,736 | exact | Passed |
| Per-env `n_steps` / rollout | 256 / 1024 | 256 / 1024 | Passed |
| Batch / learning rate | 64 / 0.0003 | exact | Passed |
| Gamma / GAE / entropy | 0.99 / 0.95 / 0.01 | exact | Passed |
| Observation shape | `[175]` | `[175]` | Passed |
| Fusion slots/features | 14 / 5 | 14 / 5 | Passed |
| Fusion scales | 200 / 20 / 10 / 5 / 20 | exact | Passed |
| Internal evaluation | offset 80K; 100 every 25K | exact | Passed |
| Intent / shield | none / false | none / false | Passed |

Independent loading confirmed 500,736 stored timesteps, stored `n_envs=4`, a
175-value Box observation, three discrete actions, `n_steps=256`, batch 64,
the fixed PPO coefficients, and finite policy parameters. A deterministic
seed-7 production-fusion reset, prediction, and step succeeded. None of these
checks measure success or collision performance.

The final checkpoint is accepted as the sole M14D candidate. Callback-best and
periodic checkpoints are ineligible regardless of their internal reward.

### 65.2 Frozen M14D comparison

M14D evaluates final Fusion V2 once on the consumed seeds 40042–40541 and binds
the run to the same committed V3 baseline SHA-256
`aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4`.
The evaluation uses deterministic actions, collision-first outcomes, the
unchanged V3 configuration, all fixed fusion scales, no intent model, no
shield, and unsafe-TTC reporting at 2.0 seconds.

This is development evidence. No untouched holdout is authorized until M14D
passes. The evaluation cannot be used to choose a stored training checkpoint,
feature scale, training seed, or further duration.

### 65.3 Prospective advancement gates

Fusion V2 was designed to remove Fusion V1's stall excess without giving back
its strong effect. It advances only if all conditions hold:

1. success is at least Fusion V1's 63.0% (315/500);
2. collision is at most Fusion V1's 34.8% (174/500);
3. incomplete rate is at most 2.0% (no more than 10/500);
4. mean minimum TTC is at least the V3 reference's 0.6003656548142169 s;
5. relative to V3, paired success change is favorable with exact two-sided
   McNemar `p < 0.05`.

The first two gates deliberately preserve Fusion V1's observed task effect
rather than reverting to the weaker original development floors. This rule is
fixed before viewing any Fusion V2 policy result. Mean reward, travel time,
unsafe-TTC events, and internal evaluation curves are descriptive.

If any gate fails, reject Fusion V2 and do not inspect other checkpoints or
retune training on seeds 40042–40541. If all pass, reserve a new untouched
500-episode paired V3/Fusion V2 holdout with the established final +3/-3-point,
2% incomplete, non-worsening TTC, and favorable `p < 0.05` rules.

### 65.4 Frozen command

First run:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only if both pass and neither output exists, run exactly:

```powershell
python scripts/evaluate_policy.py --model models/ppo_fusion_v2_multienv_seed42.zip --model-sha256 7253cf4a64785bd9d851bf91dbe3c5c2b96d159910d238f0eb5640fea7cb38b9 --config configs/intersection_reward_v3.yaml --config-sha256 433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69 --episodes 500 --seed 40042 --unsafe-ttc 2.0 --risk-fusion --fusion-neighbors 14 --fusion-range-scale 200.0 --fusion-relative-speed-scale 20.0 --fusion-ttc-scale 10.0 --fusion-cpa-horizon 5.0 --fusion-cpa-distance-scale 20.0 --reference-csv results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --reference-csv-sha256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4 --output results/ppo_fusion_v2_development_seed40042.csv --refuse-overwrite
```

Commit exactly the new CSV and summary JSON regardless of outcome. Keep every
checkpoint local. Do not evaluate Fusion V1 again, rerun V3, substitute a V2
checkpoint, change fusion parameters, or use a new seed block.

### 65.5 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-128 | Accept final Fusion V2 for one development evaluation | Inspect callback-best or periodic checkpoints | Training summary and local final ZIP pass every frozen integrity check | Retained |
| D-129 | Require V2 to preserve V1's 63.0% success and 34.8% collision | Reuse only the weaker +2/-2-point floors | V2 exists specifically to fix stalls without losing V1's task effect | Retained |
| D-130 | Keep the 2.0% incomplete ceiling unchanged | Round 2.2% down or allow one extra episode | Completion was the sole V1 failure and must be fixed prospectively | Retained |
| D-131 | Continue using consumed seeds for M14D | Spend a new holdout before V2 screening | Candidate selection remains development work | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-07 | Completed and accepted M14C training for one strict Fusion V2 development comparison | Test whether multi-stream 500K training removes V1 stalls without losing its gains | Single-summary scope/hash, exact training fields, local ZIP fingerprint, stored spaces/timesteps/n-envs/PPO settings/finiteness, and seed-7 inference verified | Result: `86a0c9e`; documentation: this update |

**Next action:** pass the complete test gate, run only the frozen M14D command, and commit its two small artifacts. Do not evaluate a holdout or any other checkpoint.


---

## 66. Milestone M14D result — Fusion V2 removes incompletion but loses driving gains

**Experiment ID:** E-M14D-FUSION-V2-DEV-S40042

**Status:** Completed; Fusion V2 rejected as an improvement; V3 remains current best accepted policy

**Date recorded:** 2026-09-08

### 66.1 Result artifacts and protocol verification

Result commit `1f25f4278b4425b02c7909d177b976f5d1ca5ec0` adds exactly the
500-row CSV and its summary JSON. At the start of this review the commit was
local while remote `main` remained at `0b22160`; the result and this analysis
are to be pushed together. Neither checkpoint is tracked.

| Artifact | SHA-256 |
|---|---|
| `results/ppo_fusion_v2_development_seed40042.csv` | `4d5100a13ee82ceff4f1114440e51178f0a3ee940a71166818a6a23ef2533589` |
| `results/ppo_fusion_v2_development_seed40042.summary.json` | `e0cfda3bb507c58fb53ea94ee834c64632674a38c43098f70b8fca40e4339193` |
| Bound V3 baseline CSV | `aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4` |
| Descriptive Fusion V1 comparator CSV | `1c54c4d36525bd84daf82c2773729aee3e4429c68db83d5ec67563b86605d632` |

The summary matches all 28 checked frozen protocol fields and all three
normalized paths: final V2 checkpoint, V3 configuration, bound baseline,
500 episodes, seeds 40042–40541, deterministic evaluator, unsafe TTC 2.0 s,
14 neighbors with five fusion features and scales 200/20/10/5/20, no intent,
and no safety shield. The local checkpoint, configuration, and baseline
fingerprints were independently recomputed and match section 65 exactly.
There are no code, configuration, or test changes in the result commit.

All three CSVs have 500 rows, no overlapping success/collision flags, and no
nonfinite numeric values. Every reported aggregate reproduces from its CSV
within `1e-12`. Collision-first classification therefore leaves these rows
unchanged. The summaries specify identical seed bounds and the evaluator
resets with `seed + episode_index`, then writes episodes in order. Since the
CSV schema has no seed column, the pairing below relies on that recorded
generation order; per-row seed identities cannot be independently certified
from the CSVs alone.

Ruff and the complete **114-test** suite passed during this result audit. This
post-run check verifies the current code; no separate pre-evaluation test log
is included in the two result artifacts. No evaluation, training, checkpoint
scan, or episode replay was run during this review.

### 66.2 Reproduced development metrics

All policies below use the same consumed seeds 40042–40541.

| Metric | V3 reference | Fusion V1 | Fusion V2 |
|---|---:|---:|---:|
| Success | 294/500 (58.8%) | 315/500 (63.0%) | 291/500 (58.2%) |
| Collision | 206/500 (41.2%) | 174/500 (34.8%) | 209/500 (41.8%) |
| Incomplete | 0/500 (0%) | 11/500 (2.2%) | 0/500 (0%) |
| Mean reward | 2.0158370123 | 2.8162111915 | 1.8878335538 |
| Mean length | 37.934 | 41.668 | 38.714 |
| Mean travel time (s) | 7.5868 | 8.3336 | 7.7428 |
| Mean minimum TTC (s) | 0.6003656548 | 0.6280867209 | 0.6174481851 |
| Mean unsafe-TTC events | 16.116 | 16.486 | 16.100 |
| Mean safety interventions | 0 | 0 | 0 |

Relative to V3, V2 changes success by **-0.6 percentage points** and collision
by **+0.6 points**. Relative to Fusion V1, it changes success by **-4.8 points**,
collision by **+7.0 points**, and incompletion by **-2.2 points**. Higher mean
minimum TTC than V3 does not establish lower collision risk.

### 66.3 Paired outcomes and exact tests

Rows are V3 outcomes; columns are Fusion V2 outcomes:

| V3 outcome | V2 success | V2 collision | V2 incomplete |
|---|---:|---:|---:|
| Success | 274 | 20 | 0 |
| Collision | 17 | 189 | 0 |
| Incomplete | 0 | 0 | 0 |

V2 gains 17 successes and loses 20. The exact two-sided McNemar test gives
`p = 0.7428293587290682`, with an unfavorable direction. The collision test
has the same discordant counts and p-value. These results do not demonstrate
a statistically significant difference from V3.

The additional Fusion V1 comparison is descriptive analysis of already
consumed development results, not an extra advancement test:

| Fusion V1 outcome | V2 success | V2 collision | V2 incomplete |
|---|---:|---:|---:|
| Success | 273 | 42 | 0 |
| Collision | 16 | 158 | 0 |
| Incomplete | 2 | 9 | 0 |

V2 loses 42 V1 successes and gains 18, giving exact two-sided success
`p = 0.002670436282807066` in the unfavorable direction. It removes 16 V1
collisions but adds 51, giving collision `p = 0.000021689238760717064`.
Of the 11 V1 incomplete episodes, **two become successes and nine become
collisions**. Eliminating incompletion consequently did not recover safe
completion in most of those cases.

### 66.4 Frozen gate decisions

| Gate from section 65 | Observed | Decision |
|---|---|---|
| Success at least 315/500 (63.0%) | 291/500 (58.2%) | Failed |
| Collision at most 174/500 (34.8%) | 209/500 (41.8%) | Failed |
| Incomplete at most 10/500 (2.0%) | 0/500 (0%) | Passed |
| Mean minimum TTC at least 0.6003656548142169 s | 0.6174481850738929 s | Passed |
| Favorable paired success versus V3, exact `p < 0.05` | 17 gains, 20 losses; `p = 0.7428293587` | Failed |

Only two of five gates pass. **Fusion V2 is rejected as an improvement.**
No untouched holdout advances from M14D. Fusion V1 remains rejected under its
original 2.0% incomplete ceiling; its promising development aggregate does
not make it an accepted replacement. V3 remains current best accepted policy.

### 66.5 Interpretation and next research direction

The V2 training package successfully removed incomplete terminal outcomes,
but failed to preserve V1's success and collision gains. Longer training,
four environment streams, and shorter per-environment rollouts changed
together, so this comparison cannot identify which training change caused
the regression. The CSVs contain terminal aggregates, not action traces;
they also cannot establish that V2 became more aggressive or explain the
timing of the added collisions.

The next work is to design and freeze a non-interventional action/timing
diagnostic using the existing V1 and V2 final policies on consumed cases,
including the incomplete-to-collision transitions. Its aim is to distinguish
unsafe entry timing from stalled but otherwise safe opportunities before
choosing another controlled training experiment. It must preserve policy
actions and reproduce the relevant committed episode outcomes. This result
section does not select new coefficients, feature scales, training duration,
checkpoints, or a new holdout, and it provides no new run command.

### 66.6 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-132 | Reject Fusion V2 and retain V3 as current best accepted policy | Advance because incompletion reached zero | Success, collision, and paired-success gates failed | Retained |
| D-133 | Preserve Fusion V1's original rejection | Relax its 2.0% incomplete ceiling after V2 failed | V1 still has 11/500 incomplete episodes | Retained |
| D-134 | Diagnose action timing before choosing another training experiment | Assume more training or greater caution will improve both outcomes | Nine of eleven V1 incomplete cases became collisions; terminal CSVs cannot explain mechanism | Retained |
| D-135 | Treat M14D pairing as supported by generation order | Claim the CSV independently certifies every seed | Seed bounds are in summaries; row-level seed IDs are absent | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-08 | Recorded and rejected M14D Fusion V2 development result | Apply all frozen gates and preserve unsuccessful evidence | Two-file scope; artifact/model/config/reference hashes; protocol fields; 1,500 existing CSV rows and aggregates; paired matrices and exact tests; Ruff and 114 tests passed | Result: `1f25f42`; documentation: this update |

**Next action:** freeze an action/timing diagnostic on consumed cases before another local run. Keep all models local and do not rerun M14D or evaluate an untouched holdout.
