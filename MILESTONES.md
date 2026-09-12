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


---

## 67. Milestone M15A implementation and frozen protocol — fusion action timing

**Experiment ID:** E-M15A-FUSION-TIMING-DEV-S40042

**Status:** Implemented and verified; full local diagnostic pending

**Date recorded:** 2026-09-09

### 67.1 Question and fixed case selection

M14D found that nine of Fusion V1's eleven incomplete episodes became V2
collisions, and 42 V1 successes also became collisions. This diagnostic
collects decision traces to examine target-speed changes, motion, and
conflict timing in those regressions. It does not alter either final policy.

The references remain the committed V1/V2 development CSVs on consumed seeds
40042–40541. Select every outcome-discordant row, plus the first ten rows in
seed order from each of the stable-success and stable-collision groups:

| V1 outcome | V2 outcome | Selected cases | Selection |
|---|---|---:|---|
| Success | Collision | 42 | All |
| Collision | Success | 16 | All |
| Incomplete | Success | 2 | All |
| Incomplete | Collision | 9 | All |
| Success | Success | 10 | First ten by seed |
| Collision | Collision | 10 | First ten by seed |
| **Total** | | **89** | **178 replays, one per policy per case** |

The implementation independently reproduced these counts before any full
diagnostic run. It processes the selected cases in ascending seed order and
records both the inferred source-row index and explicit replay seed. Original
CSVs still lack seed columns; seed identity comes from their documented row
order and summary bounds, as disclosed in section 66. These selected cases
are for diagnosis and cannot estimate population success/collision rates.
The twenty unchanged cases provide descriptive context rather than randomized
experimental controls.

### 67.2 Frozen inputs and implementation

New files:

- `scripts/diagnose_fusion_timing.py`: reference audit, deterministic case
  selection, non-interventional replay, comparison, and exclusive JSON output;
- `configs/fusion_timing_v1.json`: all frozen inputs and diagnostic settings;
- `tests/test_fusion_timing_diagnostic.py`: thirty new tests including
  parameterized cases.

Protocol SHA-256:

```text
374d4556e282e8753d42c63af37557bd3e3ae6b31d598286739c27c860eda0dc
```

The protocol binds the V3 configuration and both final Fusion checkpoints
to the fingerprints recorded in sections 63–66. It also binds each reference
CSV and summary JSON; all seven input fingerprints were checked against the
local files. Reference aggregates, column order, row count, integer counts,
collision-first outcomes, seed bounds, and fusion metadata must agree before
loading either policy. The V1 summary fingerprint is
`ef286f04a0a7bca4d40059c2bd7baadbe6f33ac541cc8c0a926579a2a783b7da`;
the V2 summary fingerprint is
`e0cfda3bb507c58fb53ea94ee834c64632674a38c43098f70b8fca40e4339193`.

The environment uses the existing V3 reward, 175-value fusion observation,
14 neighbors, and unchanged fusion scales 200/20/10/5/20. Evaluation keeps
deterministic prediction, the original automatic device selection, 5 Hz
policy frequency, and unsafe-TTC reporting at 2.0 s. No intent or shield is
enabled. The report records actual model devices, Python/package versions,
the diagnostic source hash, and the complete protocol.

### 67.3 Measurements and interpretation limits

Before every action, record the observation fingerprint, selected action,
ego speed and target speed, position, current and target lane indices, lane
longitudinal coordinate and remaining distance, radial TTC, and nearby CPA
geometry. After the unchanged action is passed to `env.step`, record actual
speed, target speed, route progress, and termination/truncation flags.

The explanatory CPA flag uses the already-established thresholds of time
at most 2.0 s and distance at most 3.0 m, with a 3.0 s horizon and 60 m range.
It checks every candidate pair and records the number that qualify, alongside
the closest-distance pair. These thresholds only annotate the trace. Geometry
comes from simulator traffic state and may include traffic beyond the policy's
visible slots; it is never inserted into the observation or action path.

Per-episode summaries count actions, flagged actions, target increases and
decreases, and low-speed samples. Low speed is defined as absolute speed at
most 0.5 m/s. The longest consecutive low-speed run and low-speed samples
without the CPA flag are also reported. Sample counts divided by 5 give
sampled durations in seconds. The terminal window uses the last two seconds
of each policy's own episode, which may end at different times.

`IDLE` continues the existing speed target and may produce acceleration or
braking. The controller derives `FASTER`/`SLOWER` targets from measured speed,
so command counts alone do not establish aggressiveness. Current/target lane
changes are geometric route indicators, not exact physical conflict-zone
crossings. Missing CPA predictions or unflagged low-speed samples do not
certify a safe opportunity to move.

The report gives the first differing action and first differing observation
within the common decision horizon. Initial observations and the complete
observation prefix through the first differing action must match. After that
action, traffic and ego trajectories may diverge; comparisons of later
actions are not counterfactual tests on the same scene. This diagnostic can
describe behavior but cannot prove which coupled V2 training change caused
its regression or which alternative action would have prevented a collision.

### 67.4 Reproduction and failure rules

Each replay must reproduce all eight original episode metrics. Boolean and
integer counts must match exactly; floating-point metrics use absolute
tolerance `1e-12` and zero relative tolerance. Metric accumulation preserves
the production evaluator's pre-step TTC and `info.get("min_ttc", pre_step_ttc)`
fallback. No post-step TTC substitution or policy action override is allowed.

Accept the diagnostic for analysis only if all 178 episode comparisons and
all 89 shared-observation-prefix checks pass. A passed diagnostic does not
advance either rejected policy or authorize an untouched holdout.

Reports include full traces stored as compact arrays with named columns.
Only descriptive trace floats are rounded to six decimals; original metrics,
their comparisons, observation hashes, and summary counts are unrounded.
Absent finite TTC/CPA predictions use JSON `null`; invalid physical state or
policy observations fail the run. Strict JSON serialization rejects NaNs.

Output creation is exclusive and refuses an existing report. Protocol-loading,
fingerprint, replay, prefix, and cleanup errors preserve a failure JSON when
the output location is writable. Metric failures include the failing seed,
policy, observed/expected metrics, and mismatched fields. Cleanup failures do
not mask an earlier error. Preserve and commit a failure report; do not delete
it and rerun or change seeds, checkpoints, or diagnostic thresholds.

### 67.5 Verification and implementation corrections

Ruff and the complete **144-test** suite passed. The thirty new tests cover
case selection and collision-first handling; exact count/tolerant float
comparison; NaN rejection; low-speed streaks; IDLE/target-speed semantics;
action/observation divergence ordering; snapshot non-interference with RNG,
route, vehicle position, and observations; deterministic action forwarding;
exclusive output; protocol-loading and metric failures; and cleanup errors.

Two short seed-7 checks, one for each existing final Fusion policy, reproduced
the uninstrumented production evaluation behavior: every episode metric,
action, and observation hash matched. Both policies loaded on CPU with
175-value observations, and their shared-observation-prefix check passed.
These checks wrote no artifacts and produced no new development-rate claims.
The full 178-episode diagnostic has not been run by Codex.

During implementation, a transcribed V1 summary hash and a lint line-length
issue were corrected before the run protocol was finalized. Initial tests
had 22 passes and three fixture setup errors because the sandbox could not
access pytest's temporary directory; the permitted rerun passed all 25 tests
then present, and the later full suite passed all 144 after five runner tests
were added. Review also prompted failure recording for protocol-loading and
cleanup errors, and rejection of nonfinite physical/observation values.

### 67.6 Frozen local command

From the project root with its virtual environment active, first run:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Only after both pass and the output does not already exist, run:

```powershell
python -m scripts.diagnose_fusion_timing --protocol configs/fusion_timing_v1.json --protocol-sha256 374d4556e282e8753d42c63af37557bd3e3ae6b31d598286739c27c860eda0dc --output results/fusion_timing_v1_development_seed40042.json
```

Commit only `results/fusion_timing_v1_development_seed40042.json`, whether it
contains a successful reproduction or a preserved failure. Keep all model
checkpoints local. No training, changed reward, altered feature scale,
checkpoint search, or untouched holdout follows automatically from this run.

### 67.7 Append-only decision and change-log additions

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-136 | Replay all changed outcomes plus twenty deterministic unchanged cases | Inspect only selected spectacular collisions | Covers every known V1/V2 outcome change with limited contextual replays | Retained |
| D-137 | Record target and actual speed alongside policy commands | Infer caution from IDLE counts | IDLE can accelerate toward an existing target | Retained |
| D-138 | Require original row reproduction and shared observation prefixes | Analyze traces without checking replay fidelity | Instrumentation must preserve behavior and common pre-divergence observations | Retained |
| D-139 | Keep the diagnostic descriptive with no policy advancement | Infer safe alternative actions from unflagged geometry | Outcome-selected cases and constant-velocity geometry cannot establish causal safety | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-09 | Implemented and froze M15A Fusion action-timing diagnostic | Explain observed V2 regressions before selecting another training experiment | Seven input hashes and 89-case selection checked; Ruff and 144 tests passed; both seed-7 metric/action/observation checks passed | This implementation update |

**Next action:** pass the test gate, run the single frozen local diagnostic command, and commit its one JSON output for analysis.


---

## 68. Milestone M15A result — controller targets distinguish waiting from continued motion

**Experiment ID:** E-M15A-FUSION-TIMING-DEV-S40042

**Status:** Diagnostic completed and accepted for descriptive analysis; policy decisions unchanged

**Date recorded:** 2026-09-09

### 68.1 Artifact and independent reproduction audit

Commit `e53ac66079d257814d4a183ad0d2503d340759fb` adds only
`results/fusion_timing_v1_development_seed40042.json` (2,084,328 bytes).
Its SHA-256 is:

```text
86e1e6b81e29f0a9761dbcf7637a7d094cbe59c2b1400d1474b530f4888de3f9
```

The local result commit initially had not been pushed: remote `main` was
still `32f7286`. The result and this analysis are to be pushed together.
The result commit changes no code, configuration, tests, or checkpoints.

The report contains the exact frozen protocol at SHA-256
`374d4556e282e8753d42c63af37557bd3e3ae6b31d598286739c27c860eda0dc`.
Its recorded diagnostic source hash is
`9c9a16f8bfb2c6767f7e70b69e556fd143956dbc8666b6780ce67203dd3c69e6`,
matching the committed script. All seven bound input files were rehashed and
matched: environment configuration, two final checkpoints, two CSVs, and two
summary JSONs. Both model devices were CPU.

Runtime recorded by the report: Python 3.12.9; NumPy 2.5.2; pandas 2.3.3;
PyTorch 2.13.0; Gymnasium 1.3.0; HighwayEnv 1.12.1; Stable-Baselines3 2.9.0.
These are the reported installed versions, not new dependency requirements.

Independent read-only analysis verified:

- all 89 selected cases, their order, explicit seeds, source row indices,
  strata, and frozen transition counts;
- all 178 observed and embedded expected metric records against the original
  CSV rows, with exact flags/counts and `1e-12` absolute float tolerance;
- all 8,213 trace rows, column counts, step/time ordering, action names,
  observation fingerprints, finite physical scalars, and terminal flags;
- every stored full-episode and terminal-window summary, reconstructed from
  the compact traces, with no discrepancies due to six-decimal rounding;
- all 89 first-action/observation-divergence comparisons and shared prefixes.

`reference_reproduced` is true, `verified_episode_count` is 178, and there are
no recorded run or cleanup failures. Ruff and all **144 tests** passed during
this result audit. No policy was replayed or trained during the analysis.
This post-run test check does not independently establish when the user's
pre-run tests were executed.

### 68.2 Incomplete cases are persistent zero-target stops

Every V1 incomplete case remained on incoming lane `o0 -> ir0` and ended
with zero recorded speed and a zero target speed. Across their last two
seconds, all **110/110 decisions** were `IDLE`, at low speed, with target zero
and no CPA flag. Their longest low-speed segment equaled their total low-speed
duration. Mean low-speed duration was **25.5273 s**, using the frozen
`abs(speed) <= 0.5 m/s` definition and 5 Hz sampling.

| Seed | V1 low-speed time (s) | V2 outcome |
|---:|---:|---|
| 40092 | 25.6 | Collision |
| 40094 | 26.4 | Collision |
| 40160 | 24.2 | Collision |
| 40168 | 26.2 | Collision |
| 40181 | 26.0 | Collision |
| 40302 | 25.8 | Collision |
| 40335 | 25.4 | Collision |
| 40382 | 25.4 | Success |
| 40495 | 25.0 | Collision |
| 40526 | 25.4 | Success |
| 40530 | 25.4 | Collision |

These V1 reference episodes contain 151 decisions and report 30.2 s under the
existing evaluator; those values are preserved exactly. Mean V2 low-speed
time on the same eleven seeds was only 0.0545 s, but nine outcomes became
collisions. The absence of a CPA flag during a V1 stop is not proof that any
particular departure time would have been safe.

### 68.3 IDLE often maintains motion in V2's added collisions

The 42 success-to-collision and nine incomplete-to-collision cases account
for all 51 collisions V2 adds relative to V1. Across V2's final two seconds
in these cases, **202 of 510 decisions** carry the frozen CPA flag. Of these
202 flagged decisions, **157 are IDLE with a positive speed target**:
109 at 9.0 m/s and 48 at 4.5 m/s. In the nine former incomplete cases alone,
42 of 51 flagged terminal-window decisions are IDLE with a positive target.

Thus the same action label appears both in long zero-target stops and in
continued motion near projected conflicts. This is a description of the
selected traces, not a claim that changing those IDLE decisions would have
prevented the collisions.

### 68.4 Command differences and effective target differences

Raw command divergence often precedes any change in the speed target. In
38 of the 42 success-to-collision cases, the first differing commands still
produce the same target speed. The median first command difference is 0.4 s,
whereas the median first target-speed difference is 3.0 s. At the first target
difference, all 42 cases still have identical pre-action observations.

The first differing targets are split in both directions:

| Outcome transition | Cases | V1 target -> V2 target at first target difference | Median time (s) |
|---|---:|---|---:|
| Success -> collision | 42 | 22: 9 -> 4.5; 20: 4.5 -> 9 | 3.0 |
| Incomplete -> collision | 9 | 4: 9 -> 4.5; 1: 4.5 -> 0; 4: 4.5 -> 9 | 1.2 |
| Collision -> success | 16 | 8: 9 -> 4.5; 8: 4.5 -> 9 | 1.8 |
| Incomplete -> success | 2 | 2: 4.5 -> 9 | 1.4 |

All 69 changed-outcome cases have a first target difference on a shared
pre-action observation. This does not isolate the effect of that single
decision: later policies and traffic can diverge. Both slowing and retaining
the higher target occur among regressions and improvements, so these results
do not support a blanket claim that V2 is too aggressive or that globally
greater caution would improve it.

The twenty unchanged cases remain descriptive context only. The first command
difference preserves the same target in 9/10 stable-collision and 10/10
stable-success cases. Seven stable-collision and eight stable-success cases
eventually have a target difference; five unchanged cases never do during
their common trace horizon. Case selection is conditioned on known outcomes,
so no new population rates or inferential significance claims are made here.

### 68.5 Observation audit and next testable hypothesis

Source inspection confirms the base observation selects
`presence, x, y, vx, vy, cos_h, sin_h`. The existing fusion wrapper adds only
neighbor range, closing speed, radial TTC, CPA time, and CPA distance.
Neither component includes the ego controller's current target speed.

The installed `MDPVehicle.act` maps FASTER/SLOWER from measured speed to a
discrete target, while IDLE retains the existing target. The low-level speed
controller uses the difference between target and measured speed. Therefore
the current target is a relevant internal controller state that is not
explicitly supplied to the feed-forward PPO policy. This source-level fact
and the observed zero-/positive-target IDLE behavior motivate the next
hypothesis; they do not prove that the missing input caused the failures.

**Proposed next direction:** add an explicit ego target-speed observation as
one isolated representation change, preserving all existing fusion values.
Test it with a prospectively specified training budget, seed protocol, and
development gates matched to its comparator. The purpose is to let PPO
distinguish its current commanded speed from its measured motion. Its effect
on success, collisions, or incompletion is unknown.

This proposal is not an automatic training authorization or an implemented
policy change. It requires an optional, backward-compatible observation
extension, non-interference tests, and a frozen comparison protocol before a
local training command is issued. No restart override, reward penalty,
additional checkpoint search, or new holdout is selected by this result.

### 68.6 Decisions and change log

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-140 | Accept M15A for descriptive analysis | Discard a completed diagnostic after policy failure | All 178 reference rows, 89 shared prefixes, and 8,213 trace rows verified | Retained |
| D-141 | Distinguish commanded speed from action labels | Treat IDLE as stationary or harmless | Zero-target V1 stalls and positive-target V2 conflict decisions both use IDLE | Retained |
| D-142 | Propose explicit target-speed observability as the next isolated experiment | Add a blanket caution or forced-restart rule | Relevant controller state is absent from the current observation; benefit remains untested | Proposed; implementation/protocol pending |
| D-143 | Retain all existing policy rejections and V3 acceptance | Promote V1/V2 on diagnostic evidence | No new policy outcome comparison was performed | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-09 | Recorded successful M15A reproduction and controller-target findings | Turn the V2 regression into a specific, testable observation hypothesis | Single-result scope/hash, seven bound inputs, full protocol/source match, all metric rows/traces/summaries/prefixes, local controller/observation source, Ruff and 144 tests checked | Result: `e53ac66`; documentation: this update |

**Next action:** implement and freeze one optional ego target-speed observation experiment before issuing another training command. V3 remains current best accepted policy; V1 and V2 remain rejected.


---

## 69. Milestone M15B implementation — collision precedence and observable target speed

**Experiment IDs:** E-M15B-CONTROLLER-STATE-IMPLEMENTATION, E-M15C-FUSION-CF-CONTROL-S42,
E-M15D-FUSION-CF-TARGET-S42, and E-M15E-FUSION-CF-PAIR-DEV-S40042

**Status:** Implementation verified; matched control training pending

**Date recorded:** 2026-09-09

### 69.1 Reward audit and isolated correction

The installed HighwayEnv reward implementation sums the weighted collision
and speed terms, then replaces that sum with the arrival reward whenever
`has_arrived` is true. A test invoking that upstream implementation confirmed
that simultaneous crash and arrival with the V3 coefficients returns a base
reward of +5.0. Collision-first evaluation classification from section 52
does not change that training reward. This edge case is a correctness concern;
its contribution to the observed collision rates has not been quantified.

`RouteProgressRewardWrapper` now accepts `collision_first=False`. The default
preserves all historical behavior. When explicitly enabled and both collision
and arrival are true, it reconstructs the ordinary collision-only weighted
base reward, excluding the arrival component and preserving the existing
on-road multiplier. Other transitions retain their original base reward.

For the new V3 configuration, let O be the original on-road indicator and
g(v) the original clipped speed reward in [0, 1]. On an overlapping terminal
event only:

```text
old base reward = O * 5
new base reward = O * (-10 + 0.05 * g(v))
shaped reward   = base reward + 2 * normalized_positive_progress - 0.005
```

With O=1 and g(v)=1, the base changes from +5.0 to -9.95. The progress/time
terms are unchanged. This narrowly repairs arrival precedence; it does not
redesign ordinary collision rewards, off-road gating, risk shaping, NPC
behavior, or terminal detection. Enabled runs expose the raw upstream base
reward and a correction flag in step info. Normalized rewards and multiple
controlled vehicles are rejected for this option rather than approximated.

The new configuration is `configs/intersection_reward_v3_collision_first.yaml`:

```text
SHA-256 4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f
```

Its parsed content differs from the existing V3 configuration only by
`reward_wrapper.collision_first: true`. The original V3 file and its SHA-256
`433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69`
are preserved. Historical checkpoints/results keep their original meaning.

### 69.2 Optional controller-state observation

`EgoTargetSpeedObservation` appends one scalar after the existing observation:

```text
augmented_observation = [original_observation, current_ego_target_speed / 9.0]
```

The feature uses the ego controller's target, not its measured speed, desired
NPC speed, hidden driver label, or next selected action. It updates on reset
and after each normal environment step. It does not change the target, route,
RNG, control action, reward, or termination. The 175 existing Fusion values
remain intact; enabling the option produces 176 values. At targets 0, 4.5,
and 9 m/s, the new feature is respectively 0, 0.5, and 1.

The environment factory, PPO trainer, and policy evaluator expose
`--target-speed-observation` and `--target-speed-scale 9.0` as appropriate
Python arguments/CLI flags. The option is off by default. Invalid scales and
nonfinite or out-of-range targets fail rather than being silently clipped.
Training/evaluation summaries record `target_speed_observation`,
`target_speed_scale`, and `collision_first_reward` alongside existing hashes.
The new input is used in both training and evaluation; an old 175-input
checkpoint cannot be substituted for the 176-input candidate.

No time-remaining, acceleration, route-stage, or additional traffic feature
is introduced in this experiment. Live watch/video entry points are not part
of this implementation; use the verified trainer/evaluator for this stage.

### 69.3 Why a new matched control is needed

Comparing a new target-aware policy trained under the repaired reward directly
to old Fusion V1 would combine a reward change and an observation change.
The prospective experiment therefore contains two fresh final checkpoints:

| Arm | Reward | Inputs | Local model stem |
|---|---|---:|---|
| Control | V3 with collision-first overlap correction | 175 | `ppo_fusion_cf_control_seed42` |
| Target-aware | Same corrected reward | 176 | `ppo_fusion_cf_target_seed42` |

The control isolates corrected-reward training relative to the historical
Fusion V1 setting. The candidate/control comparison then isolates the single
added observation. Both use the original Fusion V1 training package rather
than the coupled four-environment/500K V2 package. A shared training seed does
not imply identical training trajectories after the policies diverge, and
neither arm is an independent multi-seed replication.

### 69.4 Frozen matched training package and release order

| Setting | Both arms |
|---|---|
| Requested / expected collected steps | 200,000 / 200,704 |
| Training seed | 42 |
| Environments / stride | 1 / 1000 (only seed 42 used initially) |
| Learning rate | 0.0003 |
| `n_steps` / batch / rollout | 1024 / 64 / 1024 |
| Gamma / GAE / entropy | 0.99 / 0.95 / 0.01 |
| Policy network | [256, 256] |
| Internal evaluation | Offset 70000; 50 episodes every 10000 timesteps |
| Checkpoint interval | 25000 timesteps |
| Fusion neighbors / scales | 14 / 200, 20, 10, 5, 20 |
| Shield / intent | None / none |
| Eligible checkpoint | Final checkpoint only |

The control runs first. Its final ZIP and training JSON must pass an integrity
audit before the candidate command is released. That audit checks artifacts
and training metadata, not driving performance. Neither arm receives a
development evaluation until both final checkpoints are audited. Callback-best
and periodic checkpoints remain ineligible. Internal evaluation curves do not
select duration, coefficients, input scales, or a checkpoint.

First pass the test gate:

```powershell
python -m ruff check .
python -m pytest -p no:cacheprovider
```

Then run only this control command from the project root:

```powershell
python -m scripts.train_ppo --config configs/intersection_reward_v3_collision_first.yaml --config-sha256 4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f --timesteps 200000 --seed 42 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-envs 1 --env-seed-stride 1000 --eval-seed-offset 70000 --eval-episodes 50 --evaluation-freq 10000 --checkpoint-freq 25000 --risk-fusion --fusion-neighbors 14 --fusion-range-scale 200.0 --fusion-relative-speed-scale 20.0 --fusion-ttc-scale 10.0 --fusion-cpa-horizon 5.0 --fusion-cpa-distance-scale 20.0 --summary-output results/ppo_fusion_cf_control_seed42.training.json --output models/ppo_fusion_cf_control_seed42 --refuse-overwrite
```

Commit only `results/ppo_fusion_cf_control_seed42.training.json`. Keep its ZIP
and all callback checkpoints local. Stop and preserve outputs if a mismatch
or existing-file refusal occurs.

The candidate package is frozen now but not yet released: use the identical
command settings, add `--target-speed-observation --target-speed-scale 9.0`,
and replace only the output stem/summary with `ppo_fusion_cf_target_seed42`.
Expected candidate metadata is observation shape [176], target feature true,
scale 9.0, and collision-first reward true. The control must instead report
[175], feature false, scale null, and collision-first reward true.

### 69.5 Prospective development comparison and routing

After both audits, the pair will each be evaluated once on the already-consumed
seeds 40042–40541, using deterministic actions, collision-first outcomes,
unsafe-TTC reporting at 2.0 s, and the same corrected configuration. Evaluation
commands must include both audited model hashes and refuse existing outputs.
No evaluation command is released with this implementation.

Each arm's absolute quality screen requires all of the following:

1. success at least 63.0% (315/500);
2. collision at most 34.8% (174/500);
3. incomplete at most 2.0% (10/500);
4. mean minimum TTC at least V3's 0.6003656548142169 s;
5. favorable paired success versus the frozen V3 reference with exact
   two-sided McNemar `p < 0.05`.

The historical V3 reference remains bound to CSV SHA-256
`aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4`.
Its outcome/TTC comparison is descriptive development evidence under unchanged
dynamics; its mean reward is not comparable to corrected-reward runs.

To retain the added target feature, it must pass that absolute screen and also
improve over the freshly trained control by at least +2.0 success percentage
points and -2.0 collision points, have non-worsening mean minimum TTC versus
control, and show favorable paired success with exact two-sided `p < 0.05`.
This incremental test prevents attributing a reward-only improvement to the
new input. The 2.0% incomplete ceiling still applies.

Prospective routing: retain the target-aware direction for replication if it
passes every applicable gate; otherwise reject the feature candidate and
retain only the corrected-reward control direction if that control passes
its absolute screen. If neither qualifies, reject both. This is a planned
development selection rule, not evidence that either method already improves
driving. A retained direction requires separately frozen independent-training-
seed replication and eventual untouched evaluation before a final policy
claim. V3 remains the accepted policy throughout this screening stage.

### 69.6 Verification, engineering results, and scope limits

Ruff and the complete **168-test** suite passed, including 24 added cases.
Tests cover the installed arrival override; the corrected overlap reward;
unchanged non-overlap rewards and progress shaping; unsupported correction
settings; the single-option YAML difference; target scaling, reset/step
updates, invalid values, read-only observation/RNG/route behavior; and actual
environment-factory shapes with and without Fusion.

A seed-7 replay with the original Fusion V1 policy exercised original-reward,
corrected-reward, and corrected-reward-plus-target environments in parallel.
All 45 selected actions preserved observations (including the full 175-value
prefix), terminal flags, and RNG states. There was no overlapping terminal
event in that smoke; rewards matched, while synthetic tests specifically
covered the rare correction branch.

Both arms also passed end-to-end engineering smokes: 32 PPO timesteps with
16-step rollouts, batch 16, seed 7, one CPU thread, internal evaluation seed
offset 1, and one-episode evaluation on seed 7. Saved checkpoints contained
finite parameters, reloaded, and produced the correct 175/176 observation
and reward/target metadata. These settings differ deliberately from research
training and cannot support a performance claim.

| Engineering smoke | Internal evaluation, seed 8 | Evaluation, seed 7 |
|---|---|---|
| 175-input control | Incomplete, 151 steps, reward about -0.506 | Collision, 33 steps, reward -8.5081827487 |
| 176-input target | Collision, 30 steps, reward about -7.35 | Success, 44 steps, reward 8.93 |

All smoke outcomes, including failures, are preserved here for transparency;
none informed the frozen gate thresholds or training duration. Temporary
smoke checkpoints, logs, and output files are disposable engineering artifacts
and are removed after verification. The research checkpoints have not been
trained by Codex. Two initial line-length lint errors in the new metadata
entries were corrected before the passing suite; there were no test failures.

The earlier environment explanation also identified `yield_probability` as
stored profile metadata, not an input to the installed yielding controller.
Actual profile speed, acceleration, following-distance, and time-gap changes
remain active. This implementation does not change yielding or traffic
difficulty; any such change would require a separate environment experiment.

### 69.7 Decisions and change log

| ID | Decision | Alternatives considered | Evidence | Status |
|---|---|---|---|---|
| D-144 | Introduce collision precedence only through a new opt-in configuration | Silently alter historical V3 rewards | Upstream overlap returns +5; ordinary transitions and legacy configuration must remain reproducible | Retained |
| D-145 | Append only normalized ego target speed | Add time, acceleration, route, and traffic features together | M15A identified missing controller state; one input enables a controlled test | Retained |
| D-146 | Train a matched corrected-reward 175-input control before the 176-input candidate | Attribute a combined reward/input change to observability | Both arms need the same reward and training package | Retained |
| D-147 | Freeze final-only 200K, one-environment, seed-42 training for both arms | Reuse V2's coupled 500K/four-stream package or select checkpoints | Match the original Fusion V1 package while isolating the new feature | Retained |
| D-148 | Require absolute quality and incremental feature gates before replication | Advance on a single aggregate or reward change | Maintain the V1 completion/safety requirements and test feature value versus its matched control | Retained |
| D-149 | Leave yielding behavior unchanged and document the metadata limitation | Implement probabilistic yielding within this experiment | Traffic changes would confound reward/observation effects | Retained |

| Date | Change | Reason | Verification | Git commit |
|---|---|---|---|---|
| 2026-09-09 | Implemented isolated collision precedence and optional controller-state observation; froze matched training and development routing | Test the missing-target hypothesis without confounding it with reward correctness | Ruff and 168 tests; 45-step three-environment non-interference smoke; both 32-step train/save/load/evaluate smokes; old config hash preserved | This implementation update |

**Next action:** pass tests, run only the corrected-reward control training command, and commit its training JSON for the checkpoint audit. Do not run the target candidate or any development evaluation yet.

---

## 70. Reference-driven extension — static obstacle routing foundation

**Experiment ID:** E-M16A-STATIC-ROUTE-ENGINEERING-S7

**Date recorded:** 2026-09-09

**Status at implementation:** implemented; full tests passed; 20-case engineering gate pending.

### 70.1 Authority, reference, and benchmark separation

The user requested use of the selected reference and major changes where needed.
The selected workspace `REFERENCE.md` is the obstacle/pedestrian/rerouting reading
list, not another intersection reward-tuning proposal. README and the complete
previous research record were read before edits. The reference is preserved as
supplied and is included with this implementation.

This authorizes a separate extension, not rewriting failed intersection results.
The new `obstacle_route_v1` family implements only the first two static-obstacle
ladder stages. Pedestrians, uncertainty-aware prediction, traffic-aware lane
changes, and predictive shielding remain unimplemented. No simulator migration,
dependency change, old configuration modification, or PPO training is performed.
V3 remains the best accepted intersection policy. Section 69 remains unchanged;
an already-running corrected-reward control need not be interrupted. This stage
does not release its target candidate or development evaluation.

The selected sources were checked against primary abstracts/documentation:

- [Bastani, model predictive shielding](https://arxiv.org/abs/1905.10691):
  learned/backup policy separation is a future design rationale. This stage
  does not implement MPS or inherit its conditional safety proof.
- [Golchoubian et al., uncertainty-aware crowd navigation](https://arxiv.org/abs/2405.13969):
  pedestrian-state uncertainty motivates a later stage; the paper's reported
  improvement cannot forecast this project's rates.
- [HighwayEnv custom environments](https://highway-env.farama.org/make_your_own/):
  retain HighwayEnv road, vehicle, collision, and Gymnasium interfaces.

### 70.2 Implemented modules and protocol

New independent modules are `safeintent_rl/agents/route_graph.py`,
`safeintent_rl/envs/obstacle_route.py`, `scripts/evaluate_obstacle_route.py`,
and `tests/test_obstacle_route.py`. Existing intersection source is untouched.
The new factory is explicit; it does not register over or replace any old ID.

`configs/obstacle_route_v1.json` SHA-256:

```text
3da86256869eceaf8892169e2e3613b7858f53fd2d736b5f379dd659e2428857
```

| Property | Version 1 setting |
|---|---|
| Road | Two 4 m wide straight lanes, length 200 m |
| Ego start / goal | x=10 m in lane 0 / x>=160 m on road without collision |
| Initial speed / allowed targets | 4.5 m/s / 0, 4.5, 9 m/s |
| Blockage | 4 m by 4 m, lane 0; x uniform [70,90] m |
| Matched scenario variants | Lane 1 open; lane 1 also blocked at the same x |
| Other traffic / pedestrians / sensor noise | None / none / none |
| Dynamics / decisions / maximum duration | 15 Hz / 5 Hz / 30 s |
| Safe blocked stop | Speed <=0.1 m/s for five decisions (1 s), before blockage, on road, no crash, clearance >=2 m |
| Engineering seeds | 7–16 for each variant, 20 episodes total |
| Engineering controller | Deterministic B0 graph route + meta-actions |

The directed planning graph contains start, two lane alternatives, and goal.
Edges cost length/reference speed plus risk cost and lane-change cost in seconds.
Risk cost is zero in this static stage; lane change costs 1 second. Dijkstra
breaks equal-cost paths lexicographically. Impassable edges are **excluded**,
not assigned a large finite penalty that might still permit an impossible path.
This refines the reference's blocked-edge formula. No path returns None.
The graph is a local lane abstraction, not a city network or junction reroute.
It is computed at reset because these blockages never change during an episode.

B0 selects the open lane with LANE_RIGHT and requests FASTER afterward.
With no path it repeatedly requests SLOWER and ends in a separately classified
safe blocked stop. It has complete static lane availability from the simulator,
and assumes the adjacent lane is empty. It rejects extra road vehicles instead
of claiming to handle traffic. It is not a safety shield and logs zero overrides.

### 70.3 Observation, reward, and outcome definitions

The independent observation is 3x7 kinematics (ego plus up to two obstacles),
flattened to 21 values, then six bounded features: target speed/9, target lane,
route-available flag, clipped remaining x distance/160, and two lane-block flags.
Missing route uses lane value zero and availability zero. The resulting 27-value
input and five-action space are incompatible with old V3/Fusion checkpoints.
No hidden intent label is supplied. Native observation prefix and RNG behavior
are covered by non-interference tests.

This family has its own explicitly versioned reward, not V3 reward:

```text
collision:                -10
else off road:            -10
else successful arrival:   +5
else: 2 * positive_new_x_progress / 150 - 0.025 * 0.2
```

Progress uses a capped forward high-water mark, so backward/forward oscillation
cannot collect the same distance twice. Terminal collision/offroad/arrival
replaces shaping for that step. Safe blocked stop has no arrival bonus.
Collision takes precedence over arrival. Time limits use simulation-step counts
(150 decisions at the frozen duration) rather than accumulated float time.

JSON episode rows explicitly record seed, variant, sampled blockage position,
reward, length/time, success, collision, offroad, incomplete, safe blocked stop,
reroute success, obstacle collision, and minimum obstacle clearance. Clearance
is the signed distance between conservative circumcircles, sampled at 15 Hz;
negative clearance can occur without a polygon collision and is not labeled a
collision. This is not exact footprint separation or pedestrian clearance.
No TTC/CPA or pedestrian metrics are fabricated for this stage.

The evaluator reuses existing collision-first success/collision helpers but
uses a separate JSON schema to avoid changing historical CSV meanings. A safe
blocked stop is a terminal task fallback, **not route completion**; ordinary
timeouts remain incomplete. Report scenarios separately, never advertise
combined safe-stop-plus-arrival counts as PPO success percentage.

The evaluator verifies protocol SHA, freezes a parsed in-memory configuration,
records module hashes and package versions, and refuses output overwrite.
Exceptions preserve partial rows and a failure reason. A completed but failed
feasibility gate also retains its report and exits nonzero.

### 70.4 Tests and prospective engineering gate

Ruff passed and all **186 tests passed in 5.00 s** before the engineering run.
Eighteen added cases cover blocked/no-path routing, deterministic ties/cycles,
invalid costs, both scenario outcomes, obstacle visibility, matched layouts,
read-only augmentation, collision-first reward, a real straight-line collision,
post-terminal step refusal, forbidden reset overrides, Gymnasium API checks,
failed-report preservation, and overwrite refusal.

An initial lint check found one import-order issue and five overlong lines;
these were corrected before the full test gate. The initial 16 tests passed.
Gymnasium emitted two warnings about native HighwayEnv unbounded Box limits;
API checks still passed. These are not numerical training failures.

Before the 20-case run, require all open-lane episodes to reroute and arrive
without crash/offroad, and all blocked-lane episodes to stop safely without
crash/offroad. Every case must pass; this is an engineering feasibility gate,
not a learned-policy acceptance threshold or population safety estimate.

```powershell
python -m scripts.evaluate_obstacle_route --protocol configs/obstacle_route_v1.json --protocol-sha256 3da86256869eceaf8892169e2e3613b7858f53fd2d736b5f379dd659e2428857 --episodes 10 --seed 7 --output results/obstacle_route_v1_b0_engineering_seed7.json
```

### 70.5 Decisions and remaining stages

| ID | Decision | Reason |
|---|---|---|
| D-150 | Add a separate reference-driven environment family | Lane rerouting needs lateral control and a different task; do not corrupt V3 comparisons |
| D-151 | Validate deterministic routing before PPO | Establish a physically executable baseline before learning |
| D-152 | Exclude blocked edges and distinguish safe stop from success | Prevent impossible routes and inflated completion metrics |
| D-153 | Implement the static ladder first | Pedestrian dynamics, uncertainty, and shielding need their own measured validation |
| D-154 | Keep old experiment protocols intact | Broad change permission does not erase the append-only research record |

After this gate, the next reference stage is a crossing pedestrian with actual
time-evolving position, followed by conservative uncertainty features and
yield/resume tests. Before enabling a braking-feasibility shield, calibrate the
actual meta-action/low-level braking response: the installed MDPVehicle uses
measured-speed index transitions and proportional deceleration, not an assumed
constant road braking value. Predictive shielding and mixed traffic remain
separate later stages. No long PPO run or untouched holdout is released here.

| Date | Change | Verification |
|---|---|---|
| 2026-09-09 | Added reference-driven static route family, independent B0, observations and outcome ledger | Ruff; 186 tests; 20-case engineering result to be appended below |

### 70.6 Engineering result recorded 2026-09-10

The exact section 70.4 command completed after the 186-test gate. The retained
artifact is `results/obstacle_route_v1_b0_engineering_seed7.json`, SHA-256:

```text
1eb0620da774e231c7dacb029009eee821ba56d53993df70ba762ee7acb590d8
```

| Scenario | Cases | Route completions | Collisions | Safe blocked stops | Mean time |
|---|---:|---:|---:|---:|---:|
| Open adjacent lane | 10 | 10 | 0 | 0 | 17.4 s |
| Both lanes blocked | 10 | 0 | 0 | 10 | 3.0 s |

No case ended offroad or incomplete. Both variants contain ordered seeds 7–16
and the same sampled obstacle x for each seed. All twenty cases satisfy the
prospective engineering gate. The open-lane minimum circumcircle clearance
across cases is -1.521011 m despite no polygon collision, illustrating why this
conservative metric must not be interpreted as physical overlap. The blocked
cases stop immediately after discovering no path, with at least 54.363869 m
conservative clearance remaining. They do **not** validate last-moment braking,
a calibrated stopping-distance bound, or eventual resumption after blockage.

**Decision:** accept M16A as an engineering foundation for the next scenario
stage. Do not claim 100% learned-policy success, general traffic safety, or an
improvement over V3. The task is deliberately simple, uses privileged static
map occupancy, and has no other moving actors. No checkpoints were created.
All existing intersection experiments and section 69 commands are unchanged.

### 70.7 Publication integrity note

During repository staging, the default whitespace check flagged the raw JSON's
Windows CRLF line endings as trailing whitespace. The result bytes and recorded
SHA-256 were preserved rather than normalized after measurement. The same staged
diff passed with the command-local `core.whitespace=cr-at-eol` setting; no stored
Git setting, experiment parameter, source code, or result was changed by that
check. Publication includes no model files and no intersection trainer edits.

## 71. M15B corrected-reward control checkpoint audit (2026-09-10)

The user completed the section 69 control run and committed its training summary
in `4679cce`. This audit accepts the final artifact's integrity, not its driving
performance. The separate pedestrian implementation was deferred when the user
reported this completed run; no pedestrian code or results are claimed here.

### 71.1 Artifact and protocol checks

- Summary: `results/ppo_fusion_cf_control_seed42.training.json`, SHA-256
  `83924df16058fd830cd74085ac3a3a1ffb9ee1c282493dcf4d0eac44b907b1bd`.
- Local final ZIP: `models/ppo_fusion_cf_control_seed42.zip`, SHA-256
  `f35892f6830b1d9bc23301229ea1813c3e309a2ecd4e5293fe689cbc54c88b77`.
- Configuration SHA-256 matches the frozen section 69 value:
  `4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f`.
- ZIP integrity check passed; CPU reload succeeded; every policy state tensor
  is finite. Checkpoint and configured environment spaces match: 175 inputs,
  three discrete longitudinal actions. No driving episode was evaluated.
- Checkpoint reports seed 42, 200704 collected steps, one environment,
  n_steps 1024, batch 64, learning rate 0.0003, gamma 0.99, GAE 0.95,
  entropy coefficient 0.01 and network [256, 256], matching section 69.
- Summary reports 200000 requested steps, rollout 1024, initial seed [42],
  stride 1000, internal evaluation offset 70000 / 50 episodes / interval
  10000, checkpoint interval 25000, and the frozen 14-neighbor fusion scales.
  Collision-first reward is true; target-speed observation is false with null
  scale; shield and intent are absent. Callback settings are summary evidence,
  not an independent reconstruction of the entire training execution.

### 71.2 Verification and unsuccessful audit attempts

Ruff with `--no-cache` passed; all **186 tests passed in 5.10 s**, with the two
existing native unbounded-Box warnings. The initial restricted execution could
not write Ruff cache or pytest temporary files (166 passed, 20 setup errors).
Re-running with authorized temporary-file access resolved those access errors;
no code was changed to make tests pass. The initial audit output serializer
also rejected a NumPy int64 action count; converting the count to Python int
resolved reporting. Neither failure altered the checkpoint or configuration.

### 71.3 Decision and released next run

Accept the control artifact for the matched experiment. V3 remains the accepted
driving policy: no new success/collision rate is available. Do not run development
or holdout evaluation yet, and do not substitute a callback-best checkpoint.

Release the already-frozen target-aware candidate below. The only experimental
change is the target-speed observation (scale 9.0, expected 176 inputs); output
paths distinguish the two runs. Both candidate output paths were absent at
audit time. Re-run tests before starting if code changes after this audit.

```powershell
python -m ruff check --no-cache .
python -m pytest -p no:cacheprovider
python -m scripts.train_ppo --config configs/intersection_reward_v3_collision_first.yaml --config-sha256 4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f --timesteps 200000 --seed 42 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-envs 1 --env-seed-stride 1000 --eval-seed-offset 70000 --eval-episodes 50 --evaluation-freq 10000 --checkpoint-freq 25000 --risk-fusion --fusion-neighbors 14 --fusion-range-scale 200.0 --fusion-relative-speed-scale 20.0 --fusion-ttc-scale 10.0 --fusion-cpa-horizon 5.0 --fusion-cpa-distance-scale 20.0 --target-speed-observation --target-speed-scale 9.0 --summary-output results/ppo_fusion_cf_target_seed42.training.json --output models/ppo_fusion_cf_target_seed42 --refuse-overwrite
```

Run the training command only after both test commands pass. Long training is
left to the user's local workflow; it was not started by this audit. Commit the
candidate training JSON when complete, keep model ZIPs local, and audit the
candidate before releasing the paired development evaluation in section 69.5.
No historical experiment, reward coefficient, seed, or evaluation gate changed.

### 71.4 Target-aware candidate launched by request (2026-09-10)

The user explicitly authorized the assistant to run the released candidate.
Ruff (`--no-cache`) and all 186 tests passed again (5.71 s; two existing
unbounded-Box warnings). No source code or experiment setting needed changing.
An elevated process check found no existing candidate trainer, and final ZIP,
summary, and candidate log directory were absent before launch. A preliminary
restricted process query was denied; the authorized check resolved access.

Started the exact section 71.3 training arguments at local 13:57:50, initial
launcher PID 37092, using the project virtual environment in a hidden background
process. Python `-u` enables unbuffered logging without changing PPO settings.
Console output is retained at
`logs/ppo_fusion_cf_target_seed42/console.stdout.log` and stderr at
`logs/ppo_fusion_cf_target_seed42/console.stderr.log`. Initial output confirmed
CPU execution and TensorBoard logging to the candidate's `PPO_1` directory.

Status at this entry: **started, not yet complete or accepted**. Preserve all
outputs if execution fails; do not restart over existing artifacts. No final
candidate checkpoint audit, development evaluation, or performance claim has
been made. Model checkpoints and local logs remain uncommitted. This launch
does not implement the separate pedestrian scenario.

## 72. Candidate integrity audit and paired development release (2026-09-10)

The target-aware run finished at 200704 steps (5397 seconds reported by the
training timer, about 90 minutes). Its training summary is committed in
`92032da`. Final ZIP SHA-256:
`5ddbe06533e9dcce84ec2339db6226dadf6d5b2331fa0c87e0172e26f4f416fd`.
Training summary SHA-256:
`9b77f9b0cf94111ddeddbd1440fc0e35774784d1520d309af711c658659f9943`.

The previous-turn read-only audit verified configuration and checkpoint hashes,
ZIP integrity, CPU reload, finite policy parameters, 200704 collected steps,
and compatibility with the configured 176-input environment. Comparing training
summaries found only the intended differences: observation shape, target feature
flag/scale, and model path/hash. No performance evaluation was used in this audit.
Both final checkpoints are now eligible for the section 69.5 comparison.

Before evaluation, Ruff passed and all 186 tests passed in 5.02 seconds, with
the two existing unbounded-Box warnings. The following commands release the
already-frozen deterministic 500-episode development comparison, not a new
holdout. Each arm uses seeds 40042–40541 once, no shield/intent, the same corrected
reward and fusion settings, and the hash-bound historical V3 reference.
The section 69.5 absolute and incremental acceptance gates remain unchanged.
No source changes or model files are part of this update.

```powershell
python -u -m scripts.evaluate_policy --model models/ppo_fusion_cf_control_seed42.zip --model-sha256 f35892f6830b1d9bc23301229ea1813c3e309a2ecd4e5293fe689cbc54c88b77 --config configs/intersection_reward_v3_collision_first.yaml --config-sha256 4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f --episodes 500 --seed 40042 --unsafe-ttc 2.0 --risk-fusion --fusion-neighbors 14 --fusion-range-scale 200.0 --fusion-relative-speed-scale 20.0 --fusion-ttc-scale 10.0 --fusion-cpa-horizon 5.0 --fusion-cpa-distance-scale 20.0 --reference-csv results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --reference-csv-sha256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4 --output results/ppo_fusion_cf_control_development_seed40042.csv --refuse-overwrite
python -u -m scripts.evaluate_policy --model models/ppo_fusion_cf_target_seed42.zip --model-sha256 5ddbe06533e9dcce84ec2339db6226dadf6d5b2331fa0c87e0172e26f4f416fd --config configs/intersection_reward_v3_collision_first.yaml --config-sha256 4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f --episodes 500 --seed 40042 --unsafe-ttc 2.0 --risk-fusion --fusion-neighbors 14 --fusion-range-scale 200.0 --fusion-relative-speed-scale 20.0 --fusion-ttc-scale 10.0 --fusion-cpa-horizon 5.0 --fusion-cpa-distance-scale 20.0 --reference-csv results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --reference-csv-sha256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4 --target-speed-observation --target-speed-scale 9.0 --output results/ppo_fusion_cf_target_development_seed40042.csv --refuse-overwrite
```

Output names are arm-specific and refuse existing CSV/summary files. Background
execution may run the two independent arms concurrently, using separate console
logs; it does not change seeds, deterministic prediction, or episode order within
each arm. No result is claimed at release. If an arm fails, preserve its logs
and any artifacts before deciding whether a rerun is justified.

## 73. Paired development artifact incident and provisional results (2026-09-10)

Both evaluation console logs report 500 completed episodes and saved CSV paths;
neither stderr log contains an error. However, both expected CSVs and both
summary JSONs are absent from the actual repository. Searches including ignored
and hidden files under OneDrive Documents found no matching development result
files. The files are not tracked in the current Git checkout. The cause of
their disappearance is unknown; do not attribute it to the user or OneDrive
without evidence. The console logs are retained verbatim as
`results/ppo_fusion_cf_control_development_seed40042.console.txt` and
`results/ppo_fusion_cf_target_development_seed40042.console.txt`.

| Arm | Logged success | Logged collision | Derived incomplete | Logged mean minimum TTC |
|---|---:|---:|---:|---:|
| Corrected-reward control | 33.0% | 29.8% | 36.8% | 0.8474742909402212 s |
| Target-aware | 59.6% | 40.4% | 0.0% | 0.6152267645157733 s |

Incomplete is derived as 1 minus success minus collision, using the evaluator's
collision-exclusive success definition; it has not been recomputed from rows.
These are log-only provisional aggregates, not a verified paired comparison.
The target increases success by 26.6 percentage points versus control but also
increases collision by 10.6 points and reduces mean minimum TTC. Control fails
the frozen success and incomplete gates; target fails the absolute success and
collision gates and the incremental collision/TTC gates. Neither qualifies for
promotion on this evidence. V3 remains the accepted policy. No paired McNemar
statistic can be calculated from these aggregates alone.

Next: recover the original four artifacts if possible. Otherwise request explicit
approval for a documented same-checkpoint, same-seed recovery evaluation under
new output names, preserving these logs and checking reproduced aggregates.
Do not silently reconstruct episode rows or present recovered console text as
the original summary files. No rerun, new training, or protocol change was made
during this incident investigation; no tests were needed for these read-only
checks and documentation-only changes. Pedestrian work remains separate.

## 74. Artifact recovery confirmed and V3-PredictiveSafety v1 implementation

Date: 2026-09-10. User requests a separately named V3 successor with safety
variables/formulas informed by additional research, preserving accepted V3.

### 74.1 Resolution of section 73 artifact incident

All four original evaluation artifacts became available in commit `34cedb3`.
The subsequent read-only check verified 500 rows per CSV, collision-exclusive
success, and agreement of recomputed aggregate metrics with both JSONs and
retained console logs. Control has 165 successes, 149 collisions, 186 incomplete;
target-aware has 298 successes, 202 collisions, zero incomplete. CSV SHA-256:

- Control: `121c1c69c643c591740dd8d15c0f668579152bd4c2b006df118f6565a8cd6005`.
- Target: `e22790267eac961d1943d49e0152c7a34ba59fdebb9c1c374f4a14f33b394237`.

No recovery rerun was needed. Prior incident entries remain intact. Both arms
fail necessary frozen gates (section 73); reject both as accepted improvements.
No significance claim is needed to establish those gate failures; a full paired
statistical report was not produced by the recovery check. V3 remains accepted.

### 74.2 Research decision and scope

Added `V3_PREDICTIVE_SAFETY.md` with sourced research, equations, implementation
limits and follow-on options. Inspected primary 2025 risk-aware intersection and
reward-design papers, predictive shielding, PID Lagrangian methods, OmniSafe,
and the `menghan-xu/safe-rl-intersection` repository. No outside repository code
was installed/executed, and no reported external success rates were adopted.

Decision: implement action-conditioned forecasts as observations for PPO, not
another hardcoded brake shield or blanket TTC penalty. The 2025 work motivates
pre-collision geometric/dynamic risk; predictive shielding motivates considering
action consequences. Our observation-only method is an adaptation, not a
reproduction or a formal safety guarantee. Cost-critic/Lagrangian optimization
is documented as a later independent experiment, not misrepresented as done.

### 74.3 New configuration and formulas

Name: **V3-PredictiveSafety v1**. Config:
`configs/intersection_v3_predictive_safety_v1.yaml`, SHA-256
`e355448e7ab438840ab34b3c88d3883063dbbfba6f5327e16a2dda9adb4fca14`.
Original V3 config SHA remains
`433e6972cdf49668761bd5e55ad74b4910ed5a0128be44662d6c4577287fae69`.

All V3 dynamics, action targets, reward coefficients and episode duration remain
unchanged. New config adds only the documented collision-first overlap correction
and predictor block. It does not combine the rejected fusion/intent/shield options.

For each action [SLOWER, IDLE, FASTER], forecast an isolated ego for H=3 s at
15 Hz: apply the action once, maintain target thereafter, using actual installed
MDPVehicle proportional speed/route steering dynamics. Observed neighbors follow
`p_i(t)=p_i(0)+v_i(0)*t` with fixed heading. Use native sorted visible coverage,
up to 14 actors. No hidden NPC route, intent, behavior model or future spawn/RNG
is used. A separate road geometry copy and private fixed RNG isolate forecasts.

For rectangle separating axes u, define
`g_i(t,a)=max_u(abs((p_i-p_e) dot u)-r_e(u)-r_i(u))-m(t)`, where
`r_j(u)=(L_j*abs(long_j dot u)+W_j*abs(lat_j dot u))/2` and
`m(t)=0.5 m + (0.25 m/s)*t`. This is a signed separating-axis gap proxy with
a heuristic margin, not calibrated uncertainty or exact Euclidean clearance.

Append target speed / 9 m/s and three values per action: minimum gap / 10 m
(clipped -1..1), first nonpositive gap time / H (1 if none), and accumulated
predicted travel / (9 m/s * H) (clipped 0..1). Preserve the original 105-input
prefix, yielding 115 inputs. Predictions do not override actions or alter reward.
The info collision-cost field is diagnostic only, not a trained safety critic.
Training/evaluation summaries gain an additive predictor-config field (null for
legacy runs); old files are unchanged. Watch/evaluation load the same config.

No long model has been trained under this name. This is a new input architecture,
not a renamed or fine-tuned V3 ZIP. A fresh policy and a matched target-only
106-input corrected-V3 control must be frozen before long training. The prior
175-input fusion control is not an interchangeable ablation baseline.

### 74.4 Tests and prospective engineering smoke

Ruff and all **209 tests passed in 6.68 s** before the smoke below. The 23 new
tests cover oriented rectangle gaps, invalid parameters, input shape/bounds,
repeatability, live RNG/routes/positions preservation, exact original-observation
and reward agreement over a matched action sequence, braking versus accelerating
toward a stopped vehicle, empty traffic, forbidden option mixtures/reset overrides,
and invariance to changing hidden NPC routes/labels/behavior parameters.

Engineering-only smoke, frozen before execution: seed 7; initial-state forecast
timing over 20 calls; 32 PPO steps with rollout/batch 16, one update epoch and
[32,32] network on CPU; finite parameters; save/reload deterministic-action match.
Its temporary ZIP must be removed and no performance acceptance inferred.

```powershell
python -m scripts.check_predictive_safety --output results/v3_predictive_safety_v1_engineering_seed7.json
```

The checker verifies configuration hash, refuses existing evidence and preserves
a failed report on exceptions. No old development/holdout is consumed by it.
Engineering seed 7 is not an independent evaluation of safety or success.
Any failure and the measured overhead will be appended before a long-run release.

### 74.5 Engineering outcome

The frozen smoke passed. It trained 32 steps in 1.312037 seconds, with finite
parameters and identical deterministic action after save/reload. The observation
shape was [115]. All temporary smoke model files were removed by the checker's
temporary-directory context. No research model was saved or committed.

Twenty forecasts of the seed-7 initial state averaged 0.018471525 seconds per
observation. This is one-state engineering timing, not a throughput guarantee
for dense traffic, full training, or another machine. It adds computational work
relative to V3 and must be included in the next training-budget assessment.
The retained report is `results/v3_predictive_safety_v1_engineering_seed7.json`.
Its SHA-256 is `30f312f4eb93a83a774b4ef34c619f1ccef2657695cc028730a60b0c5dc910ca`.

Decision: accept implementation for experiment preparation only. Do not claim
fewer accidents, promotion over V3, a learned safety constraint, or pedestrian
capability. No long training, development evaluation or holdout was started.

The implementation was copied into the user's actual working repository after
checking unchanged source baselines and the append-only record prefix. There,
Ruff passed and all 209 tests passed in 6.85 seconds; the two existing native
unbounded-Box warnings remain. Unrelated maintenance-workspace static-obstacle
changes were deliberately not copied. No dependency versions changed.

## 75. Frozen V3-PredictiveSafety matched training package (2026-09-10)

User asks to proceed to the next step. No algorithm/configuration change is
needed. The scientific question is whether nine action-forecast values improve
the learned safety/progress tradeoff beyond knowing the current target speed.

### 75.1 Matched arms and compute budget

| Setting | Control | Candidate |
|---|---|---|
| Model stem | ppo_v3_predictive_control_seed42 | ppo_v3_predictive_safety_v1_seed42 |
| Inputs | 105 kinematics + target = 106 | Same prefix + nine forecasts = 115 |
| Reward | Collision-first corrected V3 | Identical |
| Shield / intent / fusion | None | None |
| Initialization | Fresh PPO | Fresh PPO |
| Requested / expected collected steps | 200000 / 200704 | 200000 / 200704 |

Both arms use training seed 42, one environment, stride 1000, learning rate
0.0003, rollout/n_steps 1024, batch 64, gamma 0.99, GAE 0.95, entropy 0.01,
[256,256] policy/value network specification, and default PPO update settings
from the same installed SB3 version. Internal deterministic evaluation uses
offset 70000, 50 episodes every 10000 timesteps; checkpoints every 25000.
Final checkpoint only is eligible, never callback-best or duration selection.
No continuing training from the accepted V3 ZIP. No coefficient/seed changes
based on internal curves. This is one training seed, not a replication study.

The candidate adds forecast computation (initial-state engineering measurement
18.47 ms per observation). Equal environment-step budgets are chosen for the
scientific comparison, not equal wall-clock budgets. Actual total training time
must be reported separately; the initial-state timing is not a runtime promise.

Ruff and 209 tests passed in the actual repository in 7.07 seconds, with two
existing Box warnings. A read-only engineering seed-7 check confirmed shapes
106/115 and exact equality of the candidate's first 106 values to the control.

### 75.2 Release order and exact commands

Run the control first, audit its final checkpoint and summary against this
package, then release the candidate. Do not start paired development evaluation
until both final checkpoints are audited. Preserve all failures and outputs.
The new control is necessary because the previous controls used 175/176 fusion
inputs and cannot isolate the action-forecast contribution.

Released control:

```powershell
python -u -m scripts.train_ppo --config configs/intersection_reward_v3_collision_first.yaml --config-sha256 4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f --timesteps 200000 --seed 42 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-envs 1 --env-seed-stride 1000 --eval-seed-offset 70000 --eval-episodes 50 --evaluation-freq 10000 --checkpoint-freq 25000 --target-speed-observation --target-speed-scale 9.0 --summary-output results/ppo_v3_predictive_control_seed42.training.json --output models/ppo_v3_predictive_control_seed42 --refuse-overwrite
```

Frozen candidate (do not run until control audit):

```powershell
python -u -m scripts.train_ppo --config configs/intersection_v3_predictive_safety_v1.yaml --config-sha256 e355448e7ab438840ab34b3c88d3883063dbbfba6f5327e16a2dda9adb4fca14 --timesteps 200000 --seed 42 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-envs 1 --env-seed-stride 1000 --eval-seed-offset 70000 --eval-episodes 50 --evaluation-freq 10000 --checkpoint-freq 25000 --summary-output results/ppo_v3_predictive_safety_v1_seed42.training.json --output models/ppo_v3_predictive_safety_v1_seed42 --refuse-overwrite
```

Tests must pass before each run. Do not pass the standalone target-speed flag
to the candidate: its config-driven predictor already includes that scalar.
Keep all model/checkpoint ZIPs local. Commit only training summaries and research
records after verification. Each new output stem must have no existing ZIP,
summary, or log directory; check for duplicate running processes before launch.

### 75.3 Prospective development gates

After both audits, evaluate each final policy once on the consumed development
seeds 40042–40541, 500 deterministic episodes, unsafe-TTC threshold 2.0 seconds,
matching configuration/observation settings and explicit model/config hashes.
Evaluation commands are not released here. No new holdout is used or reserved.

Keep the existing strict development quality screen for each arm: success at
least 63.0% (315/500), collisions at most 34.8% (174/500), incomplete at most 2.0%
(10/500), mean minimum TTC at least 0.6003656548142169 seconds, and favorable
paired success versus the frozen V3 reference with exact two-sided McNemar
p < 0.05. Reference remains
`results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv`, SHA-256
`aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4`.
V3 reference rewards are not comparable to corrected-reward means.

Additionally, retaining the predictor requires at least +2.0 success percentage
points and -2.0 collision points versus this fresh 106-input control, non-worsening
mean minimum TTC, and favorable exact paired success p < 0.05. Every applicable
gate must pass. If only the control passes the absolute screen, retain its
direction, not the predictor. If neither qualifies, reject both and preserve
their results. Do not relax gates after observing results. Any retained direction
still needs independently frozen training-seed replication and untouched testing
before replacing the accepted V3 policy.

### 75.4 Launch status

The package is frozen before launch. Only the control is authorized for the first
launch after checking artifacts/processes; the candidate remains held for audit.
No safety/performance improvement is claimed by release of these commands.

Control launch confirmed at local 19:15:07 on 2026-09-10, initial launcher PID
26952, after an authorized process check found no active PPO trainer and neither
arm had pre-existing ZIP/summary/log artifacts. The exact section 75.2 control
command runs in a hidden background process, with unbuffered stdout/stderr in
`logs/ppo_v3_predictive_control_seed42/console.stdout.log` and
`logs/ppo_v3_predictive_control_seed42/console.stderr.log`. Initial stdout confirms
CPU execution and TensorBoard output to `PPO_1`; no startup error was logged.
Status: running, not complete or accepted. Candidate training has not started.

### 75.5 Completed control audit and candidate release

The control completed 200704 steps; its final training timer reports 4466 seconds
(about 74 minutes). Final model SHA-256:
`172260972c4aad6ed73ea7080ae3458282cf046819c868820754c9d637c01473`.
Training summary SHA-256:
`8701584bc5a6175c30b0c1ce0acbca300715bdee2083272cae8413ad67a4fb08`.

Read-only audit passed: configuration hash matches section 75.2, ZIP integrity
passed, CPU reload matches the 106-input configured environment, all policy
state tensors are finite, and saved PPO steps/seed/environment count/network/
learning-rate/rollout/batch/gamma/GAE/entropy match the frozen package. Training
JSON confirms target-speed input enabled at scale 9, no fusion/intent/shield or
predictor, corrected collision precedence, and the frozen callback settings.
Callback metadata is recorded evidence, not an independent execution replay.

Ruff and all 209 tests passed in 6.93 seconds; the two existing native Box
warnings remain. No driving performance was evaluated. Accept the control's
artifact integrity, not its safety or success rate. The user's staged training
summary is preserved without restaging or committing it during this audit.

The previously frozen candidate command in section 75.2 is now released: train
`ppo_v3_predictive_safety_v1_seed42` with its hash-bound predictor configuration,
without adding the standalone target-speed flag. Candidate ZIP, summary and log
directory were absent at audit time. Candidate training has not started during
this audit. Recheck existing processes/artifacts before launch and rerun tests
if code changes. Audit the candidate before any paired development evaluation.

### 75.6 Predictive candidate launch

User explicitly approved starting the candidate. Ruff and all 209 tests passed
again in 7.09 seconds, with the two existing native Box warnings. Predictor config
hash matches the frozen section 75.2 value. The authorized process check found
no active PPO trainer, and candidate ZIP, summary and log directory were absent.

Started the exact candidate command from section 75.2 at local 20:47:20 on
2026-09-10, initial launcher PID 31580. It runs in a hidden background process
with unbuffered output in
`logs/ppo_v3_predictive_safety_v1_seed42/console.stdout.log` and errors in
`logs/ppo_v3_predictive_safety_v1_seed42/console.stderr.log`. Initial output
confirms CPU execution and TensorBoard logging to the candidate's `PPO_1`.
No startup error was logged. Status: started, not complete or accepted.

No configuration, seed, coefficient, training budget or evaluation gate changed.
The user's staged control summary remains untouched. No model files were
committed and no development evaluation was started. Audit the final candidate
checkpoint and training JSON before releasing the paired evaluation.

### 75.7 Completed predictive-candidate integrity audit (2026-09-11)

The candidate completed 200704 steps, with a final training timer of 11070 seconds
(about 3 hours 5 minutes), versus 4466 seconds for the matched control. This is
about 2.48 times the recorded training duration at the same environment-step
budget; it is not an isolated predictor microbenchmark or hardware-normalized
performance comparison. Console logs report both final ZIP and summary saved;
stderr contains no recorded error. The summary is tracked in the repository.

Final candidate model SHA-256:
`935dbd281c4f59326fdbd40a53038713629d7f74b7a48aa22566d88dce83b406`.
Training summary SHA-256:
`306b3931c0cca58b4c480b420ebb39ab8a5c36de733d1add09e30f66cc15f91e`.
Configuration matches the frozen SHA-256:
`e355448e7ab438840ab34b3c88d3883063dbbfba6f5327e16a2dda9adb4fca14`.

Read-only audit passed ZIP integrity, CPU reload with the configured 115-input
environment, finite policy tensors, 200704 steps, seed 42, one environment,
rollout 1024, batch 64, learning rate 0.0003, gamma 0.99, GAE 0.95, entropy 0.01,
and network [256,256]. Summary comparison against the audited control found
only the planned predictor config, config path/hash, observation shape,
standalone target-input flag/scale, and model path/hash differences.
The false standalone target-speed flag is correct: the predictor includes target
speed internally. Recorded predictor parameters exactly match the YAML block.

Ruff and all 209 tests passed in 7.11 seconds, with the two existing native Box
warnings. No driving evaluation was run during this audit and no success-rate
inference is made from training reward. Both final artifacts are now eligible
for the frozen section 75.3 paired development comparison. Prepare explicit
hash-bound evaluation commands and check output/duplicate-process guards before
launch. No new training, evaluation, reward change or gate relaxation occurred.

## 76. Released paired V3-PredictiveSafety development evaluation (2026-09-11)

User explicitly requested starting the evaluations. Both final checkpoints passed
their audits (75.5 and 75.7). Before launch, Ruff and all 209 tests passed in
6.93 seconds, with two existing Box warnings. Both model hashes, both config
hashes and the V3 reference CSV hash were rechecked against the frozen package.

Exact released commands:

```powershell
python -u -m scripts.evaluate_policy --model models/ppo_v3_predictive_control_seed42.zip --model-sha256 172260972c4aad6ed73ea7080ae3458282cf046819c868820754c9d637c01473 --config configs/intersection_reward_v3_collision_first.yaml --config-sha256 4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f --target-speed-observation --target-speed-scale 9.0 --episodes 500 --seed 40042 --unsafe-ttc 2.0 --reference-csv results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --reference-csv-sha256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4 --output results/ppo_v3_predictive_control_development_seed40042.csv --refuse-overwrite
python -u -m scripts.evaluate_policy --model models/ppo_v3_predictive_safety_v1_seed42.zip --model-sha256 935dbd281c4f59326fdbd40a53038713629d7f74b7a48aa22566d88dce83b406 --config configs/intersection_v3_predictive_safety_v1.yaml --config-sha256 e355448e7ab438840ab34b3c88d3883063dbbfba6f5327e16a2dda9adb4fca14 --episodes 500 --seed 40042 --unsafe-ttc 2.0 --reference-csv results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --reference-csv-sha256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4 --output results/ppo_v3_predictive_safety_v1_development_seed40042.csv --refuse-overwrite
```

Each independent arm runs once, deterministically, on the ordered 500 seeds
40042–40541. These are consumed development seeds, not a new holdout. No shield,
intent or legacy risk-fusion flag is enabled. The control has the standalone
target-speed scalar; the candidate includes it in its predictor configuration.
Section 75.3 gates remain frozen. Output CSVs, their summary JSONs and log
directories must be absent; reject duplicate active processes before starting.
Concurrent execution changes compute sharing, not episode ordering or protocol.

The evaluator prints its aggregate output only after completion; a blank stdout
log during execution is not sufficient evidence of failure. Keep separate logs
for both arms and preserve all artifacts on any failure. No rates or improvement
are claimed at launch. This entry records commands before execution.

Both launches succeeded at local 00:30:39 on 2026-09-11, initial launcher PIDs
20872 (control) and 39516 (candidate). Both processes were present on the startup
check and neither stderr log contained an error. Logs are in arm-specific
directories under `logs/`, named after each output CSV stem, as
`console.stdout.log` and `console.stderr.log`. Status: running; no result yet.
No model/checkpoint or existing result was modified by this launch.

### 76.1 Completed evaluation: summary-level audit (2026-09-11)

Both console logs report completion and saved episode CSVs. Both CSVs and both
summary JSONs are present; neither stderr log contains a recorded error. Parsed
console JSON equals each saved summary. Checkpoint/config/reference file hashes
match the frozen commands, and both summaries report 500 episodes on seeds
40042–40541, no shield/intent/fusion, and unsafe-TTC threshold 2.0 seconds.

| Arm | Success | Collision | Incomplete (derived) | Mean minimum TTC | Mean travel time |
|---|---:|---:|---:|---:|---:|
| Target-visible V3 control | 58.8% | 41.0% | 0.2% | 0.6065401801 s | 7.662 s |
| V3-PredictiveSafety v1 | 72.8% | 20.2% | 7.0% | 0.6728939420 s | 11.0012 s |

Rates above are from verified summaries; incomplete is computed as 1 minus
success minus collision under the evaluator's collision-exclusive success rule.
Episode rows were not independently recomputed in this status check, and no
paired significance statistic was calculated. Mean unsafe-TTC event counts are
16.204 (control) and 18.618 (candidate); longer episodes complicate comparison
of raw counts. Both report zero safety interventions.

The candidate improves success by 14.0 percentage points and reduces collision
by 20.8 points versus its matched control, with higher mean minimum TTC. However,
7.0% incomplete exceeds the frozen 2.0% ceiling. Reject the candidate as-is under
the all-gates rule; retain it as a promising diagnostic direction, not an accepted
replacement for V3. Control also fails its absolute success/collision gates.
No gate is relaxed, and no generalization or real-world safety claim is made.

Artifact SHA-256 values:

- Control CSV: `33ce4dc8b0e0a914033095a163cb26197ea31cf15788f9c0684e4e7818d711a3`.
- Control summary: `f4a3a623c75c6f643d14ac499d9dd5ce5f9b444c67ca25ba0388234d75929000`.
- Candidate CSV: `c5a50b1147fafa4aa61c55f9c8784455b4ce77b57ac9617dfd7466a48daae535`.
- Candidate summary: `f9521709fa74cd3a369809f9040d216384cc2ad0cb72f965b42d32acce1134eb`.

Next: independently verify episode-level metrics and paired outcomes, then
diagnose the candidate's incomplete episodes before changing rewards or training.
Do not assume incompletion proves unnecessary waiting without examining traces.
Preserve these outputs. No rerun, new training, code change or evaluation was
performed in this read-only status audit; only this append-only record changed.

## 77. Frozen five-situation stress study (2026-09-11)

User requests running all five proposed situations for both frozen policies and
explicitly reserves committing changes until completion. No commits or pushes
will be performed for this work. This is an exploratory distribution-shift study,
not a replacement for the original development benchmark or a new acceptance
holdout. No training, model selection, reward tuning or deadline change is allowed.

### 77.1 Prospective scenarios and controls

Protocol: `configs/v3_predictive_stress_v1.json`. For each arm and each scenario,
run 100 deterministic episodes on ordered seeds 80042–80141 (1000 episodes total).
Reuse the same seeds across scenarios and arms; matched seeds do not imply
identical future traffic after policy trajectories or scenario factors diverge.
These seeds are diagnostic only and cannot subsequently be treated as untouched.

| Scenario | Spawn probability | Cautious / normal / aggressive probabilities |
|---|---:|---|
| light | 0.2 | 0.30 / 0.45 / 0.25 |
| dense | 0.9 | 0.30 / 0.45 / 0.25 |
| aggressive | 0.6 | 0.10 / 0.10 / 0.80 |
| cautious | 0.6 | 0.80 / 0.10 / 0.10 |
| mixed | 0.6 | 0.30 / 0.45 / 0.25 |

Initial vehicle count stays 10: light/dense changes ongoing spawning only, not
initial traffic population. Other dynamics, reward, predictor parameters and
the native 30-second duration stay fixed. Majority profiles are probabilistic,
not exactly 80% in every episode; realized profile counts are retained per row.
The stored `safeintent_yield_probability` is not consulted by this project's
controller, so no new yielding model is claimed. Actual profile changes affect
desired speed, acceleration limits, following distance and time-gap settings.

Control remains final `ppo_v3_predictive_control_seed42.zip`, SHA-256
`172260972c4aad6ed73ea7080ae3458282cf046819c868820754c9d637c01473`,
with corrected V3 config hash `4478ae622b1a9c8d38b4163deb51589aec1acd9074aecdce23e3c8fa7546878f`
and target-speed scalar (106 inputs). Candidate remains final
`ppo_v3_predictive_safety_v1_seed42.zip`, SHA-256
`935dbd281c4f59326fdbd40a53038713629d7f74b7a48aa22566d88dce83b406`,
with predictor config hash `e355448e7ab438840ab34b3c88d3883063dbbfba6f5327e16a2dda9adb4fca14`
(115 inputs). No shield, intent inference or legacy fusion is added.

### 77.2 Isolated implementation and measurements

`scripts/evaluate_predictive_stress.py` checks protocol/config/model hashes,
loads each frozen checkpoint on CPU, and creates temporary effective scenario
YAMLs without modifying original files. Effective configurations and source /
package fingerprints are retained in the final report. Environment factory gains
only an optional driver-probability override; default callers remain unchanged.

Newline-delimited JSON episode records are flushed after each episode, preserving
completed evidence if execution fails. Every row explicitly records seed, exclusive
success/collision/incomplete, reward, steps, travel time, stopped time, zero-target
time, minimum TTC, unsafe-TTC event count, maximum NPC population, final route
progress and realized profile counts. Stopped time counts post-step ego speeds
below 0.5 m/s at the unchanged 5 Hz policy frequency; it is sampled stopped time,
not proof that waiting was unnecessary. Minimum TTC uses the historical evaluator
convention and threshold 2.0 s; nonfinite episode minima become JSON null and are
excluded from finite-TTC means. Profile labels are logged after action selection,
never provided to the learned policy.

Each arm refuses an existing output directory. Each completed scenario has a
summary and episode-file hash; the arm report retains failures and elapsed time.
Logs print progress every 10 episodes to make execution visible. Both arms may
run concurrently; scenarios within each arm run in the frozen order. No retries
or output replacement are automatic. A failed experiment remains in the record.

The initial 15 added tests and Ruff passed, covering frozen counts/seeds/deadline,
invalid profile probabilities, original-config preservation, default-override
rollout equality, exclusive NPC profile application, incomplete/null-TTC summary
semantics, failure-report preservation and overwrite refusal. Full-suite results
and launch confirmation will follow before any outcome claims.

### 77.3 Test gate and exact execution commands

Ruff and all 224 tests passed in the actual repository in 7.35 seconds (the two
existing native Box warnings remain). Protocol SHA-256:
`6c3851362df84a0aef76ee76dfad33bc203573aa843dde6c0d3c21dcd1142d47`.

```powershell
python -u -m scripts.evaluate_predictive_stress --arm control --protocol configs/v3_predictive_stress_v1.json --protocol-sha256 6c3851362df84a0aef76ee76dfad33bc203573aa843dde6c0d3c21dcd1142d47 --output-dir results/v3_predictive_stress_v1/control
python -u -m scripts.evaluate_predictive_stress --arm candidate --protocol configs/v3_predictive_stress_v1.json --protocol-sha256 6c3851362df84a0aef76ee76dfad33bc203573aa843dde6c0d3c21dcd1142d47 --output-dir results/v3_predictive_stress_v1/candidate
```

Both output directories must be absent before starting. Local console logs use
`logs/v3_predictive_stress_v1/{control,candidate}.{stdout,stderr}.log`.
Keep every scenario result regardless of performance; there is no stress-score
promotion rule. The original candidate's failed 2% incomplete gate is unchanged.

Both arm processes started successfully on 2026-09-11: control at local 01:26:40
(initial launcher PID 13184), candidate at 01:26:41 (PID 37572). Incremental
episode files and progress messages are being produced; no startup errors were
logged. No experiment source or protocol is changed while these jobs run.

### 77.4 Read-only completion verifier

Added `scripts/analyze_predictive_stress.py` separately from the running evaluator.
It requires complete reports for both arms; verifies episode-file fingerprints,
exact counts and seed ordering, exclusive outcomes, time/stopped ledgers, full
effective configs, frozen model/config/protocol hashes, source fingerprints and
agreement of recomputed versus saved summaries. It reports per-scenario paired
success rescues/regressions and exact two-sided McNemar p-values as exploratory,
not multiplicity-adjusted inference or a promotion rule. No episodes are rerun.

Four analysis-unit tests passed. Initial Ruff found two overlong lines in this
new verifier; they were corrected and Ruff passed. Neither this formatting fix
nor the analysis implementation changes the running experiment or its source
fingerprints. User retains responsibility for committing the completed changes.

### 77.5 Interrupted candidate process and explicit recovery plan

At local 01:42 on 2026-09-11 the control had completed all 500 episodes, but the
candidate process was absent (confirmed by authorized process enumeration).
Candidate stdout last reported dense 90/100; its flushed data contains all 100
light episodes and 94 dense episodes, last seed 80135. No candidate final report
exists and stderr is empty. Cause is unknown; do not describe this as a model
crash or a completed run. The initial candidate files remain untouched.

The user's continued request to finish all situations is handled with a separate
`scripts/recover_predictive_stress.py`, not by silently restarting the study.
It checks the frozen sources via the completed control report, model/config/
protocol fingerprints and saved prefix validity, copies original prefix bytes
to a new output directory, and replays the last saved partial-scenario episode
as an engineering reproduction check (not another scored episode). All metrics
must agree within 1e-9 floating tolerance, with exact discrete fields. Failure
stops recovery and preserves its report. On agreement, evaluate only the missing
306 candidate episodes: dense seeds 80136–80141 and all 100 seeds for aggressive,
cautious and mixed. The control and complete light scenario are not rerun.

Recovery results use `results/v3_predictive_stress_v1/candidate_recovery_01`.
The report records prefix hashes/counts, replay checks, recovery-code fingerprint
and recovery-only elapsed time. The final verifier accepts an explicit candidate
directory and retains recovery provenance. Original experimental functions and
their source hashes are unchanged. Initial recovery-code lint issues (one long
line and import formatting) were fixed before execution; five verifier/recovery
unit tests passed. Full-suite verification is required again before resuming.

Recovery test gate passed: Ruff and 229 tests in 13.16 seconds, with two existing
Box warnings. Launched the following command after confirming no active stress
process and no recovery output/log destination:

```powershell
python -u -m scripts.recover_predictive_stress --source-dir results/v3_predictive_stress_v1/candidate --output-dir results/v3_predictive_stress_v1/candidate_recovery_01 --control-report results/v3_predictive_stress_v1/control/report.json --protocol-sha256 6c3851362df84a0aef76ee76dfad33bc203573aa843dde6c0d3c21dcd1142d47
```

Console is retained in `logs/v3_predictive_stress_v1/recovery_01.log`. The
engineering replay of dense seed 80135 passed before continuing with seed 80136.
The completed light prefix was copied without rerunning. Recovery is running;
no completion claim is made at this checkpoint. No commits were performed.

### 77.6 Completed stress study and verified result (2026-09-11)

Recovery completed successfully: dense seeds 80136–80141 plus all aggressive,
cautious and mixed seeds, 306 new scored episodes. Together with the preserved
194-episode candidate prefix and the 500 completed control episodes, the study
contains exactly 1,000 scored episodes. The successful dense seed 80135 replay
is an engineering check only, not an additional scored episode. The original
interrupted candidate artifacts remain unchanged; the interruption cause remains
unknown. The authoritative candidate directory is `candidate_recovery_01` under
`results/v3_predictive_stress_v1`, not the original partial `candidate` directory.

Final artifact analysis returned `status: verified`, `episodes: 1000`. It checked
episode hashes, ordered seeds, exclusive outcomes, timing bounds, recomputed
summaries, effective configs, and frozen model/config/source provenance across
both arms. An additional byte-level check confirmed that the original light and
dense prefix hashes match the recovery record and their bytes are unchanged at
the start of the recovered files. No protocol or experimental source changes
were needed during recovery. The pre-recovery gate was Ruff plus 229 passing
tests (13.16 seconds; two existing Box warnings), as recorded in 77.5.

Each arm/scenario has 100 episodes, seeds 80042–80141. Entries below are
success / collision / incomplete percentages (also episode counts):

| Scenario | Corrected V3 control | V3-PredictiveSafety v1 |
|---|---:|---:|
| Light | 57 / 43 / 0 | 67 / 27 / 6 |
| Dense | 59 / 41 / 0 | 69 / 22 / 9 |
| Aggressive | 50 / 50 / 0 | 59 / 37 / 4 |
| Cautious | 70 / 30 / 0 | 78 / 20 / 2 |
| Mixed | 58 / 42 / 0 | 74 / 23 / 3 |

Candidate mean stopped times are 1.756, 2.476, 1.914, 0.908 and 1.236 seconds
in that order; control values are 0.004, 0.364, 0.160, 0.144 and 0.262 seconds.
The candidate improves observed success and collision rates in all five cases,
but unfinished episodes remain 2–9%. Dense traffic exposes the most incompletion
(9%); aggressive traffic remains the most collision-prone (37%). Stopping metrics
are consistent with conservatism but do not alone prove individual failure causes.

Exact paired-success McNemar p-values in scenario order are 0.00634765625,
0.04138946533203125, 0.14961278438568115, 0.09625244140625 and
0.0004024505615234375. These are exploratory and not multiplicity-adjusted or
promotion criteria. Equal-weight descriptive totals are control 294 successes,
206 collisions, zero incomplete; candidate 347 successes, 129 collisions and
24 incomplete. The reused 100 seeds across scenarios must not be treated as
500 independent seeds per arm. These tests cover traffic spawning and driver
profile mixtures only, not pedestrians, obstacles, weather or rerouting.

Decision: candidate remains rejected as-is under its earlier development gate;
this stress study does not relax the 2% incompletion ceiling or supersede the
previous evaluation. Accepted original V3 is unchanged. Diagnose dense waiting
and aggressive collisions before any separately named follow-up experiment.
No further training was launched.

Readable report: `STRESS_TEST_RESULTS.md`. Verified artifact:
`results/v3_predictive_stress_v1/verified_summary.json`, SHA-256
`61ebf55be7ad25deeb5af7776fecffe19e6647401db7b19669970c526330ba97`.
Analysis command:

```powershell
python -m scripts.analyze_predictive_stress --root results/v3_predictive_stress_v1 --candidate-dir results/v3_predictive_stress_v1/candidate_recovery_01 --output results/v3_predictive_stress_v1/verified_summary.json
```

That output now exists; independent reruns must use a new output path. During
the final read-only audit, an inline Python prefix check hit PowerShell quoting
syntax errors before execution; a native PowerShell byte/hash check then passed.
Repository HEAD was observed at `af3a083` during handoff, with prior stress files
already committed externally. This assistant performed no staging, commits or
pushes; the final documentation additions are left for the user to commit.

Final handoff check after documentation sync: Ruff passed; all 229 tests passed
in 9.39 seconds with the same two existing unbounded Box warnings.
`git diff --check` passed. Only `MILESTONES.md` and the new
`STRESS_TEST_RESULTS.md` remained uncommitted at that check.

## 78. Prediction audit and RL successor research (2026-09-11)

User defers CARLA, requests more research into improving the V3 predictor and a
more advanced RL-based successor. This entry records research and a read-only
configuration check, not an implemented successor or a new scored experiment.
Accepted V3, predictive v1, previous negative results and all gates are preserved.
The complete earlier research history remains unchanged. Current repository HEAD
at audit was `f715d4f`; the user had committed the section 77 completion report.

### 78.1 Verified observation issue

Ruff passed and all 229 tests passed in 10.49 seconds, with the two existing
unbounded-Box warnings, before a configuration-only reset on consumed seed 80042.
No policy decisions or scored episodes were run. The unchanged predictive YAML
produced shape 115, clip=true, see_behind=false and effective normalization ranges
x=[-200,200], y=[-4,4], vx/vy=[-80,80]. Units are metres and metres/second.
Its observation dictionary omits features_range. Installed AbstractEnv.configure
uses shallow replacement, so native intersection ranges are not inherited and
the straight-road kinematics defaults apply. Different y values above 4 m collapse
to +1 in that feature; other features need not be identical. The geometric
predictor partly compensates using raw positions but compresses them into nine
aggregate features. This is a verified information-loss mechanism, not evidence
that it alone caused a particular collision or timeout. No configuration changed.

Installed IntersectionEnv.step computes its base observation before actor
removal/spawning; the prediction wrapper then queries live actors after those
operations. The blocks can therefore use different actor sets. This static
ordering finding requires a snapshot-alignment test and episode traces before
causal claims. Native sorted selection uses lane-projected distance with rear
exclusion, not predicted collision risk. None of these facts invalidates the
already recorded outcomes under their frozen settings; they limit interpretation.

### 78.2 Existing failed-episode analysis

Authoritative candidate artifacts remain under
`results/v3_predictive_stress_v1/candidate_recovery_01/`.
Dense incomplete seeds: 80044, 80050, 80059, 80083, 80085, 80102, 80121, 80125, 80126.
All nine targeted zero speed for 25.6–27.4 s (mean 26.356 s); sampled stopped time
averages 23.756 s, with final route progress 33.2–48.0%. Persistent stopping is
established; unnecessary waiting is not. Of 37 aggressive collisions, 24 have no
sampled stopped time and 7 have at least 5 s stopped. Aggressive seed 80110 collides
with minimum recorded radial TTC 2.114 s and zero unsafe-TTC events. Preserve TTC
as the historical metric; it is not a rectangle collision probability. These
are analyses of saved rows, not replays, new training or additional scored data.

The predictor applies one action then holds its target for 3 s; it does not model
subsequent policy actions. NPC trajectories use fixed velocity and heading.
Forecast aggregation omits conflict-clearance intervals. Native speed-change
actions choose an index relative to actual speed, not merely previous target.
Current MLP has no memory/time input. Native deadline truncation is bootstrapped
by SB3, correctly for an external cutoff, while evaluation penalizes incompletion.
A finite-task-deadline formulation must be a prospective change, not a silent
termination patch or a claim of an SB3 defect.

### 78.3 Recommended research sequence (not released for training)

Full design, mathematical objectives, limitations and primary links are in
`V3_RL_RESEARCH_NEXT.md`:

1. V3-PredictiveGeometry v2: an isolated explicit normalization comparison,
   retaining PPO/reward/actions/traffic/predictor settings. Symmetric x/y ±200 m
   with unchanged velocity scales is a proposal, not an applied configuration.
   New feature semantics require a separately trained checkpoint.
2. Separately test common-snapshot observation/prediction alignment. Do not
   bundle it with geometry and then attribute results to either component alone.
3. Hash-bound failed-seed diagnostic replay, after tests, to distinguish incorrect
   forecasts from policy choices despite useful forecasts. No replay is launched.
4. More advanced V3-RecurrentConstrainedPPO: causal LSTM memory plus reward,
   collision-cost and incompletion-cost critics learned through RL. Compare
   memory-only and constraint-only ablations before interpreting their combination.

Proposed episode costs are Cc=I(collision) and
Ci=I(deadline without success/collision). Maximize expected V3 return subject to
E[Cc]<=dc and E[Ci]<=di. Budgets remain to be preregistered; old acceptance gates
are not changed. Separate Lagrange multipliers increase when the corresponding
completed-episode cost exceeds its budget. One candidate advantage is
(Ar-lambda_c*Ac-lambda_i*Ai)/(1+lambda_c+lambda_i). Event costs must fire once,
not be interpreted as collision probabilities after discounting. Remaining time
and task-deadline bootstrapping require explicit tests and matched controls.
Retaining the fixed V3 collision reward alongside an adaptive constraint is an
intentional additional objective. No formula guarantees feasible safe passage.

The policy and critics learn from rewards/costs and interaction, not expert
actions or hidden driver-class labels. Fixed geometry features are not RL.
MultiPath/Trajectron++ are useful prediction references, but their trajectory
losses are not RL and are not selected as mandatory components under the user's
RL-only learning preference. The recommended successor combines established
ideas; no scientific novelty or success percentage is claimed before evaluation.

Core primary sources inspected:

- [HighwayEnv observation contract](https://highway-env.farama.org/v1.9.1/observations/).
- [Gymnasium time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
- [Recurrent model-free RL](https://arxiv.org/abs/2110.05038).
- [SB3-Contrib recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html).
- [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html).
- [PID Lagrangian RL](https://proceedings.mlr.press/v119/stooke20a.html).
- [MultiPath](https://arxiv.org/abs/1910.05449) and
  [Trajectron++](https://arxiv.org/abs/2001.03093).
- [Previously selected uncertainty-aware DRL](https://arxiv.org/abs/2405.13969).

Only documentation is added. No production code, reward coefficients, configs,
seeds, deadlines or evaluation rules were changed; no installation or new
training occurred. This assistant did not stage, commit or push. Next actionable
implementation is observation-contract tests and a separately named repair,
not another blanket TTC shield or a larger model trained on lossy inputs.

## 79. V3-PredictiveGeometry v2 implementation and frozen run (2026-09-11)

User authorizes implementation, local execution and local Git commits, while
deferring CARLA. No push or model-checkpoint commit is authorized/needed.
This first experiment isolates the verified y-coordinate clipping issue from
section 78. Recurrent/cost/deadline changes and snapshot alignment are separate
future factors and are not bundled into this experiment.

New config: `configs/intersection_v3_predictive_geometry_v2.yaml`, SHA-256
`a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7`.
It is identical to predictor v1 except explicit observation.features_range:
x/y [-200,200] m and vx/vy [-80,80] m/s. The sole effective change is y from
[-4,4] to [-200,200]. Observation size remains 115; target speed and nine
forecasts remain identical for the same underlying state. No existing config
or production source is edited. A fresh PPO is required; no fine-tuning or
inference-time reinterpretation of an old checkpoint is allowed.

Six new geometry contract tests passed in 15.00 seconds and Ruff passed in the
maintenance checkout. They test native normalization, identical non-y inputs,
target/forecast values, rewards, executed-action info, environment RNG and
vehicle dynamics across complete fixed-action rollouts on seeds 7,42,80042.
Initial tests failed on the native reset's independently sampled, unexecuted
action placeholder. Only that reset-info field is now excluded; all executed
action fields remain exact checks. This is a test-fixture correction, not a
change to the experiment. Full actual-repository test gate follows before launch.

### 79.1 Frozen training and comparison protocol

Training uses fresh PPO MlpPolicy, seed 42, 200000 requested / 200704 expected
collected steps, one environment, learning rate .0003, n_steps 1024, batch 64,
network [256,256], gamma .99, GAE .95, entropy .01. Historical implicit settings
remain 10 epochs, clip .2, normalized advantages, value coefficient .5, gradient
clip .5, no target-KL. Internal validation offset 70000, 50 episodes every 10000
steps; checkpoints every 25000. Only the final root model is eligible, never
callback-best. No shield, intent, fusion or standalone target-speed flag.
The predictor remains 3 s / 15 Hz, margin .5 m, uncertainty growth .25 m/s,
14 neighbors, clearance scale 10 and speed scale 9. Duration 30 s, policy 5 Hz,
traffic, reward and native deadline behavior are unchanged.

Current runtime audit: Python 3.12.9, SB3 2.9.0, Torch 2.13 CPU, NumPy 2.5.2,
Gymnasium 1.3.0, HighwayEnv 1.12.1; Torch threads are 8 intra / 8 inter and CUDA
unavailable. Thread settings are not changed. Historical thread counts were
not recorded, so equality of past thread settings is not asserted.

After final-model integrity audit and a fresh full test gate, evaluate exactly
500 deterministic episodes on consumed development seeds 40042–40541 with the
same 2.0 s TTC metric. Preserve source hashes and verify raw reference rows.
References are the original V3, corrected 106-input control and predictive v1
development results already stored; do not overwrite or unnecessarily rerun them.
The historical CSVs lack seed columns: pair using their documented ordered-reset
protocol and metadata, not a claim of independently recorded row seed IDs.

Keep every section 75.3 gate unchanged. Additionally, a geometry repair is
retained as an improvement over predictive v1 only if success is at least 72.8%,
collision at most 20.2%, incomplete at most 2%, mean minimum TTC no worse than
v1's recorded development value, and paired success improves with exact two-sided
McNemar p < .05. These stricter retention conditions are fixed before training;
mixed or failed results are retained as evidence, not promoted. This remains a
single-training-seed development experiment, not final generalization evidence.

### 79.2 Exact training command and output guards

```powershell
python -u -m scripts.train_ppo --config configs/intersection_v3_predictive_geometry_v2.yaml --config-sha256 a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7 --timesteps 200000 --seed 42 --learning-rate 0.0003 --n-steps 1024 --batch-size 64 --n-envs 1 --env-seed-stride 1000 --eval-seed-offset 70000 --eval-episodes 50 --evaluation-freq 10000 --checkpoint-freq 25000 --summary-output results/ppo_v3_predictive_geometry_v2_seed42.training.json --output models/ppo_v3_predictive_geometry_v2_seed42 --refuse-overwrite
```

Final ZIP, training summary and entire model-stem log directory must be absent.
Check duplicate active trainers before launching. The initial process check
found only geometry unit tests, not a trainer; all three output paths were absent.
Keep models/checkpoints ignored. Record final tests and actual launch separately.

Actual-repository pre-training gate passed: Ruff and 235 tests in 27.27 seconds,
with the same two existing Box warnings. Training source SHA-256 is
`6dd20e7059bb0fd33f9516d3bb942003e8b2e0c386386252b9cb4e3c6d3dccc8`;
unchanged predictor source SHA-256 is
`2d3880359d26a6408cef8e5d7ae7dfc30ab2a9c4813a6888593935258fc86e2f`.
The final geometry model path is confirmed ignored by Git. No staged files
were present before the requested implementation commit.

### 79.3 Actual launch (2026-09-11)

Implementation commit: `fe7fec5` (geometry config, six contract tests, and the
append-only protocol above). Training started locally at
2026-09-11T02:35:17.3904745-04:00 using exactly section 79.2. The venv launcher
PID was 41228 and its Python worker PID was 20516; this is one training job,
not two independent trainers. Both final output files and the run log directory
were absent before launch. Standard output is
`logs/ppo_v3_predictive_geometry_v2_seed42.stdout.log`; standard error is the
matching `.stderr.log`. The first 1024 steps were confirmed, and the 06:40 UTC
check showed 4096 steps with empty stderr. Status: **training in progress**;
there is no completed geometry model, development result, or promotion claim.

### 79.4 Non-interventional failure diagnostic implementation

Added `scripts/diagnose_predictive_failures.py` and eight focused test cases
(nine test instances including parametrization). The diagnostic selects the
historical candidate's nine dense incomplete seeds and 37 aggressive collision
seeds, then replays both frozen candidate and control on those same cases:
92 engineering replays, not new scored evaluation episodes. Selection is fixed
in code and checked against the original complete, hash-verified 1000-episode
stress study. No reward, action, configuration, seed, or production source is
changed. This code is independent of the geometry training process.

The original deterministic PPO action is returned unchanged to the existing
episode evaluator. Per-decision traces record categorical action probabilities,
original input slots, the existing nine forecast features (candidate only),
ego state, and live actor geometry with episode-scoped first-sighting IDs.
Live post-spawn actors are explicitly not assumed to be the same actor set
as the native observation slots; row-alignment differences are logged. No extra
forecast rollout is performed. These traces do not supply labels to RL.

Every replay must match its frozen historical aggregate row (discrete fields
exact, numeric tolerance inherited from `rows_match`); mismatches preserve the
trace and a failed report, then stop. Outputs use a new directory and never
overwrite existing records. A positive predicted margin is not proof of a safe
counterfactual action, and this outcome-selected diagnostic cannot estimate
overall performance. Read-only review found no blocking defect; full test gate
and actual replay status are recorded separately. The replay experiment has
**not yet been launched**.

Review limitations: traces are pre-action only and cannot establish the final
collision partner/contact geometry. Current Torch version is recorded, but
historical Torch equality cannot be verified from the old stress metadata.
Added a further initialized-PPO (not trained) engineering purity test to check
the real policy-distribution path, aggregate episode reproduction, and Torch
and environment RNG preservation. This brings diagnostic coverage to ten test
instances; it is not a new policy-performance experiment.

Actual-repository verification passed: Ruff and **244 tests in 38.25 seconds**,
with the same two existing Box warnings. Coverage-count correction to the draft
description above: the diagnostic initially had seven test functions/eight
instances; the added real-PPO check makes eight functions/nine instances, not
ten. No test was removed. The entire new diagnostic is still unexecuted against
historical model checkpoints; unit/engineering tests are not its replay result.

### 79.5 Completion checks and quiet continuation

A task-attached follow-up named `Finish V3 geometry experiment` was created
(`finish-v3-geometry-experiment`) to check every 30 minutes. The user requested
no reports of minor progress; only completion, meaningful findings, failures,
or required input should be surfaced. This is a finite continuation of this
training/evaluation/diagnostic workflow, not authorization for additional
training experiments. Pause the follow-up once results are recorded and locally
committed, or if further work requires user direction. Local scheduled work
requires the computer and app to remain running; existing permissions still
apply and must not be expanded automatically.

Before evaluation require that the actual trainer has exited, the final saved
policy stdout marker exists, and both final ZIP and complete parseable training
JSON exist. The JSON is saved before environment close, so its existence alone
does not prove successful completion. Verify the final ZIP hash against the
summary; verify all section 79.1 settings in both metadata and loaded PPO,
including the historical implicit PPO settings. Use only the final root ZIP.
The geometry config and all frozen source hashes must be unchanged; a mismatch
blocks evaluation and is recorded. Re-run Ruff/full pytest before any experiment.

Additional frozen source SHA-256 values:

- `scripts/evaluate_policy.py`: `922b9f4a722b3c3a79a94e36753d4adb45c24a6d1faabb14c0c15604696aa55a`
- `safeintent_rl/envs/intersection.py`: `eca1191c1f676600ea68998642f8c417f654b14e810c948cdbd614842e03d52c`
- `safeintent_rl/evaluation/metrics.py`: `5a647b7a5616b5cc5f9a5743feb13dda386bab2e387a82d6c5af249c5342ed6b`
- `safeintent_rl/safety/ttc.py`: `75565975fa3beb8eabf6f6af75cf26eedbdad1805e7ae568630cb0575c0ae2c0`

The exact development command after replacing only the model-hash placeholder
with the audited final model hash is:

```powershell
python -u -m scripts.evaluate_policy --model models/ppo_v3_predictive_geometry_v2_seed42.zip --model-sha256 AUDITED_FINAL_MODEL_SHA256 --config configs/intersection_v3_predictive_geometry_v2.yaml --config-sha256 a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7 --episodes 500 --seed 40042 --unsafe-ttc 2.0 --reference-csv results/ppo_reward_v3_cpa_baseline_holdout_seed40042.csv --reference-csv-sha256 aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4 --output results/ppo_v3_predictive_geometry_v2_development_seed40042.csv --refuse-overwrite
```

Check active evaluator identity and absent CSV, summary, and console-log targets
before launching; do not create duplicates. The existing evaluator writes its
CSV only after all 500 episodes, then its summary. Preserve CSV-only partial
failure output; do not overwrite or automatically rerun it. Record actual
launch, frozen source/runtime evidence and failures in a subsequent entry.

`--reference-csv` only validates/records the reference hash; it does **not**
compute paired statistics. Independently recompute all metrics and exact
two-sided McNemar p-values for success versus each frozen reference. Validate
500 rows, strict exclusive success/collision, integer lengths 1–151, travel
time length/5, finite reward, nonnegative TTC (positive infinity allowed but no
NaN), unsafe count 0–length, and zero interventions. Incomplete means neither
success nor collision. Keep the existing mean-TTC convention of excluding
nonfinite episode TTC, and report the exclusion count. Verify metadata, paths,
hashes and seeds. Historical pairing remains ordered-protocol pairing only.

Reference CSV SHA-256 values in addition to original V3 in the command above:

- `results/ppo_v3_predictive_control_development_seed40042.csv`: `33ce4dc8b0e0a914033095a163cb26197ea31cf15788f9c0684e4e7818d711a3`
- `results/ppo_v3_predictive_safety_v1_development_seed40042.csv`: `c5a50b1147fafa4aa61c55f9c8784455b4ce77b57ac9617dfd7466a48daae535`

Their corresponding `.summary.json` hashes (original, control, v1) are
`51fe8e4aa86f62b2bbb5b66a7e8979655305e3a03ad571ca631d40828c7556a1`,
`f4a3a623c75c6f643d14ac499d9dd5ce5f9b444c67ca25ba0388234d75929000`,
and `f9521709fa74cd3a369809f9040d216384cc2ad0cb72f965b42d32acce1134eb`.
Exact TTC reference means are .6003656548142169, .6065401801078688, and
.6728939419963959 respectively. Counts are 294/206/0, 294/205/1, and
364/101/35 success/collision/incomplete.

Expressing the existing frozen thresholds as integer counts out of 500:
absolute gates require success >=315, collision <=174, incomplete <=10,
nonworse original TTC and favorable paired success vs original. The control
comparison additionally requires success >=304, collision <=195, nonworse
control TTC and favorable paired success vs control. The geometry-v1 comparison
requires success >=364, collision <=101, incomplete <=10, nonworse v1 TTC and
favorable paired success vs v1. Favorable means more rescued successes than
regressed successes and exact two-sided p < .05. For discordant counts b,c,
use min(1, 2*sum(comb(b+c,k), k=0..min(b,c))/2**(b+c)), or 1 if b+c=0.
All applicable gates must pass; no thresholds may be relaxed after results.

The pending diagnostic uses only the section 79.4 engineering selection, after
a fresh test gate, preserving an absent output directory:

```powershell
python -u -m scripts.diagnose_predictive_failures --protocol-sha256 6c3851362df84a0aef76ee76dfad33bc203573aa843dde6c0d3c21dcd1142d47 --output-dir results/v3_predictive_failure_diagnostic_v1
```

Sequence it after the geometry training/development evaluation to avoid adding
a second long simulator job during training. The original stress artifacts
remain authoritative; these 92 replays must not be pooled with scored results.
Commit only reviewed code, small results, and appended documentation locally,
with an unrelated-index check. No model/log/data checkpoint commits and no push.

### 79.6 Training complete and pre-evaluation audit (2026-09-11)

The 10:32 UTC follow-up confirmed training completed at 200704 collected steps.
Both original training processes had exited; stdout contains the final saved
summary and saved-policy markers, stderr is empty, and both artifacts were
written at approximately 10:28 UTC. The detached process exit code was not
retained; completion evidence is the final marker plus audited artifacts,
not an asserted captured exit code.

Final model SHA-256:
`f3bfeaa9826f989ab1934b71a46a92a97cef159c8a800c56b23b4530c17636d2`.
The ZIP matches the training JSON, the geometry config matches its frozen hash,
and all six source hashes in sections 79.2 and 79.5 are unchanged. Every training
summary field equals predictive v1's summary except the four expected
config/model path/hash fields. Loading the final PPO on CPU confirmed all
explicit and implicit section 79.1 parameters, 115 observations and three
actions. Runtime remains Python 3.12.9, SB3 2.9.0, Torch 2.13.0, NumPy 2.5.2,
Gymnasium 1.3.0, HighwayEnv 1.12.1, Torch threads 8/8.

The first pre-evaluation test attempt passed Ruff but had 221 passing tests and
23 setup errors in 59.80 seconds: Windows denied pytest access to its existing
`AppData/Local/Temp/pytest-of-hades` directory inside the restricted shell.
No assertion failed; this is not a passing full test gate. An unchanged-suite
rerun with reviewed normal local command access follows. No permission settings,
test code, experiment settings, or historical artifacts were changed.

The reviewed-access rerun passed Ruff and all **244 tests in 55.27 seconds**,
with only the same two existing Box warnings. All three frozen reference CSVs
and their summaries match the six section 79.5 hashes. Training summary SHA-256
is `6ab05310433ca6186874861f66bc2bc35e54a7474e7bbaa572b14d6220ebb861`.
The full pre-evaluation gate is now passed. The next action is exactly the
section 79.5 command with audited model SHA-256
`f3bfeaa9826f989ab1934b71a46a92a97cef159c8a800c56b23b4530c17636d2`;
record actual evaluation launch separately. No development outcome exists yet.

Evaluation launched at 2026-09-11T06:36:28.0385194-04:00 (10:36 UTC), launcher
PID 40044, with exactly the audited command above. No trainer/evaluator was
active and both result targets and both console-log targets were absent before
launch. Console paths are
`logs/ppo_v3_predictive_geometry_v2_development_seed40042.stdout.log` and its
matching `.stderr.log`. Status: **development evaluation running**. This is the
fixed 500-episode consumed development comparison, not untouched holdout or
evidence of an improvement yet. The queued diagnostic remains unstarted.

### 79.7 Independent development audit implementation (2026-09-11)

The evaluator has exited with a final saved-results marker and empty stderr;
CSV and summary were written at approximately 10:59:58 UTC. Added a standalone
standard-library audit, `scripts/audit_geometry_development.py`, with focused
tests. It reads without changing any original CSV, independently validates
typed rows and recomputes every summary metric, verifies all frozen references,
model/config/source fingerprints and evaluation metadata, and computes exact
paired success comparisons and every pre-registered gate. It explicitly keeps
the legacy finite-only mean-TTC convention and reports exclusions. No new
simulation or evaluation is performed by this audit. It refuses an existing
audit output and does not use the evaluator's aggregation implementation.

This is a post-result verification tool, not a revised acceptance protocol.
Its thresholds are copied unchanged from section 79.5. Tests include exact
McNemar direction/no-discordance, TTC-gate failure despite improved outcome
counts, malformed observations and nonfinite-TTC handling. The spreadsheet
skill's read-only scientific-data guidance is used to preserve raw observations
and separate verification from source records. No workbook is created.

Ruff and all **253 tests passed in 38.33 seconds**, with the same two existing
Box warnings. The independent audit completed successfully using bundled Python
and the standard library. All 2000 rows across the four development CSVs passed
validation and independently reconciled with their summaries. All model,
configuration, source and reference hashes matched; no TTC rows were excluded.

### 79.8 Verified geometry development result: mixed, not retained

Fixed development seeds 40042–40541, 500 episodes per policy, no safety shield:

| Policy | Success | Collision | Incomplete | Mean minimum TTC (s) |
| --- | ---: | ---: | ---: | ---: |
| Original V3 | 294 (58.8%) | 206 (41.2%) | 0 (0%) | 0.6003656548142169 |
| Corrected 106-input control | 294 (58.8%) | 205 (41.0%) | 1 (0.2%) | 0.6065401801078688 |
| Predictive V1 | 364 (72.8%) | 101 (20.2%) | 35 (7.0%) | 0.6728939419963959 |
| Geometry V2 | 394 (78.8%) | 98 (19.6%) | 8 (1.6%) | 0.6666153731858254 |

Geometry V2 mean reward 5.15308204815753, mean length 51.468, mean travel time
10.2936 s, mean unsafe-TTC events 19.208, and zero safety interventions. Compared
with V1, success increased 6.0 percentage points, collision decreased numerically
0.6 points, and incomplete decreased 5.4 points. Mean minimum TTC decreased
0.0062785688105705 s. No significance claim is made for the collision or TTC
differences. Mean unsafe-TTC events increased from 18.618 to 19.208.

Paired success checks (ordered reset-protocol pairing, not independent CSV seed
IDs): against original V3, 122 rescues / 22 regressions, exact two-sided
p=5.44807980630921e-18; against control, 121/21,
p=2.8177061253295323e-18; against V1, 67/37, p=0.004232546845609295.
All three success comparisons meet their frozen favorable-pair criterion.

**Decision: not retained under the frozen full gate set.** All absolute and
control-relative gates pass; all geometry-vs-V1 gates pass except nonworse mean
minimum TTC. Do not relax that threshold after seeing the results. Preserve the
checkpoint and results as a useful mixed ablation, but do not promote Geometry
V2 or replace accepted original V3. This is one training seed on a consumed
development set; improved development success alone is not generalization or
real-world safety evidence. A future investigation may address this trade-off,
but no further training or changed protocol is authorized by this follow-up.

Geometry CSV SHA-256:
`e145084f4ccce3d419c34886bd9434c0b5a817fb6fa3c1ceab616a89e1929030`.
Geometry summary SHA-256:
`227675bacf3f113209d970ec28763ccb57a86e352cfcd546f4ee8f6b1b5ecc61`.
Audit source SHA-256:
`80d98ac72d453b172a729f6eac2b5eef9f962172bdb7d1d3a66af75833c274d0`.
The separate `results/ppo_v3_predictive_geometry_v2_development_seed40042.audit.json`
retains recomputed metrics, counts, comparisons, gates and source fingerprints.
Original CSV/JSON evidence was neither edited nor regenerated.

### 79.9 Queued historical failure diagnostic preflight

The fresh 253-test gate above precedes the section 79.5 diagnostic command.
Diagnostic source SHA-256 is
`516c8c358ad66b4d0f093c6c7c30347bd8389ea0cc20c27cdf6815f1d9bbfee2`.
The frozen 92 replays compare historical predictive V1 and corrected control
on the selected V1 failures. They do not evaluate Geometry V2 and cannot be
used to revise its scored results. No simulator training/evaluation runs are
to overlap this diagnostic. Record launch and results separately.

The diagnostic launched at 2026-09-11T07:13:24.6091399-04:00 (11:13 UTC),
launcher PID 44204, using exactly the section 79.5 command. No simulator job
was active, and its output directory and console-log targets were absent.
Console paths are `logs/v3_predictive_failure_diagnostic_v1.stdout.log` and its
matching `.stderr.log`. Status: **historical diagnostic running**, not a
completed replay result. Geometry evaluation results above are final and are
not altered by this diagnostic.

### 79.10 Historical diagnostic complete and independently verified

The 11:44 UTC follow-up found the diagnostic complete with empty stderr and
the final `92/92 matched` marker. Runtime was 184.021145 seconds. Independently
verified every artifact hash and recorded source hash, all 92 distinct
arm/scenario/seed selections, each historical-versus-replayed aggregate row,
trace episode lengths and contiguous decision indices. All 92 replay rows
reproduce their historical counterparts. Report SHA-256:
`6709f894606fd49693967c17cd873e67489a82b1967a02cfdf94c66eb1e6adc1`.
Six diagnostic artifacts total approximately 3.29 MB, including four compressed
traces (largest approximately 1.29 MB), the episode ledger, and report. They are
small research results, not model checkpoints or a training dataset.

| Selected historical group | Replays | Decisions | Native/live row-mismatch decisions | Zero-target decisions | Stopped decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Predictive V1, dense incomplete | 9 | 1359 | 156 | 1177 | 1060 |
| Predictive V1, aggressive collisions | 37 | 1539 | 273 | 600 | 400 |
| Control, same dense seeds | 9 | 300 | 61 | 39 | 16 |
| Control, same aggressive seeds | 37 | 1238 | 218 | 87 | 48 |

Dense V1 failures had 17283 y-boundary slots among 17544 present NPC slots;
aggressive V1 failures had 12860 among 14024. These observations confirm severe
historical y-input saturation in these selected trajectories, not its frequency
across an unbiased population. Native/live mismatches are consistent with the
already documented observation-versus-post-spawn timing difference; the trace
does not assign actor identities to native slots or prove a unique cause for
each mismatch.

Among dense V1 failures, 384 stopped decisions had positive predicted FASTER
clearance margin, and FASTER was not selected in 380. Aggressive failures had
40 such decisions, with FASTER not selected in 34. All three predicted action
margins were nonpositive on 322 dense and 598 aggressive decisions. These are
pre-action samples, not the historical post-step stopped-time metric. A positive
forecast margin is not a counterfactual safety certificate: no alternative
action rollout or collision-partner identification was performed. Control has
no forecast inputs, so its forecast counters are not comparable evidence.

Interpretation: the recorded dense failures include persistent waiting even
when the limited forecast represents a positive-margin acceleration option;
the model can also encounter predicted conflicts under all three actions.
Future work should separately investigate observation/forecast consistency and
temporal decision state, while checking prediction error before changing reward
or safety formulas. This is a proposed direction, not a new approved protocol.
No causal attribution or performance improvement is claimed from these
outcome-selected replays, and no new training was started.

The finite geometry training, development evaluation, independent audit and
historical diagnostic workflow is complete. Preserve the section 79.8
not-retained decision and accepted original V3. Locally commit these verified
diagnostic artifacts and append-only record; then pause this follow-up pending
user direction. No checkpoints or logs are committed and no push is performed.

## 80. User-resumed next step: opt-in observation/forecast scene alignment

The user requested continuation after the completed section 79 workflow. The
next bounded implementation isolates the documented native-observation timing
issue before another long training run. Geometry V2's failed TTC retention gate
and original accepted V3 remain unchanged. No cost budgets, reward coefficients,
thresholds, seeds or evaluation rules are revised here.

Added `safeintent_rl/sensors/synchronized_predictive.py` containing an explicit
outer `SynchronizedPredictiveObservation` wrapper. Historical environment
factory, predictive wrapper, training/evaluation scripts, configs and model
checkpoints are untouched. The new wrapper is not enabled by default and cannot
silently change historical evaluations. It requires the native sorted relative
15-vehicle kinematics contract and a direct predictive-wrapper parent.

At each returned decision state, replace only the first 105 kinematic entries
with the native observer's current post-clear/post-spawn observation. Preserve
the target-speed scalar and all nine forecast values from the existing inner
wrapper byte-for-byte. No extra forecast, physics step, action, reward, vehicle
mutation, randomness, expert label or learned prediction model is introduced.
The new observation semantics require a separately trained policy; do not apply
the wrapper to an old checkpoint and call it an equivalent evaluation.

Engineering test protocol: compare complete fixed-action rollouts on seeds
7,42,80042 using repeated actions [2,1,0,1] (at most 160 decisions, ending at
native termination/truncation). Require identical executed-action info, rewards,
termination flags, vehicle poses/velocities/actions/routes/driver labels and
environment RNG. Require identical target/forecast entries and refreshed native
inputs equal to the current observer. A controlled post-step spawn fixture must
expose an actual changed native input while proving exactly one forecast call.
Reject stochastic/incompatible observation contracts. This is software
verification, not scored policy evaluation or a success-rate experiment.

The initial test run had four constructor failures and one passing rejection
test: the first contract guard incorrectly assumed `include_obstacles=False`.
Read-only inspection showed the frozen native configuration uses True. The
guard was corrected to require that existing value; no environment setting was
changed. Preserve this failed implementation check in the record. Verification
of the corrected implementation follows below.

Corrected focused tests passed (five instances). Actual-repository verification
passed Ruff and **258 tests in 85.19 seconds**, with the same two existing
unbounded-Box warnings. The frozen source/config/model/training-summary hashes
from the geometry audit are unchanged. All three complete seeded engineering
rollouts exercised a real input difference while preserving the checked
dynamics, rewards, terminations, target/forecast values and RNG. The forced
spawn check confirmed only one forecast call and idempotent read-only refresh.

Wrapper SHA-256:
`758d665d2c635d88693b1c71845b547d735580221830392c3f328724f68eb200`.
Test SHA-256:
`a0779818894ee417f0a63650c3f5afaed91dee03f634456a79e35ab0660fb48b`.

Explicit engineering use is
`SynchronizedPredictiveObservation(make_intersection_env(GEOMETRY_CONFIG))`.
This wrapper is deliberately not wired into any existing training/evaluation
CLI. The next training experiment needs an explicit versioned entry point and
metadata identifying synchronization, plus a pre-recorded matched training and
evaluation protocol. Do not imply that this implementation has trained a model
or improved success/TTC. No new long training run, scored evaluation or automatic
follow-up was launched in this step; the prior follow-up remains paused.

## 81. Synchronized PPO controlled training protocol (user requested next)

New model name: `ppo_v3_predictive_sync_v1_seed42`. This experiment isolates
the section 80 post-spawn input synchronization relative to Geometry V2.
Use the same geometry config and its frozen SHA-256, rewards, traffic, 115
inputs, three actions and predictive formulas. Original V3, Geometry V2 and
their source/config/model artifacts remain unchanged. No safety shield,
supervised labels, intent model or additional reward term is introduced.

Added `scripts/run_synchronized_predictive.py` as an explicit fixed-protocol
entry point. It invokes the unchanged historical training/evaluation mains with
process-local environment-factory adapters that are restored in finally.
Training's PPO constructor adds a saved observation-protocol stamp; evaluation
requires that stamp and completed, matching training evidence before loading
the new scene semantics. Both internal validation and development evaluation
use the synchronized wrapper. Old CLIs and files are not edited. The original
training/evaluation algorithms and aggregation implementations are reused.

All section 79.1 training parameters remain fixed: fresh PPO MlpPolicy,
200000 requested/200704 expected collected steps, seed 42, one environment,
LR .0003, n_steps 1024, batch 64, network [256,256], gamma .99, GAE .95,
entropy .01, 10 epochs, clip .2, value coefficient .5, gradient clip .5,
normalized advantages and no target KL. Internal validation uses seed offset
70000, 50 episodes every 10000 steps, checkpoints every 25000. Only the final
root model is eligible. Runtime versions and Torch threads 8/8 are pinned to
the prior experiment. No warm start or callback-best selection.

The runner checks frozen inputs and reference hashes before creating a run
record, records all package Python source fingerprints, requires absent output
artifacts/log directory, stamps the new summary, and preserves a failed run
record on execution failure. A completed run record and final marker, not the
underlying historical main's earlier save marker alone, establish completion.
Both completed and failed runs refuse overwrite/restart using the same name.

Exact commands (no tunable protocol overrides):

```powershell
python -u -m scripts.run_synchronized_predictive train --refuse-overwrite
python -u -m scripts.run_synchronized_predictive evaluate --refuse-overwrite
```

The second command is only eligible after the first finishes successfully,
artifact/model-parameter validation and a fresh full test gate. It uses exactly
500 deterministic development episodes, seeds 40042–40541, TTC threshold 2 s,
and the synchronized observation protocol. Preserve consumed-set limitations.

Retention: keep all section 79.5 absolute/control/V1 gates unchanged, including
mean minimum TTC >= .6728939419963959 and favorable exact paired success vs
original, corrected control and V1. Additionally require no regression against
Geometry V2 in success (>=394/500), collision (<=98/500), or incomplete
(<=8/500). Report paired success versus Geometry V2, but do not require a
significant success increase against it: this experiment targets the safety
trade-off while preserving its completion gains. These additional gates are
fixed before training, not based on the new results. No automatic promotion
from a single training seed, and no changing thresholds after outcomes.

Initial implementation checks found four lint issues (import ordering and
assigned lambdas) and one pytest temp-fixture access error; four focused tests
passed. The lint issues were corrected; no experimental outputs existed.
Full reviewed-access verification follows, including an eight-step tiny-network
engineering train/save/load test of the protocol stamp. That unit-test policy
is not a research result and is never used for scored evaluation.

### 81.1 Pre-training gate

Ruff passed and all **265 tests passed in 85.70 seconds**, with the same two
existing Box warnings. The real eight-step engineering test successfully
trained, saved and reloaded the protocol-stamped tiny policy. Adapter cleanup,
fixed arguments, failed-run preservation and overwrite rejection also passed.
All 42 recorded preflight input/source fingerprints were verified before launch.
Runner SHA-256 is
`9ab2dfbbb90b89f425cdca8e52f7c61f11e24f481efb2f5e0a8b87dc82fe4823`.
The full test gate is passed; actual launch is recorded separately below.

### 81.2 Actual synchronized training launch

Launched exactly the section 81 train command at
2026-09-11T11:47:00.1166432-04:00 (15:47 UTC), launcher PID 12808. No existing
training/evaluation job or new-model/run/summary/log targets were present.
The runner's manifest records start 15:47:03.964106 UTC, status running, and
`predictive_post_spawn_sync_v1`; stdout confirms CPU execution and the unique
TensorBoard log directory. Initial stderr is empty. Implementation/protocol
commits are `850d72b` and `326a625` (the code was committed concurrently before
the separate passing-test note). No completed checkpoint or result is claimed.

Console logs: `logs/ppo_v3_predictive_sync_v1_seed42.train.stdout.log` and the
matching `.stderr.log`. The mutable `results/ppo_v3_predictive_sync_v1_seed42.train.run.json`
is left uncommitted until its final audited status. Final model remains ignored.

The existing paused task follow-up was updated, not duplicated: name `Finish V3
synchronized comparison`, automation ID `finish-v3-geometry-experiment`, every
30 minutes, active for this finite section 81 comparison only. It must remain
quiet for routine progress, validate completion, run a fresh test gate before
the fixed evaluation, independently verify/report all gates including failures,
append results and commit only scoped small artifacts, then pause. It does not
authorize another training run, redoing the 92 historical replays, or CARLA.
Local follow-ups require the computer and app to remain running.

## 82. Matched-information sector observation prototype (2026-09-11)

The user proposed RGB/depth, LiDAR/semantic LiDAR, radar, GNSS, IMU and
collision/lane-invasion sensors, then approved continuing the staged plan.
The full sensor suite is deferred to CARLA. The current task implements only
a separately named **V3-SectorFeatures v1** engineering prototype; it does not
start or alter a long experiment. Section 81 synchronized training was still
running at the initial check (29,696 logged steps, empty stderr, run manifest
status running). That progress is not a completed checkpoint or result.

Read the README and chronological record before implementation; all prior
research entries remain intact. The main runner fingerprints every package
Python file, so even adding a never-imported file under `safeintent_rl` would
invalidate its frozen source set. The isolated prototype is therefore in
`scripts/sector_observation_v1.py`, with a new test file and the design document
`V3_SECTOR_OBSERVATION_V1.md`. No historical package source, runner, config,
model, reward, seed, action, observation contract or result is edited. The live
run manifest was already staged externally and is preserved untouched.

### 82.1 Design, formulas and limitations fixed before engineering tests

HighwayEnv's native 2-D LiDAR observes different actor coverage than the sorted
14-actor kinematics. This prototype instead computes an explicitly LiDAR-like
polar representation **only from the supplied synchronized 115-value input**.
There are no road queries, extra actors, independent sensor measurements,
semantic labels, noise, controller overrides or additional model forecasts.
It is feature engineering, not realistic sensor fusion or an accepted policy.

The 115 float32 values are retained exactly; 16 sectors with three features
each give a new 163-value observation. For present neighbor rows, decode
`p=200*[x,y]`, `v=80*[vx,vy]`; these are already relative to ego. Ego row zero
is used only for heading `theta=atan2(sin_h,cos_h)`, not subtracted again.
Set `r=||p||`, `c=-(p.v)/r`, `Delta=2*pi/16`, and
`k=floor(((atan2(p_y,p_x)-theta+Delta/2) mod 2*pi)/Delta)`.
Each half-open sector retains the nearest actor center, with equal-distance
ties selecting the first native row. Its features are
`[1, r/(200*sqrt(2)), clip(c/(80*sqrt(2)),-1,1)]`; positive c means approach.
Empty sectors emit `[0,1,0]`. A coincident center is assigned sector zero with
range and closing speed zero, retaining presence=1. The diagonal scales come
from the existing coordinate bounds, not a new sensor range or visibility rule.

This is not surface clearance, ray casting, occlusion handling or genuine free
space detection. An empty sector only lacks a retained actor. Normalization
clipping cannot be reversed; no rear visibility or extra information is added.
The original kinematics/target/forecasts remain unchanged. Current speed-only
actions do not enable lateral avoidance, pedestrians or rerouting. CARLA's
semantic LiDAR includes ground-truth object labels; collision/lane events are
outcomes rather than advance knowledge. See the design document for official
HighwayEnv and CARLA references and the complete normalization contract.

The protocol identifier is `predictive_post_spawn_sector_v1`. Existing policy
input sizes are incompatible: a future policy needs separate training and new
artifact names. Keeping the 256-wide actor/critic networks would still add
24,576 first-layer weights; any later claim must acknowledge this capacity
change, with a zero-padded capacity control an optional separately specified
future experiment, not silently bundled into the current one.

### 82.2 Implementation boundary

The opt-in wrapper directly wraps `SynchronizedPredictiveObservation`, validates
its contract and geometry ranges after resets as well as at construction,
and appends the deterministic transform without modifying physics, rewards,
termination, info, RNG or the original input prefix. It rejects malformed or
nonfinite inputs, invalid presence/ego heading and out-of-bound coordinates.
No trainer/CLI/default factory registration is provided in this engineering
stage. The existing finite heartbeat still finishes section 81 only.

First audit section 81's final training and fixed evaluation. Before a later
sector run, append exact commands, unique model/output names, source/config
hashes and retention rules, with fresh tests. The intended comparison keeps
the existing dynamics, reward, PPO settings, training seed/budget and consumed
development seeds unchanged. Do not infer permission to relax failed gates,
overwrite prior outputs, silently increase visibility or promise a success rate.
Engineering checks and any failures are recorded separately below.

### 82.3 Verified engineering checks; training remains gated

Ruff passed. All **33 new focused tests passed in 10.73 seconds** on the first
run; the complete repository suite then passed **298 tests in 92.33 seconds**,
with only the same two existing unbounded-Box warnings in the obstacle-route
interface test. No failed test or unsuccessful research run occurred in this
implementation. Tests cover sector orientation/wrap, approaching/receding and
tangential motion, nearest/tied actors, empty and coincident centers, extreme
valid coordinates, invalid contracts and read-only input preservation.

Three short paired engineering rollouts (seeds 7, 42 and 80042, at most 12
fixed actions each) preserve the complete 115-input prefix, physics, rewards,
termination/info and RNG while producing valid 163-input observations. These
are not a scored driving evaluation or evidence of improved safety. Direct
augmentation also succeeds with additional observation/forecast/step calls
forbidden. Independent read-only review found no blocking implementation issue;
its clipping/free-space and increased-capacity caveats are recorded above.

After syncing only the four owned source/test/document files, the running
section 81 runner's exact `fingerprint_inputs()` dictionary still matched all
**42** entries in its original training manifest. The append-only guard verified
that the previous MILESTONES text was preserved in full. Prototype source SHA:
`f0a78c5befe3c8e0edcc101ee49e7b2bb18c80be3afc6840e359653f99762919`;
test source SHA:
`308d30e1e2b24a340aae1a5030b65219cb94f7febe08f2bff67d4cd0f1d751f2`.

The user also asked to run the previously described 500-episode evaluation.
That remains the unchanged section 81 evaluation, eligible only after the
running synchronized training finishes, its complete checkpoint/run record is
verified and fresh tests pass. No partial checkpoint evaluation, additional
training, new seed set or change to the finite follow-up was launched here.
The live manifest's existing external staging remains untouched.

## 83. Synchronized training interruption observed (2026-09-11)

The user requested continued execution without stopping. On inspection at
19:07--19:09 UTC, neither original process (launcher 12808, worker 45488) was
present. Repeated normal process checks and reviewed-access CIM checks found
no Python training process. Stdout ended at **69,632 logged steps**, with last
write 17:24:53.9675938 UTC. Stderr remained empty. There is no `Synchronized
train complete` marker, final root model, training summary or development CSV.
The original run manifest still says `running`, but that status is stale and
must not be interpreted as a live process or successful completion.

Windows System event 1074 (User32) records a **power off** request initiated by
StartMenuExperienceHost.exe at 17:24:54.3451183 UTC (13:24:54 local EDT), less
than one second after the last stdout write. Its reason is Other (Unplanned),
code 0x0. This is the likely interruption cause; no process exit code was
captured, and no Python exception was logged. The OS boot-time property alone
did not establish a new boot, so no stronger claim about reboot timing is made.
All 42 previously recorded frozen input hashes still match.

This is an **interrupted, unsuccessful execution**, not a failed measured-policy
safety gate and not a new driving result. No completed 500-episode evaluation
is possible under the section 81 final-checkpoint protocol. Preserve every
original artifact, including the stale manifest and its existing external
staging. A separate small `results/ppo_v3_predictive_sync_v1_seed42.interruption.json`
records process absence, hashes, timestamps, event evidence and recovery limits.
The raw training logs and checkpoints remain local and ignored by Git.

### 83.1 Recovery boundary and checkpoint assessment

Only the 25,000- and 50,000-step periodic checkpoints exist. Their SHA-256
values are respectively
`039f7805fc2bab1f7378ab42740740201b2179af619a14cdeb817d987c1f4efb` and
`b07881fb9dd12e6454ab30aa9b23a662ea937cfdaec41ade27233d13ed76690a`.
The callback-best policy is not eligible as the frozen final model.

Independent read-only ZIP/metadata and source review found that the 50K ZIP
retains the policy, optimizer and protocol stamp, but not the environment,
partial rollout buffer, RNG streams or callback state needed for exact restart.
It was saved 848 steps into a 1,024-step rollout, with 480 PPO updates recorded.
Loading into a fresh environment resets/reseeds rather than restoring that
interaction state. A nominal 150K continuation would collect to 200,528 steps,
not the required 200,704, and would not recreate the interrupted trajectory.
The existing runner exposes no resume protocol and requires fresh training.

The scientifically preferred recovery is a **fresh seed-42 retry under a unique
new stem**, with identical scientific settings, new frozen command/source
provenance and explicit linkage to this interrupted attempt. Resuming 50K would
instead be a separately specified warm-start experiment. Neither is silently
launched here; the user must approve the recovery choice before further long
execution. Run fresh lint/tests before any authorized retry. No code, settings,
rewards, seeds, checkpoint selection or evaluation protocol was changed.

### 83.2 Follow-up state and verification

The existing follow-up configuration is still stored as ACTIVE. Tool discovery
in this turn did not expose `automation_update`, so it could not be paused via
the supported interface. No raw automation-file or permission-setting edit was
made. Pause this finite follow-up through the supported tool when available.
Until recovery direction is given, do not restart the old job, evaluate its
partial/checkpoint-best models, start the sensor experiment, or repeat unchanged
interruption notifications. Report a new meaningful state only.

This entry and its evidence JSON are documentation-only: the last code gate
remains 298 passing tests in section 82.3. No test suite was represented as a
new pre-training gate and no further training/evaluation was started. Evidence
JSON parsing, on-disk hash reconciliation and append-only preservation are
checked before a scoped local documentation/evidence commit; checkpoints,
logs, the externally staged live manifest and unrelated work are excluded.

## 84. Authorized fresh synchronized retry and continued research (2026-09-11)

Following the recorded power-off interruption and explicit recovery question,
the user approved a fresh retry and requested continued training, results
collection and local commits until the research succeeds. This supersedes the
section 83 wait-for-recovery-choice boundary. It does **not** authorize hiding
failures, overwriting attempts, weakening gates, selecting favorable seeds or
checkpoints, inventing performance, paid/external compute or pushing Git.

The immediate target remains all previously fixed section 79.5 and 81 gates:
at least 394/500 successes, at most 98/500 collisions and 8/500 incomplete,
mean minimum TTC at least .6728939419963959, and the already specified favorable
paired success comparisons against original V3, corrected control and V1.
Retain every other original gate and report paired success against Geometry V2
without adding a significance requirement against it. These are development
retention rules, not proof of generalization or real-world safety. One passing
training seed on a consumed set is not automatic promotion: any later
replication/fresh-evaluation stage needs a separately recorded fixed design.
Original V3 remains the accepted policy in the meantime.

### 84.1 Fresh retry implementation and exact protocol

The new entry point is `scripts/run_synchronized_retry.py`; it delegates to the
unchanged historical synchronized runner through a scoped, finally-restored
process-local adapter. Only its output-path globals and provenance callback are
replaced. No parallel threads should share that adapter. The model observation
protocol remains `predictive_post_spawn_sync_v1`, with 115 inputs and three
speed actions; it does not enable the section 82 sector prototype.

The retry asserts the complete original 42-entry source/input dictionary, pins
the original runner and preserved interrupted manifest, and includes that
manifest, the section 83 interruption evidence and the retry source in its own
fingerprints. Those fingerprints are rechecked at completion and matched before
evaluation by the original runner. The prior source mapping is copied rather
than mutated. Original model-parameter checks, protocol stamp, exclusive run
records, output overwrite guards and failure preservation remain in force.

Names and exact commands:

```powershell
python -u -m scripts.run_synchronized_retry train --refuse-overwrite
python -u -m scripts.run_synchronized_retry evaluate --refuse-overwrite
```

- Stem: `ppo_v3_predictive_sync_v1_retry01_seed42`.
- Final local model: `models/ppo_v3_predictive_sync_v1_retry01_seed42.zip`.
- Training summary: `results/ppo_v3_predictive_sync_v1_retry01_seed42.training.json`.
- Run records: `results/ppo_v3_predictive_sync_v1_retry01_seed42.train.run.json`
  and matching `.evaluate.run.json`.
- Evaluation: `results/ppo_v3_predictive_sync_v1_retry01_development_seed40042.csv`
  and matching `.summary.json`.
- Training log directory: `logs/ppo_v3_predictive_sync_v1_retry01_seed42/`.
- Console logs: `logs/ppo_v3_predictive_sync_v1_retry01_seed42.train.stdout.log`
  and matching stderr; evaluation uses separate matching `.evaluate.*.log` files.

All scientific settings are identical to section 81: geometry config SHA
`a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7`,
fresh PPO seed 42, 200,000 requested/200,704 expected collected steps,
LR .0003, n_steps 1024, batch 64, one environment, seed stride 1000,
validation offset 70000, 50 validation episodes every 10000 steps and
checkpoints every 25000. The 256/256 architecture, gamma .99, GAE .95,
entropy .01, all remaining model parameters, dependency versions, CPU and
Torch threads 8/8 stay fixed. No resume, callback-best selection or warm start.
This retry is execution recovery, not an extra independent training seed.

Only after successful process exit, final runner marker, complete run record,
matching final model/summary/source hashes and model checks may fresh Ruff/full
pytest be followed by the evaluation command. It remains 500 deterministic
episodes on consumed development seeds 40042--40541 with unsafe TTC threshold
2 seconds. Reconcile raw outcomes/metrics and paired tests independently using
the existing audit helpers, not the hard-coded Geometry V2 audit main. No old
69,632-step progress or 50K checkpoint becomes an eligible final result.

### 84.2 Continued-work limits and follow-up

Run one research training/evaluation at a time. Preserve and record all failures
and mixed results, including interruption records. After this retry's audited
results and scoped commits, use the evidence to choose the next isolated
improvement (the section 82 sector prototype is prepared but not released).
Append each new formula, config/seed choice, comparison, budget and numerical
acceptance rule before its run; verify fresh tests first. Do not silently make
the environment easier, add visibility, increase training budget or conflate
an exploratory consumed-set improvement with independent confirmation.

The user requests quiet continuation, not notices for every unchanged poll or
minor metric. Report meaningful results, completion, failure or required input.
Do not endlessly rerun identical poor experiments hoping for a favorable sample.
No CARLA expansion or large-checkpoint/log/data commits. Preserve unrelated
working-tree and index changes, especially the externally staged original run
manifest. Never push. Permission/resource limits still apply.

The root currently lacks the scheduling update tool, but the read-only review
agent verified that the supported tool is available in its context. After the
new process launches, that agent is authorized to retarget the **existing**
`finish-v3-geometry-experiment` heartbeat to this current retry and continued
research scope, keeping the same task and quiet 30-minute interval. Do not
duplicate the follow-up or edit its raw configuration. The actual update and
launch are recorded separately after tests; neither is claimed complete here.

### 84.3 Passed pre-training gate

Ruff passed; all **30 retry-specific tests passed in 8.98 seconds**, and the
full suite passed **328 tests in 53.06 seconds**, with only the two existing
unbounded-Box warnings. The first checks passed; no failed research run or test
failure occurred during this implementation. Deliberately simulated failure
fixtures verify failed-run preservation and are not actual training failures.

Tests confirm scientific argument identity except output paths, finally-based
adapter restoration, exactly-once delegation, immutable source mappings,
rejection of modified/added/removed historical inputs and changed lineage,
protocol/predecessor checks, new runner fingerprinting, and refusal of all seven
existing model/summary/log-directory/run-record/evaluation artifact targets.
Independent read-only review found no remaining blocking implementation issue.

Preflight verified **45** source/lineage fingerprints, Python 3.12.9, all frozen
dependency versions and Torch threads 8/8. Retry source SHA-256:
`bb9dfa47723325176b2727ef74fbbed0b7157dc343479e9712e7f7e003def577`.
Retry test SHA-256:
`a793f50bf9d40c4ef4a25b2eeb72ab2e0b27b1ed79e06de17d90ad52e2729f79`.
The test gate is complete. Actual launch must still confirm no conflicting
process or pre-existing retry console logs/artifacts and is recorded below.

### 84.4 Actual fresh training launch

Launched exactly the section 84 train command at
**2026-09-11T19:41:31.0498369Z**, launcher PID **47628**, worker PID **20120**.
Reviewed-access process inspection found no conflicting training/evaluation
process, and all eleven new model/summary/run/log/evaluation targets were
absent before launch. Start-Process used a hidden window with separate unique
stdout/stderr logs. The implementation and passing-test record were committed
first as `babf37d`.

The retry manifest records start **19:41:33.339861 UTC**, status `running`, the
unchanged `predictive_post_spawn_sync_v1` protocol and 45 input/lineage hashes.
Both observed command lines match the new retry command. Stdout confirms CPU
execution and the unique retry TensorBoard directory; initial stderr is empty.
This establishes actual launch, not a completed model or evaluation result.
Original interrupted artifacts remain unchanged; neither the staged original
manifest nor the mutable new run manifest is included in the documentation
commit. No checkpoint/log was added to Git and no push was performed.

### 84.5 Active continuation verified

The first 2,048 training steps were observed with no stderr output, confirming
that optimization started. The existing heartbeat was then successfully updated
through the supported automation tool at **2026-09-11T19:43:19.207Z**. The tool
returned success and a read-only configuration check verified exact saved-prompt
equality. Name: **Continue V3 safety research**; ID unchanged:
`finish-v3-geometry-experiment`; status ACTIVE; same task
`01a06baf-334e-7182-ab6e-6a4977dadc85`; same 30-minute interval. Creation time was
preserved; no duplicate task, raw-configuration edit or permission change occurred.

This supported retarget resolves section 83's unavailable-control/stale-prompt
limitation. The saved prompt now explicitly follows **retry01**, preserves the
interrupted predecessor, enforces final-model verification and fresh tests before
the fixed evaluation, independently audits all existing gates, commits only
reviewed small artifacts, and continues evidence-led, separately preregistered
V3 research under the latest user authorization. It must remain quiet for
routine progress and never confuse a development pass with replicated,
independently evaluated success. Pause only at genuinely verified completion
or when further meaningful progress requires user direction/new authority.
Local continuation requires the computer powered on and the app running.

## 85. Fresh synchronized retry training completed (2026-09-11)

Retry01 finished at **2026-09-11T23:10:14.028089Z** with **200,704 collected
steps**, as required by the 200K requested/1,024-rollout protocol. The final
`Synchronized train complete` marker and complete run record are present;
both original retry processes have exited and stderr is empty. This is a
successful training execution, not yet a measured driving improvement.

Verified final artifacts:

- Model `models/ppo_v3_predictive_sync_v1_retry01_seed42.zip`, 2,340,003 bytes,
  SHA-256 `8a363a7b5e7caef621d6b616890175c10d457531b4d4a213d05fc2011244f395`.
- Summary `results/ppo_v3_predictive_sync_v1_retry01_seed42.training.json`,
  SHA-256 `e858bc356e2f430e409ae96d53c62f164204376ed0799fa75f43fe3524d0edfa`.
- Completed run record `results/ppo_v3_predictive_sync_v1_retry01_seed42.train.run.json`,
  SHA-256 `de38b7ab02ceb7378ec07de76b3b2c95b8a25306a8ba437e42a06c9f58b97dba`.

All **45** source/input/lineage fingerprints match the completed record and
the unchanged original snapshot. Model/summary fingerprints, the
`predictive_post_spawn_sync_v1` protocol stamp, 115 inputs/3 actions, all frozen
PPO parameters, 1,960 optimization updates and finite policy weights passed
verification. Independent read-only review also passed ZIP CRC, exact argument
matching, 43 summary settings, predictor/reward settings and frozen runtime.
The selected final root model is distinct from the internal callback-best
archive, which has 140,000 steps; that earlier model is not selected or scored.

The interruption and its stale original manifest remain preserved. No partial
checkpoint was resumed and no sensor/sector observation was enabled. Fresh
Ruff/full pytest must pass before the unchanged 500-episode evaluation; its
test gate and actual launch are recorded separately below. Do not derive a
success/collision rate or policy acceptance from training reward alone.

### 85.1 Fresh pre-evaluation test gate

After completed-checkpoint verification, Ruff passed and all **328 tests passed
in 87.35 seconds**, with only the same two existing unbounded-Box warnings.
No failed checks occurred. This is the fresh gate for the section 84 retry
evaluation, not reuse of the earlier pre-training test result.

The eligible command remains exactly:

```powershell
python -u -m scripts.run_synchronized_retry evaluate --refuse-overwrite
```

It fixes the final model hash above, geometry config, synchronized observation,
500 deterministic episodes, seeds 40042--40541 and unsafe TTC threshold 2.0.
There is no shield, extra sensor input or changed reward. Actual launch still
requires a final active-process and absent-output/log check; do not duplicate
it or rerun after partial outputs. The successful training summary/completed
run record may now be committed, while the original interrupted manifest's
external staging and all local model/log artifacts remain untouched.

The first pre-commit `git diff --check` returned exit 2 because the Windows
runner's CRLF line endings on changed JSON lines were treated as trailing
whitespace. This is not a failed PPO/test result. Preserve the verified artifact
bytes and SHA-256 values; use a one-command `core.whitespace` override retaining
the normal whitespace checks and adding `cr-at-eol`. No artifact normalization,
persistent Git configuration or permission setting is changed.

### 85.2 Fixed development evaluation launched

The CRLF-aware working/index checks passed without rewriting result bytes.
Verified training records and the fresh test gate were committed as `95a07fb`;
the final model remains ignored. After confirming no active research process
and no existing evaluation CSV, summary, run record or console log targets,
launched the exact section 85.1 command once at
**2026-09-11T23:23:28.4707628Z**, launcher PID **1792**, worker PID **14736**.

The new evaluation record starts at **23:23:31.986390 UTC**, status `running`,
mode `evaluate`, with the correct unchanged observation protocol and 45 source
fingerprints. Its arguments explicitly bind final model SHA `8a363a7b...44f395`,
geometry config SHA `a9629f60...e7`, 500 episodes, first seed 40042, unsafe TTC
threshold 2.0 and the frozen original-V3 reference hash. The last episode seed
is 40541 under the preserved sequential reset protocol. Both actual process
command lines match the retry evaluation; initial stderr is empty.

Console logs are `logs/ppo_v3_predictive_sync_v1_retry01_seed42.evaluate.stdout.log`
and the matching stderr. The evaluator buffers rows and prints/writes results
at completion, so absent interim output alone is not a stall. Do not restart
the command or change source while it runs. Completion still requires actual
process exit, final `Synchronized evaluate complete` marker, complete manifest,
artifact/source hashes and independent 500-row metric/paired-gate audit. No
evaluation success rate, gate pass or policy promotion is claimed yet.

The active continuation follows this evaluation and subsequent audited research
under sections 84--85. The interrupted original experiment is never a fallback
scoring target. Preserve concurrent user/app commits and all live run manifests;
commit their completed audited versions later, never models or logs.

## 86. Synchronized retry development result: not retained (2026-09-11)

The fixed retry01 evaluation completed at **2026-09-11T23:41:13.875670Z**.
The final `Synchronized evaluate complete` marker and completed run record are
present, stderr is empty, and reviewed-access process inspection found no
remaining research process. The 500-episode evaluation was run once, using the
section 85 final model and unchanged development seeds **40042--40541**.

Independent raw-row reconciliation gives:

| Policy | Success | Collision | Incomplete | Mean minimum TTC (s) |
| --- | ---: | ---: | ---: | ---: |
| Original V3 | 294/500 (58.8%) | 206/500 (41.2%) | 0/500 (0.0%) | 0.6003656548142169 |
| Corrected control | 294/500 (58.8%) | 205/500 (41.0%) | 1/500 (0.2%) | 0.6065401801078688 |
| Predictive V1 | 364/500 (72.8%) | 101/500 (20.2%) | 35/500 (7.0%) | 0.6728939419963959 |
| Geometry V2 | 394/500 (78.8%) | 98/500 (19.6%) | 8/500 (1.6%) | 0.6666153731858254 |
| Synchronized retry01 | **376/500 (75.2%)** | **117/500 (23.4%)** | **7/500 (1.4%)** | **0.6436872601882264** |

Retry mean reward is 4.430025122161374 (summary differs only in floating-point
rounding), mean length 52.792, mean travel time 10.5584 seconds, mean unsafe-TTC
events 18.776 and mean interventions zero. All 500 TTC observations in every
arm are finite; no observations were dropped. Raw CSVs remain unchanged.

Exact two-sided paired success comparisons (rescue = reference failure becoming
candidate success; regression = reference success becoming candidate failure):

| Reference | Rescues | Regressions | Exact p | Favorable at p < .05 |
| --- | ---: | ---: | ---: | --- |
| Original V3 | 130 | 48 | 6.342544360226172e-10 | Yes |
| Corrected control | 127 | 45 | 3.0266123277781867e-10 | Yes |
| Predictive V1 | 70 | 58 | 0.33093582894221385 | No |
| Geometry V2 | 45 | 63 | 0.10143273504110499 | No |

**Decision: not retained under the unchanged section 79.5/81 gates.** Five
checks fail: V1 collision ceiling (117 > 101), V1 TTC non-regression, favorable
paired success versus V1, Geometry success floor (376 < 394), and Geometry
collision ceiling (117 > 98). The Geometry incomplete ceiling passes (7 <= 8).
There is still no added significance requirement against Geometry V2.

Relative to Geometry V2, this run has 18 fewer successes, 19 more collisions,
one fewer incomplete episode and lower mean minimum TTC. The paired success
difference against Geometry is not significant at .05; do not claim this single
run proves synchronization is intrinsically harmful. It establishes that this
trained candidate does not meet retention rules. Lower unsafe-event count or
fewer timeouts does not offset the failed safety/success gates. Geometry V2 also
remains unpromoted because of its previously failed V1 TTC gate. Original V3
remains accepted; V4's conservative-waiting rejection is unchanged.

### 86.1 Audit implementation and provenance

Added `scripts/audit_synchronized_retry.py`, reusing the frozen stdlib row,
metric and exact-paired-test helpers without changing their historical source.
It verifies the completed train/evaluate records, final model/training summary,
all 45 source/lineage hashes and full package inventory; reconciles all five
500-row CSVs against summaries; checks evaluation metadata against Geometry V2
apart from the recorded model/protocol difference; and evaluates all 16 gates.
It checks fingerprints again before returning and refuses an existing output.
This is offline analysis, not another rollout or a model-selection step.

The Spreadsheets skill's scientific-research guidance informed independent
raw-data reconciliation, preservation of observations and explicit exclusion/
pairing limits. No workbook was requested, modified or exported. The bundled
Python runtime executed the stdlib audit; the RL environment remains unchanged.

Artifacts in `results/`:

- `ppo_v3_predictive_sync_v1_retry01_development_seed40042.csv`, SHA-256
  `b2865f8d45db051a450744eb94f95850fd742089f6f12445820d6067a3514dfc`.
- Matching `.summary.json`, SHA-256
  `763cfef6e8c4a9364f69896c5ede4a6ece7e3f6dcf0e54ea33b0df140da2338d`.
- `ppo_v3_predictive_sync_v1_retry01_seed42.evaluate.run.json`, SHA-256
  `da3e4e4b98d1f4d754d521d687e300dfa3fee0afb7ed243020d5adef209b5d2d`.
- `ppo_v3_predictive_sync_v1_retry01_development_seed40042.audit.json` contains
  all recomputed metrics, counts, paired tests, gate decisions and fingerprints.

Pairing is positional under the recorded contiguous reset protocol. The CSVs
do not include per-row seed IDs, so seed identity is not independently observable
from the rows alone. This repeatedly consumed development set and one training
seed do not provide independent generalization evidence. No gates, coefficients,
seeds, checkpoints or evaluation protocol were altered in response to results.
Preserve the original interrupted attempt and every prior result. Tests and
independent review of the new audit are recorded below before its scoped commit.

### 86.2 Audit verification gate

The 25 new audit tests pass in 0.13 seconds and focused Ruff passes. They cover
all inclusive count/TTC boundaries, unchanged paired-success requirements,
absence of an extra Geometry significance gate, frozen hashes, existing-output
and create-race protection, failure preservation and all-infinite TTC rejection.
Independent read-only recomputation separately confirmed all 2,500 rows across
the five policies, 45 source hashes, 42-entry predecessor snapshot, final-model
and summary hashes, exact evaluation arguments, all five failed gates and all
paired results. No integrity discrepancy or failed test was found.

Audit source SHA-256:
`0c7ebea4537425f81edf0a9abed0cd2a1cae13e825403a764c0292127bfc2fd7`.
Audit report SHA-256:
`0292a40705889a3b6e75874c4955ecfb5e5ac19420b1909c88d1c104476bd30a`.
No simulation was run during this audit. A fresh full-suite gate is still
required before the next training/evaluation experiment, rather than treating
these analysis-only tests as that gate.

## 87. Preregistered sector-feature comparison (2026-09-11)

The synchronized retry result and failed gates were committed locally as
`9a6e5d1`. Focused checks in the actual repository also passed (25 tests in
0.07 seconds, Ruff and CRLF-aware whitespace checks). No checkpoint, log or
unrelated file was committed, and nothing was pushed.

Under the user's continued-research authorization, the next bounded experiment
is a **two-arm representation comparison**, specified in
`V3_SECTOR_COMPARISON_V1.md` before either arm trains. The section 82 sector
prototype remains byte-identical. The synchronized observation is a shared
measurement pipeline for this comparison, not adoption of its rejected trained
policy. Both arms are fresh PPO models, with no checkpoint transfer.

- Padding control: original synchronized115 inputs plus48 exact zeros; model
  stem `ppo_v3_sector_padding_v1_seed42`, protocol
  `predictive_post_spawn_padding_v1`.
- Sector candidate: same synchronized115 inputs plus the existing48 sector
  values; model stem `ppo_v3_sector_features_v1_seed42`, protocol
  `predictive_post_spawn_sector_v1`.

Both input shapes are163 and use the same256/256 separate actor/value towers.
Initial policy tensor SHA-256 and parameter count must match before either
model is compared. Padding matches shape, nominal parameter count and initial
weights, **not effective capacity**: permanently zero features cannot contribute
or give their connected weights a data gradient. Relative to115 inputs, each
model has24,576 additional first-layer weights across the two towers. Sector
features provide a nonlinear basis, not new independent information.

Frozen feature formula remains section82: p=200[x,y], v=80[vx,vy], radial closing
speed=-dot(p,v)/norm(p),16 ego-relative half-open angular sectors of width pi/8,
nearest actor-center per sector, and features
`[presence, norm(p)/(200*sqrt(2)), clip(closing/(80*sqrt(2)),-1,1)]`.
Empty=`[0,1,0]`; coincident-center and tie rules are unchanged. Retain only the
already observed14 neighbors. Original115 values, forecasts, dynamics and random
state are preserved; no extra visibility, real/noisy sensors, labels, shields,
reward changes or action modes are introduced. This does not implement CARLA,
pedestrians, lane changes or rerouting.

### 87.1 Fixed commands, order and resource boundary

New entry point: `scripts/run_sector_comparison_v1.py`. Before **each** stage,
verify fresh Ruff/full pytest and frozen sources/eligible predecessor artifacts.
One research process at a time. Fixed order, independent of interim results:

```powershell
python -u -m scripts.run_sector_comparison_v1 padding train --refuse-overwrite
python -u -m scripts.run_sector_comparison_v1 sector train --refuse-overwrite
python -u -m scripts.run_sector_comparison_v1 padding evaluate --refuse-overwrite
python -u -m scripts.run_sector_comparison_v1 sector evaluate --refuse-overwrite
```

Both arms keep section84's geometry config SHA
`a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7`,
fresh seed42,200000 requested/200704 expected steps, LR.0003, n_steps1024,
batch64, one environment, stride1000, validation offset70000,50 episodes every
10000 steps, checkpoints every25000, gamma.99, GAE.95, entropy.01, epochs10,
clip.2, vf_coef.5, max_grad.5, advantage normalization, no targetKL or value clip.
Native30 seconds,5Hz policy,15Hz simulation, collision-first reward and predictor
settings (horizon3,margin.5,growth.25,clearance10,speed9,neighbors14) are unchanged.
All section84 dependency/Python versions and CPU threads8/8 remain fixed.

Total batch budget is400000 requested/401408 expected collected training steps
and1000 development episodes, plus the unchanged internal validation. Train both
arms before evaluating either. Do not adjust the second arm from the first's
training reward, validation best or development result. Only completed final
root models are scored. No new training seed, increased per-arm budget, warm
start, resume or automatic failed-run retry is allowed by this batch design.

Each evaluation is500 deterministic episodes on consumed seeds40042--40541,
unsafe TTC threshold2.0, no safety shield/risk-fusion/intent. Output CSV stems:
`ppo_v3_sector_padding_v1_development_seed40042` and
`ppo_v3_sector_features_v1_development_seed40042` under results, with matching
summary JSONs. Model-stem training summaries/run manifests, local models and
unique logs follow section84 naming. Refuse existing targets and preserve failed
records. Bind the new runner, this separate preregistration document, existing
prototype, original45-entry lineage and completed retry evidence with hashes.
No historical source, configuration, model or CSV is replaced.

### 87.2 Frozen decision rules and limits

Audit both arms independently. Retain all16 historical gates in section86. In
aggregate these still demand at least394successes, at most98collisions and8
incomplete, mean minimum TTC>=.6728939419963959, favorable exact paired success
against original V3/corrected control/V1, and every other original condition.
Do not add a significance requirement against Geometry V2.

For a sector-feature benefit claim additionally require favorable paired success
versus padding (rescues>regressions, exact two-sided McNemar p<.05), sector
collisions<=padding collisions, incomplete<=padding incomplete and mean minimum
TTC>=padding mean minimum TTC. These are added before running, not substitutes
for historical gates. Report all arm/reference counts and paired comparisons,
including rejected synchronized retry, plus finite-TTC exclusions and metadata.
Pairing remains positional under recorded sequential resets, not per-row seed IDs.

If only padding passes historical gates, no sector benefit is established. If
sector passes historical gates but fails its padding comparison, feature benefit
is also unestablished. Any full pass is only eligibility for preregistered
replication/fresh evaluation, not acceptance on this consumed one-training-seed
development set. Record every failure or mixed outcome. Do not change gates or
continue identical failed experiments hoping for a favorable draw. Original V3
remains accepted. Implementation/test/review and actual launch are recorded
separately after completion; this preregistration itself starts no process.

### 87.3 Opt-in batch runner implementation

The new runner adds `PaddingPredictiveObservation`, preserving the direct
synchronized115-vector and appending48 float32 zeros with the same declared
space as the sector arm. Geometry, dtype, presence and ego-heading contracts
are checked without new observation/forecast/physics/random calls. Both policies
have **216,580 parameters**. Historical factories, scripts and package files
remain unchanged; process-local adapters are restored in `finally`.

An exclusive cross-process lock prevents overlapping batch stages. Every stage
requires the exact completed predecessor chain (train padding, train sector,
evaluate padding, evaluate sector), with source/model/summary/record/CSV hashes
as applicable. Future-stage artifacts are rejected before out-of-order work.
Abrupt process death can leave a lock and incomplete artifacts; this requires
diagnosis, not automatic removal/retry. Normal exit removes only its own lock.
The user-facing launcher separately checks existing console-log targets and
unrelated active research jobs. The runner preserves failed records and outputs.

Training records initial sorted policy state-dict tensor names, dtypes, shapes
and bytes in a deterministic SHA-256 before `learn`, along with parameter count.
Sector initialization must equal padding's saved **initial** fingerprint before
optimization. It is never compared to the trained padding weights. Protocol,
arm and initial fingerprint are stored on the checkpoint and in summaries.
Verifiers enforce CPU,163 inputs/3 actions, all frozen PPO settings,200704 final
steps and1960 updates, finite weights and summary protocol. Runtime and source
fingerprints are checked at preflight and source/predecessor hashes at completion.

The separate preregistration document is explicitly pinned before the first
run (SHA-256 `fbc0a4383a0d6224a78f6e4cbd7686cb6fa9593f471daa67a78686b99f74d56d`).
Keep that document and the runner unchanged throughout the four-stage batch;
append later status/results here. The old section82 prototype text records its
earlier engineering-only boundary; this separately named protocol is its opt-in
experimental release, not a default factory or accepted-policy change.
Independent review and tests must pass before any launch.

### 87.4 Implementation checks and preflight

Independent code review passed the final runner SHA-256
`e825ed428fe5ea228e4527f9e3e5daa82f9798f7f3da4064106a87cb115609aa`.
The first focused runner Ruff check found three overlong lines; formatting was
corrected and Ruff passed. The new test file's first focused run passed all79
tests but Ruff found four overlong lines. Those formatting issues were corrected,
and the repeated focused gate passed **79 tests in 4.48 seconds** with Ruff clean.
No failed PPO run or numerical research result occurred during implementation.

Tests cover exact arguments, identical163 spaces and preserved115 prefixes,
padding's lack of extra sensing/physics/random draws, stage order/predecessor
chains, all14 existing-output guards across the two arms, lock exclusion/stale
lock preservation, failed-run records, runtime/model/source mismatches and
process-local adapter restoration. Importantly, real fresh CPU PPO models for
both real wrappers have identical initial policy tensor hashes,216580 parameters
and the required seed42/256/256/frozen initial settings. These constructors were
not trained or saved; this is an initialization check, not another experiment.

Actual-repository read-only preflight passes **56 source/input/lineage hashes**,
Python3.12.9, all frozen package versions, Torch threads8/8 and an empty predecessor
chain for padding training. Final source/document bytes match the reviewed
maintenance copies. Fresh full pytest and Ruff in the actual repository remain
the required pre-training gate; its result and launch follow separately.

### 87.5 Fresh full pre-training gate passed

Actual-repository Ruff passed and **432 tests passed in 54.33 seconds**, with
only the two existing unbounded-Box warnings. No test failed. The finalized new
test file SHA-256 is
`1a671e1a0c8b636f1b94c8e66475b32377bffc0d84843be79fbb01e14296946a`.
Final runner SHA and all56 preflight fingerprints remain unchanged. This gate
authorizes only the first padding-control training stage after the final
no-active-research-process/absent-output/absent-console-log check. Each later
training or evaluation stage requires another fresh Ruff/full pytest gate.

The preregistration, new runner/tests and append-only record are committed before
launch. Models/logs/live run records are excluded. Actual process launch and
follow-up retargeting will be recorded below; neither is implied by passing tests.

### 87.6 Actual padding-control training launch

Preregistration, code, tests and the full passing gate were committed as
`88e334f`. Reviewed-access process inspection found no active research training/
evaluation, all23 prospective batch artifact/log/lock targets were absent, and
the final runner fingerprint matched. Launched exactly the first section87.1
command at **2026-09-12T00:13:55.9548674Z** (September11 local time), launcher
PID **32336**, worker PID **15560**, with a hidden window and unique stdout/stderr.

The run record starts **2026-09-12T00:13:58.241868Z**, status `running`, arm
`padding`, mode `train`, protocol `predictive_post_spawn_padding_v1`, with56
source fingerprints and no predecessors. Its lock identifies worker15560.
Both observed command lines match the preregistered padding train command.
Stdout confirms CPU and the unique TensorBoard directory, stderr is empty, and
the first1024 collected steps were observed. This is a real running training
process, not a completed policy or success-rate result.

Recorded initial policy SHA-256:
`ca35bda9cfe2f2afce2eb0cf71d2abd74fb6808d2f935b7e20a5b31c1d054c19`,
with216580 parameters. The sector arm must reproduce this initial fingerprint
before learning. No sector training or either development evaluation has begun.
Next comes verified padding completion and a fresh full gate, then sector
training, then the two fixed evaluations. Do not score padding early or select
its callback-best model. Preserve its mutable run record and ignored model/log
artifacts; this launch note is the only new documentation committed while it runs.

### 87.7 Continued batch follow-up verified

The existing `finish-v3-geometry-experiment` heartbeat was retargeted through
the supported automation tool at **2026-09-12T00:15:41.112Z**. Exact saved-prompt
equality was independently verified. It retains the name **Continue V3 safety
research**, ACTIVE status,30-minute interval, same task
`01a06baf-334e-7182-ab6e-6a4977dadc85` and original creation time. No duplicate
automation, raw configuration edit or permission change occurred.

The prompt now follows the launched padding stage, enforces the fixed four-stage
order, final-model/source/predecessor/initialization checks and fresh full tests
before every subsequent stage. It requires independent audits and all unchanged
historical plus preregistered sector-versus-padding gates, append-only records,
scoped local commits, no model/log commits or Git push, and quiet ordinary
progress. Completed retry01 and the original interrupted attempt are not restart
targets. After the complete batch audit, any further research must be separately
preregistered; no criterion relaxation, unplanned visibility or repeated identical
poor runs. Original V3 is not automatically replaced by a development result.

The OpenAI Docs skill guided this supported update and preservation of the
notification intent. Official scheduled-task guidance confirms that local
follow-up requires the computer powered on and the app running:
https://learn.chatgpt.com/docs/automations?surface=app . No power setting was changed.
The running model itself and its mutable manifest remain local/uncommitted;
the launch record was committed as `11dcee4`.
