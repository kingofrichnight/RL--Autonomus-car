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
