# SafeIntent-RL Research Milestones and Engineering Log

**Project:** Intent-Aware and Safety-Constrained Reinforcement Learning for Interactive Autonomous Driving  
**Repository:** [kingofrichnight/RL--Autonomus-car](https://github.com/kingofrichnight/RL--Autonomus-car)  
**Environment:** Gymnasium + HighwayEnv `intersection-v2`  
**Primary algorithm:** Proximal Policy Optimization (PPO)  
**Document type:** Living technical record  
**Created:** 2026-09-03  
**Last updated:** 2026-09-03  

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
| M6A | Evaluate rule-based reference and diagnose PPO V1 | In progress | Evaluator added; 500-episode run pending |
| M7 | Collect intent dataset and train GRU | Planned after M6A baseline diagnosis | Data and training scripts available |
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
| D-015 | Evaluate the rule-based controller before GRU work | Proceed directly to intent learning | PPO V1 success and collision rates require a fair non-learning reference and reward diagnosis | Active |

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
