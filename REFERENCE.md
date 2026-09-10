# Research Reference: Safe Obstacle Avoidance, Pedestrian Yielding, and Rerouting

## Purpose

This reference supports the next phase of **SafeIntent-RL**: extending the
existing unsignalized-intersection PPO project to handle static obstacles,
pedestrians, and a safe alternate route or lane. It is deliberately a curated
technical reading list, rather than a claim that it includes every paper in
the field. The sources were selected for direct relevance, reproducible code,
and methods that can be evaluated in simulation.

The recommended architecture is **hybrid**:

```text
route graph / lane availability ──> high-level route choice
                                      │
pedestrian + obstacle state ───────> PPO driving policy ──> proposed action
                                      │                            │
predicted occupancy + TTC/CPA ─────> safety shield <───────────────┘
                                      │
                              brake / yield / safe lane change
```

This avoids asking one feed-forward RL policy to simultaneously solve global
routing, tactical driving, and guaranteed collision avoidance. The current
project's PPO remains the tactical controller; deterministic planning and a
safety layer make its behavior easier to test and explain.

## Recommended reading: highest-value papers

| Priority | Paper | Why it is useful here | Adopt now? |
|---|---|---|---|
| 1 | [Uncertainty-Aware DRL for Autonomous Vehicle Crowd Navigation in Shared Space](https://arxiv.org/abs/2405.13969) — Golchoubian et al., 2024 | Closest match to pedestrian-aware AV decision-making. It joins pedestrian trajectory prediction, uncertainty, and RL navigation; its reported comparison found lower collisions and larger pedestrian clearance than a prediction-unaware baseline. | Yes: use its uncertainty-aware pedestrian-risk concept and evaluation measures; do **not** copy its older software stack. |
| 2 | [Safe Reinforcement Learning with Nonlinear Dynamics via Model Predictive Shielding](https://arxiv.org/abs/1905.10691) — Bastani, 2019 | Formalizes the idea already present in this project: a learned policy proposes an action and a backup policy overrides it when safety cannot be maintained. | Yes: use as the design rationale for a transparent, logged shield. |
| 3 | [Trajectory Planning for Autonomous Vehicles Using Hierarchical Reinforcement Learning](https://arxiv.org/abs/2011.04752) — Ben Naveed, Qiao & Dolan, 2020 | Separates high-level subgoals from low-level motion control. This is the right conceptual model for “reroute” rather than treating it as a stronger brake action. | Yes: implement a simple, deterministic route layer first; HRL is a later comparison, not the first implementation. |
| 4 | [MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning](https://arxiv.org/abs/2109.12674) — Li et al., 2021 | Demonstrates generated road networks and safe-driving tasks with cones, barriers, and broken-down vehicles. It is an excellent next simulator when the present HighwayEnv task is mature. | Study / later migration. |
| 5 | [Dynamic Model Predictive Shielding for Provably Safe Reinforcement Learning](https://arxiv.org/abs/2405.13863) — Banerjee et al., 2024 | Addresses a central failure mode of simplistic shields: a safe override can stop progress. Recovery actions should consider both immediate safety and forward progress. | Use as a future shield benchmark; too large to reproduce first. |
| 6 | [Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization](https://arxiv.org/abs/2202.10341) — Li, Peng & Zhou, 2022 | Gives a useful benchmark structure: route-progress reward, speed reward, terminal arrival reward, collision cost, and separate unseen test environments. | Yes: use the evaluation separation and cost logging, not human-in-the-loop training. |
| 7 | [Learning to Drive in a Day](https://arxiv.org/abs/1807.00412) — Kendall et al., 2019 | Useful counterpoint: end-to-end RL can learn narrow driving skills, but the paper itself motivates keeping the current project simulation-first and modular. | Background reading only. |

### What the papers collectively imply

1. **Pedestrians are dynamic and uncertain.** Do not model them only as
   stationary obstacles. Give each pedestrian a position, velocity, crossing
   state/intent, and a conservative uncertainty radius.
2. **Rerouting is a planning decision.** A safe lane or branch should be
   chosen from a route graph; PPO should decide *how* to follow/yield within
   that valid choice.
3. **Safety must be measured independently of reward.** A large negative
   collision reward is not a safety guarantee. Log safety violations, minimum
   clearance, TTC/CPA, shield interventions, and route completion separately.
4. **A shield must not become a deadlock machine.** Report completion and
   delay alongside collision outcomes. A policy that never collides because it
   never moves is not successful.

## Recommended open-source projects

### Best three to use or study

| Project | Link | Best use for this project | Decision |
|---|---|---|---|
| **HighwayEnv** | [eleurent/highway-env](https://github.com/eleurent/highway-env) | The existing project already uses its intersection environment and Gymnasium-compatible interface. It is the fastest place to add a small controlled obstacle / crossing-pedestrian extension and maintain reproducible PPO experiments. | **Use now.** Extend locally; keep the extension small and tested. |
| **MetaDrive** | [metadriverse/metadrive](https://github.com/metadriverse/metadrive) | A lightweight, extensible simulator with procedural maps, navigation state, obstacle-safety tasks, and scalar, lidar, camera, and bird's-eye observations. Its paper specifically describes cones, barriers, and broken-down vehicles. | **Use after the HighwayEnv study** as a robustness/generalization validation, not as a mid-experiment replacement. |
| **CARLA ScenarioRunner** | [carla-simulator/scenario_runner](https://github.com/carla-simulator/scenario_runner) | Defines repeatable traffic scenarios and route-based tests, and supports OpenSCENARIO. Its standard scenarios include obstacle-following and a turning vehicle that must yield to a cyclist. | **Use late** for a visually richer demo or validation. CARLA + ScenarioRunner is heavier than necessary for the main result. |

### Strong reference implementation for pedestrian uncertainty

| Project | Link | What to extract | Caution |
|---|---|---|---|
| **UncertaintyAware_DRL_CrowdNav** | [Golchoubian/UncertaintyAware_DRL_CrowdNav](https://github.com/Golchoubian/UncertaintyAware_DRL_CrowdNav) | Repository structure separating pedestrian prediction (`ped_pred`), pedestrian simulation (`ped_sim`), planning, and RL; prediction uncertainty as a planning input. | Treat as a design reference. It uses an older Python/PyTorch/OpenAI Baselines stack, so copying dependencies into this Stable-Baselines3 project would add unnecessary risk. |

### Why these are better choices than a full simulator rewrite

HighwayEnv preserves the project’s current PPO, intent pipeline, metrics, and
seed protocol. MetaDrive is the best controlled next step because it has
generated maps and obstacle tasks. CARLA is suitable when visual realism and
standardized scenarios become more important than rapid training. A simulator
migration changes too many variables to be the first test of whether the
method itself works.

## Formulas to use

All symbols should be logged with units in SI units: metres, seconds, and
metres per second. These formulas are decision aids and evaluation metrics;
they are not a proof of real-world road safety.

### 1. Relative motion, time-to-collision, and closest point of approach

For ego position/velocity \(p_e, v_e\) and obstacle or pedestrian
\(p_i, v_i\), define

\[
p = p_i-p_e, \qquad v=v_i-v_e.
\]

For radial closing speed, use

\[
v_{close}=\max\left(0,-\frac{p^T v}{\lVert p\rVert+\epsilon}\right),
\qquad TTC=\begin{cases}
\frac{\lVert p\rVert-d_{safe}}{v_{close}}, & v_{close}>\epsilon \\
\infty, & \text{otherwise.}
\end{cases}
\]

For crossing traffic, radial TTC alone can be misleading. Use closest point of
approach (CPA):

\[
t_{CPA}=\operatorname{clip}\left(-\frac{p^T v}{\lVert v\rVert^2+\epsilon},0,H\right),
\qquad d_{CPA}=\lVert p+t_{CPA}v\rVert.
\]

Flag a predicted conflict when \(t_{CPA}<H\) and
\(d_{CPA}<d_{safe}\). The project already has TTC and CPA-style utilities;
the important extension is to calculate them for each pedestrian and for
static obstacles, not only nearby cars.

### 2. Uncertainty-inflated pedestrian clearance

For a predicted pedestrian mean location \(\hat p_i(t)\) and covariance
\(\Sigma_i(t)\), use a conservative radius

\[
d_{safe,i}(t)=r_e+r_i+m+k\sqrt{\lambda_{max}(\Sigma_i(t))}.
\]

Here \(r_e\) and \(r_i\) are vehicle/pedestrian geometry radii, \(m\) is a
base margin, and \(k\) controls conservatism. If there is no trained predictor
yet, approximate uncertainty with a behaviour-dependent radius, e.g. a larger
one for an unknown or likely-crossing pedestrian. This is a lightweight
adaptation of the prediction-uncertainty idea in Golchoubian et al.^1

### 3. Braking-feasibility guard

For available deceleration \(a_{brake}>0\), reaction/actuation delay
\(\tau\), and ego speed \(v_e\), use the conservative stopping distance

\[
d_{stop}=v_e\tau+\frac{v_e^2}{2a_{brake}}+m.
\]

If the projected longitudinal free distance is below \(d_{stop}\), the shield
must prohibit acceleration and select braking/yielding. Calibrate
\(a_{brake}\), \(\tau\), and \(m\) from the simulator; do not borrow real-road
values without validation.

### 4. Control-barrier-style safety condition

For any actor, define a distance barrier

\[
h_i(x)=\lVert p_i-p_e\rVert^2-d_{safe,i}^2.
\]

Safe states satisfy \(h_i(x)\ge0\). A discrete-action shield can estimate
the next state under every candidate action and retain only actions satisfying

\[
h_i(x_{t+1}) \ge (1-\alpha)h_i(x_t)
\]

for every relevant actor. Among safe actions, choose the PPO proposal if it is
safe; otherwise choose the action that maximizes progress while maximizing the
minimum predicted barrier. This is a practical, finite-action analogue of
barrier-based safety reasoning, not a formal certificate.

### 5. Route cost and rerouting

Represent driveable lanes/branches as a directed graph \(G=(V,E)\). For an
edge \(e\), define

\[
c(e)=\ell(e)/v_{ref}+w_{risk}R(e)+w_{block}B(e)+w_{change}C(e),
\]

where \(\ell/v_{ref}\) estimates travel time, \(R(e)\) is predicted
pedestrian/vehicle risk, \(B(e)\) marks an impassable blocked edge, and
\(C(e)\) discourages needless lane changes. Recompute a shortest path (A* or
Dijkstra) when the next edge is blocked or its risk exceeds a fixed threshold.
The route layer should return a local target lane/branch and remaining distance
as part of PPO's observation.

This is intentionally not “RL learns a city map.” It makes rerouting
deterministic, inspectable, and achievable in the current project.

### 6. Reward and cost: keep them separate

Use reward to learn useful behavior, and a separate cost ledger to report
safety. A compact starting objective is

\[
r_t=w_p\Delta s_t-w_t\Delta t-w_c\mathbf{1}[collision]
-w_u\sum_i\phi(TTC_i,d_{CPA,i})-w_j|a_t-a_{t-1}|,
\]

where \(\Delta s_t\) is route progress, \(\phi\) is a bounded near-conflict
penalty, and \(w_j\) discourages oscillatory braking/accelerating. Add a
large terminal collision penalty, but do not tune the project around reward
alone. Record:

\[
J_{cost}=\sum_t \left(c_{collision}+c_{near\_miss}+c_{shield}\right).
\]

Shield interventions are *not* collisions; report them separately so a method
cannot appear safe solely because a conservative shield did all the work.

## Proposed experiment design

### Scenario ladder

Build in this order, preserving a fixed baseline before advancing:

1. **Static obstacle:** one blocked lane, safe adjacent lane available.
2. **Static obstacle, no alternate lane:** brake and wait/terminate safely.
3. **Crosswalk pedestrian:** pedestrian enters a known crossing; ego yields
   and resumes after clearance.
4. **Uncertain pedestrian:** vary walking speed, start time, crossing intent,
   and visibility; hold out combinations for evaluation.
5. **Combined intersection:** mixed vehicles, a pedestrian, and an optional
   blocked branch. This is a final robustness test, not the first training
   environment.

### Baselines and ablations

| ID | Controller | Question answered |
|---|---|---|
| B0 | Rule-based brake/yield + deterministic reroute | Is the environment and route graph valid before PPO? |
| B1 | Current PPO with new observation but no shield | Can learning react to hazards alone? |
| B2 | PPO + deterministic reroute | Does route availability help without a safety override? |
| B3 | PPO + TTC/CPA/braking shield | Does the shield reduce unsafe events, and at what progress cost? |
| P1 | PPO + uncertainty features + reroute + shield | Does the complete method improve the safety/completion tradeoff? |

Run matched seeds and retain all outcomes, including timeouts and cases where
the shield intervenes repeatedly. Train on one scenario distribution and test
on held-out pedestrian timing, obstacle position, and traffic-seed ranges.

### Minimum metrics

| Category | Metrics |
|---|---|
| Safety | Collision rate by actor type; near-miss rate; minimum pedestrian clearance; minimum TTC; minimum CPA distance; unsafe-TTC decision count |
| Task | Route completion; reroute success; correct-yield rate; blocked-route deadlock rate |
| Efficiency | Travel time; distance; mean speed; unnecessary stop time |
| Intervention | Shield intervention rate; interventions per episode; fraction of proposed actions overridden |
| Robustness | Performance on unseen traffic/pedestrian timings and obstacle locations; confidence intervals across training seeds |

## Implementation decisions for this repository

1. Retain the existing `intersection-v2` research record and do not overwrite
   its V3 baseline results.
2. Create a new environment/configuration family (for example
   `obstacle_pedestrian_v1`) rather than editing the old experiment in place.
3. Implement the deterministic route graph and a simple rule-based controller
   before PPO training. If that controller cannot complete the scenario safely,
   the RL result is not interpretable.
4. Add pedestrian/obstacle features, target-lane/route features, and the
   current controller target speed to the observation with explicit
   non-interference tests for older configurations.
5. Reuse the existing evaluator and extend its CSV schema with actor-specific
   collisions, clearance, reroute, and shield metrics.
6. Freeze scenario parameters, seed ranges, reward configuration, and safety
   thresholds before each long PPO run. Do not change them after seeing the
   holdout results.
7. Treat CARLA as a later validation/demo environment only after a method
   succeeds in the controlled HighwayEnv experiment.

## Suggested thesis/report claim

> In a controlled simulated urban-driving task, evaluate whether uncertainty-
> aware pedestrian risk features, deterministic local rerouting, and a
> TTC/CPA/braking safety shield improve safe route completion over PPO and
> rule-based baselines under held-out obstacle and pedestrian scenarios.

This is a testable, appropriately scoped claim. It does **not** claim real-road
autonomy, pedestrian perception from camera images, or universally guaranteed
safety.

## Sources

1. Golchoubian, M., Ghafurian, M., Dautenhahn, K., & Azad, N. L. (2024).
   [Uncertainty-Aware DRL for Autonomous Vehicle Crowd Navigation in Shared Space](https://arxiv.org/abs/2405.13969).
2. Bastani, O. (2019). [Safe Reinforcement Learning with Nonlinear Dynamics
   via Model Predictive Shielding](https://arxiv.org/abs/1905.10691).
3. Banerjee, A., Rahmani, K., Biswas, J., & Dillig, I. (2024). [Dynamic Model
   Predictive Shielding for Provably Safe Reinforcement Learning](https://arxiv.org/abs/2405.13863).
4. Ben Naveed, K., Qiao, Z., & Dolan, J. M. (2020). [Trajectory Planning for
   Autonomous Vehicles Using Hierarchical Reinforcement Learning](https://arxiv.org/abs/2011.04752).
5. Li, Q., Peng, Z., & Zhou, B. (2022). [Efficient Learning of Safe Driving
   Policy via Human-AI Copilot Optimization](https://arxiv.org/abs/2202.10341).
6. Li, Q. et al. (2021). [MetaDrive: Composing Diverse Driving Scenarios for
   Generalizable Reinforcement Learning](https://arxiv.org/abs/2109.12674).
7. Kendall, A. et al. (2019). [Learning to Drive in a Day](https://arxiv.org/abs/1807.00412).
8. Leurent, E. (2020). [Safe and Efficient Reinforcement Learning for
   Behavioural Planning in Autonomous Driving](https://pepite-depot.univ-lille.fr/LIBRE/EDSPI/2020/50376-2020-Leurent.pdf).
9. [HighwayEnv official repository](https://github.com/eleurent/highway-env).
10. [MetaDrive official repository](https://github.com/metadriverse/metadrive).
11. [CARLA ScenarioRunner official repository](https://github.com/carla-simulator/scenario_runner).
12. [UncertaintyAware_DRL_CrowdNav code repository](https://github.com/Golchoubian/UncertaintyAware_DRL_CrowdNav).
