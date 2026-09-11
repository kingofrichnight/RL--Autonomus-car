# V3: prediction audit and RL successor research

Date: 2026-09-11. Status: **research proposal, not implemented or evaluated**.
CARLA is deferred. Accepted V3, the predictive-v1 checkpoint, all previous
experiments, and the frozen acceptance gates remain unchanged.

## Recommendation

First repair the observation contract in separately named PPO experiments.
Then test **V3-RecurrentConstrainedPPO**: a policy with temporal memory and
separate learned reward, collision-cost and incompletion-cost critics. This is
a project-specific combination of established methods, not a claim of a novel
published algorithm or a guaranteed solution.

The driving decisions remain reinforcement learning. Do not reintroduce a
hand-written action shield or driver-class label training. Deterministic
geometry calculations are not themselves RL; neural-network layers are trained
by RL only when their optimization comes from the RL objective/value targets.

## 1. What the project evidence actually establishes

The completed stress study has 100 episodes per scenario/arm, with the same
100 seeds reused across five scenarios. Predictive v1 improves observed success
and collision rates versus its corrected matched control, but has 2–9%
incompletion. Its earlier development result has 7% incompletion and remains
rejected under the frozen 2% ceiling. See `STRESS_TEST_RESULTS.md` and
MILESTONES sections 75–77; this is not final multi-training-seed evidence.

Read-only analysis of authoritative `candidate_recovery_01` JSONL files:

- All nine dense incompletions spend 25.6–27.4 seconds targeting zero speed
  (mean 26.356); sampled stopped time averages 23.756 seconds. Their final route
  progress is 33.2–48.0%. This proves persistent stopping, not unnecessary stopping.
- Of 37 aggressive-traffic collisions, 24 have zero sampled stopped time and
  seven have at least five seconds stopped. There is no single proven cause.
- Aggressive seed 80110 collides despite recorded minimum radial TTC 2.114 s
  and zero unsafe-TTC events. The historical radial metric is not a reliable
  stand-alone collision detector. Keep it for historical comparisons, adding
  geometric diagnostics separately.

These counts are analysis of existing saved outcomes, not new evaluation runs.

## 2. Concrete observation/prediction weaknesses

### A. Position information is clipped away

The predictor YAML specifies a complete `observation` dictionary but omits
`features_range`. Installed HighwayEnv applies a shallow config update, so the
intersection's native ranges are replaced by the kinematics fallback.

After Ruff and 229 tests passed, a configuration-only reset on consumed seed
80042 confirmed:

| Setting | Effective value |
|---|---|
| Observation shape | 115 |
| x range | [-200, 200] m |
| y range | **[-4, 4] m** |
| vx/vy ranges | [-80, 80] m/s |
| clip | true |
| see_behind | false |

For example, y = 10 m and y = 50 m both encode +1 in that coordinate. Other
features can still differ; this does not mean the entire observations are
identical. The geometric predictor reads uncompressed positions but returns only
nine aggregate values, so it only partially restores the lost information.

This mechanism is verified; its causal contribution to collisions is unmeasured.
Sources: `configs/intersection_v3_predictive_safety_v1.yaml:24`, installed
`highway_env/envs/common/abstract.py:127`, `observation.py:214`.
The official [observation documentation](https://highway-env.farama.org/v1.9.1/observations/)
describes explicit normalization ranges and the absolute ego/relative NPC rows.

### B. Base observations and forecasts can use different actor sets

Installed `IntersectionEnv.step()` computes the base observation, then removes
and spawns actors before returning. `PredictiveSafetyObservation.forecast()`
subsequently queries the live road. Thus the kinematic block and forecast block
can include different actors. This is a code-ordering issue to test, not proof
of future-information leakage or the cause of a particular failed episode.

### C. The forecast is not a complete future policy rollout

Current forecasts apply one action, then hold its target for three seconds.
NPCs maintain constant velocity/heading; output per action is minimum inflated
gap, first conflict time and travel distance. This omits when a conflict clears
and how repeated braking/acceleration changes the future. At frame zero a margin
conflict can set every action's first-conflict time to zero.

The native speed-change actions select the next index relative to **actual
speed**, not simply the old target index. A prediction change must preserve that
semantics; a one-action forecast is not equivalent to repeated full braking.

Fourteen selected neighbors are not necessarily the fourteen most dangerous:
the native order is absolute lane-projected distance, with a rear exclusion.
Whether an omitted actor causes failures requires actor-identity traces.

### D. The deadline objective needs an explicit decision

Training uses a stateless MLP, no remaining-time input, and native truncation at
the deadline. SB3 bootstraps such time-limit truncations, which is correct for an
external sampling cutoff. Evaluation nevertheless counts not finishing by the
deadline as failure. If the new task is explicitly finite-horizon, remaining
time and terminal/cost semantics must be defined prospectively. This is not an
SB3 bug and must not be changed silently in historical V3.
[Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
explains the distinction.

## 3. Targeted V3 repair: controlled sequence

1. **V3-PredictiveGeometry v2:** change only the observation normalization
   contract, explicitly specifying ranges. A candidate is x/y ±200 m with the
   existing velocity scales retained; it covers the current 200 m observation
   radius. Freeze exact values and tests before training. Preserve all rewards,
   actions, dynamics, forecast settings and budgets. Retrain a new checkpoint;
   never feed reinterpreted inputs into an old checkpoint and call it a fair test.
2. **Snapshot-aligned variant:** separately construct kinematics and forecast
   from one documented actor snapshot. Verify no extra RNG draws or vehicle
   mutation. Keep this distinct from the scaling comparison.
3. On consumed development seeds, replay the nine dense timeouts and 37
   aggressive collisions with matched controls. Require reproduction of saved
   outcomes and log clipping, selected actors, action probabilities, speed/target,
   per-action conflict timing and subsequent prediction residuals.

The diagnostic order may precede training, but no replay command is released in
this research note. It must be hash-bound and test-gated first. If forecasts are
wrong, investigate tracked acceleration/turn history and conflict-clearance
intervals. If forecasts are useful but the policy ignores them, prioritize policy
memory/value learning. Neither conclusion follows from stopping averages alone.

## 4. More advanced successor: V3-RecurrentConstrainedPPO

### Memory learned through RL

Let `o_t` be the validated observation and `u_t` the fraction of task time
remaining. A proposed recurrent representation is

\[
h_t = \operatorname{LSTM}_{\theta}([o_t,u_t],h_{t-1}),\qquad
a_t\sim\pi_\theta(\cdot\mid h_t).
\]

The memory summarizes observed motion rather than predicting a driver-class
label. PPO trains the actor from rewards and advantages. Recurrent model-free
RL is a credible partial-observability baseline, but published benchmark results
are not evidence of improvement in our intersection.
[Ni et al., recurrent RL](https://arxiv.org/abs/2110.05038).

[SB3-Contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
supports discrete actions and LSTM policies. Its evaluation must carry recurrent
states and reset them independently per episode/environment. A stateless
`predict(obs)` loop would invalidate the intended memory experiment.

Scene rows can exchange actor identities when sorted; do not treat a row history
as a tracked vehicle. Attention/set encoding is a later architecture ablation,
not a required first addition. A simple recurrent baseline is easier to attribute.

### Learn completion and collision costs separately

For an episode trajectory `tau`, define simulator outcome signals:

\[
C_c(\tau)=\mathbf1[\text{collision}],\qquad
C_i(\tau)=\mathbf1[\text{deadline reached without success or collision}].
\]

Proposed constrained RL objective:

\[
\max_\theta J_R(\theta)\quad\text{subject to}\quad
\mathbb E[C_c]\le d_c,\quad \mathbb E[C_i]\le d_i.
\]

This explicitly disallows treating indefinite waiting as the solution to safety.
Budgets are design inputs, **not predicted rates or proven guarantees**. They
must be fixed before training; no new budget is selected in this note. Existing
acceptance gates remain unchanged. Some traffic states may not admit success
under the existing action set/deadline; learning cannot create a safe option.

Use a reward critic and two separate cost critics. One PPO-Lagrangian adaptation
uses the combined advantage

\[
\widetilde A_t=
\frac{A_t^R-\lambda_c A_t^c-\lambda_i A_t^i}{1+\lambda_c+\lambda_i},
\qquad
\lambda_j\leftarrow[\lambda_j+\eta_j(\widehat J_{C_j}-d_j)]_+.
\]

The actor maximizes PPO's clipped likelihood-ratio objective using this
advantage. Collision and timeout penalties adapt to different measured failures.
The cost critics learn from interaction through return/TD targets, not expert
actions or cautious/normal/aggressive labels. They estimate eventual outcome
cost under the policy, not exact future vehicle trajectories or certified risks.

This proposal follows constrained-RL principles from
[CPO](https://proceedings.mlr.press/v70/achiam17a.html) and Lagrangian/PID training
methods from [Stooke et al.](https://proceedings.mlr.press/v119/stooke20a.html);
it is not an implementation of CPO's trust-region algorithm or its theorem.
PI/PID multiplier stabilization can be a later isolated experiment if ordinary
updates oscillate. It adjusts training penalties, not steering/braking actions.

Implementation requirements: count each event once; use undiscounted episodic
costs when claiming event probabilities; distinguish rollout boundaries from
task deadlines; specify cost-advantage scaling; save critic/multiplier/recurrent
state; and audit finite gradients. Retaining V3's original collision reward plus
a collision constraint is intentional additional pressure, not a hidden reward
coefficient edit. Custom categorical cost buffers/critics are needed; existing
safe-RL libraries must not be assumed drop-in compatible.

## 5. Methods researched but not selected as the first upgrade

- [MultiPath](https://arxiv.org/abs/1910.05449) and
  [Trajectron++](https://arxiv.org/abs/2001.03093) provide useful multimodal
  trajectory-prediction ideas. Their trajectory-fitting objectives are not RL.
  Future-observation targets need no human labels, but that remains supervised
  prediction from self-collected data. Under the requested RL-only learning
  preference, these are references, not silently added training components.
- The previously selected
  [uncertainty-aware DRL crowd-navigation work](https://arxiv.org/abs/2405.13969)
  motivates uncertainty-aware inputs, not a guarantee that bigger margins help
  our traffic policy. Its pedestrian task differs from this benchmark.
- A new blanket TTC/CPA shield or globally stronger collision penalty repeats
  failure directions already in the record. No such action override is proposed.
- CVaR is not automatically a stronger choice for binary collisions. For an
  upper-tail confidence alpha and Bernoulli collision probability p,
  CVaR = min(1, p/(1-alpha)); at alpha=.95 it saturates above 5% collision.
  This elementary deduction means it cannot distinguish our 20–40% collision
  rates using a binary loss alone. Severity-based risk would be a different,
  separately defined experiment, not an immediate fix.

## 6. Evidence needed before acceptance

Compare observation-only repair, memory-only, constrained-objective-only and
combined variants with matched observation/deadline controls where needed.
Avoid attributing a combined architecture gain to one component. Preregister
seeds, environment-step budgets, cost budgets, multipliers, normalization,
checkpoint selection and all evaluation criteria; report wall-clock cost too.

Use consumed seeds for diagnosis, then multiple independent training seeds and
untouched paired evaluation. Report collision, success and incompletion jointly,
including each stress scenario and uncertainty. Preserve unsuccessful runs.
The old 2% incompletion ceiling is not relaxed. A new-policy gain is unproven
until measured; no 80%, 90% or zero-collision promise is justified.

## Work performed for this research note

Primary-source research, local source inspection, existing-JSONL analysis, Ruff,
229 passing tests (10.49 s, two existing Box warnings), and one configuration-only
reset on seed 80042. No policy steps, scored episodes, new training, production
source/config changes, installations, staging, commits or pushes were performed.
