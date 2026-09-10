# V3-PredictiveSafety v1: proposed learned-policy successor

Status: implemented engineering candidate, not an accepted driving policy.
Existing V3 configuration, checkpoint and results remain untouched. This is a
new policy trained from scratch on V3's intersection task, not a renamed model
or a fine-tune of the old ZIP. Its 115-input interface is incompatible with V3's
105-input checkpoint. No long training run is released by this implementation.

## Evidence motivating the change

The corrected-reward fusion control reached 33.0% success, 29.8% collision and
36.8% incomplete. Exposing target speed removed incomplete episodes but yielded
59.6% success and 40.4% collision. Both fail their frozen development gates.
Those failures do not establish that any added variable will improve safety.

The new question is: **does giving PPO action-specific forecasts help it learn
when to brake, maintain speed or accelerate through a gap?** Current-state TTC
does not describe the different consequences of those three actions.

## Research and repositories inspected on 2026-09-10

- [Abouelazm et al., 2025, Balancing Progress and Safety](https://arxiv.org/abs/2505.06737):
  risk should be represented before collision, with geometric/dynamic context
  and explicit consideration of the safety/progress tradeoff. Its ellipsoid/RSS
  reward is a relevant alternative, not reproduced here. Do not transfer its
  reported improvement percentage to this experiment.
- [Leng et al., 2025, Risk-Aware RL through Intersection](https://arxiv.org/abs/2503.19690):
  separate safety critics and attention are relevant next algorithmic options.
  We have not reproduced that method or its results.
- [Bastani, Model Predictive Shielding](https://arxiv.org/abs/1905.10691):
  motivates evaluating action consequences. Here forecasts are observations,
  not an override or a proof of recoverability. We do not inherit its guarantees.
- [OmniSafe](https://github.com/PKU-Alignment/omnisafe) and
  [PID Lagrangian methods](https://proceedings.mlr.press/v119/stooke20a.html):
  useful references for a future cost critic and adaptive safety multiplier.
  OmniSafe documents no official Windows support, so it is not added as a new
  dependency to this Windows/SB3 project.
- [menghan-xu/safe-rl-intersection](https://github.com/menghan-xu/safe-rl-intersection):
  a directly relevant repository using separate reward/cost critics, uncertainty
  and continuous acceleration. It also includes behavior cloning from expert
  data; that branch does not match our label-free RL preference. Its README
  safety claims are author-reported, not independently validated here.
- [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv): retain the
  current simulator and inspect the installed controller rather than assume
  constant braking acceleration or replace the task mid-comparison.

## Implemented forecast and formulas

For each action a in [SLOWER, IDLE, FASTER], simulate only an isolated ego copy
for H=3 seconds at 15 Hz. Apply a once, then maintain the resulting target.
The installed controller recomputes steering along ego's known route and uses
its proportional speed controller, rather than v^2/(2b) with an invented b.
The forecast is conditional on this action-and-maintain sequence, not on the
unknown future decisions of the learned policy.

For each visible neighboring actor i, predict constant velocity and heading:

    p_i(t) = p_i(0) + v_i(0) * t
    m(t) = 0.5 m + (0.25 m/s) * t

Only the same sorted visible slots (up to 14) used by native kinematics are
included. No NPC hidden routes, intent labels, future spawns, control law or
future random stream enter these predictions. Ego road geometry is copied;
its forecast uses a separate fixed RNG, not the live simulator RNG.

Use oriented rectangles, not the circumcircles that flagged safe side-by-side
traffic in the static-obstacle check. For axes u aligned with either rectangle:

    g_i(t,a) = max_u [ abs((p_i(t)-p_e(t,a)) dot u)
                       - radius_e(u) - radius_i(u) ] - m(t)
    radius_j(u) = (L_j * abs(long_axis_j dot u)
                  + W_j * abs(lat_axis_j dot u)) / 2

Before subtracting m, nonpositive gap indicates rectangle intersection/touching;
positive gap is a separating-axis distance proxy, not exact Euclidean distance.
Subtracting m makes the feature conservative. A nonpositive margin gap is a
forecast warning, not a measured collision.

Append current target speed / 9 and, per action:

1. clip(min_i,t g_i(t,a) / 10 m, -1, 1);
2. first t with g_i(t,a) <= 0, divided by H (1 if none);
3. accumulated predicted travel distance / (9 m/s * H), clipped to [0,1].

Empty-neighbor scenes use gap=1 and conflict-time=1. A conflict exactly at H
shares the time sentinel with no conflict; the gap feature disambiguates it.
Travel distance is not route completion or safe progress. Native observations
remain the unchanged first 105 inputs; ten appended values yield 115 inputs.

PPO still chooses every action from experience and reward. No action labels,
demonstrations, masks, automatic braking or added risk penalties are used.
Reward keeps V3's -10 collision, +5 arrival, 0.05 speed, 2.0 progress and 0.005
per-decision time cost. Only the documented collision/arrival overlap correction
is enabled. `predictive_collision_cost` is an info diagnostic, not a constrained
optimizer or a new reward. Training/evaluation JSONs record the predictor config.

## Not implemented or claimed

- The margin growth is a frozen engineering heuristic, not calibrated covariance,
  a probabilistic confidence bound, RSS compliance, or a CBF certificate.
- Constant-velocity NPC prediction misses turns, accelerations and reactions;
  15 Hz sampling can miss conflicts between frames. Unseen cars/spawns are absent.
- Horizon-limited forecasts can miss later conflicts. They do not certify any
  action safe, and a favorable forecast does not mean PPO will choose that action.
- No success-rate estimate is available. Neither near-zero accidents nor an
  increase over V3 can be promised before matched evaluation.
- This remains longitudinal intersection driving. Pedestrians, lateral obstacle
  avoidance and rerouting stay in their separate scenario-development track.

## Next experiment design, not a long-run command

First pass the full test suite and the 32-step engineering train/save/reload
check. Then freeze a compute-feasible matched training package with a 106-input
target-only corrected-V3 control and this 115-input predictor arm. Both must
share reward, dynamics, PPO settings, seeds and training budget. The old fusion
control is not this ablation: its input representation differs.

Use only the existing development seeds for initial screening; do not call them
an untouched holdout. Freeze success, collision, incomplete, TTC and paired-test
gates before running either arm. Preserve unsuccessful results. Final claims
require independent training seeds and a separately frozen untouched evaluation.
Do not add reward penalties, a Lagrange optimizer and a predictor simultaneously
and then attribute the result to the predictor alone.

A later constrained-RL option is explicitly separate:

    maximize J_reward(theta), subject to J_collision(theta) <= d
    A_combined = A_reward - lambda * A_cost
    lambda <- max(0, lambda + eta * (estimated_J_collision - d))

That requires a cost critic, defined discounted/episodic cost semantics, and a
prospectively selected budget d. It is not equivalent to adding a fixed negative
reward, and it is not implemented by the current info cost field.
