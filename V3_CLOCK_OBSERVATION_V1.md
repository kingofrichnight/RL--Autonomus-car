# V3 remaining-time observation v1: controlled RL preregistration

Status: prospective scientific design and implementation/test plan, 2026-09-12.
This document releases no implementation, command, training or evaluation.
Read MILESTONES95-96 and preserve every historical experiment and decision.

## Question and limits

Does explicit remaining-time information help fresh PPO learn more successful
crossing without worsening collisions, incomplete episodes or minimum TTC,
relative to a same-width zero-input control?

The completed timeout diagnosis reproduced111 already-scored cases. Long
zero-target intervals and repeated relaunches motivate this question; they do
not establish that earlier motion was safe. This is a clock-information test,
not an urgency reward, forced-motion rule or proof of the cause of waiting.
No expected success percentage is asserted.

[Pardo et al., Time Limits in Reinforcement Learning, ICML2018](https://proceedings.mlr.press/v80/pardo18a.html)
distinguishes finite-task deadlines from artificial interruptions of continuing
tasks. Time can be relevant state information for a finite task, whereas an
artificial interruption retains a continuation value. The
[Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
also distinguishes termination from truncation when forming learning targets.
These references motivate the hypothesis, not a safety guarantee.

Our installed HighwayEnv reports expiry as truncation, and SB3 bootstraps from
the terminal observation on pure truncations. Preserve that convention in both
arms. The proposed test therefore does not fully implement a finite-horizon
terminal-objective correction. Changing termination or bootstrapping is a
different intervention requiring a separately named experiment.

## Exactly two fresh arms

| Arm | Appended value | Model stem | Observation protocol |
| --- | --- | --- | --- |
| zero | exact float32 zero | ppo_v3_clock_zero_v1_seed42 | predictive_post_spawn_clock_zero_v1 |
| clock | remaining-time fraction | ppo_v3_clock_remaining_v1_seed42 | predictive_post_spawn_clock_remaining_v1 |

Both use the same synchronized115-input predictive parent, append exactly one
value, and produce finite float32 vectors of shape(116,). The parent is
`SynchronizedPredictiveObservation(PredictiveSafetyObservation(...))`, created
through the unchanged intersection factory and configuration below. Its native
105 values, target-speed value and nine forecast values remain byte-identical
for the same supplied trajectory. No48-value sector features are included.
Different policies may generate different trajectories; cross-arm prefix
identity is not expected after their actions diverge.

This selects a common measurement pipeline, not the rejected synchronized
checkpoint. Both policies train from scratch. Do not load a historical115- or
163-input policy into a116-input wrapper or replace a retained model.

Use separate actor/value networks with two256-unit hidden layers and three
discrete actions. Each arm has192,516 parameters:

`2 * ((116 * 256 + 256) + (256 * 256 + 256)) + (256 * 3 + 3) + (256 + 1)`.

This is512 more than the115-input architecture. Equal dimensions, parameter
counts and initial tensor hashes control initialization, not equal effective
capacity: zero-connected weights do not receive data-dependent gradients.
Initial action probabilities need not match when the clock input differs.
Measure and freeze a new116-input pre-learning hash; never reuse the historical
163-input hash or fabricate an unmeasured fingerprint.

## Clock and observation timing

For actual elapsed simulation time `t = env.unwrapped.time` in seconds and the
unchanged configured duration `D = 30.0` seconds, append:

`c(t) = float32(min(1.0, max(0.0, (D - t) / D)))`.

The zero arm appends `float32(0.0)`. Both appended Box bounds are[0,1], preserving
the parent's bounds for the first115 entries. Do not normalize another way,
round time to steps, change units, clip the original115 values, or add a second
time/progress feature. Reject nonfinite time, negative time, a changed duration,
incorrect parent/configuration contract or invalid supplied observation.

Read time only after the underlying reset or step returns, so the clock belongs
to that returned observation. On reset at actual t=0 it is1. At t=15 it is0.5;
at t=30 or later it is0. A terminal success/collision before the deadline keeps
its actual positive remaining fraction: do not set it to0 merely because done.
The zero arm is0 at every point, including resets and terminal observations.

Preserve floating-time behavior: a returned time just below30 can produce a
tiny positive fraction. The historical151-step/approximately30.2-second timeout
must not be shortened to150 steps, and no new TimeLimit wrapper is allowed.
Read actual time rather than maintaining a separate counter or RNG state.

The outer clock/zero wrapper performs exactly one delegation per reset/step and
no extra observe, forecast, reset, physics, random or action-selection calls.
Its constructor performs none of those calls either.
The common synchronized parent retains its already registered native refresh.
Preserve reward, info, terminated and truncated unchanged. Do not mutate the
supplied115-vector or underlying state. There are no labels, teacher actions,
new sensors, rear visibility, memory, shield, route-progress input or policy
override. PPO continues to learn actions from interaction and the same reward.

## Unchanged environment, reward and runtime

Config: `configs/intersection_v3_predictive_geometry_v2.yaml`, SHA-256
`a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7`.

Keep intersection-v2, duration30, simulation15Hz, policy5Hz, spawn probability0.6,
initial vehicles10, collision reward-10, arrival reward5, high-speed reward0.05,
reward normalizationFalse, progress weight2, per-step time penalty0.005,
arrival distance25 and collision-first outcome/reward handling. Driver-behavior
wrapper and its existing default probabilities remain unchanged.

Predictive settings remain horizon3, margin0.5, uncertainty growth0.25,
clearance scale10, speed scale9 and14 neighbors. Native kinematics retain15x7
sorted relative normalized values, position ranges[-200,200], velocity
ranges[-80,80], no see-behind and the existing obstacle-visibility contract.
Actions remain SLOWER/IDLE/FASTER with target speeds0/4.5/9m/s.

Runtime remains the actual project's CPython3.12.9 virtual environment,
stable-baselines3 2.9.0, Torch2.13.0, NumPy2.5.2, Gymnasium1.3.0 and HighwayEnv1.12.1.
CPU only, Torch intra/inter-op threads8/8. Preserve all frozen dependency bytes
and package origins, not just these six version strings. No package installation,
device change, paid/external compute or CARLA work is part of this experiment.

## Fixed PPO protocol, reset behavior and budget

Each arm uses fresh PPO MlpPolicy, seed42, learning rate0.0003, n_steps1024,
batch64, n_envs1, n_epochs10, gamma0.99, gae_lambda0.95, ent_coef0.01,
vf_coef0.5, max_grad_norm0.5, clip_range0.2, normalize_advantageTrue,
target_klNone, clip_range_vfNone and net_arch[256,256]. Preserve the installed
MlpPolicy defaults, optimizer and initialization through frozen source hashes.

Each arm requests200000 training steps and must collect200704 (196 rollouts,
1960 PPO epoch updates). Environment seed stride1000 with offsets[0] gives
training initial seed42. Preserve the old factory's seeded inner bootstrap
before outer wrappers and the existing DummyVecEnv/PPO reset sequence.

Internal validation starts with factory seed70042 (offset70000), then continues
the existing unseeded reset/auto-reset RNG stream. These are not explicitly
reseeded consecutive or paired validation episodes. Keep50 deterministic
episodes every10000 training steps:20 calls and1000 completed validation episodes
per arm. Checkpoint interval stays25000 (eight scheduled checkpoints). Save
callback-best for historical logging only; score only the final root model.
No validation-based tuning, early stopping, best-checkpoint selection or budget
extension is allowed.

Each development evaluation uses500 deterministic episodes with outer resets
at40042+i for i=0..499, after the original inner factory bootstrap at40042.
Keep the evaluator's pre-step minimum TTC, one prediction and step, sequential
reward accumulation, returned min_ttc with pre-step fallback, unsafe count at
TTC<=2.0, collision-first outcomes and travel time=length/5. No shield, intent
model or risk-fusion wrapper; interventions must remain zero.

Total batch:400000 requested/401408 collected training steps,2000 internal
validation episodes and1000 development episodes. Training episode count is not
fixed separately from its step budget. No extra smoke rollouts, selected-case
replays, alternative actions or holdout evaluations are included in this budget.
New clock-specific unit checks use fabricated non-driving fixtures, not consumed
diagnostic cases. The existing full test suite remains unmodified.

Fixed stage order: zero train, clock train, zero evaluate, clock evaluate.
Both training stages must complete before either development evaluation. Run
fresh full Ruff and pytest before each stage. Do not modify the second arm from
any first-arm result, and do not rerun a failed/partial stage automatically.

## Outcomes and frozen decision rules

Preserve raw CSVs, eight-column metric schema, original units and positional
pairing under the contiguous development reset protocol. Validate all500 rows,
exclusive success/collision/incomplete categories, length/travel consistency,
metadata, model/configuration/lineage hashes and zero interventions. Preserve the
existing finite-minimum-TTC averaging convention; report excluded nonfinite
counts explicitly. A wholly unavailable TTC mean cannot pass a TTC gate.
Do not substitute pooled per-step TTC, another population or a new estimator.

Apply all16 historical gates to each arm without modification:

| Comparison | Success count | Collision count | Incomplete count | Mean minimum TTC | Paired success required |
| --- | --- | --- | --- | --- | --- |
| Original V3 / absolute | >=315 | <=174 | <=10 | >=0.6003656548142169 | yes, versus original |
| Corrected control | >=304 | <=195 | no additional gate | >=0.6065401801078688 | yes, versus corrected control |
| Predictive V1 | >=364 | <=101 | no additional gate | >=0.6728939419963959 | yes, versus V1 |
| Geometry V2 | >=394 | <=98 | <=8 | no additional gate | no additional significance gate |

For paired success, rescues are reference non-success/candidate success;
regressions are reference success/candidate non-success. With n=rescues+regressions:

`p = min(1, 2 * sum(comb(n,k), k=0..min(rescues,regressions)) / 2**n)`;
if n=0, p=1. Favorable means rescues>regressions and exact two-sided p<0.05.

The additional prospective clock-benefit rule requires all five conditions:

1. Clock passes all16 historical gates.
2. Clock-versus-zero paired success is favorable under the exact rule above.
3. Clock collisions<=zero collisions.
4. Clock incomplete<=zero incomplete.
5. Clock mean minimum TTC>=zero mean minimum TTC.

These conditions supplement the historical gates, not replace them. Equal
incomplete counts establish non-regression only, not fewer timeout failures.
Do not use lower stop duration or higher reward to bypass a failed outcome gate.
Report paired comparisons with Geometry, synchronized retry01, historical sector
and historical padding descriptively; add no unregistered significance filters.
No additional statistical or success-rate target is introduced.

Both arms failing means rejection under these rules. Zero passing while clock
fails does not establish clock benefit. Clock passing historical gates without
beating zero leaves feature benefit unestablished. Record mixed or unsuccessful
results and every failed condition without changing the design.

The development seeds are already consumed. No independent generalization,
multiple-search correction, realistic-sensor performance or real-world safety
claim is supported. A complete pass only permits a separately preregistered
training-seed replication and verified-unused evaluation set; this batch neither
selects nor inspects new holdout outcomes. Geometry V2 remains the user's current
best development candidate, with its earlier TTC-gate failure preserved; original
V3 remains unchanged. No automatic policy promotion.

## Named implementation and artifact plan

Add new opt-in files only; do not edit historical observation/train/evaluate code:

- `scripts/clock_observation_v1.py`: strict zero/clock augmentation.
- `scripts/run_clock_comparison_v1.py`: release-gated four-stage adapter around
  the unchanged train/evaluate mains in dedicated processes.
- `scripts/audit_clock_comparison_v1.py`: independent, read-only result audit.
- Matching tests `tests/test_clock_observation_v1.py`,
  `tests/test_clock_comparison_v1.py`, `tests/test_clock_comparison_audit_v1.py`.

Use model stems from the arm table. For each stem, final model is
`models/<stem>.zip`, training summary `results/<stem>.training.json`, and records
`results/<stem>.train.run.json` and `results/<stem>.evaluate.run.json`.
Development CSV/summary stems are
`results/ppo_v3_clock_zero_v1_development_seed40042` and
`results/ppo_v3_clock_remaining_v1_development_seed40042`.
Aggregate audit is
`results/ppo_v3_clock_comparison_v1_development_seed40042.audit.json`.

Release/proof paths are `configs/v3_clock_comparison_v1.release.json` and
`results/v3_clock_comparison_v1.release_verification.json`.
Preparation evidence is `results/v3_clock_observation_v1.preparation.json`.
These names are reserved prospectively, not existing released artifacts.

Logs use `logs/<model-stem>/` and `logs/<model-stem>.<mode>.stdout.log` /
`.stderr.log`; the batch lock is `logs/v3_clock_comparison_v1.active.lock`.
Require exclusive creation, a verified absent-process/output preflight, one
research process at a time, and exact predecessor hashes. Preserve partial
outputs and failed locks for diagnosis; never automatically delete evidence or
resume a checkpoint. Only the owning cleanly completed invocation may remove
its live lock. Stamp models and summaries with their clock arm, observation
protocol, duration, source/config/runtime and initial-tensor provenance.

## Required tests and release blockers

1. Pure boundary fixtures: t=0,15,30,overshoot and adjacent float values; dtype,
   shape, bounds, finite time, duration30, zero-arm constant, invalid parent and
   source-observation rejection. No artificial150-step deadline.
2. Fabricated delegated trajectories: preserve first115 bytes, source buffer,
   info/reward/flags, action identity and call counts. No extra sensing/RNG calls.
   Check terminal success/collision with time remaining, pure truncation clock0,
   and subsequent reset clock1. Both arms enforce the same parent validation.
3. Synthetic vector/collector fixtures: saved terminal observation carries its
   terminal clock, auto-reset observation carries1, and pure truncation keeps
   the existing gamma*V(terminal_observation) bootstrap. Do not bootstrap using
   the reset observation or rewrite terminated/truncated flags.
4. Non-driving initialization check: construct the two116-input PPO policies
   with the frozen seeds/settings and require identical192516-parameter tensors.
   Record the measured hash before release and require each actual fresh model
   to match it before learn. Check arm/protocol tags and reject wrong checkpoints.
5. Mocked four-stage execution: verify exact settings/reset contract/order,
   callback schedule, predecessor/source/dependency hashes, exclusive outputs,
   refusal of resume/overrides, failure preservation, cleanup and restoration of
   scoped in-memory adapters. No training or driving episodes for these tests.
6. Synthetic audit cases: exact16 gates and five clock-benefit conditions,
   p=1 for no discordance, p=.05 boundary, missing/nonfinite TTC, schema/types,
   positional pairing, collisions versus timeouts and partial/mismatched records.

Before implementation changes, verify the recoverable baseline retained below;
retain any newly affected material in a new snapshot if it is not recoverable.
Before a run, independently review implementation, test it in the actual frozen
RL environment, and issue a separate hash-bound release/proof that covers this
document, all execution/test sources, exact arguments, full runtime/dependency
inventory, old comparator/configuration bytes and immutable completed evidence.
Record proof/test counts and measured initialization hash, not placeholders.
No command can bypass a missing or failed release. This document alone is not
permission to start a stage. Any scientific amendment requires another version
before collecting results, preserving this record and explaining the change.

## Retained evidence and baseline pins

Current source baseline is local commit
`df8abd0425934d0812b1b9d583678e2f20d97d41`, with the existing pending README and
append-only MILESTONES changes left intact and unstaged. No Git push or large
checkpoint/log/data commit is allowed.

The retained snapshot is
`logs/v3_sector_timeout_diagnostic_v1_snapshot_20260912T0949Z/`:
archive SHA `68b9c311587cf9fe801d0a879f23f6251be209e6946a72037012385543173406`,
manifest SHA `1f5f638d020b464c88cee753546fffe83a8e7d7a0e155e29b9d63e86a26b29f2`.
Its235 members preserve the pre-diagnostic working bytes and three final models.
It is not an installed-runtime, full Git/index, every-ignored-file or off-device
backup. Verify existing archive/member bytes; never recreate or overwrite it.
Recovery, if separately necessary, targets a new empty directory, not live files.

Immutable diagnostic release/proof and completed analysis bind the baseline:

- `configs/v3_sector_timeout_diagnostic_v1.release.json`:
  `af45b9e044dae3e418df7c04704810ca667673cd53ca0ebac53ecfea699ebe67`.
- `results/v3_sector_timeout_diagnostic_v1.release_verification.json`:
  `ab7ac9af93a0f2755b0cdef0658ba06a6c4654059b55224af4cf15b60f22cda9`.
- `results/v3_sector_timeout_diagnostic_v1.analysis.json`:
  `ac193ba5925672de86149547865e977d1d94d74824c0d45107c50be3f2d03c32`.
- `safeintent_rl/sensors/synchronized_predictive.py`:
  `758d665d2c635d88693b1c71845b547d735580221830392c3f328724f68eb200`.
- `safeintent_rl/sensors/predictive.py`:
  `2d3880359d26a6408cef8e5d7ae7dfc30ab2a9c4813a6888593935258fc86e2f`.
- `safeintent_rl/envs/intersection.py`:
  `eca1191c1f676600ea68998642f8c417f654b14e810c948cdbd614842e03d52c`.
- `scripts/train_ppo.py`:
  `6dd20e7059bb0fd33f9516d3bb942003e8b2e0c386386252b9cb4e3c6d3dccc8`.
- `scripts/evaluate_policy.py`:
  `922b9f4a722b3c3a79a94e36753d4adb45c24a6d1faabb14c0c15604696aa55a`.

The new release must retain the full old246-source/input and227-dependency pins,
explicitly inventory new files and later retained evidence, and preserve the
documented runtime-origin checks. The old diagnostic is already complete; its
consumed111 episodes and old92-case diagnostic must not be rerun. Append every
implementation, test failure, release, result and decision to MILESTONES without
editing prior research history.
