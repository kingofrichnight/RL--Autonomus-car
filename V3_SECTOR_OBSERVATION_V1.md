# V3-SectorFeatures v1: engineering prototype

Status: **implemented for engineering checks, not released for training**.
The section 81 synchronized PPO comparison must finish and be audited first.
Original V3 and all historical configurations, checkpoints and results remain
unchanged. This is not an accepted successor or a measured safety improvement.

## Research question and scope

Does an explicit angular-sector representation help PPO use the traffic state
it already receives? This is a **structured-feature augmentation** experiment,
not evidence that additional physical sensors improve safety.

CARLA supports RGB/depth cameras, LiDAR/semantic LiDAR, radar, GNSS, IMU and
collision/lane-invasion events. HighwayEnv's built-in `LidarObservation` instead
provides an idealized 2-D range/relative-velocity observation. Its native actor
coverage differs from the current sorted 14-actor input. Directly adding it
would mix representation changes with visibility changes. Accordingly this
prototype does not instantiate native LiDAR or query the road for extra actors.

References inspected for this design:

- [HighwayEnv observations](https://highway-env.farama.org/observations/#lidar)
- [CARLA sensor reference](https://carla.readthedocs.io/en/latest/ref_sensors/)

The existing radar-like fusion branch also derives features from simulated
state; this new module does not claim independent sensor measurements. True
noisy-sensor fusion would require removing clean-state bypasses from both PPO
inputs and the predictor, and is a separate later experiment. Semantic LiDAR
provides simulator object labels, not a learned classifier. Collision/lane
events are outcome signals, not advance knowledge of an accident.

## Frozen engineering observation contract

`scripts/sector_observation_v1.py` defines `SectorPredictiveObservation`, wrapping
`SynchronizedPredictiveObservation` directly. It preserves its entire 115-value
float32 input exactly and appends 48 values. The resulting input has **163
values**, so historical policies must not be loaded into it. Protocol name:
`predictive_post_spawn_sector_v1`.

The prototype deliberately lives outside `safeintent_rl`: section 81 records
every Python source in that package. Adding a new package file during that run
would change its source manifest even if the new file were never imported.
No historical factory, runner, configuration, or package source is modified.
This module has no training/evaluation CLI and does not register a new default.

Read only the already supplied synchronized observation. The first 105 entries
are 15 rows of `[presence, x, y, vx, vy, cos_h, sin_h]`. Row zero supplies only
ego heading to the projection; its absolute x/y/velocity are not subtracted a
second time. Rows 1--14 already contain relative position and relative velocity.
The original target scalar and nine forecasts remain unchanged.

For each present neighbor i:

```text
p_i = 200 * [x_i, y_i]                  # decoded, possibly clipped metres
v_i =  80 * [vx_i, vy_i]                # decoded relative m/s
theta_e = atan2(sin_h_ego, cos_h_ego)
r_i = ||p_i||_2
c_i = -(p_i . v_i) / r_i               # positive = approaching
Delta = 2*pi / 16
k_i = floor(((atan2(p_iy, p_ix) - theta_e + Delta/2) mod 2*pi) / Delta)
```

There are 16 half-open sectors, each pi/8 (22.5 degrees) wide. Sector zero is
centered on ego forward; angles increase in the simulator's x/y convention.
For each sector keep the actor with smallest center distance; equal distances
retain the lowest native row index. Flatten sector-major in index order:

```text
[present, r_i / (200*sqrt(2)), clip(c_i / (80*sqrt(2)), -1, 1)]
```

The range scale bounds the diagonal of the existing normalized position square;
it is not a new physical detection radius. No additional range or actor filter
is applied. Empty sectors emit `[0, 1, 0]`. A present actor with coincident center
uses sector zero, range zero and closing speed zero (bearing/radial speed are
undefined). The presence mask distinguishes that case from an empty sector.

Important limits:

- Ranges are actor-center distances, **not** body clearance or ray intersections.
- Empty means no retained actor in that sector, **not** verified free space.
- This does not recover clipped coordinates, add rear visibility or remove the
  14-actor cap. It adds no information beyond the existing 115 inputs.
- Native input clipping may distort a decoded bearing, distance or range-rate.
- The original kinematics and forecasts remain available to PPO; this is not a
  sensor-only, partial-observation or end-to-end perception benchmark.
- No labels, action overrides, extra physics steps, forecasts, observation
  refreshes, random draws, reward changes or cost coefficients are introduced.
- The actions remain SLOWER/IDLE/FASTER. Lane changes, pedestrians, rerouting and
  CARLA are not added by this prototype.

The wrapper checks the parent's frozen contract and exact position/velocity
normalization before augmenting, including after native reset recreates its
observation object. Invalid shape, nonfinite values, invalid presence, missing
ego, invalid ego heading and out-of-bound present coordinates are rejected.

## Training boundary and future comparison

Do not launch an additional long run now. First finish section 81 training,
run the required fresh test gate, finish its fixed evaluation and audit it.
Then separately record the exact new command, model/output names, hashes,
acceptance rules and resource budget before any sector PPO training. Never
change or reuse the in-progress synchronized command or its output names.

The intended comparison keeps section 81 traffic, dynamics, reward, actions,
PPO settings, seed, 200K requested budget and consumed development seeds fixed.
Only the observation augmentation changes. There is no automatic architecture
conversion or warm start from an existing checkpoint.

Same hidden widths do not imply equal capacity: adding 48 inputs to separate
256-wide policy and value towers adds 24,576 first-layer weights. Report this
confound. A zero-padded 163-input control could separate capacity from feature
content in a later preregistered comparison; it is not secretly added here.

Any 360-degree/expanded-actor proposal needs a coverage-matched control. Any
claim of improved success must include collisions, incomplete episodes, TTC,
paired outcomes and the consumed-development-set/single-training-seed limits.
No numerical improvement is promised. All failures and results must be appended
to `MILESTONES.md` and large checkpoints must remain outside Git.
