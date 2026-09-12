# V3 sector-feature comparison v1: preregistered development batch

Status: design frozen before either arm is trained. Launch requires tested code,
independent review and a fresh full-suite gate. See MILESTONES sections 86--87.

## Question and reason for this next experiment

Can a sector-based representation help PPO learn safer successful intersection
crossing from the same traffic observations? The synchronized retry failed five
retention gates (376 successes,117 collisions,7 incomplete). It is not adopted.
Geometry V2 remains a stronger historical development comparator, but it also
failed its TTC gate. Original V3 remains the accepted policy.

The sector prototype uses synchronized geometry so its native coordinates and
forecast features describe the same post-spawn state. This is an experimental
measurement contract, not selection of the synchronized trained policy. Both
arms train from scratch. Do not load its checkpoint, revert timing for one arm,
or add an unplanned second base. The two-arm comparison isolates explicit
feature augmentation within that common observation pipeline.

## Exactly two arms

| Arm | Observation | Model stem | Protocol |
| --- | --- | --- | --- |
| padding | 115 synchronized values followed by 48 exact zeros | ppo_v3_sector_padding_v1_seed42 | predictive_post_spawn_padding_v1 |
| sector | Same 115 values followed by the 48 existing sector features | ppo_v3_sector_features_v1_seed42 | predictive_post_spawn_sector_v1 |

Both use float32 inputs of shape (163,), the same three speed actions, and
separate 256/256 actor/value networks. Initialization tensor hashes and parameter
counts must match across these two freshly seeded models before optimization.
The new dimensions add 24,576 first-layer weights relative to a 115-input model.
Padding controls input shape, parameter count and initialization, not equal
effective capacity: zero-connected weights cannot contribute or receive data
gradients. Sector features add a nonlinear basis derived from existing inputs.
Do not call this a comparison of independent physical sensors or pure capacity.

The sector formula and validation are exactly those already tested in
V3_SECTOR_OBSERVATION_V1.md and scripts/sector_observation_v1.py (SHA-256
f0a78c5befe3c8e0edcc101ee49e7b2bb18c80be3afc6840e359653f99762919).
No range, neighbor, angular-bin or normalization coefficients are changed.
The original115 values remain byte-identical in both arms. The padding wrapper
adds no sensing calls, forecasts, dynamics steps or random draws.

In brief: decode already-normalized relative p=200[x,y], v=80[vx,vy]; assign
the closest actor center per one of16 ego-heading-centered sectors; append
[presence, norm(p)/(200 sqrt(2)), clip(-dot(p,v)/(norm(p)80 sqrt(2)),-1,1)].
Empty sectors are [0,1,0]; the prototype's documented coincident-center rule
remains unchanged. These are center distances, not body clearance, ray hits or
verified free space. No extra actors, rear visibility, measurement noise,
occlusion, labels, action override, pedestrians or rerouting are added.

## Frozen scientific protocol and budget

- Config configs/intersection_v3_predictive_geometry_v2.yaml, SHA-256
  a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7.
- PPO MlpPolicy, fresh seed42, CPU, Torch threads8/8. No resume or warm start.
- Each arm: 200000 requested steps, expected200704 collected, n_steps1024,
  batch64, n_envs1, learning_rate.0003, net_arch[256,256], gamma.99,
  gae_lambda.95, ent_coef.01, n_epochs10, clip_range.2, vf_coef.5,
  max_grad_norm.5, normalize_advantageTrue, target_klNone, clip_range_vfNone.
- Env seed stride1000, initial seed42, validation offset70000,50 validation
  episodes every10000 steps, checkpoints every25000. Final root model only,
  regardless of callback-best reward. No validation-based early stop or tuning.
- Predictive horizon3, margin.5, uncertainty_growth.25, clearance_scale10,
  speed_scale9, max_neighbors14. Collision-first reward and coefficients
  unchanged, native30-second episodes, policy5Hz, simulation15Hz.
- Python3.12.9; stable-baselines3 2.9.0, Torch2.13.0, NumPy2.5.2,
  Gymnasium1.3.0, HighwayEnv1.12.1. No dependency or device change.
- Each evaluation:500 deterministic episodes, seeds40042--40541 inclusive,
  unsafe TTC threshold2.0; no shield, intent model or risk-fusion wrapper.
- Total batch:400000 requested/401408 expected collected training steps and
  1000 development evaluation episodes, plus unchanged internal validation.
  One research process at a time; no external/paid compute or automatic retry.

Fixed order (fresh Ruff and full pytest before **each** command):

```powershell
python -u -m scripts.run_sector_comparison_v1 padding train --refuse-overwrite
python -u -m scripts.run_sector_comparison_v1 sector train --refuse-overwrite
python -u -m scripts.run_sector_comparison_v1 padding evaluate --refuse-overwrite
python -u -m scripts.run_sector_comparison_v1 sector evaluate --refuse-overwrite
```

Run both training arms before either development evaluation. Do not change the
second arm, budget, seeds, gates or checkpoint choice after any first-arm result.
An execution/integrity failure stops the batch for diagnosis; preserve partial
artifacts and append the failure before any separately authorized recovery.
Do not overwrite an attempt or interpret partial training as a final result.

Each model lives under models/<stem>.zip, summaries under results/<stem>.training.json,
and train/evaluate manifests under results/<stem>.<mode>.run.json. Evaluations:

- results/ppo_v3_sector_padding_v1_development_seed40042.csv and .summary.json.
- results/ppo_v3_sector_features_v1_development_seed40042.csv and .summary.json.

Training TensorBoard directories and unique console stdout/stderr logs live under
logs/. Preserve existing files. Source fingerprints include this preregistration,
the new runner, frozen prototype and prior synchronized lineage/result evidence.
No historical source or configuration is replaced. Check source and final-model
hashes before/after each run. Commit only reviewed code, small results and docs;
models, large data and logs stay local. No Git push.

## Frozen analysis and decision rules

Independently validate all500 raw rows per arm, exclusive success/collision/
incomplete outcomes, episode/travel-time consistency, summary metadata, hashes,
zero interventions and the existing finite-TTC convention. Report missing/
nonfinite TTC counts, not silent deletion. Positional seed pairing rests on the
recorded contiguous reset protocol because the CSV schema has no seed column.

For each arm report every unchanged section79.5/81 gate: at least315successes,
at most174collisions and10incomplete, TTC>=.6003656548142169 and favorable paired
success vs original; at least304successes, at most195collisions,
TTC>=.6065401801078688 and favorable paired success vs correctedcontrol;
at least364successes, at most101collisions, TTC>=.6728939419963959 and favorable
paired success vs V1; and at least394successes, at most98collisions and8incomplete
against Geometry V2. No added significance requirement versus Geometry V2.

Primary new sector-benefit comparison is sector versus padding. To support a
development feature-benefit claim, require all of:

1. Sector passes **all** historical gates above.
2. Sector paired success versus padding is favorable: rescues>regressions and
   exact two-sided McNemar p<.05, with the existing binomial formula.
3. Sector collisions<=padding collisions, incomplete<=padding incomplete,
   and mean minimum TTC>=padding mean minimum TTC.

These new comparisons are additional, not substitutions for failed historical
gates. Also report paired outcomes versus all historical arms and synchronized
retry for context. Do not apply an unrecorded significance filter, choose a
different TTC statistic, or relax criteria after results. No multiple-search
adjustment is claimed; this is an exploratory development batch.

If sector fails, record it without changing the criterion. If padding passes but
sector does not, padding may be eligible for separately designed validation, but
there is no sensor-feature benefit claim. If sector passes historical gates but
does not beat padding, its feature benefit is unestablished. Even a complete
pass only warrants preregistered multi-training-seed replication and untouched
evaluation. This consumed development set does not establish generalization,
real-world safety or guaranteed success. No automatic production-policy promotion.
