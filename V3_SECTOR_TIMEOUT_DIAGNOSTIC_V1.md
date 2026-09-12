# Sector batch v1: proposed matched timeout diagnosis

Status: prospective diagnostic design only. No replay implementation or command
is released by this document. Preserve the completed four-stage batch and the
user's saved Geometry V2 development candidate. See MILESTONES92.

## Question

Why do the two new policies miss the30-second deadline on some consumed
development cases? Distinguish persistent stopping, late departure, slow travel,
prediction/observation limitations and genuinely difficult traffic before
changing the RL objective. Stopping alone does not prove avoidable hesitation.

Both policies fail the existing incomplete limits: padding24/500 and sector23/500,
versus Geometry V2's8/500. On the selected union below, Geometry has9successes,
23collisions and5incomplete outcomes. Thus forcing movement could trade some
timeouts for collisions. This selected subset is not an unbiased benchmark or
proof of any causal mechanism.

## Fixed selection and bounded scope

Select every row that is incomplete in either completed padding or sector CSV;
derive its seed as40042 plus its zero-based row index under the recorded reset
protocol. Deduplicate and sort. Their overlap is10 cases, leaving37 distinct seeds:

```text
40084 40106 40125 40143 40160 40168 40173 40175 40181 40193
40203 40207 40222 40223 40243 40256 40271 40276 40286 40295
40296 40339 40341 40349 40390 40396 40403 40429 40440 40449
40476 40495 40506 40514 40530 40539 40541
```

Proposed execution budget: exactly111 non-interventional diagnostic episodes,
the same37 seeds in Geometry V2, padding and sector, in that fixed arm order and
ascending seed order within each arm. No training, search, repeated attempts,
new holdout scoring, original interrupted-policy scoring or old92-case replay.
These replays only diagnose already scored cases, not add samples to the500-row
performance estimates. Raw diagnostic traces stay outside Git; small summaries
and provenance records may be committed.

Selection inputs must be pinned before implementation release:

- Padding CSV SHA-256
  `b35aa0e97aee4417870a40433e8f0d7a36235771f7abf2ea4981c3a087c54e96`.
- Sector CSV SHA-256
  `68e7259e0c08519978b9a07bc48b812878b4f1b6481afd12d20e30abe5199cbb`.
- Geometry CSV SHA-256
  `e145084f4ccce3d419c34886bd9434c0b5a817fb6fa3c1ceab616a89e1929030`.

Use the existing final checkpoints, not callback-best or retrained models:

- Geometry: `models/ppo_v3_predictive_geometry_v2_seed42.zip`, SHA
  `f3bfeaa9826f989ab1934b71a46a92a97cef159c8a800c56b23b4530c17636d2`.
- Padding: `models/ppo_v3_sector_padding_v1_seed42.zip`, SHA
  `82da273ff74b39776b7486a6523c96102428b5e84d72499e130d9c7c22fddbd6`.
- Sector: `models/ppo_v3_sector_features_v1_seed42.zip`, SHA
  `0695d09d098ecdae60ae63d33b919c691af8844f782af5a66649c33e497bfaef`.

All use `configs/intersection_v3_predictive_geometry_v2.yaml`, SHA
`a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7`.
Preserve each model's exact historical observation wrapper:115 Geometry inputs,
163 synchronized-plus-padding inputs,163 synchronized-plus-sector inputs.
Do not feed an old policy a newly interpreted observation.

## Implementation and release conditions

1. Save and verify a recoverable source/configuration/model/result snapshot
   before any major change. Record archive/location and file hashes, retain
   model binaries locally, preserve pending README/MILESTONES edits and all prior
   source files. A hash without retained recoverable content is insufficient.
2. Implement a separately named opt-in diagnostic adapter; do not edit historical
   evaluation/training/observation code. Hash-bind this design, the selected
   CSVs, final models, source inventory, config, protocol and fixed seed list.
3. Log actual policy inputs, executed action and categorical probabilities, ego speed/target, elapsed
   time, remaining route progress, the already available per-action forecast
   outputs, actor identities/selection and observation clipping/alignment.
   Include the terminal post-step state. Diagnostic-only data must never be
   added to the policy input or reward. Do not call observe/forecast again or
   add RNG draws merely to obtain a trace; read the existing returned values.
4. Require deterministic action delegation, CPU and frozen dependency versions,
   unchanged environment/dynamics/reward/30-second cutoff and unsafe-TTC2.0.
   No shield, forced acceleration, alternative-action execution, extra RNG draws
   or newly noisy/independent sensor assumptions. A positive predicted margin
   alone is not proof of a safe gap or a certified counterfactual.
5. Unit-test selection, wrapper shapes/protocols, read-only tracing, RNG/dynamics
   invariance, action parity, exclusive outputs and failure preservation. Get
   independent code/protocol review and fresh Ruff/full pytest before release.
6. For every replay require reproduction of its saved outcome and episode
   length; reconcile all eight historical metric columns with declared numeric
   tolerances fixed before execution. Stop and preserve evidence on mismatch;
   no silent retry, overwrite or hand-picked replacement case.
7. One process at a time, unique output/log paths and explicit refusal to
   overwrite. Record every completion, mismatch or unsuccessful diagnosis in
   append-only MILESTONES. Run no command until implementation, output names,
   tolerances, complete fingerprints and the fresh test gate are recorded.

## Decision boundary after diagnosis

Report uncertainty and negative findings. Use the traces to propose one isolated
RL change only if they support it; do not silently add deadline observations,
terminal semantics, memory, reward coefficients or constrained-cost budgets.
Those require a separately named, prospectively fixed experiment and matched
controls. Retain all existing acceptance gates and reserve untouched evaluation
for a later locked policy with multi-training-seed validation. This document
neither promotes the sector policy nor promises a target success rate.
