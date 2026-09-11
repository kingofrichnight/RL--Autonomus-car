# V3 predictive safety: completed stress study

Completed 2026-09-11. All **1,000 scored episodes** passed artifact verification.
These are simulation stress tests, not evidence of real-world driving safety.

## Protocol

Two frozen checkpoints: the corrected V3 control (106 observations) and
V3-PredictiveSafety v1 (115 observations). Each ran 100 episodes in each of five
scenarios, using seeds **80042–80141** and the unchanged 30-second deadline.
No retraining, safety shield, reward changes, or acceptance-gate changes were made.
Exact settings and model/config fingerprints are in
`configs/v3_predictive_stress_v1.json` and the arm reports.

Light/dense change ongoing vehicle spawning (0.2/0.9), not the initial population
of 10 vehicles. Aggressive/cautious scenarios use an 80% probability of the named
driver profile; mixed uses the original profile mixture. The latter three retain
spawn probability 0.6. Driver profiles affect speed, acceleration, and following
gaps; they do not add a new explicit yielding model.

## Outcomes

Each cell is **success / collision / incomplete (%)**. With 100 episodes per cell,
these values also equal episode counts.

| Situation | V3 control | Predictive candidate |
|---|---:|---:|
| Light traffic | 57 / 43 / 0 | 67 / 27 / 6 |
| Dense traffic | 59 / 41 / 0 | 69 / 22 / 9 |
| Aggressive drivers | 50 / 50 / 0 | 59 / 37 / 4 |
| Cautious drivers | 70 / 30 / 0 | 78 / 20 / 2 |
| Mixed drivers | 58 / 42 / 0 | 74 / 23 / 3 |

The candidate has higher observed success and fewer collisions in all five
scenarios, but increased unfinished episodes. Aggressive traffic remains its
highest-collision condition (37%); dense traffic has the most incompletion (9%).

| Situation | Control mean stopped time (s) | Candidate mean stopped time (s) |
|---|---:|---:|
| Light | 0.004 | 1.756 |
| Dense | 0.364 | 2.476 |
| Aggressive | 0.160 | 1.914 |
| Cautious | 0.144 | 0.908 |
| Mixed | 0.262 | 1.236 |

Stopped time counts post-step samples with absolute ego speed below 0.5 m/s.
More stopping is consistent with conservative behavior, but these aggregate
metrics alone do not establish the cause of individual unfinished episodes.

Exact paired-success McNemar p-values are 0.006348 (light), 0.041389 (dense),
0.149613 (aggressive), 0.096252 (cautious), and 0.000402 (mixed). These are
exploratory, unadjusted for multiple comparisons, and not promotion criteria.

The descriptive equal-weight totals are 58.8% success / 41.2% collision / 0%
incomplete for control versus 69.4% / 25.8% / 4.8% for the candidate. The same
100 seeds recur across scenarios: pooled episodes are not 500 independent seeds
per policy. One training seed/checkpoint per arm and 100 episodes per scenario
limit generalization. Matched seeds do not guarantee identical future traffic
after policies diverge. No pedestrian, obstacle, weather, or rerouting stress
coverage is claimed by this study.

## Decision

**Do not promote this candidate.** Its earlier 500-episode development evaluation
failed the frozen 2% incompletion ceiling. This exploratory study does not replace
that evaluation or change the gate. The accepted original V3 remains unchanged;
the stress-table control is the corrected matched control, not that original
checkpoint. The next proposed investigation is episode-level analysis of dense
traffic waiting and aggressive-traffic collisions before designing a separately
named experiment. No further training was launched as part of this completion.

## Recovery and verification

Control completed 500 episodes normally. The original candidate process stopped
after 194 saved episodes without stderr; the cause is unknown. Its original files
remain under `results/v3_predictive_stress_v1/candidate/`.

Recovery copied the 100 light and 94 dense episode prefix without alteration,
reproduced dense seed 80135 successfully, and evaluated only the missing 306
episodes. That engineering replay is not scored. The authoritative candidate
directory is **`results/v3_predictive_stress_v1/candidate_recovery_01/`**; do not
count the original prefix again. Recovery provenance is retained in its report.

Ruff and 229 tests passed before recovery (two existing Box warnings). The final
analyzer verified all scenario counts, ordered seeds, exclusive outcomes, timing,
recomputed summaries, effective configs, model/config/source hashes, and matching
arm provenance. Original prefix hashes and copied prefix bytes were also checked.

Machine-readable final artifact:
`results/v3_predictive_stress_v1/verified_summary.json`

SHA-256:
`61ebf55be7ad25deeb5af7776fecffe19e6647401db7b19669970c526330ba97`

To independently regenerate verification, choose a **new**, unused output path:

```powershell
python -m scripts.analyze_predictive_stress --root results/v3_predictive_stress_v1 --candidate-dir results/v3_predictive_stress_v1/candidate_recovery_01 --output results/v3_predictive_stress_v1/verified_summary_recheck.json
```

The verifier intentionally refuses to overwrite existing results. Model
checkpoints should remain outside ordinary Git commits.
