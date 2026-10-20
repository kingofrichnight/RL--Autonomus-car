# SafeIntent-RL — Autonomous driving research

Intent-aware and safety-constrained reinforcement learning for simulated autonomous driving.
The main research package uses Gymnasium, HighwayEnv, PPO, a GRU intent classifier,
and TTC/predictive safety observations. City and multi-map experiments are a separate track.

## Start here

- [Install and run](docs/guides/quickstart.md)
- [Training, evaluation and intent workflows](docs/guides/research-workflows.md)
- [Research documents and project map](docs/README.md)
- [City / multi-map experiment catalog](experiments/city/README.md)
- [Artifacts and remaining requirements](artifacts/README.md)
- [Repository audit and preservation](docs/audit/README.md)
- [Contribution and validation](CONTRIBUTING.md)

## Folder layout

```text
safeintent_rl/       Reusable environments, agents, intent, sensors and safety code
scripts/             Existing training, evaluation and diagnostic entry points
configs/             Environment configurations and historical release manifests
tests/               Core research test suite
experiments/city/    Versioned city and multi-map source and small evidence files
experiments/runtime/Archived supporting policy/runtime source
docs/guides/         Installation and research workflows
docs/research/       Experiment design and comparison documents
docs/history/        Chronological research record
docs/audit/          Original file hashes and full local-import inventory
artifacts/           Model/dataset manifest and recovery instructions
models/, data/      Local binary artifacts (excluded from Git)
results/             Original versioned evaluation evidence
.github/workflows/  Automated core tests
```

Geometry V2 remains the recorded development candidate; see the research workflow
for its measured results and limitations. Reorganization does not change the algorithms,
retrain policies, validate historical results, or establish real-world driving safety.
Older root documentation names redirect to the organized documents.
