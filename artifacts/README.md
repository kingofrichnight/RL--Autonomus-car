# Models, datasets and external artifacts

`manifest.json` lists the recovered local models and intent dataset with SHA-256
checksums. The organized local checkout contains these files under `models/` and `data/`.
They are deliberately excluded from Git. Download the selected bundles from the
repository's GitHub Releases and verify them against `release-manifest.json`.
A GitHub clone cannot reproduce checkpoint-based results until these exact artifacts
are restored. The smoke checkpoint and four sector-history/repeat checkpoints not
referenced by the original GitHub research were excluded from publication.

## Restore

1. Download the release assets listed in `release-manifest.json` and verify their SHA-256.
2. Extract `core-models-and-dataset.zip` into the repository root. City bundles
   likewise use paths relative to the repository root.
3. Verify with `python tools/check_repository.py --artifacts`.
4. Use the configuration and wrapper settings recorded for that checkpoint.

Training a new model is possible using the research workflows, but does not recreate
the original checkpoint or its published metrics byte for byte.

## Still required

- Publication of the prepared GitHub Release bundles, including core artifacts,
  selected v13/v14 city traces, paired PPO/QR-DQN checkpoints, and the vendored
  QR-DQN package with its distribution metadata.
- Historical predecessor snapshots required by checksum-locked city release manifests.
  Superseded experiments remain local per the owner's curation request; these releases
  do not claim to support a complete historical rerun from GitHub alone.
- Original environment lock/snapshot for exact historical verification. The fresh
  install's resolved versions do not establish historical reproducibility.
- An owner-selected license before adding a license grant. None is inferred.
- A separately validated portable city release; imported historical launchers retain
  absolute paths and original checksums.

The local `city_share_tools` directory contains installed packages, not project source.
Deployment access screenshots and handoff material remain local. The local root web
app identifies itself as a fitness dashboard, so it was not merged into driving code.
