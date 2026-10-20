# Repository organization audit

Audited GitHub source: `kingofrichnight/RL--Autonomus-car`, commit recorded in
[original-files.json](original-files.json). The original 265 tracked files were
read and fingerprinted before edits.

## Changes

- Research guides, comparison notes and milestones now live in `docs/guides/`,
  `docs/research/` and `docs/history/`. Old root names provide compatibility links.
- City and multi-map versions are grouped under `experiments/city/`; referenced
  runtime Python sources are under `experiments/runtime/`.
- Core algorithms, scripts, configurations, tests and original results retain their
  original paths and bytes. Of the original files, 253 remain byte-identical;
  11 Markdown files and `.gitignore` changed.
- Added installation, contribution, artifact recovery and CI instructions.
- Recovered 25 local model checkpoints and one trajectory dataset. Publication
  selects 20 models referenced by the original research and one dataset; the smoke
  model and four unreferenced sector-history/repeat models remain local.
  Selected hashes are in the artifact manifest; binaries stay out of Git.

## Inventory and review scope

[local-import-inventory.csv](local-import-inventory.csv) accounts for 2,301 local
files considered. After owner-requested curation, only 148 useful city/runtime files
remain selected for publication; superseded snapshots, unfinished v15, and unused
archived runtime tooling remain local. Each selected file includes its original
SHA-256. Excluded files have an explicit reason. Large traces, media,
installed packages and deployment/access material remain in their original local
locations; they were not deleted.

[inspection.json](inspection.json) records whole-file format checks and the source
files containing machine-specific paths. All imported Python files were parsed,
JSON and JSONL were decoded, CSV files were read, and original compressed traces
were decompressed successfully. This is a repository organization and structural
audit, not a claim of line-by-line algorithm correctness or independent validation
of every recorded experiment.

## Validation on 2026-10-20

- `python tools/check_repository.py --artifacts`: passed.
- `python -m pip check`: passed.
- Five dependency-independent core test modules: **41 passed**.
- Full pytest run: **blocked during collection (28 errors)** by Windows Application
  Control rejecting NumPy's `_pcg64` DLL. No policy setting was changed.
- GitHub Actions runs the core suite on Python 3.11 and 3.12. Its outcome must be
  checked separately; local structural checks do not imply CI success.
- Archived city launchers are not portable yet and were not executed. They require
  additional artifact/runtime restoration and separately named successor releases.

## Preserve historical verification

Historical release manifests fingerprint old documentation and exact source inventories.
The organized branch intentionally has a different layout and must not be described
as a successful verification of those releases. Their manifests were not rewritten.
To recover the original code/config/documentation snapshot:

```sh
git worktree add ../safeintent-original 87dce7b
```

Use the full commit recorded in `original-files.json` if the short ID is ambiguous.
Restore the corresponding artifact and dependency snapshots separately. An old release
may refer to an earlier snapshot than this GitHub commit; inspect its manifest before
claiming a match.

## Outstanding owner inputs

The owner selected GitHub Releases for artifact storage. Only the assets allowlisted
in [publication-selection.json](publication-selection.json) are intended for upload;
other prepared archives remain local. Choose a license if redistribution is intended.
No license was invented.

See [artifact recovery](../../artifacts/README.md) for details.
