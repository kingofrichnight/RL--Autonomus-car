# Contributing

Install Python 3.11/3.12 and `python -m pip install -e ".[dev]"`.
Run `python tools/check_repository.py` and `python -m pytest` from the repository root.
The automated suite covers the core package. Archived city snapshots are tested
separately because their generic module names and external dependencies conflict.

Preserve old configuration, checkpoint, seed, output and source identities. New methods
need separately named configurations and result files. Never overwrite released evidence,
alter historical success rates, or replace a missing checkpoint with a different model.

Keep binaries and raw traces in artifact storage; record their hashes and download
location. Review imports for private access material before publishing. Add focused tests
when changing simulation behavior, safety metrics, observations or learning protocols.
