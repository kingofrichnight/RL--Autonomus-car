# Curated city experiments

Only the latest completed experiment and its immediate completed predecessor are
included. Older versions and the unfinished broad-curve v15 remain local and are
accounted for in the [import inventory](../../docs/audit/local-import-inventory.csv).

| Folder | Reason retained | Recorded evidence |
|---|---|---|
| [multi_map_timeout40_v13](multi_map_timeout40_v13/) | Complete 30-case city/highway/jam baseline and immediate predecessor used by v14 | FINAL_VALIDATION.json records pass, 30 browser cases and zero browser errors |
| [multi_map_jam_cross_v14](multi_map_jam_cross_v14/) | Latest completed cross-traffic extension | run_v1/complete.json records complete, native exit 0, ten episodes; preflight records pass |

These are recorded historical checks, not a fresh execution on this machine.
The import does not imply the policies always succeed: completion, collisions and
timeouts must be read from the individual result files.

Runtime inference modules are in `../runtime/`. Historical launchers retain absolute
paths and checksum-locked release inventories. They cannot be represented as portable
fresh-checkout commands. The release bundles preserve the saved outputs for inspection;
portable reruns require a separately validated successor release and restoration of
historical dependencies. Never bypass checksum guards or silently reseal old evidence.

Use the [core quick start](../../docs/guides/quickstart.md) for a standard runnable workflow.
