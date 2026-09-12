"""Reconcile the completed synchronized retry with its frozen development gates.

Read-only audit of source artifacts. Writes only a new, explicitly named report;
no simulator, model selection, seed changes, or historical-source edits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_geometry_development import (
    STEMS,
    gates,
    metrics,
    paired,
    read_rows,
    require,
    sha,
)

STEM = "ppo_v3_predictive_sync_v1_retry01_seed42"
CANDIDATE = "ppo_v3_predictive_sync_v1_retry01_development_seed40042"
PROTOCOL = "predictive_post_spawn_sync_v1"
PINNED = {
    f"results/{STEM}.train.run.json":
        "de38b7ab02ceb7378ec07de76b3b2c95b8a25306a8ba437e42a06c9f58b97dba",
    f"results/{STEM}.evaluate.run.json":
        "da3e4e4b98d1f4d754d521d687e300dfa3fee0afb7ed243020d5adef209b5d2d",
    f"results/{STEM}.training.json":
        "e858bc356e2f430e409ae96d53c62f164204376ed0799fa75f43fe3524d0edfa",
    f"models/{STEM}.zip":
        "8a363a7b5e7caef621d6b616890175c10d457531b4d4a213d05fc2011244f395",
    f"results/{CANDIDATE}.csv":
        "b2865f8d45db051a450744eb94f95850fd742089f6f12445820d6067a3514dfc",
    f"results/{CANDIDATE}.summary.json":
        "763cfef6e8c4a9364f69896c5ede4a6ece7e3f6dcf0e54ea33b0df140da2338d",
}


def check_hashes(root: Path, hashes: dict[str, str]) -> None:
    for path, digest in hashes.items():
        require(sha(root / path) == digest, f"Frozen hash mismatch: {path}")


def retention_gates(counts: dict, ttc: float, comparisons: dict) -> dict:
    result = gates(counts, ttc, comparisons)
    result.update(geometry_success=counts["success"] >= 394,
                  geometry_collision=counts["collision"] <= 98,
                  geometry_incomplete=counts["incomplete"] <= 8)
    return result


def audit(root: Path) -> dict:
    check_hashes(root, PINNED)

    def read_json(path: str) -> dict:
        return json.loads((root / path).read_text(encoding="utf-8"))

    trained = read_json(f"results/{STEM}.train.run.json")
    evaluated = read_json(f"results/{STEM}.evaluate.run.json")
    sources = trained["source_sha256"]
    require(len(sources) == 45 and sources == evaluated["source_sha256"],
            "Training/evaluation source snapshot differs")
    check_hashes(root, sources)
    package = {p.relative_to(root).as_posix()
               for p in (root / "safeintent_rl").rglob("*.py")}
    require(package == {p for p in sources if p.startswith("safeintent_rl/")},
            "Package source inventory changed")
    for mode, record, suffix in (("train", trained, "training"),
                                  ("evaluate", evaluated, "summary")):
        summary_stem = STEM if mode == "train" else CANDIDATE
        require(record["status"] == "complete" and record["mode"] == mode
                and record["observation_protocol"] == PROTOCOL,
                "Run did not complete the recorded protocol")
        require(record["model_sha256"] == PINNED[f"models/{STEM}.zip"]
                and record["summary_sha256"] ==
                PINNED[f"results/{summary_stem}.{suffix}.json"], "Run artifact mismatch")
    require(evaluated["csv_sha256"] == PINNED[f"results/{CANDIDATE}.csv"],
            "Evaluation CSV fingerprint differs")
    for key in ("versions", "python", "torch_threads"):
        require(trained[key] == evaluated[key], f"Runtime changed: {key}")

    rows, summaries, evidence = {}, {}, {}
    for name, stem in {**STEMS, "synchronized_retry": CANDIDATE}.items():
        csv_path = root / "results" / f"{stem}.csv"
        summary_path = csv_path.with_suffix(".summary.json")
        rows[name] = read_rows(csv_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        computed, counts = metrics(rows[name])
        for key, value in computed.items():
            require(math.isclose(value, summary[key], rel_tol=1e-12, abs_tol=1e-12),
                    f"Summary differs: {name} {key}")
        require(summary["first_seed"] == 40042 and summary["last_seed"] == 40541,
                "Seed metadata mismatch")
        require(summary["unsafe_ttc_threshold"] == 2.0 and not summary["safety_shield"],
                "Metric or shield mismatch")
        summaries[name] = summary
        evidence[name] = {"metrics": computed, "counts": counts,
                          "csv_sha256": sha(csv_path), "summary_sha256": sha(summary_path)}

    candidate = evidence["synchronized_retry"]
    expected = dict(summaries["geometry"])
    expected.update(candidate["metrics"], model_path=f"models/{STEM}.zip",
                    model_sha256=PINNED[f"models/{STEM}.zip"], observation_protocol=PROTOCOL)
    actual = dict(summaries["synchronized_retry"])
    for obj in (expected, actual):
        for key in ("model_path", "config_path", "reference_csv_path"):
            obj[key] = obj[key].replace("\\", "/")
    actual.update(candidate["metrics"])  # Independently reconciled above.
    require(actual == expected, "Evaluation settings differ from fixed Geometry protocol")

    comparisons = {name: paired(rows[name], rows["synchronized_retry"]) for name in STEMS}
    passed = retention_gates(candidate["counts"], candidate["metrics"]["mean_min_ttc"],
                             comparisons)
    fingerprints = {**PINNED, **sources}
    check_hashes(root, fingerprints)
    return {
        "status": "verified", "first_seed": 40042, "last_seed": 40541,
        "pairing": "positional under recorded contiguous reset protocol; no row seed IDs",
        "evidence": evidence, "paired_success": comparisons, "gates": passed,
        "failed_gates": [key for key, value in passed.items() if not value],
        "all_retention_gates_pass": all(passed.values()),
        "decision": "eligible_for_further_validation" if all(passed.values())
                    else "not_retained_under_frozen_gates",
        "limitations": ["Consumed development seeds; one training seed, not generalization",
                        "Synchronized retry is fresh execution recovery, not replication",
                        "No added significance gate against Geometry V2",
                        "TTC uses finite observations only; exclusions are reported"],
        "frozen_sha256": fingerprints, "audit_source_sha256": sha(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing audit report")
    report = audit(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("evidence", "paired_success", "failed_gates", "decision")}, indent=2))


if __name__ == "__main__":
    main()
