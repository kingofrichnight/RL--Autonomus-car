"""Read-only numerical audit of the completed, fixed two-arm sector batch.

Uses only the standard library; never imports a simulator or loads a policy.
Checkpoint execution is separately verified with the frozen experiment runner.
Only a new explicitly named JSON report may be written. Raw evidence is immutable.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from scripts.audit_geometry_development import STEMS, metrics, paired, read_rows, require, sha
from scripts.audit_synchronized_retry import CANDIDATE, check_hashes, retention_gates

ARMS = {
    "padding": "ppo_v3_sector_padding_v1_seed42",
    "sector": "ppo_v3_sector_features_v1_seed42",
}
EVALUATIONS = {
    "padding": "ppo_v3_sector_padding_v1_development_seed40042",
    "sector": "ppo_v3_sector_features_v1_development_seed40042",
}
PROTOCOLS = {arm: f"predictive_post_spawn_{arm}_v1" for arm in ARMS}
INITIAL = {
    "sha256": "ca35bda9cfe2f2afce2eb0cf71d2abd74fb6808d2f935b7e20a5b31c1d054c19",
    "parameter_count": 216580,
}
ORDER = (("padding", "train"), ("sector", "train"),
         ("padding", "evaluate"), ("sector", "evaluate"))
RECORD_HASHES = (
    "6077b4f778fa817a07d6a5933e219ca7c22b1489b1cc6dcd9407f67a60c0b19b",
    "0829a00096ed593a71f5aabab5d3e3c79d4d96406e1cb32b924948e09740469b",
    "cf7eee76af7a437fba2e1f19e6c939d59dd3b840a7d79758d0413e840e481863",
    "144b8df605a1edb7c52a5d2c3ba47d7d2c76965268448a9aa6b481b5916674c1",
)
AUDIT_DEPENDENCIES = {
    "scripts/audit_geometry_development.py":
        "80d98ac72d453b172a729f6eac2b5eef9f962172bdb7d1d3a66af75833c274d0",
    "scripts/audit_synchronized_retry.py":
        "0c7ebea4537425f81edf0a9abed0cd2a1cae13e825403a764c0292127bfc2fd7",
}


def feature_benefit_gates(sector_gates: dict, padding_evidence: dict,
                          sector_evidence: dict, comparison: dict) -> dict:
    """Additional preregistered conditions, never replacements for historical gates."""
    return {
        "all_historical_gates": all(sector_gates.values()),
        "paired_success_vs_padding": comparison["favorable"],
        "collision_vs_padding": (sector_evidence["counts"]["collision"]
                                 <= padding_evidence["counts"]["collision"]),
        "incomplete_vs_padding": (sector_evidence["counts"]["incomplete"]
                                  <= padding_evidence["counts"]["incomplete"]),
        "ttc_vs_padding": (sector_evidence["metrics"]["mean_min_ttc"]
                           >= padding_evidence["metrics"]["mean_min_ttc"]),
    }


def reconciled_evidence(root: Path, stems: dict) -> tuple[dict, dict, dict]:
    rows, summaries, evidence = {}, {}, {}
    for name, stem in stems.items():
        csv_path = root / "results" / f"{stem}.csv"
        summary_path = csv_path.with_suffix(".summary.json")
        rows[name] = read_rows(csv_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        computed, counts = metrics(rows[name])
        for key, value in computed.items():
            require(math.isclose(value, summary[key], rel_tol=1e-12, abs_tol=1e-12),
                    f"Summary differs: {name} {key}")
        require(summary["episodes"] == 500.0 and summary["first_seed"] == 40042
                and summary["last_seed"] == 40541,
                "Seed metadata mismatch")
        require(summary["unsafe_ttc_threshold"] == 2.0 and not summary["safety_shield"],
                "Metric or shield metadata mismatch")
        summaries[name] = summary
        evidence[name] = {"metrics": computed, "counts": counts,
                          "csv_sha256": sha(csv_path), "summary_sha256": sha(summary_path)}
    return rows, summaries, evidence


def checked_summary(candidate: dict, geometry: dict, arm: str, model_sha: str) -> None:
    """Only the declared arm/model/observation provenance may differ from Geometry."""
    require(arm in ARMS, "Unknown comparison arm")
    metric_keys = ("episodes", "mean_reward", "mean_length", "success_rate", "collision_rate",
                   "mean_travel_time", "mean_min_ttc", "mean_unsafe_ttc_events",
                   "mean_safety_interventions")
    expected = dict(geometry)
    expected.update({key: candidate[key] for key in metric_keys})
    expected.update(model_path=f"models/{ARMS[arm]}.zip", model_sha256=model_sha,
                    observation_protocol=PROTOCOLS[arm], sector_comparison_arm=arm,
                    initial_policy=INITIAL)
    actual = dict(candidate)
    for obj in (expected, actual):
        for key in ("model_path", "config_path", "reference_csv_path"):
            obj[key] = obj[key].replace("\\", "/")
    require(actual == expected, f"Evaluation settings differ: {arm}")


def completed_chain(root: Path) -> tuple[dict, dict]:
    """Pinned manifests bind all runtime/arguments and complete artifact/source lineage."""
    sources, dependencies, records = None, {}, {}
    fingerprints = dict(AUDIT_DEPENDENCIES)
    previous_finish = None
    for (arm, mode), expected_hash in zip(ORDER, RECORD_HASHES):
        record_path = f"results/{ARMS[arm]}.{mode}.run.json"
        check_hashes(root, {record_path: expected_hash})
        record = json.loads((root / record_path).read_text(encoding="utf-8"))
        require(record["status"] == "complete" and record["arm"] == arm
                and record["mode"] == mode and record["observation_protocol"] == PROTOCOLS[arm]
                and record["initial_policy"] == INITIAL, "Stage or initial provenance differs")
        if sources is None:
            sources = record["source_sha256"]
            require(len(sources) == 56, "Source inventory count differs")
            fingerprints.update(sources)
        require(record["source_sha256"] == sources, "Stage sources differ")
        require(record["predecessor_sha256"] == dependencies, "Predecessor chain differs")
        started = datetime.fromisoformat(record["started_utc"])
        finished = datetime.fromisoformat(record["finished_utc"])
        require(started <= finished and (previous_finish is None or previous_finish <= started),
                "Stage chronology differs")
        previous_finish = finished
        model_path = f"models/{ARMS[arm]}.zip"
        stem = ARMS[arm] if mode == "train" else EVALUATIONS[arm]
        suffix = "training" if mode == "train" else "summary"
        summary_path = f"results/{stem}.{suffix}.json"
        summary = json.loads((root / summary_path).read_text(encoding="utf-8"))
        require(summary["model_sha256"] == record["model_sha256"]
                and summary["initial_policy"] == INITIAL, "Summary model lineage differs")
        stage_hashes = {record_path: expected_hash, summary_path: record["summary_sha256"]}
        fingerprints[model_path] = record["model_sha256"]
        if mode == "train":
            stage_hashes[model_path] = record["model_sha256"]
        else:
            stage_hashes[f"results/{stem}.csv"] = record["csv_sha256"]
        dependencies.update(stage_hashes)
        fingerprints.update(stage_hashes)
        records[f"{arm}_{mode}"] = record
    package = {p.relative_to(root).as_posix()
               for p in (root / "safeintent_rl").rglob("*.py")}
    require(package == {p for p in sources if p.startswith("safeintent_rl/")},
            "Package source inventory changed")
    check_hashes(root, fingerprints)
    return records, fingerprints


def audit(root: Path) -> dict:
    records, fingerprints = completed_chain(root)
    stems = {**STEMS, "synchronized_retry": CANDIDATE, **EVALUATIONS}
    rows, summaries, evidence = reconciled_evidence(root, stems)
    for name, summary in summaries.items():
        for kind in ("model", "config"):
            path = summary[f"{kind}_path"].replace("\\", "/")
            digest = summary[f"{kind}_sha256"]
            require(path not in fingerprints or fingerprints[path] == digest,
                    f"Conflicting artifact identity: {name} {kind}")
            fingerprints[path] = digest
    check_hashes(root, fingerprints)
    comparisons, historical, decisions = {}, {}, {}
    for arm in ARMS:
        checked_summary(summaries[arm], summaries["geometry"], arm,
                        records[f"{arm}_evaluate"]["model_sha256"])
        comparisons[arm] = {name: paired(rows[name], rows[arm])
                            for name in (*STEMS, "synchronized_retry")}
        historical[arm] = retention_gates(evidence[arm]["counts"],
                                          evidence[arm]["metrics"]["mean_min_ttc"],
                                          comparisons[arm])
        decisions[arm] = ("eligible_for_separately_preregistered_validation"
                          if all(historical[arm].values())
                          else "not_retained_under_frozen_gates")
    versus_padding = paired(rows["padding"], rows["sector"])
    benefit = feature_benefit_gates(historical["sector"], evidence["padding"],
                                   evidence["sector"], versus_padding)
    check_hashes(root, fingerprints)
    return {
        "status": "verified", "first_seed": 40042, "last_seed": 40541,
        "rows_per_arm": 500, "reference_arms": list(STEMS) + ["synchronized_retry"],
        "pairing": "positional under recorded contiguous reset protocol; no row seed IDs",
        "evidence": evidence, "paired_success": comparisons,
        "historical_gates": historical,
        "failed_historical_gates": {arm: [k for k, v in values.items() if not v]
                                    for arm, values in historical.items()},
        "arm_decisions": decisions, "sector_vs_padding": versus_padding,
        "feature_benefit_gates": benefit,
        "failed_feature_benefit_gates": [k for k, v in benefit.items() if not v],
        "feature_benefit_established_under_frozen_gates": all(benefit.values()),
        "automatic_promotion": False,
        "limitations": [
            "Consumed development seeds and one training seed; not generalization",
            "No added significance gate against Geometry V2; no multiple-search adjustment",
            "Finite-TTC mean convention preserved and exclusions reported",
            "Zero padding controls nominal shape/parameters/initialization, not effective capacity",
            "Sector features reuse observed state; no independent physical sensor information",
            "No change to the saved Geometry V2 current-best-development designation",
        ],
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
    print(json.dumps({k: report[k] for k in ("evidence", "paired_success",
                     "failed_historical_gates", "sector_vs_padding",
                     "failed_feature_benefit_gates", "arm_decisions")}, indent=2))


if __name__ == "__main__":
    main()
