"""Independently audit the frozen geometry development CSVs, without simulation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

STEMS = {
    "original": "ppo_reward_v3_cpa_baseline_holdout_seed40042",
    "control": "ppo_v3_predictive_control_development_seed40042",
    "v1": "ppo_v3_predictive_safety_v1_development_seed40042",
    "geometry": "ppo_v3_predictive_geometry_v2_development_seed40042",
}
REFERENCE_HASHES = {
    "original": ("aab91174c49090dedb8702651c913f0913f89b50d3a321befa97399f84a47fb4",
                 "51fe8e4aa86f62b2bbb5b66a7e8979655305e3a03ad571ca631d40828c7556a1"),
    "control": ("33ce4dc8b0e0a914033095a163cb26197ea31cf15788f9c0684e4e7818d711a3",
                "f4a3a623c75c6f643d14ac499d9dd5ce5f9b444c67ca25ba0388234d75929000"),
    "v1": ("c5a50b1147fafa4aa61c55f9c8784455b4ce77b57ac9617dfd7466a48daae535",
           "f9521709fa74cd3a369809f9040d216384cc2ad0cb72f965b42d32acce1134eb"),
}
FROZEN = {
    "configs/intersection_v3_predictive_geometry_v2.yaml":
        "a9629f60c2261325c5cdae996573a716698b7bc65168d1cea33c04bb530c93e7",
    "models/ppo_v3_predictive_geometry_v2_seed42.zip":
        "f3bfeaa9826f989ab1934b71a46a92a97cef159c8a800c56b23b4530c17636d2",
    "results/ppo_v3_predictive_geometry_v2_seed42.training.json":
        "6ab05310433ca6186874861f66bc2bc35e54a7474e7bbaa572b14d6220ebb861",
    "scripts/train_ppo.py":
        "6dd20e7059bb0fd33f9516d3bb942003e8b2e0c386386252b9cb4e3c6d3dccc8",
    "safeintent_rl/sensors/predictive.py":
        "2d3880359d26a6408cef8e5d7ae7dfc30ab2a9c4813a6888593935258fc86e2f",
    "scripts/evaluate_policy.py":
        "922b9f4a722b3c3a79a94e36753d4adb45c24a6d1faabb14c0c15604696aa55a",
    "safeintent_rl/envs/intersection.py":
        "eca1191c1f676600ea68998642f8c417f654b14e810c948cdbd614842e03d52c",
    "safeintent_rl/evaluation/metrics.py":
        "5a647b7a5616b5cc5f9a5743feb13dda386bab2e387a82d6c5af249c5342ed6b",
    "safeintent_rl/safety/ttc.py":
        "75565975fa3beb8eabf6f6af75cf26eedbdad1805e7ae568630cb0575c0ae2c0",
}
COLUMNS = ["reward", "length", "success", "collision", "travel_time", "min_ttc",
           "unsafe_ttc_events", "safety_interventions"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path, count: int = 500) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == COLUMNS, "CSV columns differ")
        rows = list(reader)
    require(len(rows) == count, "Episode count differs")
    parsed = []
    for row in rows:
        require(set(row) == set(COLUMNS) and None not in row.values(), "Malformed row")
        item = {}
        for key, value in row.items():
            if key in ("success", "collision"):
                require(value in ("True", "False"), "Invalid boolean")
                item[key] = value == "True"
            else:
                require(bool(value), "Missing numeric value")
                item[key] = float(value)
                require(not math.isnan(item[key]), "NaN in episode")
                if key != "min_ttc":
                    require(math.isfinite(item[key]), "Nonfinite episode metric")
        require(not (item["success"] and item["collision"]), "Overlapping outcomes")
        length = item["length"]
        require(length.is_integer() and 1 <= length <= 151, "Invalid episode length")
        require(math.isclose(item["travel_time"], length / 5, abs_tol=1e-12),
                "Travel time mismatch")
        require(item["min_ttc"] >= 0, "Negative TTC")
        unsafe = item["unsafe_ttc_events"]
        require(unsafe.is_integer() and 0 <= unsafe <= length, "Invalid unsafe count")
        require(item["safety_interventions"] == 0, "Unexpected intervention")
        parsed.append(item)
    return parsed


def metrics(rows: list[dict]) -> tuple[dict, dict]:
    finite = [r["min_ttc"] for r in rows if math.isfinite(r["min_ttc"])]
    require(bool(finite), "Mean TTC unavailable: no finite observations")
    mapping = {"mean_reward": "reward", "mean_length": "length",
               "success_rate": "success", "collision_rate": "collision",
               "mean_travel_time": "travel_time",
               "mean_unsafe_ttc_events": "unsafe_ttc_events",
               "mean_safety_interventions": "safety_interventions"}
    result = {key: statistics.fmean(r[field] for r in rows)
              for key, field in mapping.items()}
    result.update(episodes=float(len(rows)), mean_min_ttc=statistics.fmean(finite))
    counts = {key: sum(r[key] for r in rows) for key in ("success", "collision")}
    counts["incomplete"] = len(rows) - counts["success"] - counts["collision"]
    counts["excluded_nonfinite_ttc"] = len(rows) - len(finite)
    return result, counts


def paired(reference: list[dict], candidate: list[dict]) -> dict:
    require(len(reference) == len(candidate), "Pair length differs")
    rescues = sum(not r["success"] and c["success"]
                  for r, c in zip(reference, candidate))
    regressions = sum(r["success"] and not c["success"]
                      for r, c in zip(reference, candidate))
    n = rescues + regressions
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(rescues, regressions) + 1))
            / 2**n) if n else 1.0
    return {"rescues": rescues, "regressions": regressions, "p_exact_two_sided": p,
            "favorable": rescues > regressions and p < 0.05}


def gates(counts: dict, ttc: float, comparisons: dict) -> dict:
    return {
        "absolute_success": counts["success"] >= 315,
        "absolute_collision": counts["collision"] <= 174,
        "incomplete": counts["incomplete"] <= 10,
        "original_ttc": ttc >= 0.6003656548142169,
        "original_paired_success": comparisons["original"]["favorable"],
        "control_success": counts["success"] >= 304,
        "control_collision": counts["collision"] <= 195,
        "control_ttc": ttc >= 0.6065401801078688,
        "control_paired_success": comparisons["control"]["favorable"],
        "v1_success": counts["success"] >= 364,
        "v1_collision": counts["collision"] <= 101,
        "v1_ttc": ttc >= 0.6728939419963959,
        "v1_paired_success": comparisons["v1"]["favorable"],
    }


def audit(root: Path) -> dict:
    fingerprints = dict(FROZEN)
    for name, hashes in REFERENCE_HASHES.items():
        for suffix, digest in zip((".csv", ".summary.json"), hashes):
            fingerprints[f"results/{STEMS[name]}{suffix}"] = digest
    for path, digest in fingerprints.items():
        require(sha(root / path) == digest, f"Frozen hash mismatch: {path}")
    rows, summaries, evidence = {}, {}, {}
    for name, stem in STEMS.items():
        csv_path = root / "results" / f"{stem}.csv"
        summary_path = csv_path.with_suffix(".summary.json")
        rows[name] = read_rows(csv_path)
        summaries[name] = json.loads(summary_path.read_text(encoding="utf-8"))
        computed, counts = metrics(rows[name])
        summary = summaries[name]
        for key, value in computed.items():
            require(math.isclose(value, summary[key], rel_tol=1e-12, abs_tol=1e-12),
                    f"Summary mismatch: {name} {key}")
        require(summary["first_seed"] == 40042 and summary["last_seed"] == 40541,
                "Seed metadata mismatch")
        require(summary["unsafe_ttc_threshold"] == 2.0 and not summary["safety_shield"],
                "Metric or shield metadata mismatch")
        evidence[name] = {"metrics": computed, "counts": counts,
                          "csv_sha256": sha(csv_path), "summary_sha256": sha(summary_path)}
    expected = dict(summaries["v1"])
    expected.update(evidence["geometry"]["metrics"])
    for key, path in (("model", "models/ppo_v3_predictive_geometry_v2_seed42.zip"),
                      ("config", "configs/intersection_v3_predictive_geometry_v2.yaml")):
        expected[f"{key}_path"] = path
        expected[f"{key}_sha256"] = FROZEN[path]
    actual = dict(summaries["geometry"])
    for obj in (expected, actual):
        for key in ("model_path", "config_path", "reference_csv_path"):
            obj[key] = obj[key].replace("\\", "/")
    for key in evidence["geometry"]["metrics"]:
        actual[key] = expected[key]  # Already independently reconciled above.
    require(actual == expected, "Evaluation metadata differs from frozen v1 protocol")
    comparisons = {name: paired(rows[name], rows["geometry"]) for name in REFERENCE_HASHES}
    passed = gates(evidence["geometry"]["counts"],
                   evidence["geometry"]["metrics"]["mean_min_ttc"], comparisons)
    for path, digest in fingerprints.items():
        require(sha(root / path) == digest, f"File changed during audit: {path}")
    return {"status": "verified", "first_seed": 40042, "last_seed": 40541,
            "pairing": "positional under recorded contiguous reset protocol; no row seed IDs",
            "evidence": evidence, "paired_success": comparisons, "gates": passed,
            "all_retention_gates_pass": all(passed.values()),
            "decision": "eligible_for_further_validation" if all(passed.values())
                        else "not_retained_under_frozen_gates",
            "frozen_sha256": fingerprints, "audit_source_sha256": sha(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing audit output")
    report = audit(args.root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("evidence", "paired_success", "gates", "decision")}, indent=2))


if __name__ == "__main__":
    main()
