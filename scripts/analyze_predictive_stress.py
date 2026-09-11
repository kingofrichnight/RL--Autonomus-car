"""Verify completed stress artifacts and summarize paired outcomes without reruns."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safeintent_rl.config import load_config
from safeintent_rl.intent.inference import file_sha256
from scripts.evaluate_predictive_stress import scenario_config, summarize, validate_protocol


def paired_success(control: list[dict], candidate: list[dict]) -> dict:
    if [r["seed"] for r in control] != [r["seed"] for r in candidate]:
        raise ValueError("Cannot pair different episode seed orders")
    rescues = sum(not a["success"] and b["success"] for a, b in zip(control, candidate))
    regressions = sum(a["success"] and not b["success"] for a, b in zip(control, candidate))
    discordant = rescues + regressions
    p = (min(1.0, 2 * sum(math.comb(discordant, k)
                         for k in range(min(rescues, regressions) + 1)) / 2**discordant)
         if discordant else 1.0)
    return {"rescues": rescues, "regressions": regressions,
            "exact_two_sided_mcnemar_p": p,
            "interpretation": "exploratory, not multiplicity-adjusted or a promotion gate"}


def verify_rows(rows: list[dict], protocol: dict) -> None:
    expected = list(range(protocol["first_seed"],
                          protocol["first_seed"] + protocol["episodes_per_scenario"]))
    if [r["seed"] for r in rows] != expected:
        raise ValueError("Wrong episode count or seed order")
    for row in rows:
        outcomes = [row[k] for k in ("success", "collision", "incomplete")]
        if any(type(value) is not bool for value in outcomes) or sum(outcomes) != 1:
            raise ValueError("Outcomes must be exclusive booleans")
        if (type(row["steps"]) is not int or not 1 <= row["steps"] <= 151
                or not math.isclose(row["travel_time"], row["steps"] / 5)):
            raise ValueError("Invalid time/step ledger")
        for key in ("stopped_time", "zero_target_time"):
            if not 0 <= row[key] <= row["travel_time"]:
                raise ValueError(f"Invalid {key}")
        if not math.isfinite(row["reward"]):
            raise ValueError("Nonfinite reward")
        if row["min_ttc"] is not None and not math.isfinite(row["min_ttc"]):
            raise ValueError("Nonfinite TTC must use null")


def analyze(root: Path, candidate_dir: Path | None = None) -> dict:
    directories = {"control": root / "control", "candidate": candidate_dir or root / "candidate"}
    reports = {arm: json.loads((directories[arm] / "report.json").read_text(encoding="utf-8"))
               for arm in ("control", "candidate")}
    protocol = reports["control"]["protocol"]
    validate_protocol(protocol)
    if any(r["status"] != "complete" for r in reports.values()):
        raise ValueError("Both arm reports must be complete")
    if reports["candidate"]["protocol"] != protocol:
        raise ValueError("Arm protocols differ")
    if file_sha256("configs/v3_predictive_stress_v1.json") != reports["control"]["protocol_sha256"]:
        raise ValueError("Protocol file hash mismatch")
    for arm, report in reports.items():
        if report["arm"] != arm:
            raise ValueError("Wrong arm report")
        for kind in ("config", "model"):
            if file_sha256(protocol["arms"][arm][kind]) != protocol["arms"][arm][kind + "_sha256"]:
                raise ValueError("Frozen artifact changed")
        for path, expected in report["source_sha256"].items():
            if file_sha256(path) != expected:
                raise ValueError("Experiment source changed during execution")
    for key in ("protocol_sha256", "source_sha256", "versions"):
        if reports["control"][key] != reports["candidate"][key]:
            raise ValueError(f"Arm {key} mismatch")
    result = {"status": "verified", "protocol_sha256": reports["control"]["protocol_sha256"],
              "episodes": 0, "scenarios": {}, "arm_report_sha256": {
                  arm: file_sha256(directories[arm] / "report.json") for arm in reports},
              "candidate_directory": str(directories["candidate"]),
              "recovery": reports["candidate"].get("recovery")}
    for scenario in protocol["scenarios"]:
        name = scenario["name"]
        records = {}
        outcome = {}
        for arm in reports:
            report = reports[arm]["scenarios"][name]
            episode_path = directories[arm] / f"{name}.episodes.jsonl"
            if file_sha256(episode_path) != report["episodes_sha256"]:
                raise ValueError("Episode-file fingerprint mismatch")
            lines = episode_path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines]
            verify_rows(rows, protocol)
            recomputed = summarize(rows)
            if recomputed != report["summary"]:
                raise ValueError("Episode metrics disagree with recorded summary")
            summary_path = directories[arm] / f"{name}.summary.json"
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if saved != report or report["status"] != "complete":
                raise ValueError("Scenario report mismatch")
            config = report["effective_config"]
            expected = scenario_config(load_config(protocol["arms"][arm]["config"]), scenario)
            if config != expected:
                raise ValueError("Effective config changed beyond the frozen scenario")
            if (config["duration"] != 30
                    or config["spawn_probability"] != scenario["spawn_probability"]):
                raise ValueError("Scenario settings mismatch")
            if report["scenario"] != scenario:
                raise ValueError("Scenario probabilities mismatch")
            records[arm] = rows
            outcome[arm] = {**recomputed, "incomplete_seeds": [
                r["seed"] for r in rows if r["incomplete"]]}
            result["episodes"] += len(rows)
        outcome["paired_success"] = paired_success(records["control"], records["candidate"])
        result["scenarios"][name] = outcome
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/v3_predictive_stress_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve existing analysis")
    result = analyze(Path(args.root), args.candidate_dir)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
