"""Offline descriptive audit of the completed, frozen 111-case timeout diagnostic.

Standard library only. No simulator, model, prediction, observation or experiment
execution. Historical source, checkpoint, CSV, release and trace bytes stay intact.
The only write is the explicitly requested, exclusively created analysis JSON.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from scripts import sector_timeout_protocol_v1 as protocol

RELEASE = "configs/v3_sector_timeout_diagnostic_v1.release.json"
RELEASE_SHA256 = "af45b9e044dae3e418df7c04704810ca667673cd53ca0ebac53ecfea699ebe67"
PROOF = "results/v3_sector_timeout_diagnostic_v1.release_verification.json"
PROOF_SHA256 = "ab7ac9af93a0f2755b0cdef0658ba06a6c4654059b55224af4cf15b60f22cda9"
SUMMARY = "results/v3_sector_timeout_diagnostic_v1.summary.json"
SUMMARY_SHA256 = "a443b95ac7beb0b70b6fd7e43f3730695377bbc12eac67c6a5e93a4e024c2f4e"
RAW_DIRECTORY = "logs/v3_sector_timeout_diagnostic_v1"
COMPLETE = f"{RAW_DIRECTORY}/complete.json"
COMPLETE_SHA256 = "06de7c97d66b54345c526e009781e1b3aba8ab08bee3cee26559f025a27b76c8"
OUTPUT = "results/v3_sector_timeout_diagnostic_v1.analysis.json"
ANALYZER = "scripts/analyze_sector_timeouts_v1.py"
TESTS = "tests/test_sector_timeout_analysis_v1.py"
ALLOWED_NEW_SOURCES = {ANALYZER, TESTS}
ARM_ORDER = ("geometry", "padding", "sector")
ACTIONS = ("SLOWER", "IDLE", "FASTER")
FEATURES = ("presence", "x", "y", "vx", "vy", "cos_h", "sin_h")
FORECAST_FEATURES = ("minimum_margin_gap", "first_margin_conflict", "travel_distance")
TARGETS = (0.0, 4.5, 9.0)
LIMITATIONS = [
    "These 37 outcome-selected, consumed seeds are not a new benchmark or 111 new samples.",
    "All summaries are descriptive, per episode then by arm/outcome; no pooled-step significance.",
    "Zero target is not zero physical speed; IDLE preserves target and need not stop the car.",
    "Positive forecast margin is not a safe gap, collision probability "
    "or certified counterfactual.",
    "First-margin-conflict value 1 cannot distinguish no conflict from conflict at the 3s horizon.",
    "Normalized forecast saturation obscures magnitudes; boundary values alone are not clipping.",
    "Full observation slots do not establish unseen hazards; "
    "actor IDs cannot be matched across arms.",
    "Subsequent actions/traffic can diverge from forecast assumptions; no alternative actions ran.",
    "Policy/action associations do not prove avoidable hesitation or a causal improvement.",
    "Progress is a clipped route fraction, not metres; "
    "sampled speed summaries are not physics integrals.",
    "All eight ledger metrics match pinned CSV rows. Six trace-recomputable metrics are checked "
    "independently; success and safety_interventions are not separately logged in raw frames.",
]
FORMULAS = {
    "interval": "Delta t_i = next_frame.time - decision_frame.time, including terminal next frame.",
    "target_seconds": "Sum Delta t_i grouped by next_frame.target_speed in {0,4.5,9}.",
    "zero_runs": "Maximal consecutive intervals with next_frame.target_speed == 0.",
    "relaunch": "At interval start when target changes from previous actual target 0 to positive; "
    "use initial decision target before the first interval.",
    "progress": "Signed next_frame.route_progress - decision_frame.route_progress; no clipping "
    "or imputation by this analysis.",
    "positive_target_progress_per_second": "Sum signed progress over positive-target intervals "
    "divided by their duration; null when duration is zero.",
    "speed": "Separate unweighted decision-state and post-transition-state sample summaries.",
    "entropy": "-sum(p_a * log(p_a)), natural logarithm, with 0*log(0)=0.",
    "clipping": "Count actual raw out-of-range masks separately for ego and neighbors by feature.",
    "aggregation": "Each episode contributes once to min/median/mean/max distributions; "
    "empty observed-outcome groups are explicitly retained.",
    "trace_reproduction": "Reward uses sequential +=, matching the evaluator. Trace independently "
    "checks reward,length,travel_time,min_ttc,unsafe_ttc_events,collision. "
    "Success and interventions are bound through the ledger/reference/source.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return protocol.sha(Path(path))


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def _file(root, relative):
    return protocol.source_path(root, relative)


def _pinned(root, relative, digest):
    path = _file(root, relative)
    require(sha(path) == digest, f"Frozen input hash mismatch: {relative}")
    return path


def _number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite/nonnumeric {label}")
    return float(value)


def _ttc(value):
    if value == "Infinity":
        return math.inf
    result = _number(value, "TTC")
    require(result >= 0, "Negative TTC")
    return result


def _integer(value, label):
    require(type(value) is int, f"Noninteger {label}")
    return value


def _bool(value, label):
    require(type(value) is bool, f"Nonboolean {label}")
    return value


def _same_number(left, right, label, *, abs_tol=1e-12):
    require(
        math.isclose(_number(left, label), _number(right, label), rel_tol=0, abs_tol=abs_tol),
        f"Numeric mismatch: {label}",
    )


def stats(values):
    values = list(values)
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in values),
        "Nonfinite descriptive sample",
    )
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _matrix(value, rows, columns, label, *, boolean=False):
    require(isinstance(value, list) and len(value) == rows, f"Invalid matrix rows: {label}")
    for row in value:
        require(isinstance(row, list) and len(row) == columns, f"Invalid matrix columns: {label}")
        for item in row:
            (_bool if boolean else _number)(item, label)
    return value


def _actors(event):
    ids = event["selected_birth_ids"]
    actors = event["selected_actors"]
    require(
        isinstance(ids, list)
        and len(ids) <= 14
        and all(type(i) is int and i >= 0 for i in ids)
        and len(set(ids)) == len(ids),
        "Invalid actor identity selection",
    )
    require(
        isinstance(actors, list) and [actor.get("birth_id") for actor in actors] == ids,
        "Actor/identity alignment differs",
    )
    for actor in actors:
        require(
            set(actor) == {"birth_id", "position", "velocity", "heading", "length", "width"},
            "Actor fields differ",
        )
        for name in ("position", "velocity"):
            require(isinstance(actor[name], list) and len(actor[name]) == 2, "Actor vector differs")
            for value in actor[name]:
                _number(value, name)
        for name in ("heading", "length", "width"):
            _number(actor[name], name)
    return ids


def _frame(frame, arm):
    require(
        set(frame)
        == {
            "arm",
            "policy_observation",
            "native_events",
            "forecast_events",
            "native_forecast_ids_match",
            "state",
            "call_order",
        },
        "Frame fields differ",
    )
    require(frame["arm"] == arm, "Frame arm differs")
    state = frame["state"]
    require(
        set(state)
        == {
            "time",
            "speed",
            "target_speed",
            "position",
            "heading",
            "crashed",
            "lane_index",
            "route_progress",
            "remaining_route_fraction",
        },
        "State fields differ",
    )
    for name in (
        "time",
        "speed",
        "target_speed",
        "heading",
        "route_progress",
        "remaining_route_fraction",
    ):
        _number(state[name], name)
    _bool(state["crashed"], "crashed")
    require(state["time"] >= 0 and state["target_speed"] in TARGETS, "Time/target differs")
    require(0 <= state["route_progress"] <= 1, "Route progress outside fraction bounds")
    _same_number(state["remaining_route_fraction"], 1 - state["route_progress"], "remaining route")
    require(
        isinstance(state["position"], list) and len(state["position"]) == 2,
        "Ego position shape differs",
    )
    for value in state["position"]:
        _number(value, "ego position")
    require(
        isinstance(state["lane_index"], list) and len(state["lane_index"]) == 3,
        "Lane index shape differs",
    )
    observation = frame["policy_observation"]
    require(
        isinstance(observation, list) and len(observation) == protocol.ARMS[arm]["shape"],
        "Policy observation shape differs",
    )
    for value in observation:
        _number(value, "policy observation")
    natives, forecasts = frame["native_events"], frame["forecast_events"]
    order = ["native", "forecast"] + ([] if arm == "geometry" else ["native"])
    require(
        frame["call_order"] == order
        and len(natives) == (1 if arm == "geometry" else 2)
        and len(forecasts) == 1,
        "Observation event sequence differs",
    )
    for native in natives:
        require(
            set(native)
            == {
                "kind",
                "time",
                "raw_rows",
                "clipped_mask",
                "normalization_ranges",
                "selected_birth_ids",
                "selected_actors",
                "normalized_rows",
                "boundary_mask",
            }
            and native["kind"] == "native",
            "Native event fields differ",
        )
        _same_number(native["time"], state["time"], "native time", abs_tol=0)
        ids = _actors(native)
        raw = _matrix(native["raw_rows"], len(ids) + 1, 7, "raw rows")
        clipped = _matrix(native["clipped_mask"], len(raw), 7, "clipped mask", boolean=True)
        normalized = _matrix(native["normalized_rows"], 15, 7, "normalized rows")
        boundary = _matrix(native["boundary_mask"], 15, 7, "boundary mask", boolean=True)
        ranges = {"x": [-200, 200], "y": [-200, 200], "vx": [-80, 80], "vy": [-80, 80]}
        require(native["normalization_ranges"] == ranges, "Native normalization ranges differ")
        for row_index, row in enumerate(raw):
            for column, feature in enumerate(FEATURES):
                expected = feature in ranges and not (
                    ranges[feature][0] <= row[column] <= ranges[feature][1]
                )
                require(
                    clipped[row_index][column] == expected, "Clipping mask differs from raw data"
                )
        require(
            boundary == [[abs(value) == 1 for value in row] for row in normalized],
            "Boundary mask differs",
        )
    forecast = forecasts[0]
    require(
        set(forecast) == {"kind", "time", "selected_birth_ids", "selected_actors", "values"}
        and forecast["kind"] == "forecast",
        "Forecast fields differ",
    )
    _same_number(forecast["time"], state["time"], "forecast time", abs_tol=0)
    _actors(forecast)
    matrix = _matrix(forecast["values"], 3, 3, "forecast")
    require(
        all(-1 <= row[0] <= 1 and 0 <= row[1] <= 1 and 0 <= row[2] <= 1 for row in matrix),
        "Normalized forecast outside bounds",
    )
    require(
        observation[:105] == [value for row in natives[-1]["normalized_rows"] for value in row]
        and observation[106:115] == [value for row in matrix for value in row],
        "Logged events differ from actual policy inputs",
    )
    _same_number(observation[105], state["target_speed"] / 9, "target observation", abs_tol=1e-7)
    aligned = natives[-1]["selected_birth_ids"] == forecast["selected_birth_ids"]
    require(
        type(frame["native_forecast_ids_match"]) is bool
        and frame["native_forecast_ids_match"] == aligned,
        "Identity alignment flag differs",
    )
    require(arm == "geometry" or aligned, "Synchronized identity mismatch")
    return state


def summarize_episode(decisions, transitions, terminal, metrics):
    """Validate one complete joined trace and return only descriptive measurements."""
    protocol.validate_episode(metrics)
    count = metrics["length"]
    require(
        len(decisions) == len(transitions) == count, "Decision/transition/metric length differs"
    )
    require(
        isinstance(terminal, dict)
        and set(terminal)
        == {"kind", "arm", "seed", "observation_frame_index", "terminated", "truncated", "frame"},
        "Terminal record fields differ",
    )
    arm, seed = terminal["arm"], _integer(terminal["seed"], "seed")
    require(arm in ARM_ORDER and terminal["kind"] == "terminal", "Terminal identity differs")
    require(
        _integer(terminal["observation_frame_index"], "terminal frame") == count,
        "Terminal frame index differs",
    )
    for name in ("terminated", "truncated"):
        _bool(terminal[name], name)
    require(terminal["terminated"] or terminal["truncated"], "Missing terminal flag")
    states = []
    probabilities = []
    for i, (decision, transition) in enumerate(zip(decisions, transitions)):
        require(
            set(decision)
            == {
                "kind",
                "arm",
                "seed",
                "decision_index",
                "observation_frame_index",
                "proposed_action",
                "action_probabilities",
                "pre_step_ttc",
                "frame",
            }
            and decision["kind"] == "decision",
            "Decision record fields differ",
        )
        require(
            set(transition)
            == {
                "kind",
                "arm",
                "seed",
                "decision_index",
                "observation_frame_index",
                "executed_action",
                "reward",
                "metric_ttc",
                "terminated",
                "truncated",
            }
            and transition["kind"] == "transition",
            "Transition record fields differ",
        )
        for record in (decision, transition):
            require(
                record["arm"] == arm
                and _integer(record["seed"], "seed") == seed
                and _integer(record["decision_index"], "decision index") == i,
                "Decision/transition identity/index differs",
            )
        require(
            _integer(decision["observation_frame_index"], "frame index") == i
            and _integer(transition["observation_frame_index"], "frame index") == i + 1,
            "Observation frame join differs",
        )
        action = _integer(decision["proposed_action"], "proposed action")
        require(
            0 <= action < 3
            and _integer(transition["executed_action"], "executed action") == action,
            "Action delegation differs",
        )
        probs = decision["action_probabilities"]
        require(isinstance(probs, list) and len(probs) == 3, "Probability shape differs")
        require(
            all(0 <= _number(p, "probability") <= 1 for p in probs)
            and math.isclose(sum(probs), 1, rel_tol=0, abs_tol=1e-6)
            and action == max(range(3), key=lambda j: probs[j]),
            "Probability/argmax differs",
        )
        probabilities.append(probs)
        _number(transition["reward"], "reward")
        _ttc(decision["pre_step_ttc"])
        _ttc(transition["metric_ttc"])
        done = _bool(transition["terminated"], "terminated") | _bool(
            transition["truncated"], "truncated"
        )
        require(done == (i == count - 1), "Premature/missing transition termination")
        states.append(_frame(decision["frame"], arm))
    states.append(_frame(terminal["frame"], arm))
    require(
        terminal["terminated"] == transitions[-1]["terminated"]
        and terminal["truncated"] == transitions[-1]["truncated"],
        "Terminal flags differ",
    )
    _same_number(states[0]["time"], 0, "reset time", abs_tol=0)
    elapsed = states[-1]["time"] - states[0]["time"]
    _same_number(elapsed, metrics["travel_time"], "elapsed/travel time")
    for i, (before, after) in enumerate(zip(states, states[1:])):
        require(after["time"] > before["time"], "Nonincreasing frame time")
        _same_number(after["time"] - before["time"], 0.2, "policy interval")
        _same_number(before["time"], i / 5, "decision time")
    reward_sum = 0.0
    for transition in transitions:
        reward_sum += float(transition["reward"])
    rebuilt = dict(
        metrics,
        reward=reward_sum,
        length=len(transitions),
        travel_time=len(transitions) / 5,
        min_ttc=min(_ttc(row["metric_ttc"]) for row in transitions),
        unsafe_ttc_events=sum(_ttc(row["metric_ttc"]) <= 2 for row in transitions),
        collision=states[-1]["crashed"],
    )
    protocol.compare_episode(rebuilt, metrics)
    target_seconds = {"0": 0.0, "4.5": 0.0, "9": 0.0}
    runs, relaunches = [], []
    active = None
    previous_target = states[0]["target_speed"]
    positive_seconds = positive_progress = 0.0
    for before, after in zip(states, states[1:]):
        target, duration = after["target_speed"], after["time"] - before["time"]
        target_seconds[f"{target:g}"] += duration
        delta = after["route_progress"] - before["route_progress"]
        if target == 0:
            if active is None:
                active = {
                    "start_time": before["time"],
                    "end_time": after["time"],
                    "duration": 0.0,
                    "start_progress": before["route_progress"],
                    "end_progress": after["route_progress"],
                }
            active.update(end_time=after["time"], end_progress=after["route_progress"])
            active["duration"] += duration
        else:
            positive_seconds += duration
            positive_progress += delta
            if active is not None:
                runs.append(active)
                active = None
            if previous_target == 0:
                relaunches.append(
                    {
                        "time": before["time"],
                        "progress": before["route_progress"],
                        "target_speed": target,
                    }
                )
        previous_target = target
    tail = active["duration"] if active is not None else 0.0
    if active is not None:
        runs.append(active)
    counts = Counter(ACTIONS[item["executed_action"]] for item in transitions)
    ordered_mismatch = same_set_reordered = native_only = forecast_only = 0
    native_counts, forecast_counts = [], []
    clipping = {role: {feature: 0 for feature in FEATURES} for role in ("ego", "neighbors")}
    forecasts = {action: {feature: [] for feature in FORECAST_FEATURES} for action in ACTIONS}
    selected_forecasts = {feature: [] for feature in FORECAST_FEATURES}
    for decision in decisions:
        frame = decision["frame"]
        native, forecast = frame["native_events"][-1], frame["forecast_events"][0]
        left, right = native["selected_birth_ids"], forecast["selected_birth_ids"]
        ordered_mismatch += left != right
        same_set_reordered += left != right and set(left) == set(right)
        native_only += len(set(left) - set(right))
        forecast_only += len(set(right) - set(left))
        native_counts.append(len(left))
        forecast_counts.append(len(right))
        for row_index, mask in enumerate(native["clipped_mask"]):
            for j, value in enumerate(mask):
                clipping["ego" if row_index == 0 else "neighbors"][FEATURES[j]] += value
        for a, action in enumerate(ACTIONS):
            for j, feature in enumerate(FORECAST_FEATURES):
                forecasts[action][feature].append(forecast["values"][a][j])
                if a == decision["proposed_action"]:
                    selected_forecasts[feature].append(forecast["values"][a][j])

    def forecast_summary(values):
        return {
            feature: {
                **stats(samples),
                "at_lower_bound": sum(
                    v == (-1 if feature == "minimum_margin_gap" else 0) for v in samples
                ),
                "at_upper_bound": sum(v == 1 for v in samples),
            }
            for feature, samples in values.items()
        }

    outcome = (
        "collision" if metrics["collision"] else "success" if metrics["success"] else "incomplete"
    )
    return {
        "arm": arm,
        "seed": seed,
        "outcome": outcome,
        "metrics": dict(metrics),
        "elapsed_seconds": elapsed,
        "target_seconds": target_seconds,
        "zero_target": {
            "runs": runs,
            "total_seconds": target_seconds["0"],
            "longest_seconds": max((run["duration"] for run in runs), default=0.0),
            "tail_seconds": tail,
            "relaunches": relaunches,
        },
        "speed": {
            "decision_samples": stats(state["speed"] for state in states[:-1]),
            "post_transition_samples": stats(state["speed"] for state in states[1:]),
        },
        "progress": {
            "start": states[0]["route_progress"],
            "final": states[-1]["route_progress"],
            "remaining": states[-1]["remaining_route_fraction"],
            "net_change": states[-1]["route_progress"] - states[0]["route_progress"],
            "positive_target_seconds": positive_seconds,
            "positive_target_net_change": positive_progress,
            "positive_target_progress_per_second": (
                positive_progress / positive_seconds if positive_seconds else None
            ),
        },
        "actions": {
            "counts": {action: counts[action] for action in ACTIONS},
            "mean_probabilities": [statistics.fmean(p[a] for p in probabilities) for a in range(3)],
            "entropy": stats(
                -sum(p * math.log(p) for p in probs if p > 0) for probs in probabilities
            ),
            "chosen_probability": stats(
                probs[d["proposed_action"]] for probs, d in zip(probabilities, decisions)
            ),
            "top_two_probability_gap": stats(sorted(p)[-1] - sorted(p)[-2] for p in probabilities),
        },
        "observations": {
            "ordered_identity_mismatch_decisions": ordered_mismatch,
            "same_set_reordered_decisions": same_set_reordered,
            "native_only_actor_occurrences": native_only,
            "forecast_only_actor_occurrences": forecast_only,
            "native_selected_count": stats(native_counts),
            "forecast_selected_count": stats(forecast_counts),
            "native_full_slots_decisions": sum(v == 14 for v in native_counts),
            "forecast_full_slots_decisions": sum(v == 14 for v in forecast_counts),
            "clipped_values_by_feature": clipping,
        },
        "forecasts": {
            "by_action": {a: forecast_summary(v) for a, v in forecasts.items()},
            "chosen_action": forecast_summary(selected_forecasts),
        },
    }


def read_trace_episodes(path, arm, seeds=protocol.SEEDS):
    """Stream complete gzip episodes in exact decision/transition/terminal order."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for seed in seeds:
            decisions, transitions = [], []
            while True:
                line = stream.readline()
                require(bool(line) and line.endswith("\n"), "Truncated trace/record")
                record = _json(line)
                require(
                    isinstance(record, dict)
                    and record.get("arm") == arm
                    and type(record.get("seed")) is int
                    and record["seed"] == seed,
                    "Trace arm/seed order differs",
                )
                if record.get("kind") == "terminal":
                    require(
                        bool(decisions) and len(decisions) == len(transitions),
                        "Terminal before a completed transition",
                    )
                    yield seed, decisions, transitions, record
                    break
                require(
                    record.get("kind") == "decision" and len(decisions) == len(transitions),
                    "Trace record order differs",
                )
                require(len(decisions) < 151, "Trace exceeds historical maximum recorded length")
                decisions.append(record)
                line = stream.readline()
                require(bool(line) and line.endswith("\n"), "Truncated transition record")
                transition = _json(line)
                require(
                    isinstance(transition, dict)
                    and transition.get("kind") == "transition"
                    and transition.get("arm") == arm
                    and type(transition.get("seed")) is int
                    and transition["seed"] == seed,
                    "Trace transition order/identity differs",
                )
                transitions.append(transition)
        require(stream.read() == "", "Extra trace records or episodes")


def _read_references(root):
    """Independent strict CSV parsing; never import a dataframe or simulator stack."""
    references = {}
    for arm, spec in protocol.ARMS.items():
        for key in ("model", "csv", "summary"):
            _pinned(root, spec[key], spec[f"{key}_sha256"])
        with _file(root, spec["csv"]).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == list(protocol.COLUMNS), "Reference CSV columns differ")
            rows = []
            for row in reader:
                require(
                    set(row) == set(protocol.COLUMNS) and None not in row.values(),
                    "Malformed reference CSV row",
                )
                parsed = {}
                for key, value in row.items():
                    if key in protocol.BOOL_FIELDS:
                        require(value in ("True", "False"), "Invalid CSV boolean")
                        parsed[key] = value == "True"
                    else:
                        require(bool(value), "Missing CSV numeric value")
                        number = float(value)
                        require(math.isfinite(number), "Nonfinite reference metric")
                        if key in protocol.COUNT_FIELDS:
                            require(number.is_integer(), "Nonintegral CSV count")
                            number = int(number)
                        parsed[key] = number
                protocol.validate_episode(parsed)
                rows.append(parsed)
        require(len(rows) == 500, "Reference CSV episode count differs")
        saved = _json(_file(root, spec["summary"]).read_text(encoding="utf-8"))
        protocol._summary_checks(arm, spec, saved, rows)
        references[arm] = rows
    require(protocol.selected_seeds(references) == protocol.SEEDS, "Reference selection differs")
    return references


def _utc(value):
    require(
        isinstance(value, str) and datetime.fromisoformat(value).utcoffset() is not None,
        "Missing/timezone-naive provenance timestamp",
    )


def _check_source_inventory(root, fingerprints):
    expected_py = {
        name for name in fingerprints
        if name.endswith(".py") and name.split("/")[0] in ("safeintent_rl", "scripts", "tests")
    }
    actual_py = {
        path.relative_to(root).as_posix()
        for folder in ("safeintent_rl", "scripts", "tests")
        for path in (root / folder).rglob("*.py")
    }
    require(actual_py == expected_py | ALLOWED_NEW_SOURCES,
            "Only the separately recorded analyzer and test sources may be newly present")


def _verify_inputs(root):
    """Check frozen bytes without recomputing a now-expanded runtime source inventory."""
    root = root.resolve(strict=True)
    input_hashes = {RELEASE: RELEASE_SHA256, PROOF: PROOF_SHA256, SUMMARY: SUMMARY_SHA256}
    release = _json(_pinned(root, RELEASE, RELEASE_SHA256).read_text(encoding="utf-8"))
    proof = _json(_pinned(root, PROOF, PROOF_SHA256).read_text(encoding="utf-8"))
    summary = _json(_pinned(root, SUMMARY, SUMMARY_SHA256).read_text(encoding="utf-8"))
    require(
        release["status"] == "released"
        and release["validation"]["verification_sha256"] == PROOF_SHA256,
        "Release status/proof pin differs",
    )
    require(
        release["protocol"]["arm_order"] == list(ARM_ORDER)
        and release["protocol"]["selected_seeds"] == list(protocol.SEEDS)
        and release["protocol"]["episodes"] == 111,
        "Released case order differs",
    )
    recorded_validation = {
        key: value for key, value in release["validation"].items() if key != "verification_sha256"
    }
    require(proof["validation"] == recorded_validation, "Released test/review proof differs")
    for key in ("source_sha256", "dependency_sha256", "runtime", "snapshot_archive_sha256"):
        require(proof[key] == release[key], "Release/proof fingerprint agreement differs")
    require(
        len(release["source_sha256"]) == 246 and len(release["dependency_sha256"]) == 227,
        "Released fingerprint inventory count differs",
    )
    require(
        sha(Path(protocol.__file__)) == release["source_sha256"][protocol.PROTOCOL],
        "Executing protocol helper differs from frozen source",
    )
    for name, digest in release["source_sha256"].items():
        _pinned(root, name, digest)
        input_hashes[name] = digest
    _check_source_inventory(root, release["source_sha256"])
    for name in ALLOWED_NEW_SOURCES:
        input_hashes[name] = sha(_file(root, name))
    require(sha(Path(__file__)) == input_hashes[ANALYZER], "Executing analyzer bytes differ")
    for name, digest in release["dependency_sha256"].items():
        path = Path(name)
        require(
            path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == digest,
            f"Frozen dependency bytes changed: {name}",
        )
        input_hashes[name] = digest
    for package in release["runtime"]["discovered_packages"]:
        for name, digest in package["metadata_sha256"].items():
            path = Path(name)
            require(
                path.is_absolute()
                and path.is_file()
                and not path.is_symlink()
                and sha(path) == digest,
                f"Frozen metadata bytes changed: {name}",
            )
            input_hashes[name] = digest
    snapshot = protocol.verify_snapshot(root)
    require(
        snapshot["archive_fingerprint"]["sha256"] == release["snapshot_archive_sha256"],
        "Snapshot release pin differs",
    )
    require(
        summary["status"] == "complete"
        and summary["current_phase"] == "finished"
        and summary["release_sha256"] == RELEASE_SHA256
        and summary["release"] == release
        and all(
            type(summary[k]) is int and summary[k] == 111
            for k in ("attempted_episodes", "completed_episodes", "matched_episodes")
        ),
        "Diagnostic summary is not the complete released run",
    )
    require(
        not any(
            key in summary
            for key in ("error", "failure_phase", "finalization_errors", "close_error")
        ),
        "Diagnostic records a failure",
    )
    _utc(summary["started_utc"])
    _utc(summary["finished_utc"])
    artifacts = summary["artifact_sha256"]
    expected_names = {
        "started.json",
        "episodes.jsonl",
        *[f"{arm}.trace.jsonl.gz" for arm in ARM_ORDER],
    }
    require(set(artifacts) == expected_names, "Raw artifact hash inventory differs")
    directory = root / RAW_DIRECTORY
    require(
        {path.name for path in directory.iterdir()} == expected_names | {"complete.json"},
        "Raw artifact directory inventory differs",
    )
    for name, digest in artifacts.items():
        relative = f"{RAW_DIRECTORY}/{name}"
        _pinned(root, relative, digest)
        input_hashes[relative] = digest
    marker_path = _pinned(root, COMPLETE, COMPLETE_SHA256)
    marker = _json(marker_path.read_text(encoding="utf-8"))
    require(
        set(marker) == {"status", "matched_episodes", "summary_sha256", "finished_utc"}
        and marker["status"] == "complete"
        and type(marker["matched_episodes"]) is int
        and marker["matched_episodes"] == 111
        and marker["summary_sha256"] == SUMMARY_SHA256,
        "Completion marker differs",
    )
    _utc(marker["finished_utc"])
    input_hashes[COMPLETE] = COMPLETE_SHA256
    require(
        not (root / "logs/v3_sector_timeout_diagnostic_v1.active.lock").exists(),
        "Diagnostic active/failed lock still exists",
    )
    return release, summary, _read_references(root), input_hashes


def _read_ledger(root, summary, references):
    path = _file(root, f"{RAW_DIRECTORY}/episodes.jsonl")
    with path.open(encoding="utf-8") as stream:
        records = [_json(line) for line in stream]
    expected = [(arm, seed) for arm in ARM_ORDER for seed in protocol.SEEDS]
    require(
        len(records) == 222 and len(summary["episode_checks"]) == 111,
        "Completed ledger count differs",
    )
    completed = {}
    for i, (arm, seed) in enumerate(expected):
        start, finish = records[2 * i : 2 * i + 2]
        require(
            set(start) == {"kind", "arm", "seed", "attempt_index", "utc"}
            and start["kind"] == "start",
            "Ledger start record differs",
        )
        _utc(start["utc"])
        require(
            set(finish) == {"kind", "arm", "seed", "attempt_index", "actual", "reference", "deltas"}
            and finish["kind"] == "complete"
            and finish == summary["episode_checks"][i],
            "Ledger completion/summary differs",
        )
        for record in (start, finish):
            require(
                record["arm"] == arm
                and type(record["seed"]) is int
                and record["seed"] == seed
                and type(record["attempt_index"]) is int
                and record["attempt_index"] == i,
                "Ledger order/identity differs",
            )
        reference = references[arm][seed - protocol.FIRST_SEED]
        protocol.compare_episode(finish["reference"], reference)
        require(
            finish["reference"] == reference, "Ledger reference values are not exact CSV values"
        )
        deltas = protocol.compare_episode(finish["actual"], reference)
        require(
            deltas == finish["deltas"] and all(value == 0 for value in deltas.values()),
            "Completed diagnostic metrics are not exactly reproduced",
        )
        completed[(arm, seed)] = finish["actual"]
    return completed


def _group(episodes):
    """Aggregate per-episode quantities, never treating individual decisions as cases."""
    fields = {
        "elapsed_seconds": lambda e: e["elapsed_seconds"],
        "zero_target_seconds": lambda e: e["zero_target"]["total_seconds"],
        "longest_zero_target_run_seconds": lambda e: e["zero_target"]["longest_seconds"],
        "tail_zero_target_seconds": lambda e: e["zero_target"]["tail_seconds"],
        "zero_target_run_count": lambda e: len(e["zero_target"]["runs"]),
        "relaunch_count": lambda e: len(e["zero_target"]["relaunches"]),
        "decision_sample_mean_speed": lambda e: e["speed"]["decision_samples"]["mean"],
        "post_transition_sample_mean_speed": lambda e: e["speed"]["post_transition_samples"][
            "mean"
        ],
        "terminal_route_progress": lambda e: e["progress"]["final"],
        "terminal_remaining_route_fraction": lambda e: e["progress"]["remaining"],
        "positive_target_progress_per_second": lambda e: e["progress"][
            "positive_target_progress_per_second"
        ],
        "mean_entropy": lambda e: e["actions"]["entropy"]["mean"],
        "mean_chosen_probability": lambda e: e["actions"]["chosen_probability"]["mean"],
        "mean_top_two_probability_gap": lambda e: e["actions"]["top_two_probability_gap"]["mean"],
        "identity_mismatch_decisions": lambda e: e["observations"][
            "ordered_identity_mismatch_decisions"
        ],
        "native_mean_selected_count": lambda e: e["observations"]["native_selected_count"]["mean"],
        "forecast_mean_selected_count": lambda e: e["observations"]["forecast_selected_count"][
            "mean"
        ],
    }
    results = {}
    for key, getter in fields.items():
        values = [getter(episode) for episode in episodes]
        results[key] = {
            **stats(value for value in values if value is not None),
            "undefined_episodes": sum(value is None for value in values),
        }
    return {
        "episode_count": len(episodes),
        "seeds": [e["seed"] for e in episodes],
        "episode_distributions": results,
        "target_seconds": {
            target: stats(e["target_seconds"][target] for e in episodes)
            for target in ("0", "4.5", "9")
        },
        "action_counts": {
            action: stats(e["actions"]["counts"][action] for e in episodes) for action in ACTIONS
        },
    }


def _verify_unchanged(root, fingerprints):
    _check_source_inventory(root, fingerprints)
    for name, digest in fingerprints.items():
        path = Path(name) if Path(name).is_absolute() else _file(root, name)
        require(
            path.is_file() and not path.is_symlink() and sha(path) == digest,
            f"Analysis input changed during read: {name}",
        )


def analyze(root):
    """Return a full analysis without writing any file or executing any research code."""
    root = Path(root).resolve(strict=True)
    release, summary, references, input_hashes = _verify_inputs(root)
    ledger = _read_ledger(root, summary, references)
    episodes = []
    for arm in ARM_ORDER:
        path = _file(root, f"{RAW_DIRECTORY}/{arm}.trace.jsonl.gz")
        for seed, decisions, transitions, terminal in read_trace_episodes(path, arm):
            episodes.append(
                summarize_episode(decisions, transitions, terminal, ledger[(arm, seed)])
            )
    require(
        [(e["arm"], e["seed"]) for e in episodes]
        == [(arm, seed) for arm in ARM_ORDER for seed in protocol.SEEDS],
        "Descriptive episodes do not cover all 111 ordered cases",
    )
    groups = {}
    for arm in ARM_ORDER:
        subset = [e for e in episodes if e["arm"] == arm]
        groups[arm] = {
            "all_selected": _group(subset),
            "by_observed_outcome": {
                outcome: _group([e for e in subset if e["outcome"] == outcome])
                for outcome in ("success", "collision", "incomplete")
            },
        }
    _verify_unchanged(root, input_hashes)
    return {
        "schema_version": 1,
        "name": "v3_sector_timeout_diagnostic_v1_descriptive_analysis",
        "status": "complete",
        "episode_count": len(episodes),
        "selected_seeds": list(protocol.SEEDS),
        "arm_order": list(ARM_ORDER),
        "research_episodes_executed_by_analysis": 0,
        "formulas": FORMULAS,
        "limitations": LIMITATIONS,
        "verification": {
            "ledger_reference_columns": list(protocol.COLUMNS),
            "trace_recomputed_columns": [
                "reward",
                "length",
                "travel_time",
                "min_ttc",
                "unsafe_ttc_events",
                "collision",
            ],
            "ledger_source_bound_columns": ["success", "safety_interventions"],
            "historical_source_count": len(release["source_sha256"]),
            "historical_dependency_count": len(release["dependency_sha256"]),
            "diagnostic_finished_utc": summary["finished_utc"],
        },
        "analysis_python": sys.version,
        "analysis_executable": sys.executable,
        "analysis_source_sha256": {
            name: input_hashes[name] for name in sorted(ALLOWED_NEW_SOURCES)
        },
        "input_sha256": dict(sorted(input_hashes.items())),
        "episodes": episodes,
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--refuse-overwrite", action="store_true", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    destination = root / OUTPUT
    require(destination.parent.is_dir(), "Analysis results directory missing")
    require(not destination.exists(), "Preserve existing analysis output")
    result = analyze(root)
    # Exclusive creation also protects against a target appearing during analysis.
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Offline analysis saved to {destination}")


if __name__ == "__main__":
    main()
