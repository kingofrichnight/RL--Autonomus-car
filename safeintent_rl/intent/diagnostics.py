from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from safeintent_rl.intent.evaluation import classification_metrics


@dataclass
class OnlineIntentDiagnostics:
    decisions: int = 0
    slot_observations: int = 0
    missing_slots: int = 0
    obstacle_slots: int = 0
    vehicle_slots: int = 0
    warmup_vehicle_slots: int = 0
    predicted_vehicle_slots: int = 0
    labeled_predictions: int = 0
    correct_predictions: int = 0
    confidence_sum: float = 0.0
    entropy_sum: float = 0.0
    true_labels: list[int] = field(default_factory=list)
    predicted_labels: list[int] = field(default_factory=list)

    def update(self, snapshot: dict[str, Any]) -> None:
        count_keys = (
            "slot_count", "missing_slots", "obstacle_slots", "vehicle_slots",
            "warmup_vehicle_slots", "predicted_vehicle_slots", "labeled_predictions",
            "correct_predictions",
        )
        if any(not isinstance(snapshot[key], Integral) for key in count_keys):
            raise ValueError("Intent diagnostic counts must be integers")
        raw_labels = list(snapshot["true_labels"]) + list(snapshot["predicted_labels"])
        if any(not isinstance(value, Integral) or not 0 <= value < 3 for value in raw_labels):
            raise ValueError("Intent diagnostic labels must be project class indices")
        slot_count = int(snapshot["slot_count"])
        missing = int(snapshot["missing_slots"])
        obstacles = int(snapshot["obstacle_slots"])
        vehicles = int(snapshot["vehicle_slots"])
        warmup = int(snapshot["warmup_vehicle_slots"])
        predicted = int(snapshot["predicted_vehicle_slots"])
        labeled = int(snapshot["labeled_predictions"])
        true_labels = [int(value) for value in snapshot["true_labels"]]
        predicted_labels = [int(value) for value in snapshot["predicted_labels"]]

        values = [slot_count, missing, obstacles, vehicles, warmup, predicted, labeled]
        if any(value < 0 for value in values):
            raise ValueError("Intent diagnostic counts must be non-negative")
        if missing + obstacles + vehicles != slot_count:
            raise ValueError("Intent diagnostic slot counts are inconsistent")
        if warmup + predicted != vehicles:
            raise ValueError("Intent diagnostic vehicle counts are inconsistent")
        if labeled > predicted or len(true_labels) != labeled:
            raise ValueError("Intent diagnostic labeled counts are inconsistent")
        if len(predicted_labels) != labeled:
            raise ValueError("Intent diagnostic prediction counts are inconsistent")
        correct = sum(a == b for a, b in zip(true_labels, predicted_labels, strict=True))
        if int(snapshot["correct_predictions"]) != correct:
            raise ValueError("Intent diagnostic correct prediction count is inconsistent")
        confidence = float(snapshot["confidence_sum"])
        entropy = float(snapshot["entropy_sum"])
        if (
            not math.isfinite(confidence) or not math.isfinite(entropy)
            or not 0 <= confidence <= labeled
            or not 0 <= entropy <= labeled * math.log(3) + 1e-5
        ):
            raise ValueError("Intent diagnostic confidence or entropy is invalid")

        self.decisions += 1
        self.slot_observations += slot_count
        self.missing_slots += missing
        self.obstacle_slots += obstacles
        self.vehicle_slots += vehicles
        self.warmup_vehicle_slots += warmup
        self.predicted_vehicle_slots += predicted
        self.labeled_predictions += labeled
        self.correct_predictions += correct
        self.confidence_sum += confidence
        self.entropy_sum += entropy
        self.true_labels.extend(true_labels)
        self.predicted_labels.extend(predicted_labels)

    def summarize(self, label_names: list[str]) -> dict[str, Any]:
        def ratio(numerator: float, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        metrics = (
            classification_metrics(self.true_labels, self.predicted_labels, label_names)
            if self.labeled_predictions else None
        )
        expected_correct = sum(
            true == predicted
            for true, predicted in zip(self.true_labels, self.predicted_labels, strict=True)
        )
        if self.correct_predictions != expected_correct:
            raise ValueError("Online intent correct count does not match the confusion matrix")

        return {
            "decisions": self.decisions,
            "slot_observations": self.slot_observations,
            "missing_slots": self.missing_slots,
            "obstacle_slots": self.obstacle_slots,
            "vehicle_slots": self.vehicle_slots,
            "warmup_vehicle_slots": self.warmup_vehicle_slots,
            "predicted_vehicle_slots": self.predicted_vehicle_slots,
            "labeled_predictions": self.labeled_predictions,
            "vehicle_slot_rate": ratio(self.vehicle_slots, self.slot_observations),
            "prediction_coverage": ratio(self.predicted_vehicle_slots, self.vehicle_slots),
            "labeled_prediction_rate": ratio(
                self.labeled_predictions, self.predicted_vehicle_slots
            ),
            "mean_confidence": ratio(self.confidence_sum, self.labeled_predictions),
            "mean_entropy": ratio(self.entropy_sum, self.labeled_predictions),
            "mean_normalized_entropy": (
                ratio(self.entropy_sum / math.log(len(label_names)), self.labeled_predictions)
            ),
            "classification_available": metrics is not None,
            "all_classes_observed": set(self.true_labels) == set(range(len(label_names))),
            "classification": metrics,
        }


@dataclass
class HistoryCoverageComparison:
    """Compare readiness of the policy tracker and a non-interventional shadow."""

    decisions: int = 0
    vehicle_slots: int = 0
    both_ready: int = 0
    current_only: int = 0
    shadow_only: int = 0
    neither_ready: int = 0

    def update(self, current: dict[str, Any], shadow: dict[str, Any]) -> None:
        identity_keys = ("slot_count", "missing_slots", "obstacle_slots", "vehicle_slots")
        if any(current[key] != shadow[key] for key in identity_keys):
            raise ValueError("Current and shadow diagnostic slots do not match")
        slot_count = int(current["slot_count"])
        vehicle_rows = {int(value) for value in current["vehicle_rows"]}
        shadow_vehicle_rows = {int(value) for value in shadow["vehicle_rows"]}
        current_ready = {int(value) for value in current["predicted_rows"]}
        shadow_ready = {int(value) for value in shadow["predicted_rows"]}
        if (
            vehicle_rows != shadow_vehicle_rows
            or len(vehicle_rows) != int(current["vehicle_slots"])
            or any(not 0 <= value < slot_count for value in vehicle_rows)
            or not current_ready <= vehicle_rows
            or not shadow_ready <= vehicle_rows
            or len(current_ready) != int(current["predicted_vehicle_slots"])
            or len(shadow_ready) != int(shadow["predicted_vehicle_slots"])
        ):
            raise ValueError("Current and shadow readiness rows are inconsistent")

        both = current_ready & shadow_ready
        current_only = current_ready - shadow_ready
        shadow_only = shadow_ready - current_ready
        neither = vehicle_rows - (current_ready | shadow_ready)
        self.decisions += 1
        self.vehicle_slots += len(vehicle_rows)
        self.both_ready += len(both)
        self.current_only += len(current_only)
        self.shadow_only += len(shadow_only)
        self.neither_ready += len(neither)

    def summarize(self) -> dict[str, Any]:
        accounted = self.both_ready + self.current_only + self.shadow_only + self.neither_ready
        if accounted != self.vehicle_slots:
            raise ValueError("Coverage comparison counts do not cover every vehicle slot")

        def ratio(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        current_ready = self.both_ready + self.current_only
        shadow_ready = self.both_ready + self.shadow_only
        return {
            "decisions": self.decisions,
            "vehicle_slots": self.vehicle_slots,
            "both_ready": self.both_ready,
            "current_only": self.current_only,
            "shadow_only": self.shadow_only,
            "neither_ready": self.neither_ready,
            "current_coverage": ratio(current_ready, self.vehicle_slots),
            "shadow_coverage": ratio(shadow_ready, self.vehicle_slots),
            "absolute_coverage_gain": ratio(
                shadow_ready - current_ready,
                self.vehicle_slots,
            ),
            "current_warmup_recovery": ratio(
                self.shadow_only,
                self.shadow_only + self.neither_ready,
            ),
            "shadow_is_readiness_superset": self.current_only == 0,
        }


@dataclass
class HistoryLengthCoverageCurve:
    """Count counterfactual readiness for predeclared shorter history windows."""

    candidate_lengths: tuple[int, ...]
    decisions: int = 0
    vehicle_slots: int = 0
    ready_counts: dict[int, int] = field(default_factory=dict)
    observed_length_counts: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = tuple(int(value) for value in self.candidate_lengths)
        if (
            not normalized
            or normalized != tuple(sorted(set(normalized)))
            or any(value <= 0 for value in normalized)
        ):
            raise ValueError("History coverage candidates must be unique positive lengths")
        self.candidate_lengths = normalized
        self.ready_counts = {value: 0 for value in normalized}

    def update(self, shadow: dict[str, Any]) -> None:
        lengths = list(shadow["history_lengths"])
        vehicle_slots = int(shadow["vehicle_slots"])
        if (
            len(lengths) != vehicle_slots
            or any(not isinstance(value, Integral) or value <= 0 for value in lengths)
            or any(value > self.candidate_lengths[-1] for value in lengths)
        ):
            raise ValueError("Shadow history lengths are inconsistent with vehicle slots")
        normalized = [int(value) for value in lengths]
        additions = {
            candidate: sum(length >= candidate for length in normalized)
            for candidate in self.candidate_lengths
        }

        self.decisions += 1
        self.vehicle_slots += vehicle_slots
        for candidate, count in additions.items():
            self.ready_counts[candidate] += count
        for length in normalized:
            self.observed_length_counts[length] = self.observed_length_counts.get(length, 0) + 1

    def summarize(self, *, minimum_coverage: float) -> dict[str, Any]:
        if not 0 <= minimum_coverage <= 1:
            raise ValueError("Minimum coverage must be between zero and one")
        coverage = {
            str(candidate): {
                "ready_vehicle_slots": self.ready_counts[candidate],
                "warmup_vehicle_slots": self.vehicle_slots - self.ready_counts[candidate],
                "coverage": (
                    self.ready_counts[candidate] / self.vehicle_slots
                    if self.vehicle_slots
                    else None
                ),
            }
            for candidate in self.candidate_lengths
        }
        eligible = [
            candidate
            for candidate in self.candidate_lengths
            if self.vehicle_slots
            and self.ready_counts[candidate] / self.vehicle_slots >= minimum_coverage
        ]
        return {
            "decisions": self.decisions,
            "vehicle_slots": self.vehicle_slots,
            "candidate_lengths": list(self.candidate_lengths),
            "minimum_coverage": minimum_coverage,
            "coverage_by_history_length": coverage,
            "observed_history_length_counts": {
                str(length): self.observed_length_counts.get(length, 0)
                for length in range(1, self.candidate_lengths[-1] + 1)
            },
            "selected_history_length": max(eligible) if eligible else None,
            "selection_rule": "longest candidate meeting minimum coverage",
        }
