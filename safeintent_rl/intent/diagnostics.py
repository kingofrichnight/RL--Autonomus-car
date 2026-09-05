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
