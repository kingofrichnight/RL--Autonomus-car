from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot policy evaluation comparisons")
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", default="results/policy_comparison.png")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.csv_files):
        parser.error("--labels must contain one label for every CSV file")
    labels = args.labels or [Path(path).stem for path in args.csv_files]
    frames = [pd.read_csv(path) for path in args.csv_files]

    success = [100 * frame["success"].mean() for frame in frames]
    collision = [100 * frame["collision"].mean() for frame in frames]
    travel_time = [frame["travel_time"].mean() for frame in frames]
    finite_min_ttc = [
        frame.loc[np.isfinite(frame["min_ttc"]), "min_ttc"].mean() for frame in frames
    ]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = ["#2f6fed", "#e34a33", "#31a354", "#756bb1", "#636363"]
    metrics = (
        (success, "Success rate (%)", True),
        (collision, "Collision rate (%)", False),
        (travel_time, "Mean travel time (s)", False),
        (finite_min_ttc, "Mean minimum TTC (s)", True),
    )
    for axis, (values, title, higher_is_better) in zip(axes.flat, metrics, strict=True):
        bars = axis.bar(labels, values, color=colors[: len(labels)])
        axis.set_title(title + (" ↑" if higher_is_better else " ↓"))
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f", padding=3)

    figure.suptitle("SafeIntent-RL policy comparison", fontsize=15)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved comparison plot to {output}")


if __name__ == "__main__":
    main()
