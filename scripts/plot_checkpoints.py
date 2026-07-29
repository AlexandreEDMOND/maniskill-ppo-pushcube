"""Plot fixed-evaluation learning curves from custom PPO checkpoint reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} does not contain checkpoint metrics")
    return data


def plot_series(axis, metrics: list[dict], label: str, key: str, title: str) -> None:
    steps_millions = [entry["step"] / 1_000_000 for entry in metrics]
    if key in metrics[0]["evaluation"]:
        values = [entry["evaluation"][key] for entry in metrics]
    else:
        values = [entry["diagnostics"][key] for entry in metrics]
    axis.plot(steps_millions, values, marker="o", linewidth=2, label=label)
    axis.set_title(title)
    axis.set_xlabel("Environment interactions (millions)")
    axis.grid(alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, nargs="+", help="checkpoint_metrics.json files")
    parser.add_argument("--labels", nargs="+", help="one label per metrics file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = args.labels or [path.parent.name for path in args.metrics]
    if len(labels) != len(args.metrics):
        raise ValueError("--labels must contain exactly one entry per metrics file")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    charts = [
        ("success_rate", "Fixed-evaluation success rate"),
        ("final_cube_to_goal_distance_mean", "Final cube-to-goal distance (m)"),
        ("return_mean", "Undiscounted episode return"),
        ("action_std_mean", "Policy action standard deviation"),
    ]
    for path, label in zip(args.metrics, labels, strict=True):
        metrics = load_metrics(path)
        for axis, (key, title) in zip(axes.flat, charts, strict=True):
            plot_series(axis, metrics, label, key, title)

    axes[0, 0].set_ylim(bottom=0, top=1)
    axes[0, 1].axhline(0.1, color="tab:red", linestyle="--", label="success threshold")
    for axis in axes.flat:
        axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"Plot: {args.output}")


if __name__ == "__main__":
    main()
