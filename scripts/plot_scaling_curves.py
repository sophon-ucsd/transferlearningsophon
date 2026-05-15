#!/usr/bin/env python3
"""Generate scaling curve plots from collected results.

Usage:
    python scripts/plot_scaling_curves.py results/results_summary.csv
    python scripts/plot_scaling_curves.py results/results_summary.csv --output-dir figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Style per strategy
STRATEGY_CONFIG = {
    "from_scratch": {"color": "#888888", "linestyle": "--", "marker": "s", "label": "From Scratch"},
    "frozen":       {"color": "#4477AA", "linestyle": "-",  "marker": "o", "label": "Frozen Sophon + MLP"},
    "partial_ft":   {"color": "#EE6677", "linestyle": "-",  "marker": "^", "label": "Partial Fine-tuning"},
    "full_ft":      {"color": "#228833", "linestyle": "-",  "marker": "D", "label": "Full Fine-tuning"},
}


def plot_accuracy_vs_data(df: pd.DataFrame, output: str) -> None:
    """Mentor's Plot: X = dataset size (log), Y = test accuracy (%)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for strategy, style in STRATEGY_CONFIG.items():
        subset = df[df["strategy"] == strategy].sort_values("train_size")
        if subset.empty:
            continue

        x = subset["train_size"].values
        y = subset["test_acc_mean"].values * 100
        yerr = subset["test_acc_std"].values * 100 if "test_acc_std" in subset else None

        ax.plot(
            x, y, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markersize=6, linewidth=2,
            label=style["label"], zorder=3,
        )
        if yerr is not None and len(x) > 1:
            ax.fill_between(x, y - yerr, y + yerr, color=style["color"], alpha=0.15, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Jets", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Transfer Learning Scaling: Sophon on JetClass", fontsize=13)
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor="gray")
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_loss_vs_data(df: pd.DataFrame, output: str) -> None:
    """Loss version for power-law fitting. X = dataset size (log), Y = loss (log)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for strategy, style in STRATEGY_CONFIG.items():
        subset = df[df["strategy"] == strategy].sort_values("train_size")
        if subset.empty:
            continue

        x = subset["train_size"].values
        y = subset["test_loss_mean"].values
        yerr = subset["test_loss_std"].values if "test_loss_std" in subset else None

        ax.plot(
            x, y, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markersize=6, linewidth=2,
            label=style["label"], zorder=3,
        )
        if yerr is not None and len(x) > 1:
            ax.fill_between(x, y - yerr, y + yerr, color=style["color"], alpha=0.15, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Training Jets", fontsize=12)
    ax.set_ylabel("Test Cross-Entropy Loss", fontsize=12)
    ax.set_title("Loss Scaling Curves", fontsize=13)
    ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor="gray")
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", help="Path to results_summary.csv")
    parser.add_argument("--output-dir", default="figures/", help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    out = Path(args.output_dir)
    plot_accuracy_vs_data(df, output=str(out / "accuracy_scaling.pdf"))
    plot_loss_vs_data(df, output=str(out / "loss_scaling.pdf"))


if __name__ == "__main__":
    main()
