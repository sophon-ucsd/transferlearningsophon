#!/usr/bin/env python3
"""Plot substructure probing results.

Figure A: Bar chart of R² per observable (all classes)
Figure B: Heatmap of R² per observable × class

Usage:
    python scripts/plot_probing.py probing_results.json --output-dir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Observable categories for color-coding
OBS_CATEGORIES = {
    "jet_mass": "kinematics",
    "jet_pt": "kinematics",
    "multiplicity": "kinematics",
    "tau1": "substructure",
    "tau2": "substructure",
    "tau3": "substructure",
    "tau21": "substructure",
    "tau32": "substructure",
    "C2": "substructure",
    "width": "substructure",
    "mean_d0": "tracking",
    "mean_dz": "tracking",
    "charged_frac": "tracking",
}

CATEGORY_COLORS = {
    "kinematics": "#4477AA",
    "substructure": "#228833",
    "tracking": "#EE6677",
}

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def plot_bar_chart(results, output):
    """Figure A: R² bar chart for all classes combined."""
    all_cls = results["all_classes"]
    obs_names = list(all_cls.keys())
    r2_means = [all_cls[o]["R2_mean"] for o in obs_names]
    r2_stds = [all_cls[o]["R2_std"] for o in obs_names]
    colors = [CATEGORY_COLORS[OBS_CATEGORIES[o]] for o in obs_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(obs_names))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=3, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", "\n") for o in obs_names], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Linear Probing R²", fontsize=12)
    ax.set_title("Physics Information in Sophon Pretrained Embeddings", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["kinematics"], label="Kinematics"),
        Patch(facecolor=CATEGORY_COLORS["substructure"], label="Substructure"),
        Patch(facecolor=CATEGORY_COLORS["tracking"], label="Tracking"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_heatmap(results, output):
    """Figure B: R² heatmap per observable × class."""
    per_class = results["per_class"]
    obs_names = list(results["all_classes"].keys())
    classes = [c for c in LABEL_NAMES if c in per_class]

    # Build matrix
    matrix = np.zeros((len(obs_names), len(classes)))
    for j, cls in enumerate(classes):
        for i, obs in enumerate(obs_names):
            if obs in per_class[cls]:
                matrix[i, j] = per_class[cls][obs]["R2_mean"]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(obs_names)))
    ax.set_yticklabels([o.replace("_", " ") for o in obs_names], fontsize=9)

    # Add text annotations
    for i in range(len(obs_names)):
        for j in range(len(classes)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="R²", shrink=0.8)
    ax.set_title("Probing R² by Observable and Jet Class", fontsize=13)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="Path to probing_results.json")
    parser.add_argument("--output-dir", default="figures/")
    args = parser.parse_args()

    with open(args.results_json) as f:
        results = json.load(f)

    out = Path(args.output_dir)
    plot_bar_chart(results, str(out / "probing_bar.pdf"))
    plot_heatmap(results, str(out / "probing_heatmap.pdf"))


if __name__ == "__main__":
    main()
