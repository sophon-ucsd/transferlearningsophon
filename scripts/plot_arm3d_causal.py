#!/usr/bin/env python3
"""Arm 3.3 figure 3d — causal ablation bar chart.

Reads causal_ablation_results.csv and produces a grouped bar chart per class:
  - intact d0 baseline (blue)
  - d0 zeroed at input (vermillion / red)

The H->bb drop is annotated prominently. Z->qq, W->qq, Hgg expected to be unchanged.

Usage:
    python scripts/plot_arm3d_causal.py \\
        --csv results/arm3/causal_ablation_results.csv \\
        --output results/arm3/arm3d_causal
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import apply_style, save_fig, OKABE_ITO


# Class display order — flavor-tagging-relevant first, others after
DISPLAY_ORDER = ["Hbb", "Hcc", "Tbqq", "Tbl", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "QCD"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/arm3/arm3d_causal")
    p.add_argument("--classes", default=None,
                   help="Comma list — restrict to these classes (default: all 10)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    apply_style()

    classes = args.classes.split(",") if args.classes else DISPLAY_ORDER
    classes = [c for c in classes if c in df["class"].values]

    intact, zerod, deltas = [], [], []
    for c in classes:
        row = df[df["class"] == c].iloc[0]
        intact.append(row["auc_intact"])
        zerod.append(row["auc_d0_zeroed"])
        deltas.append(row["delta"])

    intact = np.array(intact)
    zerod = np.array(zerod)
    deltas = np.array(deltas)

    n = len(classes)
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))
    blue = OKABE_ITO["blue"]
    red = OKABE_ITO["vermillion"]

    bars_intact = ax.bar(x - width/2, intact, width=width,
                          color=blue, edgecolor="black", linewidth=0.6,
                          label=r"$d_0$ intact (baseline)")
    bars_zero = ax.bar(x + width/2, zerod, width=width,
                       color=red, edgecolor="black", linewidth=0.6,
                       label=r"$d_0=0$ at input")

    # Delta annotation per class
    for i, (a, b, d) in enumerate(zip(intact, zerod, deltas)):
        # Down arrow for drops > 0.02
        if d > 0.02:
            ax.annotate(
                f"−{d:.3f}",
                xy=(i, b - 0.02), xytext=(i, b - 0.06),
                ha="center", va="top", fontsize=9, color="#7C2A1D", fontweight="bold",
            )
            ax.annotate("",
                        xy=(i + width/2, b), xytext=(i - width/2, a),
                        arrowprops=dict(arrowstyle="->", color="#7C2A1D", lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_ylabel("Per-class AUC (one-vs-rest)")
    ax.set_title(r"Causal ablation: zeroing $d_0$ at the input to pretrained Sophon")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="#888", lw=0.5)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="gray")

    save_fig(fig, args.output)
    print(f"Saved: {args.output}.{{pdf,png}}")
    plt.close()

    # Headline numbers
    if "Hbb" in classes:
        i = classes.index("Hbb")
        print(f"\nHeadline: Hbb AUC {intact[i]:.4f} -> {zerod[i]:.4f}  (Δ = {deltas[i]:+.4f})")


if __name__ == "__main__":
    main()
