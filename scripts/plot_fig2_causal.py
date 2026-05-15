#!/usr/bin/env python3
"""Figure 2 — Arm 3.3 causal ablation (per-class AUC, three interventions).

Reads:
    results/arm3/causal_ablation_results.csv

Outputs:
    results/arm3/arm3d_causal.{pdf,png}
"""
from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import set_publishable_style, save_fig, OKABE_ITO


# Per spec: order by resample-drop magnitude descending, "macro" at right with a gap
ORDER_FIXED = ["Hcc", "Hgg", "Hbb", "Zqq", "QCD",
               "Wqq", "Tbqq", "H4q", "Hqql", "Tbl"]
GAP_BEFORE_MACRO = 0.7

BASELINE_COLOR = "#999999"          # neutral gray
ZERO_COLOR     = OKABE_ITO["orange"]      # secondary intervention
RESAMPLE_COLOR = OKABE_ITO["vermillion"]  # principal causal estimate


def main():
    set_publishable_style()
    df = pd.read_csv("results/arm3/causal_ablation_results.csv")

    # Reorder by spec, then macro at the end
    df_idx = df.set_index("class")
    classes = ORDER_FIXED
    rows = [df_idx.loc[c] for c in classes]
    macro = df_idx.loc["macro"]

    n = len(classes)
    pos = np.concatenate([np.arange(n, dtype=float),
                          np.array([n + GAP_BEFORE_MACRO])])
    labels = classes + ["macro"]

    baseline = np.array([r["baseline_AUC"] for r in rows] + [macro["baseline_AUC"]])
    zerod    = np.array([r["zero_AUC"]     for r in rows] + [macro["zero_AUC"]])
    resamp   = np.array([r["resample_AUC"] for r in rows] + [macro["resample_AUC"]])
    delta_r  = np.array([r["delta_resample"] for r in rows] + [macro["delta_resample"]])

    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.0))

    ax.bar(pos - width, baseline, width=width,
           color=BASELINE_COLOR, linewidth=0,
           label=r"Baseline ($d_0$ intact)", zorder=2)
    ax.bar(pos,         zerod,    width=width,
           color=ZERO_COLOR, linewidth=0,
           label=r"Zero-ablation ($d_0 = 0$)", zorder=2)
    ax.bar(pos + width, resamp,   width=width,
           color=RESAMPLE_COLOR, linewidth=0,
           label=r"Resample-ablation (marginal pool)", zorder=2)

    # Resample-drop annotation above each group
    for i, (p, b, r, d) in enumerate(zip(pos, baseline, resamp, delta_r)):
        ax.annotate(rf"$\Delta = {d:+.3f}$",
                    xy=(p, max(b, r) + 0.005),
                    xytext=(p, max(b, r) + 0.018),
                    ha="center", va="bottom", fontsize=9,
                    color="#333", clip_on=False)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=11)
    # Make macro x-tick distinct
    for tl, lbl in zip(ax.get_xticklabels(), labels):
        if lbl == "macro":
            tl.set_fontweight("bold")
            tl.set_fontstyle("italic")

    # Visual divider before macro
    ax.axvline(n + GAP_BEFORE_MACRO/2 - 0.5, color="#bbb", lw=0.7, ls="--", zorder=1)

    ax.set_ylim(0.70, 1.04)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_ylabel("Per-class AUC (one-vs-rest)")
    ax.set_title(r"Causal ablation of $d_0$ at the input to pretrained Sophon", pad=14)

    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="y", alpha=1.0)  # color from style.py

    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.20)
    save_fig(fig, "results/arm3/arm3d_causal")
    print("Saved: results/arm3/arm3d_causal.{pdf,png}")


if __name__ == "__main__":
    main()
