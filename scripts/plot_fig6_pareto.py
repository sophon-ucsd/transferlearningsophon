#!/usr/bin/env python3
"""Figure 6 — Arm 2 Pareto: trainable parameters vs JetClass-1 macro AUC.

Numbers verified against:
  - Qu+ 2022 Table 1 (PFN, P-CNN, ParticleNet, ParT)
  - LLoCa Table 5 (MIParT-L, L-GATr)
  - This work (Frozen / Partial / Full Sophon adaptations)

Models with FLOPs-only counts (no clear param count) are excluded.
"""
from __future__ import annotations

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import set_publishable_style, save_fig, STRATEGY_COLORS


# (name, trainable_params, macro_AUC, citation)
BASELINES = [
    ("PFN",          86_000,  0.9714, "Qu+'22"),
    ("P-CNN",        354_000, 0.9789, "Qu+'22"),
    ("ParticleNet",  366_000, 0.9849, "Qu+'22"),
    ("ParT",         2_140_000, 0.9877, "Qu+'22"),
    ("MIParT-L",     225_000, 0.9878, "LLoCa Tab 5"),
    ("L-GATr",       30_000_000, 0.9885, "LLoCa Tab 5"),
]

# Reference point: ParT trained on 2M jets (smaller training set; same params)
PARTSMALL = ("ParT (2M jets)", 2_140_000, 0.9834)

SOPHON = [
    ("Frozen Sophon + MLP",    35_594,    0.9792),
    ("Partial-FT Sophon (3M)", 1_400_000, 0.9793),
    ("Full-FT Sophon (3M)",    2_177_790, 0.9844),
]


def main():
    set_publishable_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Baselines
    bx = np.array([b[1] for b in BASELINES], dtype=float)
    by = np.array([b[2] for b in BASELINES], dtype=float)
    ax.scatter(bx, by, s=70, facecolors="white", edgecolors="#333",
               linewidths=1.2, zorder=3)
    label_offsets = {
        "PFN":          ( 8, -2),
        "P-CNN":        ( 8, -2),
        "ParticleNet":  ( 8, -2),
        "ParT":         ( 8, -2),
        "MIParT-L":     ( 8,  6),
        "L-GATr":       (-8, -10),
    }
    for n, x, y, _ in BASELINES:
        dx, dy = label_offsets.get(n, (8, -2))
        ha = "left" if dx > 0 else "right"
        ax.annotate(n, xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=9, color="#333", ha=ha, va="center")

    # ParT-2M reference (filled gray for "ParT trained with same data scale as ours")
    ax.scatter([PARTSMALL[1]], [PARTSMALL[2]], s=70, marker="o",
               color="#aaa", edgecolor="#333", linewidths=1.2, zorder=3)
    ax.annotate(PARTSMALL[0], xy=(PARTSMALL[1], PARTSMALL[2]),
                xytext=(8, -2), textcoords="offset points",
                fontsize=9, color="#666", ha="left", va="center", style="italic")

    # Sophon strategies
    sophon_colors = [STRATEGY_COLORS["frozen"],
                     STRATEGY_COLORS["partial_ft"],
                     STRATEGY_COLORS["full_ft"]]
    for (n, x, y), color in zip(SOPHON, sophon_colors):
        ax.scatter([x], [y], s=180, marker="*", color=color,
                   edgecolor="#222", linewidths=0.8, zorder=4,
                   label=n)

    # Pareto frontier across baselines + Sophon (by params ascending; keep dominators)
    pts = [(b[1], b[2]) for b in BASELINES] + [(s[1], s[2]) for s in SOPHON]
    pts = sorted(pts, key=lambda p: p[0])
    frontier = []
    best = -np.inf
    for x, y in pts:
        if y > best:
            frontier.append((x, y))
            best = y
    fx, fy = zip(*frontier)
    ax.plot(fx, fy, color="#888", lw=1.0, linestyle=(0, (4, 3)),
            alpha=0.85, zorder=2)

    # Frozen Sophon callout
    fz = SOPHON[0]
    ax.annotate(
        f"Frozen Sophon: AUC = {fz[2]:.3f}\n{fz[1]/1000:.0f}k trainable params",
        xy=(fz[1], fz[2]),
        xytext=(2.0e5, 0.973),
        ha="left", va="top",
        fontsize=9, color="#222",
        arrowprops=dict(arrowstyle="-", lw=0.7, color="#666"),
        bbox=dict(facecolor="white", edgecolor="#bbb",
                  boxstyle="round,pad=0.3", alpha=0.9),
    )

    ax.set_xscale("log")
    ax.set_xlim(2e4, 6e7)
    ax.set_ylim(0.965, 0.992)
    ax.set_xticks([1e5, 1e6, 1e7])
    ax.set_xticklabels(["100K", "1M", "10M"])
    ax.minorticks_off()
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel(r"Macro AUC on JetClass-1")
    ax.set_title("Pareto: parameters vs. AUC on JetClass-1 (10-class)")

    # Legend (only the 3 Sophon entries + frontier marker)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#888", lw=1.0,
                          linestyle=(0, (4, 3)), label="Pareto frontier"))
    labels.append("Pareto frontier")
    ax.legend(handles, labels, loc="lower right", fontsize=9)

    save_fig(fig, "results/arm2/arm2_pareto")
    save_fig(fig, "results/main_plots/pareto")
    print("Saved: results/arm2/arm2_pareto.{pdf,png} + results/main_plots/pareto.{pdf,png}")


if __name__ == "__main__":
    main()
