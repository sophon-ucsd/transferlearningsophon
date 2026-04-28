#!/usr/bin/env python3
"""Figure 4 — Arm 3.2 cluster metrics bar chart.

Three side-by-side bars (Silhouette / k-NN / Linear AUC), pretrained vs full-FT.

Reads:
    results/arm3/arm3b_cluster_metrics.json

Outputs:
    results/arm3/arm3b_cluster_metrics.{pdf,png}
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import set_publishable_style, save_fig, OKABE_ITO


def main():
    set_publishable_style()
    data = json.loads(
        open("results/arm3/arm3b_cluster_metrics.json").read()
    )["results"]
    pre = data["pretrained"]
    ft  = data["finetuned"]

    metrics = [
        ("silhouette", "Silhouette",        (-0.05, 0.30)),
        ("knn",        "k-NN accuracy",     ( 0.0,  1.00)),
        ("linear",     "Linear AUC",        ( 0.5,  1.00)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.4),
                             gridspec_kw=dict(wspace=0.40))
    for ax, (k, title, (ymin, ymax)) in zip(axes, metrics):
        pre_v = pre[k]["overall"]
        ft_v  = ft[k]["overall"]
        ax.bar(0, pre_v, width=0.6,
               color=OKABE_ITO["blue"], linewidth=0)
        ax.bar(1, ft_v, width=0.6,
               color=OKABE_ITO["vermillion"], linewidth=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pretrained", "Full-FT 3M"], fontsize=10)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, pad=8)
        # value labels above bars
        for i, v in enumerate([pre_v, ft_v]):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=10, color="#333")
        # delta annotation
        ax.text(0.5, ymax * 0.95,
                rf"$\Delta = {ft_v - pre_v:+.3f}$",
                ha="center", va="top", fontsize=10,
                color="#666", transform=ax.transData)

    save_fig(fig, "results/arm3/arm3b_cluster_metrics")
    print("Saved: results/arm3/arm3b_cluster_metrics.{pdf,png}")


if __name__ == "__main__":
    main()
