#!/usr/bin/env python3
"""Arm 3.3 figure 3c — probing R² bar chart.

Reads probing_results.csv and produces a grouped bar chart of R² per observable,
two bars per observable (pretrained gray, fine-tuned blue), with bootstrap 95% CI.

Observables grouped along x-axis:
  Substructure | Displacement (d0 family) | Displacement (dz family)

The displacement group is highlighted to emphasize the mean(|d0|) → max(|d0|) story.

Usage:
    python scripts/plot_arm3c_probing.py \\
        --csv results/arm3/probing_results.csv \\
        --output results/arm3/arm3c_probing
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
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import apply_style, save_fig, PRE_VS_FT, FIG_SIZE


# Display order on the x-axis. Group dividers between sets.
SUBSTRUCTURE = [
    ("multiplicity",       "multiplicity"),
    ("jet_mass",           "jet mass"),
    ("jet_pt",             r"$p_T$"),
    ("width",              "jet width"),
    ("tau21",              r"$\tau_{21}$"),
    ("tau32",              r"$\tau_{32}$"),
    ("C2",                 r"$C_2$"),
    ("charged_frac",       "charged fraction"),
]
D0_GROUP = [
    ("mean_abs_d0",        r"mean$|d_0|$"),
    ("max_abs_d0",         r"max$|d_0|$"),
    ("top3_sum_abs_d0",    r"top-3 $|d_0|$"),
    ("count_d0_gt_1sigma", r"#$|d_0|/\sigma>1$"),
    ("count_d0_gt_2sigma", r"#$|d_0|/\sigma>2$"),
]
DZ_GROUP = [
    ("mean_abs_dz",        r"mean$|d_z|$"),
    ("max_abs_dz",         r"max$|d_z|$"),
    ("count_dz_gt_2sigma", r"#$|d_z|/\sigma>2$"),
]
GROUPS = [
    ("Substructure",       SUBSTRUCTURE),
    (r"Displacement: $d_0$", D0_GROUP),
    (r"Displacement: $d_z$", DZ_GROUP),
]


def get_value(df: pd.DataFrame, model: str, observable: str):
    sub = df[(df.model == model) & (df.observable == observable)]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return r["R2_mean"], r["R2_lo"], r["R2_hi"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="results/arm3/arm3c_probing")
    p.add_argument("--pretrained-tag", default="pretrained")
    p.add_argument("--finetune-tag", default="full_ft_3M_seed42")
    p.add_argument("--finetune-label", default="Full FT 3M, seed=42")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    apply_style()

    # Compose the x-axis order
    obs_order = []
    obs_labels = []
    group_spans = []  # list of (start_idx, end_idx, group_name)
    cur = 0
    for group_name, items in GROUPS:
        start = cur
        for key, lbl in items:
            obs_order.append(key)
            obs_labels.append(lbl)
            cur += 1
        group_spans.append((start, cur, group_name))

    n = len(obs_order)
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 5))

    # Highlight band behind the d0 group
    for start, end, name in group_spans:
        if "d_0" in name:
            ax.axvspan(start - 0.5, end - 0.5, color="#FFF4E0", alpha=0.7, zorder=0)

    pre_vals, pre_lo, pre_hi = [], [], []
    ft_vals,  ft_lo,  ft_hi  = [], [], []
    for obs in obs_order:
        v = get_value(df, args.pretrained_tag, obs)
        if v is None:
            pre_vals.append(np.nan); pre_lo.append(0); pre_hi.append(0)
        else:
            pre_vals.append(v[0]); pre_lo.append(v[0] - v[1]); pre_hi.append(v[2] - v[0])
        v = get_value(df, args.finetune_tag, obs)
        if v is None:
            ft_vals.append(np.nan); ft_lo.append(0); ft_hi.append(0)
        else:
            ft_vals.append(v[0]); ft_lo.append(v[0] - v[1]); ft_hi.append(v[2] - v[0])

    ax.bar(x - width/2, pre_vals,
           width=width, color=PRE_VS_FT["pretrained"],
           yerr=[pre_lo, pre_hi], capsize=2, ecolor="#222",
           label="Pretrained Sophon", edgecolor="black", linewidth=0.6, zorder=2)
    ax.bar(x + width/2, ft_vals,
           width=width, color=PRE_VS_FT["full_ft"],
           yerr=[ft_lo, ft_hi], capsize=2, ecolor="#222",
           label=args.finetune_label, edgecolor="black", linewidth=0.6, zorder=2)

    # Group dividers and labels
    ymax = max(np.nanmax(pre_vals), np.nanmax(ft_vals))
    for start, end, name in group_spans[1:]:
        ax.axvline(start - 0.5, color="black", lw=0.7, alpha=0.4)
    for start, end, name in group_spans:
        mid = (start + end - 1) / 2
        ax.text(mid, ymax * 1.08, name, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(obs_labels, rotation=45, ha="right")
    ax.set_ylabel(r"Linear probe $R^2$")
    ax.set_title("Probing the Sophon embedding for jet observables")
    ax.set_ylim(-0.05, max(ymax * 1.18, 1.0))
    ax.axhline(0, color="#888", lw=0.5)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="gray")

    save_fig(fig, args.output)
    print(f"Saved: {args.output}.{{pdf,png}}")
    plt.close()

    # Print headline contrast for the caption
    pre_mean = get_value(df, args.pretrained_tag, "mean_abs_d0")
    pre_max  = get_value(df, args.pretrained_tag, "max_abs_d0")
    if pre_mean and pre_max:
        print(f"\nHeadline: pretrained mean|d0| R^2 = {pre_mean[0]:.3f}, "
              f"max|d0| R^2 = {pre_max[0]:.3f}  (ratio {pre_max[0]/max(pre_mean[0],1e-3):.1f}x)")


if __name__ == "__main__":
    main()
