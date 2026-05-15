#!/usr/bin/env python3
"""Re-render the 3-panel ROC figure from a cached _curves.npz.

Usage:
    python scripts/render_roc_from_cache.py \\
        --cache results/main_plots/roc_3strategies_10M_curves.npz \\
        --size-label "10M" \\
        --output results/main_plots/roc_3strategies_10M
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import set_publishable_style, save_fig, CLASS_COLORS

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql",
               "Zqq", "Wqq", "Tbqq", "Tbl"]


def load_strategy(z, tag):
    out = {}
    for name in LABEL_NAMES:
        fkey = f"{tag}__{name}__fpr"
        if fkey not in z.files:
            continue
        out[name] = (z[f"{tag}__{name}__fpr"], z[f"{tag}__{name}__tpr"],
                     float(z[f"{tag}__{name}__auc"][0]))
    return out


def draw_panel(ax, curves, macro, title, show_ylabel=True, show_legend=False):
    for name, (fpr, tpr, auc) in curves.items():
        fpr_safe = np.clip(fpr, 1e-4, 1.0)
        ax.plot(fpr_safe, tpr, color=CLASS_COLORS[name], lw=1.3,
                label=f"{name}  ({auc:.3f})")
    ax.plot([1e-4, 1], [1e-4, 1], color="#888", lw=0.7,
            linestyle=(0, (4, 3)), zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    if show_ylabel:
        ax.set_ylabel("True positive rate")
    ax.set_title(f"{title}\nmacro AUC = {macro:.3f}", fontsize=11, pad=8)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, ncol=1,
                  title="class (AUC)", title_fontsize=9,
                  handlelength=1.2, handletextpad=0.4, borderpad=0.4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--size-label", required=True,
                   help="e.g. '10M' or '3M'")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    set_publishable_style()
    z = np.load(args.cache)

    curves_frozen = load_strategy(z, "frozen")
    curves_partial = load_strategy(z, "partial")
    curves_full = load_strategy(z, "full")
    macro_frozen, macro_partial, macro_full = z["macro"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8),
                             gridspec_kw=dict(wspace=0.18))
    sz = args.size_label
    draw_panel(axes[0], curves_frozen,  macro_frozen,
               f"Frozen Sophon + MLP ({sz})", show_ylabel=True)
    draw_panel(axes[1], curves_partial, macro_partial,
               f"Partial FT ({sz}, last 4 of 8 blocks)", show_ylabel=False)
    draw_panel(axes[2], curves_full,    macro_full,
               f"Full FT ({sz})", show_ylabel=False, show_legend=True)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.13)
    save_fig(fig, args.output)
    print(f"Saved: {args.output}.{{pdf,png}}")


if __name__ == "__main__":
    main()
