#!/usr/bin/env python3
"""Figure 1 — Arm 3.3 probing R² (clean two-panel, vertical bars)."""
from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import apply_style, save_fig, OKABE_ITO


# Compact selection: 12 observables, ordered by physics group.
# Order is left-to-right: substructure → shape → d0 magnitude → d0 significance.
# A bigger gap separates each group; no group labels (caption explains).
ORDER = [
    # substructure
    ("multiplicity", r"multiplicity"),
    ("n_charged",    r"$n_{\rm ch}$"),
    ("jet_mass",     r"jet mass"),
    ("width",        r"jet width"),
    None,  # gap
    # shape
    ("tau21", r"$\tau_{21}$"),
    ("tau32", r"$\tau_{32}$"),
    ("C2",    r"$C_2$"),
    None,  # gap
    # displacement magnitude
    ("mean_abs_d0",     r"mean$|d_0|$"),
    ("max_abs_d0",      r"max$|d_0|$"),
    ("top3_sum_abs_d0", r"top-3 $|d_0|$"),
    None,  # gap
    # displacement significance
    ("count_d0_gt_1sigma", r"#$\,d_0/\sigma{>}1$"),
    ("count_d0_gt_2sigma", r"#$\,d_0/\sigma{>}2$"),
]
HIGHLIGHT = "count_d0_gt_2sigma"

# Per-strategy colors (consistent across both panels)
COLOR_FROZEN  = OKABE_ITO["blue"]          # Frozen Sophon == pretrained embedding
COLOR_PARTIAL = OKABE_ITO["bluish_green"]  # partial FT
COLOR_FULL    = OKABE_ITO["vermillion"]    # full FT


def fetch(df, model, obs):
    s = df[(df.model == model) & (df.observable == obs)]
    if not len(s):
        return None
    r = s.iloc[0]
    return float(r["R2_mean"]), float(r["R2_lo"]), float(r["R2_hi"])


def draw_panel(ax, df_pre_full, df_partial, title: str):
    """3 bars per observable: Frozen Sophon (=pretrained), Partial-FT, Full-FT.

    df_pre_full has model tags ["pretrained", "full_ft_3M_seed42"].
    df_partial  has model tags ["pretrained", "partial_ft_3M_seed42"]; we read
    only the partial column from this one.
    """
    GAP = 0.7
    pos = []; labels = []
    cur = 0.0
    for entry in ORDER:
        if entry is None:
            cur += GAP
            continue
        pos.append(cur)
        labels.append(entry)
        cur += 1.0
    pos = np.array(pos)
    width = 0.27   # 3 bars per group

    frozen_v, frozen_lo, frozen_hi = [], [], []
    part_v,   part_lo,   part_hi   = [], [], []
    full_v,   full_lo,   full_hi   = [], [], []
    h_idx = -1
    for i, (key, _) in enumerate(labels):
        # Frozen Sophon embedding == pretrained embedding
        v = fetch(df_pre_full, "pretrained", key)
        if v is None:
            frozen_v.append(np.nan); frozen_lo.append(0); frozen_hi.append(0)
        else:
            r, lo, hi = v
            frozen_v.append(r); frozen_lo.append(max(0, r - lo)); frozen_hi.append(max(0, hi - r))

        # Partial-FT (may be missing if df_partial is None — gracefully skip)
        if df_partial is not None:
            v = fetch(df_partial, "partial_ft_3M_seed42", key)
        else:
            v = None
        if v is None:
            part_v.append(np.nan); part_lo.append(0); part_hi.append(0)
        else:
            r, lo, hi = v
            part_v.append(r); part_lo.append(max(0, r - lo)); part_hi.append(max(0, hi - r))

        # Full-FT
        v = fetch(df_pre_full, "full_ft_3M_seed42", key)
        if v is None:
            full_v.append(np.nan); full_lo.append(0); full_hi.append(0)
        else:
            r, lo, hi = v
            full_v.append(r); full_lo.append(max(0, r - lo)); full_hi.append(max(0, hi - r))

        if key == HIGHLIGHT:
            h_idx = i

    ax.bar(pos - width, frozen_v, width=width,
           color=COLOR_FROZEN, linewidth=0,
           yerr=[frozen_lo, frozen_hi], capsize=2.0, ecolor="black",
           error_kw=dict(elinewidth=0.8),
           label="Frozen Sophon", zorder=2)
    ax.bar(pos,         part_v,   width=width,
           color=COLOR_PARTIAL, linewidth=0,
           yerr=[part_lo, part_hi], capsize=2.0, ecolor="black",
           error_kw=dict(elinewidth=0.8),
           label="Partial-FT 3M", zorder=2)
    ax.bar(pos + width, full_v,   width=width,
           color=COLOR_FULL, linewidth=0,
           yerr=[full_lo, full_hi], capsize=2.0, ecolor="black",
           error_kw=dict(elinewidth=0.8),
           label="Full-FT 3M", zorder=2)

    ax.set_xticks(pos)
    ax.set_xticklabels([lbl for _, lbl in labels],
                       rotation=35, ha="right", fontsize=11)
    if h_idx >= 0:
        for tl in ax.get_xticklabels():
            if tl.get_text() == labels[h_idx][1]:
                tl.set_fontweight("bold")

    ax.set_xlim(-0.7, pos[-1] + 0.7)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel(r"Probe $R^2$")
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.tick_params(axis="x", which="major", pad=2)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0)


def _load_csv_or_none(path: str):
    p = os.path.join(os.getcwd(), path)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    apply_style()
    df_ridge          = pd.read_csv("results/arm3/probing_results.csv")
    df_ridge_partial  = _load_csv_or_none("results/arm3/probing_partial_results.csv")
    df_mlp            = pd.read_csv("results/arm3/mlp_probing_results.csv")
    df_mlp_partial    = _load_csv_or_none("results/arm3/mlp_probing_partial_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0),
                             gridspec_kw=dict(wspace=0.18))
    draw_panel(axes[0], df_ridge, df_ridge_partial, r"Linear (ridge) probe")
    draw_panel(axes[1], df_mlp,   df_mlp_partial,   r"Nonlinear (MLP) probe")

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.01))

    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18)
    save_fig(fig, "results/arm3/arm3c_probing")
    print("Saved: results/arm3/arm3c_probing.{pdf,png}")
    if df_mlp_partial is None:
        print("  NOTE: partial-FT MLP probe CSV not found yet; "
              "MLP panel shows partial as gaps. Re-run after "
              "mlp-probe-partial-raunav completes.")


if __name__ == "__main__":
    main()
