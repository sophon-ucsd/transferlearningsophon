#!/usr/bin/env python3
"""Figure 5 — Arm 1 scaling curves with power-law extrapolation.

Solid line + filled markers = measured. Dashed line + open markers = power-law
projection. Power law: AUC(N) = AUC_inf - A * N^(-alpha), fit per strategy on
the strategy's available data points only.

Reads:
    results/arm1/sweep_results.csv

Outputs:
    results/arm1/arm1_scaling.{pdf,png}
"""
from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import set_publishable_style, save_fig, STRATEGY_COLORS


PARTICLE_TRANSFORMER_AUC = 0.988
SIZES_ALL = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8]

STRATEGY_DISPLAY = {
    "frozen":     ("Frozen Sophon + MLP",  STRATEGY_COLORS["frozen"],     "o"),
    "partial_ft": ("Partial-FT (4 frozen)", STRATEGY_COLORS["partial_ft"], "s"),
    "full_ft":    ("Full FT",               STRATEGY_COLORS["full_ft"],   "D"),
}


def power_law(N, auc_inf, A, alpha):
    return auc_inf - A * np.power(N, -alpha)


def fit_strategy(sizes: np.ndarray, aucs: np.ndarray):
    """Fit AUC(N) = AUC_inf - A * N^(-alpha). Returns (auc_inf, A, alpha)."""
    p0 = (aucs.max() + 0.005, 0.5, 0.25)
    bounds = ([aucs.max(), 1e-4, 1e-3],
              [1.0,        50.0,  2.0])
    popt, _ = curve_fit(power_law, sizes, aucs, p0=p0, bounds=bounds, maxfev=20000)
    return popt


def main():
    set_publishable_style()
    df = pd.read_csv("results/arm1/sweep_results.csv")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    fits = {}
    for strategy, (label, color, marker) in STRATEGY_DISPLAY.items():
        sub = df[df.strategy == strategy]
        if not len(sub):
            continue
        agg = sub.groupby("train_size")["test_auc"].mean().reset_index()
        agg = agg.sort_values("train_size")
        x_meas = agg["train_size"].values.astype(float)
        y_meas = agg["test_auc"].values

        # --- Fit power law on measured points ---
        try:
            auc_inf, A, alpha = fit_strategy(x_meas, y_meas)
        except Exception as e:
            print(f"{strategy}: fit failed ({e})")
            auc_inf = A = alpha = None

        # --- Plot measured (solid + filled markers) ---
        ax.plot(x_meas, y_meas, color=color, lw=1.6, marker=marker,
                markersize=5, markeredgecolor="white", markeredgewidth=0.5,
                label=label, zorder=4)

        # --- Plot extrapolation (dashed line + open markers for projected sizes) ---
        if auc_inf is not None:
            projected_sizes = np.array([s for s in SIZES_ALL if s > x_meas.max()])
            if len(projected_sizes):
                # Dense dashed continuation curve
                x_dense = np.logspace(np.log10(x_meas.max()),
                                      np.log10(projected_sizes.max() * 1.05), 60)
                y_dense = power_law(x_dense, auc_inf, A, alpha)
                ax.plot(x_dense, y_dense, color=color, lw=1.4,
                        linestyle=(0, (5, 4)), alpha=0.85, zorder=3)
                # Open-marker projection points
                y_proj = power_law(projected_sizes, auc_inf, A, alpha)
                ax.plot(projected_sizes, y_proj, marker=marker, linestyle="",
                        markersize=5, markeredgecolor=color,
                        markerfacecolor="white", markeredgewidth=1.2, zorder=4)

            fits[strategy] = (auc_inf, A, alpha)

    # Published-ParT reference
    ax.axhline(PARTICLE_TRANSFORMER_AUC, color="#444", lw=1.0,
               linestyle=(0, (4, 3)), zorder=1)
    ax.text(1.0e8, PARTICLE_TRANSFORMER_AUC + 0.0010,
            r"published ParT (AUC $\approx$ 0.988)",
            ha="right", va="bottom", fontsize=9, color="#444")

    # Frozen+MLP callout at 100M
    df_frozen = df[df.strategy == "frozen"]
    if len(df_frozen):
        big_auc = float(df_frozen[df_frozen.train_size == 1e8]["test_auc"].mean())
        params = int(df_frozen["total_params"].iloc[0])
        ax.annotate(
            f"Frozen + MLP @ 100M\nAUC = {big_auc:.3f}, {params/1000:.0f}k params",
            xy=(1e8, big_auc),
            xytext=(2.5e6, big_auc - 0.012),
            ha="left", va="top",
            fontsize=9, color="#222",
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#666"),
            bbox=dict(facecolor="white", edgecolor="#bbb",
                      boxstyle="round,pad=0.3", alpha=0.9),
        )

    # Tiny inline note about projection (bottom-left)
    ax.text(0.02, 0.04,
            r"dashed = power-law projection $\mathrm{AUC}(N)=\mathrm{AUC}_\infty - A N^{-\alpha}$",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color="#666", style="italic")

    ax.set_xscale("log")
    ax.set_xlim(8e3, 1.5e8)
    ax.set_ylim(0.93, 0.995)
    ax.set_xlabel(r"Number of fine-tuning jets $N$")
    ax.set_ylabel(r"Macro AUC (10-class JetClass)")
    ax.set_title("Transfer-learning scaling: Sophon adaptations")
    ax.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels(["10K", "100K", "1M", "10M", "100M"])
    ax.minorticks_off()
    ax.legend(loc="lower right")
    ax.grid(True, which="major", alpha=1.0)

    save_fig(fig, "results/arm1/arm1_scaling")
    print("Saved: results/arm1/arm1_scaling.{pdf,png}")
    if fits:
        print("\nFitted power-law parameters (AUC_inf, A, alpha):")
        for s, (a_inf, A, al) in fits.items():
            print(f"  {s:>12}: AUC_inf={a_inf:.4f}  A={A:.3f}  alpha={al:.3f}")


if __name__ == "__main__":
    main()
