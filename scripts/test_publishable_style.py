#!/usr/bin/env python3
"""Sanity-check the publishable style with a line plot + scatter plot."""
from __future__ import annotations

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import (
    set_publishable_style, save_fig,
    OKABE_ITO, PALETTE, CLASS_COLORS, single_col, double_col,
)

set_publishable_style()


# ---- Line plot (single column, golden ratio) ----------------------------
fig, ax = plt.subplots(figsize=single_col())
x = np.linspace(0, 10, 100)
for i, freq in enumerate([0.5, 1.0, 1.5, 2.0]):
    ax.plot(x, np.sin(freq * x) / (freq + 0.5),
            label=fr"$f={freq}$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\sin(fx)/(f + 0.5)$")
ax.set_title("Damped sinusoid family")
ax.legend(loc="upper right")
save_fig(fig, "results/style_test_lines")
print("Saved: results/style_test_lines.{pdf,png}")
plt.close()


# ---- Scatter (double column) --------------------------------------------
fig, ax = plt.subplots(figsize=double_col(0.7))
rng = np.random.default_rng(0)
for name, color in CLASS_COLORS.items():
    cx, cy = rng.uniform(-3, 3, size=2)
    pts = rng.normal(loc=(cx, cy), scale=0.45, size=(100, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.55,
               color=color, label=name, edgecolors="none")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("Per-class palette (Okabe-Ito + neutrals)")
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.30),
          markerscale=1.6)
ax.set_xticks([]); ax.set_yticks([])
save_fig(fig, "results/style_test_scatter")
print("Saved: results/style_test_scatter.{pdf,png}")
plt.close()
