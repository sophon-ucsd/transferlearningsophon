"""Shared matplotlib style for poster figures.

Use:
    from plots.style import apply_style, OKABE_ITO, CLASS_COLORS, save_fig
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ...
    save_fig(fig, "results/arm3/arm3b_cluster_metrics")  # writes .pdf and .png at 300 dpi
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito 8-color palette (colorblind-safe, used widely in physics)
OKABE_ITO = {
    "black":         "#000000",
    "orange":        "#E69F00",
    "sky_blue":      "#56B4E9",
    "bluish_green":  "#009E73",
    "yellow":        "#F0E442",
    "blue":          "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple":"#CC79A7",
}
PALETTE = list(OKABE_ITO.values())

# 10-class jet color mapping. Order matches LABEL_NAMES across the codebase
# (QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl).
# QCD is gray (background); each class gets a distinct Okabe-Ito hue.
CLASS_COLORS = {
    "QCD":  "#666666",
    "Hbb":  OKABE_ITO["vermillion"],
    "Hcc":  OKABE_ITO["orange"],
    "Hgg":  OKABE_ITO["reddish_purple"],
    "H4q":  OKABE_ITO["blue"],
    "Hqql": OKABE_ITO["bluish_green"],
    "Zqq":  "#7E5109",   # darker brown
    "Wqq":  OKABE_ITO["sky_blue"],
    "Tbqq": "#1F77B4",
    "Tbl":  "#17A2B8",
}

# Strategy palette (frozen / partial / full / from_scratch)
STRATEGY_COLORS = {
    "frozen":       OKABE_ITO["sky_blue"],
    "partial_ft":   OKABE_ITO["bluish_green"],
    "full_ft":      OKABE_ITO["vermillion"],
    "from_scratch": OKABE_ITO["reddish_purple"],
}

# Pretrained vs fine-tuned 2-color set (used in cluster-metrics + probing bars)
PRE_VS_FT = {
    "pretrained": "#666666",        # neutral gray
    "full_ft":    OKABE_ITO["blue"], # blue for trained
}


def apply_style() -> None:
    """Apply the project-wide matplotlib style. Call once at the top of any plotting script."""
    mpl.rcParams.update({
        # Font
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.title_fontsize": 10,
        "figure.titlesize":  14,

        # Lines
        "lines.linewidth":   1.6,
        "lines.markersize":  6,

        # Axes
        "axes.linewidth":    1.0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.5,

        # Figure
        "figure.dpi":        100,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,

        # PDF / PS — embed fonts
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def save_fig(fig, base_path, formats=("pdf", "png"), dpi: int = 300) -> None:
    """Save a figure to multiple formats with consistent settings.

    Parameters
    ----------
    fig
        Matplotlib figure.
    base_path
        Path WITHOUT extension. Parent dir is created if missing.
    formats
        Iterable of extensions to write.
    dpi
        Resolution for raster outputs.
    """
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(f"{base}.{ext}", dpi=dpi, bbox_inches="tight")


# Standard figure sizes (width, height in inches) for poster context
FIG_SIZE = {
    "small":      (5.0, 3.5),   # single-panel inset
    "medium":     (7.0, 5.0),   # standard panel
    "wide":       (10.0, 4.5),  # 2-panel side-by-side base
    "wide_tall":  (10.0, 6.0),  # 2-panel taller
    "scaling":    (8.0, 5.5),   # scaling-curve plot
    "umap_pair":  (14.0, 6.5),  # paired UMAP
    "bar_panel":  (9.0, 4.5),   # grouped bar chart
}
