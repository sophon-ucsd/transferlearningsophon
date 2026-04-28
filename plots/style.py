"""Publication-ready plotting style.

Single entry point: ``set_publishable_style()`` (alias: ``apply_style()``).

Spec:
    - DPI 300 for saved figures
    - Single-column width 3.5 in, double-column 7 in (academic journal standard);
      height = width / golden ratio (~1.618)
    - Sans-serif font (Helvetica / Arial / DejaVu Sans)
    - Tick labels 10 pt, axis labels + legend 11 pt, titles 12 pt
    - Okabe-Ito 8-color colorblind-safe default cycle
    - Top + right spines removed; remaining spines/ticks dark gray (#333)
    - Pure white background; subtle dashed gridlines (#E0E0E0) behind data
    - Frameless legend with semi-transparent background

Typical use::

    from plots.style import (
        set_publishable_style, save_fig, single_col, double_col,
        OKABE_ITO, CLASS_COLORS,
    )
    set_publishable_style()
    fig, ax = plt.subplots(figsize=double_col())
    ...
    save_fig(fig, "results/arm3/foo")  # writes .pdf and .png at 300 dpi
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# Okabe-Ito 8-color colorblind-safe palette (Nature/Wong 2011).
OKABE_ITO = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}
# Cycle order — black is poor for thin lines, so it goes last
PALETTE = [
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]

# Deterministic per-class colors used across all poster figures
CLASS_COLORS = {
    "QCD":  "#666666",
    "Hbb":  OKABE_ITO["vermillion"],
    "Hcc":  OKABE_ITO["orange"],
    "Hgg":  OKABE_ITO["reddish_purple"],
    "H4q":  OKABE_ITO["blue"],
    "Hqql": OKABE_ITO["bluish_green"],
    "Zqq":  "#7E5109",
    "Wqq":  OKABE_ITO["sky_blue"],
    "Tbqq": "#1F77B4",
    "Tbl":  "#17A2B8",
}

# Strategy palette for scaling-curve / Pareto plots
STRATEGY_COLORS = {
    "frozen":       OKABE_ITO["sky_blue"],
    "partial_ft":   OKABE_ITO["bluish_green"],
    "full_ft":      OKABE_ITO["vermillion"],
    "from_scratch": OKABE_ITO["reddish_purple"],
}

# Two-color Pretrained / Full-FT comparison
PRE_VS_FT = {
    "pretrained": OKABE_ITO["blue"],
    "full_ft":    OKABE_ITO["vermillion"],
}


# ---------------------------------------------------------------------------
# Sizing — golden-ratio heights at journal-column widths
# ---------------------------------------------------------------------------

GOLDEN = 1.61803398875

def single_col(scale: float = 1.0) -> tuple[float, float]:
    """3.5-inch single-column figure, golden-ratio height. ``scale`` multiplies both."""
    w = 3.5 * scale
    return (w, w / GOLDEN)

def double_col(scale: float = 1.0) -> tuple[float, float]:
    """7-inch double-column figure, golden-ratio height."""
    w = 7.0 * scale
    return (w, w / GOLDEN)

# Convenience size aliases for non-golden layouts when needed
FIG_SIZE = {
    "single":      single_col(1.0),
    "single_tall": (3.5, 3.5 * 1.0),
    "double":      double_col(1.0),
    "double_tall": (7.0, 7.0 * 0.7),     # taller than golden for grouped plots
    "double_wide": (10.0, 10.0 / GOLDEN), # extra wide if needed
}


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

DARK_GRAY  = "#333333"
GRID_GRAY  = "#E0E0E0"
WHITE      = "#FFFFFF"


def set_publishable_style() -> None:
    """Apply the publication-ready Matplotlib style globally.

    Uses Computer Modern / STIX serif fonts so output mimics LaTeX without
    requiring a LaTeX install (``text.usetex`` stays off for portability).

    Idempotent — safe to call repeatedly.
    """
    mpl.rcParams.update({
        # ---- Font: serif, LaTeX-flavored ----
        "font.family":       "serif",
        # CMU Serif is the LaTeX default; fall back to STIX (free Times) and Times.
        "font.serif":        ["CMU Serif", "Computer Modern Roman",
                              "STIX Two Text", "STIXGeneral",
                              "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":  "cm",            # Computer Modern math
        "mathtext.rm":       "serif",
        "mathtext.it":       "serif:italic",
        "mathtext.bf":       "serif:bold",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   11,
        "legend.title_fontsize": 11,
        "figure.titlesize":  12,

        # ---- Lines / markers ----
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
        "patch.linewidth":   0.8,            # bar borders, etc

        # ---- Color cycle ----
        "axes.prop_cycle":   mpl.cycler(color=PALETTE),

        # ---- Axes / spines (clean look) ----
        "axes.facecolor":    WHITE,
        "axes.edgecolor":    DARK_GRAY,
        "axes.linewidth":    0.8,
        "axes.labelcolor":   DARK_GRAY,
        "axes.titlecolor":   DARK_GRAY,
        "axes.titleweight":  "regular",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.axisbelow":    True,            # grid behind data

        # ---- Ticks ----
        "xtick.color":       DARK_GRAY,
        "ytick.color":       DARK_GRAY,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size":  2.0,
        "ytick.minor.size":  2.0,

        # ---- Grid ----
        "axes.grid":         True,
        "grid.color":        GRID_GRAY,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "grid.alpha":        1.0,             # color carries the subtlety

        # ---- Legend ----
        "legend.frameon":    False,
        "legend.framealpha": 0.7,
        "legend.facecolor":  WHITE,
        "legend.edgecolor":  "none",
        "legend.borderpad":  0.4,
        "legend.handlelength": 1.6,

        # ---- Figure ----
        "figure.facecolor":  WHITE,
        "figure.dpi":        100,
        "figure.figsize":    double_col(),

        # ---- Saving ----
        "savefig.dpi":           300,
        "savefig.facecolor":     WHITE,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.05,

        # ---- Editable PDFs (TrueType, not Type 3) ----
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


# Backwards-compatible alias
apply_style = set_publishable_style


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_fig(fig, base_path, formats=("pdf", "png"), dpi: int = 300) -> None:
    """Save a figure to .pdf and .png at the given DPI with tight bbox.

    The parent directory is created automatically.
    """
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(f"{base}.{ext}", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
