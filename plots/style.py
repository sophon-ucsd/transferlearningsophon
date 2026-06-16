"""Shared Matplotlib style for the poster figures.

One entry point: ``apply_style()``. Sets fonts (serif), colors (Okabe-Ito),
sizes (golden ratio at journal-column widths), gridlines, and saves PDFs as
TrueType so they edit cleanly in Illustrator.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


# Okabe-Ito 8-color colorblind-safe palette (Wong, Nature Methods 2011).
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

# Cycle order — black is poor for thin lines, so it goes last.
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

# Deterministic per-class colors used across all poster figures.
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

STRATEGY_COLORS = {
    "frozen":       OKABE_ITO["sky_blue"],
    "partial_ft":   OKABE_ITO["bluish_green"],
    "full_ft":      OKABE_ITO["vermillion"],
    "from_scratch": OKABE_ITO["reddish_purple"],
}

PRE_VS_FT = {
    "pretrained": OKABE_ITO["blue"],
    "full_ft":    OKABE_ITO["vermillion"],
}


GOLDEN = 1.61803398875


def single_col(scale: float = 1.0) -> tuple[float, float]:
    """3.5-inch single-column figure at golden-ratio height."""
    w = 3.5 * scale
    return (w, w / GOLDEN)


def double_col(scale: float = 1.0) -> tuple[float, float]:
    """7-inch double-column figure at golden-ratio height."""
    w = 7.0 * scale
    return (w, w / GOLDEN)


FIG_SIZE = {
    "single":      single_col(1.0),
    "single_tall": (3.5, 3.5),
    "double":      double_col(1.0),
    "double_tall": (7.0, 4.9),
    "double_wide": (10.0, 10.0 / GOLDEN),
}


DARK_GRAY  = "#333333"
GRID_GRAY  = "#E0E0E0"
WHITE      = "#FFFFFF"


def apply_style() -> None:
    """Set the Matplotlib rcParams used by every figure in this repo.

    Idempotent — safe to call repeatedly.
    """
    mpl.rcParams.update({
        # Serif font (CMU / STIX) with mathtext rendered in Computer Modern.
        "font.family":       "serif",
        "font.serif":        ["CMU Serif", "Computer Modern Roman",
                              "STIX Two Text", "STIXGeneral",
                              "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":  "cm",
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

        "lines.linewidth":   1.5,
        "lines.markersize":  5,
        "patch.linewidth":   0.8,

        "axes.prop_cycle":   mpl.cycler(color=PALETTE),

        "axes.facecolor":    WHITE,
        "axes.edgecolor":    DARK_GRAY,
        "axes.linewidth":    0.8,
        "axes.labelcolor":   DARK_GRAY,
        "axes.titlecolor":   DARK_GRAY,
        "axes.titleweight":  "regular",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.axisbelow":    True,

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

        "axes.grid":         True,
        "grid.color":        GRID_GRAY,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "grid.alpha":        1.0,

        "legend.frameon":    False,
        "legend.framealpha": 0.7,
        "legend.facecolor":  WHITE,
        "legend.edgecolor":  "none",
        "legend.borderpad":  0.4,
        "legend.handlelength": 1.6,

        "figure.facecolor":  WHITE,
        "figure.dpi":        100,
        "figure.figsize":    double_col(),

        "savefig.dpi":           300,
        "savefig.facecolor":     WHITE,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.05,

        # TrueType (Illustrator-editable) instead of Type 3.
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def save_fig(fig, base_path, formats=("pdf", "png"), dpi: int = 300) -> None:
    """Save ``fig`` as both .pdf and .png at the given DPI with a tight bbox."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(f"{base}.{ext}", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
