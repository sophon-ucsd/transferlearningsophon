#!/usr/bin/env python3
"""Embedding geometry analysis: how Sophon organizes jet classes.

Computes class centroids, pairwise distances, UMAP visualization,
and correlates embedding geometry with classification difficulty.

Usage:
    python scripts/embedding_geometry.py --embeddings-dir embeddings/ --output-dir figures/
    python scripts/embedding_geometry.py --embeddings-dir /data/embeddings_val_5M/ --output-dir figures/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

# Physics-ordered indices for the distance matrix
PHYSICS_ORDER = [
    1,  # Hbb
    2,  # Hcc
    3,  # Hgg
    4,  # H4q
    5,  # Hqql
    8,  # Tbqq
    9,  # Tbl
    7,  # Wqq
    6,  # Zqq
    0,  # QCD
]
PHYSICS_NAMES = [LABEL_NAMES[i] for i in PHYSICS_ORDER]

# Colors per class
CLASS_COLORS = {
    "QCD": "#888888",
    "Hbb": "#e41a1c",
    "Hcc": "#ff7f00",
    "Hgg": "#984ea3",
    "H4q": "#377eb8",
    "Hqql": "#4daf4a",
    "Zqq": "#a65628",
    "Wqq": "#f781bf",
    "Tbqq": "#999999",
    "Tbl": "#00ced1",
}


def load_embeddings(embeddings_dir, n_per_class=5000, seed=42):
    """Load embeddings, subsample to n_per_class per class."""
    emb, labels = _load_dir(embeddings_dir)
    total = n_per_class * len(np.unique(labels))
    idx = stratified_subsample(labels, total, seed)
    return emb[idx].astype(np.float32), labels[idx]


def compute_class_stats(embeddings, labels):
    """Compute per-class centroids and spreads."""
    classes = sorted(np.unique(labels))
    centroids = {}
    spreads = {}
    for cls in classes:
        mask = labels == cls
        cls_emb = embeddings[mask]
        centroids[cls] = cls_emb.mean(axis=0)
        spreads[cls] = cls_emb.std(axis=0).mean()  # mean std across dims
    return centroids, spreads


def compute_distance_matrix(centroids):
    """Compute 10x10 pairwise distance matrix between centroids."""
    classes = sorted(centroids.keys())
    n = len(classes)
    centroid_array = np.array([centroids[c] for c in classes])

    cos_dist = cosine_distances(centroid_array)
    euc_dist = euclidean_distances(centroid_array)

    return classes, cos_dist, euc_dist


def plot_umap(embeddings, labels, output, n_samples=10000, seed=42):
    """Figure 1: UMAP of embeddings colored by class."""
    try:
        import umap
    except ImportError:
        print("umap-learn not installed, skipping UMAP plot")
        print("Install with: pip install umap-learn")
        return

    # Subsample for UMAP
    idx = stratified_subsample(labels, min(n_samples, len(labels)), seed)
    emb_sub = embeddings[idx]
    lab_sub = labels[idx]

    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                        random_state=seed, n_components=2)
    coords = reducer.fit_transform(emb_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx in sorted(np.unique(lab_sub)):
        mask = lab_sub == cls_idx
        name = LABEL_NAMES[cls_idx]
        color = CLASS_COLORS.get(name, "#333333")
        ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.4,
                   color=color, label=name, rasterized=True)

    ax.legend(fontsize=9, markerscale=5, frameon=True, fancybox=False,
              edgecolor="gray", loc="best")
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Sophon Pretrained Embedding Space (UMAP)", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_distance_matrix(cos_dist, classes, output):
    """Figure 2: 10x10 cosine distance heatmap, physics-ordered."""
    # Reorder to physics groups
    idx_map = {c: i for i, c in enumerate(classes)}
    order = [idx_map[c] for c in PHYSICS_ORDER if c in idx_map]
    ordered_names = [LABEL_NAMES[classes[i]] for i in order]

    matrix = cos_dist[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")

    ax.set_xticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels(ordered_names, fontsize=9)

    # Annotate cells
    for i in range(len(ordered_names)):
        for j in range(len(ordered_names)):
            val = matrix[i, j]
            color = "white" if val > 0.4 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Cosine Distance", shrink=0.8)
    ax.set_title("Pairwise Cosine Distance Between Class Centroids", fontsize=13)

    # Add group dividers
    for pos in [5, 7, 9]:  # after Higgs, Top, Bosons
        if pos < len(ordered_names):
            ax.axhline(y=pos - 0.5, color="black", linewidth=1.5)
            ax.axvline(x=pos - 0.5, color="black", linewidth=1.5)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_geometry_vs_auc(cos_dist, classes, per_class_auc, output):
    """Figure 3: Nearest-class distance vs per-class AUC."""
    # Compute nearest-neighbor distance for each class
    nn_dist = {}
    for i, cls in enumerate(classes):
        distances = cos_dist[i].copy()
        distances[i] = np.inf  # exclude self
        nn_dist[cls] = np.min(distances)

    # Match with AUC
    x, y, names = [], [], []
    for cls in classes:
        cls_name = LABEL_NAMES[cls]
        if cls_name in per_class_auc:
            x.append(nn_dist[cls])
            y.append(per_class_auc[cls_name])
            names.append(cls_name)

    if len(x) < 3:
        print("Not enough per-class AUC data for geometry-AUC plot")
        return

    x, y = np.array(x), np.array(y)

    # Fit line
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(7, 5))
    for xi, yi, name in zip(x, y, names):
        color = CLASS_COLORS.get(name, "#333333")
        ax.scatter(xi, yi, s=80, color=color, zorder=3, edgecolor="black", linewidth=0.5)
        ax.annotate(name, (xi, yi), fontsize=8, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")

    # Fit line
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), "k--", alpha=0.5,
            label=f"R={r:.3f}, p={p:.3f}")

    ax.set_xlabel("Nearest-Class Cosine Distance", fontsize=12)
    ax.set_ylabel("Per-Class AUC (Frozen, 100K jets)", fontsize=12)
    ax.set_title("Embedding Geometry Predicts Classification Difficulty", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--output-dir", default="figures/")
    parser.add_argument("--n-per-class", type=int, default=5000)
    parser.add_argument("--per-class-auc-json", default=None,
                        help="JSON with per-class AUC for geometry-AUC plot")
    args = parser.parse_args()

    out = Path(args.output_dir)

    # Load
    print(f"Loading embeddings from {args.embeddings_dir}")
    embeddings, labels = load_embeddings(args.embeddings_dir, args.n_per_class)
    print(f"Loaded: {len(labels):,} jets, {len(np.unique(labels))} classes")

    # Class stats
    centroids, spreads = compute_class_stats(embeddings, labels)
    print("\nClass spreads (mean std across dims):")
    for cls in sorted(centroids.keys()):
        print(f"  {LABEL_NAMES[cls]:>8}: spread={spreads[cls]:.4f}")

    # Distance matrix
    classes, cos_dist, euc_dist = compute_distance_matrix(centroids)

    # Find closest and farthest pairs
    n = len(classes)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((cos_dist[i, j], LABEL_NAMES[classes[i]], LABEL_NAMES[classes[j]]))
    pairs.sort()
    print(f"\nClosest pairs (cosine distance):")
    for dist, a, b in pairs[:5]:
        print(f"  {a} - {b}: {dist:.4f}")
    print(f"\nFarthest pairs:")
    for dist, a, b in pairs[-3:]:
        print(f"  {a} - {b}: {dist:.4f}")

    # Save stats
    stats = {
        "n_jets": len(labels),
        "n_per_class": args.n_per_class,
        "spreads": {LABEL_NAMES[k]: float(v) for k, v in spreads.items()},
        "closest_pairs": [{"a": a, "b": b, "cosine_dist": float(d)} for d, a, b in pairs[:5]],
        "farthest_pairs": [{"a": a, "b": b, "cosine_dist": float(d)} for d, a, b in pairs[-3:]],
    }
    stats_path = out / "embedding_geometry.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Plots
    plot_umap(embeddings, labels, str(out / "umap_embeddings.pdf"))
    plot_distance_matrix(cos_dist, classes, str(out / "distance_matrix.pdf"))

    # Geometry vs AUC (if data available)
    if args.per_class_auc_json:
        with open(args.per_class_auc_json) as f:
            per_class_auc = json.load(f)
        plot_geometry_vs_auc(cos_dist, classes, per_class_auc, str(out / "geometry_vs_auc.pdf"))

    print(f"\nAll outputs in {out}/")


if __name__ == "__main__":
    main()
