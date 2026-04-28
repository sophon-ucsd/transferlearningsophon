#!/usr/bin/env python3
"""Figure 3 — Arm 3.1 paired UMAP (pretrained vs full-FT-3M-seed42).

Axis-locked panels, class centroids overlaid + labeled, silhouette annotation.

Reads embeddings from PVC (or local), saves UMAP coords to a cache .npz so
subsequent layout edits don't repeat the slow fit.

Outputs:
    results/arm3/arm3a_umap.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample
from plots.style import set_publishable_style, save_fig, CLASS_COLORS, double_col

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql",
               "Zqq", "Wqq", "Tbqq", "Tbl"]


def fit_umap(emb: np.ndarray, seed: int = 42, n_neighbors: int = 30,
             min_dist: float = 0.30) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric="cosine", random_state=seed,
                        n_components=2, verbose=False)
    return reducer.fit_transform(emb)


def load_subsample(emb_dir: str, n_per_class: int, seed: int):
    emb, lab = _load_dir(emb_dir)
    n_classes = len(np.unique(lab))
    target = n_per_class * n_classes
    idx = stratified_subsample(lab, min(target, len(lab)), seed)
    return emb[idx].astype(np.float32), lab[idx]


def silhouette_overall(emb: np.ndarray, lab: np.ndarray, n: int = 10000,
                       seed: int = 42) -> float:
    from sklearn.metrics import silhouette_samples
    rng = np.random.default_rng(seed)
    if len(lab) > n:
        sub = rng.choice(len(lab), size=n, replace=False)
        emb, lab = emb[sub], lab[sub]
    return float(silhouette_samples(emb, lab, metric="euclidean").mean())


def panel(ax, coords: np.ndarray, lab: np.ndarray, title: str,
          xlim: tuple[float, float], ylim: tuple[float, float],
          subtitle: str = ""):
    for cls_idx in sorted(np.unique(lab)):
        m = (lab == cls_idx)
        name = LABEL_NAMES[cls_idx]
        color = CLASS_COLORS[name]
        ax.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.40,
                   color=color, edgecolors="none", rasterized=True)

    # Class centroids — larger filled markers labeled with class abbreviations
    for cls_idx in sorted(np.unique(lab)):
        m = (lab == cls_idx)
        cx, cy = coords[m, 0].mean(), coords[m, 1].mean()
        name = LABEL_NAMES[cls_idx]
        ax.scatter([cx], [cy], s=120, color=CLASS_COLORS[name],
                   edgecolor="black", linewidth=0.8, zorder=4)
        ax.annotate(name, (cx, cy), xytext=(0, 9),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#222",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=1.0),
                    zorder=5)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title, pad=10)

    if subtitle:
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes,
                ha="left", va="top", fontsize=10, color="#444",
                bbox=dict(facecolor="white", edgecolor="#bbb",
                          alpha=0.9, boxstyle="round,pad=0.3"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-dir", default="/data/embeddings_test_20M")
    p.add_argument("--finetuned-dir",
                   default="/data/embeddings_ft_full_3M_seed42_test100k")
    p.add_argument("--n-per-class", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache",
                   default="/data/results/poster/umap_coords_cache.npz")
    p.add_argument("--output",
                   default="/data/results/poster/arm3a_umap")
    p.add_argument("--cluster-metrics-json",
                   default="/data/results/poster/arm3b_cluster_metrics.json",
                   help="If present, read silhouette numbers from here.")
    args = p.parse_args()

    set_publishable_style()

    cache = Path(args.cache)
    if cache.exists():
        print(f"Using cached UMAP coords from {cache}")
        z = np.load(cache)
        pre_coords = z["pre_coords"]; pre_lab = z["pre_lab"]
        ft_coords  = z["ft_coords"];  ft_lab  = z["ft_lab"]
        sil_pre = float(z["sil_pre"]); sil_ft = float(z["sil_ft"])
    else:
        print(f"Loading pretrained from {args.pretrained_dir}")
        pre_emb, pre_lab = load_subsample(args.pretrained_dir,
                                          args.n_per_class, args.seed)
        print(f"  {len(pre_lab):,} jets")
        print(f"Loading fine-tuned from {args.finetuned_dir}")
        ft_emb,  ft_lab  = load_subsample(args.finetuned_dir,
                                          args.n_per_class, args.seed)
        print(f"  {len(ft_lab):,} jets")

        # Read silhouette from cluster-metrics JSON if available; else compute
        cm_path = Path(args.cluster_metrics_json)
        if cm_path.exists():
            cm = json.loads(cm_path.read_text())
            sil_pre = float(cm["results"]["pretrained"]["silhouette"]["overall"])
            sil_ft  = float(cm["results"]["finetuned"]["silhouette"]["overall"])
        else:
            sil_pre = silhouette_overall(pre_emb, pre_lab)
            sil_ft  = silhouette_overall(ft_emb,  ft_lab)
        print(f"Silhouette: pre={sil_pre:.3f}  ft={sil_ft:.3f}")

        print("Running UMAP on pretrained...")
        pre_coords = fit_umap(pre_emb, seed=args.seed)
        print("Running UMAP on fine-tuned...")
        ft_coords  = fit_umap(ft_emb,  seed=args.seed)

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache,
                 pre_coords=pre_coords, pre_lab=pre_lab,
                 ft_coords=ft_coords, ft_lab=ft_lab,
                 sil_pre=sil_pre, sil_ft=sil_ft)
        print(f"Cached coords -> {cache}")

    # Lock axes to a common range covering BOTH panels
    pad = 1.0
    xlim = (min(pre_coords[:, 0].min(), ft_coords[:, 0].min()) - pad,
            max(pre_coords[:, 0].max(), ft_coords[:, 0].max()) + pad)
    ylim = (min(pre_coords[:, 1].min(), ft_coords[:, 1].min()) - pad,
            max(pre_coords[:, 1].max(), ft_coords[:, 1].max()) + pad)

    # Slight scaling above golden ratio for the paired-UMAP layout
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6),
                             gridspec_kw=dict(wspace=0.06))
    panel(axes[0], pre_coords, pre_lab, "Pretrained Sophon",
          xlim, ylim, subtitle=f"silhouette = {sil_pre:.3f}")
    panel(axes[1], ft_coords,  ft_lab,  "Full-FT 3M (seed 42)",
          xlim, ylim, subtitle=f"silhouette = {sil_ft:.3f}")

    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.07)
    save_fig(fig, args.output)
    print(f"Saved {args.output}.{{pdf,png}}")


if __name__ == "__main__":
    main()
