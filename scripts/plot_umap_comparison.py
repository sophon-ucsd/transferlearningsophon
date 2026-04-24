#!/usr/bin/env python3
"""Side-by-side UMAP: pretrained Sophon vs fine-tuned embedding space.

Loads both embedding directories, runs UMAP on each (same hyperparams, same seed,
same per-class subsample), and plots 1x2 with a shared legend.

Usage:
    python scripts/plot_umap_comparison.py \
        --pretrained-dir /data/embeddings_val_5M \
        --finetuned-dir /data/embeddings_ft_full_3M_seed42 \
        --finetune-label "Full FT, 3M jets" \
        --output /data/results/figures/umap_compare_full_ft_3M.pdf \
        --n-per-class 3000
"""
from __future__ import annotations

import argparse
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

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

CLASS_COLORS = {
    "QCD":  "#888888", "Hbb":  "#e41a1c", "Hcc":  "#ff7f00",
    "Hgg":  "#984ea3", "H4q":  "#377eb8", "Hqql": "#4daf4a",
    "Zqq":  "#a65628", "Wqq":  "#f781bf", "Tbqq": "#2b8cbe",
    "Tbl":  "#00ced1",
}


def load_subsampled(emb_dir: str, n_per_class: int, seed: int):
    emb, lab = _load_dir(emb_dir)
    total = n_per_class * len(np.unique(lab))
    idx = stratified_subsample(lab, min(total, len(lab)), seed)
    return emb[idx].astype(np.float32), lab[idx]


def run_umap(emb, seed):
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                        random_state=seed, n_components=2)
    return reducer.fit_transform(emb)


def plot_panel(ax, coords, labels, title):
    for cls_idx in sorted(np.unique(labels)):
        mask = labels == cls_idx
        name = LABEL_NAMES[cls_idx]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.4,
                   color=CLASS_COLORS[name], label=name, rasterized=True)
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-dir", required=True)
    p.add_argument("--finetuned-dir", required=True)
    p.add_argument("--finetune-label", required=True,
                   help="e.g. 'Full FT, 3M jets, seed=42'")
    p.add_argument("--output", required=True)
    p.add_argument("--n-per-class", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("Loading pretrained...")
    pre_emb, pre_lab = load_subsampled(args.pretrained_dir, args.n_per_class, args.seed)
    print(f"  {len(pre_lab):,} embeddings")

    print("Loading fine-tuned...")
    ft_emb, ft_lab = load_subsampled(args.finetuned_dir, args.n_per_class, args.seed)
    print(f"  {len(ft_lab):,} embeddings")

    print("Running UMAP on pretrained...")
    pre_coords = run_umap(pre_emb, args.seed)
    print("Running UMAP on fine-tuned...")
    ft_coords = run_umap(ft_emb, args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_panel(axes[0], pre_coords, pre_lab,
               "Pretrained Sophon\n(frozen backbone, JetClass-II 188-class pretrain)")
    plot_panel(axes[1], ft_coords, ft_lab,
               f"Fine-Tuned Sophon\n({args.finetune_label})")

    # Shared legend on the right
    handles, labels_l = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_l, loc="center right", fontsize=10,
               markerscale=5, frameon=True, fancybox=False, edgecolor="gray",
               bbox_to_anchor=(1.02, 0.5))

    fig.suptitle("Embedding Space: Pretrained vs Fine-Tuned", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.93, 0.96])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
