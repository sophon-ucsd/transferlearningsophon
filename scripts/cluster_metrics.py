#!/usr/bin/env python3
"""Arm 3.2 — quantitative cluster metrics for pretrained vs fine-tuned Sophon embeddings.

Computes on a stratified test subset (default 10K/class = 100K jets):
  1. Per-class silhouette score (sklearn.metrics.silhouette_samples)
  2. k-NN classification accuracy (n_neighbors=10) on raw 128-d embeddings
  3. Linear separability via multinomial LogisticRegression — per-class AUC

Outputs a 3-panel grouped bar chart comparing pretrained vs fine-tuned, and a JSON
with the raw numbers for poster captions.

Usage:
    python scripts/cluster_metrics.py \\
        --pretrained-dir /data/embeddings_test_20M \\
        --finetuned-dir /data/embeddings_ft_full_3M_seed42_test100k \\
        --finetune-label "Full FT, 3M jets, seed=42" \\
        --output-dir results/arm3 \\
        --n-per-class 10000
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

from sklearn.metrics import silhouette_samples, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from plots.style import apply_style, save_fig, PRE_VS_FT, FIG_SIZE, OKABE_ITO

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


def load_subsample(emb_dir: str, n_per_class: int, seed: int):
    emb, lab = _load_dir(emb_dir)
    n_classes = len(np.unique(lab))
    target = n_per_class * n_classes
    idx = stratified_subsample(lab, min(target, len(lab)), seed)
    return emb[idx].astype(np.float32), lab[idx]


def per_class_silhouette(emb, lab, sample_size=10000, seed=42):
    """Silhouette computed on a subsample (full set is O(N^2) memory)."""
    rng = np.random.RandomState(seed)
    if len(lab) > sample_size:
        sub = rng.choice(len(lab), size=sample_size, replace=False)
        emb_s = emb[sub]
        lab_s = lab[sub]
    else:
        emb_s = emb
        lab_s = lab
    sil = silhouette_samples(emb_s, lab_s, metric="euclidean")
    overall = float(sil.mean())
    per_class = {}
    for c in sorted(np.unique(lab_s)):
        per_class[LABEL_NAMES[c]] = float(sil[lab_s == c].mean())
    return overall, per_class


def knn_accuracy(emb, lab, k=10, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(emb, lab, test_size=0.2, random_state=seed, stratify=lab)
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(Xtr, ytr)
    overall = float(clf.score(Xte, yte))
    # per-class precision/recall
    yhat = clf.predict(Xte)
    per_class = {}
    for c in sorted(np.unique(yte)):
        mask = yte == c
        prec_mask = yhat == c
        recall = float((yhat[mask] == c).mean()) if mask.sum() else 0.0
        precision = float((yte[prec_mask] == c).mean()) if prec_mask.sum() else 0.0
        per_class[LABEL_NAMES[c]] = {"precision": precision, "recall": recall}
    return overall, per_class


def linear_separability(emb, lab, seed=42):
    """Multinomial LR -> per-class AUC (one-vs-rest)."""
    scaler = StandardScaler()
    emb_z = scaler.fit_transform(emb)
    Xtr, Xte, ytr, yte = train_test_split(emb_z, lab, test_size=0.2, random_state=seed, stratify=lab)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=-1, C=1.0)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    overall = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
    per_class = {}
    for i, c in enumerate(sorted(np.unique(yte))):
        binary = (yte == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        per_class[LABEL_NAMES[c]] = float(roc_auc_score(binary, proba[:, i]))
    return overall, per_class


def plot_summary(results, output_path: str, finetune_label: str):
    """3-panel grouped bar chart: silhouette, k-NN accuracy, linear-AUC."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    metrics = [
        ("silhouette", "Silhouette score",      "Mean per-class silhouette",      (-0.05, 0.55)),
        ("knn",        "k-NN accuracy (k=10)",  "Test accuracy on 80/20 split",   (0.0, 1.0)),
        ("linear",     "Linear AUC (LogReg)",   "Macro-averaged one-vs-rest AUC", (0.5, 1.0)),
    ]

    pre_color = PRE_VS_FT["pretrained"]
    ft_color = PRE_VS_FT["full_ft"]

    for ax, (key, title, ylabel, ylim) in zip(axes, metrics):
        pre_val = results["pretrained"][key]["overall"]
        ft_val  = results["finetuned"][key]["overall"]
        delta   = ft_val - pre_val

        ax.bar(["Pretrained", finetune_label],
               [pre_val, ft_val],
               color=[pre_color, ft_color], edgecolor="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

        # value labels
        for i, v in enumerate([pre_val, ft_val]):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=10, fontweight="bold")

        # delta annotation
        ax.annotate(
            f"Δ = {delta:+.3f}",
            xy=(1, ft_val), xytext=(0.5, max(pre_val, ft_val) * 1.05),
            ha="center", fontsize=9, color="#444",
        )

    fig.suptitle("Cluster geometry: pretrained Sophon vs fine-tuned", y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-dir", required=True)
    p.add_argument("--finetuned-dir", required=True)
    p.add_argument("--finetune-label", default="Full FT 3M, seed=42")
    p.add_argument("--output-dir", default="results/arm3")
    p.add_argument("--n-per-class", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained embeddings from {args.pretrained_dir}")
    pre_emb, pre_lab = load_subsample(args.pretrained_dir, args.n_per_class, args.seed)
    print(f"  {len(pre_lab):,} jets, dim={pre_emb.shape[1]}")

    print(f"Loading fine-tuned embeddings from {args.finetuned_dir}")
    ft_emb, ft_lab = load_subsample(args.finetuned_dir, args.n_per_class, args.seed)
    print(f"  {len(ft_lab):,} jets, dim={ft_emb.shape[1]}")

    results = {}
    for tag, emb, lab in [("pretrained", pre_emb, pre_lab),
                          ("finetuned", ft_emb, ft_lab)]:
        print(f"\n=== {tag} ===")
        sil_o, sil_c = per_class_silhouette(emb, lab, seed=args.seed)
        print(f"  silhouette (overall): {sil_o:.4f}")
        knn_o, knn_c = knn_accuracy(emb, lab, k=10, seed=args.seed)
        print(f"  knn accuracy:         {knn_o:.4f}")
        lin_o, lin_c = linear_separability(emb, lab, seed=args.seed)
        print(f"  linear AUC (macro):   {lin_o:.4f}")
        results[tag] = {
            "silhouette": {"overall": sil_o, "per_class": sil_c},
            "knn":        {"overall": knn_o, "per_class": knn_c},
            "linear":     {"overall": lin_o, "per_class": lin_c},
            "n_jets":     int(len(lab)),
        }

    # Save JSON
    json_path = out_dir / "arm3b_cluster_metrics.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_per_class": args.n_per_class,
            "seed": args.seed,
            "finetune_label": args.finetune_label,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Plot
    plot_path = out_dir / "arm3b_cluster_metrics"
    plot_summary(results, str(plot_path), args.finetune_label)
    print(f"Saved figure: {plot_path}.{{pdf,png}}")


if __name__ == "__main__":
    main()
