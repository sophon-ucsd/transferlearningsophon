#!/usr/bin/env python3
"""Arm 4 — held-out-class OOD detection via Mahalanobis distance.

Setup:
  Held-out classes: Tbl + Hqql (lepton-bearing). These are the labels
  Sophon "should" treat as anomalies w.r.t. the hadronic SM background.
  Training classes (8 hadronic): QCD, Hbb, Hcc, Hgg, H4q, Zqq, Wqq, Tbqq.

For each embedding type (Sophon 128-d vs 8-d HLF baseline):
  1. Compute 8 per-class centroids and a single shared covariance matrix from
     the training-class embeddings.
  2. For each test jet, compute Mahalanobis score:
        s(x) = -min_c (x - mu_c)^T Sigma^-1 (x - mu_c)
     (negative because higher = more anomalous)
  3. Compute ROC: score s as anomaly score, label = 1 if Tbl or Hqql.
  4. Report AUC.

Outputs:
  results/arm4/ood_results.json  — AUCs and per-class score histograms
  results/arm4/arm4_ood_roc.{pdf,png}

Usage:
    python scripts/ood_holdout_roc.py \\
        --pretrained-dir /data/embeddings_test_20M \\
        --observables /data/results/poster/observables_test_100k.parquet \\
        --output-dir /data/results/poster \\
        --n-per-class 10000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample
from plots.style import apply_style, save_fig, OKABE_ITO


LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
HELD_OUT = {"Hqql", "Tbl"}

HLF_FEATURES = ["jet_mass", "jet_pt", "multiplicity",
                "tau1", "tau2", "tau3", "tau21", "tau32"]


def fit_centroids_and_cov(emb: np.ndarray, lab: np.ndarray, training_classes: set[int],
                           ridge: float = 1e-3):
    """Compute per-class centroids and pooled covariance from training-class embeddings.

    Adds ridge*I*trace(Sigma)/d to keep the covariance well-conditioned.
    Returns (centroids dict[int -> ndarray], inv_cov ndarray).
    """
    train_mask = np.isin(lab, list(training_classes))
    Xtr = emb[train_mask]
    ytr = lab[train_mask]

    centroids: dict[int, np.ndarray] = {}
    Xc_list = []
    for c in sorted(training_classes):
        m = (ytr == c)
        if m.sum() == 0:
            continue
        mu = Xtr[m].mean(axis=0)
        centroids[c] = mu
        Xc_list.append(Xtr[m] - mu)
    Xc = np.concatenate(Xc_list, axis=0)
    cov = np.cov(Xc, rowvar=False)
    d = cov.shape[0]
    cov_reg = cov + ridge * (np.trace(cov) / d) * np.eye(d)
    inv_cov = np.linalg.inv(cov_reg)
    return centroids, inv_cov


def mahalanobis_score(emb: np.ndarray, centroids: dict[int, np.ndarray],
                      inv_cov: np.ndarray) -> np.ndarray:
    """Higher score => more anomalous.
    score(x) = min_c (x - mu_c)^T inv_cov (x - mu_c)
    The min Mahalanobis distance to any TRAINING-class centroid is itself the
    "how far from any seen class" measure: small for in-distribution, large for OOD.
    """
    n = len(emb)
    n_classes = len(centroids)
    distances = np.empty((n, n_classes), dtype=np.float64)
    for j, (_, mu) in enumerate(sorted(centroids.items())):
        diff = emb - mu  # (n, d)
        distances[:, j] = np.einsum("nd,de,ne->n", diff, inv_cov, diff)
    return distances.min(axis=1)  # higher = more anomalous


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-dir", required=True,
                   help="Directory of pretrained Sophon test embeddings")
    p.add_argument("--observables", required=True,
                   help="Parquet of per-jet observables (for HLF baseline)")
    p.add_argument("--output-dir", default="results/arm4")
    p.add_argument("--n-per-class", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Sophon embeddings, stratified subsample 10K/class -------------
    print(f"Loading Sophon embeddings from {args.pretrained_dir}")
    emb, lab = _load_dir(args.pretrained_dir)
    n_classes = len(np.unique(lab))
    target = args.n_per_class * n_classes
    idx = stratified_subsample(lab, min(target, len(lab)), args.seed)
    emb = emb[idx].astype(np.float64)
    lab = lab[idx]
    print(f"  Sophon: {len(lab):,} jets, dim={emb.shape[1]}")

    # --- Load HLF baseline from observables (use stratified subsample matching) -
    print(f"Loading HLF observables from {args.observables}")
    df = pd.read_parquet(args.observables)
    df_idx = stratified_subsample(df["label"].values, min(target, len(df)), args.seed)
    df_sub = df.iloc[df_idx].reset_index(drop=True)
    hlf = df_sub[HLF_FEATURES].values.astype(np.float64)
    hlf_lab = df_sub["label"].values
    # z-score the HLF features so Mahalanobis is well-conditioned
    hlf_mu = hlf.mean(axis=0)
    hlf_sd = hlf.std(axis=0) + 1e-9
    hlf_z = (hlf - hlf_mu) / hlf_sd
    print(f"  HLF: {len(hlf):,} jets, dim={hlf.shape[1]}")

    # --- Set up training (8 hadronic) and held-out (Tbl + Hqql) labels -------
    training_classes = {LABEL_NAMES.index(c) for c in LABEL_NAMES if c not in HELD_OUT}
    held_out_classes = {LABEL_NAMES.index(c) for c in HELD_OUT}
    print(f"\nTraining classes: {sorted(training_classes)}")
    print(f"Held-out classes: {sorted(held_out_classes)}")

    # Test-set labels: 1 if held-out else 0
    is_ood = np.isin(lab, list(held_out_classes)).astype(int)
    is_ood_hlf = np.isin(hlf_lab, list(held_out_classes)).astype(int)

    # --- Fit Mahalanobis from TRAINING-class subset only ---------------------
    print("\nFitting Sophon Mahalanobis...")
    cent_s, inv_s = fit_centroids_and_cov(emb, lab, training_classes)
    score_s = mahalanobis_score(emb, cent_s, inv_s)
    auc_s = float(roc_auc_score(is_ood, score_s))
    fpr_s, tpr_s, _ = roc_curve(is_ood, score_s)

    print("Fitting HLF Mahalanobis...")
    cent_h, inv_h = fit_centroids_and_cov(hlf_z, hlf_lab, training_classes)
    score_h = mahalanobis_score(hlf_z, cent_h, inv_h)
    auc_h = float(roc_auc_score(is_ood_hlf, score_h))
    fpr_h, tpr_h, _ = roc_curve(is_ood_hlf, score_h)

    print(f"\nSophon AUC: {auc_s:.4f}")
    print(f"HLF AUC:    {auc_h:.4f}")
    print(f"Gap:        {auc_s - auc_h:+.4f}")

    # --- Plot ROC ------------------------------------------------------------
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr_s, tpr_s, color=OKABE_ITO["blue"], lw=2.0,
            label=f"Sophon 128-d  (AUC = {auc_s:.3f})")
    ax.plot(fpr_h, tpr_h, color=OKABE_ITO["vermillion"], lw=2.0,
            label=f"HLF 8-feature  (AUC = {auc_h:.3f})")
    ax.plot([0, 1], [0, 1], color="#888", lw=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("False positive rate (hadronic accepted as anomaly)")
    ax.set_ylabel("True positive rate (Tbl + Hqql identified)")
    ax.set_title("Held-out-class OOD detection\n(8 hadronic vs Tbl+Hqql)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="gray")

    save_fig(fig, str(out_dir / "arm4_ood_roc"))
    print(f"Saved figure: {out_dir / 'arm4_ood_roc'}.{{pdf,png}}")
    plt.close()

    # Also produce per-class score histogram (Sophon)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(np.percentile(score_s, 1), np.percentile(score_s, 99), 60)
    for c in sorted(training_classes | held_out_classes):
        m = (lab == c)
        if m.sum() == 0:
            continue
        name = LABEL_NAMES[c]
        ax.hist(score_s[m], bins=bins, histtype="step", lw=1.4,
                density=True,
                color=OKABE_ITO["vermillion"] if c in held_out_classes else "#888888",
                label=f"{name}{' (OOD)' if c in held_out_classes else ''}",
                alpha=0.95 if c in held_out_classes else 0.5)
    ax.set_xlabel("Mahalanobis anomaly score (higher = more anomalous)")
    ax.set_ylabel("Density")
    ax.set_title("Sophon anomaly-score distribution per class")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    save_fig(fig, str(out_dir / "arm4_ood_per_class_score"))
    plt.close()

    # --- Save numbers --------------------------------------------------------
    out_json = out_dir / "arm4_ood_results.json"
    out = {
        "n_per_class": args.n_per_class,
        "training_classes": sorted([LABEL_NAMES[c] for c in training_classes]),
        "held_out_classes": sorted([LABEL_NAMES[c] for c in held_out_classes]),
        "sophon": {"auc": auc_s, "n_test_jets": int(len(score_s))},
        "hlf": {"auc": auc_h, "features": HLF_FEATURES, "n_test_jets": int(len(score_h))},
        "gap": auc_s - auc_h,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
