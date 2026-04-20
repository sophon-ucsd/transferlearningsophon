#!/usr/bin/env python3
"""Linear probing: what physics does Sophon's embedding encode?

Fits Ridge regression from 128-dim Sophon embeddings to jet substructure
observables. High R² = the embedding encodes that observable well.

Usage:
    python scripts/probe_substructure.py \
        --observables substructure_observables.npz \
        --embeddings-dir embeddings/ \
        --output probing_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

OBS_NAMES = ["jet_mass", "jet_pt", "multiplicity", "tau1", "tau2", "tau3",
             "tau21", "tau32", "C2", "width", "mean_d0", "mean_dz", "charged_frac"]


def load_embeddings_matched(embeddings_dir, n_jets, seed=42):
    """Load Sophon embeddings matched to the substructure observable order.

    The substructure observables were computed from stratified_subsample
    of val_5M with the given seed. We need the same jets' embeddings.

    For simplicity: load all val embeddings and subsample with same seed.
    """
    from src.data.embedding_dataset import _load_dir
    from src.data.subsampler import stratified_subsample

    emb, labels = _load_dir(embeddings_dir)
    indices = stratified_subsample(labels, n_jets, seed)
    return emb[indices].astype(np.float32), labels[indices]


def probe_one(embeddings, target, n_folds=5):
    """Fit Ridge regression and return cross-validated R²."""
    # Remove NaN/inf
    valid = np.isfinite(target)
    if valid.sum() < 100:
        return {"R2_mean": 0.0, "R2_std": 0.0, "n_valid": int(valid.sum())}

    X = StandardScaler().fit_transform(embeddings[valid])
    y = StandardScaler().fit_transform(target[valid].reshape(-1, 1)).ravel()

    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, X, y, cv=n_folds, scoring="r2")

    return {
        "R2_mean": float(np.mean(scores)),
        "R2_std": float(np.std(scores)),
        "n_valid": int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observables", required=True, help="Path to substructure_observables.npz")
    parser.add_argument("--embeddings-dir", required=True, help="Dir with Sophon embedding .npy or .csv files")
    parser.add_argument("--output", default="probing_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load observables
    data = np.load(args.observables)
    labels = data["labels"]
    n_jets = len(labels)
    print(f"Loaded {n_jets:,} jets with {len(OBS_NAMES)} observables")

    # Load matched embeddings
    print(f"Loading embeddings from {args.embeddings_dir}")
    embeddings, emb_labels = load_embeddings_matched(args.embeddings_dir, n_jets, args.seed)
    print(f"Embeddings shape: {embeddings.shape}")

    # Verify labels match
    if not np.array_equal(labels, emb_labels):
        print("WARNING: Labels don't match between observables and embeddings!")
        print(f"  Obs labels: {np.bincount(labels)}")
        print(f"  Emb labels: {np.bincount(emb_labels)}")

    # --- All-classes probing ---
    print("\n=== All classes ===")
    all_results = {}
    for obs_name in OBS_NAMES:
        target = data[obs_name]
        result = probe_one(embeddings, target)
        all_results[obs_name] = result
        print(f"  {obs_name:>15}: R² = {result['R2_mean']:.4f} ± {result['R2_std']:.4f}")

    # --- Per-class probing ---
    print("\n=== Per class ===")
    per_class = {}
    for cls_idx in range(10):
        cls_name = LABEL_NAMES[cls_idx]
        mask = labels == cls_idx
        if mask.sum() < 100:
            continue
        cls_emb = embeddings[mask]
        per_class[cls_name] = {}
        print(f"\n  {cls_name} ({mask.sum():,} jets):")
        for obs_name in OBS_NAMES:
            target = data[obs_name][mask]
            result = probe_one(cls_emb, target)
            per_class[cls_name][obs_name] = result
            print(f"    {obs_name:>15}: R² = {result['R2_mean']:.4f} ± {result['R2_std']:.4f}")

    # Save
    output = {
        "n_jets": n_jets,
        "embedding_dim": embeddings.shape[1],
        "all_classes": all_results,
        "per_class": per_class,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
