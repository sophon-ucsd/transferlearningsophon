#!/usr/bin/env python3
"""Compute substructure + probe embeddings with guaranteed alignment.

Processes ROOT files and embeddings class-by-class in the same order.
No subsampling mismatch possible.

Usage:
    python scripts/probe_matched.py \
        --root-dir /data/JetClass/Pythia/val_5M \
        --emb-dir /data/embeddings_val_5M \
        --output /data/results/probing_results.json \
        --n-per-class 5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import uproot
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.compute_substructure import compute_observables_for_jet

TREE_NAME = "tree"
PARTICLE_KEYS = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_d0val", "part_dzval", "part_charge",
]
LABEL_KEYS = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
    "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
    "label_Tbqq", "label_Tbl",
]
LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

# Embedding file class name -> label index (same as NPY_CLASS_MAP)
EMB_CLASS_MAP = {
    "HToBB": 1, "HToCC": 2, "HToGG": 3, "HToWW4Q": 4, "HToWW2Q1L": 5,
    "TTBar": 8, "TTBarLep": 9, "WToQQ": 7, "ZToQQ": 6, "ZJetsToNuNu": 0,
}

OBS_NAMES = ["jet_mass", "jet_pt", "multiplicity", "tau1", "tau2", "tau3",
             "tau21", "tau32", "C2", "width", "mean_d0", "mean_dz", "charged_frac"]


def probe_one(embeddings, target, n_folds=5):
    valid = np.isfinite(target) & np.isfinite(embeddings).all(axis=1)
    if valid.sum() < 100:
        return {"R2_mean": 0.0, "R2_std": 0.0, "n_valid": int(valid.sum())}
    X = StandardScaler().fit_transform(embeddings[valid])
    y = StandardScaler().fit_transform(target[valid].reshape(-1, 1)).ravel()
    scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=n_folds, scoring="r2")
    return {"R2_mean": float(np.mean(scores)), "R2_std": float(np.std(scores)),
            "n_valid": int(valid.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="ROOT files dir (val_5M)")
    parser.add_argument("--emb-dir", required=True, help="Embedding .npy dir (embeddings_val_5M)")
    parser.add_argument("--output", default="/data/results/probing_results.json")
    parser.add_argument("--n-per-class", type=int, default=5000)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    emb_dir = Path(args.emb_dir)
    n_per_class = args.n_per_class

    # Process each class: load ROOT files in sorted order, load corresponding embeddings
    all_obs = {name: [] for name in OBS_NAMES}
    all_emb = []
    all_labels = []

    for emb_file in sorted(emb_dir.glob("*_embeddings.npy")):
        cls_name = emb_file.stem.replace("_embeddings", "")
        if cls_name not in EMB_CLASS_MAP:
            continue
        label = EMB_CLASS_MAP[cls_name]

        # Load embeddings for this class
        emb = np.load(str(emb_file))  # (N_class, 128)
        print(f"\n{cls_name}: {len(emb):,} embeddings, label={label}")

        # Find ROOT files for this class
        root_files = sorted(root_dir.glob(f"{cls_name}_*.root"))
        if not root_files:
            print(f"  WARNING: No ROOT files for {cls_name}")
            continue

        # Load ROOT files in order, count jets to match embedding order
        root_jets = 0
        cls_obs = {name: [] for name in OBS_NAMES}
        cls_emb = []
        done = False

        for rf in root_files:
            if done:
                break
            with uproot.open(f"{rf}:{TREE_NAME}") as tree:
                arrays = tree.arrays(PARTICLE_KEYS, library="np")
                n_jets = len(arrays["part_px"])

            for i in range(n_jets):
                if len(cls_emb) >= n_per_class:
                    done = True
                    break

                obs = compute_observables_for_jet(
                    px=arrays["part_px"][i],
                    py=arrays["part_py"][i],
                    pz=arrays["part_pz"][i],
                    energy=arrays["part_energy"][i],
                    d0val=arrays["part_d0val"][i],
                    dzval=arrays["part_dzval"][i],
                    charge=arrays["part_charge"][i],
                )
                for name in OBS_NAMES:
                    cls_obs[name].append(obs[name])

                # Corresponding embedding at the same global index
                emb_idx = root_jets + i
                if emb_idx < len(emb):
                    cls_emb.append(emb[emb_idx])

            root_jets += n_jets

        n_matched = min(len(cls_emb), min(len(v) for v in cls_obs.values()))
        print(f"  Matched: {n_matched} jets")

        for name in OBS_NAMES:
            all_obs[name].extend(cls_obs[name][:n_matched])
        all_emb.extend(cls_emb[:n_matched])
        all_labels.extend([label] * n_matched)

    # Convert to arrays
    embeddings = np.array(all_emb, dtype=np.float32)
    labels = np.array(all_labels, dtype=int)
    obs_arrays = {name: np.array(all_obs[name]) for name in OBS_NAMES}

    print(f"\nTotal: {len(labels):,} matched jets, {len(np.unique(labels))} classes")
    print(f"Embeddings: {embeddings.shape}")

    # --- All-classes probing ---
    print("\n=== All classes ===")
    all_results = {}
    for obs_name in OBS_NAMES:
        result = probe_one(embeddings, obs_arrays[obs_name])
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
            result = probe_one(cls_emb, obs_arrays[obs_name][mask])
            per_class[cls_name][obs_name] = result
            print(f"    {obs_name:>15}: R² = {result['R2_mean']:.4f} ± {result['R2_std']:.4f}")

    # Save
    output = {
        "n_jets": len(labels),
        "embedding_dim": int(embeddings.shape[1]),
        "all_classes": all_results,
        "per_class": per_class,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
