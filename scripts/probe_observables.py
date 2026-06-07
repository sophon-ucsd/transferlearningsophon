#!/usr/bin/env python3
"""Linear ridge probing across jet observables.

For each (model in [pretrained, fine-tuned]) x (observable in parquet):
  X = 128-d Sophon embedding
  y = observable value
  RidgeCV (alpha grid) on 80% train, R^2 on 20% held-out
  Bootstrap 100x on the held-out test set for the R^2 95% CI

Output: probing_results.csv with columns
    [model, observable, R2_mean, R2_lo, R2_hi, n_train, n_test, alpha]

Usage:
    python scripts/probe_observables.py \\
        --observables results/substructure_observables.npz \\
        --pretrained-dir /data/embeddings_test_20M \\
        --finetuned-dir  /data/embeddings_ft_full_3M_seed42_test100k \\
        --output results/probing_results.csv \\
        --n-bootstrap 100 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

# Observables we probe by default (everything except non-target columns).
NON_TARGET = {"label", "jet_id"}

# Observables to skip (integer-valued or trivial columns we don't need to probe)
SKIP = set()


def load_aligned(emb_dir: str, n_per_class: int, seed: int):
    """Load embeddings, stratified-subsample to n_per_class * 10 jets, return (emb, labels)."""
    emb, lab = _load_dir(emb_dir)
    n_classes = len(np.unique(lab))
    target = n_per_class * n_classes
    idx = stratified_subsample(lab, min(target, len(lab)), seed)
    return emb[idx].astype(np.float32), lab[idx]


def probe_one(X_train, y_train, X_test, y_test,
              n_bootstrap: int, rng: np.random.Generator, alphas) -> dict:
    """Fit RidgeCV on train, evaluate R^2 on test with bootstrap CI."""
    clf = RidgeCV(alphas=alphas)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)

    base_r2 = float(r2_score(y_test, yhat))

    n = len(y_test)
    boots = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        boots[b] = r2_score(y_test[idx], yhat[idx])

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "R2_mean": base_r2,
        "R2_lo": float(lo),
        "R2_hi": float(hi),
        "alpha": float(clf.alpha_),
    }


def run_model(model_tag: str, emb: np.ndarray, lab: np.ndarray,
              df_obs: pd.DataFrame, observables: list[str],
              n_bootstrap: int, seed: int) -> list[dict]:
    """Probe every observable for this model. Returns list of result dicts."""
    print(f"\n=== Probing {model_tag} (n={len(emb):,}) ===")

    # Verify embeddings and observables are aligned by label distribution
    obs_labels = df_obs["label"].values
    if not np.array_equal(np.bincount(lab, minlength=10),
                          np.bincount(obs_labels, minlength=10)):
        print("WARNING: per-class jet counts differ between embeddings and observables.")
        print(f"  emb: {np.bincount(lab, minlength=10)}")
        print(f"  obs: {np.bincount(obs_labels, minlength=10)}")
        # Fall back to per-class alignment
        emb, lab, df_obs = align_by_class(emb, lab, df_obs)
        print(f"  after align: emb={len(emb):,}, obs={len(df_obs):,}")

    rng = np.random.default_rng(seed)
    alphas = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

    # Train/test split, shared across all observables (so error bars are comparable)
    Xs = StandardScaler().fit_transform(emb)
    Xtr, Xte, ytr_idx, yte_idx = train_test_split(
        Xs, np.arange(len(Xs)), test_size=0.2, random_state=seed, stratify=lab
    )

    results = []
    t0 = time.time()
    for obs in observables:
        y = df_obs[obs].values.astype(np.float64)
        # Drop NaN/inf
        finite = np.isfinite(y)
        if finite.sum() < 100:
            print(f"  SKIP {obs}: only {finite.sum()} finite values")
            continue

        # Index into split
        train_mask = finite[ytr_idx]
        test_mask = finite[yte_idx]
        Xtr_o, ytr_o = Xtr[train_mask], y[ytr_idx][train_mask]
        Xte_o, yte_o = Xte[test_mask],  y[yte_idx][test_mask]

        # Standardize y on train, apply same to test (for cross-comparable R^2 across obs)
        scl = StandardScaler()
        ytr_z = scl.fit_transform(ytr_o.reshape(-1, 1)).ravel()
        yte_z = scl.transform(yte_o.reshape(-1, 1)).ravel()

        out = probe_one(Xtr_o, ytr_z, Xte_o, yte_z, n_bootstrap, rng, alphas)
        out.update({
            "model": model_tag,
            "observable": obs,
            "n_train": int(len(Xtr_o)),
            "n_test": int(len(Xte_o)),
        })
        results.append(out)
        print(f"  {obs:>26}: R2 = {out['R2_mean']:.4f}  "
              f"[{out['R2_lo']:.4f}, {out['R2_hi']:.4f}]  alpha={out['alpha']:.3g}")
    print(f"  total time: {time.time() - t0:.1f}s")
    return results


def align_by_class(emb, lab, df_obs):
    """Reorder both side so per-class blocks align (simplest robust alignment)."""
    classes = sorted(np.unique(lab))
    emb_idx = []
    obs_idx = []
    for c in classes:
        ei = np.where(lab == c)[0]
        oi = np.where(df_obs["label"].values == c)[0]
        n = min(len(ei), len(oi))
        emb_idx.append(ei[:n])
        obs_idx.append(oi[:n])
    emb_idx = np.concatenate(emb_idx)
    obs_idx = np.concatenate(obs_idx)
    return emb[emb_idx], lab[emb_idx], df_obs.iloc[obs_idx].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observables", required=True, help="Parquet with per-jet observables")
    p.add_argument("--pretrained-dir", required=True)
    p.add_argument("--finetuned-dir", required=True)
    p.add_argument("--finetune-label", default="full_ft_3M_seed42")
    p.add_argument("--output", default="results/probing_results.csv")
    p.add_argument("--n-per-class", type=int, default=10000)
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--observables-include", default=None,
                   help="Comma list — restrict to these observable names")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading observables: {args.observables}")
    df_obs = pd.read_parquet(args.observables)
    print(f"  {len(df_obs):,} jets, {len(df_obs.columns)} columns")

    if args.observables_include:
        obs_list = args.observables_include.split(",")
    else:
        obs_list = [c for c in df_obs.columns if c not in NON_TARGET and c not in SKIP]
    print(f"  Probing {len(obs_list)} observables: {obs_list}")

    # Load embeddings (use same stratified subsample seed as observables for alignment)
    pre_emb, pre_lab = load_aligned(args.pretrained_dir, args.n_per_class, args.seed)
    ft_emb, ft_lab = load_aligned(args.finetuned_dir, args.n_per_class, args.seed)

    rows = []
    rows += run_model("pretrained", pre_emb, pre_lab, df_obs, obs_list, args.n_bootstrap, args.seed)
    rows += run_model(args.finetune_label, ft_emb, ft_lab, df_obs, obs_list, args.n_bootstrap, args.seed)

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved: {out_path}")

    # CHECKPOINT — print the d0 contrast prominently for sanity
    if "max_abs_d0" in df.observable.values and "mean_abs_d0" in df.observable.values:
        print("\n--- d0 sanity check ---")
        for tag in df.model.unique():
            sub = df[df.model == tag]
            mean = sub[sub.observable == "mean_abs_d0"]["R2_mean"].iloc[0]
            mx = sub[sub.observable == "max_abs_d0"]["R2_mean"].iloc[0]
            count = sub[sub.observable == "count_d0_gt_2sigma"]["R2_mean"].iloc[0] \
                    if "count_d0_gt_2sigma" in sub.observable.values else None
            print(f"  {tag}: mean_abs_d0 R2={mean:.3f}  max_abs_d0 R2={mx:.3f}"
                  + (f"  count_>2sigma R2={count:.3f}" if count is not None else ""))
            if mx <= mean + 0.05:
                print(f"  WARNING: max_abs_d0 NOT substantially higher than mean_abs_d0 for {tag}.")


if __name__ == "__main__":
    main()
