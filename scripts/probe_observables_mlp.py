#!/usr/bin/env python3
"""Arm 3.3 STEP 3 — MLP probing across observables.

Tests whether observable info is encoded NONLINEARLY in the Sophon embedding.
If MLP recovers max_abs_d0 at high R² while ridge gets ~0.07, magnitude info
is encoded — just not linearly.

Architecture per prompt: 128 -> 256 -> 64 -> 1, ReLU, dropout 0.1, MSE loss,
Adam lr=1e-3, 50 epochs, early stopping on val loss with patience=8.

Train/val/test split: 70/10/20. Bootstrap 100x on test for R² 95% CI.

Usage:
    python scripts/probe_observables_mlp.py \\
        --observables results/arm3/observables_test_100k.parquet \\
        --pretrained-dir /data/embeddings_test_20M \\
        --finetuned-dir  /data/embeddings_ft_full_3M_seed42_test100k \\
        --output results/arm3/mlp_probing_results.csv \\
        --n-bootstrap 100 \\
        --observables-include max_abs_d0,top3_sum_abs_d0,count_d0_gt_2sigma,mean_abs_d0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
NON_TARGET = {"label", "jet_id"}


class MLPProbe(nn.Module):
    """Small MLP probe per the prompt: 128 -> 256 -> 64 -> 1."""
    def __init__(self, input_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_aligned(emb_dir: str, n_per_class: int, seed: int):
    emb, lab = _load_dir(emb_dir)
    n_classes = len(np.unique(lab))
    target = n_per_class * n_classes
    idx = stratified_subsample(lab, min(target, len(lab)), seed)
    return emb[idx].astype(np.float32), lab[idx]


def align_by_class(emb, lab, df_obs):
    classes = sorted(np.unique(lab))
    emb_idx, obs_idx = [], []
    for c in classes:
        ei = np.where(lab == c)[0]
        oi = np.where(df_obs["label"].values == c)[0]
        n = min(len(ei), len(oi))
        emb_idx.append(ei[:n]); obs_idx.append(oi[:n])
    return (emb[np.concatenate(emb_idx)],
            lab[np.concatenate(emb_idx)],
            df_obs.iloc[np.concatenate(obs_idx)].reset_index(drop=True))


def train_probe(X_tr, y_tr, X_va, y_va, device,
                epochs: int = 50, batch_size: int = 512, patience: int = 8,
                lr: float = 1e-3) -> nn.Module:
    model = MLPProbe(input_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    Xv = torch.from_numpy(X_va).float().to(device)
    yv = torch.from_numpy(y_va).float().to(device)

    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(Xv), yv).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def probe_one(X_tr, y_tr, X_va, y_va, X_te, y_te,
              n_bootstrap: int, rng: np.random.Generator, device):
    model = train_probe(X_tr, y_tr, X_va, y_va, device)
    model.eval()
    with torch.no_grad():
        yhat = model(torch.from_numpy(X_te).float().to(device)).cpu().numpy()
    base_r2 = float(r2_score(y_te, yhat))

    n = len(y_te)
    boots = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = r2_score(y_te[idx], yhat[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"R2_mean": base_r2, "R2_lo": float(lo), "R2_hi": float(hi)}


def run_model(model_tag: str, emb: np.ndarray, lab: np.ndarray,
              df_obs: pd.DataFrame, observables: list[str],
              n_bootstrap: int, seed: int, device) -> list[dict]:
    print(f"\n=== MLP probing {model_tag} (n={len(emb):,}) ===")

    if not np.array_equal(np.bincount(lab, minlength=10),
                          np.bincount(df_obs["label"].values, minlength=10)):
        emb, lab, df_obs = align_by_class(emb, lab, df_obs)

    rng = np.random.default_rng(seed)

    Xs = StandardScaler().fit_transform(emb).astype(np.float32)
    # 70/10/20 split
    Xtr_full, Xte, ytr_full_idx, yte_idx = train_test_split(
        Xs, np.arange(len(Xs)), test_size=0.2, random_state=seed, stratify=lab
    )
    Xtr, Xva, ytr_idx, yva_idx = train_test_split(
        Xtr_full, ytr_full_idx, test_size=0.125, random_state=seed,
        stratify=lab[ytr_full_idx]  # 10% of 80% = 8% overall, close to prompt 70/10/20
    )

    results = []
    t_global = time.time()
    for obs in observables:
        y = df_obs[obs].values.astype(np.float64)
        finite = np.isfinite(y)
        if finite.sum() < 100:
            print(f"  SKIP {obs}: only {finite.sum()} finite values")
            continue

        scl = StandardScaler()
        ytr = scl.fit_transform(y[ytr_idx][finite[ytr_idx]].reshape(-1, 1)).ravel()
        yva = scl.transform(y[yva_idx][finite[yva_idx]].reshape(-1, 1)).ravel()
        yte = scl.transform(y[yte_idx][finite[yte_idx]].reshape(-1, 1)).ravel()

        Xtr_o = Xtr[finite[ytr_idx]]
        Xva_o = Xva[finite[yva_idx]]
        Xte_o = Xte[finite[yte_idx]]

        t0 = time.time()
        out = probe_one(Xtr_o, ytr, Xva_o, yva, Xte_o, yte,
                        n_bootstrap, rng, device)
        out.update({
            "model": model_tag,
            "observable": obs,
            "n_train": int(len(Xtr_o)),
            "n_val":   int(len(Xva_o)),
            "n_test":  int(len(Xte_o)),
        })
        results.append(out)
        print(f"  {obs:>26}: R2 = {out['R2_mean']:.4f}  "
              f"[{out['R2_lo']:.4f}, {out['R2_hi']:.4f}]  "
              f"({time.time() - t0:.1f}s)")
    print(f"  total: {time.time() - t_global:.0f}s")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observables", required=True)
    p.add_argument("--pretrained-dir", required=True)
    p.add_argument("--finetuned-dir", required=True)
    p.add_argument("--finetune-label", default="full_ft_3M_seed42")
    p.add_argument("--output", default="results/arm3/mlp_probing_results.csv")
    p.add_argument("--n-per-class", type=int, default=10000)
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--observables-include", default=None,
                   help="Comma list. If unset, probes all parquet columns.")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_obs = pd.read_parquet(args.observables)
    if args.observables_include:
        obs_list = args.observables_include.split(",")
    else:
        obs_list = [c for c in df_obs.columns if c not in NON_TARGET]
    print(f"Probing {len(obs_list)} observables: {obs_list}")

    pre_emb, pre_lab = load_aligned(args.pretrained_dir, args.n_per_class, args.seed)
    ft_emb,  ft_lab  = load_aligned(args.finetuned_dir,  args.n_per_class, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = []
    rows += run_model("pretrained", pre_emb, pre_lab, df_obs, obs_list,
                      args.n_bootstrap, args.seed, device)
    rows += run_model(args.finetune_label, ft_emb, ft_lab, df_obs, obs_list,
                      args.n_bootstrap, args.seed, device)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved: {out_path}")

    # Headline
    if "max_abs_d0" in df.observable.values:
        for tag in df.model.unique():
            sub = df[df.model == tag]
            for obs in ["mean_abs_d0", "max_abs_d0", "count_d0_gt_2sigma"]:
                if obs in sub.observable.values:
                    r = sub[sub.observable == obs]["R2_mean"].iloc[0]
                    print(f"  {tag:>22} | {obs:>20}: MLP R2 = {r:.4f}")


if __name__ == "__main__":
    main()
