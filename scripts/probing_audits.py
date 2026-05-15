#!/usr/bin/env python3
"""Audit suite for the d0 / FT-compression probing story.

Three CPU audits + one MLP-dependent audit, all on existing artifacts:
  Audit 1 — Multiplicity-residualized ridge probing
            For y in [count_d0_gt_1sigma, count_d0_gt_2sigma, count_dz_gt_2sigma,
                     multiplicity (sanity check)]:
              residualize y w.r.t. n_charged via out-of-fold polynomial regression,
              run the same ridge-probe pipeline on the residual.
            -> audit1_residualized_R2.csv

  Audit 2 — Per-class ridge probing
            Slice the test set by class (10K each); run ridge probe within each
            class for: count_d0_gt_1sigma, count_d0_gt_2sigma, count_dz_gt_2sigma,
            multiplicity, jet_mass, width.
            -> audit2_per_class_R2.csv

  Audit 4 — MLP probe selectivity (Hewitt-Liang control)
            For each observable: refit the same MLP (128->256->64->1) on shuffled
            labels; report R^2_real (from existing mlp_probing_results.csv) and
            R^2_shuffled. Selectivity = R^2_real - R^2_shuffled.
            -> audit4_mlp_selectivity.csv

Usage:
    # Audits 1+2 (no MLP probe needed):
    python scripts/probing_audits.py \\
        --observables /data/results/poster/observables_test_100k.parquet \\
        --pretrained-dir /data/embeddings_test_20M \\
        --finetuned-dir  /data/embeddings_ft_full_3M_seed42_test100k \\
        --output-dir /data/results/poster/probing_audits \\
        --audits 1,2

    # Audit 4 (after mlp_probing_results.csv exists):
    python scripts/probing_audits.py \\
        --observables ... --pretrained-dir ... --finetuned-dir ... \\
        --mlp-probing-csv /data/results/poster/mlp_probing_results.csv \\
        --output-dir /data/results/poster/probing_audits \\
        --audits 4
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

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]


# ===== Shared helpers =======================================================

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


def ridge_probe(X_train, y_train, X_test, y_test, n_bootstrap: int,
                rng: np.random.Generator, alphas) -> dict:
    clf = RidgeCV(alphas=alphas)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    base_r2 = float(r2_score(y_test, yhat))
    n = len(y_test)
    boots = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = r2_score(y_test[idx], yhat[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"R2_mean": base_r2, "R2_lo": float(lo), "R2_hi": float(hi),
            "alpha": float(clf.alpha_)}


def residualize(y: np.ndarray, x: np.ndarray, degree: int = 3, n_folds: int = 5,
                seed: int = 42) -> np.ndarray:
    """Out-of-fold polynomial regression of y on x; return y - E[y|x] (test fold-wise)."""
    y = np.asarray(y).reshape(-1).astype(float)
    x = np.asarray(x).reshape(-1, 1).astype(float)
    pred = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(x):
        pf = PolynomialFeatures(degree, include_bias=False)
        Xtr = pf.fit_transform(x[tr])
        Xte = pf.transform(x[te])
        m = LinearRegression().fit(Xtr, y[tr])
        pred[te] = m.predict(Xte)
    return y - pred


# ===== Audit 1: residualized probing =======================================

def audit1(args, df_obs, emb_pre, lab_pre, emb_ft, lab_ft):
    """For each target y, residualize against n_charged and re-run ridge probe."""
    targets = ["count_d0_gt_1sigma", "count_d0_gt_2sigma",
               "count_dz_gt_2sigma", "multiplicity"]
    if "n_charged" not in df_obs.columns:
        raise RuntimeError("audit1 requires 'n_charged' column in observables parquet")
    n_charged = df_obs["n_charged"].values

    print(f"\n=== Audit 1: residualized probing (degree-3 poly, k=5 OOF) ===")
    rng = np.random.default_rng(args.seed)
    alphas = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

    rows = []
    for tag, emb, lab in [("pretrained", emb_pre, lab_pre),
                          (args.finetune_label, emb_ft, lab_ft)]:
        # Align embeddings to observables (identical stratified seed; this
        # guards against any class-count mismatch).
        e, l, df = emb, lab, df_obs
        if not np.array_equal(np.bincount(l, minlength=10),
                              np.bincount(df["label"].values, minlength=10)):
            e, l, df = align_by_class(emb, lab, df_obs)
        nc = df["n_charged"].values

        Xs = StandardScaler().fit_transform(e)
        Xtr, Xte, idx_tr, idx_te = train_test_split(
            Xs, np.arange(len(Xs)), test_size=0.2, random_state=args.seed, stratify=l
        )

        for tgt in targets:
            y = df[tgt].values.astype(float)
            finite = np.isfinite(y)
            if finite.sum() < 100:
                continue

            # Raw ridge R^2 (recompute for reference)
            scl = StandardScaler()
            ytr_z = scl.fit_transform(y[idx_tr][finite[idx_tr]].reshape(-1, 1)).ravel()
            yte_z = scl.transform(y[idx_te][finite[idx_te]].reshape(-1, 1)).ravel()
            raw = ridge_probe(Xtr[finite[idx_tr]], ytr_z,
                              Xte[finite[idx_te]], yte_z,
                              args.n_bootstrap, rng, alphas)

            # Residualized: y_resid = y - E[y|n_charged]
            y_resid = residualize(y[finite], nc[finite], degree=3,
                                  n_folds=5, seed=args.seed)
            # Map back to full-length array (residual NaN for non-finite y)
            yr_full = np.full_like(y, np.nan, dtype=float)
            yr_full[finite] = y_resid

            scl_r = StandardScaler()
            ytr_r = scl_r.fit_transform(yr_full[idx_tr][finite[idx_tr]].reshape(-1, 1)).ravel()
            yte_r = scl_r.transform(yr_full[idx_te][finite[idx_te]].reshape(-1, 1)).ravel()
            res = ridge_probe(Xtr[finite[idx_tr]], ytr_r,
                              Xte[finite[idx_te]], yte_r,
                              args.n_bootstrap, rng, alphas)

            rows.append({
                "model": tag,
                "observable": tgt,
                "raw_R2": raw["R2_mean"],
                "residualized_R2": res["R2_mean"],
                "delta": raw["R2_mean"] - res["R2_mean"],
                "ci_lo": res["R2_lo"],
                "ci_hi": res["R2_hi"],
            })
            print(f"  {tag:>22} | {tgt:>22}: raw={raw['R2_mean']:.4f}  "
                  f"resid={res['R2_mean']:.4f}  Δ={raw['R2_mean']-res['R2_mean']:+.4f}")

    out = Path(args.output_dir) / "audit1_residualized_R2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"  Saved {out}")

    # Decision-rule sanity print for poster
    df_a1 = pd.DataFrame(rows)
    pre = df_a1[(df_a1.model == "pretrained")
                & (df_a1.observable == "count_d0_gt_2sigma")]
    if len(pre):
        v = float(pre.iloc[0]["residualized_R2"])
        print(f"\n  DECISION RULE (pretrained, count_d0_gt_2sigma residualized R²):")
        print(f"    R² = {v:.4f}")
        if v >= 0.4:
            print("    → strong form: 'Sophon encodes the displacement count BEYOND multiplicity'")
        elif v >= 0.2:
            print("    → weaker form: 'partly explained by multiplicity'")
        else:
            print("    → kill b-tag-invariant framing; reframe as 'multiplicity-correlated'")


# ===== Audit 2: per-class probing ===========================================

def audit2(args, df_obs, emb_pre, lab_pre, emb_ft, lab_ft):
    """Within-class ridge probing."""
    targets = ["count_d0_gt_1sigma", "count_d0_gt_2sigma", "count_dz_gt_2sigma",
               "multiplicity", "jet_mass", "width"]

    print(f"\n=== Audit 2: per-class ridge probing ===")
    rng = np.random.default_rng(args.seed)
    alphas = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)

    rows = []
    for tag, emb, lab in [("pretrained", emb_pre, lab_pre),
                          (args.finetune_label, emb_ft, lab_ft)]:
        e, l, df = emb, lab, df_obs
        if not np.array_equal(np.bincount(l, minlength=10),
                              np.bincount(df["label"].values, minlength=10)):
            e, l, df = align_by_class(emb, lab, df_obs)

        for cls_idx, cls_name in enumerate(LABEL_NAMES):
            mask = (l == cls_idx)
            if mask.sum() < 200:
                continue
            ec = StandardScaler().fit_transform(e[mask])
            for tgt in targets:
                y = df.loc[mask, tgt].values.astype(float)
                finite = np.isfinite(y)
                if finite.sum() < 200:
                    continue
                Xtr, Xte, ytr, yte = train_test_split(
                    ec[finite], y[finite], test_size=0.2,
                    random_state=args.seed,
                )
                scl = StandardScaler()
                ytr_z = scl.fit_transform(ytr.reshape(-1, 1)).ravel()
                yte_z = scl.transform(yte.reshape(-1, 1)).ravel()
                out = ridge_probe(Xtr, ytr_z, Xte, yte_z,
                                  args.n_bootstrap, rng, alphas)
                rows.append({
                    "model": tag,
                    "observable": tgt,
                    "class": cls_name,
                    "R2": out["R2_mean"],
                    "ci_lo": out["R2_lo"],
                    "ci_hi": out["R2_hi"],
                    "n": int(finite.sum()),
                })
        print(f"  {tag} done")

    out = Path(args.output_dir) / "audit2_per_class_R2.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"  Saved {out}")

    # Decision-rule print for poster
    df_a2 = pd.DataFrame(rows)
    for cls in ("QCD", "Hbb"):
        sub = df_a2[(df_a2.model == "pretrained")
                    & (df_a2.observable == "count_d0_gt_2sigma")
                    & (df_a2["class"] == cls)]
        if len(sub):
            v = float(sub.iloc[0]["R2"])
            print(f"  PER-CLASS pretrained {cls} count_d0_gt_2sigma R² = {v:.4f}")


# ===== Audit 4: MLP selectivity ============================================

class MLPProbe(nn.Module):
    def __init__(self, input_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(X_tr, y_tr, X_va, y_va, device, epochs=50, batch_size=512,
            patience=8, lr=1e-3) -> nn.Module:
    model = MLPProbe(input_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_ds = TensorDataset(torch.from_numpy(X_tr).float(),
                             torch.from_numpy(y_tr).float())
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    Xv = torch.from_numpy(X_va).float().to(device)
    yv = torch.from_numpy(y_va).float().to(device)

    best_val = float("inf"); best_state = None; bad = 0
    for _ in range(epochs):
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


def audit4(args, df_obs, emb_pre, lab_pre, emb_ft, lab_ft, mlp_probing_csv: str):
    """Refit MLP on SHUFFLED labels for each observable; compute selectivity."""
    print(f"\n=== Audit 4: MLP selectivity vs shuffled-label control ===")
    if not Path(mlp_probing_csv).exists():
        print(f"  SKIPPING: {mlp_probing_csv} not found yet (mlp-probe job hasn't landed)")
        return
    df_real = pd.read_csv(mlp_probing_csv)
    obs_list = df_real.observable.unique().tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    rows = []
    for tag, emb, lab in [("pretrained", emb_pre, lab_pre),
                          (args.finetune_label, emb_ft, lab_ft)]:
        e, l, df = emb, lab, df_obs
        if not np.array_equal(np.bincount(l, minlength=10),
                              np.bincount(df["label"].values, minlength=10)):
            e, l, df = align_by_class(emb, lab, df_obs)

        Xs = StandardScaler().fit_transform(e).astype(np.float32)
        Xtr_full, Xte, idx_tr_full, idx_te = train_test_split(
            Xs, np.arange(len(Xs)), test_size=0.2, random_state=args.seed, stratify=l
        )
        Xtr, Xva, idx_tr, idx_va = train_test_split(
            Xtr_full, idx_tr_full, test_size=0.125, random_state=args.seed,
            stratify=l[idx_tr_full],
        )

        for obs in obs_list:
            real_row = df_real[(df_real.model == tag) & (df_real.observable == obs)]
            if len(real_row) == 0:
                continue
            r2_real = float(real_row.iloc[0]["R2_mean"])

            y = df[obs].values.astype(float)
            finite = np.isfinite(y)
            scl = StandardScaler()
            # SHUFFLED y
            y_shuf = y.copy()
            shuf_idx = np.arange(len(y_shuf))
            rng.shuffle(shuf_idx)
            y_shuf = y_shuf[shuf_idx]

            ytr = scl.fit_transform(
                y_shuf[idx_tr][finite[idx_tr]].reshape(-1, 1)).ravel()
            yva = scl.transform(
                y_shuf[idx_va][finite[idx_va]].reshape(-1, 1)).ravel()
            yte = scl.transform(
                y_shuf[idx_te][finite[idx_te]].reshape(-1, 1)).ravel()

            Xtr_o = Xtr[finite[idx_tr]]
            Xva_o = Xva[finite[idx_va]]
            Xte_o = Xte[finite[idx_te]]

            t0 = time.time()
            model = fit_mlp(Xtr_o, ytr, Xva_o, yva, device)
            model.eval()
            with torch.no_grad():
                yhat = model(torch.from_numpy(Xte_o).float().to(device)).cpu().numpy()
            r2_shuf = float(r2_score(yte, yhat))

            n = len(yte)
            boots = np.empty(args.n_bootstrap, dtype=float)
            for b in range(args.n_bootstrap):
                ix = rng.integers(0, n, size=n)
                boots[b] = r2_score(yte[ix], yhat[ix])
            lo, hi = np.percentile(boots, [2.5, 97.5])

            sel = r2_real - r2_shuf
            rows.append({
                "model": tag,
                "observable": obs,
                "R2_real": r2_real,
                "R2_shuffled": r2_shuf,
                "selectivity": sel,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            })
            print(f"  {tag:>22} | {obs:>22}: real={r2_real:.4f}  "
                  f"shuf={r2_shuf:.4f}  sel={sel:+.4f}  ({time.time() - t0:.1f}s)")

    out = Path(args.output_dir) / "audit4_mlp_selectivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"  Saved {out}")


# ===== Driver ===============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observables", required=True)
    p.add_argument("--pretrained-dir", required=True)
    p.add_argument("--finetuned-dir", required=True)
    p.add_argument("--finetune-label", default="full_ft_3M_seed42")
    p.add_argument("--output-dir", default="/data/results/poster/probing_audits")
    p.add_argument("--n-per-class", type=int, default=10000)
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audits", default="1,2",
                   help="Comma list of audits to run (1, 2, 4). "
                        "Audit 4 requires --mlp-probing-csv to exist.")
    p.add_argument("--mlp-probing-csv",
                   default="/data/results/poster/mlp_probing_results.csv")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df_obs = pd.read_parquet(args.observables)
    print(f"Loaded observables: {len(df_obs):,} jets, {len(df_obs.columns)} columns")

    emb_pre, lab_pre = load_aligned(args.pretrained_dir, args.n_per_class, args.seed)
    emb_ft,  lab_ft  = load_aligned(args.finetuned_dir,  args.n_per_class, args.seed)
    print(f"Pretrained: {len(emb_pre):,} emb. FT: {len(emb_ft):,} emb.")

    audits = set(args.audits.split(","))
    if "1" in audits:
        audit1(args, df_obs, emb_pre, lab_pre, emb_ft, lab_ft)
    if "2" in audits:
        audit2(args, df_obs, emb_pre, lab_pre, emb_ft, lab_ft)
    if "4" in audits:
        audit4(args, df_obs, emb_pre, lab_pre, emb_ft, lab_ft, args.mlp_probing_csv)


if __name__ == "__main__":
    main()
