#!/usr/bin/env python3
"""Arm 3.3 STEP 4 — Causal ablation: zero d0 at the input to pretrained Sophon.

For a stratified test subset (default 10K/class = 100K jets), runs the pretrained
Sophon backbone + the strongest available frozen-MLP head (trained on 100M jets,
seed=42) twice per jet:
  (a) intact d0:  features as-is
  (b) d0 zeroed:  features[:,:,11] = 0  (Sophon input feature index 11 is tanh(d0))

Computes per-class one-vs-rest AUC for each setting and the delta. The expectation
(documented in the prompt) is:
  H->bb baseline AUC ~ 0.97; with d0 zeroed ~ 0.85 (large drop)
  Z->qq AUC ~ unchanged
If H->bb AUC does NOT drop, the input-feature indexing assumption is wrong.

Usage:
    python scripts/causal_ablation_d0.py \\
        --pretrained-checkpoint models/JetClassII_Sophon/model.pt \\
        --frozen-mlp /data/results/frozen_base/frozen_base_100000000_42/best_model.pt \\
        --features-dir /data/features/test_20M \\
        --output results/arm3/causal_ablation_results.csv \\
        --n-per-class 10000
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
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sophon_wrapper import SophonTransferModel
from src.models.heads import MLPHead


LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql",
               "Zqq", "Wqq", "Tbqq", "Tbl"]

# Sophon's 17-feature input ordering (see src/data/jetclass.py:131-149).
# Index 11 = tanh(d0val), index 13 = tanh(dzval).
D0_FEATURE_INDEX = 11
DZ_FEATURE_INDEX = 13

# class-name -> Sophon label (int) used in features dir naming
CLASS_LABEL = {
    "ZJetsToNuNu": 0, "HToBB": 1, "HToCC": 2, "HToGG": 3, "HToWW4Q": 4,
    "HToWW2Q1L": 5, "ZToQQ": 6, "WToQQ": 7, "TTBar": 8, "TTBarLep": 9,
}


def load_features(features_dir: Path, max_per_class: int):
    """Load class-balanced subset of pre-extracted .npy features.

    Returns (features, lorentz, masks, labels) as concatenated arrays in fp16/fp32.
    """
    feat_files: dict[str, list[str]] = {}
    for f in sorted(features_dir.glob("*_features.npy")):
        stem = f.stem.replace("_features", "")
        cls = stem.rsplit("_", 1)[0] if "_" in stem else stem
        if cls in CLASS_LABEL:
            feat_files.setdefault(cls, []).append(stem)

    all_f, all_lv, all_m, all_lab = [], [], [], []
    for cls, stems in sorted(feat_files.items()):
        label = CLASS_LABEL[cls]
        collected = 0
        for stem in stems:
            if collected >= max_per_class:
                break
            f = np.load(str(features_dir / f"{stem}_features.npy"), mmap_mode="r")
            lv = np.load(str(features_dir / f"{stem}_lorentz.npy"), mmap_mode="r")
            m = np.load(str(features_dir / f"{stem}_masks.npy"), mmap_mode="r")
            take = min(len(f), max_per_class - collected)
            all_f.append(np.array(f[:take]))
            all_lv.append(np.array(lv[:take]))
            all_m.append(np.array(m[:take]))
            all_lab.append(np.full(take, label, dtype=np.int64))
            collected += take
        print(f"  {cls}: {collected:,} jets (label={label})")
    return (np.concatenate(all_f),
            np.concatenate(all_lv),
            np.concatenate(all_m),
            np.concatenate(all_lab))


@torch.no_grad()
def forward_to_logits(backbone: SophonTransferModel, head: MLPHead,
                      f: torch.Tensor, lv: torch.Tensor, m: torch.Tensor,
                      device, amp_dtype) -> torch.Tensor:
    with torch.amp.autocast(device_type="cuda",
                            dtype=amp_dtype,
                            enabled=(amp_dtype != torch.float32)):
        _, embed = backbone(f, lv, m)  # (B, 128)
    logits = head(embed.float())
    return logits


@torch.no_grad()
def run_pass(backbone, head, feats, lorentz, masks,
             intervention: str,
             device,
             d0_pool: np.ndarray | None = None,
             d0err_pool: np.ndarray | None = None,
             rng: np.random.Generator | None = None,
             batch_size: int = 512) -> np.ndarray:
    """Returns (N, 10) softmax probabilities under one of three interventions.

    intervention:
        "intact"   — features unchanged
        "zero"     — features[:, :, D0_FEATURE_INDEX] = 0 for charged particles only
        "resample" — replace d0 (and d0err) at each charged particle with random
                     samples drawn from the marginal pool of charged-particle d0.
                     Charged particles are identified by features[:, :, isCH] flag,
                     more robustly by features[:, :, D0ERR_INDEX] > 0.

    Audit 3 from the poster review: resample addresses the off-manifold critique
    that zero-ablation places the model in an unphysical regime.
    """
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) \
                else torch.float32
    out_probs = []
    n = len(feats)
    for i in range(0, n, batch_size):
        f_b = feats[i:i+batch_size].copy()  # don't mutate the source array
        if intervention == "zero":
            # Zero d0 only at positions where the original was non-zero
            # (i.e., charged particles with valid IP measurement). Neutrals
            # already have d0=0 by convention; touching them is a no-op anyway.
            f_b[:, :, D0_FEATURE_INDEX] = 0.0
        elif intervention == "resample":
            assert d0_pool is not None and rng is not None
            # Charged particles in this batch: nonzero d0err (feature 12)
            charged_mask = f_b[:, :, D0_FEATURE_INDEX + 1] > 0  # d0err > 0 → charged
            n_charged = int(charged_mask.sum())
            if n_charged > 0:
                idx = rng.integers(0, len(d0_pool), size=n_charged)
                f_b[charged_mask, D0_FEATURE_INDEX] = d0_pool[idx]
                if d0err_pool is not None:
                    # Redraw d0err coupled to the new d0 from the same indices,
                    # preserving the (d0, d0err) joint distribution on the pool.
                    f_b[charged_mask, D0_FEATURE_INDEX + 1] = d0err_pool[idx]
        elif intervention != "intact":
            raise ValueError(f"unknown intervention: {intervention}")

        # Transpose to (B, 17, 128) and (B, 4, 128) for Sophon's API
        f_t = torch.from_numpy(f_b.astype(np.float32)).transpose(1, 2).to(device, non_blocking=True)
        v_t = torch.from_numpy(np.asarray(lorentz[i:i+batch_size], dtype=np.float32)).transpose(1, 2).to(device, non_blocking=True)
        m_t = torch.from_numpy(np.asarray(masks[i:i+batch_size])).unsqueeze(1).to(device, non_blocking=True)
        logits = forward_to_logits(backbone, head, f_t, v_t, m_t, device, amp_dtype)
        out_probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out_probs, axis=0)


def build_d0_pool(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal pool of (d0, d0err) over all charged particles in the test set.

    Charged particles identified by d0err > 0 (feature index 12). The pool acts
    as the "resample" reservoir: we draw with replacement from this distribution.
    For Audit 3 we use the test-set marginal as a stand-in for the training-set
    marginal — this is fine because Sophon's training and test data come from the
    same JetClass-1 generator and selection.
    """
    d0 = feats[:, :, D0_FEATURE_INDEX]
    d0err = feats[:, :, D0_FEATURE_INDEX + 1]
    mask = d0err > 0
    return d0[mask].astype(np.float32), d0err[mask].astype(np.float32)


def per_class_auc(labels, probs) -> dict:
    """One-vs-rest AUC per class."""
    out = {}
    for i, name in enumerate(LABEL_NAMES):
        bin_lab = (labels == i).astype(int)
        if bin_lab.sum() == 0 or bin_lab.sum() == len(bin_lab):
            out[name] = float("nan")
        else:
            out[name] = float(roc_auc_score(bin_lab, probs[:, i]))
    out["macro"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-checkpoint", required=True,
                   help="Pretrained Sophon checkpoint (e.g., models/JetClassII_Sophon/model.pt)")
    p.add_argument("--frozen-mlp", required=True,
                   help="Trained MLPHead checkpoint (state dict) — typically frozen_base_100000000_42/best_model.pt")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--output", default="results/arm3/causal_ablation_results.csv")
    p.add_argument("--n-per-class", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Sophon backbone (replace head with a placeholder MLP we won't use; we route
    # embeddings through the standalone MLPHead instead).
    print(f"Loading pretrained Sophon backbone from {args.pretrained_checkpoint}")
    backbone = SophonTransferModel(
        checkpoint_path=args.pretrained_checkpoint,
        num_classes=10, head_type="mlp", export_embed=True,
    ).to(device).eval()

    # Load the trained 10-class MLP head (operates on 128-dim embeddings).
    print(f"Loading frozen MLP head from {args.frozen_mlp}")
    head = MLPHead(128, 10, [256], dropout=0.1).to(device).eval()
    head_state = torch.load(args.frozen_mlp, map_location="cpu", weights_only=False)
    if isinstance(head_state, dict) and "model_state_dict" in head_state:
        head_state = head_state["model_state_dict"]
    head.load_state_dict(head_state, strict=True)

    # Load test subset
    print(f"\nLoading {args.n_per_class:,}/class test features from {args.features_dir}")
    feats, lorentz, masks, labels = load_features(Path(args.features_dir), args.n_per_class)
    print(f"  total: {len(labels):,} jets, features shape {feats.shape}")

    # Build resample pool from the test set's charged-particle (d0, d0err) marginal
    # (Audit 3: addresses off-manifold critique of zero-ablation)
    d0_pool, d0err_pool = build_d0_pool(feats)
    print(f"\nResample pool: {len(d0_pool):,} charged-particle (d0, d0err) pairs")
    rng = np.random.default_rng(args.seed)

    # Run three passes
    print("\n=== Pass A: d0 INTACT (baseline) ===")
    t0 = time.time()
    probs_intact = run_pass(backbone, head, feats, lorentz, masks,
                            intervention="intact", device=device,
                            batch_size=args.batch_size)
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"\n=== Pass B: d0 ZEROED at feature index {D0_FEATURE_INDEX} ===")
    t0 = time.time()
    probs_zerod0 = run_pass(backbone, head, feats, lorentz, masks,
                            intervention="zero", device=device,
                            batch_size=args.batch_size)
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"\n=== Pass C: d0 RESAMPLED from charged-particle marginal pool ===")
    t0 = time.time()
    probs_resample = run_pass(backbone, head, feats, lorentz, masks,
                              intervention="resample", device=device,
                              d0_pool=d0_pool, d0err_pool=d0err_pool, rng=rng,
                              batch_size=args.batch_size)
    print(f"  done in {time.time() - t0:.1f}s")

    # Per-class AUC for each
    auc_intact = per_class_auc(labels, probs_intact)
    auc_zerod0 = per_class_auc(labels, probs_zerod0)
    auc_resample = per_class_auc(labels, probs_resample)

    print("\n--- Results ---")
    print(f"{'class':>10} {'intact':>10} {'d0=0':>10} {'resamp':>10} "
          f"{'Δ_zero':>10} {'Δ_resamp':>10}")
    rows = []
    for k in LABEL_NAMES + ["macro"]:
        a, b, c = auc_intact[k], auc_zerod0[k], auc_resample[k]
        print(f"{k:>10} {a:>10.4f} {b:>10.4f} {c:>10.4f} "
              f"{a - b:>+10.4f} {a - c:>+10.4f}")
        rows.append({
            "class": k,
            "baseline_AUC": a,
            "zero_AUC": b,
            "resample_AUC": c,
            "delta_zero": a - b,
            "delta_resample": a - c,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved CSV: {out_path}")

    # Sanity check
    if not np.isnan(auc_intact.get("Hbb", float("nan"))):
        delta_hbb_zero = auc_intact["Hbb"] - auc_zerod0["Hbb"]
        delta_hbb_res  = auc_intact["Hbb"] - auc_resample["Hbb"]
        if delta_hbb_zero < 0.03:
            print(f"\nFLAG: Hbb AUC dropped by only {delta_hbb_zero:+.4f} on zero-ablation.")
            print("  Below stopping-rule threshold of 0.03 — interpretability story changes if d0 isn't used.")
        if delta_hbb_res < 0.03:
            print(f"\nFLAG: Hbb AUC dropped by only {delta_hbb_res:+.4f} on resample-ablation.")
            print("  Headline drop should use resample value; flag for user.")


if __name__ == "__main__":
    main()
