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
             zero_index: int | None,
             device, batch_size: int = 512) -> np.ndarray:
    """Returns (N, 10) softmax probabilities. If zero_index is set, sets
    features[:, :, zero_index] = 0 before each forward pass.
    """
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) \
                else torch.float32
    out_probs = []
    n = len(feats)
    for i in range(0, n, batch_size):
        f_b = feats[i:i+batch_size].copy()  # don't mutate the source array
        if zero_index is not None:
            f_b[:, :, zero_index] = 0
        # Transpose to (B, 17, 128) and (B, 4, 128) for Sophon's API
        f_t = torch.from_numpy(f_b.astype(np.float32)).transpose(1, 2).to(device, non_blocking=True)
        v_t = torch.from_numpy(np.asarray(lorentz[i:i+batch_size], dtype=np.float32)).transpose(1, 2).to(device, non_blocking=True)
        m_t = torch.from_numpy(np.asarray(masks[i:i+batch_size])).unsqueeze(1).to(device, non_blocking=True)
        logits = forward_to_logits(backbone, head, f_t, v_t, m_t, device, amp_dtype)
        out_probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out_probs, axis=0)


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

    # Run two passes
    print("\n=== Pass A: d0 INTACT ===")
    t0 = time.time()
    probs_intact = run_pass(backbone, head, feats, lorentz, masks,
                            zero_index=None, device=device, batch_size=args.batch_size)
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"\n=== Pass B: d0 ZEROED at feature index {D0_FEATURE_INDEX} ===")
    t0 = time.time()
    probs_zerod0 = run_pass(backbone, head, feats, lorentz, masks,
                            zero_index=D0_FEATURE_INDEX, device=device, batch_size=args.batch_size)
    print(f"  done in {time.time() - t0:.1f}s")

    # Per-class AUC for each
    auc_intact = per_class_auc(labels, probs_intact)
    auc_zerod0 = per_class_auc(labels, probs_zerod0)

    print("\n--- Results ---")
    print(f"{'class':>10} {'intact':>10} {'d0=0':>10} {'delta':>10}")
    rows = []
    for k in LABEL_NAMES + ["macro"]:
        a, b = auc_intact[k], auc_zerod0[k]
        print(f"{k:>10} {a:>10.4f} {b:>10.4f} {a - b:>+10.4f}")
        rows.append({
            "class": k,
            "auc_intact": a,
            "auc_d0_zeroed": b,
            "delta": a - b,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSaved CSV: {out_path}")

    # Sanity check
    if not np.isnan(auc_intact.get("Hbb", float("nan"))):
        delta_hbb = auc_intact["Hbb"] - auc_zerod0["Hbb"]
        if delta_hbb < 0.05:
            print(f"\nWARNING: Hbb AUC dropped by only {delta_hbb:+.4f} after zeroing d0.")
            print("Expected drop ~0.10. Check input feature indexing.")


if __name__ == "__main__":
    main()
