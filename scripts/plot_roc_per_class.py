#!/usr/bin/env python3
"""Per-class ROC plot for the best Frozen Sophon + MLP model (trained on 100M).

Loads the saved MLP head (frozen_base_100000000_42/best_model.pt), applies it
to the pretrained Sophon test embeddings, computes per-class one-vs-rest ROC.

Outputs:
    results/main_plots/roc_per_class.{pdf,png}
    results/arm1/roc_per_class.{pdf,png}
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample
from src.models.heads import MLPHead

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots.style import set_publishable_style, save_fig, CLASS_COLORS

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql",
               "Zqq", "Wqq", "Tbqq", "Tbl"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp", required=True,
                   help="Path to frozen MLP head state dict")
    p.add_argument("--embeddings-dir", required=True,
                   help="Pretrained Sophon test embeddings dir")
    p.add_argument("--n-per-class", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/main_plots/roc_per_class")
    args = p.parse_args()

    set_publishable_style()

    # Load embeddings (subsample 100K)
    print(f"Loading embeddings from {args.embeddings_dir}")
    emb, lab = _load_dir(args.embeddings_dir)
    target = args.n_per_class * 10
    idx = stratified_subsample(lab, min(target, len(lab)), args.seed)
    emb = emb[idx].astype(np.float32)
    lab = lab[idx]
    print(f"  {len(lab):,} jets")

    # Load MLP head
    print(f"Loading MLP head from {args.mlp}")
    head = MLPHead(128, 10, [256], dropout=0.1).eval()
    state = torch.load(args.mlp, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    head.load_state_dict(state, strict=True)

    # Forward through head
    with torch.no_grad():
        logits = head(torch.from_numpy(emb))
        probs = F.softmax(logits, dim=-1).numpy()

    # ROC per class
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    aucs = {}
    for cls_idx in range(10):
        name = LABEL_NAMES[cls_idx]
        y_bin = (lab == cls_idx).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, cls_idx])
        auc = float(roc_auc_score(y_bin, probs[:, cls_idx]))
        aucs[name] = auc
        ax.plot(fpr, tpr, color=CLASS_COLORS[name], lw=1.4,
                label=f"{name}  ({auc:.3f})")

    macro = float(roc_auc_score(lab, probs, multi_class="ovr", average="macro"))
    ax.plot([0, 1], [0, 1], color="#888", lw=0.7, linestyle=(0, (4, 3)))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Per-class ROC: Frozen Sophon + MLP (100M jets)\n"
                 f"macro AUC = {macro:.4f}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, ncol=2,
              title="Class (AUC)", title_fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    save_fig(fig, args.output)
    print(f"Saved {args.output}.{{pdf,png}}")
    print(f"Macro AUC = {macro:.4f}")


if __name__ == "__main__":
    main()
