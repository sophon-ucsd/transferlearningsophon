#!/usr/bin/env python3
"""3-panel per-class ROC at 3M jets, seed=42, for Frozen / Partial / Full FT.

Forwards a stratified 10K/class test subset (100K total) through each
strategy's saved checkpoint, computes per-class one-vs-rest ROC,
and writes a side-by-side 3-panel figure with semi-log FPR.

Outputs:
    /data/results/poster/roc_3strategies_3M.{pdf,png}    (when run on cluster)
    /data/results/poster/roc_3strategies_3M_curves.npz   (saved curves for re-render)
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample
from src.models.sophon_wrapper import SophonTransferModel
from src.models.heads import MLPHead
from plots.style import apply_style, save_fig, CLASS_COLORS

LABEL_NAMES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql",
               "Zqq", "Wqq", "Tbqq", "Tbl"]

CLASS_LABEL = {
    "ZJetsToNuNu": 0, "HToBB": 1, "HToCC": 2, "HToGG": 3, "HToWW4Q": 4,
    "HToWW2Q1L": 5, "ZToQQ": 6, "WToQQ": 7, "TTBar": 8, "TTBarLep": 9,
}


def load_test_features(features_dir: Path, max_per_class: int):
    """Load class-balanced subset of pre-extracted .npy features."""
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
        print(f"  {cls}: {collected:,}")
    return (np.concatenate(all_f),
            np.concatenate(all_lv),
            np.concatenate(all_m),
            np.concatenate(all_lab))


def load_test_embeddings(emb_dir: str, max_per_class: int, seed: int = 42):
    """Pretrained Sophon embeddings for the frozen-MLP path."""
    emb, lab = _load_dir(emb_dir)
    target = max_per_class * 10
    idx = stratified_subsample(lab, min(target, len(lab)), seed)
    return emb[idx].astype(np.float32), lab[idx]


@torch.no_grad()
def forward_frozen_mlp(mlp_path: str, emb: np.ndarray, device) -> np.ndarray:
    head = MLPHead(128, 10, [256], dropout=0.1).to(device).eval()
    state = torch.load(mlp_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    head.load_state_dict(state, strict=True)
    logits = head(torch.from_numpy(emb).to(device))
    return F.softmax(logits.float(), dim=-1).cpu().numpy()


@torch.no_grad()
def forward_full_model(checkpoint: str, pretrained: str,
                       feats: np.ndarray, lorentz: np.ndarray, masks: np.ndarray,
                       device, batch_size: int = 256) -> np.ndarray:
    """Forward features through a saved SophonTransferModel checkpoint."""
    model = SophonTransferModel(checkpoint_path=pretrained,
                                num_classes=10, head_type="mlp",
                                export_embed=True).to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded {len(state) - len(unexpected)} keys ({len(missing)} missing, {len(unexpected)} unexpected)")

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    out_probs = []
    for i in range(0, len(feats), batch_size):
        f_t = torch.from_numpy(np.asarray(feats[i:i+batch_size], dtype=np.float32)
                               ).transpose(1, 2).to(device, non_blocking=True)
        v_t = torch.from_numpy(np.asarray(lorentz[i:i+batch_size], dtype=np.float32)
                               ).transpose(1, 2).to(device, non_blocking=True)
        m_t = torch.from_numpy(np.asarray(masks[i:i+batch_size])
                               ).unsqueeze(1).to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(amp_dtype != torch.float32)):
            logits, _ = model(f_t, v_t, m_t)
        out_probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out_probs, axis=0)


def per_class_curves(probs: np.ndarray, lab: np.ndarray):
    out = {}
    for c in range(10):
        binary = (lab == c).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, c])
        auc = float(roc_auc_score(binary, probs[:, c]))
        out[LABEL_NAMES[c]] = (fpr, tpr, auc)
    macro = float(roc_auc_score(lab, probs, multi_class="ovr", average="macro"))
    return out, macro


def draw_panel(ax, curves: dict, macro: float, title: str, show_ylabel: bool = True,
               show_legend: bool = False):
    for name, (fpr, tpr, auc) in curves.items():
        # log x-axis: clamp to 1e-4 minimum
        fpr_safe = np.clip(fpr, 1e-4, 1.0)
        ax.plot(fpr_safe, tpr, color=CLASS_COLORS[name], lw=1.3,
                label=f"{name}  ({auc:.3f})")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    if show_ylabel:
        ax.set_ylabel("True positive rate")
    ax.set_title(f"{title}\nmacro AUC = {macro:.3f}", fontsize=11, pad=8)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, ncol=1,
                  title="class (AUC)", title_fontsize=9,
                  handlelength=1.2, handletextpad=0.4, borderpad=0.4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-mlp",
                   default="/data/results/frozen_base/frozen_base_3000000_42/best_model.pt")
    p.add_argument("--partial-ckpt",
                   default="/data/results/partial_ft/partial_ft_3000000_42/best_model.pt")
    p.add_argument("--full-ckpt",
                   default="/data/results/full_ft/full_ft_3000000_42/best_model.pt")
    p.add_argument("--pretrained",
                   default="models/JetClassII_Sophon/model.pt")
    p.add_argument("--features-dir", default="/data/features/test_20M")
    p.add_argument("--embeddings-dir", default="/data/embeddings_test_20M")
    p.add_argument("--n-per-class", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="/data/results/poster/roc_3strategies_3M")
    args = p.parse_args()

    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Frozen path
    print("\nFrozen MLP (pretrained embeddings + 35K head, trained on 3M)")
    emb, lab_emb = load_test_embeddings(args.embeddings_dir, args.n_per_class, args.seed)
    print(f"  {len(emb):,} test embeddings")
    probs_frozen = forward_frozen_mlp(args.frozen_mlp, emb, device)
    curves_frozen, macro_frozen = per_class_curves(probs_frozen, lab_emb)
    print(f"  macro AUC = {macro_frozen:.4f}")

    # Partial / Full FT paths (need raw features)
    print(f"\nLoading {args.n_per_class:,}/class test features from {args.features_dir}")
    t0 = time.time()
    feats, lorentz, masks, lab_feat = load_test_features(Path(args.features_dir),
                                                         args.n_per_class)
    print(f"  total: {len(lab_feat):,} jets, features shape {feats.shape}, "
          f"loaded in {time.time()-t0:.0f}s")

    print("\nPartial FT (3M, seed=42)")
    probs_partial = forward_full_model(args.partial_ckpt, args.pretrained,
                                       feats, lorentz, masks, device, args.batch_size)
    curves_partial, macro_partial = per_class_curves(probs_partial, lab_feat)
    print(f"  macro AUC = {macro_partial:.4f}")

    print("\nFull FT (3M, seed=42)")
    probs_full = forward_full_model(args.full_ckpt, args.pretrained,
                                    feats, lorentz, masks, device, args.batch_size)
    curves_full, macro_full = per_class_curves(probs_full, lab_feat)
    print(f"  macro AUC = {macro_full:.4f}")

    # Cache curves so re-rendering doesn't need GPU
    cache_arrays = {}
    for tag, curves in [("frozen", curves_frozen),
                        ("partial", curves_partial),
                        ("full", curves_full)]:
        for name, (fpr, tpr, auc) in curves.items():
            cache_arrays[f"{tag}__{name}__fpr"] = fpr
            cache_arrays[f"{tag}__{name}__tpr"] = tpr
            cache_arrays[f"{tag}__{name}__auc"] = np.array([auc])
    cache_arrays["macro"] = np.array([macro_frozen, macro_partial, macro_full])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path) + "_curves.npz", **cache_arrays)
    print(f"\nCached curves -> {out_path}_curves.npz")

    # Plot the 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8),
                             gridspec_kw=dict(wspace=0.18))
    draw_panel(axes[0], curves_frozen,  macro_frozen,
               r"Frozen Sophon + MLP (3M)", show_ylabel=True)
    draw_panel(axes[1], curves_partial, macro_partial,
               r"Partial FT (3M, last 4 of 8 blocks)", show_ylabel=False)
    draw_panel(axes[2], curves_full,    macro_full,
               r"Full FT (3M)", show_ylabel=False, show_legend=True)

    # Diagonal reference on each panel
    for ax in axes:
        ax.plot([1e-4, 1], [1e-4, 1], color="#888", lw=0.7,
                linestyle=(0, (4, 3)), zorder=0)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.13)
    save_fig(fig, str(out_path))
    print(f"Saved: {out_path}.{{pdf,png}}")


if __name__ == "__main__":
    main()
