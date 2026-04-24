#!/usr/bin/env python3
"""Extract 128-dim embeddings from a fine-tuned Sophon checkpoint.

Runs preprocessed .npy features through the fine-tuned backbone and dumps
per-class *_embeddings.npy in the same format as /data/embeddings_val_5M/.

Usage:
    python scripts/extract_embeddings_ft.py \
        --checkpoint /data/results/full_ft/full_ft_3000000_42/best_model.pt \
        --features-dir /data/features/val_5M \
        --output-dir /data/embeddings_ft_full_3M_seed42 \
        --max-per-class 5000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sophon_wrapper import SophonTransferModel


# class name → Sophon label (matches NPY_CLASS_MAP used elsewhere)
CLASS_LABEL = {
    "ZJetsToNuNu": 0, "HToBB": 1, "HToCC": 2, "HToGG": 3, "HToWW4Q": 4,
    "HToWW2Q1L": 5, "ZToQQ": 6, "WToQQ": 7, "TTBar": 8, "TTBarLep": 9,
}


def load_class_files(features_dir: Path) -> dict[str, list[str]]:
    files_by_class: dict[str, list[str]] = {}
    for f in sorted(features_dir.glob("*_features.npy")):
        stem = f.stem.replace("_features", "")
        parts = stem.rsplit("_", 1)
        cls = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        if cls in CLASS_LABEL:
            files_by_class.setdefault(cls, []).append(stem)
    return files_by_class


@torch.no_grad()
def extract_class(model, device, stems, features_dir: Path, max_per_class: int,
                  batch_size: int = 512) -> np.ndarray:
    out = []
    collected = 0
    for stem in stems:
        if collected >= max_per_class:
            break
        feats = np.load(str(features_dir / f"{stem}_features.npy"))
        lv = np.load(str(features_dir / f"{stem}_lorentz.npy"))
        masks = np.load(str(features_dir / f"{stem}_masks.npy"))

        take = min(len(feats), max_per_class - collected)
        feats, lv, masks = feats[:take], lv[:take], masks[:take]

        for i in range(0, take, batch_size):
            f = torch.from_numpy(feats[i:i+batch_size].astype(np.float32)).transpose(1, 2).to(device)
            v = torch.from_numpy(lv[i:i+batch_size].astype(np.float32)).transpose(1, 2).to(device)
            m = torch.from_numpy(masks[i:i+batch_size]).unsqueeze(1).to(device)
            _, emb = model(f, v, m)
            out.append(emb.cpu().numpy().astype(np.float16))
        collected += take
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="fine-tuned best_model.pt")
    p.add_argument("--pretrained-checkpoint", default="models/JetClassII_Sophon/model.pt",
                   help="Sophon pretrained checkpoint for architecture init")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-per-class", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build architecture from pretrained checkpoint (matches training setup)
    model = SophonTransferModel(
        checkpoint_path=args.pretrained_checkpoint,
        num_classes=10, head_type="mlp", export_embed=True,
    )

    # 2. Overwrite with fine-tuned weights
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded fine-tuned: {len(state) - len(unexpected)} keys, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")

    model.to(device).eval()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_by_class = load_class_files(features_dir)
    print(f"Found {len(files_by_class)} classes in {features_dir}")

    for cls, stems in sorted(files_by_class.items()):
        print(f"\nExtracting {cls} (label={CLASS_LABEL[cls]}) from {len(stems)} file(s)...")
        emb = extract_class(model, device, stems, features_dir,
                            args.max_per_class, args.batch_size)
        out_path = output_dir / f"{cls}_embeddings.npy"
        np.save(str(out_path), emb)
        print(f"  Saved: {out_path}  shape={emb.shape} dtype={emb.dtype}")

    print(f"\nAll embeddings in {output_dir}")


if __name__ == "__main__":
    main()
