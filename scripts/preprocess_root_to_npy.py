#!/usr/bin/env python3
"""Pre-process ROOT files into .npy feature files for fast loading.

Computes 17 Sophon features + Lorentz vectors + masks + labels from ROOT
files and saves as numpy arrays. Subsequent training runs load .npy instead
of re-processing ROOT files — seconds instead of hours.

Usage:
    python scripts/preprocess_root_to_npy.py \
        --input-dir /data/JetClass/Pythia/train_100M \
        --output-dir /data/features/train_100M

    python scripts/preprocess_root_to_npy.py \
        --input-dir /data/JetClass/Pythia/val_5M \
        --output-dir /data/features/val_5M
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jetclass import _process_file, _discover_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="ROOT files directory")
    parser.add_argument("--output-dir", required=True, help="Output .npy directory")
    parser.add_argument("--max-files", type=int, default=None, help="Max files to process (for testing)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files_by_class = _discover_files(args.input_dir)
    if not files_by_class:
        print(f"No ROOT files found in {args.input_dir}")
        return

    total_jets = 0
    t0 = time.time()

    for cls_name, flist in sorted(files_by_class.items()):
        if args.max_files:
            flist = flist[:args.max_files]

        all_feats, all_lv, all_masks, all_labels = [], [], [], []

        for fpath in flist:
            print(f"  {fpath.name}...", end="", flush=True)
            feats, lv, masks, labels = _process_file(str(fpath))
            all_feats.append(feats)
            all_lv.append(lv)
            all_masks.append(masks)
            all_labels.append(labels)
            print(f" {len(labels):,} jets")

        feats = np.concatenate(all_feats)
        lv = np.concatenate(all_lv)
        masks = np.concatenate(all_masks)
        labels = np.concatenate(all_labels)

        # Save as float16 to save space (same as embeddings)
        np.save(out / f"{cls_name}_features.npy", feats.astype(np.float16))
        np.save(out / f"{cls_name}_lorentz.npy", lv.astype(np.float16))
        np.save(out / f"{cls_name}_masks.npy", masks)
        np.save(out / f"{cls_name}_labels.npy", labels)

        n = len(labels)
        total_jets += n
        size_mb = (feats.nbytes + lv.nbytes + masks.nbytes + labels.nbytes) / 1e6
        print(f"  {cls_name}: {n:,} jets, {size_mb:.0f} MB saved")

    elapsed = time.time() - t0
    print(f"\nDone: {total_jets:,} jets in {elapsed/60:.1f} min")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
