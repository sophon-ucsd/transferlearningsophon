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

        for fpath in flist:
            fname = fpath.stem  # e.g. HToBB_042
            print(f"  {fpath.name}...", end="", flush=True)
            feats, lv, masks, labels = _process_file(str(fpath))

            np.save(out / f"{fname}_features.npy", feats.astype(np.float16))
            np.save(out / f"{fname}_lorentz.npy", lv.astype(np.float16))
            np.save(out / f"{fname}_masks.npy", masks)
            np.save(out / f"{fname}_labels.npy", labels)

            total_jets += len(labels)
            print(f" {len(labels):,} jets saved")

    elapsed = time.time() - t0
    print(f"\nDone: {total_jets:,} jets in {elapsed/60:.1f} min")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
