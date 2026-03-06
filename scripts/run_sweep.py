#!/usr/bin/env python3
"""
Run sweeps over architecture or dataset size.

Examples:
    # Architecture sweep (fixed 1M/class, vary MLP shape)
    python scripts/run_sweep.py --sweep arch --emb-dir embeddings/ --out-dir results/arch_sweep/

    # Dataset size sweep (fixed medium MLP, vary data, early stopping)
    python scripts/run_sweep.py --sweep size --emb-dir embeddings/ --out-dir results/size_sweep/
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd


# Sweep Configurations
ARCH_SWEEP = {
    "tiny": [64],
    "small": [128, 64],
    "medium": [256, 128, 64],
    "large": [512, 256, 128, 64],
    "xlarge": [1024, 512, 256, 128],
    "xxlarge": [1024, 512, 256, 128, 64],
    "wide": [2048, 1024, 512],
}

SIZE_SWEEP = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]


def run_command(cmd: list, verbose: bool = True):
    """Run a command and return exit code."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=not verbose)
    return result.returncode


# Sweep Functions
def run_arch_sweep(args):
    """Run architecture sweep at fixed dataset size."""
    results = []
    t0 = time.time()

    for arch_name, hidden_layers in ARCH_SWEEP.items():
        print(f"\n\n{'#'*60}")
        print(f"# Architecture: {arch_name} = {hidden_layers}")
        print(f"{'#'*60}")

        arch_out_dir = Path(args.out_dir) / arch_name
        arch_out_dir.mkdir(parents=True, exist_ok=True)

        hidden_str = ",".join(map(str, hidden_layers))

        cmd = [
            sys.executable, "scripts/train_mlp.py",
            "--emb-dir", args.emb_dir,
            "--out-dir", str(arch_out_dir),
            "--hidden-layers", hidden_str,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--per-class-cap", str(args.per_class_cap) if args.per_class_cap else "1000000",
            "--seed", str(args.seed),
        ]
        if args.classes:
            cmd += ["--classes", args.classes]

        ret = run_command(cmd)

        if ret == 0:
            results_file = arch_out_dir / "train_results.csv"
            if results_file.exists():
                df = pd.read_csv(results_file)
                df["arch_name"] = arch_name
                results.append(df)

    # Aggregate results
    if results:
        all_results = pd.concat(results, ignore_index=True)
        out_file = Path(args.out_dir) / "arch_sweep_results.csv"
        all_results.to_csv(out_file, index=False)
        print(f"\n\nSaved aggregated results to {out_file}")
        print(all_results.to_string())

    elapsed = time.time() - t0
    print(f"\nArch sweep total time: {elapsed / 60:.1f} min")


def run_size_sweep(args):
    """Run dataset size sweep with fixed architecture and early stopping."""
    results = []
    t0 = time.time()

    for size in SIZE_SWEEP:
        print(f"\n\n{'#'*60}")
        print(f"# Dataset size: {size:,} per class ({size * 10:,} total)")
        print(f"{'#'*60}")

        size_out_dir = Path(args.out_dir) / f"size_{size}"
        size_out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "scripts/train_mlp.py",
            "--emb-dir", args.emb_dir,
            "--out-dir", str(size_out_dir),
            "--hidden-layers", args.hidden_layers,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--per-class-cap", str(size),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
        ]
        if args.classes:
            cmd += ["--classes", args.classes]

        ret = run_command(cmd)

        if ret == 0:
            results_file = size_out_dir / "train_results.csv"
            if results_file.exists():
                df = pd.read_csv(results_file)
                df["per_class_cap"] = size
                results.append(df)

    # Aggregate results
    if results:
        all_results = pd.concat(results, ignore_index=True)
        out_file = Path(args.out_dir) / "size_sweep_results.csv"
        all_results.to_csv(out_file, index=False)
        print(f"\n\nSaved aggregated results to {out_file}")
        print(all_results.to_string())

    elapsed = time.time() - t0
    print(f"\nSize sweep total time: {elapsed / 60:.1f} min")


# Main
def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps")
    parser.add_argument("--sweep", type=str, required=True, choices=["arch", "size"], help="Sweep type")
    parser.add_argument("--emb-dir", type=str, default="embeddings/", help="Embeddings directory")
    parser.add_argument("--out-dir", type=str, default="results/sweep/", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs per run")
    parser.add_argument("--batch-size", type=int, default=8192, help="Training batch size")
    parser.add_argument("--per-class-cap", type=int, default=1000000, help="Samples per class (for arch sweep)")
    parser.add_argument("--hidden-layers", type=str, default="256,128,64", help="Architecture (for size sweep)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience for size sweep (0=disabled)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names (passed to train_mlp.py / evaluate_baseline.py)")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.sweep == "arch":
        run_arch_sweep(args)
    elif args.sweep == "size":
        run_size_sweep(args)


if __name__ == "__main__":
    main()
