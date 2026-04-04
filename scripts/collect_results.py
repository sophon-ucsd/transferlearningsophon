#!/usr/bin/env python3
"""Scan results directory, collect all results.json files into a single CSV.

Usage:
    python scripts/collect_results.py results/
    python scripts/collect_results.py results/full_sweep/ --output results_sweep.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect_results(results_dir: str) -> pd.DataFrame:
    """Recursively find all results.json files, return as DataFrame."""
    results = []
    for rfile in sorted(Path(results_dir).rglob("results.json")):
        try:
            with open(rfile) as f:
                data = json.load(f)
            data["results_path"] = str(rfile)
            results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: could not read {rfile}: {e}")

    if not results:
        print(f"No results.json files found in {results_dir}")
        return pd.DataFrame()

    return pd.DataFrame(results)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (strategy, train_size), compute mean and std across seeds."""
    if df.empty:
        return df

    group_cols = ["strategy", "train_size"]
    metric_cols = [
        c for c in [
            "test_loss", "test_acc", "test_auc_macro",
            "best_val_loss", "best_val_acc", "val_auc_macro",
            "wall_clock_seconds", "num_epochs",
        ]
        if c in df.columns
    ]

    agg_dict = {col: ["mean", "std", "count"] for col in metric_cols}
    summary = df.groupby(group_cols).agg(agg_dict)
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing results.json files")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path")
    parser.add_argument("--summary", "-s", default=None, help="Summary CSV path")
    args = parser.parse_args()

    df = collect_results(args.results_dir)
    if df.empty:
        return

    print(f"Collected {len(df)} results")
    print(f"Strategies: {sorted(df['strategy'].unique())}")
    print(f"Train sizes: {sorted(df['train_size'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    out_path = args.output or str(Path(args.results_dir) / "results_all.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved all results to {out_path}")

    summary = summarize(df)
    sum_path = args.summary or str(Path(args.results_dir) / "results_summary.csv")
    summary.to_csv(sum_path, index=False)
    print(f"Saved summary to {sum_path}")

    print("\n" + "=" * 80)
    print("Summary (mean ± std across seeds):")
    print("=" * 80)
    for strategy in sorted(summary["strategy"].unique()):
        print(f"\n{strategy}:")
        s = summary[summary["strategy"] == strategy].sort_values("train_size")
        for _, row in s.iterrows():
            size = int(row["train_size"])
            acc_mean = row.get("test_acc_mean", float("nan"))
            acc_std = row.get("test_acc_std", float("nan"))
            loss_mean = row.get("test_loss_mean", float("nan"))
            n = int(row.get("test_acc_count", 0))
            print(
                f"  {size:>12,} jets: acc={acc_mean:.4f}±{acc_std:.4f}  "
                f"loss={loss_mean:.4f}  (n={n})"
            )


if __name__ == "__main__":
    main()
