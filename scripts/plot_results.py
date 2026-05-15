#!/usr/bin/env python3
"""
Generate plots from experiment results.

Examples:
    python scripts/plot_results.py --plot-type arch_sweep --results-csv results/arch_sweep_results.csv
    python scripts/plot_results.py --plot-type size_sweep --results-csv results/size_sweep_results.csv
    python scripts/plot_results.py --plot-type comparison --mlp-results results/train_results.csv --baseline-results results/baseline_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# Use non-interactive backend for cluster
matplotlib.use("Agg")

# Style settings
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# Plot Functions
def plot_arch_sweep(results_csv: Path, out_dir: Path):
    """Plot architecture sweep results."""
    df = pd.read_csv(results_csv)
    
    # Sort by model size (sum of hidden layers)
    df["model_size"] = df["hidden_layers"].apply(lambda x: sum(eval(x)))
    df = df.sort_values("model_size")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    ax = axes[0]
    ax.bar(df["arch_name"], df["test_acc"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Architecture")
    ax.set_ylim(0, max(df["test_acc"]) * 1.15)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["test_acc"] + 0.01, f"{row['test_acc']:.3f}", ha="center", fontsize=9)
    
    # AUC plot
    ax = axes[1]
    ax.bar(df["arch_name"], df["test_auc_macro_ovr"], color="darkorange", alpha=0.8)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Test AUC (macro OvR)")
    ax.set_title("AUC vs Architecture")
    ax.set_ylim(0.5, 1.0)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["test_auc_macro_ovr"] + 0.01, f"{row['test_auc_macro_ovr']:.3f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    out_path = out_dir / "arch_sweep_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_size_sweep(results_csv: Path, out_dir: Path):
    """Plot dataset size sweep results."""
    df = pd.read_csv(results_csv)
    df = df.sort_values("per_class_cap")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy vs size
    ax = axes[0]
    ax.plot(df["per_class_cap"], df["test_acc"], "o-", color="steelblue", markersize=8, linewidth=2)
    ax.set_xlabel("Samples per Class")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Dataset Size")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    
    # AUC vs size
    ax = axes[1]
    ax.plot(df["per_class_cap"], df["test_auc_macro_ovr"], "o-", color="darkorange", markersize=8, linewidth=2)
    ax.set_xlabel("Samples per Class")
    ax.set_ylabel("Test AUC (macro OvR)")
    ax.set_title("AUC vs Dataset Size")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    out_path = out_dir / "size_sweep_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison(mlp_csv: Path, baseline_csv: Path, out_dir: Path):
    """Plot MLP vs Baseline comparison."""
    mlp_df = pd.read_csv(mlp_csv)
    baseline_df = pd.read_csv(baseline_csv)
    
    mlp_acc = mlp_df["test_acc"].iloc[0]
    mlp_auc = mlp_df["test_auc_macro_ovr"].iloc[0]
    baseline_acc = baseline_df["test_acc"].iloc[0]
    baseline_auc = baseline_df["test_auc_macro_ovr"].iloc[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Accuracy comparison
    ax = axes[0]
    x = [0, 1]
    values = [baseline_acc, mlp_acc]
    colors = ["gray", "steelblue"]
    labels = ["Linear Probe", "MLP Head"]
    bars = ax.bar(x, values, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(0, max(values) * 1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=11)
    
    # AUC comparison
    ax = axes[1]
    values = [baseline_auc, mlp_auc]
    bars = ax.bar(x, values, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test AUC (macro OvR)")
    ax.set_title("AUC Comparison")
    ax.set_ylim(0.5, 1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=11)
    
    plt.tight_layout()
    out_path = out_dir / "comparison_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# Main
def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("--plot-type", type=str, required=True, 
                        choices=["arch_sweep", "size_sweep", "comparison"],
                        help="Type of plot to generate")
    parser.add_argument("--results-csv", type=str, help="Path to results CSV (for sweep plots)")
    parser.add_argument("--mlp-results", type=str, help="Path to MLP results CSV (for comparison)")
    parser.add_argument("--baseline-results", type=str, help="Path to baseline results CSV (for comparison)")
    parser.add_argument("--out-dir", type=str, default="results/plots/", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_type == "arch_sweep":
        if not args.results_csv:
            parser.error("--results-csv required for arch_sweep")
        plot_arch_sweep(Path(args.results_csv), out_dir)
        
    elif args.plot_type == "size_sweep":
        if not args.results_csv:
            parser.error("--results-csv required for size_sweep")
        plot_size_sweep(Path(args.results_csv), out_dir)
        
    elif args.plot_type == "comparison":
        if not args.mlp_results or not args.baseline_results:
            parser.error("--mlp-results and --baseline-results required for comparison")
        plot_comparison(Path(args.mlp_results), Path(args.baseline_results), out_dir)


if __name__ == "__main__":
    main()
