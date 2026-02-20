#!/usr/bin/env python3
"""
Evaluate baseline Sophon model (using pre-computed logits, no MLP).

Example:
    python scripts/evaluate_baseline.py --emb-dir embeddings/ --out-dir results/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt



# Constants
CLASSES_10 = [
    "HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q",
    "TTBar", "TTBarLep", "WToQQ", "ZToQQ", "ZToNuNu",
]


def parse_classes(arg: str) -> list:
    """Parse comma-separated class names, falling back to CLASSES_10."""
    if arg is None:
        return CLASSES_10
    names = [c.strip() for c in arg.split(",") if c.strip()]
    if not names:
        return CLASSES_10
    return names



# Data Loading
def find_csv_for_class(emb_dir: Path, class_name: str) -> Path:
    """Find CSV file for a given class."""
    patterns = [
        f"{class_name}_inference_with_embedding.csv",
        f"{class_name}.csv",
        f"{class_name}_embeddings.csv",
    ]
    for p in patterns:
        fp = emb_dir / p
        if fp.exists():
            return fp
    raise FileNotFoundError(f"No CSV found for {class_name} in {emb_dir}")


def load_data_with_logits(emb_dir: Path, class_names: list, per_class_cap: int = None):
    """Load embeddings with logit columns."""
    rows = []
    for idx, cls in enumerate(class_names):
        fp = find_csv_for_class(emb_dir, cls)
        df = pd.read_csv(fp)
        if per_class_cap and len(df) > per_class_cap:
            df = df.sample(n=per_class_cap, random_state=42)
        df["label"] = idx
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def get_logit_columns(df: pd.DataFrame) -> list:
    """Get logit column names."""
    cols = [c for c in df.columns if c.startswith("logit_")]
    if not cols:
        raise ValueError("No logit columns found. Run inference_all_classes.py first.")
    return sorted(cols, key=lambda x: int(x.replace("logit_", "")))


def softmax(x):
    """Apply softmax to logits."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)



# Plotting
def plot_roc_curves(y_true, y_prob, class_names, title, out_path):
    """Plot one-vs-rest ROC curves."""
    plt.figure(figsize=(10, 8))
    for k, cls in enumerate(class_names):
        y_bin = (y_true == k).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, k])
        auc_score = roc_auc_score(y_bin, y_prob[:, k])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc_score:.4f})")
    
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC plot to {out_path}")



# Main
def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline Sophon model")
    parser.add_argument("--emb-dir", type=str, default="embeddings/", help="Embeddings directory")
    parser.add_argument("--out-dir", type=str, default="results/", help="Output directory")
    parser.add_argument("--per-class-cap", type=int, default=None, help="Max samples per class")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names (default: all 10 JetClass classes)")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_classes(args.classes)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Load data
    print("Loading data with logits...")
    df = load_data_with_logits(emb_dir, class_names, args.per_class_cap)
    print(f"Total samples: {len(df):,}")
    
    logit_cols = get_logit_columns(df)
    print(f"Found {len(logit_cols)} logit columns")
    
    logits = df[logit_cols].values.astype(np.float32)
    labels = df["label"].values
    
    # Split (same as MLP training for fair comparison)
    _, logits_test, _, y_test = train_test_split(
        logits, labels, test_size=args.test_frac, stratify=labels, random_state=args.seed
    )
    
    print(f"Test samples: {len(logits_test):,}")
    
    # Convert logits to probabilities
    probs = softmax(logits_test)
    preds = probs.argmax(axis=1)
    
    # Compute metrics
    acc = accuracy_score(y_test, preds)
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_test, probs[:, 1])
        else:
            auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    print(f"\nBaseline Sophon Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC (macro OvR): {auc:.4f}")

    # Per-class AUCs
    print("\n  Per-class AUC:")
    per_class_aucs = {}
    for k, cls in enumerate(class_names):
        y_bin = (y_test == k).astype(int)
        if y_bin.sum() > 0:
            cls_auc = roc_auc_score(y_bin, probs[:, k])
            per_class_aucs[cls] = cls_auc
            print(f"    {cls}: {cls_auc:.4f}")
    
    # Save results
    results = {
        "model": "baseline_sophon",
        "per_class_cap": args.per_class_cap or "all",
        "test_acc": acc,
        "test_auc_macro_ovr": auc,
        **{f"auc_{cls}": v for cls, v in per_class_aucs.items()}
    }
    
    results_file = out_dir / "baseline_results.csv"
    pd.DataFrame([results]).to_csv(results_file, index=False)
    print(f"\nSaved results to {results_file}")
    
    # Plot ROC
    roc_path = out_dir / "roc_baseline.png"
    plot_roc_curves(y_test, probs, class_names, "Baseline Sophon ROC (test set)", roc_path)


if __name__ == "__main__":
    main()
