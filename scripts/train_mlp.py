#!/usr/bin/env python3
"""
Train an MLP head on Sophon embeddings for JetClass classification.

Example:
    python scripts/train_mlp.py --emb-dir embeddings/ --hidden-layers 256,128,64 --epochs 25
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Constants
CLASSES_10 = [
    "HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q",
    "TTBar", "TTBarLep", "WToQQ", "ZToQQ", "ZJetsToNuNu",
]
EMB_PREFIX = "emb_"


def parse_classes(arg: str) -> list:
    """Parse comma-separated class names, falling back to CLASSES_10."""
    if arg is None:
        return CLASSES_10
    names = [c.strip() for c in arg.split(",") if c.strip()]
    if not names:
        return CLASSES_10
    return names


# MLP Model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



# Data Loading
def find_emb_file(emb_dir: Path, class_name: str) -> Path:
    """Find embedding file (NPY or CSV) for a given class."""
    npy_patterns = [
        f"{class_name}_embeddings.npy",
    ]
    csv_patterns = [
        f"{class_name}_inference_with_embedding.csv",
        f"{class_name}.csv",
        f"{class_name}_embeddings.csv",
    ]
    if class_name == "ZJetsToNuNu":
        npy_patterns.append("ZToNuNu_embeddings.npy")
        csv_patterns.append("ZToNuNu_inference_with_embedding.csv")
    for p in npy_patterns + csv_patterns:
        fp = emb_dir / p
        if fp.exists():
            return fp
    raise FileNotFoundError(f"No embedding file found for {class_name} in {emb_dir}")


def load_embeddings(emb_dir: Path, class_names: list, per_class_cap: int = None):
    """Load embeddings from NPY or CSV files. Uses mmap for NPY to avoid loading all into RAM."""
    all_X = []
    all_y = []
    for idx, cls in enumerate(class_names):
        fp = find_emb_file(emb_dir, cls)

        if fp.suffix == ".npy":
            arr = np.load(fp, mmap_mode="r")
            n = arr.shape[0]
            if per_class_cap and n > per_class_cap:
                rng = np.random.RandomState(42)
                indices = rng.choice(n, per_class_cap, replace=False)
                indices.sort()
                X_cls = arr[indices].astype(np.float32)
            else:
                X_cls = np.array(arr, dtype=np.float32)
            all_X.append(X_cls)
            all_y.append(np.full(len(X_cls), idx, dtype=np.int64))
        else:
            header = pd.read_csv(fp, nrows=0).columns.tolist()
            emb_cols = [c for c in header if c.startswith("emb_")]
            df = pd.read_csv(fp, usecols=emb_cols)
            if per_class_cap and len(df) > per_class_cap:
                df = df.sample(n=per_class_cap, random_state=42)
            all_X.append(df.values.astype(np.float32))
            all_y.append(np.full(len(df), idx, dtype=np.int64))

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    emb_cols = [f"emb_{i}" for i in range(X.shape[1])]
    result = pd.DataFrame(X, columns=emb_cols)
    result["label"] = y
    return result


def get_embedding_columns(df: pd.DataFrame) -> list:
    """Get embedding column names."""
    return sorted([c for c in df.columns if c.startswith(EMB_PREFIX)],
                  key=lambda x: int(x.replace(EMB_PREFIX, "")))


# Training
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    probs = np.array(all_probs)
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(all_labels, probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return avg_loss, acc, auc, np.array(all_labels), probs



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


def plot_loss_curves(history, out_path):
    """Plot training and validation loss curves."""
    epochs = [h["epoch"] for h in history]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [h["val_acc"] for h in history], label="Val Accuracy")
    ax2.plot(epochs, [h["val_auc"] for h in history], label="Val AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curves to {out_path}")


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


# Main
def main():
    parser = argparse.ArgumentParser(description="Train MLP on Sophon embeddings")
    parser.add_argument("--emb-dir", type=str, default="embeddings/", help="Embeddings directory")
    parser.add_argument("--out-dir", type=str, default="results/", help="Output directory")
    parser.add_argument("--hidden-layers", type=str, default="256,128,64", help="Hidden layer sizes, comma-separated")
    parser.add_argument("--per-class-cap", type=int, default=None, help="Max samples per class")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled, stop after N epochs with no val loss improvement)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation set fraction")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names (default: all 10 JetClass classes)")
    args = parser.parse_args()

    t_start = time.time()

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]
    class_names = parse_classes(args.classes)
    num_classes = len(class_names)
    print(f"Hidden layers: {hidden_layers}")
    print(f"Classes ({num_classes}): {class_names}")

    # Load data
    print("Loading embeddings...")
    t_load = time.time()
    df = load_embeddings(emb_dir, class_names, args.per_class_cap)
    print(f"Total samples: {len(df):,} (loaded in {format_time(time.time() - t_load)})")

    emb_cols = get_embedding_columns(df)
    print(f"Embedding dimension: {len(emb_cols)}")

    X = df[emb_cols].values.astype(np.float32)
    y = df["label"].values

    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_frac, stratify=y, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_frac, stratify=y_trainval, random_state=args.seed
    )
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create dataloaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model
    model = MLPClassifier(len(emb_cols), hidden_layers, num_classes, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Training loop
    best_val_loss = float("inf")
    best_val_auc = 0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"\nTraining (max {args.epochs} epochs, patience={args.patience or 'off'})...")
    t_train = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - t_epoch

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "epoch_time_s": epoch_time,
        })

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        marker = "*" if improved else ""
        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | {format_time(epoch_time)} {marker}")

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    train_time = time.time() - t_train
    print(f"\nTraining complete in {format_time(train_time)} ({epoch} epochs, best epoch: {best_epoch})")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    # Plot loss curves
    plot_loss_curves(history, out_dir / "loss_curves.png")

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_auc, y_test_labels, y_test_probs = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC (macro OvR): {test_auc:.4f}")
    print(f"  Best Val AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    # Per-class AUC
    per_class_aucs = {}
    for k, cls in enumerate(class_names):
        y_bin = (y_test_labels == k).astype(int)
        if y_bin.sum() > 0:
            cls_auc = roc_auc_score(y_bin, y_test_probs[:, k])
            per_class_aucs[cls] = cls_auc
            print(f"    {cls}: AUC={cls_auc:.4f}")

    total_time = time.time() - t_start

    # Save results
    results = {
        "hidden_layers": str(hidden_layers),
        "n_params": n_params,
        "per_class_cap": args.per_class_cap or "all",
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_auc_macro_ovr": test_auc,
        "best_val_auc": best_val_auc,
        "train_time_s": round(train_time, 1),
        "total_time_s": round(total_time, 1),
        **{f"auc_{cls}": v for cls, v in per_class_aucs.items()},
    }

    results_file = out_dir / "train_results.csv"
    pd.DataFrame([results]).to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

    # Plot ROC
    arch_name = "_".join(map(str, hidden_layers))
    roc_path = out_dir / f"roc_mlp_{arch_name}.png"
    plot_roc_curves(y_test_labels, y_test_probs, class_names, f"MLP ROC (arch={hidden_layers})", roc_path)

    # Always save model weights
    model_path = out_dir / f"mlp_{arch_name}.pt"
    torch.save({
        "model_state_dict": best_model_state,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "hidden_layers": hidden_layers,
        "num_classes": num_classes,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "args": vars(args),
    }, model_path)
    print(f"Saved model to {model_path}")

    print(f"\nTotal time: {format_time(total_time)}")


if __name__ == "__main__":
    main()
