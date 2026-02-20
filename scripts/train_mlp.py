#!/usr/bin/env python3
"""
Train an MLP head on Sophon embeddings for JetClass classification.

Example:
    python scripts/train_mlp.py --emb-dir embeddings/ --hidden-layers 256,128,64 --epochs 25
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# Constants
CLASSES_10 = [
    "HToBB", "HToCC", "HToGG", "HToWW2Q1L", "HToWW4Q",
    "TTBar", "TTBarLep", "WToQQ", "ZToQQ", "ZToNuNu",
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


def load_embeddings(emb_dir: Path, class_names: list, per_class_cap: int = None):
    """Load embeddings from CSV files."""
    rows = []
    for idx, cls in enumerate(class_names):
        fp = find_csv_for_class(emb_dir, cls)
        df = pd.read_csv(fp)
        if per_class_cap and len(df) > per_class_cap:
            df = df.sample(n=per_class_cap, random_state=42)
        df["label"] = idx
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def get_embedding_columns(df: pd.DataFrame) -> list:
    """Get embedding column names."""
    return sorted([c for c in df.columns if c.startswith(EMB_PREFIX)],
                  key=lambda x: int(x.replace(EMB_PREFIX, "")))


def get_logit_columns(df: pd.DataFrame) -> list:
    """Get logit column names if present."""
    cols = [c for c in df.columns if c.startswith("logit_")]
    return sorted(cols, key=lambda x: int(x.replace("logit_", ""))) if cols else []



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
            # Binary: use probability of class 1
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



# Main
def main():
    parser = argparse.ArgumentParser(description="Train MLP on Sophon embeddings")
    parser.add_argument("--emb-dir", type=str, default="embeddings/", help="Embeddings directory")
    parser.add_argument("--out-dir", type=str, default="results/", help="Output directory")
    parser.add_argument("--hidden-layers", type=str, default="256,128,64", help="Hidden layer sizes, comma-separated")
    parser.add_argument("--per-class-cap", type=int, default=None, help="Max samples per class")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation set fraction")
    parser.add_argument("--save-model", action="store_true", help="Save model checkpoint")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names (default: all 10 JetClass classes)")
    args = parser.parse_args()

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
    df = load_embeddings(emb_dir, class_names, args.per_class_cap)
    print(f"Total samples: {len(df):,}")
    
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
    
    # Training loop
    best_val_auc = 0
    best_model_state = None
    
    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
    
    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_auc, y_test_labels, y_test_probs = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC (macro OvR): {test_auc:.4f}")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    
    # Save results
    results = {
        "hidden_layers": str(hidden_layers),
        "per_class_cap": args.per_class_cap or "all",
        "epochs": args.epochs,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_auc_macro_ovr": test_auc,
        "best_val_auc": best_val_auc,
    }
    
    results_file = out_dir / "train_results.csv"
    pd.DataFrame([results]).to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")
    
    # Plot ROC
    arch_name = "_".join(map(str, hidden_layers))
    roc_path = out_dir / f"roc_mlp_{arch_name}.png"
    plot_roc_curves(y_test_labels, y_test_probs, class_names, f"MLP ROC (arch={hidden_layers})", roc_path)
    
    # Save model
    if args.save_model:
        model_path = out_dir / f"mlp_{arch_name}.pt"
        torch.save({
            "model_state_dict": best_model_state,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "hidden_layers": hidden_layers,
            "args": vars(args),
        }, model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
