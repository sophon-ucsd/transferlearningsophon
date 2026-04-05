#!/usr/bin/env python3
"""Run the full frozen MLP sweep in a single process.

Loads train/val/test embeddings ONCE, then loops over all
(architecture, train_size, seed) combinations. Each MLP training
takes seconds — the expensive part is the one-time data load.

Usage:
    python scripts/run_frozen_sweep.py \
        --train-dir /data/embeddings_pretrained_full_02_174909_100M \
        --val-dir /data/embeddings_val_5M \
        --test-dir /data/embeddings_test_20M
"""
from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import _load_dir, NPY_CLASS_MAP
from src.data.subsampler import stratified_subsample
from src.models.heads import MLPHead
from src.utils.reproducibility import seed_everything


ARCH_MAP = {
    "small": [128],
    "base": [256],
    "large": [512],
    "deep": [256, 128],
}

SIZES = [10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, 100000000]
SEEDS = [42, 123, 456]
ARCHS = ["small", "base", "large", "deep"]


class ArrayDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embeddings": torch.from_numpy(self.embeddings[idx]),
            "label": int(self.labels[idx]),
        }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in loader:
        emb = batch["embeddings"].to(device)
        labels = batch["label"].to(device)
        logits = model(emb)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embeddings"].to(device)
            labels = batch["label"].to(device)
            logits = model(emb)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)
            all_probs.append(F.softmax(logits, dim=-1).cpu())
            all_labels.append(labels.cpu())
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels.numpy(), all_probs.numpy(),
                            multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return total_loss / total, correct / total, auc


def run_single(train_emb, train_lab, val_loader, test_loader,
               arch, train_size, seed, device, output_dir,
               epochs=50, patience=10, lr=1e-3, wd=0.01, batch_size=4096):
    """Train one MLP config. Data is already loaded — just subsample and go."""
    seed_everything(seed)

    # Subsample training set
    if train_size < len(train_lab):
        indices = stratified_subsample(train_lab, train_size, seed)
        emb = np.ascontiguousarray(train_emb[indices]).astype(np.float32)
        lab = train_lab[indices].copy()
    else:
        emb = np.ascontiguousarray(train_emb).astype(np.float32)
        lab = train_lab.copy()

    train_ds = ArrayDataset(emb, lab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)

    hidden_dims = ARCH_MAP[arch]
    model = MLPHead(128, 10, hidden_dims, dropout=0.1).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start = time.time()
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, device)
    elapsed = time.time() - start

    results = {
        "strategy": "frozen",
        "architecture": arch,
        "hidden_dims": hidden_dims,
        "train_size": train_size,
        "seed": seed,
        "num_classes": 10,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "val_auc_macro": best_val_auc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_auc_macro": test_auc,
        "total_params": param_count,
        "trainable_params": param_count,
        "num_epochs": epoch + 1,
        "wall_clock_seconds": elapsed,
        "lr": lr,
        "dropout": 0.1,
    }

    out = Path(output_dir) / f"frozen_{arch}_{train_size}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="/data/results/frozen_sweep_v3")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === LOAD DATA ONCE ===
    print("\nLoading training embeddings (this takes a few minutes)...")
    t0 = time.time()
    train_emb, train_lab = _load_dir(args.train_dir)
    print(f"Train: {len(train_lab):,} embeddings, loaded in {time.time()-t0:.0f}s")

    print("\nLoading validation embeddings...")
    val_emb, val_lab = _load_dir(args.val_dir)
    val_emb_f32 = np.ascontiguousarray(val_emb).astype(np.float32)
    val_lab = val_lab.copy()
    print(f"Val: {len(val_lab):,} embeddings")

    # Small val subset for per-epoch early stopping (100K is plenty)
    VAL_SUBSET = 100_000
    if len(val_lab) > VAL_SUBSET:
        val_sub_idx = stratified_subsample(val_lab, VAL_SUBSET, seed=0)
        val_small_ds = ArrayDataset(val_emb_f32[val_sub_idx], val_lab[val_sub_idx])
        print(f"Val subset for early stopping: {len(val_sub_idx):,} embeddings")
    else:
        val_small_ds = ArrayDataset(val_emb_f32, val_lab)
    val_small_loader = DataLoader(val_small_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("\nLoading test embeddings...")
    test_emb, test_lab = _load_dir(args.test_dir)
    test_ds = ArrayDataset(np.ascontiguousarray(test_emb).astype(np.float32), test_lab)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test: {len(test_lab):,} embeddings (full eval only at end of each run)")

    # === RUN ALL CONFIGS ===
    total_runs = len(ARCHS) * len(SIZES) * len(SEEDS)
    run_idx = 0

    for arch in ARCHS:
        for size in SIZES:
            if size > len(train_lab):
                print(f"\nSkipping size={size:,} (only {len(train_lab):,} available)")
                run_idx += len(SEEDS)
                continue
            for seed in SEEDS:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] arch={arch} size={size:,} seed={seed}")

                results = run_single(
                    train_emb, train_lab, val_small_loader, test_loader,
                    arch, size, seed, device, args.output_dir,
                    epochs=args.epochs, patience=args.patience,
                    lr=args.lr, batch_size=args.batch_size,
                )

                print(f"  acc={results['test_acc']:.4f} auc={results['test_auc_macro']:.4f} "
                      f"time={results['wall_clock_seconds']:.1f}s")

    print(f"\nALL DONE — {run_idx} runs, results in {args.output_dir}")


if __name__ == "__main__":
    main()
