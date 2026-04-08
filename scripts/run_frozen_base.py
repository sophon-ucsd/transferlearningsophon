#!/usr/bin/env python3
"""Run frozen MLP sweep — base architecture [256] only.

Loads data once, runs 27 configs (9 sizes × 3 seeds).
Uses larger batch sizes for bigger datasets to reduce steps per epoch.
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

from src.data.embedding_dataset import _load_dir
from src.data.subsampler import stratified_subsample
from src.models.heads import MLPHead
from src.utils.reproducibility import seed_everything


SIZES = [10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, 100000000]
SEEDS = [42, 123, 456]


class ArrayDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {"embeddings": torch.from_numpy(self.embeddings[idx]), "label": int(self.labels[idx])}


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for batch in loader:
        emb = batch["embeddings"].to(device)
        labels = batch["label"].to(device)
        logits = model(emb)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embeddings"].to(device)
            labels = batch["label"].to(device)
            logits = model(emb)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += len(labels)
            all_probs.append(F.softmax(logits, -1).cpu())
            all_labels.append(labels.cpu())
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels.numpy(), all_probs.numpy(), multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return total_loss / total, correct / total, auc


def run_single(train_emb, train_lab, val_loader, test_loader,
               train_size, seed, device, output_dir,
               epochs=50, patience=10, lr=1e-3):
    seed_everything(seed)

    if train_size < len(train_lab):
        indices = stratified_subsample(train_lab, train_size, seed)
        emb = np.ascontiguousarray(train_emb[indices]).astype(np.float32)
        lab = train_lab[indices].copy()
    else:
        emb = np.ascontiguousarray(train_emb).astype(np.float32)
        lab = train_lab.copy()

    # Bigger batch for bigger datasets — fewer steps, faster epochs
    if train_size >= 10_000_000:
        bs = 16384
    elif train_size >= 1_000_000:
        bs = 8192
    else:
        bs = 4096

    train_ds = ArrayDataset(emb, lab)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

    model = MLPHead(128, 10, [256], dropout=0.1).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start = time.time()
    best_val_loss = float("inf")
    best_val_acc = best_val_auc = 0.0
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

    # Final test on full test set
    model.load_state_dict(best_state)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, device)
    elapsed = time.time() - start

    results = {
        "strategy": "frozen", "architecture": "base", "hidden_dims": [256],
        "train_size": train_size, "seed": seed, "num_classes": 10,
        "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
        "val_auc_macro": best_val_auc,
        "test_loss": test_loss, "test_acc": test_acc, "test_auc_macro": test_auc,
        "total_params": param_count, "trainable_params": param_count,
        "num_epochs": epoch + 1, "wall_clock_seconds": elapsed,
        "lr": lr, "dropout": 0.1, "batch_size": bs,
    }

    out = Path(output_dir) / f"frozen_base_{train_size}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="/data/results/frozen_base")
    parser.add_argument("--sizes", default=None,
                        help="Comma-separated list of train sizes (default: all 9)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    if args.sizes:
        global SIZES
        SIZES = [int(s) for s in args.sizes.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all data once
    print("\nLoading training embeddings...")
    t0 = time.time()
    train_emb, train_lab = _load_dir(args.train_dir)
    print(f"Train: {len(train_lab):,} loaded in {time.time()-t0:.0f}s")

    # Small val subset for early stopping
    print("\nLoading validation embeddings...")
    val_emb, val_lab = _load_dir(args.val_dir)
    VAL_SUBSET = 100_000
    if len(val_lab) > VAL_SUBSET:
        val_idx = stratified_subsample(val_lab, VAL_SUBSET, seed=0)
        val_ds = ArrayDataset(np.ascontiguousarray(val_emb[val_idx]).astype(np.float32), val_lab[val_idx])
        print(f"Val: {VAL_SUBSET:,} subset for early stopping")
    else:
        val_ds = ArrayDataset(np.ascontiguousarray(val_emb).astype(np.float32), val_lab)
    val_loader = DataLoader(val_ds, batch_size=8192, shuffle=False, num_workers=0)

    # Full test set — only used once per run at the end
    print("\nLoading test embeddings...")
    test_emb, test_lab = _load_dir(args.test_dir)
    test_ds = ArrayDataset(np.ascontiguousarray(test_emb).astype(np.float32), test_lab)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=0)
    print(f"Test: {len(test_lab):,} (full eval at end of each run)")

    # Run 27 configs
    total = len(SIZES) * len(SEEDS)
    idx = 0
    for size in SIZES:
        if size > len(train_lab):
            print(f"\nSkipping size={size:,} (only {len(train_lab):,} available)")
            idx += len(SEEDS)
            continue
        for seed in SEEDS:
            idx += 1
            print(f"\n[{idx}/{total}] size={size:,} seed={seed}")
            results = run_single(train_emb, train_lab, val_loader, test_loader,
                                 size, seed, device, args.output_dir,
                                 epochs=args.epochs, patience=args.patience)
            print(f"  acc={results['test_acc']:.4f} auc={results['test_auc_macro']:.4f} "
                  f"time={results['wall_clock_seconds']:.1f}s epochs={results['num_epochs']}")

    print(f"\nALL DONE — {idx} runs in {args.output_dir}")


if __name__ == "__main__":
    main()
