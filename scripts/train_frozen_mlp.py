#!/usr/bin/env python3
"""Train MLP classifier on precomputed Sophon embeddings.

Fast path for the "frozen" transfer strategy. Each run takes seconds to minutes.

Usage:
    python scripts/train_mlp.py --embeddings-dir embeddings/ --train-size 100000 --seed 42
    python scripts/train_mlp.py --embeddings-dir embeddings/ --train-size 10000 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.embedding_dataset import create_embedding_dataloaders
from src.models.heads import MLPHead
from src.utils.reproducibility import seed_everything


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
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            emb = batch["embeddings"].to(device)
            labels = batch["label"].to(device)

            logits = model(emb)
            loss = F.cross_entropy(logits, labels)
            probs = F.softmax(logits, dim=-1)

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(
            all_labels.numpy(), all_probs.numpy(),
            multi_class="ovr", average="macro",
        )
    except ValueError:
        auc = float("nan")

    return total_loss / total, correct / total, auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=100000)
    parser.add_argument("--test-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sweep-name", default="frozen_mlp")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = create_embedding_dataloaders(
        args.embeddings_dir, args.train_size, args.val_size, args.test_size,
        args.batch_size, num_workers=0, seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset):,}, Val: {len(val_loader.dataset):,}, "
          f"Test: {len(test_loader.dataset):,}")

    # Model
    model = MLPHead(128, args.num_classes, args.hidden_dims, args.dropout).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"MLP params: {param_count:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    start_time = time.time()
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
            )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Test with best model
    model.load_state_dict(best_state)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time

    print(f"\nTest: loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")
    print(f"Time: {elapsed:.1f}s")

    # Save results.json (same format as full pipeline for compatibility)
    results = {
        "strategy": "frozen",
        "train_size": args.train_size or len(train_loader.dataset),
        "seed": args.seed,
        "num_classes": args.num_classes,
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
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "lr": args.lr,
    }

    if args.output_dir is None:
        args.output_dir = f"results/{args.sweep_name}/frozen_{args.train_size}_{args.seed}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
