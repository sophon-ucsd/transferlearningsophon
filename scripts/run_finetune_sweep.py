#!/usr/bin/env python3
"""Run fine-tune sweep in a single process with pre-processed .npy features.

Loads features ONCE, then loops over (strategy, size, seed) configs.
Each run: subsample train data, create model, train, evaluate, save results.

Usage:
    python scripts/run_finetune_sweep.py \
        --train-dir /data/features/train_100M \
        --val-dir /data/features/val_5M \
        --test-dir /data/features/test_20M \
        --strategy full_ft \
        --checkpoint models/JetClassII_Sophon/model.pt
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sophon_wrapper import create_model, SophonTransferModel
from src.utils.reproducibility import seed_everything


SIZES = [10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, 100000000]
SEEDS = [42, 123, 456]


class FeatureDataset(Dataset):
    """Dataset from pre-processed .npy feature files."""

    def __init__(self, features, lorentz, masks, labels):
        self.features = features      # (N, 128, 17) float16 or float32
        self.lorentz = lorentz         # (N, 128, 4)
        self.masks = masks             # (N, 128) bool
        self.labels = labels           # (N,) int

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Transpose to model format: (17, 128) and (4, 128)
        f = torch.from_numpy(self.features[idx].astype(np.float32)).T
        lv = torch.from_numpy(self.lorentz[idx].astype(np.float32)).T
        m = torch.from_numpy(self.masks[idx]).unsqueeze(0)  # (1, 128)
        return {
            "features": f,
            "lorentz_vectors": lv,
            "mask": m,
            "label": int(self.labels[idx]),
        }


def collate_fn(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "lorentz_vectors": torch.stack([b["lorentz_vectors"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def load_npy_features(data_dir, max_jets=None):
    """Load pre-processed .npy feature files from a directory.

    Distributes max_jets evenly across classes to ensure all classes
    are represented. Groups files by class name prefix.
    """
    d = Path(data_dir)
    feature_files = sorted(d.glob("*_features.npy"))
    if not feature_files:
        raise FileNotFoundError(f"No *_features.npy files in {data_dir}")

    # Group files by class (e.g. HToBB_000, HToBB_001 -> HToBB)
    files_by_class: dict[str, list[str]] = {}
    for fpath in feature_files:
        stem = fpath.stem.replace("_features", "")
        # Class name: everything before the last _NNN
        parts = stem.rsplit("_", 1)
        cls = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        files_by_class.setdefault(cls, []).append(stem)

    num_classes = len(files_by_class)
    per_class = max_jets // num_classes if max_jets else None

    all_f, all_lv, all_m, all_lab = [], [], [], []
    total = 0

    for cls, stems in sorted(files_by_class.items()):
        cls_count = 0
        for stem in stems:
            if per_class and cls_count >= per_class:
                break
            feats = np.load(str(d / f"{stem}_features.npy"))
            lv = np.load(str(d / f"{stem}_lorentz.npy"))
            masks = np.load(str(d / f"{stem}_masks.npy"))
            labels = np.load(str(d / f"{stem}_labels.npy"))

            if per_class and cls_count + len(feats) > per_class:
                need = per_class - cls_count
                feats, lv, masks, labels = feats[:need], lv[:need], masks[:need], labels[:need]

            all_f.append(feats)
            all_lv.append(lv)
            all_m.append(masks)
            all_lab.append(labels)
            cls_count += len(feats)
            total += len(feats)

        print(f"  {cls}: {cls_count:,} jets")

    features = np.concatenate(all_f)
    lorentz = np.concatenate(all_lv)
    masks = np.concatenate(all_m)
    labels = np.concatenate(all_lab)

    print(f"  Total loaded: {len(labels):,} jets, {num_classes} classes")
    return features, lorentz, masks, labels


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for batch in loader:
        f = batch["features"].to(device)
        lv = batch["lorentz_vectors"].to(device)
        m = batch["mask"].to(device)
        lab = batch["label"].to(device)

        logits, _ = model(f, lv, m)
        loss = F.cross_entropy(logits, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(lab)
        correct += (logits.argmax(-1) == lab).sum().item()
        total += len(lab)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(device)
            lv = batch["lorentz_vectors"].to(device)
            m = batch["mask"].to(device)
            lab = batch["label"].to(device)

            logits, _ = model(f, lv, m)
            loss = F.cross_entropy(logits, lab)
            total_loss += loss.item() * len(lab)
            correct += (logits.argmax(-1) == lab).sum().item()
            total += len(lab)
            all_probs.append(F.softmax(logits, -1).cpu())
            all_labels.append(lab.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return total_loss / total, correct / total, auc


def run_single(train_dir,
               val_loader, test_loader,
               strategy, checkpoint, train_size, seed, device, output_dir,
               frozen_layers=4, lr=1e-3, backbone_lr=1e-4,
               epochs=100, patience=10, batch_size=256):
    seed_everything(seed)

    # Load only enough training data for this run
    print(f"    Loading {train_size:,} training jets...")
    f, lv, m, lab = load_npy_features(train_dir, max_jets=train_size)

    train_ds = FeatureDataset(f, lv, m, lab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)

    # Create fresh model for each run
    model = create_model(strategy, checkpoint, num_classes=10, frozen_layers=frozen_layers)
    model = model.to(device)
    total_params, trainable_params = SophonTransferModel.count_params(model)

    # Optimizer — differential LR for any pretrained strategy
    if strategy in ("full_ft", "partial_ft"):
        param_groups = model.get_param_groups(backbone_lr, lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

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

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"    early stopping at epoch {epoch+1}")
            break

    # Final test
    model.load_state_dict(best_state)
    test_loss, test_acc, test_auc = evaluate(model, test_loader, device)
    elapsed = time.time() - start

    results = {
        "strategy": strategy,
        "train_size": train_size,
        "seed": seed,
        "num_classes": 10,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "val_auc_macro": best_val_auc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_auc_macro": test_auc,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "num_epochs": epoch + 1,
        "wall_clock_seconds": elapsed,
    }

    out = Path(output_dir) / f"{strategy}_{train_size}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, help="Pre-processed train features dir")
    parser.add_argument("--val-dir", required=True, help="Pre-processed val features dir")
    parser.add_argument("--test-dir", required=True, help="Pre-processed test features dir")
    parser.add_argument("--strategy", required=True, choices=["from_scratch", "frozen", "partial_ft", "full_ft"])
    parser.add_argument("--checkpoint", default=None, help="Sophon checkpoint path")
    parser.add_argument("--output-dir", default="/data/results")
    parser.add_argument("--sizes", default=None, help="Comma-separated sizes")
    parser.add_argument("--frozen-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=100000)
    parser.add_argument("--test-size", type=int, default=500000)
    args = parser.parse_args()

    if args.sizes:
        global SIZES
        SIZES = [int(s) for s in args.sizes.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Strategy: {args.strategy}")

    # Load val and test once (small, stays in memory)
    print("\nLoading val features...")
    val_f, val_lv, val_m, val_lab = load_npy_features(args.val_dir, max_jets=args.val_size)
    val_ds = FeatureDataset(val_f, val_lv, val_m, val_lab)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    print(f"Val: {len(val_lab):,} jets")

    print("\nLoading test features...")
    test_f, test_lv, test_m, test_lab = load_npy_features(args.test_dir, max_jets=args.test_size)
    test_ds = FeatureDataset(test_f, test_lv, test_m, test_lab)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)
    print(f"Test: {len(test_lab):,} jets")

    # Run sweep — train data loaded per-run (only what's needed)
    total = len(SIZES) * len(SEEDS)
    idx = 0
    for size in SIZES:
        for seed in SEEDS:
            idx += 1
            print(f"\n[{idx}/{total}] strategy={args.strategy} size={size:,} seed={seed}")

            results = run_single(
                args.train_dir,
                val_loader, test_loader,
                args.strategy, args.checkpoint, size, seed, device,
                args.output_dir,
                frozen_layers=args.frozen_layers,
                lr=args.lr, backbone_lr=args.backbone_lr,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size,
            )

            print(f"  acc={results['test_acc']:.4f} auc={results['test_auc_macro']:.4f} "
                  f"time={results['wall_clock_seconds']:.1f}s epochs={results['num_epochs']}")

    print(f"\nALL DONE — {idx} runs in {args.output_dir}")


if __name__ == "__main__":
    main()
