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

# Embedding file class name -> label index
EMB_CLASS_MAP = {
    "HToBB": 1, "HToCC": 2, "HToGG": 3, "HToWW4Q": 4, "HToWW2Q1L": 5,
    "TTBar": 8, "TTBarLep": 9, "WToQQ": 7, "ZToQQ": 6, "ZJetsToNuNu": 0,
}


def _load_embeddings_subset(emb_dir, target_size, seed):
    """Load embeddings class-balanced, only enough files to cover target_size."""
    from pathlib import Path as P
    d = P(emb_dir)
    emb_files = sorted(d.glob("*_embeddings.npy"))

    # Group by class
    files_by_class = {}
    for f in emb_files:
        cls = f.stem.replace("_embeddings", "")
        if cls in EMB_CLASS_MAP:
            files_by_class[cls] = f

    num_classes = len(files_by_class)
    per_class = target_size // num_classes

    all_emb, all_lab = [], []
    rng = np.random.RandomState(seed)

    for cls, fpath in sorted(files_by_class.items()):
        label = EMB_CLASS_MAP[cls]
        emb = np.load(str(fpath), mmap_mode="r")
        n = len(emb)

        if per_class < n:
            idx = rng.choice(n, size=per_class, replace=False)
            idx.sort()
            chunk = emb[idx]
        else:
            chunk = np.array(emb)

        # Keep as float16 for large sizes to avoid OOM; convert per-batch in __getitem__
        all_emb.append(chunk)
        all_lab.append(np.full(len(chunk), label, dtype=np.int64))

    total = sum(len(e) for e in all_emb)
    if total <= 30_000_000:
        # Small enough to concatenate safely
        return np.concatenate(all_emb), np.concatenate(all_lab)
    else:
        # Return lists — use MultiArrayDataset to avoid np.concatenate OOM
        return all_emb, all_lab


class ArrayDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        emb = torch.from_numpy(self.embeddings[idx].astype(np.float32))
        return {"embeddings": emb, "label": int(self.labels[idx])}


class MultiArrayDataset(Dataset):
    """Dataset from multiple per-class arrays without concatenating.
    Avoids the 50GB peak memory from np.concatenate on 100M embeddings."""
    def __init__(self, emb_list, lab_list):
        self.emb_list = emb_list
        self.lab_list = lab_list
        self.cum_sizes = []
        total = 0
        for e in emb_list:
            total += len(e)
            self.cum_sizes.append(total)
        self._total = total

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        # Find which array this index belongs to
        for i, cum in enumerate(self.cum_sizes):
            if idx < cum:
                local = idx - (self.cum_sizes[i-1] if i > 0 else 0)
                emb = torch.from_numpy(self.emb_list[i][local].astype(np.float32))
                label = int(self.lab_list[i][local])
                return {"embeddings": emb, "label": label}
        raise IndexError(f"Index {idx} out of range")


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

    # Handle both concatenated arrays and list-of-arrays (for large datasets)
    if isinstance(train_emb, list):
        # Multi-array mode — use MultiArrayDataset directly, no concatenation
        emb_list = train_emb
        lab_list = train_lab
        is_multi = True
        total_size = sum(len(e) for e in emb_list)
    else:
        is_multi = False
        total_size = len(train_lab)
        if train_size < total_size:
            indices = stratified_subsample(train_lab, train_size, seed)
            emb = train_emb[indices].astype(np.float32)
            lab = train_lab[indices].copy()
        else:
            emb = np.asarray(train_emb)
            lab = train_lab

    # Scale batch size with dataset — ensure at least ~50 steps per epoch
    if train_size >= 10_000_000:
        bs = 16384
    elif train_size >= 1_000_000:
        bs = 8192
    elif train_size >= 100_000:
        bs = 4096
    elif train_size >= 10_000:
        bs = 512
    else:
        bs = 128

    if is_multi:
        train_ds = MultiArrayDataset(emb_list, lab_list)
    else:
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
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)
        scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

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

    # Load best model for final test
    model.load_state_dict(best_state)

    # Final test — collect per-sample predictions
    model.eval()
    all_probs, all_labels_test = [], []
    test_loss_total = test_correct = test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            emb_b = batch["embeddings"].to(device)
            lab_b = batch["label"].to(device)
            logits = model(emb_b)
            loss = F.cross_entropy(logits, lab_b)
            test_loss_total += loss.item() * len(lab_b)
            test_correct += (logits.argmax(-1) == lab_b).sum().item()
            test_total += len(lab_b)
            all_probs.append(F.softmax(logits, -1).cpu())
            all_labels_test.append(lab_b.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels_test = torch.cat(all_labels_test).numpy()
    test_loss = test_loss_total / test_total
    test_acc = test_correct / test_total
    from sklearn.metrics import roc_auc_score
    try:
        test_auc = roc_auc_score(all_labels_test, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        test_auc = 0.0

    # Per-class AUC
    label_names = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
    per_class_auc = {}
    for cls_idx in range(10):
        binary = (all_labels_test == cls_idx).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            per_class_auc[label_names[cls_idx]] = 0.0
            continue
        try:
            per_class_auc[label_names[cls_idx]] = float(
                roc_auc_score(binary, all_probs[:, cls_idx]))
        except ValueError:
            per_class_auc[label_names[cls_idx]] = 0.0

    elapsed = time.time() - start

    # === Save everything ===
    out = Path(output_dir) / f"frozen_base_{train_size}_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Results JSON
    results = {
        "strategy": "frozen", "architecture": "base", "hidden_dims": [256],
        "train_size": train_size, "seed": seed, "num_classes": 10,
        "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
        "val_auc_macro": best_val_auc,
        "test_loss": test_loss, "test_acc": test_acc, "test_auc_macro": test_auc,
        "per_class_auc": per_class_auc,
        "total_params": param_count, "trainable_params": param_count,
        "num_epochs": epoch + 1, "wall_clock_seconds": elapsed,
        "lr": lr, "dropout": 0.1, "batch_size": bs,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 2. Training history CSV
    import csv
    with open(out / "training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history.keys())
        w.writeheader()
        for i in range(len(history["epoch"])):
            w.writerow({k: history[k][i] for k in history})

    # 3. Model checkpoint
    torch.save(best_state, out / "best_model.pt")

    # 4. Loss curves plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    eps = history["epoch"]
    ax1.plot(eps, history["train_loss"], label="Train", color="#4477AA")
    ax1.plot(eps, history["val_loss"], label="Val", color="#EE6677")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Frozen {train_size:,} s{seed} — Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(eps, history["train_acc"], label="Train Acc", color="#4477AA")
    ax2.plot(eps, history["val_acc"], label="Val Acc", color="#EE6677")
    ax2.plot(eps, history["val_auc"], label="Val AUC", color="#228833", linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Metric")
    ax2.set_title(f"Frozen {train_size:,} s{seed} — Accuracy & AUC")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150); plt.close()

    # 5. ROC curves per class
    from sklearn.metrics import roc_curve, auc as sk_auc
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls_idx in range(10):
        binary = (all_labels_test == cls_idx).astype(int)
        if binary.sum() == 0: continue
        fpr, tpr, _ = roc_curve(binary, all_probs[:, cls_idx])
        area = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label_names[cls_idx]} (AUC={area:.3f})", linewidth=1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — Frozen {train_size:,} s{seed}")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=150); plt.close()

    # 6. Confusion matrix
    from sklearn.metrics import confusion_matrix
    preds = all_probs.argmax(axis=1)
    cm = confusion_matrix(all_labels_test, preds, labels=list(range(10)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(label_names, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(label_names, fontsize=7)
    for i in range(10):
        for j in range(10):
            color = "white" if cm_norm[i,j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", fontsize=6, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion — Frozen {train_size:,} s{seed}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    print(f"    Saved: results.json, training_history.csv, best_model.pt, "
          f"loss_curves.png, roc_curves.png, confusion_matrix.png")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="/data/results/frozen_base")
    parser.add_argument("--sizes", default=None,
                        help="Comma-separated list of train sizes (default: all 9)")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated list of seeds (default: 42,123,456)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs where results.json + best_model.pt already exist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    if args.sizes:
        global SIZES
        SIZES = [int(s) for s in args.sizes.split(",")]
    if args.seeds:
        global SEEDS
        SEEDS = [int(s) for s in args.seeds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load val once (small, 100K subset)
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
    del val_emb, val_lab  # free memory

    # Load test once (500K subset to avoid OOM on sklearn AUC)
    print("\nLoading test embeddings...")
    test_emb, test_lab = _load_dir(args.test_dir)
    TEST_SUBSET = 500_000
    if len(test_lab) > TEST_SUBSET:
        test_idx = stratified_subsample(test_lab, TEST_SUBSET, seed=0)
        test_ds = ArrayDataset(np.ascontiguousarray(test_emb[test_idx]).astype(np.float32), test_lab[test_idx])
        print(f"Test: {TEST_SUBSET:,} subset")
    else:
        test_ds = ArrayDataset(np.ascontiguousarray(test_emb).astype(np.float32), test_lab)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=0)
    print(f"Test: {len(test_ds):,} jets")
    del test_emb, test_lab  # free memory

    # Training embeddings loaded per-run to avoid 25GB+ in memory
    # Just scan the dir to know what's available
    from pathlib import Path as P
    train_files = sorted(P(args.train_dir).glob("*_embeddings.npy"))
    print(f"\nTrain dir: {len(train_files)} embedding files")

    # Run configs
    total = len(SIZES) * len(SEEDS)
    idx = 0
    for size in SIZES:
        for seed in SEEDS:
            idx += 1
            print(f"\n[{idx}/{total}] size={size:,} seed={seed}")

            if args.skip_existing:
                run_dir = Path(args.output_dir) / f"frozen_base_{size}_{seed}"
                if (run_dir / "results.json").exists() and (run_dir / "best_model.pt").exists():
                    print(f"    SKIP — already complete at {run_dir}")
                    continue

            # Load only what we need for this run (class-balanced)
            print(f"    Loading {size:,} training embeddings...")
            train_emb_run, train_lab_run = _load_embeddings_subset(
                args.train_dir, size, seed)
            from collections import Counter
            if isinstance(train_lab_run, list):
                label_counts = Counter()
                for arr in train_lab_run:
                    label_counts.update(arr.tolist())
                total_n = sum(len(e) for e in train_emb_run)
                print(f"    Train labels: {dict(sorted(label_counts.items()))}")
                print(f"    Train shape: ({total_n}, {train_emb_run[0].shape[1]}), "
                      f"dtype: {train_emb_run[0].dtype} (multi-array)")
                print(f"    Train range: [{min(e.min() for e in train_emb_run):.4f}, "
                      f"{max(e.max() for e in train_emb_run):.4f}]")
            else:
                print(f"    Train labels: {dict(sorted(Counter(train_lab_run.tolist()).items()))}")
                print(f"    Train shape: {train_emb_run.shape}, dtype: {train_emb_run.dtype}")
                print(f"    Train range: [{train_emb_run.min():.4f}, {train_emb_run.max():.4f}]")

            results = run_single(train_emb_run, train_lab_run, val_loader, test_loader,
                                 size, seed, device, args.output_dir,
                                 epochs=args.epochs, patience=args.patience)
            print(f"  acc={results['test_acc']:.4f} auc={results['test_auc_macro']:.4f} "
                  f"time={results['wall_clock_seconds']:.1f}s epochs={results['num_epochs']}")

    print(f"\nALL DONE — {idx} runs in {args.output_dir}")


if __name__ == "__main__":
    main()
