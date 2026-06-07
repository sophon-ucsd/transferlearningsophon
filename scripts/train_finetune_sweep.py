#!/usr/bin/env python3
"""Run fine-tune sweep in a single process with pre-processed .npy features.

Loads features ONCE, then loops over (strategy, size, seed) configs.
Each run: subsample train data, create model, train, evaluate, save results.

Usage:
    python scripts/train_finetune_sweep.py \
        --train-dir /data/features/train_100M \
        --val-dir /data/features/val_5M \
        --test-dir /data/features/test_20M \
        --strategy full_ft \
        --checkpoint models/JetClassII_Sophon/model.pt
"""
from __future__ import annotations

import argparse
import gc
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


torch.backends.cudnn.benchmark = True

SIZES = [10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, 100000000]
SEEDS = [42, 123, 456]


def _amp_dtype(device: torch.device) -> torch.dtype:
    """Return bfloat16 if supported, else float32 (no autocast effect)."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


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
        f = torch.from_numpy(np.asarray(self.features[idx], dtype=np.float32)).T
        lv = torch.from_numpy(np.asarray(self.lorentz[idx], dtype=np.float32)).T
        m = torch.from_numpy(np.asarray(self.masks[idx])).unsqueeze(0)  # (1, 128)
        return {
            "features": f,
            "lorentz_vectors": lv,
            "mask": m,
            "label": int(self.labels[idx]),
        }


class MultiFeatureDataset(Dataset):
    """Dataset from per-class lists of feature arrays — avoids concatenation OOM at 30M+.

    Each list element is one mmap'd .npy slice (one class). Index dispatch is a
    short linear scan over cumulative sizes (10 classes => 10 comparisons).
    """

    def __init__(self, feats_list, lv_list, masks_list, labels_list):
        self.feats = feats_list
        self.lv = lv_list
        self.masks = masks_list
        self.labels = labels_list
        self.cum_sizes: list[int] = []
        total = 0
        for arr in feats_list:
            total += len(arr)
            self.cum_sizes.append(total)
        self._total = total

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        for i, cum in enumerate(self.cum_sizes):
            if idx < cum:
                local = idx - (self.cum_sizes[i - 1] if i > 0 else 0)
                f = torch.from_numpy(np.asarray(self.feats[i][local], dtype=np.float32)).T
                lv = torch.from_numpy(np.asarray(self.lv[i][local], dtype=np.float32)).T
                m = torch.from_numpy(np.asarray(self.masks[i][local])).unsqueeze(0)
                return {
                    "features": f,
                    "lorentz_vectors": lv,
                    "mask": m,
                    "label": int(self.labels[i][local]),
                }
        raise IndexError(f"Index {idx} out of range")


def collate_fn(batch):
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "lorentz_vectors": torch.stack([b["lorentz_vectors"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def load_npy_features(data_dir, max_jets=None, return_list=False, materialize=False):
    """Load pre-processed .npy feature files from a directory.

    Distributes max_jets evenly across classes to ensure all classes
    are represented. Groups files by class name prefix.

    If return_list=True, returns lists of per-file mmap'd arrays (no concatenation,
    avoids OOM at 30M+). Caller must use MultiFeatureDataset.
    Otherwise concatenates into single ndarrays (FeatureDataset compatible).

    If materialize=True (only with return_list=True), reads each mmap'd array
    fully into RAM before returning. Eliminates PVC reads during training but
    requires enough pod RAM for total dataset size.
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
            feats = np.load(str(d / f"{stem}_features.npy"), mmap_mode="r")
            lv = np.load(str(d / f"{stem}_lorentz.npy"), mmap_mode="r")
            masks = np.load(str(d / f"{stem}_masks.npy"), mmap_mode="r")
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

    if return_list:
        if materialize:
            print(f"  Materializing {total:,} jets into RAM (avoids PVC reads during training)...")
            for i in range(len(all_f)):
                all_f[i] = np.array(all_f[i])
                all_lv[i] = np.array(all_lv[i])
                all_m[i] = np.array(all_m[i])
            print(f"  Total loaded: {total:,} jets, {num_classes} classes (multi-array, materialized)")
        else:
            print(f"  Total loaded: {total:,} jets, {num_classes} classes (multi-array, mmap'd)")
        return all_f, all_lv, all_m, all_lab

    features = np.concatenate(all_f)
    lorentz = np.concatenate(all_lv)
    masks = np.concatenate(all_m)
    labels = np.concatenate(all_lab)

    print(f"  Total loaded: {len(labels):,} jets, {num_classes} classes")
    return features, lorentz, masks, labels


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    amp_dtype = _amp_dtype(device)
    total_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device, dtype=torch.long)
    total = 0
    for batch in loader:
        f = batch["features"].to(device, non_blocking=True)
        lv = batch["lorentz_vectors"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            logits, _ = model(f, lv, m)
            loss = F.cross_entropy(logits, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * lab.size(0)
        correct += (logits.argmax(-1) == lab).sum()
        total += lab.size(0)
    return (total_loss.item() / total), (correct.item() / total)


def evaluate(model, loader, device):
    model.eval()
    amp_dtype = _amp_dtype(device)
    total_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device, dtype=torch.long)
    total = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            f = batch["features"].to(device, non_blocking=True)
            lv = batch["lorentz_vectors"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            lab = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits, _ = model(f, lv, m)
                loss = F.cross_entropy(logits, lab)
            total_loss += loss.detach() * lab.size(0)
            correct += (logits.argmax(-1) == lab).sum()
            total += lab.size(0)
            # softmax in fp32 for AUC precision; bf16 softmax loses 4th-decimal AUC
            all_probs.append(F.softmax(logits.float(), -1).cpu())
            all_labels.append(lab.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return (total_loss.item() / total), (correct.item() / total), auc


def run_single(train_dir,
               val_loader, test_loader,
               strategy, checkpoint, train_size, seed, device, output_dir,
               frozen_layers=4, lr=1e-3, backbone_lr=1e-4,
               epochs=100, patience=10, batch_size=256,
               materialize_train=False):
    seed_everything(seed)

    # Load only enough training data for this run.
    # At >=3M, return list-of-arrays to avoid concat hang under PVC contention.
    # If materialize_train=True, read fully into RAM (no per-batch PVC reads).
    print(f"    Loading {train_size:,} training jets...")
    use_multi = train_size >= 3_000_000
    f, lv, m, lab = load_npy_features(
        train_dir, max_jets=train_size,
        return_list=use_multi, materialize=(use_multi and materialize_train))

    if use_multi:
        train_ds = MultiFeatureDataset(f, lv, m, lab)
    else:
        train_ds = FeatureDataset(f, lv, m, lab)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=collate_fn, drop_last=True,
    )

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

    # 5-epoch linear warmup then cosine — mitigates LR confound from larger batch.
    warmup_epochs = min(5, max(1, epochs // 10))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    start = time.time()
    best_val_loss = float("inf")
    best_val_acc = best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    # Training history
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

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"    early stopping at epoch {epoch+1}")
            break

    # Load best model for final evaluation
    model.load_state_dict(best_state)

    # Final test — collect per-sample predictions for ROC curves
    model.eval()
    amp_dtype = _amp_dtype(device)
    all_probs, all_labels_test = [], []
    test_loss_total = torch.zeros(1, device=device)
    test_correct = torch.zeros(1, device=device, dtype=torch.long)
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            f_b = batch["features"].to(device, non_blocking=True)
            lv_b = batch["lorentz_vectors"].to(device, non_blocking=True)
            m_b = batch["mask"].to(device, non_blocking=True)
            lab_b = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                logits, _ = model(f_b, lv_b, m_b)
                loss = F.cross_entropy(logits, lab_b)
            test_loss_total += loss.detach() * lab_b.size(0)
            test_correct += (logits.argmax(-1) == lab_b).sum()
            test_total += lab_b.size(0)
            all_probs.append(F.softmax(logits.float(), -1).cpu())
            all_labels_test.append(lab_b.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels_test = torch.cat(all_labels_test).numpy()
    test_loss = test_loss_total.item() / test_total
    test_acc = test_correct.item() / test_total
    try:
        test_auc = roc_auc_score(all_labels_test, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        test_auc = 0.0

    # Per-class AUC
    per_class_auc = {}
    label_names = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
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
    out = Path(output_dir) / f"{strategy}_{train_size}_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Results JSON
    results = {
        "strategy": strategy, "train_size": train_size, "seed": seed,
        "num_classes": 10,
        "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
        "val_auc_macro": best_val_auc,
        "test_loss": test_loss, "test_acc": test_acc,
        "test_auc_macro": test_auc,
        "per_class_auc": per_class_auc,
        "total_params": total_params, "trainable_params": trainable_params,
        "num_epochs": epoch + 1, "wall_clock_seconds": elapsed,
    }
    with open(out / "results.json", "w") as f_out:
        json.dump(results, f_out, indent=2)

    # 2. Training history CSV
    import csv
    with open(out / "training_history.csv", "w", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=history.keys())
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
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{strategy} {train_size:,} s{seed} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps, history["train_acc"], label="Train", color="#4477AA")
    ax2.plot(eps, history["val_acc"], label="Val", color="#EE6677")
    ax2.plot(eps, history["val_auc"], label="Val AUC", color="#228833", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title(f"{strategy} {train_size:,} s{seed} — Accuracy & AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close()

    # 5. ROC curves per class
    from sklearn.metrics import roc_curve, auc as sk_auc
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls_idx in range(10):
        binary = (all_labels_test == cls_idx).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, all_probs[:, cls_idx])
        area = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label_names[cls_idx]} (AUC={area:.3f})", linewidth=1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {strategy} {train_size:,} s{seed}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=150)
    plt.close()

    # 6. Confusion matrix
    from sklearn.metrics import confusion_matrix
    preds = all_probs.argmax(axis=1)
    cm = confusion_matrix(all_labels_test, preds, labels=list(range(10)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_xticklabels(label_names, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(10))
    ax.set_yticklabels(label_names, fontsize=7)
    for i in range(10):
        for j in range(10):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {strategy} {train_size:,} s{seed}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"    Saved: results.json, training_history.csv, best_model.pt, "
          f"loss_curves.png, roc_curves.png, confusion_matrix.png")

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
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (default: 42,123,456)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs where results.json + best_model.pt already exist")
    parser.add_argument("--materialize-train", action="store_true",
                        help="Read mmap'd train arrays fully into RAM (avoids PVC reads during training)")
    parser.add_argument("--frozen-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=100000)
    parser.add_argument("--test-size", type=int, default=500000)
    args = parser.parse_args()

    if args.sizes:
        global SIZES
        SIZES = [int(s) for s in args.sizes.split(",")]
    if args.seeds:
        global SEEDS
        SEEDS = [int(s) for s in args.seeds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}")
    print(f"Strategy: {args.strategy}")
    print(f"bf16 supported: {bf16_supported}; AMP dtype: {_amp_dtype(device)}")
    print(f"Batch size: {args.batch_size}")

    # Load val and test once (small, stays in memory)
    print("\nLoading val features...")
    val_f, val_lv, val_m, val_lab = load_npy_features(args.val_dir, max_jets=args.val_size)
    val_ds = FeatureDataset(val_f, val_lv, val_m, val_lab)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True,
                            prefetch_factor=2, collate_fn=collate_fn)
    print(f"Val: {len(val_lab):,} jets")

    print("\nLoading test features...")
    test_f, test_lv, test_m, test_lab = load_npy_features(args.test_dir, max_jets=args.test_size)
    test_ds = FeatureDataset(test_f, test_lv, test_m, test_lab)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True,
                             prefetch_factor=2, collate_fn=collate_fn)
    print(f"Test: {len(test_lab):,} jets")

    # Run sweep — train data loaded per-run (only what's needed)
    total = len(SIZES) * len(SEEDS)
    idx = 0
    for size in SIZES:
        for seed in SEEDS:
            idx += 1
            print(f"\n[{idx}/{total}] strategy={args.strategy} size={size:,} seed={seed}")

            if args.skip_existing:
                run_dir = Path(args.output_dir) / f"{args.strategy}_{size}_{seed}"
                if (run_dir / "results.json").exists() and (run_dir / "best_model.pt").exists():
                    print(f"    SKIP — already complete at {run_dir}")
                    continue

            results = run_single(
                args.train_dir,
                val_loader, test_loader,
                args.strategy, args.checkpoint, size, seed, device,
                args.output_dir,
                frozen_layers=args.frozen_layers,
                lr=args.lr, backbone_lr=args.backbone_lr,
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size,
                materialize_train=args.materialize_train,
            )

            print(f"  acc={results['test_acc']:.4f} auc={results['test_auc_macro']:.4f} "
                  f"time={results['wall_clock_seconds']:.1f}s epochs={results['num_epochs']}")

            # Free model + GPU cache between runs to avoid fragmentation OOM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nALL DONE — {idx} runs in {args.output_dir}")


if __name__ == "__main__":
    main()
