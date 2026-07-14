# 
# Pretrain Sophon from scratch on JetClass-II at variable class granularity
# beginning of raunavs b.1 task
#  

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from torch.optim import AdamW

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sophon_wrapper import SophonTransferModel
from src.utils.reproducibility import seed_everything


torch.backends.cudnn.benchmark = True


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


class FeatureDataset(Dataset):
    def __init__(self, features, lorentz, masks, labels):
        self.features = features
        self.lorentz = lorentz
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        f = torch.from_numpy(np.asarray(self.features[idx], dtype=np.float32)).T
        lv = torch.from_numpy(np.asarray(self.lorentz[idx], dtype=np.float32)).T
        m = torch.from_numpy(np.asarray(self.masks[idx])).unsqueeze(0)
        return {
            "features": f,
            "lorentz_vectors": lv,
            "mask": m,
            "label": int(self.labels[idx]),
        }


class MultiFeatureDataset(Dataset):
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
    d = Path(data_dir)
    feature_files = sorted(d.glob("*_features.npy"))
    if not feature_files:
        raise FileNotFoundError(f"No *_features.npy files in {data_dir}")

    files_by_class: dict[str, list[str]] = {}
    for fpath in feature_files:
        stem = fpath.stem.replace("_features", "")
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
            print(f"  Materializing {total:,} jets into RAM...")
            for i in range(len(all_f)):
                all_f[i] = np.array(all_f[i])
                all_lv[i] = np.array(all_lv[i])
                all_m[i] = np.array(all_m[i])
            print(f"  Total loaded: {total:,} jets, {num_classes} classes (materialized)")
        else:
            print(f"  Total loaded: {total:,} jets, {num_classes} classes (mmap'd)")
        return all_f, all_lv, all_m, all_lab

    features = np.concatenate(all_f)
    lorentz = np.concatenate(all_lv)
    masks = np.concatenate(all_m)
    labels = np.concatenate(all_lab)

    print(f"  Total loaded: {len(labels):,} jets, {num_classes} classes")
    return features, lorentz, masks, labels


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


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
            all_probs.append(F.softmax(logits.float(), -1).cpu())
            all_labels.append(lab.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return (total_loss.item() / total), (correct.item() / total), auc


def run_pretrain(train_dir, val_loader, num_classes, seed, device, output_dir,
                 lr=5e-4, # changed from 1e-3 to match the default 5e-4 on the repo
                 epochs=80, # changed from 200 to match the default 80 on the repo
                 patience=20, 
                 batch_size=512,
                 ema_decay=0.9999,
                materialize_train=False):
    seed_everything(seed)

    print(f"  Loading training jets (all classes)...")
    f, lv, m, lab = load_npy_features(train_dir, return_list=True, materialize=materialize_train)
    train_ds = MultiFeatureDataset(f, lv, m, lab)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=collate_fn, drop_last=True,
    )

    model = SophonTransferModel(checkpoint_path=None, num_classes=num_classes, head_type="mlp")
    model = model.to(device)
    total_params, trainable_params = SophonTransferModel.count_params(model)
    print(f"  Params: {total_params:,} total / {trainable_params:,} trainable")

    ema = EMA(model, decay=ema_decay)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ## changing this to use ranger instead, AdamW declaration up top

    try:
        from ranger import Ranger
        optimizer = Ranger(model.parameters(), lr=lr, weight_decay=0.01)
        print("using ranger optimizer")
    except ImportError:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        print("ranger not found, using AdamW optimizer instead")

    warmup_epochs = min(10, max(1, epochs // 20))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    start = time.time()
    best_val_loss = float("inf")
    best_val_acc = best_val_auc = 0.0
    patience_counter = 0

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        ema.update(model)

        # evaluate with ema weights
        ema.apply_shadow(model)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)
        ema.restore(model)

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
            patience_counter = 0
            # save ema weights as checkpoint — these are what downstream fine-tuning uses
            ema.apply_shadow(model)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.restore(model)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"  early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "num_classes": num_classes, "seed": seed,
        "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
        "val_auc_macro": best_val_auc,
        "total_params": total_params, "trainable_params": trainable_params,
        "num_epochs": epoch + 1, "wall_clock_seconds": elapsed,
        "lr": lr, "ema_decay": ema_decay, "batch_size": batch_size,
    }
    with open(out / "pretrain_results.json", "w") as f_out:
        json.dump(results, f_out, indent=2)

    import csv
    with open(out / "training_history.csv", "w", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=history.keys())
        w.writeheader()
        for i in range(len(history["epoch"])):
            w.writerow({k: history[k][i] for k in history})

    # save in format compatible with train_finetune_sweep.py --checkpoint
    torch.save({"model_state_dict": best_state}, out / "best_model.pt")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    eps = history["epoch"]
    ax1.plot(eps, history["train_loss"], label="Train", color="#4477AA")
    ax1.plot(eps, history["val_loss"], label="Val", color="#EE6677")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Pretrain {num_classes}cls s{seed} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps, history["train_acc"], label="Train", color="#4477AA")
    ax2.plot(eps, history["val_acc"], label="Val", color="#EE6677")
    ax2.plot(eps, history["val_auc"], label="Val AUC", color="#228833", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title(f"Pretrain {num_classes}cls s{seed} — Accuracy & AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close()

    print(f"  Saved: pretrain_results.json, training_history.csv, best_model.pt, loss_curves.png")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, help="JetClass-II train .npy dir")
    parser.add_argument("--val-dir", required=True, help="JetClass-II val .npy dir")
    parser.add_argument("--num-classes", type=int, required=True, choices=[10, 42, 188])
    parser.add_argument("--output-dir", default="/data/pretrain/sophon")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--materialize-train", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}")
    print(f"Num classes: {args.num_classes}")
    print(f"bf16 supported: {bf16_supported}; AMP dtype: {_amp_dtype(device)}")

    print("\nLoading val features...")
    val_f, val_lv, val_m, val_lab = load_npy_features(args.val_dir, max_jets=100000)
    val_ds = FeatureDataset(val_f, val_lv, val_m, val_lab)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True,
                            prefetch_factor=2, collate_fn=collate_fn)
    print(f"Val: {len(val_lab):,} jets")

    out_dir = Path(args.output_dir) / f"sophon_{args.num_classes}cls_seed{args.seed}"

    print(f"\nPretraining sophon {args.num_classes}cls seed={args.seed}")
    results = run_pretrain(
        args.train_dir, val_loader,
        num_classes=args.num_classes,
        seed=args.seed,
        device=device,
        output_dir=out_dir,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay,
        materialize_train=args.materialize_train,
    )
    print(f"\nDone — val_acc={results['best_val_acc']:.4f} val_auc={results['val_auc_macro']:.4f} "
          f"time={results['wall_clock_seconds']:.1f}s epochs={results['num_epochs']}")
    print(f"Checkpoint: {out_dir}/best_model.pt")
    print("Next: pass to train_finetune_sweep.py --checkpoint")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()