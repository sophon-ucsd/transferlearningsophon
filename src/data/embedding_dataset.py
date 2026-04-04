"""Dataset for precomputed Sophon 128-dim embeddings (.npy or .csv files).

Supports two modes:
1. Separate directories per split (for official JetClass train/val/test):
   - train_dir: /data/embeddings_pretrained_full_*/
   - val_dir:   /data/embeddings_val_5M/
   - test_dir:  /data/embeddings_test_20M/

2. Single directory with internal splitting (fallback for local testing).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .subsampler import stratified_subsample

# Class name -> integer label (matching LABEL_KEYS argmax order)
NPY_CLASS_MAP = {
    "HToBB": 1, "HToCC": 2, "HToGG": 3, "HToWW4Q": 4, "HToWW2Q1L": 5,
    "TTBar": 8, "TTBarLep": 9, "WToQQ": 7, "ZToQQ": 6, "ZJetsToNuNu": 0,
}


def _detect_format(embeddings_dir: Path) -> str:
    if list(embeddings_dir.glob("*_embeddings.npy")):
        return "npy"
    if list(embeddings_dir.glob("*_inference_with_embedding.csv")):
        return "csv"
    raise FileNotFoundError(
        f"No embedding files found in {embeddings_dir}. "
        "Expected *_embeddings.npy or *_inference_with_embedding.csv"
    )


def _load_npy(embeddings_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    all_emb, all_lab = [], []
    for fpath in sorted(embeddings_dir.glob("*_embeddings.npy")):
        cls_name = fpath.stem.replace("_embeddings", "")
        if cls_name not in NPY_CLASS_MAP:
            print(f"Warning: unknown class {cls_name}, skipping")
            continue
        label = NPY_CLASS_MAP[cls_name]
        emb = np.load(str(fpath), mmap_mode="r")
        all_emb.append(emb)
        all_lab.append(np.full(len(emb), label, dtype=np.int64))
        print(f"  {cls_name}: {len(emb):,} embeddings (label={label})")
    return np.concatenate(all_emb), np.concatenate(all_lab)


def _load_csv(embeddings_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    all_emb, all_lab = [], []
    emb_cols = [f"emb_{i}" for i in range(128)]
    for fpath in sorted(embeddings_dir.glob("*_inference_with_embedding.csv")):
        df = pd.read_csv(fpath)
        all_emb.append(df[emb_cols].values.astype(np.float32))
        all_lab.append(df["truth_label"].values.astype(np.int64))
        cls_name = fpath.stem.replace("_inference_with_embedding", "")
        print(f"  {cls_name}: {len(df):,} embeddings")
    return np.concatenate(all_emb), np.concatenate(all_lab)


def _load_dir(embeddings_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all embeddings from a directory, auto-detecting format."""
    emb_path = Path(embeddings_dir)
    fmt = _detect_format(emb_path)
    print(f"Loading {fmt} embeddings from {emb_path}")
    if fmt == "npy":
        return _load_npy(emb_path)
    return _load_csv(emb_path)


class EmbeddingDataset(Dataset):
    """Loads precomputed Sophon 128-dim embeddings from a single directory.

    Each directory IS one split. No internal train/val/test splitting.
    Optionally subsamples to a target size (stratified by class).
    """

    def __init__(
        self,
        embeddings_dir: str,
        subset_size: int | None = None,
        seed: int = 42,
    ) -> None:
        all_emb, all_lab = _load_dir(embeddings_dir)
        print(f"Total: {len(all_lab):,} embeddings, {len(np.unique(all_lab))} classes")

        if subset_size is not None and subset_size < len(all_lab):
            indices = stratified_subsample(all_lab, subset_size, seed)
            all_emb = all_emb[indices]
            all_lab = all_lab[indices]
            print(f"Subsampled to {len(all_lab):,} embeddings")

        self.embeddings = np.ascontiguousarray(all_emb).astype(np.float32)
        self.labels = all_lab.copy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        return {
            "embeddings": torch.from_numpy(self.embeddings[idx]),
            "label": int(self.labels[idx]),
        }


def create_embedding_dataloaders(
    train_dir: str,
    val_dir: str | None = None,
    test_dir: str | None = None,
    train_size: int | None = None,
    batch_size: int = 4096,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from embedding directories.

    If val_dir and test_dir are provided, uses separate directories per split
    (official JetClass splits). Otherwise falls back to internal 80/10/10 split
    of train_dir (for local testing with a single embedding directory).
    """
    if val_dir is not None and test_dir is not None:
        train_ds = EmbeddingDataset(train_dir, subset_size=train_size, seed=seed)
        val_ds = EmbeddingDataset(val_dir)
        test_ds = EmbeddingDataset(test_dir)
    else:
        # Fallback: split single directory internally
        train_ds, val_ds, test_ds = _split_single_dir(train_dir, train_size, seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


class _ArrayDataset(Dataset):
    """Simple dataset from pre-split numpy arrays."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        return {
            "embeddings": torch.from_numpy(self.embeddings[idx]),
            "label": int(self.labels[idx]),
        }


def _split_single_dir(
    embeddings_dir: str,
    train_size: int | None,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> tuple[_ArrayDataset, _ArrayDataset, _ArrayDataset]:
    """Fallback: load one directory and split 80/10/10 internally."""
    all_emb, all_lab = _load_dir(embeddings_dir)

    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(all_lab):
        cls_indices = np.where(all_lab == cls)[0]
        rng.shuffle(cls_indices)
        n_cls = len(cls_indices)
        n_train = int(n_cls * train_frac)
        n_val = int(n_cls * val_frac)
        train_idx.append(cls_indices[:n_train])
        val_idx.append(cls_indices[n_train:n_train + n_val])
        test_idx.append(cls_indices[n_train + n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    if train_size is not None and train_size < len(train_idx):
        sub = stratified_subsample(all_lab[train_idx], train_size, seed)
        train_idx = train_idx[sub]

    def _make(idx):
        return _ArrayDataset(
            np.ascontiguousarray(all_emb[idx]).astype(np.float32),
            all_lab[idx].copy(),
        )

    print(f"  train: {len(train_idx):,}, val: {len(val_idx):,}, test: {len(test_idx):,}")
    return _make(train_idx), _make(val_idx), _make(test_idx)
