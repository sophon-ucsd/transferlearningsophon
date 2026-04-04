"""Dataset for precomputed Sophon 128-dim embeddings (.npy or .csv files).

Directory structure (NPY — from cluster inference job):
    embeddings_dir/
        HToBB_embeddings.npy      # shape (N, 128), float16
        HToCC_embeddings.npy
        ...

Directory structure (CSV — from local inference):
    embeddings_dir/
        HToBB_inference_with_embedding.csv
        HToCC_inference_with_embedding.csv
        ...
    CSV columns: source_file, entry_index, row_index, truth_label, label_name,
                 jet_sdmass, jet_mass, jet_pt, jet_eta, jet_phi,
                 logit_0..logit_9, emb_0..emb_127
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .subsampler import stratified_subsample

# Class name -> integer label (matching LABEL_KEYS argmax order in inference_all_classes.py)
# label_QCD=0, label_Hbb=1, label_Hcc=2, label_Hgg=3, label_H4q=4,
# label_Hqql=5, label_Zqq=6, label_Wqq=7, label_Tbqq=8, label_Tbl=9
NPY_CLASS_MAP = {
    "HToBB": 1, "HToCC": 2, "HToGG": 3, "HToWW4Q": 4, "HToWW2Q1L": 5,
    "TTBar": 8, "TTBarLep": 9, "WToQQ": 7, "ZToQQ": 6, "ZJetsToNuNu": 0,
}


def _detect_format(embeddings_dir: Path) -> str:
    """Auto-detect whether directory has .npy or .csv embedding files."""
    if list(embeddings_dir.glob("*_embeddings.npy")):
        return "npy"
    if list(embeddings_dir.glob("*_inference_with_embedding.csv")):
        return "csv"
    raise FileNotFoundError(
        f"No embedding files found in {embeddings_dir}. "
        "Expected *_embeddings.npy or *_inference_with_embedding.csv"
    )


def _load_npy(embeddings_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all .npy embedding files. Returns (embeddings, labels)."""
    all_emb = []
    all_lab = []
    for fpath in sorted(embeddings_dir.glob("*_embeddings.npy")):
        # Extract class name: e.g. "HToBB_embeddings.npy" -> "HToBB"
        cls_name = fpath.stem.replace("_embeddings", "")
        if cls_name not in NPY_CLASS_MAP:
            print(f"Warning: unknown class {cls_name} in {fpath.name}, skipping")
            continue
        label = NPY_CLASS_MAP[cls_name]
        emb = np.load(str(fpath), mmap_mode="r")  # memory-mapped
        all_emb.append(emb)
        all_lab.append(np.full(len(emb), label, dtype=np.int64))
        print(f"  {cls_name}: {len(emb):,} embeddings (label={label})")

    return np.concatenate(all_emb), np.concatenate(all_lab)


def _load_csv(embeddings_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all CSV embedding files. Returns (embeddings, labels)."""
    import pandas as pd

    all_emb = []
    all_lab = []
    emb_cols = [f"emb_{i}" for i in range(128)]

    for fpath in sorted(embeddings_dir.glob("*_inference_with_embedding.csv")):
        df = pd.read_csv(fpath)
        emb = df[emb_cols].values.astype(np.float32)
        labels = df["truth_label"].values.astype(np.int64)
        cls_name = fpath.stem.replace("_inference_with_embedding", "")
        all_emb.append(emb)
        all_lab.append(labels)
        print(f"  {cls_name}: {len(emb):,} embeddings")

    return np.concatenate(all_emb), np.concatenate(all_lab)


class EmbeddingDataset(Dataset):
    """Loads precomputed Sophon 128-dim embeddings from .npy or .csv files."""

    def __init__(
        self,
        embeddings_dir: str,
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        subset_size: int | None = None,
        seed: int = 42,
    ) -> None:
        emb_path = Path(embeddings_dir)
        fmt = _detect_format(emb_path)
        print(f"Loading {fmt} embeddings from {emb_path}")

        if fmt == "npy":
            all_emb, all_lab = _load_npy(emb_path)
        else:
            all_emb, all_lab = _load_csv(emb_path)

        n = len(all_lab)
        print(f"Total: {n:,} embeddings, {len(np.unique(all_lab))} classes")

        # Deterministic split per class
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

        if split == "train":
            indices = train_idx
        elif split == "val":
            indices = val_idx
        elif split == "test":
            indices = test_idx
        else:
            raise ValueError(f"Unknown split: {split!r}")

        # Subsample training set if requested
        if subset_size is not None and subset_size < len(indices):
            sub_labels = all_lab[indices]
            sub_idx = stratified_subsample(sub_labels, subset_size, seed)
            indices = indices[sub_idx]

        # Copy into contiguous arrays for fast DataLoader access
        # (breaks the memory map but we need fast random access)
        self.embeddings = np.ascontiguousarray(all_emb[indices]).astype(np.float32)
        self.labels = all_lab[indices].copy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        return {
            "embeddings": torch.from_numpy(self.embeddings[idx]),
            "label": int(self.labels[idx]),
        }


def create_embedding_dataloaders(
    embeddings_dir: str,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    batch_size: int = 4096,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from precomputed embeddings."""
    train_ds = EmbeddingDataset(
        embeddings_dir, split="train", subset_size=train_size, seed=seed,
    )
    val_ds = EmbeddingDataset(
        embeddings_dir, split="val", subset_size=val_size, seed=seed,
    )
    test_ds = EmbeddingDataset(
        embeddings_dir, split="test", subset_size=test_size, seed=seed,
    )

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
