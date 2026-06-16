"""JetClass dataset and data module for ROOT files.

Feature computation matches inference_all_classes.py exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uproot
from torch.utils.data import Dataset, DataLoader

try:
    import pytorch_lightning as pl
except ImportError:
    pl = None  # allow import without lightning for testing

from .subsampler import stratified_subsample

# Constants must match inference_all_classes.py exactly.
MAX_PART = 128
TREE_NAME = "tree"

PARTICLE_KEYS = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_d0err",
    "part_dzval", "part_dzerr", "part_charge",
    "part_isChargedHadron", "part_isNeutralHadron",
    "part_isPhoton", "part_isElectron", "part_isMuon",
]

SCALAR_KEYS = ["jet_pt", "jet_energy"]

LABEL_KEYS = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
    "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
    "label_Tbqq", "label_Tbl",
]

ALL_KEYS = PARTICLE_KEYS + SCALAR_KEYS + LABEL_KEYS

# Class name (from filename) -> integer label index. Kept only as a fallback;
# we read labels from the ROOT file's one-hot LABEL_KEYS at runtime.
CLASS_TO_LABEL = {
    "HToBB": 1, "HToCC": 2, "HToGG": 3, "HToWW4Q": 4, "HToWW2Q1L": 5,
    "TTBar": 8, "TTBarLep": 9, "WToQQ": 7, "ZToQQ": 6, "ZJetsToNuNu": 0,
}

NUM_CLASSES = 10


def _norm(x: np.ndarray, subtract: float, multiply: float,
          clip_lo: float = -5.0, clip_hi: float = 5.0) -> np.ndarray:
    return np.clip((x - subtract) * multiply, clip_lo, clip_hi)


def compute_sophon_features(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    energy: np.ndarray,
    deta: np.ndarray,
    dphi: np.ndarray,
    d0val: np.ndarray,
    d0err: np.ndarray,
    dzval: np.ndarray,
    dzerr: np.ndarray,
    charge: np.ndarray,
    isChargedHadron: np.ndarray,
    isNeutralHadron: np.ndarray,
    isPhoton: np.ndarray,
    isElectron: np.ndarray,
    isMuon: np.ndarray,
    jet_pt: float,
    jet_energy: float,
) -> np.ndarray:
    """Compute the 17 Sophon input features. Returns (n_part, 17) float32.

    Normalization constants match JetClassII_full.yaml exactly.
    """
    eps = 1e-20
    jet_pt_safe = max(jet_pt, eps)
    jet_energy_safe = max(jet_energy, eps)

    pt = np.sqrt(px ** 2 + py ** 2)
    pt_scale = pt * 500.0 / jet_pt_safe
    energy_scale = energy * 500.0 / jet_pt_safe

    pt_scale_log = _norm(np.log(np.clip(pt_scale, eps, None)), subtract=1.7, multiply=0.7)
    e_scale_log = _norm(np.log(np.clip(energy_scale, eps, None)), subtract=2.0, multiply=0.7)
    logptrel = _norm(np.log(np.clip(pt / jet_pt_safe, eps, None)), subtract=-4.7, multiply=0.7)
    logerel = _norm(np.log(np.clip(energy / jet_energy_safe, eps, None)), subtract=-4.7, multiply=0.7)
    deltaR = _norm(np.sqrt(deta ** 2 + dphi ** 2), subtract=0.2, multiply=4.0)

    d0 = np.tanh(d0val)
    d0err_c = np.clip(d0err, 0.0, 1.0)
    dz = np.tanh(dzval)
    dzerr_c = np.clip(dzerr, 0.0, 1.0)

    feats = np.stack([
        pt_scale_log,
        e_scale_log,
        logptrel,
        logerel,
        deltaR,
        charge,
        isChargedHadron,
        isNeutralHadron,
        isPhoton,
        isElectron,
        isMuon,
        d0,
        d0err_c,
        dz,
        dzerr_c,
        deta,
        dphi,
    ], axis=1).astype(np.float32)

    return feats


def _process_file(fpath: str, max_particles: int = MAX_PART):
    """Load one ROOT file and compute all features/LV/masks/labels.

    Returns: features (N, max_part, 17), lorentz_vectors (N, max_part, 4),
             masks (N, max_part), labels (N,)
    """
    with uproot.open(f"{fpath}:{TREE_NAME}") as tree:
        arrays = tree.arrays(ALL_KEYS, library="np")

    n_jets = len(arrays["jet_pt"])
    all_feats = np.zeros((n_jets, max_particles, 17), dtype=np.float32)
    all_lv = np.zeros((n_jets, max_particles, 4), dtype=np.float32)
    all_masks = np.zeros((n_jets, max_particles), dtype=bool)

    # Labels from one-hot
    label_matrix = np.stack([arrays[k] for k in LABEL_KEYS], axis=1)
    all_labels = np.argmax(label_matrix, axis=1)

    for i in range(n_jets):
        px = arrays["part_px"][i]
        py = arrays["part_py"][i]
        pz = arrays["part_pz"][i]
        energy = arrays["part_energy"][i]
        n_part = len(px)

        pt = np.sqrt(px ** 2 + py ** 2)
        if n_part > max_particles:
            keep = np.argsort(pt)[::-1][:max_particles]
            n_part = max_particles
        else:
            keep = np.argsort(pt)[::-1]

        px, py, pz, energy = px[keep], py[keep], pz[keep], energy[keep]
        jet_pt = float(arrays["jet_pt"][i])
        jet_energy = float(arrays["jet_energy"][i])
        jet_pt_safe = max(jet_pt, 1e-20)

        lv = np.stack([
            px * 500.0 / jet_pt_safe, py * 500.0 / jet_pt_safe,
            pz * 500.0 / jet_pt_safe, energy * 500.0 / jet_pt_safe,
        ], axis=1).astype(np.float32)

        feats = compute_sophon_features(
            px=px, py=py, pz=pz, energy=energy,
            deta=arrays["part_deta"][i][keep], dphi=arrays["part_dphi"][i][keep],
            d0val=arrays["part_d0val"][i][keep], d0err=arrays["part_d0err"][i][keep],
            dzval=arrays["part_dzval"][i][keep], dzerr=arrays["part_dzerr"][i][keep],
            charge=arrays["part_charge"][i][keep],
            isChargedHadron=arrays["part_isChargedHadron"][i][keep],
            isNeutralHadron=arrays["part_isNeutralHadron"][i][keep],
            isPhoton=arrays["part_isPhoton"][i][keep],
            isElectron=arrays["part_isElectron"][i][keep],
            isMuon=arrays["part_isMuon"][i][keep],
            jet_pt=jet_pt, jet_energy=jet_energy,
        )

        all_feats[i, :n_part] = feats
        all_lv[i, :n_part] = lv
        all_masks[i, :n_part] = True

    return all_feats, all_lv, all_masks, all_labels


class JetClassDataset(Dataset):
    """Pre-loaded JetClass ROOT dataset.

    Loads all files at init time, computes features, stores in memory.
    __getitem__ is a simple array index — no I/O.
    """

    def __init__(self, file_list: list[str | Path], max_particles: int = MAX_PART) -> None:
        self.max_particles = max_particles

        all_feats, all_lv, all_masks, all_labels = [], [], [], []
        for fpath in file_list:
            print(f"  Loading {Path(fpath).name}...", end="", flush=True)
            feats, lv, masks, labels = _process_file(str(fpath), max_particles)
            all_feats.append(feats)
            all_lv.append(lv)
            all_masks.append(masks)
            all_labels.append(labels)
            print(f" {len(labels):,} jets")

        self.features = np.concatenate(all_feats)      # (N, 128, 17)
        self.lorentz_vectors = np.concatenate(all_lv)   # (N, 128, 4)
        self.masks = np.concatenate(all_masks)           # (N, 128)
        self.labels = np.concatenate(all_labels)         # (N,)
        print(f"  Total: {len(self.labels):,} jets loaded into memory")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "lorentz_vectors": torch.from_numpy(self.lorentz_vectors[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
            "label": int(self.labels[idx]),
        }

    def get_all_labels(self) -> np.ndarray:
        return self.labels.copy()


class LazyJetClassDataset(Dataset):
    """Lazy-loading JetClass dataset for val/test (sequential access).

    Processes files on demand with an LRU cache. Much lower memory than
    pre-loading, works well for sequential (non-shuffled) access.

    If max_jets is set, only indexes that many jets total (stratified
    across files). This ensures len(dataset) == max_jets exactly.
    """

    def __init__(self, file_list: list[str | Path], max_particles: int = MAX_PART,
                 cache_size: int = 5, max_jets: int | None = None) -> None:
        self.file_list = [str(p) for p in file_list]
        self.max_particles = max_particles

        # Build index: scan all files for entry counts
        self._file_entry_counts: list[int] = []
        for fpath in self.file_list:
            with uproot.open(f"{fpath}:{TREE_NAME}") as tree:
                self._file_entry_counts.append(tree.num_entries)

        # Distribute max_jets evenly across files (ensures class balance)
        total_available = sum(self._file_entry_counts)
        if max_jets is not None and max_jets < total_available:
            ratio = max_jets / total_available
            self._file_use_counts = [max(1, int(n * ratio)) for n in self._file_entry_counts]
            # Trim to exact target
            while sum(self._file_use_counts) > max_jets:
                # Remove from largest
                idx = max(range(len(self._file_use_counts)), key=lambda i: self._file_use_counts[i])
                self._file_use_counts[idx] -= 1
        else:
            self._file_use_counts = list(self._file_entry_counts)

        # Build cumulative sizes and trim empty files
        self._cum_sizes: list[int] = []
        total = 0
        keep = []
        for i, use in enumerate(self._file_use_counts):
            if use > 0:
                total += use
                self._cum_sizes.append(total)
                keep.append(i)
        self._total = total
        self.file_list = [self.file_list[i] for i in keep]
        self._file_entry_counts = [self._file_entry_counts[i] for i in keep]
        self._file_use_counts = [self._file_use_counts[i] for i in keep]

        # LRU cache: file_idx -> (feats, lv, masks, labels)
        self._cache: dict[int, tuple] = {}
        self._cache_order: list[int] = []
        self._cache_size = cache_size
        print(f"  LazyDataset: {self._total:,} jets from {len(self.file_list)} files")

    def __len__(self) -> int:
        return self._total

    def _file_and_local(self, idx: int) -> tuple[int, int]:
        """Map global index to (file_idx, local_idx)."""
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range [0, {self._total})")
        for file_idx, cum in enumerate(self._cum_sizes):
            if idx < cum:
                local = idx - (self._cum_sizes[file_idx - 1] if file_idx > 0 else 0)
                return file_idx, local
        raise IndexError(f"Index {idx} out of range")

    def _load_file(self, file_idx: int) -> tuple:
        if file_idx in self._cache:
            return self._cache[file_idx]
        data = _process_file(self.file_list[file_idx], self.max_particles)
        # Evict oldest if cache full
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[file_idx] = data
        self._cache_order.append(file_idx)
        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, local = self._file_and_local(idx)
        feats, lv, masks, labels = self._load_file(file_idx)
        return {
            "features": torch.from_numpy(feats[local]),
            "lorentz_vectors": torch.from_numpy(lv[local]),
            "mask": torch.from_numpy(masks[local]),
            "label": int(labels[local]),
        }

    def get_all_labels(self) -> np.ndarray:
        """Load labels from all files (needed for subsampling)."""
        all_labels = []
        for file_idx in range(len(self.file_list)):
            _, _, _, labels = self._load_file(file_idx)
            all_labels.append(labels)
        return np.concatenate(all_labels)


def jetclass_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate JetClassDataset samples into model-ready batches.

    Output shapes:
        features: (B, 17, 128)
        lorentz_vectors: (B, 4, 128)
        mask: (B, 1, 128)
        label: (B,)
    """
    features = torch.stack([s["features"] for s in batch]).permute(0, 2, 1)       # (B,128,17) -> (B,17,128)
    lorentz_vectors = torch.stack([s["lorentz_vectors"] for s in batch]).permute(0, 2, 1)  # (B,128,4) -> (B,4,128)
    mask = torch.stack([s["mask"] for s in batch]).unsqueeze(1)                    # (B,128) -> (B,1,128)
    label = torch.tensor([s["label"] for s in batch], dtype=torch.long)

    return {
        "features": features,
        "lorentz_vectors": lorentz_vectors,
        "mask": mask,
        "label": label,
    }


def _discover_files(data_dir: str | Path) -> dict[str, list[Path]]:
    """Discover ROOT files per class in data_dir."""
    data_path = Path(data_dir)
    files_by_class: dict[str, list[Path]] = {}
    for f in sorted(data_path.glob("*.root")):
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        cls_name = parts[0]
        files_by_class.setdefault(cls_name, []).append(f)
    for cls in files_by_class:
        files_by_class[cls].sort(key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return files_by_class


if pl is not None:
    class JetClassDataModule(pl.LightningDataModule):
        """Lightning data module for JetClass ROOT files.

        Supports two modes:
        1. Separate dirs (official splits): train_dir + val_dir + test_dir
        2. Single dir (fallback): data_dir split internally 80/10/10
        """

        def __init__(
            self,
            data_dir: str = "",
            train_dir: str | None = None,
            val_dir: str | None = None,
            test_dir: str | None = None,
            train_size: int | None = None,
            val_size: int | None = None,
            test_size: int | None = None,
            batch_size: int = 512,
            num_workers: int = 8,
            seed: int = 42,
        ) -> None:
            super().__init__()
            self.data_dir = data_dir
            self.train_dir = train_dir
            self.val_dir = val_dir
            self.test_dir = test_dir
            self.train_size = train_size
            self.val_size = val_size
            self.test_size = test_size
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.seed = seed

            self.train_dataset: Optional[Dataset] = None
            self.val_dataset: Optional[Dataset] = None
            self.test_dataset: Optional[Dataset] = None

        def setup(self, stage: str | None = None) -> None:
            if self.train_dir and self.val_dir and self.test_dir:
                train_files = self._all_files(self.train_dir)
                val_files = self._all_files(self.val_dir)
                test_files = self._all_files(self.test_dir)
            else:
                train_files, val_files, test_files = self._split_single_dir(self.data_dir)

            # Pre-select files to avoid loading more than needed
            train_files = self._preselect_files(train_files, self.train_size)
            val_files = self._preselect_files(val_files, self.val_size)
            test_files = self._preselect_files(test_files, self.test_size)

            print(f"Files — train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

            # Load train into memory with streaming subsample
            self.train_dataset = self._load_and_subsample(
                train_files, self.train_size, "train")

            # Val/test: lazy load with max_jets cap
            self.val_dataset = LazyJetClassDataset(val_files, max_jets=self.val_size)
            self.test_dataset = LazyJetClassDataset(test_files, max_jets=self.test_size)

        def train_dataloader(self) -> DataLoader:
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, collate_fn=jetclass_collate,
                pin_memory=True, drop_last=True,
            )

        def val_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, collate_fn=jetclass_collate,
                pin_memory=True,
            )

        def test_dataloader(self) -> DataLoader:
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, collate_fn=jetclass_collate,
                pin_memory=True,
            )

        @staticmethod
        def _preselect_files(file_list: list[Path], target_size: int | None,
                             jets_per_file: int = 100_000) -> list[Path]:
            """Reduce file list if target_size is much smaller than total."""
            if target_size is None:
                return file_list
            total_available = len(file_list) * jets_per_file
            if target_size >= total_available:
                return file_list
            # Group by class, select enough files per class
            files_by_class: dict[str, list[Path]] = {}
            for f in file_list:
                cls = f.stem.rsplit("_", 1)[0]
                files_by_class.setdefault(cls, []).append(f)
            num_classes = max(1, len(files_by_class))
            per_class = target_size // num_classes
            files_needed = max(1, (per_class + jets_per_file - 1) // jets_per_file + 1)
            selected = []
            for cls, flist in sorted(files_by_class.items()):
                selected.extend(flist[:files_needed])
            if len(selected) < len(file_list):
                print(f"Pre-selected {len(selected)} files for target_size={target_size:,}")
            return selected

        def _load_and_subsample(self, file_list: list[Path],
                                target_size: int | None, split_name: str) -> JetClassDataset:
            """Load files one at a time, subsample on the fly, store only kept jets.

            Memory usage = target_size * (128*17 + 128*4 + 128 + 1) * 4 bytes.
            For 100K jets ≈ 1GB. For 10M jets ≈ 100GB (falls back to full load).
            """
            # First pass: count jets per class across files (fast, just entry counts)
            class_counts: dict[str, int] = {}
            file_classes: list[str] = []
            for f in file_list:
                cls = f.stem.rsplit("_", 1)[0]
                file_classes.append(cls)
                with uproot.open(f"{f}:{TREE_NAME}") as tree:
                    class_counts[cls] = class_counts.get(cls, 0) + tree.num_entries

            total_jets = sum(class_counts.values())
            num_classes = len(class_counts)

            if target_size is None or target_size >= total_jets:
                # Load everything — no subsampling needed
                print(f"  Loading all {total_jets:,} {split_name} jets...")
                return JetClassDataset(file_list)

            # Compute how many jets to keep per class
            per_class_target = target_size // num_classes
            # Track how many we've kept per class
            kept_per_class: dict[str, int] = {cls: 0 for cls in class_counts}

            all_feats, all_lv, all_masks, all_labels = [], [], [], []
            rng = np.random.RandomState(self.seed)

            for fpath, cls in zip(file_list, file_classes):
                remaining = per_class_target - kept_per_class[cls]
                if remaining <= 0:
                    continue

                print(f"  Loading {Path(fpath).name}...", end="", flush=True)
                feats, lv, masks, labels = _process_file(str(fpath), MAX_PART)
                n = len(labels)

                if remaining < n:
                    # Subsample this file
                    idx = rng.choice(n, size=remaining, replace=False)
                    idx.sort()
                    feats, lv, masks, labels = feats[idx], lv[idx], masks[idx], labels[idx]

                all_feats.append(feats)
                all_lv.append(lv)
                all_masks.append(masks)
                all_labels.append(labels)
                kept_per_class[cls] += len(labels)
                print(f" kept {len(labels):,}/{n:,}")

            # Build dataset from collected arrays
            ds = JetClassDataset.__new__(JetClassDataset)
            ds.max_particles = MAX_PART
            ds.features = np.concatenate(all_feats)
            ds.lorentz_vectors = np.concatenate(all_lv)
            ds.masks = np.concatenate(all_masks)
            ds.labels = np.concatenate(all_labels)
            total_kept = len(ds.labels)
            print(f"  {split_name}: {total_kept:,} jets in memory "
                  f"({total_kept * 128 * 21 * 4 / 1e9:.1f} GB)")
            return ds

        @staticmethod
        def _all_files(data_dir: str) -> list[Path]:
            """Get all ROOT files from a directory (no splitting)."""
            files_by_class = _discover_files(data_dir)
            if not files_by_class:
                raise FileNotFoundError(f"No ROOT files found in {data_dir}")
            all_files = []
            for flist in sorted(files_by_class.values()):
                all_files.extend(flist)
            return all_files

        @staticmethod
        def _split_single_dir(data_dir: str) -> tuple[list[Path], list[Path], list[Path]]:
            """Split files from a single directory 80/10/10."""
            files_by_class = _discover_files(data_dir)
            if not files_by_class:
                raise FileNotFoundError(f"No ROOT files found in {data_dir}")
            train_files, val_files, test_files = [], [], []
            for cls_name, flist in sorted(files_by_class.items()):
                n = len(flist)
                if n <= 2:
                    train_files.extend(flist)
                    val_files.extend(flist[:1])
                    test_files.extend(flist[:1])
                elif n <= 5:
                    train_files.extend(flist[:-2])
                    val_files.append(flist[-2])
                    test_files.append(flist[-1])
                else:
                    n_train = int(n * 0.8)
                    n_val = max(1, int(n * 0.1))
                    train_files.extend(flist[:n_train])
                    val_files.extend(flist[n_train:n_train + n_val])
                    test_files.extend(flist[n_train + n_val:])
            return train_files, val_files, test_files
