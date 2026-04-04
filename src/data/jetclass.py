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

# ---------------------------------------------------------------------------
# Constants — must match inference_all_classes.py exactly
# ---------------------------------------------------------------------------
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

# Class name (from filename) -> integer label (matching label_keys order)
CLASS_TO_LABEL = {
    "HToBB": 0, "HToCC": 1, "HToGG": 2, "HToWW4Q": 3, "HToWW2Q1L": 4,
    "TTBar": 5, "TTBarLep": 6, "WToQQ": 7, "ZToQQ": 8, "ZJetsToNuNu": 9,
}

# Reverse: label index -> label key order matches LABEL_KEYS
# label_QCD=0, label_Hbb=1, ... but wait — the label from argmax of label_keys
# gives: QCD=0, Hbb=1, Hcc=2, Hgg=3, H4q=4, Hqql=5, Zqq=6, Wqq=7, Tbqq=8, Tbl=9
# And CLASS_TO_LABEL maps filename -> these same indices. Verify below.
# HToBB -> label_Hbb is at index 1 in LABEL_KEYS... but CLASS_TO_LABEL says 0.
# Actually looking at the existing code: label_names = ["QCD","Hbb","Hcc","Hgg",
# "H4q","Hqql","Zqq","Wqq","Tbqq","Tbl"] and get_truth_label does argmax over
# label_keys which are [label_QCD, label_Hbb, ...]. So QCD=0, Hbb=1, etc.
# But HToBB files have label_Hbb=1 → argmax=1. So HToBB -> label 1, not 0.
# Let me fix CLASS_TO_LABEL to match the actual argmax:
CLASS_TO_LABEL = {
    "HToBB": 1,       # label_Hbb
    "HToCC": 2,       # label_Hcc
    "HToGG": 3,       # label_Hgg
    "HToWW4Q": 4,     # label_H4q
    "HToWW2Q1L": 5,   # label_Hqql
    "TTBar": 8,        # label_Tbqq
    "TTBarLep": 9,     # label_Tbl
    "WToQQ": 7,        # label_Wqq
    "ZToQQ": 6,        # label_Zqq
    "ZJetsToNuNu": 0,  # label_QCD  (ZJetsToNuNu is QCD-like)
}
# NOTE: We won't hardcode labels — we read them from the ROOT file directly.
# CLASS_TO_LABEL is only a fallback / for reference.

NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Feature computation — copied verbatim from inference_all_classes.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class JetClassDataset(Dataset):
    """Lazy-loading JetClass ROOT dataset.

    Builds an index over all files, loads one file at a time on demand.
    """

    def __init__(self, file_list: list[str | Path], max_particles: int = MAX_PART) -> None:
        self.file_list = [str(p) for p in file_list]
        self.max_particles = max_particles

        # Build global index: list of (file_idx, local_idx, label)
        self._index: list[tuple[int, int, int]] = []
        self._file_num_entries: list[int] = []

        for file_idx, fpath in enumerate(self.file_list):
            with uproot.open(f"{fpath}:{TREE_NAME}") as tree:
                n_entries = tree.num_entries
            self._file_num_entries.append(n_entries)
            # We'll read labels when we actually load the file
            for local_idx in range(n_entries):
                self._index.append((file_idx, local_idx, -1))  # label filled on load

        # Cache: last loaded file data
        self._cached_file_idx: int = -1
        self._cached_arrays: dict | None = None
        self._cached_labels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._index)

    def _load_file(self, file_idx: int) -> None:
        """Load a ROOT file into the cache."""
        if file_idx == self._cached_file_idx:
            return
        fpath = self.file_list[file_idx]
        with uproot.open(f"{fpath}:{TREE_NAME}") as tree:
            arrays = tree.arrays(ALL_KEYS, library="np")
        # Compute labels from label columns
        label_matrix = np.stack([arrays[k] for k in LABEL_KEYS], axis=1)  # (N, 10)
        labels = np.argmax(label_matrix, axis=1)  # (N,)
        self._cached_file_idx = file_idx
        self._cached_arrays = arrays
        self._cached_labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, local_idx, _ = self._index[idx]
        self._load_file(file_idx)
        arrays = self._cached_arrays
        i = local_idx

        # --- Get raw particle arrays for this jet ---
        px = arrays["part_px"][i]
        py = arrays["part_py"][i]
        pz = arrays["part_pz"][i]
        energy = arrays["part_energy"][i]
        n_part = len(px)

        # pT-sort and truncate to max_particles
        pt = np.sqrt(px ** 2 + py ** 2)
        if n_part > self.max_particles:
            keep = np.argsort(pt)[::-1][:self.max_particles]
            n_part = self.max_particles
        else:
            keep = np.argsort(pt)[::-1]  # still sort by descending pT

        # Apply keep index to all particle arrays
        px = px[keep]
        py = py[keep]
        pz = pz[keep]
        energy = energy[keep]

        jet_pt = float(arrays["jet_pt"][i])
        jet_energy = float(arrays["jet_energy"][i])
        jet_pt_safe = max(jet_pt, 1e-20)

        # --- Lorentz vectors: scaled by 500/jet_pt ---
        lv = np.stack([
            px * 500.0 / jet_pt_safe,
            py * 500.0 / jet_pt_safe,
            pz * 500.0 / jet_pt_safe,
            energy * 500.0 / jet_pt_safe,
        ], axis=1).astype(np.float32)  # (n_part, 4)

        # --- 17 Sophon features ---
        feats = compute_sophon_features(
            px=px, py=py, pz=pz, energy=energy,
            deta=arrays["part_deta"][i][keep],
            dphi=arrays["part_dphi"][i][keep],
            d0val=arrays["part_d0val"][i][keep],
            d0err=arrays["part_d0err"][i][keep],
            dzval=arrays["part_dzval"][i][keep],
            dzerr=arrays["part_dzerr"][i][keep],
            charge=arrays["part_charge"][i][keep],
            isChargedHadron=arrays["part_isChargedHadron"][i][keep],
            isNeutralHadron=arrays["part_isNeutralHadron"][i][keep],
            isPhoton=arrays["part_isPhoton"][i][keep],
            isElectron=arrays["part_isElectron"][i][keep],
            isMuon=arrays["part_isMuon"][i][keep],
            jet_pt=jet_pt,
            jet_energy=jet_energy,
        )  # (n_part, 17)

        # --- Pad to max_particles ---
        feat_padded = np.zeros((self.max_particles, 17), dtype=np.float32)
        feat_padded[:n_part] = feats

        lv_padded = np.zeros((self.max_particles, 4), dtype=np.float32)
        lv_padded[:n_part] = lv

        mask = np.zeros(self.max_particles, dtype=bool)
        mask[:n_part] = True

        label = int(self._cached_labels[local_idx])

        return {
            "features": torch.from_numpy(feat_padded),          # (128, 17)
            "lorentz_vectors": torch.from_numpy(lv_padded),     # (128, 4)
            "mask": torch.from_numpy(mask),                     # (128,)
            "label": label,
        }

    def get_all_labels(self) -> np.ndarray:
        """Return labels for the full dataset (loads all files)."""
        labels = []
        for file_idx in range(len(self.file_list)):
            self._load_file(file_idx)
            labels.append(self._cached_labels.copy())
        return np.concatenate(labels)


# ---------------------------------------------------------------------------
# Collate function — transposes to model format
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Data Module
# ---------------------------------------------------------------------------
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

        Splits files per class: first 80% train, next 10% val, last 10% test.
        Applies stratified subsampling to train set if train_size is specified.
        """

        def __init__(
            self,
            data_dir: str,
            train_size: int | None = None,
            val_size: int | None = None,
            test_size: int | None = None,
            batch_size: int = 512,
            num_workers: int = 8,
            seed: int = 42,
        ) -> None:
            super().__init__()
            self.data_dir = data_dir
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
            files_by_class = _discover_files(self.data_dir)
            if not files_by_class:
                raise FileNotFoundError(f"No ROOT files found in {self.data_dir}")

            train_files, val_files, test_files = [], [], []

            for cls_name, flist in sorted(files_by_class.items()):
                n = len(flist)
                if n <= 2:
                    # Very few files: use all for train, first for val/test
                    train_files.extend(flist)
                    val_files.extend(flist[:1])
                    test_files.extend(flist[:1])
                elif n <= 5:
                    # Small dataset (e.g. val_5M with 5 files): 3 train, 1 val, 1 test
                    train_files.extend(flist[:-2])
                    val_files.append(flist[-2])
                    test_files.append(flist[-1])
                else:
                    # Normal split: 80/10/10
                    n_train = int(n * 0.8)
                    n_val = max(1, int(n * 0.1))
                    train_files.extend(flist[:n_train])
                    val_files.extend(flist[n_train:n_train + n_val])
                    test_files.extend(flist[n_train + n_val:])

            print(f"Files — train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

            self.train_dataset = JetClassDataset(train_files)
            self.val_dataset = JetClassDataset(val_files)
            self.test_dataset = JetClassDataset(test_files)

            # Apply subsampling to train set
            if self.train_size is not None and self.train_size < len(self.train_dataset):
                labels = self.train_dataset.get_all_labels()
                indices = stratified_subsample(labels, self.train_size, self.seed)
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
                print(f"Subsampled train set: {len(self.train_dataset)} jets")

            # Apply subsampling to val/test if specified
            if self.val_size is not None and self.val_size < len(self.val_dataset):
                labels = self.val_dataset.get_all_labels()
                indices = stratified_subsample(labels, self.val_size, self.seed)
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, indices)

            if self.test_size is not None and self.test_size < len(self.test_dataset):
                labels = self.test_dataset.get_all_labels()
                indices = stratified_subsample(labels, self.test_size, self.seed)
                self.test_dataset = torch.utils.data.Subset(self.test_dataset, indices)

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
