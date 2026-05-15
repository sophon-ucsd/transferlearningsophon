"""Deterministic stratified subsampling for dataset size sweeps."""
from __future__ import annotations

import numpy as np


def stratified_subsample(
    labels: np.ndarray,
    target_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Subsample indices stratified by class label.

    Args:
        labels: 1D array of integer class labels for the full dataset.
        target_size: desired total number of samples.
        seed: random seed for reproducibility.

    Returns:
        Sorted 1D array of selected indices.
    """
    if target_size >= len(labels):
        return np.arange(len(labels))

    rng = np.random.RandomState(seed)
    unique_classes = np.unique(labels)
    per_class = target_size // len(unique_classes)

    selected = []
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        if per_class >= len(cls_indices):
            selected.append(cls_indices)
        else:
            chosen = rng.choice(cls_indices, size=per_class, replace=False)
            selected.append(chosen)

    return np.sort(np.concatenate(selected))
