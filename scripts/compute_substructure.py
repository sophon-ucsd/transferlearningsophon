#!/usr/bin/env python3
"""Compute jet substructure observables from JetClass ROOT files.

Computes: jet mass, pT, multiplicity, tau1/2/3, tau21/32, C2, width,
mean |d0|, mean |dz|, charged fraction.

Usage:
    python scripts/compute_substructure.py \
        --data-dir data/val_5M \
        --output substructure_observables.npz \
        --n-jets 50000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.subsampler import stratified_subsample

TREE_NAME = "tree"
PARTICLE_KEYS = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_dzval", "part_charge",
]
LABEL_KEYS = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
    "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
    "label_Tbqq", "label_Tbl",
]
JET_R = 0.8  # AK8 jets


def compute_observables_for_jet(px, py, pz, energy, d0val, dzval, charge):
    """Compute all substructure observables for one jet.

    Args:
        px, py, pz, energy: particle 4-momenta arrays
        d0val, dzval: impact parameters
        charge: particle charges

    Returns:
        dict of observable name -> value
    """
    n_part = len(px)
    pt = np.sqrt(px**2 + py**2)

    # --- Jet kinematics ---
    sum_px = np.sum(px)
    sum_py = np.sum(py)
    sum_pz = np.sum(pz)
    sum_E = np.sum(energy)

    jet_pt = np.sqrt(sum_px**2 + sum_py**2)
    m2 = sum_E**2 - sum_px**2 - sum_py**2 - sum_pz**2
    jet_mass = np.sqrt(max(m2, 0.0))

    # --- Multiplicity (pT > 0.5 GeV) ---
    multiplicity = int(np.sum(pt > 0.5))

    # --- Jet axis (pT-weighted centroid in eta-phi) ---
    eta = np.arctanh(np.clip(pz / np.sqrt(px**2 + py**2 + pz**2 + 1e-20), -0.9999, 0.9999))
    phi = np.arctan2(py, px)
    pt_sum = np.sum(pt) + 1e-20
    jet_eta = np.sum(pt * eta) / pt_sum
    jet_phi = np.arctan2(np.sum(pt * np.sin(phi)), np.sum(pt * np.cos(phi)))

    # deltaR from jet axis
    deta = eta - jet_eta
    dphi = phi - jet_phi
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # wrap to [-pi, pi]
    deltaR = np.sqrt(deta**2 + dphi**2)

    # --- Jet width ---
    width = np.sum(pt * deltaR) / pt_sum

    # --- N-subjettiness (simplified: N hardest particles as axes) ---
    d0 = np.sum(pt) * JET_R  # normalization

    # Sort by pT descending
    order = np.argsort(pt)[::-1]

    def compute_tau(n_axes):
        if n_part < n_axes or d0 < 1e-20:
            return 0.0
        # Use N hardest particles as axes
        axes_idx = order[:n_axes]
        axes_eta = eta[axes_idx]
        axes_phi = phi[axes_idx]

        # For each particle, find min deltaR to any axis
        min_dr = np.full(n_part, np.inf)
        for a in range(n_axes):
            de = eta - axes_eta[a]
            dp = phi - axes_phi[a]
            dp = np.arctan2(np.sin(dp), np.cos(dp))
            dr = np.sqrt(de**2 + dp**2)
            min_dr = np.minimum(min_dr, dr)

        return np.sum(pt * min_dr) / d0

    tau1 = compute_tau(1)
    tau2 = compute_tau(2)
    tau3 = compute_tau(3)
    tau21 = tau2 / max(tau1, 1e-20)
    tau32 = tau3 / max(tau2, 1e-20)

    # --- C2 energy correlation function (use 30 hardest particles) ---
    n_ecf = min(30, n_part)
    top_idx = order[:n_ecf]
    pt_top = pt[top_idx]
    eta_top = eta[top_idx]
    phi_top = phi[top_idx]
    pt_sum_top = np.sum(pt_top) + 1e-20

    # e2 = sum_{i<j} pT_i * pT_j * deltaR_ij / (sum pT)^2
    e2 = 0.0
    e3 = 0.0
    # Precompute pairwise deltaR
    dr_matrix = np.zeros((n_ecf, n_ecf))
    for i in range(n_ecf):
        for j in range(i + 1, n_ecf):
            de = eta_top[i] - eta_top[j]
            dp = phi_top[i] - phi_top[j]
            dp = np.arctan2(np.sin(dp), np.cos(dp))
            dr_matrix[i, j] = np.sqrt(de**2 + dp**2)
            dr_matrix[j, i] = dr_matrix[i, j]
            e2 += pt_top[i] * pt_top[j] * dr_matrix[i, j]

    e2 /= pt_sum_top**2

    # e3 = sum_{i<j<k} pT_i*pT_j*pT_k * dR_ij*dR_ik*dR_jk / (sum pT)^3
    for i in range(n_ecf):
        for j in range(i + 1, n_ecf):
            for k in range(j + 1, n_ecf):
                e3 += (pt_top[i] * pt_top[j] * pt_top[k] *
                       dr_matrix[i, j] * dr_matrix[i, k] * dr_matrix[j, k])
    e3 /= pt_sum_top**3

    C2 = e3 / max(e2**3, 1e-30) if e2 > 1e-20 else 0.0

    # --- Track-based observables ---
    mean_d0 = np.mean(np.abs(d0val)) if len(d0val) > 0 else 0.0
    mean_dz = np.mean(np.abs(dzval)) if len(dzval) > 0 else 0.0
    charged_frac = np.mean(charge != 0) if len(charge) > 0 else 0.0

    return {
        "jet_mass": jet_mass,
        "jet_pt": jet_pt,
        "multiplicity": multiplicity,
        "tau1": tau1,
        "tau2": tau2,
        "tau3": tau3,
        "tau21": tau21,
        "tau32": tau32,
        "C2": C2,
        "width": width,
        "mean_d0": mean_d0,
        "mean_dz": mean_dz,
        "charged_frac": charged_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="substructure_observables.npz")
    parser.add_argument("--n-jets", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Discover ROOT files
    data_dir = Path(args.data_dir)
    root_files = sorted(data_dir.glob("*.root"))
    print(f"Found {len(root_files)} ROOT files in {data_dir}")

    # Load all jets, collect labels for stratified sampling
    print("Scanning files for jet counts...")
    file_entries = []
    for f in root_files:
        with uproot.open(f"{f}:{TREE_NAME}") as tree:
            file_entries.append(tree.num_entries)
    total_jets = sum(file_entries)
    print(f"Total: {total_jets:,} jets")

    # Build global index -> (file_idx, local_idx) and collect labels
    print("Loading labels for stratified sampling...")
    all_labels = []
    for f in root_files:
        with uproot.open(f"{f}:{TREE_NAME}") as tree:
            lab_arrays = tree.arrays(LABEL_KEYS, library="np")
        label_matrix = np.stack([lab_arrays[k] for k in LABEL_KEYS], axis=1)
        all_labels.append(np.argmax(label_matrix, axis=1))
    all_labels = np.concatenate(all_labels)

    # Stratified subsample
    indices = stratified_subsample(all_labels, args.n_jets, args.seed)
    print(f"Selected {len(indices):,} jets (stratified)")

    # Map global indices to (file_idx, local_idx)
    cum_sizes = np.cumsum(file_entries)
    file_for_idx = np.searchsorted(cum_sizes, indices, side="right")
    local_for_idx = indices - np.concatenate([[0], cum_sizes[:-1]])[file_for_idx]

    # Group by file for efficient loading
    from collections import defaultdict
    file_groups = defaultdict(list)
    for i, (fi, li) in enumerate(zip(file_for_idx, local_for_idx)):
        file_groups[fi].append((i, li))

    # Compute observables
    obs_names = ["jet_mass", "jet_pt", "multiplicity", "tau1", "tau2", "tau3",
                 "tau21", "tau32", "C2", "width", "mean_d0", "mean_dz", "charged_frac"]
    results = {name: np.zeros(len(indices)) for name in obs_names}
    result_labels = np.zeros(len(indices), dtype=int)

    t0 = time.time()
    files_done = 0
    for fi, group in sorted(file_groups.items()):
        fpath = root_files[fi]
        with uproot.open(f"{fpath}:{TREE_NAME}") as tree:
            arrays = tree.arrays(PARTICLE_KEYS + LABEL_KEYS, library="np")

        label_matrix = np.stack([arrays[k] for k in LABEL_KEYS], axis=1)
        labels = np.argmax(label_matrix, axis=1)

        for out_idx, local_idx in group:
            obs = compute_observables_for_jet(
                px=arrays["part_px"][local_idx],
                py=arrays["part_py"][local_idx],
                pz=arrays["part_pz"][local_idx],
                energy=arrays["part_energy"][local_idx],
                d0val=arrays["part_d0val"][local_idx],
                dzval=arrays["part_dzval"][local_idx],
                charge=arrays["part_charge"][local_idx],
            )
            for name in obs_names:
                results[name][out_idx] = obs[name]
            result_labels[out_idx] = labels[local_idx]

        files_done += 1
        if files_done % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {files_done}/{len(file_groups)} files, {elapsed:.0f}s")

    # Save
    np.savez(args.output, labels=result_labels, **results)
    elapsed = time.time() - t0
    print(f"\nDone: {len(indices):,} jets, {elapsed:.0f}s")
    print(f"Saved to {args.output}")

    # Print summary
    from collections import Counter
    print(f"\nClass distribution: {dict(sorted(Counter(result_labels).items()))}")
    for name in obs_names:
        vals = results[name]
        print(f"  {name:>15}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")


if __name__ == "__main__":
    main()
