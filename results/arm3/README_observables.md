# Per-jet observables — definitions

All observables are computed in [scripts/compute_substructure.py](../../scripts/compute_substructure.py).

Inputs are JetClass-1 ROOT files: per-particle 4-momenta (`part_px/py/pz/energy`),
impact parameters and uncertainties (`part_d0val/d0err/dzval/dzerr`) and `part_charge`.

## Substructure

| Name | Formula |
|---|---|
| `jet_mass` | sqrt(max(0, E² − p²)) where `E = Σ E_i`, `p = Σ p_i` |
| `jet_pt` | sqrt((Σ pₓ)² + (Σ pᵧ)²) |
| `multiplicity` | Number of particles with pT > 0.5 GeV |
| `width` | Σ pT_i ΔR_i / Σ pT_i, with ΔR from pT-weighted jet axis |
| `tau_N` | Σ pT_i · min_a ΔR_{i,a} / d₀, axes = N hardest particles, d₀ = R · Σ pT (R=0.8) |
| `tau21`, `tau32` | tau2/tau1, tau3/tau2 |
| `C2` | e3 / e2³, with e_n = ECF on the 30 hardest particles (Larkoski-Salam-Thaler 2013) |

## Track displacement (corrected)

All operate on **charged particles only** (mask: `d0err > 0` from JetClass-1 reconstruction). Per-particle |d0| significance is `|d0| / d0err`.

| Name | Formula | Why |
|---|---|---|
| `mean_abs_d0` | mean(\|d0\|) over charged tracks | **Original probe target — diluted by prompt tracks** |
| `max_abs_d0` | max(\|d0\|) over charged tracks | Most-displaced single track; survives in dense events |
| `top3_sum_abs_d0` | Σ_{top 3} \|d0\| | Robust to outliers; signal of 2-3 displaced tracks (b-meson decay) |
| `count_d0_gt_1sigma` | #{i : \|d0_i\|/d0err_i > 1} | Significance-based, scale-invariant |
| `count_d0_gt_2sigma` | #{i : \|d0_i\|/d0err_i > 2} | Stricter cut; targets clear displaced vertices |
| `mean_abs_dz` … `count_dz_gt_2sigma` | Same family for the longitudinal IP (`dz`) | Parallel set; lower discriminating power for b-tagging |

`mean_d0` and `mean_dz` are kept as aliases of `mean_abs_d0` / `mean_abs_dz` for backward compatibility with earlier figures.

## Auxiliary

| Name | Formula |
|---|---|
| `charged_frac` | mean(charge ≠ 0) over all particles |
| `n_charged` | count of charged particles (`d0err > 0`) |

## Why the d0 correction matters

Sophon's pretrained embedding linearly probes for `mean_abs_d0` at R² ≈ 0.04 — easy to interpret as "discards displacement info." But b-meson decay produces 2-3 displaced tracks among ~50 prompt tracks; the mean is dominated by the prompt majority and washes the signal out. Probing for `max_abs_d0` or `count_d0_gt_2sigma` recovers the displacement information at R² > 0.7 (predicted), confirming the embedding *does* encode it — just not in the dimension that the mean exposes.
