#!/usr/bin/env python3
"""Hernandez-style transfer scaling analysis.

Takes collected results CSV and produces:
- Power-law fits per strategy: L(D) = L_inf + A * D^(-beta)
- Effective data transferred D_T for each pretrained strategy
- Hernandez fit: D_T = k * D^alpha
- Figures: D_T vs D_F, data multiplier vs D_F
- fits.json with all parameters

Usage:
    python scripts/hernandez_analysis.py results/results_summary.csv
    python scripts/hernandez_analysis.py --dummy  # test with fake data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


STRATEGY_STYLE = {
    "from_scratch": {"color": "#888888", "linestyle": "--", "marker": "s", "label": "From Scratch"},
    "frozen":       {"color": "#4477AA", "linestyle": "-",  "marker": "o", "label": "Frozen + MLP"},
    "partial_ft":   {"color": "#EE6677", "linestyle": "-",  "marker": "^", "label": "Partial Fine-tune"},
    "full_ft":      {"color": "#228833", "linestyle": "-",  "marker": "D", "label": "Full Fine-tune"},
}


def power_law(D, L_inf, A, beta):
    """L(D) = L_inf + A * D^(-beta)"""
    return L_inf + A * np.power(D, -beta)


def hernandez_law(D, k, alpha):
    """D_T(D) = k * D^alpha"""
    return k * np.power(D, alpha)


def fit_power_law(sizes, losses):
    """Fit L(D) = L_inf + A * D^(-beta). Returns dict with params and R^2."""
    x = np.asarray(sizes, dtype=float)
    y = np.asarray(losses, dtype=float)
    try:
        p0 = [y.min() * 0.9, (y.max() - y.min()) * x.min() ** 0.3, 0.3]
        bounds = ([0, 0, 0.01], [y.max(), np.inf, 5.0])
        popt, pcov = curve_fit(power_law, x, y, p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        y_pred = power_law(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {
            "L_inf": popt[0], "L_inf_err": perr[0],
            "A": popt[1], "A_err": perr[1],
            "beta": popt[2], "beta_err": perr[2],
            "R2": r2, "success": True,
        }
    except RuntimeError as e:
        return {"success": False, "error": str(e)}


def compute_effective_data(scratch_sizes, scratch_losses, pretrained_sizes, pretrained_losses):
    """For each pretrained point, find scratch size that achieves same loss.

    D_T = equivalent_scratch_size - pretrained_size
    multiplier = equivalent_scratch_size / pretrained_size
    """
    # Fit power law to scratch curve for smooth interpolation
    fit = fit_power_law(scratch_sizes, scratch_losses)
    if not fit["success"]:
        return None, fit

    L_inf, A, beta = fit["L_inf"], fit["A"], fit["beta"]

    rows = []
    for size, loss in zip(pretrained_sizes, pretrained_losses):
        gap = loss - L_inf
        if gap <= 0:
            equiv = float("inf")
        else:
            equiv = (A / gap) ** (1.0 / beta)
        d_t = max(0, equiv - size)
        mult = equiv / size if size > 0 else float("inf")
        rows.append({
            "pretrained_size": int(size),
            "pretrained_loss": loss,
            "equiv_scratch_size": equiv,
            "D_T": d_t,
            "multiplier": mult,
        })

    return pd.DataFrame(rows), fit


def fit_hernandez(d_f, d_t):
    """Fit D_T = k * D_F^alpha."""
    mask = np.isfinite(d_t) & (d_t > 0) & np.isfinite(d_f) & (d_f > 0)
    if mask.sum() < 2:
        return {"success": False, "error": "Not enough valid points"}
    x, y = d_f[mask], d_t[mask]
    try:
        popt, pcov = curve_fit(hernandez_law, x, y, p0=[1.0, 0.5], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        y_pred = hernandez_law(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {
            "k": popt[0], "k_err": perr[0],
            "alpha": popt[1], "alpha_err": perr[1],
            "R2": r2, "success": True,
        }
    except RuntimeError as e:
        return {"success": False, "error": str(e)}


def generate_dummy_data():
    """Generate plausible fake results for testing the analysis pipeline."""
    sizes = [10000, 30000, 100000, 300000, 1000000, 3000000, 10000000]
    rows = []
    for strategy, base_loss, decay_rate in [
        ("from_scratch", 2.3, 0.25),
        ("frozen", 2.1, 0.30),
        ("partial_ft", 2.0, 0.32),
        ("full_ft", 1.9, 0.35),
    ]:
        L_inf = {"from_scratch": 0.45, "frozen": 0.35, "partial_ft": 0.30, "full_ft": 0.25}[strategy]
        A = base_loss - L_inf
        for size in sizes:
            loss = L_inf + A * size ** (-decay_rate)
            acc = 1.0 - loss / 2.5  # rough conversion
            auc = 0.85 + 0.15 * (1 - loss / base_loss)
            for seed in [42, 123, 456]:
                noise = np.random.RandomState(seed + size).normal(0, 0.002)
                rows.append({
                    "strategy": strategy, "train_size": size, "seed": seed,
                    "test_loss_mean": loss + noise, "test_acc_mean": acc + noise,
                    "test_auc_macro_mean": auc + noise,
                    "test_loss_std": 0.005, "test_acc_std": 0.005, "test_auc_macro_std": 0.003,
                    "test_loss_count": 1,
                })
    df = pd.DataFrame(rows)
    # Aggregate to summary format
    summary = df.groupby(["strategy", "train_size"]).agg({
        "test_loss_mean": "mean", "test_acc_mean": "mean", "test_auc_macro_mean": "mean",
    }).reset_index()
    summary.columns = ["strategy", "train_size", "test_loss_mean", "test_acc_mean", "test_auc_macro_mean"]
    return summary


def plot_effective_data(all_dt: dict[str, pd.DataFrame], output: str):
    """Figure 3: D_T vs D_F for each pretrained strategy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, dt_df in all_dt.items():
        if dt_df is None:
            continue
        style = STRATEGY_STYLE.get(strategy, {"color": "black", "marker": "o", "label": strategy})
        valid = dt_df[np.isfinite(dt_df["D_T"]) & (dt_df["D_T"] > 0)]
        if valid.empty:
            continue
        ax.plot(valid["pretrained_size"], valid["D_T"],
                color=style["color"], marker=style["marker"], markersize=6,
                linewidth=2, label=style["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fine-tuning Dataset Size $D_F$", fontsize=12)
    ax.set_ylabel("Effective Data Transferred $D_T$", fontsize=12)
    ax.set_title("Effective Data Transferred (Hernandez Framework)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_data_multiplier(all_dt: dict[str, pd.DataFrame], output: str):
    """Figure 4: Data multiplier vs D_F."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, dt_df in all_dt.items():
        if dt_df is None:
            continue
        style = STRATEGY_STYLE.get(strategy, {"color": "black", "marker": "o", "label": strategy})
        valid = dt_df[np.isfinite(dt_df["multiplier"]) & (dt_df["multiplier"] > 1)]
        if valid.empty:
            continue
        ax.plot(valid["pretrained_size"], valid["multiplier"],
                color=style["color"], marker=style["marker"], markersize=6,
                linewidth=2, label=style["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fine-tuning Dataset Size $D_F$", fontsize=12)
    ax.set_ylabel("Data Multiplier ($D_{scratch} / D_F$)", fontsize=12)
    ax.set_title("Data Efficiency Gain from Pretraining", fontsize=13)
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5, label="No gain (1×)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", nargs="?", default=None)
    parser.add_argument("--dummy", action="store_true", help="Run with fake data")
    parser.add_argument("--output-dir", default="figures/")
    parser.add_argument("--fits-output", default="figures/fits.json")
    args = parser.parse_args()

    if args.dummy:
        print("Generating dummy data...")
        df = generate_dummy_data()
    elif args.summary_csv:
        df = pd.read_csv(args.summary_csv)
    else:
        print("Provide a summary CSV or --dummy")
        return

    strategies = sorted(df["strategy"].unique())
    print(f"Strategies: {strategies}")

    # Fit power laws per strategy
    all_fits = {}
    for strategy in strategies:
        sub = df[df["strategy"] == strategy].sort_values("train_size")
        sizes = sub["train_size"].values
        losses = sub["test_loss_mean"].values
        fit = fit_power_law(sizes, losses)
        all_fits[strategy] = fit
        if fit["success"]:
            print(f"\n{strategy}: L(D) = {fit['L_inf']:.4f} + {fit['A']:.4f} * D^(-{fit['beta']:.4f})  R²={fit['R2']:.4f}")
        else:
            print(f"\n{strategy}: fit failed — {fit.get('error')}")

    # Compute effective data transferred
    if "from_scratch" not in strategies:
        print("\nNo from_scratch baseline — cannot compute D_T")
        all_dt = {}
    else:
        scratch = df[df["strategy"] == "from_scratch"].sort_values("train_size")
        scratch_sizes = scratch["train_size"].values
        scratch_losses = scratch["test_loss_mean"].values

        all_dt = {}
        all_hernandez = {}
        for strategy in strategies:
            if strategy == "from_scratch":
                continue
            sub = df[df["strategy"] == strategy].sort_values("train_size")
            dt_df, scratch_fit = compute_effective_data(
                scratch_sizes, scratch_losses,
                sub["train_size"].values, sub["test_loss_mean"].values,
            )
            all_dt[strategy] = dt_df

            if dt_df is not None:
                print(f"\n{strategy} effective data transferred:")
                for _, row in dt_df.iterrows():
                    print(f"  D_F={row['pretrained_size']:>12,}  D_T={row['D_T']:>14,.0f}  "
                          f"multiplier={row['multiplier']:.1f}×")

                # Fit Hernandez law
                h_fit = fit_hernandez(dt_df["pretrained_size"].values, dt_df["D_T"].values)
                all_hernandez[strategy] = h_fit
                if h_fit["success"]:
                    print(f"  Hernandez fit: D_T = {h_fit['k']:.2f} * D^{h_fit['alpha']:.4f}  R²={h_fit['R2']:.4f}")

    # Save fits
    output = {
        "power_law_fits": {k: {kk: vv for kk, vv in v.items() if kk != "func"}
                           for k, v in all_fits.items()},
    }
    if all_dt:
        output["hernandez_fits"] = {k: v for k, v in all_hernandez.items()} if 'all_hernandez' in dir() else {}
        output["effective_data"] = {
            k: v.to_dict(orient="records") for k, v in all_dt.items() if v is not None
        }

    Path(args.fits_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.fits_output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFits saved to {args.fits_output}")

    # Generate plots
    if all_dt:
        plot_effective_data(all_dt, str(Path(args.output_dir) / "effective_data_transferred.pdf"))
        plot_data_multiplier(all_dt, str(Path(args.output_dir) / "data_multiplier.pdf"))


if __name__ == "__main__":
    main()
