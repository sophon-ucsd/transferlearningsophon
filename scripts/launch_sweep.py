#!/usr/bin/env python3
"""Launch experiment sweeps locally or on Kubernetes.

Usage:
    python scripts/launch_sweep.py configs/sweeps/full_sweep.yaml --dry-run
    python scripts/launch_sweep.py configs/sweeps/smoke_test.yaml --local
    python scripts/launch_sweep.py configs/sweeps/full_sweep.yaml --kubernetes
    python scripts/launch_sweep.py configs/sweeps/full_sweep.yaml --kubernetes --submit
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_sweep(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_grid(grid: dict) -> list[dict]:
    """Expand a grid config into all combinations."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def config_to_hydra_overrides(config: dict) -> str:
    """Convert config dict to Hydra override string."""
    return " ".join(f"{k}={v}" for k, v in config.items())


def config_to_job_name(sweep_name: str, config: dict) -> str:
    """Generate a K8s-safe job name from config."""
    strategy = config.get("transfer.strategy", "unknown")
    train_size = config.get("data.train_size", "full")
    seed = config.get("project.seed", 0)
    name = f"{sweep_name}-{strategy}-{train_size}-s{seed}"
    name = name.lower().replace("_", "-")[:63]
    return name


def generate_k8s_job(
    template_path: str, job_name: str, overrides: str,
    sweep_name: str, config: dict,
) -> str:
    """Read K8s job template and fill in placeholders."""
    with open(template_path) as f:
        template = f.read()

    result = template.replace("{{JOB_NAME}}", job_name)
    result = result.replace("{{HYDRA_OVERRIDES}}", overrides)
    result = result.replace("{{SWEEP_NAME}}", sweep_name)

    # Estimate resources based on train_size
    train_size = config.get("data.train_size", 1_000_000)
    if isinstance(train_size, str):
        train_size = int(train_size)

    if train_size >= 10_000_000:
        memory, cpu, timeout = "32Gi", "8", "86400"
    elif train_size >= 1_000_000:
        memory, cpu, timeout = "16Gi", "4", "43200"
    else:
        memory, cpu, timeout = "8Gi", "4", "14400"

    result = result.replace("{{MEMORY}}", memory)
    result = result.replace("{{CPU}}", cpu)
    result = result.replace("{{TIMEOUT}}", timeout)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch experiment sweep")
    parser.add_argument("sweep_config", help="Path to sweep YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument("--local", action="store_true", help="Run sequentially on this machine")
    parser.add_argument("--kubernetes", action="store_true", help="Generate K8s job YAMLs")
    parser.add_argument("--k8s-template", default="k8s/job_template.yaml")
    parser.add_argument("--k8s-output-dir", default="k8s/jobs/")
    parser.add_argument("--submit", action="store_true", help="kubectl apply generated jobs")
    parser.add_argument("--data-dir", default=None, help="Override data.data_dir for all runs")
    parser.add_argument("--checkpoint", default=None, help="Override transfer.checkpoint for all runs")
    args = parser.parse_args()

    sweep = load_sweep(args.sweep_config)
    sweep_name = sweep.get("sweep_name", "default")
    grid = sweep["grid"]
    configs = expand_grid(grid)

    print(f"Sweep: {sweep_name}")
    print(f"Total runs: {len(configs)}")
    print(f"Grid: {', '.join(f'{k}: {len(v)} values' for k, v in grid.items())}")
    print()

    # --dry-run: just print
    if args.dry_run:
        for i, cfg in enumerate(configs):
            print(f"  [{i+1:3d}/{len(configs)}] {config_to_hydra_overrides(cfg)}")
        print(f"\nTotal: {len(configs)} runs")
        return

    # --local: run sequentially on this machine
    if args.local:
        project_root = str(Path(__file__).resolve().parent.parent)
        for i, cfg in enumerate(configs):
            overrides = config_to_hydra_overrides(cfg)
            if args.data_dir:
                overrides += f" data.data_dir={args.data_dir}"
            if args.checkpoint:
                overrides += f" transfer.checkpoint={args.checkpoint}"
            overrides += f" sweep_name={sweep_name}"

            print(f"\n{'='*60}")
            print(f"Run [{i+1}/{len(configs)}]: {overrides}")
            print(f"{'='*60}")

            cmd = f"{sys.executable} scripts/train.py {overrides}"
            result = subprocess.run(cmd, shell=True, cwd=project_root)
            if result.returncode != 0:
                print(f"FAILED: {overrides}")
        return

    # --kubernetes: generate job YAMLs
    if args.kubernetes:
        output_dir = Path(args.k8s_output_dir) / sweep_name
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = []
        for cfg in configs:
            job_name = config_to_job_name(sweep_name, cfg)
            overrides = config_to_hydra_overrides(cfg)
            if args.data_dir:
                overrides += f" data.data_dir={args.data_dir}"
            if args.checkpoint:
                overrides += f" transfer.checkpoint={args.checkpoint}"
            overrides += f" sweep_name={sweep_name}"

            job_yaml = generate_k8s_job(
                args.k8s_template, job_name, overrides, sweep_name, cfg,
            )
            job_path = output_dir / f"{job_name}.yaml"
            with open(job_path, "w") as f:
                f.write(job_yaml)
            manifest.append({
                "name": job_name,
                "path": str(job_path),
                "overrides": overrides,
            })

            if args.submit:
                subprocess.run(["kubectl", "apply", "-f", str(job_path)])
                print(f"Submitted: {job_name}")

        manifest_path = output_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f)

        print(f"\nGenerated {len(manifest)} job YAMLs in {output_dir}/")
        if not args.submit:
            print(f"To submit all: kubectl apply -f {output_dir}/")
        return

    print("Specify --dry-run, --local, or --kubernetes")


if __name__ == "__main__":
    main()
