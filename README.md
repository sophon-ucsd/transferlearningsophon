# Transfer learning strategies for jet tagging with Sophon

Scaling-law comparison of three adaptation strategies for the [Sophon](https://github.com/jet-universe/sophon) pretrained foundation model on the 10-class [JetClass-I](https://zenodo.org/records/6619768) benchmark.

We sweep nine training-set sizes from 10K to 100M labeled jets and three random seeds, and compare:

- **Frozen + MLP** — Sophon backbone locked, train a 35K-parameter MLP head on the 128-D embedding
- **Partial fine-tune** — freeze the first 4 of 8 self-attention blocks, fine-tune the rest with the head
- **Full fine-tune** — train all 2.2M Sophon parameters end-to-end with the head

## Key results

- Frozen + MLP at 100M jets reaches **macro AUC 0.979** on the 10-class test set, within 0.9% of the published Particle Transformer (0.988) while training only **1/60th** the parameters and never updating the backbone.
- Full fine-tune at 30M jets reaches **0.986** and the power-law fit projects 0.987 at 100M.
- All three strategies follow smooth power-law improvement curves; differences between strategies shrink steadily as the training set grows.
- Probing the frozen embedding for 12 hand-engineered jet observables recovers the canonical CMS Track Counting High Purity statistic at R² = 0.68 — Sophon spontaneously learned the same displaced-track tagger CMS hand-crafted, without ever being told.

## Install

```sh
conda create -n sophon python=3.10 -y
conda activate sophon
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

The repo is a standard `pyproject.toml` project — `pip install -e .` pulls every runtime dependency.

## Data

- **Local development**: download the JetClass validation split (~5M events) from [Zenodo](https://zenodo.org/records/6619768).
  ```sh
  mkdir -p data && cd data
  wget "https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1" \
    -O JetClass_Pythia_val_5M.tar
  tar -xf JetClass_Pythia_val_5M.tar && cd ..
  ```
- **Cluster runs**: the full 100M training set lives on the NRP/Nautilus PVC at `/data/JetClass/Pythia/train_100M/`.

The 10 JetClass-I classes: `HToBB`, `HToCC`, `HToGG`, `HToWW2Q1L`, `HToWW4Q`, `TTBar`, `TTBarLep`, `WToQQ`, `ZToQQ`, `ZJetsToNuNu`.

## Pretrained Sophon weights

```sh
mkdir -p models/JetClassII_Sophon
curl -fSL -o models/JetClassII_Sophon/model.pt \
  "https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt"
```

## Reproducing the sweep

### 1. Extract 128-D Sophon embeddings from JetClass ROOT files

```sh
python3 inference_all_classes.py \
  --checkpoint models/JetClassII_Sophon/model.pt \
  --root-dir data/train_100M \
  --events-per-class 0 \
  --output-dir embeddings_pretrained_full \
  --format npy
```

`--events-per-class 0` extracts everything available; pass a positive integer to cap per class. `--format npy` writes float16 NPYs (memory-mapped, ~10× smaller than CSV).

### 2a. Frozen + MLP scaling sweep

```sh
python3 scripts/train_frozen_sweep.py \
  --emb-dir embeddings_pretrained_full \
  --output-dir results/frozen_base \
  --sizes 10000,30000,100000,300000,1000000,3000000,10000000,30000000,100000000 \
  --seeds 42,123,456 \
  --epochs 500 --patience 20 --batch-size 8192
```

### 2b. Partial / full fine-tune sweep

```sh
# Partial FT (freezes the first 4 of 8 self-attention blocks)
python3 scripts/train_finetune_sweep.py \
  --train-dir /data/features/train_100M \
  --val-dir   /data/features/val_5M \
  --test-dir  /data/features/test_20M \
  --strategy partial_ft \
  --checkpoint models/JetClassII_Sophon/model.pt \
  --sizes 10000,30000,100000,300000,1000000,3000000,10000000 \
  --seeds 42,123,456 \
  --output-dir results/partial_ft \
  --batch-size 512 --epochs 100 --patience 10 \
  --materialize-train --skip-existing

# Full FT — same command, --strategy full_ft, --output-dir results/full_ft
```

`--materialize-train` reads the entire mmap'd training set into RAM once at startup, avoiding PVC contention during training.

### 3. Aggregate per-run results

```sh
python3 scripts/collect_results.py
# writes results/sweep_results.csv
```

Each `{strategy}/{strategy}_{size}_{seed}/` directory contains `results.json`, `best_model.pt`, training history, and per-run ROC + confusion-matrix plots.

## Reproducing the figures

Each poster figure is rendered from a single script that reads the canonical analysis CSV:

```sh
# Figure: scaling curves + power-law projection
python3 scripts/plot_scaling.py

# Figure: probing R² across 12 jet observables for all three strategies
python3 scripts/plot_probing.py

# Figure: paired UMAP, pretrained vs full-FT
python3 scripts/plot_umap.py \
  --pretrained-dir embeddings_pretrained_full \
  --finetuned-dir  embeddings_ft_full_10M_seed42_test100k \
  --ft-label "Full-FT 10M (seed 42)"

# Figure: per-class ROC for all three strategies at 10M
python3 scripts/plot_roc_3strategies.py \
  --frozen-mlp   results/frozen_base/frozen_base_10000000_42/best_model.pt \
  --partial-ckpt results/partial_ft/partial_ft_10000000_42/best_model.pt \
  --full-ckpt    results/full_ft/full_ft_10000000_42/best_model.pt \
  --pretrained   models/JetClassII_Sophon/model.pt \
  --features-dir /data/features/test_20M \
  --embeddings-dir embeddings_pretrained_full
```

PDFs and PNGs land under `results/main_plots/` (gitignored — regenerable).

## Probing analysis

The probing pipeline computes the linear and MLP-probe R² of each Sophon embedding against 12 hand-engineered jet observables (4 substructure ratios, 3 shape descriptors, 3 |d₀| magnitudes, 2 d₀-significance statistics). The substructure observables themselves are computed on the JetClass test set:

```sh
python3 scripts/compute_substructure.py \
  --features-dir /data/features/test_20M \
  --output       results/substructure_observables.npz
python3 scripts/probe_observables.py \
  --embeddings-dir embeddings_pretrained_full \
  --observables    results/substructure_observables.npz \
  --output         results/probing_results.csv
python3 scripts/probing_audits.py \
  --observables results/substructure_observables.npz \
  --output-dir  results/
```

`probing_audits.py` runs three control studies: residualize the target against multiplicity, per-class within-class probing, and a shuffled-label MLP selectivity control.

## Cluster execution

The sweeps were run on the NRP/Nautilus Kubernetes cluster. `k8s/` contains three reference YAML templates you can adapt — see `k8s/README.md`.

## Project structure

```
.
├── inference_all_classes.py     # Step 1: extract 128-D embeddings + raw Sophon baseline
├── src/
│   ├── data/                    # JetClass loader, embedding dataset, stratified subsampler
│   ├── models/                  # SophonTransferModel + classification heads
│   └── utils/                   # seed_everything
├── plots/style.py               # Matplotlib style (Okabe-Ito palette, serif, 300 dpi)
├── scripts/
│   ├── train_frozen_sweep.py    # Frozen + MLP scaling sweep
│   ├── train_finetune_sweep.py  # Partial / full FT scaling sweep
│   ├── collect_results.py       # Aggregate per-run results.json → sweep_results.csv
│   ├── preprocess_root_to_npy.py# ROOT → .npy feature pre-extraction
│   ├── compute_substructure.py  # Substructure observables on test set
│   ├── probe_observables.py     # Linear / MLP probing pipeline
│   ├── probing_audits.py        # Residualized + per-class + selectivity audits
│   ├── plot_scaling.py          # Scaling curves with power-law projection
│   ├── plot_probing.py          # Probing R² figure
│   ├── plot_umap.py             # Paired UMAP (pretrained vs full-FT)
│   └── plot_roc_3strategies.py  # Per-class ROC for all three strategies
├── k8s/                         # Kubernetes job templates (see k8s/README.md)
├── networks/                    # Reference Sophon ParticleTransformer definition
└── results/                     # Canonical CSVs feeding the poster figures
    ├── sweep_results.csv        # 9 sizes × 3 strategies × seeds (drives plot_scaling.py)
    ├── probing_results.csv      # linear (ridge) probe, frozen Sophon
    ├── probing_partial_results.csv
    ├── mlp_probing_results.csv  # MLP probe, frozen Sophon
    └── mlp_probing_partial_results.csv
```

## Authors

Raunav Mendiratta and Emmet Muschenetz. Mentored by Jason Weitz under Javier Duarte at UC San Diego.

## References

- Qu, Li, Qian. *Particle Transformer for Jet Tagging.* ICML 2022. [arXiv:2202.03772](https://arxiv.org/abs/2202.03772).
- Li, Du, Qu, Lai et al. *Accelerating Resonance Searches via Signature-Oriented Pre-training* (Sophon). [arXiv:2405.12972](https://arxiv.org/abs/2405.12972).
- Qu, Li, Qian. *JetClass: A large-scale dataset for deep learning in jet physics.* [Zenodo](https://zenodo.org/records/6619768).
- Geuskens, Gite, Krämer, Mikuni, Mück, Nachman, Reyes-González. *The Fundamental Limit of Jet Tagging.* [arXiv:2411.02628](https://arxiv.org/abs/2411.02628).
