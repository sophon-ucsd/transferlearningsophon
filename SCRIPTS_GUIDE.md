# Scripts & Code Guide

## Overview

There are two training pipelines:

1. **Frozen MLP pipeline** — trains a small MLP on precomputed Sophon embeddings (.npy files). Fast (seconds per run). No GPU required but faster with one.
2. **Full model pipeline** — trains the full Sophon ParticleTransformer on ROOT files. Supports frozen/partial_ft/full_ft/from_scratch strategies. Needs GPU. Hours per run.

There are also legacy scripts from the original codebase (before the scaling experiment infrastructure was built).

---

## Training Scripts

### scripts/train_sophon.py — Full Model Training (Hydra)

**What it does:** Trains the full Sophon ParticleTransformer model on JetClass ROOT files. This is the main training script for partial_ft, full_ft, and from_scratch strategies.

**How it works:**
1. Reads config from `configs/base.yaml` with Hydra overrides from command line
2. Creates `JetClassDataModule` which loads ROOT files, computes the 17 Sophon features, and builds train/val/test dataloaders
3. Creates `SophonTransferModel` via `create_model()` — loads pretrained weights (or not), applies freeze strategy
4. Wraps in `JetClassifier` (PyTorch Lightning module) which handles training loop, metrics, optimizer with differential LR
5. Trains with early stopping, saves best checkpoint, evaluates on test set
6. Saves `results.json` with all metrics

**Usage:**
```bash
# Full fine-tune with pretrained Sophon, 100K training jets
python scripts/train_sophon.py \
    transfer.strategy=full_ft \
    transfer.checkpoint=models/JetClassII_Sophon/model.pt \
    transfer.backbone_lr=1e-4 \
    data.train_dir=/data/JetClass/Pythia/train_100M \
    data.val_dir=/data/JetClass/Pythia/val_5M \
    data.test_dir=/data/JetClass/Pythia/test_20M \
    data.train_size=100000 \
    training.optimizer.lr=1e-3 \
    training.max_epochs=100 \
    project.seed=42

# From scratch (random init, no checkpoint)
python scripts/train_sophon.py \
    transfer.strategy=from_scratch \
    transfer.checkpoint=null \
    data.train_dir=/data/JetClass/Pythia/train_100M \
    data.train_size=100000
```

**Config (configs/base.yaml):** Contains all defaults — LR, batch size, epochs, patience, data paths, strategy, etc. Command line overrides any value.

**Strategies explained:**
- `from_scratch`: Random weights, train all 2.2M params. LR=1e-3.
- `frozen`: Load Sophon, freeze backbone, train only MLP head (35K params). LR=1e-3. (Normally use train_frozen_mlp.py instead — much faster on precomputed embeddings.)
- `partial_ft`: Load Sophon, freeze first 4 of 8 transformer blocks, train rest + head (1.2M params). LR=5e-4.
- `full_ft`: Load Sophon, unfreeze everything (2.2M params). Head LR=1e-3, backbone LR=1e-4 (differential).

**Output:** `results/{sweep_name}/{strategy}_{train_size}_{seed}/results.json`

---

### scripts/train_frozen_mlp.py — Frozen MLP Training (argparse)

**What it does:** Trains a small MLP classifier on precomputed Sophon 128-dim embeddings. This is the fast path for the "frozen" strategy — no Sophon forward pass needed.

**How it works:**
1. Loads precomputed embeddings from .npy or .csv files via `EmbeddingDataset`
2. Creates an `MLPHead` (e.g. Linear(128→256) → ReLU → Dropout → Linear(256→10))
3. Trains with AdamW + cosine LR schedule
4. Early stopping on val_loss with configurable patience
5. Final evaluation on test set with sklearn AUC
6. Saves `results.json`

**Usage:**
```bash
# With separate train/val/test embedding dirs (official splits)
python scripts/train_frozen_mlp.py \
    --train-embeddings-dir /data/embeddings_pretrained_full_02_174909_100M \
    --val-embeddings-dir /data/embeddings_val_5M \
    --test-embeddings-dir /data/embeddings_test_20M \
    --train-size 100000 \
    --architecture base \
    --epochs 50 \
    --seed 42

# With single dir (local testing, splits internally 80/10/10)
python scripts/train_frozen_mlp.py \
    --embeddings-dir embeddings/ \
    --train-size 1000 \
    --epochs 5
```

**Architecture presets:**
- `small`: hidden_dims=[128], ~17K params
- `base`: hidden_dims=[256], ~35K params (default)
- `large`: hidden_dims=[512], ~68K params
- `deep`: hidden_dims=[256, 128], ~43K params

---

### scripts/run_frozen_base.py — Frozen MLP Sweep (single process)

**What it does:** Runs the entire frozen MLP scaling sweep in one process. Loads all 100M training embeddings ONCE into memory, then loops over (size, seed) combinations. Each MLP training takes seconds — the expensive part is the one-time data load.

**How it works:**
1. Loads 100M training embeddings (~25GB float16) from .npy files
2. Loads 100K val subset for per-epoch early stopping
3. Loads 20M test set for final evaluation (once per run)
4. For each (size, seed): subsamples training data, trains MLP, evaluates, saves results.json

**Usage:**
```bash
python scripts/run_frozen_base.py \
    --train-dir /data/embeddings_pretrained_full_02_174909_100M \
    --val-dir /data/embeddings_val_5M \
    --test-dir /data/embeddings_test_20M \
    --output-dir /data/results/frozen_base \
    --sizes 3000000,10000000,30000000,100000000 \
    --epochs 200 \
    --patience 10
```

**Key details:**
- `--sizes`: Comma-separated list of training sizes (default: all 9 from 10K to 100M)
- Batch size scales automatically: 4096 for <1M, 8192 for 1M-10M, 16384 for 10M+
- For 100M size, keeps embeddings as float16 to avoid OOM (converts per-batch)
- Results saved to `{output_dir}/frozen_base_{size}_{seed}/results.json`

---

### scripts/run_frozen_sweep.py — Frozen Multi-Architecture Sweep

**What it does:** Same as run_frozen_base.py but sweeps over 4 MLP architectures × 9 sizes × 3 seeds = 108 runs. Currently not used (we focus on base only).

---

## Analysis Scripts

### scripts/collect_results.py — Aggregate Results

**What it does:** Scans a directory tree for `results.json` files, aggregates into CSV.

**Usage:**
```bash
python scripts/collect_results.py /data/results/frozen_base/
# Produces: results_all.csv and results_summary.csv
```

**Output:**
- `results_all.csv`: One row per run with all metrics
- `results_summary.csv`: Grouped by (strategy, train_size) with mean/std across seeds

---

### scripts/plot_scaling_curves.py — Scaling Curve Plots

**What it does:** Reads results_summary.csv and generates accuracy vs dataset size and loss vs dataset size plots.

**Usage:**
```bash
python scripts/plot_scaling_curves.py results/results_summary.csv --output-dir figures/
# Produces: accuracy_scaling.pdf/png, loss_scaling.pdf/png
```

**Plot details:**
- X-axis: number of training jets (log scale)
- Y-axis: test accuracy (%) or test loss
- One line per strategy with error bands (±1 std across seeds)
- Colors: from_scratch=gray, frozen=blue, partial_ft=red, full_ft=green

---

### scripts/hernandez_analysis.py — Transfer Scaling Analysis

**What it does:** Fits power laws to scaling curves and computes Hernandez effective data transferred.

**Usage:**
```bash
# On real data
python scripts/hernandez_analysis.py results/results_summary.csv

# Test with dummy data
python scripts/hernandez_analysis.py --dummy
```

**What it produces:**
- Power law fit per strategy: L(D) = L_inf + A × D^(-beta)
- Effective data transferred D_T for each pretrained strategy vs from-scratch
- Hernandez fit: D_T = k × D^alpha
- `figures/fits.json` with all fitted parameters
- `figures/effective_data_transferred.pdf` — D_T vs fine-tuning data size
- `figures/data_multiplier.pdf` — data efficiency multiplier vs data size

**Fitting method:** Log-space linear regression (grid search over L_inf, then fit log(L - L_inf) = log(A) - beta × log(D)). Bootstrap CIs on beta. Matches Kaplan/Hoffmann/Hernandez methodology.

---

### scripts/launch_sweep.py — Generate Sweep Configs / K8s Jobs

**What it does:** Reads a sweep YAML, expands the grid into all combinations, generates K8s job YAMLs.

**Usage:**
```bash
# Print all configs without running
python scripts/launch_sweep.py configs/sweeps/full_sweep.yaml --dry-run

# Generate K8s job YAMLs
python scripts/launch_sweep.py configs/sweeps/from_scratch.yaml --kubernetes

# Run locally (sequential)
python scripts/launch_sweep.py configs/sweeps/smoke_test.yaml --local
```

---

## Source Modules (src/)

### src/models/sophon_wrapper.py — Sophon Model Wrapper

**What it provides:**
- `SophonTransferModel`: wraps weaver's ParticleTransformer with Sophon's exact architecture
- `create_model(strategy, checkpoint_path, ...)`: factory that returns a configured model
- Freeze/unfreeze methods: `freeze_backbone()`, `unfreeze_backbone()`, `freeze_first_n_layers(n)`
- Differential LR: `get_param_groups(backbone_lr, head_lr)`
- Checkpoint loading with key remapping (handles 'mod.' prefix)
- Includes the `.view()` fix for the 4D attn_mask bug

**Architecture (Sophon defaults):**
- input_dim=17, embed_dims=[128, 512, 128], num_heads=8, num_layers=8
- num_cls_layers=2, pair_embed_dims=[64, 64, 64], activation='gelu'
- Embedding dim: 128 (output size of encoder)
- Total params: 2,177,790

### src/models/heads.py — Classification Heads

- `LinearHead(128, 10)`: single linear layer
- `MLPHead(128, 10, hidden_dims=[256], dropout=0.1)`: configurable MLP

### src/data/jetclass.py — ROOT File Data Pipeline

**Two dataset classes:**
- `JetClassDataset`: Pre-loads ROOT files into memory. Fast `__getitem__` (array index). Used for training data.
- `LazyJetClassDataset`: Lazy-loads with LRU cache. Low memory. Used for val/test. Respects `max_jets` to cap total size.

**Data module:**
- `JetClassDataModule`: Lightning DataModule. Supports separate train_dir/val_dir/test_dir (official splits) or single data_dir (fallback).
- Pre-selects files when train_size is small (avoids indexing 1000 files for 10K jets)
- Stream-subsamples training data (loads files one at a time, keeps only needed jets)
- Pre-selects val/test files based on val_size/test_size

**Feature computation:**
- `compute_sophon_features()`: Computes the 17 Sophon input features from raw ROOT arrays
- Matches `inference_all_classes.py` exactly (verified byte-for-byte)
- `_process_file()`: Loads one ROOT file, computes features/LV/masks/labels for all jets

### src/data/embedding_dataset.py — Precomputed Embedding Pipeline

- `EmbeddingDataset`: Loads .npy or .csv embedding files. Each directory IS one split.
- `create_embedding_dataloaders()`: Creates train/val/test loaders from separate dirs or single dir with internal split.
- Auto-detects format (npy vs csv)
- Label mapping: `NPY_CLASS_MAP` maps class names to integer labels matching LABEL_KEYS argmax order

### src/data/subsampler.py — Stratified Subsampling

- `stratified_subsample(labels, target_size, seed)`: Returns indices for equal-per-class subsampling. Deterministic given seed.

### src/training/trainer.py — Lightning Training Module

- `JetClassifier`: Handles forward pass, loss, metrics (torchmetrics Accuracy + AUROC), optimizer config
- Differential LR for full_ft (backbone_lr vs head_lr)
- Cosine LR schedule with optional linear warmup
- `save_results()`: Saves results.json with all metrics

### src/utils/reproducibility.py — Seed Utility

- `seed_everything(seed)`: Sets torch, numpy, random, CUDA seeds + deterministic flags

---

## Legacy Scripts (from before scaling experiment)

These are from the original codebase. Still functional but not used in the scaling sweeps.

- `scripts/train_mlp.py` — Original MLP trainer (uses older data loading, different interface)
- `scripts/run_sweep.py` — Original sweep runner for arch/size sweeps
- `scripts/plot_results.py` — Original plotting script
- `inference_all_classes.py` — Sophon inference script (extracts embeddings from ROOT files)
- `networks/example_ParticleTransformer_sophon.py` — Original Sophon model wrapper (used by inference_all_classes.py)

---

## Data Flow Diagram

```
ROOT files (.root)                    Precomputed embeddings (.npy)
     │                                        │
     ▼                                        ▼
 jetclass.py                          embedding_dataset.py
 (compute 17 features,                (load 128-dim vectors,
  Lorentz vectors, masks)              map class→label)
     │                                        │
     ▼                                        ▼
 JetClassDataModule                   EmbeddingDataset /
 (train/val/test splits,              create_embedding_dataloaders
  subsampling)                        (train/val/test splits)
     │                                        │
     ▼                                        ▼
 train_sophon.py                      train_frozen_mlp.py /
 (full model: frozen/                 run_frozen_base.py
  partial_ft/full_ft/                 (MLP head only)
  from_scratch)                              │
     │                                        │
     ▼                                        ▼
 results.json                         results.json
     │                                        │
     └──────────────┬─────────────────────────┘
                    ▼
            collect_results.py
            (aggregate → CSV)
                    │
                    ▼
         plot_scaling_curves.py
         hernandez_analysis.py
         (scaling laws, figures)
```
