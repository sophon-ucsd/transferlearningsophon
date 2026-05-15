# transferlearningsophon

## Transfer Learning with Sophon for Jet Classification

This project explores representation learning in jet physics using Sophon, a pretrained ParticleTransformer foundation model. We extract frozen 128-D embeddings from Sophon and train lightweight classifiers (MLP heads and linear probes) on them to evaluate transfer quality across all 10 JetClass-I jet types.

The pipeline:
1. Run Sophon inference on JetClass ROOT files to extract embeddings (+ raw Sophon zero-shot baseline)
2. Train MLP classifiers on frozen embeddings, sweeping over architecture and dataset size
3. Compare against the raw Sophon baseline

## Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU inference)
- weaver-core >= 0.4.0
- uproot, awkward, numpy, tqdm, scikit-learn, pandas, matplotlib

## Install

```sh
conda create -n sophon python=3.10 -y
conda activate sophon
# Install PyTorch (pick the right command for your CUDA)
# See https://pytorch.org/get-started/locally/ for your platform
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

The full training set (`train_100M`) is available on the Nautilus PVC at `/data/JetClass/Pythia/train_100M/`. It contains 100 ROOT files per class (~10M events/class, 100M total).

For local development, download the smaller validation set (~5M events) from https://zenodo.org/records/6619768:

```sh
mkdir -p data
cd data
wget "https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1" -O JetClass_Pythia_val_5M.tar
tar -xf JetClass_Pythia_val_5M.tar
cd ..
```

The 10 JetClass-I classes: `HToBB`, `HToCC`, `HToGG`, `HToWW2Q1L`, `HToWW4Q`, `TTBar`, `TTBarLep`, `WToQQ`, `ZToQQ`, `ZJetsToNuNu`.

**Note:** `data/`, `embeddings*/`, `results/`, and model weights (`*.pt`) are all in `.gitignore`.

## Pretrained Weights

Download the Sophon pretrained checkpoint from HuggingFace:

```sh
mkdir -p models/JetClassII_Sophon
curl -fSL -o models/JetClassII_Sophon/model.pt \
  "https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt"
```

## Pipeline

### Step 1: Extract Embeddings

`inference_all_classes.py` reads ROOT files, computes the 17 derived Sophon input features, runs each jet through the model, and saves 128-D embeddings. It also evaluates the raw Sophon baseline (zero-shot accuracy and AUC from the model's own logit outputs).

```sh
# Pretrained Sophon — full dataset, NPY format
python3 inference_all_classes.py \
  --checkpoint models/JetClassII_Sophon/model.pt \
  --root-dir data/train_100M \
  --events-per-class 0 \
  --output-dir embeddings_pretrained_full \
  --format npy

# Local dev — small run, CSV format (default)
python3 inference_all_classes.py \
  --checkpoint models/JetClassII_Sophon/model.pt \
  --root-dir data/val_5M \
  --events-per-class 1000 \
  --output-dir embeddings_test
```

Key flags:
- `--events-per-class 0` extracts all available events (use a number to cap per class)
- `--format npy` saves as float16 NPY (memory-mapped loading, ~10x smaller than CSV)
- `--skip-existing` skips classes that already have output files

Output: one file per class (`{ClassName}_embeddings.npy` or `{ClassName}_inference_with_embedding.csv`) plus `raw_sophon_baseline.json` with zero-shot metrics.

### Step 2: Train MLP Classifiers

`scripts/train_mlp.py` trains an MLP head on frozen embeddings with configurable architecture, early stopping, and automatic output of loss curves, ROC plots, training history, and model weights.

```sh
python3 scripts/train_mlp.py \
  --emb-dir embeddings_pretrained_full/ \
  --out-dir results/mlp_medium/ \
  --hidden-layers 256,128,64 \
  --epochs 50 \
  --batch-size 8192 \
  --patience 10
```

Key flags:
- `--hidden-layers` comma-separated layer sizes (e.g., `64` for tiny, `1024,512,256,128` for xlarge)
- `--patience N` enables early stopping (stops if val loss doesn't improve for N epochs, 0 = disabled)
- `--per-class-cap N` limits samples per class

Output per run:
- `train_results.csv` — test accuracy, AUC, per-class AUCs, timing, parameter count
- `training_history.csv` — per-epoch metrics
- `loss_curves.png` — train/val loss + val accuracy/AUC plots
- `roc_mlp_*.png` — per-class ROC curves
- `mlp_*.pt` — saved model weights + scaler + metadata

### Step 3: Run Sweeps

`scripts/run_sweep.py` orchestrates multiple training runs.

**Architecture sweep** — 7 MLP architectures at fixed 1M events/class:
```sh
python3 scripts/run_sweep.py --sweep arch \
  --emb-dir embeddings_pretrained_full/ \
  --out-dir results/arch_sweep/ \
  --per-class-cap 1000000 \
  --epochs 50 \
  --batch-size 8192
```

Architectures: tiny `[64]`, small `[128,64]`, medium `[256,128,64]`, large `[512,256,128,64]`, xlarge `[1024,512,256,128]`, xxlarge `[1024,512,256,128,64]`, wide `[2048,1024,512]`.

**Dataset size sweep** — medium MLP at 5 dataset sizes with early stopping:
```sh
python3 scripts/run_sweep.py --sweep size \
  --emb-dir embeddings_pretrained_full/ \
  --out-dir results/size_sweep/ \
  --hidden-layers 256,128,64 \
  --epochs 50 \
  --batch-size 8192 \
  --patience 10
```

Sizes per class: 10K, 100K, 1M, 10M (100K to 100M total events).

## Running on Nautilus (Kubernetes)

The pipeline is split into 3 jobs for parallel execution by multiple team members. All jobs run in the `cms-ml` namespace using the `transfer-learning-vol` PVC.

**Before submitting:** replace `YOUR_INITIALS` in each job YAML with your initials.

### Job 1: Inference (everyone runs this)

Extracts all 100M embeddings from `train_100M` as NPY and computes the raw Sophon baseline.

```sh
kubectl apply -f k8s/job-1-inference.yaml
kubectl logs -f job/sophon-inference-YOUR_INITIALS -n cms-ml
```

Estimated time: ~11 hours. Output: `/data/embeddings_pretrained_full/`

### Job 2: Architecture Sweep (1 person, after Job 1)

Trains 7 MLP architectures at 1M events/class.

```sh
kubectl apply -f k8s/job-2-arch-sweep.yaml
kubectl logs -f job/sophon-arch-sweep-YOUR_INITIALS -n cms-ml
```

Estimated time: ~1-2 hours. Output: `/data/results/arch_sweep/`

### Job 3: Dataset Size Sweep (1 person, after Job 1)

Trains medium MLP `[256,128,64]` at 5 dataset sizes with early stopping (patience=10).

```sh
kubectl apply -f k8s/job-3-size-sweep.yaml
kubectl logs -f job/sophon-size-sweep-YOUR_INITIALS -n cms-ml
```

Estimated time: ~2-4 hours. Output: `/data/results/size_sweep/`

## Project Structure

```
inference_all_classes.py    # Sophon inference + embedding extraction + raw baseline
scripts/
  train_mlp.py              # MLP training with early stopping, loss curves, weight saving
  run_sweep.py              # Orchestrates arch and size sweeps
  plot_results.py           # Plotting utilities
configs/
  arch_sweep.yaml           # Architecture sweep configuration
  size_sweep.yaml           # Dataset size sweep configuration
k8s/
  job-1-inference.yaml      # K8s job: inference (all team members)
  job-2-arch-sweep.yaml     # K8s job: architecture sweep
  job-3-size-sweep.yaml     # K8s job: dataset size sweep
networks/
  example_ParticleTransformer_sophon.py  # Sophon model definition
```

## References

- [Sophon](https://github.com/jet-universe/sophon) — Pretrained ParticleTransformer for jet physics
- [JetClass dataset](https://zenodo.org/records/6619768) — Jet classification benchmark
- [weaver-core](https://github.com/hqucms/weaver-core) — ML framework for HEP
- [Sophon pretrained weights](https://huggingface.co/jet-universe/sophon) — HuggingFace model hub
