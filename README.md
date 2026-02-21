# transferlearningsophon

## Transfer Learning Project

This README page aims to be an introduction for the ongoing transfer learning project with applications to particle physics. The project in it of itself is aimed at advancing and observing the machine learning applications to the world of particle physics, and specifically, to the task of jet-tagging. As described in the original repository for Sophon, _"...the model Sophon (Sophon (Signature-Oriented Pre-training for Heavy-resonance ObservatioN) is a method proposed for developing foundation AI models tailored for future usage in LHC experimental analyses..."_ More specifically, the Sophon is a deep learning framework developed with the goal of better classifying jets—AKA, collimated sprays of particles produced in high-energy collisions at places like the LHC (Large Hadron Collider)—using both particle-level and jet-level features.

The bigger and more universal goal, however, is to explore representation learning in jet physics, focusing on how neural network embeddings capture physical information across different datasets and simulation domains. By doing all of this, we are aiming for the following, overarching goal: Evaluate transfer learning potential across deep learning models and jet types (Sophon vs. ParT & Higgs, top, QCD, etc.)

This README covers the full pipeline: running Sophon inference on the JetClass dataset to extract embeddings, then training lightweight classifiers (MLP heads and linear probes) on those embeddings to evaluate transfer quality.

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

Download the JetClass validation set (~5M events) from https://zenodo.org/records/6619768 -> "JetClass_Pythia_val_5M.tar" -> Download & Extract.

Place the `.root` files so they live at `data/val_5M/`. For example:
```sh
mkdir -p data
cd data
wget "https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1" -O JetClass_Pythia_val_5M.tar
tar -xf JetClass_Pythia_val_5M.tar
cd ..
```

You should see files like `data/val_5M/HToBB_120.root`, `data/val_5M/HToCC_120.root`, etc.

**Note:** `data/`, `embeddings/`, `results/`, and model weights (`*.pt`) are all in `.gitignore` — you must download/generate these yourself.

## Pretrained Weights

Download the Sophon pretrained checkpoint from HuggingFace:
```sh
mkdir -p models/JetClassII_Sophon
curl -fSL -o models/JetClassII_Sophon/model.pt \
  "https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt"
```

## Running the Pipeline

### Step 1: Extract Embeddings

The inference script reads ROOT files, computes the 17 derived Sophon input features (matching the [official preprocessing](https://github.com/jet-universe/sophon/blob/main/data/JetClassII/JetClassII_full.yaml)), runs each jet through the model, and writes a CSV per class with 128-D embeddings, logits, and kinematic features.

**Pretrained Sophon:**
```sh
python inference_all_classes.py --checkpoint models/JetClassII_Sophon/model.pt
mv embeddings embeddings_pretrained
```

**Random-init (no pretraining, for comparison):**
```sh
python inference_all_classes.py
mv embeddings embeddings_random
```

### Step 2: Train Classifiers on Embeddings

**MLP head** (nonlinear classifier on frozen embeddings):
```sh
python scripts/train_mlp.py \
  --emb-dir embeddings_pretrained/ \
  --out-dir results/mlp/ \
  --hidden-layers 256,128,64 \
  --epochs 25
```

**Linear probe baseline** (logistic regression on frozen embeddings):
```sh
python scripts/evaluate_baseline.py \
  --emb-dir embeddings_pretrained/ \
  --out-dir results/baseline/
```

### Step 3: Run Sweeps

Architecture sweep (varies MLP size from tiny to xlarge):
```sh
python scripts/run_sweep.py --sweep arch \
  --emb-dir embeddings_pretrained/ \
  --out-dir results/pretrained/arch_sweep/
```

Dataset size sweep (varies samples per class from 1K to 100K, also runs baseline):
```sh
python scripts/run_sweep.py --sweep size \
  --emb-dir embeddings_pretrained/ \
  --out-dir results/pretrained/size_sweep/
```

### Step 4: Generate Plots

```sh
python scripts/plot_results.py --plot-type comparison \
  --mlp-results results/pretrained/arch_sweep/medium/train_results.csv \
  --baseline-results results/pretrained/size_sweep/baseline/baseline_results.csv \
  --out-dir results/comparison/
```

## Running on Nautilus (Kubernetes)

The full pipeline (pretrained + random-init inference, all sweeps, comparison plots) can be run as a single GPU job:
```sh
kubectl apply -f k8s/job-inference-gpu.yaml
```
This clones the repo, downloads weights, symlinks data from the PVC (`transfer-learning-vol`), runs everything, and saves results back to the PVC.

## References

- [Sophon](https://github.com/jet-universe/sophon) — Pretrained ParticleTransformer for jet physics
- [JetClass dataset](https://zenodo.org/records/6619768) — Jet classification benchmark
- [weaver-core](https://github.com/hqucms/weaver-core) — ML framework for HEP
- [Sophon pretrained weights](https://huggingface.co/jet-universe/sophon) — HuggingFace model hub
