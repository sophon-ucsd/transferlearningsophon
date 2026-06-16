# Transfer learning with Sophon on JetClass-I

Scaling-law comparison of three adaptation strategies (frozen + MLP, partial fine-tune, full fine-tune) for the [Sophon](https://github.com/jet-universe/sophon) foundation model on the 10-class [JetClass-I](https://zenodo.org/records/6619768) benchmark.

## Setup

```sh
conda create -n sophon python=3.10 -y && conda activate sophon
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .

mkdir -p models/JetClassII_Sophon
curl -fSL -o models/JetClassII_Sophon/model.pt \
  "https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/model.pt"
```

## 1. Extract embeddings

```sh
python3 inference_all_classes.py \
  --checkpoint models/JetClassII_Sophon/model.pt \
  --root-dir data/train_100M \
  --events-per-class 0 \
  --output-dir embeddings_pretrained_full --format npy
```

## 2. Train

```sh
# Frozen + MLP sweep
python3 scripts/train_frozen_sweep.py \
  --emb-dir embeddings_pretrained_full \
  --output-dir results/frozen_base \
  --sizes 10000,30000,100000,300000,1000000,3000000,10000000,30000000,100000000 \
  --seeds 42,123,456

# Partial / full fine-tune sweep
python3 scripts/train_finetune_sweep.py \
  --train-dir /data/features/train_100M \
  --val-dir   /data/features/val_5M \
  --test-dir  /data/features/test_20M \
  --strategy partial_ft \
  --checkpoint models/JetClassII_Sophon/model.pt \
  --sizes 10000,30000,100000,300000,1000000,3000000,10000000 \
  --seeds 42,123,456 \
  --output-dir results/partial_ft \
  --materialize-train --skip-existing

python3 scripts/collect_results.py   # writes results/sweep_results.csv
```

## 3. Figures

```sh
python3 scripts/plot_scaling.py
python3 scripts/plot_umap.py --pretrained-dir embeddings_pretrained_full --finetuned-dir <ft_embeddings>
python3 scripts/plot_roc_3strategies.py \
  --frozen-mlp   results/frozen_base/frozen_base_10000000_42/best_model.pt \
  --partial-ckpt results/partial_ft/partial_ft_10000000_42/best_model.pt \
  --full-ckpt    results/full_ft/full_ft_10000000_42/best_model.pt \
  --pretrained   models/JetClassII_Sophon/model.pt \
  --features-dir /data/features/test_20M \
  --embeddings-dir embeddings_pretrained_full
```

## Layout

```
inference_all_classes.py        Step-1 embedding extraction
src/{data,models,utils}/        package code
plots/style.py                  figure style
scripts/                        training sweeps + figure renderers
k8s/                            kubernetes job templates
networks/                       reference Sophon ParT definition
results/sweep_results.csv       9 sizes × 3 strategies × seeds
```

