# Cluster Guide — Nautilus K8s for Transfer Learning

## Safety Rules

- **NEVER** delete, modify, or interact with any pod/job that doesn't have your name in it
- The `cms-ml` namespace is shared with many researchers
- Always grep for your name when listing or deleting: `kubectl get pods -n cms-ml | grep raunav`
- When cleaning up, only target your specific job names
- Completed/errored pods don't use compute but stick around — delete yours periodically

## PVC Layout

All data lives on the shared PVC `transfer-learning-vol`, mounted at `/data/` inside any pod/job.

```
/data/
├── JetClass/Pythia/
│   ├── train_100M/          # 1000 ROOT files, 100K jets each, 10 classes
│   ├── val_5M/              # 50 ROOT files (5 per class)
│   └── test_20M/            # 200 ROOT files (20 per class)
├── embeddings_pretrained_full_02_174909_100M/   # Sophon embeddings (100M, .npy)
├── embeddings_val_5M/       # Sophon embeddings for val (5M, .npy)
├── embeddings_test_20M/     # Sophon embeddings for test (20M, .npy)
├── results/
│   ├── frozen_base/         # Frozen MLP sweep results
│   ├── from_scratch/        # From-scratch sweep results
│   ├── full_ft/             # Full fine-tune sweep results
│   └── partial_ft/          # Partial fine-tune sweep results
└── transferlearningsophon/  # Old repo clone (ignore)
```

Each embedding file: `{ClassName}_embeddings.npy`, shape `(N, 128)`, dtype float16.

Each results directory: `{strategy}_{train_size}_{seed}/results.json`.

## Common Commands

### Checking your pods/jobs

```bash
# List only your pods
kubectl get pods -n cms-ml | grep <your-name>

# Check job logs (latest output)
kubectl logs job/<job-name> -n cms-ml --tail=20

# Follow logs in real time
kubectl logs -f job/<job-name> -n cms-ml

# Check which GPU you got
kubectl exec <pod-name> -n cms-ml -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check what node your pod is on
kubectl get pods -n cms-ml -l job-name=<job-name> -o wide
```

### Launching jobs

```bash
# Submit a job
kubectl apply -f k8s/<job-file>.yaml

# IMPORTANT: Change the job name to include YOUR name before submitting
# e.g. name: full-ft-sweep-emmet (not raunav)
```

### Inspecting the PVC

```bash
# Create a lightweight pod for browsing
kubectl apply -f k8s/pod.yaml
kubectl wait --for=condition=Ready pod/raunav-sophon-pod -n cms-ml --timeout=300s
kubectl exec -it raunav-sophon-pod -n cms-ml -- /bin/bash

# Inside the pod:
ls /data/
ls /data/results/frozen_base/

# Delete the pod when done
kubectl delete pod raunav-sophon-pod -n cms-ml
```

### Cleaning up

```bash
# Delete a specific job (and its pods)
kubectl delete job <job-name> -n cms-ml

# Check for your stale completed/errored pods
kubectl get pods -n cms-ml | grep <your-name> | grep -E "Completed|Error|OOMKilled"

# Delete a specific stale pod
kubectl delete pod <pod-name> -n cms-ml
```

### Downloading results locally

```bash
# Copy a file from PVC (need a running pod first)
kubectl cp cms-ml/<pod-name>:/data/results/frozen_base/frozen_base_10000_42/results.json ./results.json

# Copy a whole directory
kubectl cp cms-ml/<pod-name>:/data/results/frozen_base/ ./frozen_base_results/
```

## Running Sweeps

### Frozen MLP sweep (on precomputed embeddings, no GPU needed but faster with one)

```bash
kubectl apply -f k8s/job-frozen-base-large.yaml
```

Uses `scripts/run_frozen_base.py` which loads embeddings from `.npy` files and trains a small MLP.

### Full model sweeps (need GPU)

```bash
# Change job name to yours first!
kubectl apply -f k8s/job-from-scratch-sweep.yaml   # Random init baseline
kubectl apply -f k8s/job-full-ft-sweep.yaml         # Full fine-tune
kubectl apply -f k8s/job-partial-ft-sweep.yaml      # Partial fine-tune (freeze first 4 layers)
```

These use `scripts/train_sophon.py` which loads ROOT files, builds the ParticleTransformer model, and trains end-to-end.

### Collecting results

```bash
# From a pod with PVC access:
python3 scripts/collect_results.py /data/results/frozen_base/
python3 scripts/collect_results.py /data/results/  # all strategies

# Generates results_all.csv and results_summary.csv
```

### Plotting

```bash
python3 scripts/plot_scaling_curves.py /data/results/results_summary.csv
python3 scripts/hernandez_analysis.py /data/results/results_summary.csv
```

## Hyperparameters per Strategy

| Strategy | LR (head) | LR (backbone) | Frozen layers | Batch size | Max epochs | Patience |
|---|---|---|---|---|---|---|
| frozen | 1e-3 | N/A | all | 4096-16384 | 200 | 10 |
| from_scratch | 1e-3 | N/A | none | 256 | 100 | 10 |
| full_ft | 1e-3 | 1e-4 | none | 256 | 100 | 10 |
| partial_ft | 5e-4 | N/A | first 4 of 8 | 256 | 100 | 10 |

## Dataset Sizes for Sweeps

9 sizes × 3 seeds = 27 runs per strategy:

`10K, 30K, 100K, 300K, 1M, 3M, 10M, 30M, 100M` jets

Seeds: `42, 123, 456`

All evaluate on official val_5M and test_20M splits.

## Troubleshooting

**OOMKilled**: Pod exceeded memory limit. Check:
- Are you loading the full 20M test set into RAM? Use lazy loading.
- For 100M training, keep embeddings as float16 — don't convert to float32 in bulk.
- Request more memory in the YAML (64Gi for embedding jobs, 32Gi for ROOT file jobs).

**ContainerCreating for 10+ min**: Pulling the Docker image. Normal on first run on a new node.

**Job stays Pending**: No resources available. Check `kubectl describe pod <pod-name> -n cms-ml` for events. The cluster is shared — wait or reduce resource requests.

**Auth expired**: `kubectl` returns OIDC errors. Re-authenticate to the cluster.

**Wrong results path**: Hydra changes cwd. Use `logging.results_dir=/data/results` in overrides to save directly to PVC.
