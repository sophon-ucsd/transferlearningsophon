import os
import sys
import csv
import json
import math
import argparse
import torch
import uproot
import numpy as np
from tqdm import tqdm
from math import cos, sin, sinh
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from networks.example_ParticleTransformer_sophon import get_model

MAX_PART = 128
STEP_SIZE = 5000
TREE_NAME = "tree"

# The 17 derived Sophon input features (matches pretrained model's input_dim)
SOPHON_FEATURE_NAMES = [
    "part_pt_scale_log",
    "part_e_scale_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0",
    "part_d0err",
    "part_dz",
    "part_dzerr",
    "part_deta",
    "part_dphi",
]
NUM_SOPHON_FEATURES = len(SOPHON_FEATURE_NAMES)  # 17

def discover_classes(root_dir):
    """Auto-discover jet classes from .root files in the given directory."""
    root_path = Path(root_dir)
    if not root_path.is_dir():
        sys.exit(f"Root directory not found: {root_dir}")

    files_by_class = {}
    for f in sorted(root_path.glob("*.root")):
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            print(f"Warning: skipping unrecognized file {f.name}")
            continue
        files_by_class.setdefault(parts[0], []).append(f.name)

    for cls in files_by_class:
        files_by_class[cls].sort(key=lambda x: int(x.rsplit("_", 1)[1].replace(".root", "")))

    print(f"Discovered {len(files_by_class)} classes in {root_dir}:")
    for cls, flist in sorted(files_by_class.items()):
        print(f"  {cls}: {len(flist)} files")

    return files_by_class

particle_keys = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi", "part_d0val", "part_d0err",
    "part_dzval", "part_dzerr", "part_charge",
    "part_isChargedHadron", "part_isNeutralHadron",
    "part_isPhoton", "part_isElectron", "part_isMuon",
]

scalar_keys_for_model = [
    "jet_pt", "jet_eta", "jet_phi",
    "jet_energy", "jet_nparticles", "jet_sdmass",
    "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4",
]

label_keys = [
    "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
    "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
    "label_Tbqq", "label_Tbl",
]

pf_keys = particle_keys + label_keys + scalar_keys_for_model

label_names = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

class DummyDataConfig:
    input_dicts = {"pf_features": list(range(NUM_SOPHON_FEATURES))}
    input_names = ["pf_points"]
    input_shapes = {"pf_points": (MAX_PART, NUM_SOPHON_FEATURES)}
    label_names = ["label"]
    num_classes = 10

def _norm(x, subtract, multiply, clip_lo=-5.0, clip_hi=5.0):
    """Apply (x - subtract) * multiply then clip to [clip_lo, clip_hi]."""
    return np.clip((x - subtract) * multiply, clip_lo, clip_hi)


def compute_sophon_features(arrays, i, keep_idx=None):
    """Derive the 17 Sophon input features from raw particle arrays.

    Preprocessing matches the official JetClassII_full.yaml config exactly:
      - momentum/energy logs are computed from jet_pt*500-scaled values
      - 5 kinematic features are shift/scale normalised then clipped to [-5,5]
      - d0 / dz use tanh transform;  d0err / dzerr are clipped to [0,1]

    Returns an (n_part, 17) float32 array.
    """
    get = (lambda k: arrays[k][i][keep_idx]) if keep_idx is not None else (lambda k: arrays[k][i])

    px = get("part_px")
    py = get("part_py")
    energy = get("part_energy")

    jet_pt_val = float(arrays["jet_pt"][i])
    jet_energy_val = float(arrays["jet_energy"][i])

    eps = 1e-20
    jet_pt_safe = max(jet_pt_val, eps)
    jet_energy_safe = max(jet_energy_val, eps)

    # Scaled kinematics (official: part_*_scale = part_* * 500 / jet_pt)
    pt = np.sqrt(px ** 2 + py ** 2)
    pt_scale = pt * 500.0 / jet_pt_safe
    energy_scale = energy * 500.0 / jet_pt_safe

    # Logarithmic features with normalization (subtract, multiply, clip)
    pt_scale_log = _norm(np.log(np.clip(pt_scale, eps, None)),
                         subtract=1.7, multiply=0.7)
    e_scale_log  = _norm(np.log(np.clip(energy_scale, eps, None)),
                         subtract=2.0, multiply=0.7)
    logptrel     = _norm(np.log(np.clip(pt / jet_pt_safe, eps, None)),
                         subtract=-4.7, multiply=0.7)
    logerel      = _norm(np.log(np.clip(energy / jet_energy_safe, eps, None)),
                         subtract=-4.7, multiply=0.7)

    deta = get("part_deta")
    dphi = get("part_dphi")
    deltaR = _norm(np.sqrt(deta ** 2 + dphi ** 2),
                   subtract=0.2, multiply=4.0)

    # Impact parameters: tanh transform for d0/dz, clip [0,1] for errors
    d0    = np.tanh(get("part_d0val"))
    d0err = np.clip(get("part_d0err"), 0.0, 1.0)
    dz    = np.tanh(get("part_dzval"))
    dzerr = np.clip(get("part_dzerr"), 0.0, 1.0)

    feats = np.stack([
        pt_scale_log,
        e_scale_log,
        logptrel,
        logerel,
        deltaR,
        get("part_charge"),
        get("part_isChargedHadron"),
        get("part_isNeutralHadron"),
        get("part_isPhoton"),
        get("part_isElectron"),
        get("part_isMuon"),
        d0,
        d0err,
        dz,
        dzerr,
        deta,
        dphi,
    ], axis=1).astype(np.float32)

    return feats


def build_pf_tensor(arrays, i, device):
    """Build (points, features, lorentz_vectors, mask) for one jet."""
    n_part = arrays["part_px"][i].shape[0]

    if n_part > MAX_PART:
        px = arrays["part_px"][i]
        py = arrays["part_py"][i]
        pt = np.sqrt(px * px + py * py)
        keep_idx = np.argsort(pt)[::-1][:MAX_PART]
        n_part = MAX_PART
    else:
        keep_idx = None

    get = (lambda k: arrays[k][i][keep_idx]) if keep_idx is not None else (lambda k: arrays[k][i])

    # Scaled Lorentz 4-vectors: part_*_scale = part_* * 500 / jet_pt
    jet_pt_val = float(arrays["jet_pt"][i])
    jet_pt_safe = max(jet_pt_val, 1e-20)
    lv = np.stack([
        get("part_px") * 500.0 / jet_pt_safe,
        get("part_py") * 500.0 / jet_pt_safe,
        get("part_pz") * 500.0 / jet_pt_safe,
        get("part_energy") * 500.0 / jet_pt_safe,
    ], axis=1).astype(np.float32)

    # 17 derived Sophon features — shape (n_part, 17)
    sophon_feats = compute_sophon_features(arrays, i, keep_idx)

    # Pad to MAX_PART
    lv_padded = np.zeros((MAX_PART, 4), dtype=np.float32)
    lv_padded[:n_part] = lv

    feat_padded = np.zeros((MAX_PART, NUM_SOPHON_FEATURES), dtype=np.float32)
    feat_padded[:n_part] = sophon_feats

    # Tensors — model expects (batch, channels, particles)
    lv_tensor = torch.tensor(lv_padded).unsqueeze(0).transpose(1, 2).to(device)
    feat_tensor = torch.tensor(feat_padded).unsqueeze(0).transpose(1, 2).to(device)
    mask = torch.tensor(lv_padded[:, 3] != 0, dtype=torch.bool).unsqueeze(0).unsqueeze(1).to(device)

    return None, feat_tensor, lv_tensor, mask

def get_truth_label(arrays, i):
    labs = np.array([arrays[k][i] for k in label_keys])
    y = int(np.argmax(labs))
    return y, label_names[y]

def jet_masses(arrays, i):
    jet_sdmass = float(arrays["jet_sdmass"][i])
    pt = float(arrays["jet_pt"][i])
    eta = float(arrays["jet_eta"][i])
    phi = float(arrays["jet_phi"][i])
    E = float(arrays["jet_energy"][i])
    px = pt * cos(phi)
    py = pt * sin(phi)
    pz = pt * sinh(eta)
    m2 = max(E * E - (px * px + py * py + pz * pz), 0.0)
    return jet_sdmass, math.sqrt(m2), pt, eta, phi

NPY_FLUSH_INTERVAL = 500_000  # flush embeddings to disk every 500K events


def _flush_npy(emb_list, output_path, first_flush):
    """Append embeddings to an NPY file on disk, clearing the in-memory list."""
    if not emb_list:
        return
    chunk = np.stack(emb_list)
    if first_flush or not os.path.exists(output_path):
        np.save(output_path, chunk)
    else:
        existing = np.load(output_path)
        np.save(output_path, np.concatenate([existing, chunk], axis=0))
    emb_list.clear()


def process_class(class_name, root_files, root_dir, output_dir, target_events,
                   model, device, num_classes=10, skip_existing=False, fmt="csv"):
    if fmt == "npy":
        output_path = os.path.join(output_dir, f"{class_name}_embeddings.npy")
    else:
        output_path = os.path.join(output_dir, f"{class_name}_inference_with_embedding.csv")

    if skip_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"Skipping {class_name} (exists): {output_path}")
        return 0, np.array([]), np.array([]), np.array([])

    print(f"\nProcessing {class_name}")

    total_written = 0
    wrote_header = False
    target = target_events if target_events > 0 else None
    emb_list = [] if fmt == "npy" else None
    first_flush = True

    # Accumulate raw Sophon predictions for baseline evaluation
    all_preds = []   # argmax of logits
    all_truths = []  # truth label index
    all_probs = []   # softmax probabilities (for AUC)

    csvfile = open(output_path, "w", newline="") if fmt == "csv" else None
    writer = csv.writer(csvfile) if csvfile else None

    try:
        paths = [f"{os.path.join(root_dir, fn)}:{TREE_NAME}" for fn in root_files]

        it = uproot.iterate(
            paths,
            expressions=pf_keys,
            step_size=STEP_SIZE,
            library="np",
            report=True,
        )

        for batch_idx, (arrays, report) in enumerate(it):
            batch_len = len(arrays["jet_pt"])

            source_file = os.path.basename(getattr(report, "file_path", "unknown"))
            batch_start_entry = getattr(report, "entry_start", 0)

            pbar = tqdm(range(batch_len), desc=f"{class_name} batch {batch_idx}", leave=False)

            for i in pbar:
                if target is not None and total_written >= target:
                    break

                try:
                    points, features, lorentz_vectors, mask = build_pf_tensor(arrays, i, device)

                    with torch.no_grad():
                        out = model(points, features, lorentz_vectors, mask)

                    if isinstance(out, tuple):
                        logits, embedding = out
                        logits_np = logits.squeeze(0).detach().cpu().numpy()
                    else:
                        embedding = out
                        logits_np = np.zeros(num_classes, dtype=np.float32)

                    emb = embedding.squeeze(0).detach().cpu().numpy()

                    # Track raw Sophon predictions for baseline eval
                    truth_label, label_name = get_truth_label(arrays, i)
                    pred_label = int(np.argmax(logits_np))
                    all_preds.append(pred_label)
                    all_truths.append(truth_label)
                    # Compute softmax in-place to avoid storing raw logits
                    exp_l = np.exp(logits_np - logits_np.max())
                    all_probs.append((exp_l / exp_l.sum()).astype(np.float16))

                    if fmt == "npy":
                        emb_list.append(emb.astype(np.float16))
                        # Periodically flush to disk to avoid OOM on large datasets
                        if len(emb_list) >= NPY_FLUSH_INTERVAL:
                            _flush_npy(emb_list, output_path, first_flush)
                            first_flush = False
                            print(f"  {class_name}: flushed {total_written + 1:,} embeddings to disk")
                    else:
                        if not wrote_header:
                            base = [
                                "source_file", "entry_index", "row_index",
                                "truth_label", "label_name",
                                "jet_sdmass", "jet_mass", "jet_pt", "jet_eta", "jet_phi",
                            ]
                            logit_cols = [f"logit_{j}" for j in range(logits_np.shape[0])]
                            emb_cols = [f"emb_{j}" for j in range(emb.shape[-1])]
                            writer.writerow(base + logit_cols + emb_cols)
                            wrote_header = True

                        jet_sdmass, jet_mass, pt, eta, phi = jet_masses(arrays, i)
                        entry_index = int(batch_start_entry + i)

                        row = [
                            source_file, entry_index, total_written,
                            truth_label, label_name,
                            jet_sdmass, jet_mass, pt, eta, phi,
                            *logits_np.astype(np.float32).tolist(),
                            *emb.astype(np.float32).tolist(),
                        ]
                        writer.writerow(row)

                    total_written += 1

                except Exception as e:
                    pbar.set_postfix_str(f"err: {e}")
                    continue

            if target is not None and total_written >= target:
                break

    finally:
        if csvfile:
            csvfile.close()

    # Flush any remaining embeddings
    if fmt == "npy" and emb_list:
        _flush_npy(emb_list, output_path, first_flush)

    print(f"{class_name}: saved {total_written:,} rows -> {output_path}")
    return total_written, np.array(all_preds), np.array(all_truths), np.array(all_probs)



def main():
    parser = argparse.ArgumentParser(description="JetClass Sophon inference & embedding extraction")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained model.pt weights file. "
                             "If omitted, runs with random-init weights (baseline).")
    parser.add_argument("--root-dir", type=str, default="data/val_5M",
                        help="Directory containing .root files (auto-discovers classes)")
    parser.add_argument("--events-per-class", type=int, default=100_000,
                        help="Target events per class (0 = all available)")
    parser.add_argument("--output-dir", type=str, default="embeddings",
                        help="Directory for output CSV files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip classes whose output CSV already exists")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "npy"],
                        help="Output format: csv (full, ~1.8KB/event) or npy (embeddings only, ~256B/event)")
    args = parser.parse_args()

    # --- Discover classes from root dir ---
    classes = discover_classes(args.root_dir)
    if not classes:
        sys.exit(f"No .root files found in {args.root_dir}")

    data_config = DummyDataConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint (if any) and detect num_classes ---
    num_classes = 10  # default for random-init
    checkpoint_state = None

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        raw = torch.load(str(ckpt_path), map_location=device)
        checkpoint_state = raw["model_state_dict"] if "model_state_dict" in raw else raw
        # Detect num_classes from FC layer weight shape
        for key in ["mod.fc.0.weight", "fc.0.weight"]:
            if key in checkpoint_state:
                num_classes = checkpoint_state[key].shape[0]
                print(f"Detected num_classes={num_classes} from checkpoint")
                break

    # --- Create model with correct num_classes ---
    print(f"Loading model on {device} (num_classes={num_classes})...")
    model, _ = get_model(data_config, num_classes=num_classes, export_embed=True)

    if checkpoint_state is not None:
        # Try loading; if keys lack 'mod.' prefix, remap them
        missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
        if len(unexpected) > 0 and all(not k.startswith("mod.") for k in checkpoint_state):
            remapped = {"mod." + k: v for k, v in checkpoint_state.items()}
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            print(f"Remapped checkpoint keys (added 'mod.' prefix)")
        loaded = len(checkpoint_state) - len(unexpected)
        print(f"Loaded pretrained weights from {ckpt_path} "
              f"({loaded} params loaded, {len(missing)} missing, {len(unexpected)} unexpected)")
    else:
        print("No --checkpoint provided; using random-init weights")

    model.eval().to(device)
    print("Model ready")

    os.makedirs(args.output_dir, exist_ok=True)

    epc_str = str(args.events_per_class) if args.events_per_class > 0 else "ALL"
    print("JetClass embedding generator")
    print(f"classes: {len(classes)}")
    print(f"events per class: {epc_str}")
    print(f"root dir: {args.root_dir}")
    print(f"out dir: {args.output_dir}")
    print(f"skip existing: {args.skip_existing}")
    print(f"format: {args.format}")

    total_events = 0
    all_preds = []
    all_truths = []
    all_probs = []

    for class_name, file_list in sorted(classes.items()):
        n_written, preds, truths, probs = process_class(
            class_name, file_list, args.root_dir, args.output_dir,
            args.events_per_class, model, device, num_classes,
            skip_existing=args.skip_existing, fmt=args.format,
        )
        total_events += n_written
        if len(preds) > 0:
            all_preds.append(preds)
            all_truths.append(truths)
            all_probs.append(probs)

    print(f"\nDone. Total events written: {total_events:,}")
    print(f"Outputs in: {args.output_dir}/")

    # --- Raw Sophon baseline evaluation ---
    if len(all_preds) > 0:
        preds = np.concatenate(all_preds)
        truths = np.concatenate(all_truths)
        probs = np.concatenate(all_probs).astype(np.float32)

        acc = float(np.mean(preds == truths))
        print(f"\n{'='*50}")
        print(f"RAW SOPHON BASELINE (zero-shot transfer)")
        print(f"{'='*50}")
        print(f"  Total events evaluated: {len(truths):,}")
        print(f"  Overall accuracy: {acc:.4f}")

        # Per-class accuracy
        unique_labels = sorted(set(truths))
        per_class = {}
        for lab in unique_labels:
            mask = truths == lab
            cls_acc = float(np.mean(preds[mask] == truths[mask]))
            cls_name = label_names[lab] if lab < len(label_names) else f"class_{lab}"
            per_class[cls_name] = {"accuracy": cls_acc, "count": int(mask.sum())}
            print(f"  {cls_name}: acc={cls_acc:.4f} (n={mask.sum():,})")

        # Macro AUC (one-vs-rest)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(truths, probs, multi_class="ovr", average="macro")
            print(f"  Macro AUC (OvR): {auc:.4f}")
        except Exception as e:
            auc = None
            print(f"  AUC computation failed: {e}")

        # Save baseline metrics
        baseline_results = {
            "model": "raw_sophon_pretrained",
            "total_events": len(truths),
            "accuracy": acc,
            "auc_macro_ovr": auc,
            "per_class": per_class,
        }
        metrics_path = os.path.join(args.output_dir, "raw_sophon_baseline.json")
        with open(metrics_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"\n  Baseline metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
