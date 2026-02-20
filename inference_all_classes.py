import os
import sys
import csv
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

TARGET_EVENTS_PER_CLASS = 100_000
MAX_PART = 128
STEP_SIZE = 5000
TREE_NAME = "tree"
ROOT_DIR = "val_5M"
OUTPUT_DIR = "embeddings"
SKIP_IF_EXISTS = False

# ---------------------------------------------------------------------------
# The 17 derived Sophon input features (matches pretrained model's input_dim)
# ---------------------------------------------------------------------------
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

JET_CLASSES = {
    "HToBB": {
        "files": ["HToBB_120.root", "HToBB_121.root", "HToBB_122.root", "HToBB_123.root", "HToBB_124.root"],
        "output": "HToBB_inference_with_embedding.csv"
    },
    "HToCC": {
        "files": ["HToCC_120.root", "HToCC_121.root", "HToCC_122.root", "HToCC_123.root", "HToCC_124.root"],
        "output": "HToCC_inference_with_embedding.csv"
    },
    "HToGG": {
        "files": ["HToGG_120.root", "HToGG_121.root", "HToGG_122.root", "HToGG_123.root", "HToGG_124.root"],
        "output": "HToGG_inference_with_embedding.csv"
    },
    "HToWW4Q": {
        "files": ["HToWW4Q_120.root", "HToWW4Q_121.root", "HToWW4Q_122.root", "HToWW4Q_123.root", "HToWW4Q_124.root"],
        "output": "HToWW4Q_inference_with_embedding.csv"
    },
    "HToWW2Q1L": {
        "files": ["HToWW2Q1L_120.root", "HToWW2Q1L_121.root", "HToWW2Q1L_122.root", "HToWW2Q1L_123.root", "HToWW2Q1L_124.root"],
        "output": "HToWW2Q1L_inference_with_embedding.csv"
    },
    "ZToQQ": {
        "files": ["ZToQQ_120.root", "ZToQQ_121.root", "ZToQQ_122.root", "ZToQQ_123.root", "ZToQQ_124.root"],
        "output": "ZToQQ_inference_with_embedding.csv"
    },
    "WToQQ": {
        "files": ["WToQQ_120.root", "WToQQ_121.root", "WToQQ_122.root", "WToQQ_123.root", "WToQQ_124.root"],
        "output": "WToQQ_inference_with_embedding.csv"
    },
    "TTBar": {
        "files": ["TTBar_120.root", "TTBar_121.root", "TTBar_122.root", "TTBar_123.root", "TTBar_124.root"],
        "output": "TTBar_inference_with_embedding.csv"
    },
    "TTBarLep": {
        "files": ["TTBarLep_120.root", "TTBarLep_121.root", "TTBarLep_122.root", "TTBarLep_123.root", "TTBarLep_124.root"],
        "output": "TTBarLep_inference_with_embedding.csv"
    },
    "ZJetsToNuNu": {
        "files": ["ZJetsToNuNu_120.root", "ZJetsToNuNu_121.root", "ZJetsToNuNu_122.root", "ZJetsToNuNu_123.root", "ZJetsToNuNu_124.root"],
        "output": "ZToNuNu_inference_with_embedding.csv"
    },
}

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
    scale = max(jet_pt_val * 500.0, eps)

    # Scaled kinematics (official: part_*_scale = part_* / (jet_pt * 500))
    pt = np.sqrt(px ** 2 + py ** 2)
    pt_scale = pt / scale
    energy_scale = energy / scale

    # Logarithmic features with normalization (subtract, multiply, clip)
    pt_scale_log = _norm(np.log(np.clip(pt_scale, eps, None)),
                         subtract=1.7, multiply=0.7)
    e_scale_log  = _norm(np.log(np.clip(energy_scale, eps, None)),
                         subtract=2.0, multiply=0.7)
    logptrel     = _norm(np.log(np.clip(pt / max(jet_pt_val, eps), eps, None)),
                         subtract=-4.7, multiply=0.7)
    logerel      = _norm(np.log(np.clip(energy / max(jet_energy_val, eps), eps, None)),
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

    # Scaled Lorentz 4-vectors: part_*_scale = part_* / (jet_pt * 500)
    jet_pt_val = float(arrays["jet_pt"][i])
    scale = max(jet_pt_val * 500.0, 1e-20)
    lv = np.stack([
        get("part_px") / scale,
        get("part_py") / scale,
        get("part_pz") / scale,
        get("part_energy") / scale,
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

def process_class(class_name, class_info, model, device):
    output_path = os.path.join(OUTPUT_DIR, class_info["output"])

    if SKIP_IF_EXISTS and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"Skipping {class_name} (exists): {output_path}")
        return 0

    print(f"\nProcessing {class_name}")

    root_files = class_info["files"]
    total_written = 0
    wrote_header = False
    target = TARGET_EVENTS_PER_CLASS

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        paths = [f"{os.path.join(ROOT_DIR, fn)}:{TREE_NAME}" for fn in root_files]

        it = uproot.iterate(
            paths,
            expressions=pf_keys,
            entry_step=STEP_SIZE,
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
                        logits_np = np.zeros(10, dtype=np.float32)  # fallback

                    emb = embedding.squeeze(0).detach().cpu().numpy()

                    if not wrote_header:
                        base = [
                            "source_file", "entry_index", "row_index",
                            "truth_label", "label_name",
                            "jet_sdmass", "jet_mass", "jet_pt", "jet_eta", "jet_phi",
                        ]
                        logit_cols = [f"logit_{j}" for j in range(10)]
                        emb_cols = [f"emb_{j}" for j in range(emb.shape[-1])]
                        writer.writerow(base + logit_cols + emb_cols)
                        wrote_header = True

                    truth_label, label_name = get_truth_label(arrays, i)
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

    print(f"{class_name}: saved {total_written:,} rows -> {output_path}")
    return total_written

def main():
    parser = argparse.ArgumentParser(description="JetClass Sophon inference & embedding extraction")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained model.pt weights file. "
                             "If omitted, runs with random-init weights (baseline).")
    args = parser.parse_args()

    data_config = DummyDataConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on {device}...")
    model, _ = get_model(data_config, num_classes=data_config.num_classes, export_embed=True)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=device)
        # Handle both raw state_dict and wrapped checkpoint formats
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {ckpt_path}")
    else:
        print("No --checkpoint provided; using random-init weights")

    model.eval().to(device)
    print("Model ready")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("JetClass embedding generator")
    print(f"classes: {len(JET_CLASSES)}")
    print(f"events per class: {TARGET_EVENTS_PER_CLASS if TARGET_EVENTS_PER_CLASS else 'ALL'}")
    print(f"root dir: {ROOT_DIR}")
    print(f"out dir: {OUTPUT_DIR}")
    print(f"skip existing: {SKIP_IF_EXISTS}")

    total_events = 0
    for class_name, class_info in JET_CLASSES.items():
        total_events += process_class(class_name, class_info, model, device)

    print(f"Done. Total events written: {total_events:,}")
    print(f"Outputs in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
