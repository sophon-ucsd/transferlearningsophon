import os
import gc
import sys
import csv
import torch
import uproot
import numpy as np
from tqdm import tqdm
from math import cos, sin, sinh
import argparse

sys.path.append(".")
from networks.example_ParticleTransformer_sophon import get_model

# particle and scalar feature keys
particle_keys = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
    'part_dzval', 'part_dzerr', 'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon'
]

scalar_keys = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
    'label_Tbqq', 'label_Tbl', 'jet_pt', 'jet_eta', 'jet_phi',
    'jet_energy', 'jet_nparticles', 'jet_sdmass', 'jet_tau1',
    'jet_tau2', 'jet_tau3', 'jet_tau4', 'aux_genpart_eta',
    'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt',
    'aux_truth_match'
]

pf_keys = particle_keys + scalar_keys

root_dir = "./data/JetClass/val_5M"
root_files = [f for f in os.listdir(root_dir) if f.endswith('.root')]
for f in root_files:
    print(f)

# dummy config for model
class DummyDataConfig:
    input_dicts = {"pf_features": list(range(37))}
    input_names = ["pf_points"]
    input_shapes = {"pf_points": (128, 37)}
    label_names = ["label"]
    num_classes = 10

data_config = DummyDataConfig()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

batch_size = 256

print(f"Device: {device.type} | fp16: True | batch_size: {batch_size}")
model, _ = get_model(data_config, num_classes=data_config.num_classes, export_embed=True)
model.eval().to(device)

output_csv_path = "val_5M_inference_with_probs_and_embedding.csv"

with open(output_csv_path, mode="w", newline="", encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)

    base_header = (
        ["file", "event_index"] +
        ["truth_label", "label_name",
         "jet_sdmass", "jet_mass", "jet_pt", "jet_eta", "jet_phi"]
    )
    prob_header = [f"prob_{i}" for i in range(10)]
    emb_header = [f"emb_{j}" for j in range(128)]
    writer.writerow(base_header + prob_header + emb_header)

    for file_name in root_files:
        print(f"\nRunning inference on: {file_name}")
        file_path = os.path.join(root_dir, file_name)
        with uproot.open(file_path) as f:
            tree = f["tree"]
            arrays = tree.arrays(pf_keys, library="np")

        max_part = 128
        total_events = len(arrays["part_px"])

        batch = []
        meta = []  # (event_index, truth_label, label_name, jet_sdmass, jet_mass, pt, eta, phi)
        batch_counter = 0

        for i in tqdm(range(total_events), desc=f"{file_name}"):
            try:
                n_part = arrays["part_px"][i].shape[0]
                if n_part > max_part:
                    continue

                particle_feats = [arrays[k][i] for k in particle_keys]
                scalar_feats = [np.full(n_part, arrays[k][i]) for k in scalar_keys]
                pf_features = np.stack(particle_feats + scalar_feats, axis=1).astype(np.float32)

                padded = np.zeros((max_part, pf_features.shape[1]), dtype=np.float32)
                padded[:n_part, :] = pf_features
                batch.append(padded)

                # truth labels
                label_array = np.array([arrays[k][i] for k in [
                    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
                    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
                    'label_Tbqq', 'label_Tbl'
                ]])
                truth_label = int(np.argmax(label_array))
                label_names = ["QCD","Hbb","Hcc","Hgg","H4q","Hqql","Zqq","Wqq","Tbqq","Tbl"]
                label_name = label_names[truth_label]

                # kinematics
                jet_sdmass = float(arrays["jet_sdmass"][i])
                pt  = float(arrays["jet_pt"][i])
                eta = float(arrays["jet_eta"][i])
                phi = float(arrays["jet_phi"][i])
                E   = float(arrays["jet_energy"][i])
                px = pt * cos(phi)
                py = pt * sin(phi)
                pz = pt * sinh(eta)
                p2 = px*px + py*py + pz*pz
                m2 = max(E*E - p2, 0.0)
                jet_mass = float(np.sqrt(m2))

                meta.append((i, truth_label, label_name, jet_sdmass, jet_mass, pt, eta, phi))

                # run batch if full or last
                if len(batch) >= batch_size or i == (total_events - 1):
                    jet_tensor = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)  # [B, 128, 37]
                    lorentz_vectors = jet_tensor[:, :, 0:4].transpose(1, 2)  # [B, 4, 128]
                    features = jet_tensor[:, :, 4:].transpose(1, 2)         # [B, 33, 128]
                    mask = (jet_tensor.sum(dim=2) != 0).unsqueeze(1)        # [B, 1, 128]
                    points = None

                    with torch.no_grad():
                        if device.type == "cuda":
                            from torch.cuda.amp import autocast
                            with autocast():
                                out = model(points, features, lorentz_vectors, mask)
                        else:
                            out = model(points, features, lorentz_vectors, mask)

                    if isinstance(out, tuple):
                        logits, embedding = out
                        logits = logits.detach().cpu().numpy()
                        embedding = embedding.detach().cpu().numpy()
                    else:
                        out_np = out.detach().cpu().numpy()  # [B, 138]
                        logits = out_np[:, :10]
                        embedding = out_np[:, 10:]

                    lmax = np.max(logits, axis=1, keepdims=True)
                    exp_logits = np.exp(logits - lmax)
                    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                    for (evt_idx, tl, lname, sdm, jm, ptt, et, ph), pr, emb in zip(meta, probs, embedding):
                        row = [file_name, evt_idx, tl, lname, sdm, jm, ptt, et, ph] + list(pr) + list(emb)
                        writer.writerow(row)

                    # reset batch buffers
                    batch.clear()
                    meta.clear()

            except Exception as e:
                print(f"Error in event {i}: {e}")
                continue

        # Clean up memory after each file
        del arrays
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

print(f"Saved CSV data to {output_csv_path}")
