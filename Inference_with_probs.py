import os
import sys
import csv
import math
import torch
import uproot
import numpy as np
from tqdm import tqdm
from math import cos, sin, sinh

# Ensure the current directory is in sys.path to import local modules
sys.path.append(".")
# Import the get_model function from the local networks module
from networks.example_ParticleTransformer_sophon import get_model

# Percentage of events to process from each file (0 to 100)
PERCENTAGE = 1.0  # Process 5% of events from each file
# Defines the maximum number of particles to consider per jet
MAX_PART = 128                 
# Tells uproot how many events to read into memory at one time
STEP_SIZE = 1000  # Reduced for better memory handling
# The name of the data structure inside the ROOT files              
TREE_NAME = "tree"            

# Define the input data location 
root_dir = "data/JetClass/val_5M"
# Defines the list of ROOT files to process
root_files = [
    "HToBB_120.root",
    "HtoCC_120.root",
    "HToGG_120.root",
    "HToWW2Q1L_120.root",
    "HToWW4Q_120.root",
    "HtoWW4Q_120.root",
    "TTBar_120.root",
    "TTBarLep_120.root",
    "WToQQ_120.root",
    "ZJetsToNuNu_120.root",
    "ZToQQ_120.root"
]
# Define the output CSV file path (will include model probabilities)
OUTPUT_CSV = "inference_5M_percentage_with_probs.csv"

# Define the list of particle and scalar feature keys to read from the ROOT files
particle_keys = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
    'part_dzval', 'part_dzerr', 'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon'
]

# Define scalar features that have a single value per jet
scalar_keys = [
    'label_QCD','label_Hbb','label_Hcc','label_Hgg',
    'label_H4q','label_Hqql','label_Zqq','label_Wqq',
    'label_Tbqq','label_Tbl','jet_pt','jet_eta','jet_phi',
    'jet_energy','jet_nparticles','jet_sdmass','jet_tau1',
    'jet_tau2','jet_tau3','jet_tau4','aux_genpart_eta',
    'aux_genpart_phi','aux_genpart_pid','aux_genpart_pt',
    'aux_truth_match'
]

# Combine all feature keys
pf_keys = particle_keys + scalar_keys

# Define the list of the different types of jets
label_names = ["QCD","Hbb","Hcc","Hgg","Htoww4q","Hqql","Zqq","Znn","Htoww2q1L","Ttbar", "Ttbarlep"]

#  Container to hold configuration settings for get_model function
class DummyDataConfig:
    # map of input feature indices
    input_dicts = {"pf_features": list(range(37))}
    # Name of the model input
    input_names = ["pf_points"]
    # Expected tensor shape for the model input
    input_shapes = {"pf_points": (MAX_PART, 37)}
    # Names of output labels
    label_names = ["label"]
    # Number of output classes
    num_classes = 10

data_config = DummyDataConfig()
# Check for GPU availability and load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create the model
model, _ = get_model(data_config, num_classes=data_config.num_classes, export_embed=True)
# Put the model in evaluation mode and move it to the appropriate device
model.eval().to(device)

def build_pf_tensor(arrays, i):
    """Return model inputs for event i, or None if too many particles."""
    n_part = arrays["part_px"][i].shape[0]
    if n_part > MAX_PART:
        return None
    particle_feats = [arrays[k][i] for k in particle_keys]
    scalar_feats = [np.full(n_part, arrays[k][i]) for k in scalar_keys]
    all_feats = particle_feats + scalar_feats
    pf_features = np.stack(all_feats, axis=1).astype(np.float32)
    padded = np.zeros((MAX_PART, pf_features.shape[1]), dtype=np.float32)
    padded[:n_part, :] = pf_features
    jet_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
    lorentz_vectors = jet_tensor[:, :, 0:4].transpose(1, 2)
    features = jet_tensor[:, :, 4:].transpose(1, 2)
    mask = (jet_tensor.sum(dim=2) != 0).unsqueeze(1)
    points = lorentz_vectors  # Use lorentz vectors as points since they contain the spatial information ##############################################
    return points, features, lorentz_vectors, mask

def get_truth_label(arrays, i):
    labs = np.array([arrays[k][i] for k in [
        'label_QCD','label_Hbb','label_Hcc','label_Hgg',
        'label_H4q','label_Hqql','label_Zqq','label_Wqq',
        'label_Tbqq','label_Tbl'
    ]])
    y = int(np.argmax(labs))
    return y, label_names[y]

def jet_masses(arrays, i):
    jet_sdmass = float(arrays["jet_sdmass"][i])
    pt  = float(arrays["jet_pt"][i])
    eta = float(arrays["jet_eta"][i])
    phi = float(arrays["jet_phi"][i])
    E   = float(arrays["jet_energy"][i])
    px = pt * cos(phi); py = pt * sin(phi); pz = pt * sinh(eta)
    m2 = max(E*E - (px*px + py*py + pz*pz), 0.0)
    return jet_sdmass, math.sqrt(m2), pt, eta, phi


import random

# reproducible sampling
np.random.seed(42)
random.seed(42)


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    wrote_header = False
    print(f"\nProcessing {PERCENTAGE}% of events from each ROOT file...")

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        for fn in root_files:
            file_path = os.path.join(root_dir, fn)
            print(f"\nProcessing file: {fn}")
            
            try:
                # Process each file individually
                it = uproot.iterate(
                    f"{file_path}:{TREE_NAME}",
                    expressions=pf_keys,
                    library="np",
                    step_size=STEP_SIZE
                )

                for arrays in it:
                    batch_len = len(arrays["jet_pt"])
                    # Calculate how many events to process in this batch
                    n_to_process = max(1, int(batch_len * PERCENTAGE / 100))
                    # Randomly select indices to process
                    indices = np.random.choice(batch_len, size=n_to_process, replace=False)
                    indices.sort()  # Sort for efficiency
                    
                    for i in tqdm(indices, desc=f"Processing {fn}"):
                        try:
                            built = build_pf_tensor(arrays, i)
                            if built is None:
                                continue
                            points, features, lorentz_vectors, mask = built
                            with torch.no_grad():
                                logits, embedding = model(points, features, lorentz_vectors, mask)
                                # compute probabilities from logits (softmax)
                                try:
                                    probs_t = torch.nn.functional.softmax(logits, dim=1)
                                except Exception:
                                    probs_t = torch.tensor(logits)
                                    probs_t = torch.nn.functional.softmax(probs_t, dim=1)

                                probs = probs_t.squeeze(0).cpu().numpy()
                                emb = embedding.squeeze(0).cpu().numpy()

                            if not wrote_header:
                                base = ["file","global_index","truth_label","label_name",
                                        "jet_sdmass","jet_mass","jet_pt","jet_eta","jet_phi"]
                                prob_cols = [f"prob_{j}" for j in range(probs.shape[-1])]
                                emb_cols = [f"emb_{j}" for j in range(emb.shape[-1])]
                                writer.writerow(base + prob_cols + emb_cols)
                                wrote_header = True

                            truth_label, label_name = get_truth_label(arrays, i)
                            jet_sdmass, jet_mass, pt, eta, phi = jet_masses(arrays, i)

                            row = [fn, i, truth_label, label_name,
                                   jet_sdmass, jet_mass, pt, eta, phi] + list(probs.astype(np.float32)) + list(emb.astype(np.float32))
                            writer.writerow(row)

                        except Exception as e:
                            print(f"Error processing event {i} in file {fn}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Error processing file {fn}: {str(e)}")
                continue

    print(f"\n✅ Finished processing. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()