# JetClass Particle Transformer Inference

This repository provides a complete workflow for High Energy Physics jet classification. It consists of two main components:

1. Inference Script (inference_with_probs.py): Processes raw ROOT files through a Particle Transformer (ParT) model to generate predictions and embeddings.

2. Analysis Notebook (ROC_sophon.ipynb): Visualizes the performance of the model using ROC curves and validates the quality of the feature embeddings by training a downstream MLP classifier.


## Directory Structure

The script relies on specific relative paths. Ensure your directory is organized as follows:

```text
.
├── inference_with_probs.py              # Main inference script (ROOT -> CSV)
├── ROC_sophon.ipynb                     # Analysis notebook (CSV -> Plots/MLP)
├── networks/
│   └── example_ParticleTransformer_sophon.py  # Model definition
├── data/
│   └── JetClass/
│       └── val_5M/                      # Input ROOT files
│           ├── HToBB_120.root
│           └── ...
└── Val_5M_10_percent_with_probs.csv     # Generated output file
```
## Prerequisites
You will need Python 3.8+ and the following libraries for both the script and the notebook:
```bash
pip install torch numpy uproot tqdm pandas scikit-learn matplotlib seaborn joblib
```
Note: If you are running on a machine with a GPU, ensure you install the CUDA-enabled version of PyTorch.

## Part 1: Running Inference
The inference_script.py handles the heavy lifting of reading complex ROOT data, calculating jet physics properties, and running the deep learning model.

### 1. Configuration
Open `inference_with_probs.py` and adjust the global variables if necessary:

* `PERCENTAGE`: Fraction of events to process (default 1.0).
* `OUTPUT_CSV`: Name of the result file (e.g., "Val_5M_10_percent_with_probs.csv").

### 2. Execution
Run the script from the terminal:
```bash
python inference_script.py
```
This generates a CSV file containing Kinematics, Softmax Probabilities (prob_X), and Embeddings (emb_X).

## Part 2: Analysis & Evaluation
The ROC_sophon.ipynb notebook takes the CSV generated in Part 1 and performs detailed performance metrics.

### 1. Features
ROC Analysis: Computes One-vs-Rest ROC curves for all 10 jet classes.
AUC Metrics: Calculates per-class AUC, as well as Micro/Macro averages to quantify overall model performance.
Embedding Validation: Trains a lightweight Multi-Layer Perceptron (MLP) on the extracted embeddings to verify they contain linearly separable class information.
Confusion Matrix: Visualizes where the model is confusing specific jet labels.

### 2. Usage
1. Open the notebook:
```bash
jupyter notebook ROC_sophon.ipynb
```
2. Important: Locate the CSV_PATH variable in the second code cell and ensure it matches the output name from Part 1:
```python
CSV_PATH = 'Val_5M_10_percent_with_probs.csv'
```
3. Run all cells.

### 3. Notebook Outputs
The notebook will generate:
* roc_sophon.png: A high-resolution plot of the ROC curves.
* mlp_embeddings.joblib: The trained downstream MLP classifier.
* mlp_scaler.joblib: The standardization scaler used for the embeddings.
