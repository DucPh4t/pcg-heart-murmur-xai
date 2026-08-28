# Patient-Level Explainable Heart Murmur Detection From PCG Recordings

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/EAI%20RAIDS-2026-brightgreen.svg)](https://raids.eai-conferences.org/2026/accepted-papers/)

Research code for 3-class heart murmur classification on the PhysioNet/CinC 2022 phonocardiogram (PCG) dataset. The target is the `Murmur` label with three classes: `Absent`, `Present`, and `Unknown`.

The project provides a recording-level ResNet18-FGA baseline and a patient-level Multiple-Instance Learning (MIL) extension that aggregates all available recording locations for each patient.

> *This repository extends earlier conference work accepted at the 2nd EAI International Conference on Responsible Artificial Intelligence and Data Science ([EAI RAIDS 2026](https://raids.eai-conferences.org/2026/accepted-papers/)), titled **"An Explainable Attention-Based Deep Learning Framework for Pediatric Heart Murmur Screening Using Phonocardiograms."***

---

## Method Overview

```text
               +-------------------------------------------------------------+
               |                  PCG Audio Recordings                       |
               |        (AV, PV, TV, MV locations per patient)               |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               |            Stage 1: Preprocessing & Log-Mel                 |
               |  - 4000 Hz resample, Butterworth bandpass (25-600 Hz)       |
               |  - First 10.0s windowing, 96 Mel bins (20-800 Hz)           |
               |  - Fold-specific train-only z-score normalization           |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               |            Stage 2: Recording-Level Encoder                 |
               |  - ResNet18 Backbone                                        |
               |  - Channel Attention + Frequency-Guided Attention (FGA)     |
               |  - Temporal Attention Pooling (512-dim embedding)           |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               |          Stage 3: Patient-Level Aggregation (MIL)           |
               |  - Location Embedding (AV, PV, TV, MV, Phc)                 |
               |  - Learned Attention Pooling over patient recording bag     |
               |  - 3-Class Classifier (Absent / Present / Unknown)          |
               +-------------------------------------------------------------+
                                              |
                                              v
               +-------------------------------------------------------------+
               |            Stage 4: Post-Processing & Explainability        |
               |  - OOF Entropy Threshold Calibration for Unknown class      |
               |  - SHAP Frequency-Band Decomposition & MIL Attention Maps   |
               +-------------------------------------------------------------+
```

---

## Highlights

- **Patient-Level Split & Evaluation:** Strict patient-level stratification to prevent recording leakage across train, validation, and test splits.
- **Leak-Free Normalization:** Fold-specific normalization statistics computed strictly on training folds.
- **Frequency-Guided Attention (FGA):** Dual channel-frequency attention preserving spatial-temporal cardiac acoustic characteristics.
- **Patient-Level MIL:** Multi-instance learning with learnable attention pooling across arbitrary recording locations (AV, PV, TV, MV, Phc).
- **Post-Hoc Uncertainty Calibration:** Out-of-fold (OOF) entropy thresholding to calibrate predictions for the ambiguous `Unknown` class.
- **Clinical Interpretability:** SHAP frequency-band attribution and MIL recording attention maps.

---

## Results

Evaluation on the internal held-out patient-level test split with 189 patients (`Absent=139`, `Present=36`, `Unknown=14`). *(Note: These are internal validation/test results, not the hidden official challenge test set).*

### Main Classification Metrics

| Method | Level | Weighted Acc. (%) | Present Sens. (%) | Absent Spec. (%) | Unknown Recall (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **FGA Baseline** | Patient | 77.56 | 80.56 | 92.81 | 14.29 |
| **Patient MIL**  | Patient | **77.84** | **80.56** | 91.37 | **21.43** |

### Additional Performance Metrics

| Method | Accuracy (%) | Macro-F1 (%) |
|---|:---:|:---:|
| **FGA Baseline** | 84.66 | 63.60 |
| **Patient MIL**  | 84.13 | **65.24** |

### Bootstrap Confidence Intervals (Patient MIL, 95% CI)

| Metric | 95% Confidence Interval |
|---|:---:|
| **Weighted Accuracy (WA)** | 69.57 - 85.47 |
| **Present Sensitivity**     | 67.57 - 92.31 |
| **Absent Specificity**      | 86.71 - 95.45 |
| **Unknown Recall**          | 0.00 - 44.44  |

---

## Explainability & Visualizations

Curated figures are stored in `docs/figures/`:

| MIL Attention Distribution | Confusion Matrix | SHAP Frequency Bands |
|:---:|:---:|:---:|
| ![MIL attention by location](docs/figures/mil_attention_by_location.png) | ![Patient MIL confusion matrix](docs/figures/confusion_matrix_mil.png) | ![SHAP signed frequency bands](docs/figures/shap_signed_frequency_bands.png) |
| *Attention weights across auscultation locations* | *Patient-level MIL confusion matrix* | *Signed feature contributions by clinical band* |

---

## Repository Layout

```text
pcg-heart-murmur-xai/
├── src/
│   ├── config.py                 # Central experiment configuration
│   ├── dataset.py                # Dataset loading, mel-spectrogram, MIL bags
│   ├── model.py                  # ResNet18-FGA and Patient-Level MIL architectures
│   ├── train.py                  # 5-fold baseline training
│   ├── evaluate.py               # Baseline inference & patient evaluation
│   ├── train_mil.py              # Patient-level MIL 5-fold training
│   ├── evaluate_mil.py           # Patient-level MIL inference & evaluation
│   ├── export_mil_attention.py   # MIL attention CSV export
│   ├── compute_shap.py           # SHAP analysis for correct cases
│   ├── compute_shap_errors.py    # SHAP analysis for misclassifications
│   ├── compute_xai.py            # Grad-CAM++ overlap metrics
│   ├── generate_xai.py           # Grad-CAM++ visualization export
│   ├── experiment_utils.py       # Metrics, split management, seed utilities
│   └── utils.py                  # Loss functions and DSP helpers
├── checkpoints/                  # Trained model checkpoints (*.pth) and norm stats (*.npz)
├── results/                      # Evaluation JSONs, confusion matrices, and XAI figures
├── docs/figures/                 # Curated figures for public documentation
├── PIPELINE.md                   # Comprehensive pipeline specification
├── requirements.txt              # Python package dependencies
├── splits_seed42.json            # Deterministic patient-level split definition
└── .gitignore                    # Artifact exclusion rules
```

---

## Prerequisites & Environment

- **Python:** 3.9, 3.10, or 3.11
- **Hardware Acceleration:** CUDA (NVIDIA GPU), MPS (Apple Silicon), or CPU

```bash
# Clone the repository
git clone https://github.com/DucPh4t/pcg-heart-murmur-xai.git
cd pcg-heart-murmur-xai

# Install dependencies
pip install -r requirements.txt
```

> **PyTorch Installation Note:** For optimal GPU acceleration, install the appropriate PyTorch build matching your hardware from [pytorch.org](https://pytorch.org/).

---

## Dataset Setup

Download the dataset from the [PhysioNet Challenge 2022](https://physionet.org/content/challenge-2022/1.0.0/) and place the files under `data/raw/`:

```text
data/
└── raw/
    ├── training_data.csv
    └── training_data/
        ├── 2530_AV.wav
        ├── 2530_MV.wav
        └── ...
```

---

## Reproducing Experiments

### 1. Recording-Level Baseline (ResNet18-FGA)

```bash
# Train 5-fold baseline models
python src/train.py

# Evaluate on test set with OOF entropy calibration
python src/evaluate.py

# Compare uncertainty strategies (Entropy vs Energy)
python src/compare_uncertainty.py
```

### 2. Patient-Level Multiple-Instance Learning (MIL)

```bash
# Train 5-fold Patient-MIL models
python src/train_mil.py

# Evaluate Patient-MIL
python src/evaluate_mil.py

# Export recording attention weights
python src/export_mil_attention.py
```

### 3. Explainability (SHAP & Grad-CAM++)

```bash
# Generate SHAP frequency-band analysis
python src/compute_shap.py
python src/compute_shap_errors.py

# Compute and visualize Grad-CAM++ maps
python src/compute_xai.py
python src/generate_xai.py
```

---

## Limitations & Disclaimer

- **Research Prototype:** This software is an academic research prototype and is **not** certified for clinical diagnosis or medical use.
- **Class Imbalance on Unknown:** The `Unknown` murmur category has limited sample support (14 patients in the held-out split), resulting in wider confidence intervals for Unknown recall.
- **Internal Split:** All reported numbers reflect the fixed internal patient-level split (`splits_seed42.json`), not the official hidden PhysioNet test set.

---

## Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@inproceedings{heartmurmur2026raids,
  title     = {An Explainable Attention-Based Deep Learning Framework for Pediatric Heart Murmur Screening Using Phonocardiograms},
  booktitle = {Proceedings of the 2nd EAI International Conference on Responsible Artificial Intelligence and Data Science (EAI RAIDS 2026)},
  year      = {2026},
  publisher = {Springer}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
