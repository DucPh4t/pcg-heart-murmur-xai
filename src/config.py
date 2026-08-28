"""
config.py
================
Central config for the 3-class FGA experiment (Present / Unknown / Absent).

3-Class Mapping (aligns with PhysioNet 2022 Weighted Accuracy formula):
  0 = Absent   (weight 1 in WA)
  1 = Present  (weight 5 in WA)
  2 = Unknown  (weight 3 in WA)
"""

import os

# Paths
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.dirname(BASE_DIR)
DATA_DIR       = os.path.join(PROJECT_ROOT, "data", "raw", "training_data")
CSV_PATH       = os.path.join(PROJECT_ROOT, "data", "raw", "training_data.csv")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")


def get_checkpoint_path(filename):
    """Find a checkpoint/norm file in CHECKPOINT_DIR or fallback to PROJECT_ROOT."""
    p = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(p):
        return p
    root_p = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(root_p):
        return root_p
    return p


def get_results_path(filename):
    """Find a results/XAI file in RESULTS_DIR or fallback to PROJECT_ROOT."""
    p = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(p):
        return p
    root_p = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(root_p):
        return root_p
    return p

# Audio Parameters
SAMPLE_RATE  = 4000
DURATION     = 10.0
NUM_SAMPLES  = int(SAMPLE_RATE * DURATION)
BANDPASS_LOW  = 25
BANDPASS_HIGH = 600

# Spectrogram Parameters
N_FFT        = 256
HOP_LENGTH   = 64
N_MELS       = 96
FMIN         = 20
FMAX         = 800

# Location-aware input. Channel 0 is the audio spectrogram; the remaining
# channels are one-hot location maps broadcast over the spectrogram.
USE_LOCATION_CHANNELS = False
LOCATION_LABELS = {
    "AV": 0,
    "PV": 1,
    "TV": 2,
    "MV": 3,
    "Phc": 4,
}
NUM_LOCATION_CHANNELS = len(LOCATION_LABELS)
INPUT_CHANNELS = 1 + NUM_LOCATION_CHANNELS if USE_LOCATION_CHANNELS else 1

# Experiment naming keeps ablations from overwriting each other.
# Main paper protocol: first 10 seconds, matching the prior conference setup.
#
# Multi-crop ablation preset:
#   EXPERIMENT_NAME = "multicrop"
#   CROP_MODE_TRAIN = "random"
#   CROP_MODE_EVAL = "center"
#   NORM_CROP_MODE = "center"
#   EVAL_MULTI_CROP = True
EXPERIMENT_NAME = "first10_D_patientckpt_focal_g2_unknown4"

# Cropping / evaluation windows
CROP_MODE_TRAIN = "start"
CROP_MODE_EVAL = "start"
NORM_CROP_MODE = "start"
EVAL_MULTI_CROP = False
EVAL_CROP_STRIDE_SEC = 5.0

# 3-class labels and metric weights
NUM_CLASSES  = 3
CLASS_LABELS = {
    'Absent':  0,
    'Present': 1,
    'Unknown': 2,
}

# PhysioNet 2022 WA weights: Present=5, Unknown=3, Absent=1
# Index aligns with CLASS_LABELS above
WA_WEIGHTS   = [1, 5, 3]   # [Absent, Present, Unknown]

# Training Parameters
BATCH_SIZE        = 32
LEARNING_RATE     = 1e-4
NUM_EPOCHS        = 25       # Slightly more epochs for 3 classes
NUM_WORKERS       = 0        # Keep 0 for portable runs without shared-memory issues

# Data Augmentation
AUGMENTATION_ENABLED   = True
AUGMENTATION_MULTIPLIER = 2
TIME_SHIFT_ENABLED     = True
TIME_SHIFT_MAX_MS      = 10
TIME_SHIFT_PROB        = 0.3
GAUSSIAN_NOISE_ENABLED = True
GAUSSIAN_NOISE_SCALE   = 0.003
GAUSSIAN_NOISE_PROB    = 0.3
FREQ_MASK_ENABLED      = True
FREQ_MASK_PARAM        = 10
FREQ_MASK_PROB         = 0.2

# Normalization is estimated from train folds only and then reused for val/test.
NORMALIZATION_MODE = "train_stats_per_frequency"
NORM_STATS_PREFIX = f"norm_stats_{EXPERIMENT_NAME}_fold"

# Imbalance handling: avoid stacking sampler + heavy class weights + focal too aggressively.
# Options: "none" or "weighted".
SAMPLER_MODE = "none"

# Metric weights follow PhysioNet 2022. Training loss uses the same scale by
# default; tune in ablations if needed.
LOSS_WEIGHTS = [1, 5, 4]   # [Absent, Present, Unknown]
LOSS_TYPE = "focal"        # Options: "focal", "ce"
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1

# Model and artifact prefixes keep experiments from overwriting each other.
MODEL_PREFIX = f"best_model_{EXPERIMENT_NAME}"
RESULTS_FILENAME = f"results_{EXPERIMENT_NAME}.json"
CONFUSION_MATRIX_FILENAME = f"confusion_matrix_{EXPERIMENT_NAME}.png"
XAI_FILENAME = f"xai_{EXPERIMENT_NAME}.json"

# Patient-level MIL experiment. This is intentionally separate from
# EXPERIMENT_NAME so the stable recording-level D baseline remains untouched.
MIL_EXPERIMENT_NAME = "first10_F_patient_mil_focal_g2_unknown4"
MIL_MODEL_PREFIX = f"best_model_{MIL_EXPERIMENT_NAME}"
MIL_NORM_STATS_PREFIX = f"norm_stats_{MIL_EXPERIMENT_NAME}_fold"
MIL_RESULTS_FILENAME = f"results_{MIL_EXPERIMENT_NAME}.json"
MIL_CONFUSION_MATRIX_FILENAME = f"confusion_matrix_{MIL_EXPERIMENT_NAME}.png"
MIL_BATCH_SIZE = 8
MIL_USE_LOCATION_EMBEDDING = True
MIL_LOCATION_EMBED_DIM = 16
