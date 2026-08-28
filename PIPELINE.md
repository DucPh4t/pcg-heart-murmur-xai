# HEART MURMUR DETECTION RESEARCH PIPELINE (3-CLASS)

This document describes the current paper-facing FGA pipeline. The active
configuration is defined in `src/config.py`.

## Task Definition

- Dataset/task: PhysioNet/CinC 2022 heart murmur classification.
- Target label: `Murmur`.
- Classes: `Absent`, `Present`, `Unknown`.
- Patient split: yes, all recordings from one patient stay in the same split.
- Main report level: patient-level.
- Secondary report level: recording-level.
- Metric: PhysioNet weighted accuracy, with weights:
  - `Absent = 1`
  - `Present = 5`
  - `Unknown = 3`

The pipeline does not train on `Outcome` (`Normal`/`Abnormal`), and does not use
binary 2-class labels.

## Pipeline Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — DATA AND LABEL SETUP                                       │
├──────────────────────────────────────────────────────────────────────┤
│ Input files: PhysioNet/CinC 2022 PCG recordings                       │
│ Metadata: training_data.csv                                           │
│ Target: Murmur only                                                   │
│ Classes: Absent / Present / Unknown                                   │
│ Split: fixed patient-level split in splits_seed42.json                │
│ Guarantee: recordings from one patient never cross train/val/test     │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — FIRST-10s PREPROCESSING                                    │
├──────────────────────────────────────────────────────────────────────┤
│ 1. Load WAV with librosa                                              │
│ 2. Resample to 4000 Hz                                                │
│ 3. Butterworth bandpass: 25-600 Hz                                    │
│ 4. Crop/pad first 10.0 seconds                                        │
│ 5. Log-mel spectrogram                                                │
│    - n_fft=256, hop=64, n_mels=96, fmin=20, fmax=800                 │
│ 6. Fold-specific train-only z-score normalization                     │
│ 7. Train-only augmentation                                            │
│    - small time shift, Gaussian noise, frequency masking              │
│ Output tensor: (1, 96, 626)                                           │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — 5-FOLD FGA TRAINING                                        │
├──────────────────────────────────────────────────────────────────────┤
│ Backbone: ImageNet-pretrained ResNet18                                │
│ Input adaptation: RGB conv1 averaged to 1 channel                     │
│ Attention: Channel Attention + Frequency-Guided Attention             │
│ Pooling: Temporal Attention Pooling                                   │
│ Output: 3 logits (Absent, Present, Unknown)                           │
│ Loss: FocalLoss(alpha=[1,5,4], gamma=2, label_smoothing=0.1)          │
│ Sampler: none                                                         │
│ Checkpoint selection: best validation patient-level WA                 │
│ Saved artifacts:                                                      │
│ - checkpoints/best_model_first10_D_patientckpt_focal_g2_unknown4_fold*.pth
│ - checkpoints/norm_stats_first10_D_patientckpt_focal_g2_unknown4_fold*.npz
└──────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 4 — OOF CALIBRATION                                            │
├──────────────────────────────────────────────────────────────────────┤
│ Collect validation predictions from all five folds                    │
│ Tune entropy threshold on OOF validation only                         │
│ No held-out test labels are used for threshold selection              │
│ Calibrate separately for:                                             │
│ - recording-level predictions                                         │
│ - patient-level predictions                                           │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 5 — HELD-OUT TEST EVALUATION                                   │
├──────────────────────────────────────────────────────────────────────┤
│ Fold ensemble: average probabilities/logits                           │
│ Recording-level metrics: reported as secondary evidence               │
│ Patient-level metrics: main paper result                              │
│ Patient aggregation: average recording probabilities per patient       │
│ Reported variants:                                                    │
│ - argmax                                                              │
│ - OOF-calibrated entropy for Unknown handling                         │
│ Metrics: WA, Present sensitivity, Absent specificity, Unknown recall  │
│ Uncertainty: bootstrap confidence intervals                           │
│ Saved artifacts:                                                      │
│ - results/results_first10_D_patientckpt_focal_g2_unknown4.json        │
│ - results/confusion_matrix_first10_D_patientckpt_focal_g2_unknown4.png│
└──────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 6 — EXPLAINABILITY AND ANALYSIS                                │
├──────────────────────────────────────────────────────────────────────┤
│ Grad-CAM++: fold-specific normalized inputs for ensemble XAI          │
│ SHAP: fixed single fold to keep one stable input normalization space  │
│ XAI caution: regenerate figures after each experiment config change   │
│ Saved artifact: results/xai_first10_D_patientckpt_focal_g2_unknown4.json
└──────────────────────────────────────────────────────────────────────┘
```

## Patient-Level MIL Extension

The paper-facing extension is a patient-level multiple-instance learning (MIL)
model. It encodes each recording with the same ResNet18-FGA feature extractor,
adds a recording-location embedding, and aggregates all available recordings
from one patient with learned attention pooling.

```python
MIL_EXPERIMENT_NAME = "first10_F_patient_mil_focal_g2_unknown4"
MIL_USE_LOCATION_EMBEDDING = True
```

This keeps the stable D recording-level baseline intact while testing whether
patient-level aggregation improves the main evaluation target.

## Main Protocol: First 10 Seconds

The active main protocol uses the first 10 seconds of each recording, matching
the prior conference protocol.

```python
EXPERIMENT_NAME = "first10_D_patientckpt_focal_g2_unknown4"
CROP_MODE_TRAIN = "start"
CROP_MODE_EVAL = "start"
NORM_CROP_MODE = "start"
EVAL_MULTI_CROP = False
LOSS_WEIGHTS = [1, 5, 4]
```

Expected artifacts:

- `checkpoints/best_model_first10_D_patientckpt_focal_g2_unknown4_fold0.pth` ...
  `checkpoints/best_model_first10_D_patientckpt_focal_g2_unknown4_fold4.pth`
- `checkpoints/norm_stats_first10_D_patientckpt_focal_g2_unknown4_fold0.npz` ...
  `checkpoints/norm_stats_first10_D_patientckpt_focal_g2_unknown4_fold4.npz`
- `results/results_first10_D_patientckpt_focal_g2_unknown4.json`
- `results/confusion_matrix_first10_D_patientckpt_focal_g2_unknown4.png`
- `results/xai_first10_D_patientckpt_focal_g2_unknown4.json`

Multi-crop is no longer the active main protocol. It can be used as an ablation
by changing the config preset.

## Preprocessing

1. Load `.wav` with `librosa`.
2. Resample to `4000 Hz`.
3. Apply Butterworth bandpass filter `25-600 Hz`.
4. Crop/pad to exactly `10.0 s`.
5. Compute log-mel spectrogram:
   - `n_fft = 256`
   - `hop_length = 64`
   - `n_mels = 96`
   - `fmin = 20`
   - `fmax = 800`
6. Normalize with fold-specific train-only per-frequency mean/std.
7. Apply train-time augmentation only:
   - small time shift
   - Gaussian noise
   - frequency masking

Normalization stats are computed using the same crop protocol as the experiment:

```python
NORM_CROP_MODE = "start"
```

This avoids mixing first-10s evaluation with center-crop statistics.

## Model

- Backbone: ImageNet-pretrained ResNet18.
- Input adaptation: RGB conv1 weights averaged into one spectrogram channel.
- Attention:
  - channel attention
  - frequency-guided attention (FGA)
  - temporal attention pooling
- Output head: 3 logits for `Absent`, `Present`, `Unknown`.

## Imbalance Handling

Current main setting:

```python
SAMPLER_MODE = "none"
LOSS_WEIGHTS = [1, 5, 4]
```

The DataLoader does not use `WeightedRandomSampler` in the main run. Batches are
shuffled normally. Imbalance is handled by:

- `FocalLoss`
- class weights `[1, 5, 4]`
- `gamma = 2.0`
- `label_smoothing = 0.1`

This avoids stacking sampler + heavy class weights + focal loss too aggressively.

Recommended imbalance ablation:

| Setup | Sampler | Loss | Purpose |
|---|---:|---|---|
| D stable baseline | none | Focal `[1,5,4]`, gamma 2 | active recording-level baseline |
| F patient MIL | none | Focal `[1,5,4]`, gamma 2 | patient-level extension |
| Weighted CE | none | CE class weights | loss ablation |
| Sampler CE | weighted | CE no class weight | oversampling ablation |

For Q2, report D and F first. Keep the other settings as ablations if runtime
allows.

## Training And Evaluation

1. Create or load one fixed patient-level split: `splits_seed42.json`.
2. Train 5 folds on trainval patients only.
3. Compute normalization stats from each training fold only.
4. Save fold-specific stats and the checkpoint with best validation patient-level WA.
5. Calibrate Unknown threshold on OOF validation predictions only.
6. Apply fixed threshold to held-out test.
7. Report:
   - recording-level argmax
   - recording-level OOF entropy
   - patient-level argmax
   - patient-level OOF entropy
   - bootstrap confidence intervals

Run order:

```bash
python src/train.py
python src/evaluate.py
python src/compare_uncertainty.py
python src/train_mil.py
python src/evaluate_mil.py
```

## Unknown Handling

`Unknown` is treated as a labeled third class, not true external OOD.

The uncertainty layer is calibrated with OOF validation predictions:

- entropy threshold
- optional energy comparison in `src/compare_uncertainty.py`

Test labels are not used for threshold selection.

## XAI

- Grad-CAM++ uses the FGA layer.
- XAI ensemble creates fold-specific inputs with the corresponding fold
  normalization stats.
- SHAP is explained on a single fixed fold because SHAP needs one stable input
  normalization space.

XAI figures from old checkpoints should not be reused after changing the active
experiment config.

## Paper Claim Boundary

Safe claim:

> The proposed FGA pipeline provides a reproducible patient-level 3-class murmur
> classification protocol with fold-specific normalization, patient-level
> evaluation, and OOF-calibrated uncertainty handling for Unknown.

Do not overclaim:

> The model solves Unknown detection.

Unknown support is small and must be reported with confidence intervals.
