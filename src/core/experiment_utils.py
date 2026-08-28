import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from . import config as cfg
except (ImportError, ValueError):
    import config as cfg


NUM_FOLDS = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42
SPLIT_PATH = os.path.join(cfg.PROJECT_ROOT, "splits_seed42.json")


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_physionet_wa(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    mAA, mPP, mUU = cm[0, 0], cm[1, 1], cm[2, 2]
    N_A, N_P, N_U = cm[0].sum(), cm[1].sum(), cm[2].sum()
    denominator = 5 * N_P + 3 * N_U + N_A

    return {
        "wa": float((5 * mPP + 3 * mUU + mAA) / denominator) if denominator > 0 else 0.0,
        "sensitivity_present": float(cm[1, 1] / N_P) if N_P > 0 else 0.0,
        "specificity_absent": float(cm[0, 0] / N_A) if N_A > 0 else 0.0,
        "recall_unknown": float(cm[2, 2] / N_U) if N_U > 0 else 0.0,
        "confusion_matrix": cm.astype(int).tolist(),
        "N_A": int(N_A),
        "N_P": int(N_P),
        "N_U": int(N_U),
    }


def get_or_create_split(patients, patient_labels, patient_indices_map):
    """Create or load one patient-level split reused by all experiments."""
    patient_set = set(patients)
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        saved = set(split["trainval_patients"]) | set(split["test_patients"])
        if saved == patient_set:
            return split

    trainval_patients, test_patients = train_test_split(
        patients,
        test_size=TEST_SIZE,
        stratify=patient_labels,
        random_state=RANDOM_STATE,
    )
    trainval_labels = [patient_labels[patients.index(p)] for p in trainval_patients]

    folds = []
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_pos, val_pos in skf.split(trainval_patients, trainval_labels):
        folds.append(
            {
                "train_patients": [trainval_patients[i] for i in train_pos],
                "val_patients": [trainval_patients[i] for i in val_pos],
            }
        )

    split = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "num_folds": NUM_FOLDS,
        "trainval_patients": list(trainval_patients),
        "test_patients": list(test_patients),
        "folds": folds,
        "patient_file_counts": {
            p: len(patient_indices_map[p]) for p in patients
        },
    }
    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)
    return split


def indices_for_patients(patient_ids, patient_indices_map):
    return [idx for patient_id in patient_ids for idx in patient_indices_map[patient_id]]


def entropy_from_probs(probs):
    probs = np.clip(np.asarray(probs), 1e-9, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def energy_from_logits(logits):
    logits = np.asarray(logits, dtype=np.float64)
    max_logits = np.max(logits, axis=1, keepdims=True)
    logsumexp = max_logits + np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))
    return -logsumexp.squeeze(1)


def predict_with_unknown_threshold(probs, threshold=None):
    preds = np.argmax(probs, axis=1).astype(int)
    if threshold is not None and np.isfinite(threshold):
        preds[entropy_from_probs(probs) > threshold] = 2
    return preds


def sweep_entropy_threshold(probs, labels, thresholds=None):
    """Tune threshold on validation/OOF data only, maximizing WA then Unknown recall."""
    if thresholds is None:
        thresholds = np.r_[np.arange(0.45, 1.101, 0.025), np.inf]

    best = None
    for threshold in thresholds:
        preds = predict_with_unknown_threshold(probs, threshold)
        metrics = compute_physionet_wa(labels, preds)
        candidate = {
            "threshold": float(threshold) if np.isfinite(threshold) else None,
            "metrics": metrics,
        }
        if best is None:
            best = candidate
            continue
        current_key = (metrics["wa"], metrics["recall_unknown"])
        best_key = (best["metrics"]["wa"], best["metrics"]["recall_unknown"])
        if current_key > best_key:
            best = candidate
    return best


def sweep_unknown_score_threshold(base_preds, unknown_scores, labels, thresholds=None):
    """Tune a score threshold on validation/OOF data; higher score predicts Unknown."""
    if thresholds is None:
        thresholds = np.r_[np.linspace(np.min(unknown_scores), np.max(unknown_scores), 150), np.inf]

    best = None
    for threshold in thresholds:
        preds = np.asarray(base_preds, dtype=int).copy()
        if np.isfinite(threshold):
            preds[unknown_scores > threshold] = 2
        metrics = compute_physionet_wa(labels, preds)
        candidate = {
            "threshold": float(threshold) if np.isfinite(threshold) else None,
            "metrics": metrics,
        }
        if best is None:
            best = candidate
            continue
        current_key = (metrics["wa"], metrics["recall_unknown"])
        best_key = (best["metrics"]["wa"], best["metrics"]["recall_unknown"])
        if current_key > best_key:
            best = candidate
    return best


def aggregate_patient_probs(record_probs, record_labels, record_patient_ids, method="mean"):
    """
    Aggregate recording probabilities into one patient-level probability vector.

    Methods:
    - mean: average all recording probabilities. This is the conservative baseline.
    - max_present: use the full probability vector from the recording with the
      highest Present probability. This follows the clinical idea that a murmur
      may be audible in only one auscultation location.
    - noisy_or_present: combine Present as a multi-instance event using noisy-OR,
      then distribute the remaining probability mass over Absent/Unknown using
      their mean relative support.
    """
    probs_by_patient = defaultdict(list)
    labels_by_patient = {}

    for probs, label, patient_id in zip(record_probs, record_labels, record_patient_ids):
        probs_by_patient[patient_id].append(np.asarray(probs, dtype=float))
        labels_by_patient[patient_id] = int(label)

    patient_ids = sorted(probs_by_patient)
    patient_rows = []
    for patient_id in patient_ids:
        rows = np.vstack(probs_by_patient[patient_id])

        if method == "mean":
            patient_prob = np.mean(rows, axis=0)
        elif method == "max_present":
            patient_prob = rows[int(np.argmax(rows[:, 1]))]
        elif method == "noisy_or_present":
            present = 1.0 - np.prod(1.0 - rows[:, 1])
            present = float(np.clip(present, 0.0, 1.0))
            other_mean = np.mean(rows[:, [0, 2]], axis=0)
            other_total = float(other_mean.sum())
            residual = 1.0 - present
            if other_total > 0:
                absent = residual * float(other_mean[0] / other_total)
                unknown = residual * float(other_mean[1] / other_total)
            else:
                absent = residual / 2.0
                unknown = residual / 2.0
            patient_prob = np.asarray([absent, present, unknown], dtype=float)
        else:
            raise ValueError(
                f"Unsupported patient aggregation method: {method!r}. "
                "Use 'mean', 'max_present', or 'noisy_or_present'."
            )

        patient_prob = np.clip(patient_prob, 1e-9, 1.0)
        patient_prob = patient_prob / patient_prob.sum()
        patient_rows.append(patient_prob)

    patient_probs = np.vstack(patient_rows)
    patient_labels = np.asarray([labels_by_patient[p] for p in patient_ids], dtype=int)
    return patient_ids, patient_probs, patient_labels


def bootstrap_metric_ci(labels, preds, n_boot=1000, seed=RANDOM_STATE):
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    rng = np.random.default_rng(seed)
    rows = []

    if len(labels) == 0:
        return {}

    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        m = compute_physionet_wa(labels[idx], preds[idx])
        rows.append(
            [
                m["wa"],
                m["sensitivity_present"],
                m["specificity_absent"],
                m["recall_unknown"],
            ]
        )

    arr = np.asarray(rows)
    names = ["wa", "sensitivity_present", "specificity_absent", "recall_unknown"]
    return {
        name: {
            "low": float(np.percentile(arr[:, i], 2.5)),
            "high": float(np.percentile(arr[:, i], 97.5)),
        }
        for i, name in enumerate(names)
    }
