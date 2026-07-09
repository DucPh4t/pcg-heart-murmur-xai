import os
import sys

import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import config as cfg
from dataset import HeartMurmurDataset, get_patient_info
from evaluate import infer_ensemble, infer_model, load_fold_norm_stats, load_model
from experiment_utils import (
    RANDOM_STATE,
    compute_physionet_wa,
    energy_from_logits,
    get_or_create_split,
    indices_for_patients,
    predict_with_unknown_threshold,
    set_seed,
    sweep_entropy_threshold,
    sweep_unknown_score_threshold,
)


def collect_oof_logits(dataset, split, patient_indices_map, device):
    all_probs, all_logits, all_labels = [], [], []
    for fold_idx, fold in enumerate(split["folds"]):
        val_indices = indices_for_patients(fold["val_patients"], patient_indices_map)
        model = load_model(fold_idx, device)
        dataset.norm_stats = load_fold_norm_stats(fold_idx)
        probs, logits, labels = infer_model(model, dataset, val_indices, device)
        all_probs.append(probs)
        all_logits.append(logits)
        all_labels.extend(labels.tolist())
    return np.vstack(all_probs), np.vstack(all_logits), np.asarray(all_labels, dtype=int)


def print_metrics(name, labels, preds):
    metrics = compute_physionet_wa(labels, preds)
    print(
        f"{name:<24} | {metrics['wa']:<8.4f} | "
        f"{metrics['sensitivity_present']:<8.2%} | "
        f"{metrics['specificity_absent']:<8.2%} | "
        f"{metrics['recall_unknown']:<8.2%}"
    )


def main():
    set_seed(RANDOM_STATE)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(
        f"Device: {device} | Unknown uncertainty methods with OOF calibration | "
        f"experiment={cfg.EXPERIMENT_NAME}"
    )

    dataset = HeartMurmurDataset(
        cfg.CSV_PATH,
        cfg.DATA_DIR,
        class_map=cfg.CLASS_LABELS,
        mode="val",
    )
    patients, patient_labels, patient_indices_map = get_patient_info(dataset)
    split = get_or_create_split(patients, patient_labels, patient_indices_map)

    print("\n[1/2] OOF calibration")
    oof_probs, oof_logits, oof_labels = collect_oof_logits(
        dataset, split, patient_indices_map, device
    )
    oof_base = np.argmax(oof_probs, axis=1)
    entropy_best = sweep_entropy_threshold(oof_probs, oof_labels)
    energy_best = sweep_unknown_score_threshold(
        oof_base,
        energy_from_logits(oof_logits),
        oof_labels,
    )

    print(f"  Entropy threshold: {entropy_best['threshold']}")
    print(f"  Energy threshold : {energy_best['threshold']}")

    print("\n[2/2] Held-out test application")
    test_indices = indices_for_patients(split["test_patients"], patient_indices_map)
    test_probs, test_logits, test_labels = infer_ensemble(dataset, test_indices, device)
    base_preds = np.argmax(test_probs, axis=1)

    entropy_preds = predict_with_unknown_threshold(test_probs, entropy_best["threshold"])
    energy_preds = base_preds.copy()
    if energy_best["threshold"] is not None:
        energy_preds[energy_from_logits(test_logits) > energy_best["threshold"]] = 2

    print(f"\n{'Method':<24} | {'WA':<8} | {'Sens P':<8} | {'Spec A':<8} | {'Rec U':<8}")
    print("-" * 76)
    print_metrics("No handling", test_labels, base_preds)
    print_metrics("Entropy OOF", test_labels, entropy_preds)
    print_metrics("Energy OOF", test_labels, energy_preds)


if __name__ == "__main__":
    main()
