import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import config as cfg
from dataset import HeartMurmurDataset, get_patient_info, load_normalization_stats
from experiment_utils import (
    NUM_FOLDS,
    RANDOM_STATE,
    aggregate_patient_probs,
    bootstrap_metric_ci,
    compute_physionet_wa,
    get_or_create_split,
    indices_for_patients,
    predict_with_unknown_threshold,
    seed_worker,
    set_seed,
    sweep_entropy_threshold,
)
from model import get_model


CLASS_NAMES = ["Absent", "Present", "Unknown"]
PATIENT_AGGREGATION_METHODS = ["mean", "max_present", "noisy_or_present"]


def load_model(fold_idx, device):
    model_path = os.path.join(cfg.PROJECT_ROOT, f"{cfg.MODEL_PREFIX}_fold{fold_idx}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    model = get_model(num_classes=cfg.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_fold_norm_stats(fold_idx):
    stats_path = os.path.join(cfg.PROJECT_ROOT, f"{cfg.NORM_STATS_PREFIX}{fold_idx}.npz")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing normalization stats for fold {fold_idx}: {stats_path}. "
            "Retrain with src/train.py to create fold-specific stats."
        )
    return load_normalization_stats(stats_path)


def infer_model(model, dataset, indices, device):
    if cfg.EVAL_MULTI_CROP:
        probs, logits, labels = [], [], []
        with torch.no_grad():
            for idx in indices:
                crops, label = dataset.get_crops(idx)
                crop_logits = model(crops.to(device))
                mean_logits = crop_logits.mean(dim=0, keepdim=True)
                logits.append(mean_logits.cpu().numpy())
                probs.append(F.softmax(mean_logits, dim=1).cpu().numpy())
                labels.append(int(label.item()))
        return np.vstack(probs), np.vstack(logits), np.asarray(labels, dtype=int)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        worker_init_fn=seed_worker,
    )
    probs, logits, labels = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            batch_logits = model(imgs.to(device))
            logits.append(batch_logits.cpu().numpy())
            probs.append(F.softmax(batch_logits, dim=1).cpu().numpy())
            labels.extend(lbls.numpy().tolist())
    return np.vstack(probs), np.vstack(logits), np.asarray(labels, dtype=int)


def infer_ensemble(dataset, indices, device):
    fold_probs, fold_logits = [], []
    labels = None

    for fold_idx in range(NUM_FOLDS):
        model = load_model(fold_idx, device)
        dataset.norm_stats = load_fold_norm_stats(fold_idx)
        probs, logits, batch_labels = infer_model(model, dataset, indices, device)
        fold_probs.append(probs)
        fold_logits.append(logits)
        if labels is None:
            labels = batch_labels
        print(f"  Fold {fold_idx + 1}: inference complete")

    return (
        np.mean(np.stack(fold_probs, axis=0), axis=0),
        np.mean(np.stack(fold_logits, axis=0), axis=0),
        labels,
    )


def collect_oof_predictions(dataset, split, patient_indices_map, device):
    oof_probs, oof_labels, oof_patient_ids = [], [], []

    for fold_idx, fold in enumerate(split["folds"]):
        val_patients = fold["val_patients"]
        val_indices = indices_for_patients(val_patients, patient_indices_map)
        model = load_model(fold_idx, device)
        dataset.norm_stats = load_fold_norm_stats(fold_idx)
        probs, _, labels = infer_model(model, dataset, val_indices, device)
        patient_ids = [dataset.file_list[i][2] for i in val_indices]

        oof_probs.append(probs)
        oof_labels.extend(labels.tolist())
        oof_patient_ids.extend(patient_ids)
        print(f"  OOF fold {fold_idx + 1}: {len(val_indices)} recordings")

    return np.vstack(oof_probs), np.asarray(oof_labels, dtype=int), oof_patient_ids


def summarize(labels, probs, threshold):
    preds = predict_with_unknown_threshold(probs, threshold)
    metrics = compute_physionet_wa(labels, preds)
    return {
        "threshold": threshold_for_json(threshold),
        "metrics": metrics,
        "classification_report": classification_report(
            labels,
            preds,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "bootstrap_ci": bootstrap_metric_ci(labels, preds),
    }


def threshold_for_json(threshold):
    return None if threshold is None or not np.isfinite(threshold) else float(threshold)


def plot_confusion_matrix(labels, probs, threshold, title, out_path):
    preds = predict_with_unknown_threshold(probs, threshold)
    cm = np.asarray(compute_physionet_wa(labels, preds)["confusion_matrix"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(title)
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def choose_patient_aggregation(oof_calibrations):
    """Select aggregation using OOF only; held-out test is not used for selection."""
    return max(
        PATIENT_AGGREGATION_METHODS,
        key=lambda method: (
            oof_calibrations[method]["entropy"]["metrics"]["wa"],
            oof_calibrations[method]["entropy"]["metrics"]["recall_unknown"],
        ),
    )


def main():
    set_seed(RANDOM_STATE)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(
        f"Device: {device} | FGA 3-class ensemble evaluation | "
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

    print("\n[1/4] OOF validation predictions for threshold calibration")
    oof_probs, oof_labels, oof_patient_ids = collect_oof_predictions(
        dataset, split, patient_indices_map, device
    )

    oof_record_best = sweep_entropy_threshold(oof_probs, oof_labels)
    oof_patient_calibrations = {}
    for method in PATIENT_AGGREGATION_METHODS:
        _, method_probs, method_labels = aggregate_patient_probs(
            oof_probs,
            oof_labels,
            oof_patient_ids,
            method=method,
        )
        oof_patient_calibrations[method] = {
            "argmax": summarize(method_labels, method_probs, threshold=None),
            "entropy": sweep_entropy_threshold(method_probs, method_labels),
        }

    selected_patient_aggregation = choose_patient_aggregation(oof_patient_calibrations)
    oof_patient_best = oof_patient_calibrations["mean"]["entropy"]

    record_threshold = oof_record_best["threshold"]
    patient_threshold = oof_patient_calibrations["mean"]["entropy"]["threshold"]
    selected_patient_threshold = oof_patient_calibrations[
        selected_patient_aggregation
    ]["entropy"]["threshold"]

    print(
        f"  Recording threshold from OOF: {threshold_for_json(record_threshold)} "
        f"(OOF WA={oof_record_best['metrics']['wa']:.4f})"
    )
    print(
        f"  Patient threshold from OOF: {threshold_for_json(patient_threshold)} "
        f"(OOF WA={oof_patient_best['metrics']['wa']:.4f})"
    )
    print("  Patient aggregation OOF comparison:")
    for method in PATIENT_AGGREGATION_METHODS:
        arg_m = oof_patient_calibrations[method]["argmax"]["metrics"]
        ent_m = oof_patient_calibrations[method]["entropy"]["metrics"]
        marker = " <= selected" if method == selected_patient_aggregation else ""
        print(
            f"    {method:<16} argmax WA={arg_m['wa']:.4f} | "
            f"entropy WA={ent_m['wa']:.4f} | "
            f"Sens_P={ent_m['sensitivity_present']:.3f} | "
            f"Spec_A={ent_m['specificity_absent']:.3f} | "
            f"Rec_U={ent_m['recall_unknown']:.3f}{marker}"
        )

    print("\n[2/4] Held-out test ensemble inference")
    test_indices = indices_for_patients(split["test_patients"], patient_indices_map)
    test_probs, test_logits, test_labels = infer_ensemble(dataset, test_indices, device)
    test_patient_ids = [dataset.file_list[i][2] for i in test_indices]
    patient_ids, test_patient_probs, test_patient_labels = aggregate_patient_probs(
        test_probs,
        test_labels,
        test_patient_ids,
        method="mean",
    )
    patient_aggregation_results = {}
    for method in PATIENT_AGGREGATION_METHODS:
        method_patient_ids, method_probs, method_labels = aggregate_patient_probs(
            test_probs,
            test_labels,
            test_patient_ids,
            method=method,
        )
        method_threshold = oof_patient_calibrations[method]["entropy"]["threshold"]
        patient_aggregation_results[method] = {
            "patient_ids": method_patient_ids,
            "argmax": summarize(method_labels, method_probs, threshold=None),
            "oof_entropy": summarize(method_labels, method_probs, threshold=method_threshold),
            "oof_calibration": {
                "threshold": threshold_for_json(method_threshold),
                "argmax_oof_metrics": oof_patient_calibrations[method]["argmax"]["metrics"],
                "entropy_oof_metrics": oof_patient_calibrations[method]["entropy"]["metrics"],
            },
        }

    print("\n[3/4] Final metrics")
    record_argmax = summarize(test_labels, test_probs, threshold=None)
    record_entropy = summarize(test_labels, test_probs, threshold=record_threshold)
    patient_argmax = summarize(test_patient_labels, test_patient_probs, threshold=None)
    patient_entropy = summarize(test_patient_labels, test_patient_probs, threshold=patient_threshold)

    for name, block in [
        ("Recording argmax", record_argmax),
        ("Recording OOF entropy", record_entropy),
        ("Patient argmax", patient_argmax),
        ("Patient OOF entropy", patient_entropy),
    ]:
        m = block["metrics"]
        print(
            f"  {name}: WA={m['wa']:.4f} | "
            f"Sens_P={m['sensitivity_present']:.3f} | "
            f"Spec_A={m['specificity_absent']:.3f} | "
            f"Rec_U={m['recall_unknown']:.3f} "
            f"(A/P/U={m['N_A']}/{m['N_P']}/{m['N_U']})"
        )
    print("  Patient aggregation held-out comparison:")
    for method in PATIENT_AGGREGATION_METHODS:
        block = patient_aggregation_results[method]["oof_entropy"]
        m = block["metrics"]
        marker = " <= selected by OOF" if method == selected_patient_aggregation else ""
        print(
            f"    {method:<16} WA={m['wa']:.4f} | "
            f"Sens_P={m['sensitivity_present']:.3f} | "
            f"Spec_A={m['specificity_absent']:.3f} | "
            f"Rec_U={m['recall_unknown']:.3f}{marker}"
        )

    print("\n[4/4] Saving artifacts")
    cm_path = os.path.join(cfg.PROJECT_ROOT, cfg.CONFUSION_MATRIX_FILENAME)
    plot_confusion_matrix(
        test_patient_labels,
        test_patient_probs,
        patient_threshold,
        "Patient-level Confusion Matrix FGA (OOF-calibrated Entropy)",
        cm_path,
    )

    out = {
        "model": "FGA_3Class_Ensemble",
        "experiment": cfg.EXPERIMENT_NAME,
        "preprocessing": {
            "duration_sec": cfg.DURATION,
            "crop_mode_train": cfg.CROP_MODE_TRAIN,
            "crop_mode_eval": cfg.CROP_MODE_EVAL,
            "norm_crop_mode": cfg.NORM_CROP_MODE,
            "eval_multi_crop": cfg.EVAL_MULTI_CROP,
            "eval_crop_stride_sec": cfg.EVAL_CROP_STRIDE_SEC,
            "sample_rate": cfg.SAMPLE_RATE,
            "bandpass_low": cfg.BANDPASS_LOW,
            "bandpass_high": cfg.BANDPASS_HIGH,
            "n_fft": cfg.N_FFT,
            "hop_length": cfg.HOP_LENGTH,
            "n_mels": cfg.N_MELS,
            "fmin": cfg.FMIN,
            "fmax": cfg.FMAX,
            "normalization_mode": cfg.NORMALIZATION_MODE,
            "location_channels_enabled": cfg.USE_LOCATION_CHANNELS,
            "location_labels": cfg.LOCATION_LABELS,
            "input_channels": cfg.INPUT_CHANNELS,
        },
        "imbalance": {
            "sampler_mode": cfg.SAMPLER_MODE,
            "loss": cfg.LOSS_TYPE,
            "loss_weights": cfg.LOSS_WEIGHTS,
            "focal_gamma": cfg.FOCAL_GAMMA if cfg.LOSS_TYPE.lower() == "focal" else None,
            "label_smoothing": cfg.LABEL_SMOOTHING,
        },
        "split_file": os.path.relpath(os.path.join(cfg.PROJECT_ROOT, "splits_seed42.json"), cfg.PROJECT_ROOT),
        "calibration": {
            "method": "OOF entropy threshold; no test labels used for threshold selection",
            "recording_threshold": threshold_for_json(record_threshold),
            "recording_oof_metrics": oof_record_best["metrics"],
            "patient_threshold": threshold_for_json(patient_threshold),
            "patient_oof_metrics": oof_patient_best["metrics"],
            "patient_aggregation_methods": {
                method: {
                    "argmax_oof_metrics": oof_patient_calibrations[method]["argmax"]["metrics"],
                    "entropy_threshold": threshold_for_json(
                        oof_patient_calibrations[method]["entropy"]["threshold"]
                    ),
                    "entropy_oof_metrics": oof_patient_calibrations[method]["entropy"]["metrics"],
                }
                for method in PATIENT_AGGREGATION_METHODS
            },
        },
        "patient_aggregation": {
            "methods": PATIENT_AGGREGATION_METHODS,
            "selected_by_oof": selected_patient_aggregation,
            "selected_threshold": threshold_for_json(selected_patient_threshold),
            "selection_rule": "Maximize OOF patient-level WA, then Unknown recall.",
        },
        "recording_level": {
            "argmax": record_argmax,
            "oof_entropy": record_entropy,
        },
        "patient_level": {
            "patient_ids": patient_ids,
            "argmax": patient_argmax,
            "oof_entropy": patient_entropy,
        },
        "patient_level_by_aggregation": patient_aggregation_results,
    }

    json_path = os.path.join(cfg.PROJECT_ROOT, cfg.RESULTS_FILENAME)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"  Saved: {json_path}")
    print(f"  Saved: {cm_path}")


if __name__ == "__main__":
    main()
