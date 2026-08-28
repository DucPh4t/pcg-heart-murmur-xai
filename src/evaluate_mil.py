import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import config as cfg
from dataset import (
    HeartMurmurDataset,
    PatientMurmurBagDataset,
    get_patient_info,
    load_normalization_stats,
    patient_bag_collate,
)
from evaluate import CLASS_NAMES, plot_confusion_matrix, threshold_for_json
from experiment_utils import (
    NUM_FOLDS,
    RANDOM_STATE,
    bootstrap_metric_ci,
    compute_physionet_wa,
    get_or_create_split,
    predict_with_unknown_threshold,
    set_seed,
    sweep_entropy_threshold,
)
from model import get_patient_mil_model


def mil_norm_stats_path(fold_idx):
    return cfg.get_checkpoint_path(f"{cfg.MIL_NORM_STATS_PREFIX}{fold_idx}.npz")


def load_mil_norm_stats(fold_idx):
    stats_path = mil_norm_stats_path(fold_idx)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing MIL normalization stats for fold {fold_idx}: {stats_path}. "
            "Run src/train_mil.py first."
        )
    return load_normalization_stats(stats_path)


def load_mil_model(fold_idx, device):
    model_path = cfg.get_checkpoint_path(f"{cfg.MIL_MODEL_PREFIX}_fold{fold_idx}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing MIL checkpoint: {model_path}")
    model = get_patient_mil_model(num_classes=cfg.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def make_patient_loader(recording_dataset, patient_ids, patient_indices_map, patient_label_map):
    ds = PatientMurmurBagDataset(
        recording_dataset,
        patient_ids,
        patient_indices_map,
        patient_label_map,
    )
    return DataLoader(
        ds,
        batch_size=cfg.MIL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=patient_bag_collate,
    )


def infer_mil_model(model, recording_dataset, patient_ids, patient_indices_map, patient_label_map, device):
    loader = make_patient_loader(recording_dataset, patient_ids, patient_indices_map, patient_label_map)
    probs, logits, labels, out_patient_ids = [], [], [], []
    with torch.no_grad():
        for bags, mask, locations, batch_labels, batch_patient_ids in loader:
            batch_logits = model(
                bags.to(device),
                mask.to(device),
                locations.to(device),
            )
            logits.append(batch_logits.cpu().numpy())
            probs.append(F.softmax(batch_logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
            out_patient_ids.extend(batch_patient_ids)
    return (
        np.vstack(probs),
        np.vstack(logits),
        np.asarray(labels, dtype=int),
        out_patient_ids,
    )


def collect_oof_predictions(recording_dataset, split, patient_indices_map, patient_label_map, device):
    oof_probs, oof_logits, oof_labels, oof_patient_ids = [], [], [], []
    for fold_idx, fold in enumerate(split["folds"]):
        model = load_mil_model(fold_idx, device)
        recording_dataset.norm_stats = load_mil_norm_stats(fold_idx)
        probs, logits, labels, patient_ids = infer_mil_model(
            model,
            recording_dataset,
            fold["val_patients"],
            patient_indices_map,
            patient_label_map,
            device,
        )
        oof_probs.append(probs)
        oof_logits.append(logits)
        oof_labels.extend(labels.tolist())
        oof_patient_ids.extend(patient_ids)
        print(f"  OOF fold {fold_idx + 1}: {len(patient_ids)} patients")
    return (
        np.vstack(oof_probs),
        np.vstack(oof_logits),
        np.asarray(oof_labels, dtype=int),
        oof_patient_ids,
    )


def infer_mil_ensemble(recording_dataset, patient_ids, patient_indices_map, patient_label_map, device):
    fold_probs, fold_logits = [], []
    labels = None
    out_patient_ids = None
    for fold_idx in range(NUM_FOLDS):
        model = load_mil_model(fold_idx, device)
        recording_dataset.norm_stats = load_mil_norm_stats(fold_idx)
        probs, logits, batch_labels, batch_patient_ids = infer_mil_model(
            model,
            recording_dataset,
            patient_ids,
            patient_indices_map,
            patient_label_map,
            device,
        )
        fold_probs.append(probs)
        fold_logits.append(logits)
        if labels is None:
            labels = batch_labels
            out_patient_ids = batch_patient_ids
        print(f"  Fold {fold_idx + 1}: inference complete")
    return (
        np.mean(np.stack(fold_probs, axis=0), axis=0),
        np.mean(np.stack(fold_logits, axis=0), axis=0),
        labels,
        out_patient_ids,
    )


def summarize(labels, probs, threshold):
    preds = predict_with_unknown_threshold(probs, threshold)
    return {
        "threshold": threshold_for_json(threshold),
        "metrics": compute_physionet_wa(labels, preds),
        "classification_report": classification_report(
            labels,
            preds,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "bootstrap_ci": bootstrap_metric_ci(labels, preds),
    }


def main():
    set_seed(RANDOM_STATE)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device} | Patient-level MIL evaluation | experiment={cfg.MIL_EXPERIMENT_NAME}")

    dataset = HeartMurmurDataset(
        cfg.CSV_PATH,
        cfg.DATA_DIR,
        class_map=cfg.CLASS_LABELS,
        mode="val",
    )
    patients, patient_labels, patient_indices_map = get_patient_info(dataset)
    patient_label_map = dict(zip(patients, patient_labels))
    split = get_or_create_split(patients, patient_labels, patient_indices_map)

    print("\n[1/3] OOF validation predictions for threshold calibration")
    oof_probs, _, oof_labels, _ = collect_oof_predictions(
        dataset,
        split,
        patient_indices_map,
        patient_label_map,
        device,
    )
    oof_best = sweep_entropy_threshold(oof_probs, oof_labels)
    patient_threshold = oof_best["threshold"]
    print(
        f"  MIL patient threshold from OOF: {threshold_for_json(patient_threshold)} "
        f"(OOF WA={oof_best['metrics']['wa']:.4f})"
    )

    print("\n[2/3] Held-out test MIL ensemble inference")
    test_probs, test_logits, test_labels, test_patient_ids = infer_mil_ensemble(
        dataset,
        split["test_patients"],
        patient_indices_map,
        patient_label_map,
        device,
    )

    argmax = summarize(test_labels, test_probs, threshold=None)
    entropy = summarize(test_labels, test_probs, threshold=patient_threshold)

    print("\n[3/3] Final metrics")
    for name, block in [("MIL patient argmax", argmax), ("MIL patient OOF entropy", entropy)]:
        m = block["metrics"]
        print(
            f"  {name}: WA={m['wa']:.4f} | "
            f"Sens_P={m['sensitivity_present']:.3f} | "
            f"Spec_A={m['specificity_absent']:.3f} | "
            f"Rec_U={m['recall_unknown']:.3f} "
            f"(A/P/U={m['N_A']}/{m['N_P']}/{m['N_U']})"
        )

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(cfg.RESULTS_DIR, cfg.MIL_CONFUSION_MATRIX_FILENAME)
    plot_confusion_matrix(
        test_labels,
        test_probs,
        patient_threshold,
        "Patient-level MIL Confusion Matrix (OOF-calibrated Entropy)",
        cm_path,
    )

    out = {
        "model": "FGA_Patient_MIL",
        "experiment": cfg.MIL_EXPERIMENT_NAME,
        "base_recording_experiment": cfg.EXPERIMENT_NAME,
        "preprocessing": {
            "duration_sec": cfg.DURATION,
            "crop_mode_train": cfg.CROP_MODE_TRAIN,
            "crop_mode_eval": cfg.CROP_MODE_EVAL,
            "norm_crop_mode": cfg.NORM_CROP_MODE,
            "sample_rate": cfg.SAMPLE_RATE,
            "bandpass_low": cfg.BANDPASS_LOW,
            "bandpass_high": cfg.BANDPASS_HIGH,
            "n_fft": cfg.N_FFT,
            "hop_length": cfg.HOP_LENGTH,
            "n_mels": cfg.N_MELS,
            "fmin": cfg.FMIN,
            "fmax": cfg.FMAX,
            "normalization_mode": cfg.NORMALIZATION_MODE,
        },
        "mil": {
            "location_embedding": cfg.MIL_USE_LOCATION_EMBEDDING,
            "location_embed_dim": cfg.MIL_LOCATION_EMBED_DIM,
            "batch_size": cfg.MIL_BATCH_SIZE,
            "aggregation": "learned attention pooling over patient recordings",
        },
        "imbalance": {
            "sampler_mode": cfg.SAMPLER_MODE,
            "loss": cfg.LOSS_TYPE,
            "loss_weights": cfg.LOSS_WEIGHTS,
            "focal_gamma": cfg.FOCAL_GAMMA,
            "label_smoothing": cfg.LABEL_SMOOTHING,
        },
        "calibration": {
            "method": "OOF entropy threshold; no test labels used for threshold selection",
            "patient_threshold": threshold_for_json(patient_threshold),
            "patient_oof_metrics": oof_best["metrics"],
        },
        "patient_level": {
            "patient_ids": test_patient_ids,
            "argmax": argmax,
            "oof_entropy": entropy,
        },
    }

    json_path = os.path.join(cfg.RESULTS_DIR, cfg.MIL_RESULTS_FILENAME)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"  Saved: {json_path}")
    print(f"  Saved: {cm_path}")


if __name__ == "__main__":
    main()
