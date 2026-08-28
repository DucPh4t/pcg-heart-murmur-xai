import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(SRC_DIR, "core")

for p in [SRC_DIR, CORE_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from core import config as cfg
    from core.dataset import (
        HeartMurmurDataset,
        PatientMurmurBagDataset,
        get_patient_info,
        load_normalization_stats,
        patient_bag_collate,
    )
    from core.experiment_utils import NUM_FOLDS, RANDOM_STATE, get_or_create_split, set_seed
    from core.model import get_patient_mil_model
except (ImportError, ValueError):
    import config as cfg
    from dataset import (
        HeartMurmurDataset,
        PatientMurmurBagDataset,
        get_patient_info,
        load_normalization_stats,
        patient_bag_collate,
    )
    from experiment_utils import NUM_FOLDS, RANDOM_STATE, get_or_create_split, set_seed
    from model import get_patient_mil_model


CLASS_NAMES = {0: "Absent", 1: "Present", 2: "Unknown"}
LOCATIONS = list(cfg.LOCATION_LABELS.keys())


def mil_norm_stats_path(fold_idx):
    return cfg.get_checkpoint_path(f"{cfg.MIL_NORM_STATS_PREFIX}{fold_idx}.npz")


def load_mil_norm_stats(fold_idx):
    return load_normalization_stats(mil_norm_stats_path(fold_idx))


def load_mil_model(fold_idx, device):
    model_path = cfg.get_checkpoint_path(f"{cfg.MIL_MODEL_PREFIX}_fold{fold_idx}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing MIL checkpoint: {model_path}")
    model = get_patient_mil_model(num_classes=cfg.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def location_from_recording(recording_name):
    return recording_name.split("_")[-1]


def make_loader(recording_dataset, patient_ids, patient_indices_map, patient_label_map):
    bag_ds = PatientMurmurBagDataset(
        recording_dataset,
        patient_ids,
        patient_indices_map,
        patient_label_map,
    )
    return DataLoader(
        bag_ds,
        batch_size=cfg.MIL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=patient_bag_collate,
    )


def main():
    set_seed(RANDOM_STATE)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device} | Exporting MIL attention | experiment={cfg.MIL_EXPERIMENT_NAME}")

    dataset = HeartMurmurDataset(
        cfg.CSV_PATH,
        cfg.DATA_DIR,
        class_map=cfg.CLASS_LABELS,
        mode="val",
    )
    patients, patient_labels, patient_indices_map = get_patient_info(dataset)
    patient_label_map = dict(zip(patients, patient_labels))
    split = get_or_create_split(patients, patient_labels, patient_indices_map)
    test_patients = split["test_patients"]

    accum = {
        patient_id: {
            "probs": [],
            "attn": [],
            "label": patient_label_map[patient_id],
        }
        for patient_id in test_patients
    }

    for fold_idx in range(NUM_FOLDS):
        print(f"  Fold {fold_idx + 1}: attention inference")
        model = load_mil_model(fold_idx, device)
        dataset.norm_stats = load_mil_norm_stats(fold_idx)
        loader = make_loader(dataset, test_patients, patient_indices_map, patient_label_map)

        with torch.no_grad():
            for bags, mask, locations, labels, patient_ids in loader:
                logits, attn = model(
                    bags.to(device),
                    mask.to(device),
                    locations.to(device),
                    return_attention=True,
                )
                probs = F.softmax(logits, dim=1).cpu().numpy()
                attn = attn.cpu().numpy()
                mask_np = mask.numpy()

                for row, patient_id in enumerate(patient_ids):
                    n_valid = int(mask_np[row].sum())
                    accum[patient_id]["probs"].append(probs[row])
                    accum[patient_id]["attn"].append(attn[row, :n_valid])

    long_rows = []
    summary_rows = []
    for patient_id in test_patients:
        item = accum[patient_id]
        mean_probs = np.mean(np.vstack(item["probs"]), axis=0)
        pred = int(np.argmax(mean_probs))
        true = int(item["label"])

        attn_stack = np.vstack(item["attn"])
        mean_attn = np.mean(attn_stack, axis=0)
        rec_indices = patient_indices_map[patient_id]

        loc_weights = {loc: 0.0 for loc in LOCATIONS}
        rec_names = []
        for rec_idx, weight in zip(rec_indices, mean_attn):
            file_path = dataset.file_list[rec_idx][0]
            recording = os.path.splitext(os.path.basename(file_path))[0]
            location = location_from_recording(recording)
            rec_names.append(recording)
            loc_weights[location] = float(loc_weights.get(location, 0.0) + weight)
            long_rows.append({
                "patient_id": patient_id,
                "recording": recording,
                "location": location,
                "true_label": CLASS_NAMES[true],
                "pred_label": CLASS_NAMES[pred],
                "correct": int(true == pred),
                "attention_weight": float(weight),
                "prob_absent": float(mean_probs[0]),
                "prob_present": float(mean_probs[1]),
                "prob_unknown": float(mean_probs[2]),
            })

        top_location = max(loc_weights, key=loc_weights.get)
        row = {
            "patient_id": patient_id,
            "true_label": CLASS_NAMES[true],
            "pred_label": CLASS_NAMES[pred],
            "correct": int(true == pred),
            "top_location": top_location,
            "top_location_weight": float(loc_weights[top_location]),
            "prob_absent": float(mean_probs[0]),
            "prob_present": float(mean_probs[1]),
            "prob_unknown": float(mean_probs[2]),
            "recordings": "+".join(rec_names),
        }
        for loc in LOCATIONS:
            row[f"attention_{loc}"] = float(loc_weights.get(loc, 0.0))
        summary_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    long_path = os.path.join(
        cfg.RESULTS_DIR,
        f"mil_attention_recordings_{cfg.MIL_EXPERIMENT_NAME}.csv",
    )
    summary_path = os.path.join(
        cfg.RESULTS_DIR,
        f"mil_attention_patients_{cfg.MIL_EXPERIMENT_NAME}.csv",
    )
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"  Saved: {long_path}")
    print(f"  Saved: {summary_path}")

    print("\nTop-location counts by true label:")
    counts = summary_df.groupby(["true_label", "top_location"]).size().reset_index(name="count")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
