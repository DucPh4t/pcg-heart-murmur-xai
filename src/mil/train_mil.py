import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(SRC_DIR, "core")
BASELINE_DIR = os.path.join(SRC_DIR, "baseline")

for p in [SRC_DIR, CORE_DIR, BASELINE_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from core import config as cfg
    from core.dataset import (
        HeartMurmurDataset,
        PatientMurmurBagDataset,
        compute_normalization_stats,
        get_patient_info,
        load_normalization_stats,
        patient_bag_collate,
    )
    from core.experiment_utils import (
        NUM_FOLDS,
        RANDOM_STATE,
        compute_physionet_wa,
        get_or_create_split,
        indices_for_patients,
        seed_worker,
        set_seed,
    )
    from core.model import get_patient_mil_model
    from core.utils import FocalLoss
except (ImportError, ValueError):
    import config as cfg
    from dataset import (
        HeartMurmurDataset,
        PatientMurmurBagDataset,
        compute_normalization_stats,
        get_patient_info,
        load_normalization_stats,
        patient_bag_collate,
    )
    from experiment_utils import (
        NUM_FOLDS,
        RANDOM_STATE,
        compute_physionet_wa,
        get_or_create_split,
        indices_for_patients,
        seed_worker,
        set_seed,
    )
    from model import get_patient_mil_model
    from utils import FocalLoss


CV_FOLDS = NUM_FOLDS


def mil_norm_stats_path(fold_idx):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    return cfg.get_checkpoint_path(f"{cfg.MIL_NORM_STATS_PREFIX}{fold_idx}.npz")


def make_patient_loader(recording_dataset, patient_ids, patient_indices_map, patient_label_map, shuffle):
    ds = PatientMurmurBagDataset(
        recording_dataset,
        patient_ids,
        patient_indices_map,
        patient_label_map,
    )
    return DataLoader(
        ds,
        batch_size=cfg.MIL_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        worker_init_fn=seed_worker,
        collate_fn=patient_bag_collate,
    )


def train_one_fold(fold_idx, train_patients, val_patients, train_ds_full, val_ds_full, patient_indices_map, patient_label_map, device):
    print(f"\n{'='*60}")
    print(f"MIL FOLD {fold_idx + 1}/{CV_FOLDS}")
    print(f"{'='*60}")

    train_idx = indices_for_patients(train_patients, patient_indices_map)
    stats_path = mil_norm_stats_path(fold_idx)
    if os.path.exists(stats_path):
        norm_stats = load_normalization_stats(stats_path)
    else:
        print(
            f"  Computing train-only normalization stats: "
            f"fold {fold_idx}, crop={cfg.NORM_CROP_MODE}"
        )
        norm_stats = compute_normalization_stats(
            train_ds_full,
            train_idx,
            stats_path,
            crop_mode=cfg.NORM_CROP_MODE,
        )
    train_ds_full.norm_stats = norm_stats
    val_ds_full.norm_stats = norm_stats

    train_loader = make_patient_loader(
        train_ds_full,
        train_patients,
        patient_indices_map,
        patient_label_map,
        shuffle=True,
    )
    val_loader = make_patient_loader(
        val_ds_full,
        val_patients,
        patient_indices_map,
        patient_label_map,
        shuffle=False,
    )

    model = get_patient_mil_model(num_classes=cfg.NUM_CLASSES).to(device)
    cw = None if cfg.LOSS_WEIGHTS is None else torch.FloatTensor(cfg.LOSS_WEIGHTS).to(device)
    criterion = FocalLoss(
        alpha=cw,
        gamma=cfg.FOCAL_GAMMA,
        label_smoothing=cfg.LABEL_SMOOTHING,
    )
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_wa = 0.0
    best_metrics = None
    patience_ctr = 0
    patience = 8

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        for i, (bags, mask, locations, labels, _) in enumerate(
            tqdm(train_loader, desc=f"Ep{epoch+1}/{cfg.NUM_EPOCHS} [MIL-Tr]", leave=False)
        ):
            bags = bags.to(device)
            mask = mask.to(device)
            locations = locations.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(bags, mask, locations)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch + i / max(len(train_loader), 1))

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for bags, mask, locations, labels, _ in val_loader:
                logits = model(
                    bags.to(device),
                    mask.to(device),
                    locations.to(device),
                )
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.numpy().tolist())

        all_probs = np.vstack(all_probs)
        all_labels = np.asarray(all_labels, dtype=int)
        preds = np.argmax(all_probs, axis=1)
        metrics = compute_physionet_wa(all_labels, preds)
        wa = metrics["wa"]

        if wa > best_wa:
            best_wa = wa
            best_metrics = metrics
            patience_ctr = 0
            os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.MIL_MODEL_PREFIX}_fold{fold_idx}.pth"),
            )
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0:
            print(
                f"  Ep{epoch+1}: Patient WA={wa:.4f} | "
                f"Sens_P={metrics['sensitivity_present']:.3f} | "
                f"Spec_A={metrics['specificity_absent']:.3f} | "
                f"Rec_U={metrics['recall_unknown']:.3f}"
            )

        if patience_ctr >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    print(f"\n  - Best MIL patient-level WA = {best_wa:.4f}")
    return {
        "fold": fold_idx + 1,
        "wa": best_wa,
        "selection_level": "patient_mil",
        **best_metrics,
    }


def main():
    set_seed(RANDOM_STATE)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device} | Patient-level MIL training | experiment={cfg.MIL_EXPERIMENT_NAME}")
    print(
        f"MIL location embedding: {cfg.MIL_USE_LOCATION_EMBEDDING} | "
        f"batch={cfg.MIL_BATCH_SIZE}"
    )

    train_ds = HeartMurmurDataset(
        cfg.CSV_PATH,
        cfg.DATA_DIR,
        class_map=cfg.CLASS_LABELS,
        mode="train",
        augment_prob=0.5,
    )
    val_ds = HeartMurmurDataset(
        cfg.CSV_PATH,
        cfg.DATA_DIR,
        class_map=cfg.CLASS_LABELS,
        mode="val",
    )

    patients, patient_labels, patient_indices_map = get_patient_info(val_ds)
    patient_label_map = dict(zip(patients, patient_labels))
    split = get_or_create_split(patients, patient_labels, patient_indices_map)

    fold_results = []
    for fold_idx, fold in enumerate(split["folds"]):
        result = train_one_fold(
            fold_idx,
            fold["train_patients"],
            fold["val_patients"],
            train_ds,
            val_ds,
            patient_indices_map,
            patient_label_map,
            device,
        )
        fold_results.append(result["wa"])

    wa_arr = np.asarray(fold_results)
    print("\n- Patient MIL Training Completed!")
    print(f"- Cross-Validation patient-level WA = {np.mean(wa_arr):.4f} +/- {np.std(wa_arr):.4f}")
    print("- Run src/evaluate_mil.py next.")


if __name__ == "__main__":
    main()
