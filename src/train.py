import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import config as cfg
from dataset import (
    HeartMurmurDataset,
    compute_normalization_stats,
    get_patient_info,
    load_normalization_stats,
)
from experiment_utils import (
    NUM_FOLDS,
    RANDOM_STATE,
    aggregate_patient_probs,
    compute_physionet_wa,
    get_or_create_split,
    indices_for_patients,
    seed_worker,
    set_seed,
)
from model import get_model

CV_FOLDS     = NUM_FOLDS

class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) with optional class weights.

    Formula: L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    alpha_t: class-specific weight.
    (1 - p_t)^gamma: down-weights easy examples.
    """
    def __init__(self, alpha=None, gamma=2.0, eps=1e-7, label_smoothing=0.1):
        super().__init__()
        if alpha is not None:
            self.alpha = alpha / alpha.sum()
        else:
            self.alpha = None
        self.gamma = gamma
        self.eps = eps
        self.label_smoothing = label_smoothing
        self.num_classes = 3

    def forward(self, logits, targets):
        # 1. Compute CE Loss with Label Smoothing natively using PyTorch
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction='none')
        
        # 2. Compute probability of the target class (p_t) for Focal weight
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p_t.clamp(min=self.eps, max=1.0)
        
        # 3. Compute Focal Weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # 4. Combine: Focal Weight * CE
        loss = focal_weight * ce_loss
        
        # 5. Apply class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        
        return loss.mean()

def create_sampler(labels):
    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=cfg.NUM_CLASSES)
    class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
    sample_weights = class_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler

def train_one_fold(fold_idx, train_idx, val_idx, train_ds_full, val_ds_full, device):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{CV_FOLDS}")
    print(f"{'='*60}")

    train_ds = Subset(train_ds_full, train_idx)
    val_ds   = Subset(val_ds_full,   val_idx)

    train_labels = [train_ds_full.file_list[i][1] for i in train_idx]
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    stats_path = cfg.get_checkpoint_path(f"{cfg.NORM_STATS_PREFIX}{fold_idx}.npz")
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

    sampler = create_sampler(train_labels) if cfg.SAMPLER_MODE == "weighted" else None
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg.NUM_WORKERS,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        worker_init_fn=seed_worker,
    )

    # ResNet18-FGA backbone with temporal attention pooling.
    model = get_model(num_classes=cfg.NUM_CLASSES).to(device)

    cw = None if cfg.LOSS_WEIGHTS is None else torch.FloatTensor(cfg.LOSS_WEIGHTS).to(device)
    loss_type = cfg.LOSS_TYPE.lower()
    if loss_type == "focal":
        criterion = FocalLoss(
            alpha=cw,
            gamma=cfg.FOCAL_GAMMA,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )
    elif loss_type == "ce":
        criterion = nn.CrossEntropyLoss(
            weight=cw,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )
    else:
        raise ValueError(f"Unsupported LOSS_TYPE={cfg.LOSS_TYPE!r}. Use 'focal' or 'ce'.")
    
    # AdamW with weight decay for regularization.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    # Cosine warm restarts are stepped per batch below.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_wa = 0.0
    best_metrics = None
    best_recording_metrics = None
    patience_ctr = 0
    PATIENCE     = 8
    val_patient_ids = [val_ds_full.file_list[i][2] for i in val_idx]

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        for i, (imgs, lbls) in enumerate(tqdm(train_loader, desc=f"Ep{epoch+1}/{cfg.NUM_EPOCHS} [Tr]", leave=False)):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            
            # Clip gradients to stabilize the attention modules.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            scheduler.step(epoch + i / len(train_loader))

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                logits = model(imgs.to(device))
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.extend(lbls.numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.asarray(all_labels, dtype=int)
        record_preds = np.argmax(all_probs, axis=1)
        record_m = compute_physionet_wa(all_labels, record_preds)

        _, patient_probs, patient_labels = aggregate_patient_probs(
            all_probs,
            all_labels,
            val_patient_ids,
        )
        patient_preds = np.argmax(patient_probs, axis=1)
        m = compute_physionet_wa(patient_labels, patient_preds)
        wa = m['wa']

        if wa > best_wa:
            best_wa = wa
            best_metrics = m
            best_recording_metrics = record_m
            patience_ctr = 0
            os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.MODEL_PREFIX}_fold{fold_idx}.pth"),
            )
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(
                f"  Ep{epoch+1}: Patient WA={wa:.4f} | "
                f"Sens_P={m['sensitivity_present']:.3f} | "
                f"Spec_A={m['specificity_absent']:.3f} | "
                f"Rec_U={m['recall_unknown']:.3f} | "
                f"Recording WA={record_m['wa']:.4f}"
            )

    print(f"\n  - Best patient-level WA = {best_wa:.4f}")
    return {
        'fold': fold_idx + 1,
        'wa': best_wa,
        'selection_level': 'patient',
        'recording_wa_at_best_patient_epoch': (
            best_recording_metrics['wa'] if best_recording_metrics is not None else None
        ),
        **best_metrics,
    }

def main():
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_ds = HeartMurmurDataset(cfg.CSV_PATH, cfg.DATA_DIR, class_map=cfg.CLASS_LABELS, mode='train', augment_prob=0.5)
    val_ds   = HeartMurmurDataset(cfg.CSV_PATH, cfg.DATA_DIR, class_map=cfg.CLASS_LABELS, mode='val')

    patients, pt_labels, pt_map = get_patient_info(val_ds)
    split = get_or_create_split(patients, pt_labels, pt_map)
    
    fold_results = []
    for fold_idx, fold in enumerate(split["folds"]):
        tr_idx = indices_for_patients(fold["train_patients"], pt_map)
        va_idx = indices_for_patients(fold["val_patients"], pt_map)

        res = train_one_fold(fold_idx, tr_idx, va_idx, train_ds, val_ds, device)
        fold_results.append(res['wa'])
        
    print(f"\n- Training FGA 3 Class Completed!")
    wa_arr = np.array(fold_results)
    print(f"- Cross-Validation patient-level WA = {np.mean(wa_arr):.4f} +/- {np.std(wa_arr):.4f}")
    print("- Run evaluate script next.")

if __name__ == "__main__":
    main()
