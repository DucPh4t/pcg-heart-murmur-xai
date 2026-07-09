import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import config as cfg
from dataset import HeartMurmurDataset, get_patient_info
from evaluate import load_fold_norm_stats
from experiment_utils import RANDOM_STATE, get_or_create_split, indices_for_patients, set_seed
from model import get_model
from utils import GradCAMPlusPlus

NUM_FOLDS    = 5
THRESHOLDS = [80] # To speed up, just run top 20%
GT_CARDIAC_CONTEXT_STATES = {1, 2, 3} # S1, Systole, S2
GT_SYSTOLE_STATES = {2}
GT_VALID_STATES = {1, 2, 3, 4}
MIN_VALID_FRAMES = 50
FAITH_THRESH_PCT = 70

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class GradCAMPlusPlusEnsemble:
    def __init__(self, models, fold_stats):
        self.models = models
        self.fold_stats = fold_stats
        # Hook fga2 because it preserves more temporal resolution than deeper blocks.
        self.extractors = [GradCAMPlusPlus(m, m.fga2) for m in models]

    def _make_input(self, dataset, idx, fold_idx):
        old_stats = dataset.norm_stats
        dataset.norm_stats = self.fold_stats[fold_idx]
        try:
            img, _ = dataset[idx]
        finally:
            dataset.norm_stats = old_stats
        return img.unsqueeze(0).to(device), img

    def predict_logits(self, dataset, idx):
        fold_logits = []
        ref_img = None
        with torch.no_grad():
            for fold_idx, model in enumerate(self.models):
                x, img = self._make_input(dataset, idx, fold_idx)
                fold_logits.append(model(x))
                if ref_img is None:
                    ref_img = img
        return torch.mean(torch.stack(fold_logits), 0), ref_img

    def generate_cam(self, dataset, idx, cls):
        heatmaps = []
        for fold_idx, ext in enumerate(self.extractors):
            x, _ = self._make_input(dataset, idx, fold_idx)
            hm = ext(x, cls)
            heatmaps.append(hm)
        avg = np.mean(heatmaps, axis=0)
        mx = avg.max()
        if mx > 0:
            avg /= mx
        return avg

def load_tsv_masks(tsv_path, duration, positive_states, valid_states, fps=100):
    if not os.path.exists(tsv_path): return None, None
    try: df = pd.read_csv(tsv_path, sep='\t', header=None, names=['start', 'end', 'state'])
    except: return None, None

    n = int(duration * fps)
    gt_mask    = np.zeros(n, dtype=np.int32)
    valid_mask = np.zeros(n, dtype=np.int32)

    for _, row in df.iterrows():
        state = int(row['state'])
        s = max(0, int(row['start'] * fps))
        e = min(n, int(row['end']   * fps))
        if state in valid_states:   valid_mask[s:e] = 1
        if state in positive_states: gt_mask[s:e] = 1
    return gt_mask, valid_mask

def cam_to_1d(cam2d, n_frames):
    curve = np.mean(cam2d, axis=0) 
    mx = curve.max()
    if mx > 0: curve /= mx
    t = torch.tensor(curve, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=n_frames, mode='linear', align_corners=False)
    return out.squeeze().numpy()

def compute_overlap_metrics(cam_1d, gt_mask, valid_mask, thresh_pct):
    v_idx = np.where(valid_mask == 1)[0]
    if len(v_idx) == 0: return {'iou': 0.0, 'tor': 0.0}
    
    cam_valid = cam_1d[v_idx]
    gt_valid  = gt_mask[v_idx]
    
    thr = np.percentile(cam_valid, thresh_pct)
    cam_bin = (cam_valid >= thr).astype(np.int32)
    
    inter = np.logical_and(cam_bin, gt_valid).sum()
    union = np.logical_or(cam_bin,  gt_valid).sum()
    
    return {
        'iou': float(inter / union) if union > 0 else 0.0,
        'tor': float(inter / cam_bin.sum()) if cam_bin.sum() > 0 else 0.0
    }

def main():
    set_seed(RANDOM_STATE)
    print(f"Device: {device} | FGA 3-CLASS XAI EVALUATION")
    
    ds = HeartMurmurDataset(
        cfg.CSV_PATH,
        cfg.DATA_DIR,
        class_map=cfg.CLASS_LABELS,
        mode='val',
        crop_mode='start',
    )
    patients, labels, pmap = get_patient_info(ds)
    split = get_or_create_split(patients, labels, pmap)
    
    test_indices = indices_for_patients(split["test_patients"], pmap)
        
    models = []
    fold_stats = []
    for f in range(NUM_FOLDS):
        m = get_model(num_classes=3).to(device)
        m.load_state_dict(torch.load(os.path.join(cfg.PROJECT_ROOT, f"{cfg.MODEL_PREFIX}_fold{f}.pth"), map_location=device))
        m.eval()
        models.append(m)
        fold_stats.append(load_fold_norm_stats(f))
        
    gc = GradCAMPlusPlusEnsemble(models, fold_stats)
    
    fps = 100
    n_frames = int(cfg.DURATION * fps)
    results = []
    
    for idx in tqdm(test_indices, desc="Computing XAI"):
        fpath, y_true, _ = ds.file_list[idx]
        rec_name = os.path.basename(fpath).replace('.wav', '')
        
        tsv_path = os.path.join(cfg.DATA_DIR, f"{rec_name}.tsv")
        cardiac_mask, valid_mask = load_tsv_masks(
            tsv_path, cfg.DURATION, GT_CARDIAC_CONTEXT_STATES, GT_VALID_STATES, fps
        )
        systole_mask, _ = load_tsv_masks(
            tsv_path, cfg.DURATION, GT_SYSTOLE_STATES, GT_VALID_STATES, fps
        )
        if cardiac_mask is None or valid_mask.sum() < MIN_VALID_FRAMES: continue
            
        logits, _ = gc.predict_logits(ds, idx)
        pred = int(torch.argmax(logits, 1).item())
            
        cam2d = gc.generate_cam(ds, idx, pred)
        cam1d = cam_to_1d(cam2d, n_frames)
        
        cardiac_overlap = compute_overlap_metrics(cam1d, cardiac_mask, valid_mask, 80)
        systole_overlap = compute_overlap_metrics(cam1d, systole_mask, valid_mask, 80)
        results.append({
            'file': rec_name,
            'true_label': y_true,
            'pred': pred,
            'tor_cardiac_context': cardiac_overlap['tor'],
            'iou_cardiac_context': cardiac_overlap['iou'],
            'tor_systole_only': systole_overlap['tor'],
            'iou_systole_only': systole_overlap['iou'],
        })
        
    # --- REPORT ---
    print("\n" + "="*60)
    print("XAI EVALUATION: TEMPORAL OVERLAP RATIO (TOR)")
    print("Reports both cardiac-context overlap (S1/Systole/S2) and systole-only overlap.")
    print("="*60)
    
    label_names = {0: 'Absent', 1: 'Present', 2: 'Unknown'}
    for lbl in [np.int64(1), np.int64(0), np.int64(2)]:
        lbl_res = [r['tor_cardiac_context'] for r in results if r['true_label'] == lbl]
        sys_res = [r['tor_systole_only'] for r in results if r['true_label'] == lbl]
        if lbl_res:
            print(
                f"[{label_names.get(int(lbl))}] "
                f"TOR cardiac: {np.mean(lbl_res):.4f} +/- {np.std(lbl_res):.4f} | "
                f"TOR systole: {np.mean(sys_res):.4f} +/- {np.std(sys_res):.4f} "
                f"(N={len(lbl_res)})"
            )

    # Save machine-readable overlap metrics for later reporting.
    with open(os.path.join(cfg.PROJECT_ROOT, cfg.XAI_FILENAME), 'w') as f:
        json.dump([
            {
                'file': r['file'],
                'true_label': int(r['true_label']),
                'pred': int(r['pred']),
                'tor_cardiac_context': float(r['tor_cardiac_context']),
                'iou_cardiac_context': float(r['iou_cardiac_context']),
                'tor_systole_only': float(r['tor_systole_only']),
                'iou_systole_only': float(r['iou_systole_only']),
            }
            for r in results
        ], f, indent=2)

if __name__ == "__main__":
    main()
