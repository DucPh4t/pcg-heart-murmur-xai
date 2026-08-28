import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import config as cfg
from dataset import HeartMurmurDataset, get_patient_info
from evaluate import load_fold_norm_stats
from experiment_utils import RANDOM_STATE, get_or_create_split, indices_for_patients, set_seed
from model import get_model
from utils import GradCAMPlusPlus

NUM_FOLDS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CLASS_NAMES = {0: "Absent", 1: "Present", 2: "Unknown"}

OUT_DIR = os.path.join(cfg.RESULTS_DIR, "XAI_Figures_3Class")
for c in CLASS_NAMES.values():
    os.makedirs(os.path.join(OUT_DIR, c), exist_ok=True)

class GradCAMPlusPlusEnsemble:
    def __init__(self, models, fold_stats):
        self.models = models
        self.fold_stats = fold_stats
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

def load_tsv_blocks(tsv_path, duration):
    """Load TSV cardiac-state annotations for overlay blocks."""
    if not os.path.exists(tsv_path): return []
    try: df = pd.read_csv(tsv_path, sep='\t', header=None, names=['start', 'end', 'state'])
    except: return []
    blocks = []
    for _, row in df.iterrows():
        s = max(0, row['start'])
        e = min(duration, row['end'])
        st = int(row['state'])
        if st in [1, 2, 3, 4]:
            blocks.append((s, e, st))
    return blocks

def plot_xai_figure(wav_path, spec, cam2d, gt_blocks, true_label, pred_label, rec_name, out_path):
    # Load raw audio and use the same duration as the model input.
    y, sr = librosa.load(wav_path, sr=None, duration=cfg.DURATION)
    time_audio = np.linspace(0, len(y)/sr, len(y))
    
    # spectrogram (1, H, W) -> numpy (H, W)
    spec_np = spec[0].cpu().numpy()
    
    # Resize cam2d up to spectrogram size
    cam_tensor = torch.tensor(cam2d).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(cam_tensor, size=spec_np.shape, mode='bilinear', align_corners=False).squeeze().numpy()
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2, 2]})
    
    time_axis = np.linspace(0, cfg.DURATION, spec_np.shape[1])
    freq_axis = np.linspace(cfg.FMIN, cfg.FMAX, spec_np.shape[0])
    colors = {1: 'blue', 2: 'red', 3: 'blue', 4: 'green'} # S1/S2: Blue, Sys: Red, Dia: Green
    
    # Set display limits for all axes
    DISPLAY_FMAX = cfg.FMAX
    
    # --- AXIS 0: Raw Audio Waveform ---
    ax0.plot(time_audio, y, color='black', linewidth=0.5)
    ax0.set_title(f"File: {rec_name} | GT: {CLASS_NAMES[true_label]} | Pred: {CLASS_NAMES[pred_label]}")
    ax0.set_ylabel("Amplitude")
    ax0.set_xlim(0, cfg.DURATION) # Force 10s crop
    
    y_min, y_max = ax0.get_ylim()
    for s, e, st in gt_blocks:
        rect = patches.Rectangle((s, y_min), e - s, y_max - y_min, 
                                 linewidth=0, edgecolor='none', facecolor=colors[st], alpha=0.15)
        ax0.add_patch(rect)
    
    # --- AXIS 1: Original Spectrogram ---
    im1 = ax1.pcolormesh(time_axis, freq_axis, spec_np, shading='auto', cmap='magma')
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim(cfg.FMIN, DISPLAY_FMAX)
    
    # --- AXIS 2: Attention Heatmap (Grad-CAM++) ---
    im2_bg = ax2.pcolormesh(time_axis, freq_axis, spec_np, shading='auto', cmap='gray') # BG
    im2 = ax2.pcolormesh(time_axis, freq_axis, cam_resized, shading='auto', cmap='jet', alpha=0.55) # Overlay
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_ylim(cfg.FMIN, DISPLAY_FMAX)
    
    # Add Heatmap Colorbar at the bottom
    cbar = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.18, aspect=40)
    cbar.set_label('Attention importance: low (blue) to high (red)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    set_seed(RANDOM_STATE)
    print(f"Device: {device} | Generating FGA 3-Class XAI Figures...")
    
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
    print("Loading folds...")
    for f in range(NUM_FOLDS):
        m = get_model(num_classes=3).to(device)
        m.load_state_dict(torch.load(cfg.get_checkpoint_path(f"{cfg.MODEL_PREFIX}_fold{f}.pth"), map_location=device))
        m.eval()
        models.append(m)
        fold_stats.append(load_fold_norm_stats(f))
        
    gc = GradCAMPlusPlusEnsemble(models, fold_stats)
    
    # Select cases to draw (avoiding drawing thousands of images)
    # Target: 10 Present, 10 Absent, 10 Unknown
    drawn_counts = {0: 0, 1: 0, 2: 0}
    max_per_class = 15
    
    np.random.shuffle(test_indices) # Shuffle to get varied cases
    
    for idx in tqdm(test_indices):
        if sum(drawn_counts.values()) >= 3 * max_per_class:
            break
            
        fpath, y_true, _ = ds.file_list[idx]
        if drawn_counts[y_true] >= max_per_class:
            continue
            
        rec_name = os.path.basename(fpath).replace('.wav', '')
        
        tsv_path = os.path.join(cfg.DATA_DIR, f"{rec_name}.tsv")
        gt_blocks = load_tsv_blocks(tsv_path, cfg.DURATION)
        
        # Require enough annotated signal within the 10-second model input.
        total_gt_time = sum([e - s for s, e, st in gt_blocks])
        if len(gt_blocks) < 4 or total_gt_time < 2.5:
            continue 
            
        logits, img = gc.predict_logits(ds, idx)
        pred = int(torch.argmax(logits, 1).item())
        # For XAI plotting, we want to draw cases where model makes the CORRECT prediction (or confident)
        # So we prefer files where pred == y_true
        if pred != y_true and drawn_counts[y_true] > 5:
            continue # Only take a few mistakes for error analysis if needed
            
        cam2d = gc.generate_cam(ds, idx, pred)
        
        cname = CLASS_NAMES[y_true]
        prefix = "Correct" if pred == y_true else f"Mispred_as_{CLASS_NAMES[pred]}"
        out_path = os.path.join(OUT_DIR, cname, f"{prefix}_{rec_name}.png")
        
        plot_xai_figure(fpath, img, cam2d, gt_blocks, y_true, pred, rec_name, out_path)
        drawn_counts[y_true] += 1
        
    print(f"\nAll figures successfully generated and saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
