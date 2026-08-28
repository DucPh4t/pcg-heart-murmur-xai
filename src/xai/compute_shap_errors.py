"""
compute_shap_errors.py
======================
SHAP error analysis for FGA 3-class heart murmur model.
Computes GradientExplainer (Expected Gradients) SHAP values per-sample,
then aggregates into clinically-meaningful frequency bands.

Outputs (in SHAP_Figures_Errors/):
  1. shap_bar_plot.png         - Global abs. importance (% contribution)
  2. shap_signed_bar.png       - Signed importance: direction of evidence
  3. shap_beeswarm.png         - Per-sample distribution (Bee Swarm)
  4. Waterfalls/Absent/*.png   - Waterfall per incorrectly predicted Absent sample
  5. Waterfalls/Present/*.png  - Waterfall per incorrectly predicted Present sample
  6. Waterfalls/Unknown/*.png  - Waterfall per incorrectly predicted Unknown sample
"""
import json
import os, sys, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap
from tqdm import tqdm

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
CORE_DIR = os.path.join(SRC_DIR, "core")
BASELINE_DIR = os.path.join(SRC_DIR, "baseline")

for p in [SRC_DIR, CORE_DIR, BASELINE_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from core import config as cfg
    from core.dataset import HeartMurmurDataset, get_patient_info
    from core.experiment_utils import (
        RANDOM_STATE,
        aggregate_patient_probs,
        get_or_create_split,
        indices_for_patients,
        predict_with_unknown_threshold,
        set_seed,
    )
    from core.model import get_model
    from baseline.evaluate import infer_ensemble, load_fold_norm_stats
except (ImportError, ValueError):
    import config as cfg
    from dataset import HeartMurmurDataset, get_patient_info
    from experiment_utils import (
        RANDOM_STATE,
        aggregate_patient_probs,
        get_or_create_split,
        indices_for_patients,
        predict_with_unknown_threshold,
        set_seed,
    )
    from model import get_model
    from evaluate import infer_ensemble, load_fold_norm_stats

# Configuration
NUM_FOLDS       = 5
RANDOM_STATE    = 42
NUM_BACKGROUND  = 20     # Background samples for GradientExplainer
NSAMPLES        = 50     # SHAP integration steps (convergence verified at 50)
MAX_PER_CLASS   = 99999  # Evaluate all misclassified test samples
EXPLAIN_FOLD    = 0      # Keep SHAP in one fold-specific normalization space.

CLASS_NAMES = {0: "Absent", 1: "Present", 2: "Unknown"}


FREQ_BANDS = {
    'Heart Sounds (20-100 Hz)': (20, 100),
    'Low Murmur (100-300 Hz)':  (100, 300),
    'High Murmur (300-600 Hz)': (300, 600),
    'Upper (600-800 Hz)':       (600, 800),
}

OUT_DIR = os.path.join(cfg.RESULTS_DIR, "SHAP_Figures_Errors")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")  # SHAP hooks incompatible with MPS


def load_entropy_threshold(level="patient"):
    results_path = cfg.get_results_path(cfg.RESULTS_FILENAME)
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        results = json.load(f)
    key = "patient_threshold" if level == "patient" else "recording_threshold"
    return results.get("calibration", {}).get(key)


def load_patient_aggregation_config():
    results_path = cfg.get_results_path(cfg.RESULTS_FILENAME)
    if not os.path.exists(results_path):
        return "mean", load_entropy_threshold("patient")
    with open(results_path) as f:
        results = json.load(f)
    aggregation = results.get("patient_aggregation", {})
    method = aggregation.get("selected_by_oof", "mean")
    threshold = aggregation.get("selected_threshold")
    if threshold is None:
        threshold = results.get("calibration", {}).get("patient_threshold")
    return method, threshold


def choose_representative_recording(patient_id, patient_pred, patient_indices, index_to_prob):
    """Pick the recording that contributes most to the wrong patient-level decision."""
    candidates = patient_indices[patient_id]
    scored = []
    for idx in candidates:
        probs = index_to_prob[idx]
        rec_pred = int(np.argmax(probs))
        score = float(probs[patient_pred])
        scored.append((rec_pred == patient_pred, score, idx))
    scored.sort(reverse=True)
    return scored[0][2]


# Model wrapper
class EnsembleWrapper(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
    def forward(self, x):
        return torch.stack([m(x) for m in self.models], dim=0).mean(dim=0)


def disable_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False


# Frequency band aggregation
def aggregate_bands(shap_2d, freq_axis, use_abs=False):
    """Aggregate pixel SHAP (n_mels, T) into frequency-band scalars."""
    result = {}
    for name, (flo, fhi) in FREQ_BANDS.items():
        mask = (freq_axis >= flo) & (freq_axis < fhi)
        if mask.any():
            vals = shap_2d[mask, :]
            result[name] = float(np.mean(np.abs(vals)) if use_abs else np.mean(vals))
        else:
            result[name] = 0.0
    return result


# Plotting
def plot_bar_chart(class_imp):
    """Bar chart with % contribution per band per class."""
    band_names = list(FREQ_BANDS.keys())
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(band_names))
    width = 0.25
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    for ci in range(3):
        if not class_imp[ci]:
            continue
        # Average absolute importance per band
        raw = []
        for b in band_names:
            raw.append(np.mean([d[b] for d in class_imp[ci]]))
        raw = np.array(raw)
        # Convert to percentage
        total = raw.sum()
        pct = (raw / total * 100) if total > 0 else raw

        bars = ax.bar(x + ci * width, pct, width,
                      label=f"{CLASS_NAMES[ci]} (n={len(class_imp[ci])})",
                      color=colors[ci], alpha=0.85, edgecolor='white')
        for bar, p in zip(bars, pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{p:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Frequency Band', fontsize=13)
    ax.set_ylabel('Contribution (%)', fontsize=13)
    ax.set_title('SHAP Feature Importance by Frequency Band (% Contribution)',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(band_names, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(0, max(45, ax.get_ylim()[1] + 5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_bar_plot.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved shap_bar_plot.png")


def plot_signed_bar_chart(class_imp_signed):
    """Signed Bar chart: shows mean SHAP direction per band per class.
    Positive values support the predicted class.
    Negative values push probability away from the predicted class.
    This reveals the functional difference between Absent vs Present.
    """
    band_names = list(FREQ_BANDS.keys())
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(band_names))
    width = 0.25
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    for ci in range(3):
        if not class_imp_signed[ci]:
            continue
        # Average SIGNED SHAP per band (keeps +/- direction)
        raw = []
        for b in band_names:
            raw.append(np.mean([d[b] for d in class_imp_signed[ci]]))
        raw = np.array(raw)
        # Normalize by total absolute sum to get signed %
        total = np.abs(raw).sum()
        pct = (raw / total * 100) if total > 0 else raw

        bars = ax.bar(x + ci * width, pct, width,
                      label=f"{CLASS_NAMES[ci]} (n={len(class_imp_signed[ci])})",
                      color=colors[ci], alpha=0.85, edgecolor='white')
        for bar, p in zip(bars, pct):
            va = 'bottom' if p >= 0 else 'top'
            offset = 0.4 if p >= 0 else -0.4
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f'{p:+.1f}%', ha='center', va=va,
                    fontsize=8.5, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.9)
    ax.set_xlabel('Frequency Band', fontsize=13)
    ax.set_ylabel('Mean Signed SHAP Contribution (%)', fontsize=12)
    ax.set_title(
        'SHAP Error Analysis - Direction of Evidence for Incorrect Samples\n'
        '(+) = Supports wrong prediction   |   (-) = Contradicts wrong prediction',
        fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(band_names, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_signed_bar.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved shap_signed_bar.png")


def plot_beeswarm(shap_matrix, labels_list):
    """Bee swarm: one column per class, dots = samples."""
    band_names = list(FREQ_BANDS.keys())
    n_bands = len(band_names)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)

    for ci, ax in enumerate(axes):
        mask = np.array(labels_list) == ci
        n_samples = mask.sum()
        if n_samples == 0:
            ax.set_title(f"{CLASS_NAMES[ci]} (n=0)")
            continue

        cls_shap = shap_matrix[mask]  # (n, 4)
        # Normalize to % for readability
        row_totals = np.abs(cls_shap).sum(axis=1, keepdims=True)
        cls_pct = cls_shap / (row_totals + 1e-12) * 100

        for bi in range(n_bands):
            vals = cls_pct[:, bi]
            jitter = np.random.normal(0, 0.12, size=len(vals))
            y = np.full_like(vals, bi) + jitter
            # Color: red (+) blue (-)
            c = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
            ax.scatter(vals, y, c=c, s=40, alpha=0.7, edgecolors='white', linewidths=0.3)

        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4)
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels(band_names, fontsize=10)
        ax.set_xlabel("SHAP Contribution (%)", fontsize=10)
        ax.set_title(f"{CLASS_NAMES[ci]} (n={n_samples})", fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.15)

    # Legend
    legend_el = [Patch(facecolor='#e74c3c', label='Positive (+)'),
                 Patch(facecolor='#3498db', label='Negative (-)')]
    axes[2].legend(handles=legend_el, fontsize=9, loc='lower right')
    axes[0].set_ylabel("Frequency Band", fontsize=11)
    plt.suptitle("SHAP Bee Swarm - Per-sample Contribution Distribution",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_beeswarm.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  saved shap_beeswarm.png")


def plot_waterfall(bands_signed, true_label, pred_label, rec_name):
    """Waterfall for 1 sample: shows each band push/pull in %."""
    names = list(bands_signed.keys())
    vals = np.array(list(bands_signed.values()))
    total = np.abs(vals).sum()
    pct = vals / (total + 1e-12) * 100  # signed %

    # Sort by absolute
    order = np.argsort(np.abs(pct))
    names_s = [names[i] for i in order]
    pct_s = pct[order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(names_s))
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in pct_s]

    bars = ax.barh(y_pos, pct_s, color=colors, alpha=0.85, height=0.55,
                   edgecolor='white', linewidth=0.5)

    # Scale label offsets to the current axis range.
    x_max = np.max(np.abs(pct_s)) if len(pct_s) > 0 else 100
    offset = x_max * 0.05

    for bar, p in zip(bars, pct_s):
        if p > 0:
            x_txt = bar.get_width() + offset
            ha = 'left'
        else:
            x_txt = bar.get_width() - offset
            ha = 'right'

        ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                f'{p:+.1f}%', ha=ha, va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_s, fontsize=11)
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    # Expand limits so labels do not touch the frame.
    x_limits = ax.get_xlim()
    ax.set_xlim(min(x_limits[0], -x_max*0.2 - 5), max(x_limits[1], x_max + 15))
    
    ax.set_xlabel("SHAP Contribution (%)", fontsize=12)
    ax.set_title(f"SHAP Waterfall - {rec_name}\n"
                 f"Ground Truth: {CLASS_NAMES[true_label]}  |  "
                 f"Prediction: {CLASS_NAMES[pred_label]}",
                 fontsize=13, fontweight='bold')

    legend_el = [Patch(facecolor='#e74c3c', label='Pushes toward prediction (+)'),
                 Patch(facecolor='#3498db', label='Pushes away (-)')]
    ax.legend(handles=legend_el, fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.15)

    plt.tight_layout()
    cname = CLASS_NAMES[true_label]
    waterfall_dir = os.path.join(OUT_DIR, "Waterfalls", cname)
    os.makedirs(waterfall_dir, exist_ok=True)
    path = os.path.join(waterfall_dir, f"{rec_name}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


# Main
def main():
    set_seed(RANDOM_STATE)
    print(f"Device: {device}")

    # Data
    ds = HeartMurmurDataset(cfg.CSV_PATH, cfg.DATA_DIR,
                              class_map=cfg.CLASS_LABELS, mode='val')
    patients, labels, pmap = get_patient_info(ds)
    split = get_or_create_split(patients, labels, pmap)
    test_patients = split["test_patients"]

    test_indices = indices_for_patients(test_patients, pmap)
    test_index_set = set(test_indices)
    test_patient_indices = {
        p: [idx for idx in pmap[p] if idx in test_index_set]
        for p in test_patients
    }

    # Model
    # SHAP needs a single, fixed input normalization space. Explain one fold
    # consistently instead of mixing fold-specific normalizers inside SHAP.
    ds.norm_stats = load_fold_norm_stats(EXPLAIN_FOLD)
    models = []
    print(f"Loading fold {EXPLAIN_FOLD} model for SHAP error analysis...")
    m = get_model(num_classes=cfg.NUM_CLASSES).to(device)
    m.load_state_dict(torch.load(
        cfg.get_checkpoint_path(f"{cfg.MODEL_PREFIX}_fold{EXPLAIN_FOLD}.pth"),
        map_location=device))
    m.eval()
    models.append(m)

    ensemble = EnsembleWrapper(models).to(device)
    ensemble.eval()
    disable_inplace_relu(ensemble)

    # Background
    background_pool = indices_for_patients(split["trainval_patients"], pmap)
    np.random.seed(RANDOM_STATE)
    bg_idx = np.random.choice(background_pool, size=min(NUM_BACKGROUND, len(background_pool)),
                              replace=False)
    background = torch.stack([ds[i][0] for i in bg_idx]).to(device)
    print(f"Background: {background.shape}")

    explainer = shap.GradientExplainer(ensemble, background)
    freq_axis = np.linspace(cfg.FMIN, cfg.FMAX, cfg.N_MELS)

    # Step 1: Select patient-level incorrect cases, then one representative recording.
    print("\nSelecting patient-level incorrect cases and representative recordings...")
    patient_aggregation_method, patient_threshold = load_patient_aggregation_config()
    print(
        f"  Patient aggregation for SHAP error selection: "
        f"{patient_aggregation_method} | threshold={patient_threshold}"
    )
    test_probs, _, test_labels = infer_ensemble(ds, test_indices, device)
    test_patient_ids = [ds.file_list[i][2] for i in test_indices]
    patient_ids, patient_probs, patient_labels = aggregate_patient_probs(
        test_probs,
        test_labels,
        test_patient_ids,
        method=patient_aggregation_method,
    )
    patient_preds = predict_with_unknown_threshold(patient_probs, patient_threshold)
    index_to_prob = {idx: test_probs[pos] for pos, idx in enumerate(test_indices)}

    selected_per_class = {0: [], 1: [], 2: []}
    for pid, y_true, y_pred in zip(patient_ids, patient_labels, patient_preds):
        y_true = int(y_true)
        y_pred = int(y_pred)
        if y_true == y_pred or len(selected_per_class[y_true]) >= MAX_PER_CLASS:
            continue
        rep_idx = choose_representative_recording(
            pid, y_pred, test_patient_indices, index_to_prob
        )
        selected_per_class[y_true].append((rep_idx, y_pred, pid))

    for ci in range(3):
        print(f"  {CLASS_NAMES[ci]}: {len(selected_per_class[ci])} patient-level error cases selected")

    selected_rows = []
    for ci, cases in selected_per_class.items():
        for idx, pred, pid in cases:
            rec_name = os.path.basename(ds.file_list[idx][0]).replace(".wav", "")
            selected_rows.append({
                "patient_id": pid,
                "recording": rec_name,
                "true_label": CLASS_NAMES[ci],
                "pred_label": CLASS_NAMES[pred],
            })
    pd.DataFrame(selected_rows).to_csv(
        os.path.join(OUT_DIR, "shap_selected_patient_errors.csv"),
        index=False,
    )
    ds.norm_stats = load_fold_norm_stats(EXPLAIN_FOLD)

    # Step 2: Compute SHAP only for pre-filtered samples.
    class_imp_abs    = {0: [], 1: [], 2: []}
    class_imp_signed = {0: [], 1: [], 2: []}  # For signed bar chart
    all_shap_bands = []
    all_labels = []
    all_waterfalls = []

    for ci in range(3):
        cname = CLASS_NAMES[ci]
        cases = selected_per_class[ci]
        print(f"\nSHAP for {cname}: {len(cases)} patient-level representative errors")

        for idx, pred_label, patient_id in tqdm(cases, desc=f"SHAP {cname}"):
            img, _ = ds[idx]
            x = img.unsqueeze(0).to(device)

            # SHAP (NSAMPLES steps; convergence verified at 50 for band-aggregated analysis).
            with torch.autograd.set_detect_anomaly(False):
                shap_values = explainer.shap_values(x, nsamples=NSAMPLES)

            # Handle return format
            if isinstance(shap_values, np.ndarray):
                if shap_values.ndim >= 5 and shap_values.shape[-1] == 3:
                    sv_raw = shap_values[..., pred_label]
                else:
                    sv_raw = shap_values
            elif isinstance(shap_values, (list, tuple)) and len(shap_values) == cfg.NUM_CLASSES:
                sv_raw = shap_values[pred_label]
            elif isinstance(shap_values, list) and len(shap_values) == 1:
                sv_raw = shap_values[0]
            else:
                sv_raw = shap_values

            sv = np.squeeze(np.array(sv_raw))
            if sv.ndim == 3:
                if sv.shape[0] == img.shape[0]:
                    sv = sv[0]
                elif sv.shape[-1] == img.shape[0]:
                    sv = sv[..., 0]
            while sv.ndim > 2:
                # Collapse any extra batch or pooling dimensions.
                sv = sv.mean(axis=0)

            spec_shape = img[0].numpy().shape
            if sv.shape != spec_shape:
                if sv.shape == spec_shape[::-1]:
                    sv = sv.T
                else:
                    continue

            # Aggregate
            bands_abs    = aggregate_bands(sv, freq_axis, use_abs=True)
            bands_signed = aggregate_bands(sv, freq_axis, use_abs=False)

            class_imp_abs[ci].append(bands_abs)
            class_imp_signed[ci].append(bands_signed)
            all_shap_bands.append(list(bands_signed.values()))
            all_labels.append(ci)

            rec_name = os.path.basename(ds.file_list[idx][0]).replace('.wav', '')
            all_waterfalls.append((bands_signed, ci, pred_label, rec_name))

            print(f"  [{cname}] patient={patient_id} recording={rec_name} pred={CLASS_NAMES[pred_label]} done")

        print(f"  {cname}: {len(class_imp_abs[ci])} samples done")

    # Summary
    print(f"\n{'='*50}")
    for ci in range(3):
        print(f"  {CLASS_NAMES[ci]}: {len(class_imp_abs[ci])} samples")

    # Plots
    print("\nGenerating SHAP figures...")

    plot_bar_chart(class_imp_abs)
    plot_signed_bar_chart(class_imp_signed)

    if all_shap_bands:
        plot_beeswarm(np.array(all_shap_bands), all_labels)

    print(f"\nGenerating {len(all_waterfalls)} Waterfall figures into SHAP_Figures_3Class/Waterfalls/ ...")
    for bands_signed, true_label, pred_label, rec_name in tqdm(all_waterfalls, desc="Waterfalls"):
        plot_waterfall(bands_signed, true_label, pred_label, rec_name)

    print(f"\nAll done: {OUT_DIR}")


if __name__ == "__main__":
    main()
