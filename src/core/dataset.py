"""
dataset.py
==========
Dataset loading, audio preprocessing, mel-spectrogram extraction,
normalization, data augmentation, and MIL bag construction for heart murmur detection.
"""

import os
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

try:
    from . import config as config
    from .utils import butter_bandpass_filter
except (ImportError, ValueError):
    import config as config
    from utils import butter_bandpass_filter


class HeartMurmurDataset(Dataset):
    """
    Heart Murmur Dataset with proper augmentation handling.
    
    Augmentation is controlled by:
    1. mode='train' enables augmentation
    2. augment_prob controls how often augmentation is applied
    
    Oversampling is handled EXTERNALLY via WeightedRandomSampler
    """
    
    def __init__(self, csv_file, root_dir, class_map=config.CLASS_LABELS,
                 mode='train', augment_prob=0.5, norm_stats=None, crop_mode=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wav files.
            class_map (dict): Mapping from label string to integer.
            mode (string): 'train' or 'val'. Augmentation only in 'train'.
            augment_prob (float): Probability of applying augmentation (0-1).
        """
        self.mode = mode
        self.root_dir = root_dir
        self.class_map = class_map
        self.augment_prob = augment_prob if mode == 'train' else 0.0
        self.norm_stats = norm_stats
        if crop_mode is None:
            crop_mode = config.CROP_MODE_TRAIN if mode == 'train' else config.CROP_MODE_EVAL
        self.crop_mode = crop_mode
        
        # Load and process CSV
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['Murmur'].isin(self.class_map.keys())]
        
        # Build file list: (file_path, label, patient_id)
        self.file_list = []
        
        for idx, row in self.df.iterrows():
            patient_id = str(row['Patient ID'])
            locations = str(row['Recording locations:']).split('+')
            label_str = row['Murmur']
            label = self.class_map[label_str]
            
            for loc in locations:
                file_name = f"{patient_id}_{loc}.wav"
                file_path = os.path.join(self.root_dir, file_name)
                if os.path.exists(file_path):
                    self.file_list.append((file_path, label, patient_id))
        
        print(f"Dataset loaded with {len(self.file_list)} audio files (mode={mode}).")
        
        # Print class distribution
        labels = [item[1] for item in self.file_list]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Class distribution: {dict(zip(unique, counts))}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label, patient_id = self.file_list[idx]
        signal = self._load_signal(file_path)
        signal = self._crop_or_pad(signal, mode=self.crop_mode)

        apply_augment = (
            self.mode == 'train'
            and config.AUGMENTATION_ENABLED
            and np.random.rand() < self.augment_prob
        )
        if apply_augment:
            signal = self._apply_audio_augmentation(signal)

        log_melspec = self._signal_to_log_mel(signal)
        log_melspec = self._normalize(log_melspec)
        if apply_augment:
            log_melspec = self._apply_freq_masking(log_melspec)

        img_tensor = torch.tensor(log_melspec, dtype=torch.float32).unsqueeze(0)
        img_tensor = self._add_location_channels(img_tensor, file_path)
        return img_tensor, torch.tensor(label, dtype=torch.long)

    def _location_from_path(self, file_path):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        return stem.split("_")[-1]

    def _add_location_channels(self, img_tensor, file_path):
        if not getattr(config, "USE_LOCATION_CHANNELS", False):
            return img_tensor

        loc = self._location_from_path(file_path)
        h, w = img_tensor.shape[-2:]
        loc_channels = torch.zeros(
            config.NUM_LOCATION_CHANNELS,
            h,
            w,
            dtype=img_tensor.dtype,
        )
        loc_id = config.LOCATION_LABELS.get(loc)
        if loc_id is not None:
            loc_channels[loc_id].fill_(1.0)
        return torch.cat([img_tensor, loc_channels], dim=0)

    def _load_signal(self, file_path):
        try:
            signal, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
            return butter_bandpass_filter(
                signal,
                config.BANDPASS_LOW,
                config.BANDPASS_HIGH,
                config.SAMPLE_RATE,
                order=4,
            )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros(config.NUM_SAMPLES, dtype=np.float32)

    def _crop_or_pad(self, signal, mode='center', start=None):
        length = len(signal)
        if length < config.NUM_SAMPLES:
            padding = config.NUM_SAMPLES - length
            return np.pad(signal, (0, padding), 'constant')
        if length == config.NUM_SAMPLES:
            return signal

        max_start = length - config.NUM_SAMPLES
        if start is not None:
            start = int(np.clip(start, 0, max_start))
        elif mode == 'random':
            start = np.random.randint(0, max_start + 1)
        elif mode == 'center':
            start = max_start // 2
        else:
            start = 0
        return signal[start:start + config.NUM_SAMPLES]

    def _signal_to_log_mel(self, signal):
        melspec = librosa.feature.melspectrogram(
            y=signal,
            sr=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            fmin=config.FMIN,
            fmax=config.FMAX,
        )
        return librosa.power_to_db(melspec, ref=np.max)

    def _normalize(self, log_melspec):
        if self.norm_stats is None:
            mean = log_melspec.mean(axis=1, keepdims=True)
            std = log_melspec.std(axis=1, keepdims=True)
        else:
            mean = self.norm_stats['mean'][:, None]
            std = self.norm_stats['std'][:, None]
        log_melspec = (log_melspec - mean) / (std + 1e-6)
        return np.clip(log_melspec, -5.0, 5.0)

    def get_crops(self, idx, stride_sec=None):
        """Return all deterministic evaluation crops for one recording."""
        file_path, label, patient_id = self.file_list[idx]
        signal = self._load_signal(file_path)
        if len(signal) <= config.NUM_SAMPLES:
            starts = [0]
        else:
            stride = int((stride_sec or config.EVAL_CROP_STRIDE_SEC) * config.SAMPLE_RATE)
            stride = max(1, stride)
            last_start = len(signal) - config.NUM_SAMPLES
            starts = list(range(0, last_start + 1, stride))
            if starts[-1] != last_start:
                starts.append(last_start)

        tensors = []
        for start in starts:
            crop = self._crop_or_pad(signal, mode='start', start=start)
            spec = self._normalize(self._signal_to_log_mel(crop))
            img = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
            tensors.append(self._add_location_channels(img, file_path))
        return torch.stack(tensors), torch.tensor(label, dtype=torch.long)
    
    def _apply_audio_augmentation(self, signal):
        """Apply time shift and gaussian noise to audio signal."""
        # Time Shift
        if config.TIME_SHIFT_ENABLED and np.random.rand() < config.TIME_SHIFT_PROB:
            shift_samples = int(config.TIME_SHIFT_MAX_MS * config.SAMPLE_RATE / 1000)
            shift = np.random.randint(-shift_samples, shift_samples + 1)
            signal = np.roll(signal, shift)
        
        # Gaussian Noise
        if config.GAUSSIAN_NOISE_ENABLED and np.random.rand() < config.GAUSSIAN_NOISE_PROB:
            noise = np.random.normal(0, config.GAUSSIAN_NOISE_SCALE, signal.shape)
            signal = signal + noise
        
        return signal
    
    def _apply_freq_masking(self, spectrogram):
        """Apply frequency masking to spectrogram."""
        if not config.FREQ_MASK_ENABLED or np.random.rand() >= config.FREQ_MASK_PROB:
            return spectrogram
        
        n_mels = spectrogram.shape[0]
        f = np.random.randint(0, config.FREQ_MASK_PARAM + 1)
        f0 = np.random.randint(0, n_mels - f + 1)
        
        masked_spec = spectrogram.copy()
        masked_spec[f0:f0+f, :] = 0
        
        return masked_spec
    
    def get_labels(self):
        """Return all labels for creating samplers."""
        return [item[1] for item in self.file_list]
    
    def get_patient_ids(self):
        """Return all patient IDs."""
        return [item[2] for item in self.file_list]


def create_weighted_sampler(dataset, indices=None):
    """
    Create a WeightedRandomSampler to handle class imbalance.
    
    This replaces the pre-augmentation approach with proper oversampling
    that happens AFTER the train/val split.
    
    Args:
        dataset: HeartMurmurDataset instance
        indices: Optional list of indices (for Subset)
    
    Returns:
        WeightedRandomSampler for the DataLoader
    """
    if indices is None:
        labels = dataset.get_labels()
    else:
        all_labels = dataset.get_labels()
        labels = [all_labels[i] for i in indices]
    
    # Count classes
    class_counts = np.bincount(labels)
    
    # Weight per class (inverse frequency)
    class_weights = 1.0 / class_counts
    
    # Weight per sample
    sample_weights = [class_weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    
    return sampler


def location_id_from_path(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    loc = stem.split("_")[-1]
    return config.LOCATION_LABELS.get(loc, config.NUM_LOCATION_CHANNELS)


class PatientMurmurBagDataset(Dataset):
    """Patient-level dataset for multiple-instance learning.

    Each item is a variable-length bag of recordings from one patient. The
    recording_dataset controls preprocessing, crop mode, augmentation, and
    normalization, so MIL uses the same DSP pipeline as the D baseline.
    """

    def __init__(self, recording_dataset, patient_ids, patient_indices_map, patient_label_map):
        self.recording_dataset = recording_dataset
        self.patient_ids = list(patient_ids)
        self.patient_indices_map = patient_indices_map
        self.patient_label_map = patient_label_map

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        indices = self.patient_indices_map[patient_id]
        xs = []
        locations = []
        for rec_idx in indices:
            x, _ = self.recording_dataset[rec_idx]
            file_path = self.recording_dataset.file_list[rec_idx][0]
            xs.append(x)
            locations.append(location_id_from_path(file_path))

        return {
            "x": torch.stack(xs, dim=0),
            "locations": torch.tensor(locations, dtype=torch.long),
            "label": torch.tensor(self.patient_label_map[patient_id], dtype=torch.long),
            "patient_id": patient_id,
        }


def patient_bag_collate(batch):
    max_records = max(item["x"].shape[0] for item in batch)
    batch_size = len(batch)
    channels, height, width = batch[0]["x"].shape[1:]

    x = torch.zeros(batch_size, max_records, channels, height, width, dtype=batch[0]["x"].dtype)
    mask = torch.zeros(batch_size, max_records, dtype=torch.bool)
    pad_location = config.NUM_LOCATION_CHANNELS
    locations = torch.full((batch_size, max_records), pad_location, dtype=torch.long)
    labels = torch.empty(batch_size, dtype=torch.long)
    patient_ids = []

    for row, item in enumerate(batch):
        n = item["x"].shape[0]
        x[row, :n] = item["x"]
        mask[row, :n] = True
        locations[row, :n] = item["locations"]
        labels[row] = item["label"]
        patient_ids.append(item["patient_id"])

    return x, mask, locations, labels, patient_ids


def get_patient_info(dataset):
    """
    Extract patient-level information from dataset.
    
    Use majority vote over all recordings for a patient. If there is a tie,
    prefer clinically actionable murmur detection first: Present > Unknown > Absent.
    """
    patient_labels_all  = {}  # patient_id -> list of all labels
    patient_indices_map = {}
    
    for idx, (file_path, label, patient_id) in enumerate(dataset.file_list):
        if patient_id not in patient_labels_all:
            patient_labels_all[patient_id]  = []
            patient_indices_map[patient_id] = []
        patient_labels_all[patient_id].append(label)
        patient_indices_map[patient_id].append(idx)
    
    tie_priority = [1, 2, 0]  # Present > Unknown > Absent
    patient_label_map = {}
    for pid, lbls in patient_labels_all.items():
        label_counts = np.bincount(lbls, minlength=3)
        max_count = label_counts.max()
        tied = set(np.where(label_counts == max_count)[0].tolist())
        patient_label_map[pid] = next(lbl for lbl in tie_priority if lbl in tied)
    
    unique_patients = list(patient_label_map.keys())
    patient_labels  = [patient_label_map[p] for p in unique_patients]
    
    return unique_patients, patient_labels, patient_indices_map


def compute_normalization_stats(dataset, indices, save_path=None, crop_mode=None):
    """Estimate per-frequency mean/std from training recordings only."""
    crop_mode = crop_mode or config.NORM_CROP_MODE
    total_sum = None
    total_sq = None
    total_count = 0

    old_norm_stats = dataset.norm_stats
    old_crop_mode = dataset.crop_mode
    dataset.norm_stats = None
    dataset.crop_mode = crop_mode

    for idx in indices:
        file_path, _, _ = dataset.file_list[idx]
        signal = dataset._crop_or_pad(dataset._load_signal(file_path), mode=crop_mode)
        spec = dataset._signal_to_log_mel(signal)
        if total_sum is None:
            total_sum = np.zeros(spec.shape[0], dtype=np.float64)
            total_sq = np.zeros(spec.shape[0], dtype=np.float64)
        total_sum += spec.sum(axis=1)
        total_sq += np.square(spec).sum(axis=1)
        total_count += spec.shape[1]

    dataset.norm_stats = old_norm_stats
    dataset.crop_mode = old_crop_mode

    mean = total_sum / max(total_count, 1)
    var = total_sq / max(total_count, 1) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-6))
    stats = {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}

    if save_path is not None:
        np.savez(
            save_path,
            mean=stats['mean'],
            std=stats['std'],
            crop_mode=np.array(crop_mode),
            n_mels=np.array(config.N_MELS),
            fmax=np.array(config.FMAX),
        )
    return stats


def load_normalization_stats(path):
    data = np.load(path)
    mean = data['mean'].astype(np.float32)
    std = data['std'].astype(np.float32)
    if mean.shape[0] != config.N_MELS or std.shape[0] != config.N_MELS:
        raise ValueError(
            f"Normalization stats at {path} use shape {mean.shape}/{std.shape}, "
            f"but config.N_MELS={config.N_MELS}. Recompute stats by retraining."
        )
    return {'mean': mean, 'std': std}


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset, DataLoader
    
    print("Testing HeartMurmurDataset...")
    print("="*60)
    
    # Load dataset
    dataset = HeartMurmurDataset(
        config.CSV_PATH, 
        config.DATA_DIR, 
        mode='train',  # Enable augmentation
        augment_prob=0.5
    )
    
    # Get patient info
    unique_patients, patient_labels, patient_indices_map = get_patient_info(dataset)
    
    print(f"\nTotal patients: {len(unique_patients)}")
    print(f"Total files: {len(dataset)}")
    
    # Split patients (NOT files)
    train_patients, val_patients = train_test_split(
        unique_patients,
        test_size=0.2,
        stratify=patient_labels,
        random_state=42
    )
    
    print(f"\nTrain patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    
    # Get file indices for each split
    train_indices = []
    for pid in train_patients:
        train_indices.extend(patient_indices_map[pid])
    
    val_indices = []
    for pid in val_patients:
        val_indices.extend(patient_indices_map[pid])
    
    print(f"Train files: {len(train_indices)}")
    print(f"Val files: {len(val_indices)}")
    
    # Verify no leakage
    assert len(set(train_patients) & set(val_patients)) == 0, "Patient leakage!"
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    
    # Create sampler for class imbalance
    train_sampler = create_weighted_sampler(dataset, train_indices)
    
    # Create DataLoader with sampler (shuffle must be False when using sampler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=2
    )
    
    # Test a few batches
    print("\nTesting DataLoader...")
    total_0, total_1 = 0, 0
    for i, (images, labels) in enumerate(train_loader):
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        batch_dist = dict(zip(unique, counts))
        total_0 += batch_dist.get(0, 0)
        total_1 += batch_dist.get(1, 0)
        
        if i < 3:
            print(f"  Batch {i}: {dict(zip(unique, counts))}, shape: {images.shape}")
        if i >= 10:
            break
    
    print(f"\nFirst 10 batches - Class 0: {total_0}, Class 1: {total_1}")
    print("Ratio should be closer to 1:1 due to weighted sampling")
    
    # Test single item
    print("\n" + "="*60)
    print("Testing single item retrieval...")
    img, lbl = dataset[0]
    print(f"Spectrogram shape: {img.shape}")
    print(f"Label: {lbl}")
    
    # Show sample
    plt.figure(figsize=(10, 4))
    plt.imshow(img.squeeze().numpy(), aspect='auto', origin='lower', cmap='magma')
    plt.title(f"Log-Mel Spectrogram (Label: {lbl})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(config.PROJECT_ROOT, "sample_spec.png"))
    print("\nSaved sample_spec.png")
