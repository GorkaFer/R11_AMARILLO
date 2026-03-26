import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import get_window, stft

COLUMNS = [
    'tachometer',
    'acc_under_axial',
    'acc_under_radiale',
    'acc_under_tangencial',
    'acc_over_axial',
    'acc_over_radiale',
    'acc_over_tangencial',
    'microphone'
]

FEATURE_CHANNELS = [
    'acc_under_axial',
    'acc_under_radiale',
    'acc_under_tangencial',
    'acc_over_axial',
    'acc_over_radiale',
    'acc_over_tangencial',
    'microphone'
]

FEATURE_CHANNEL_IDX = [COLUMNS.index(ch) for ch in FEATURE_CHANNELS]

FAULT_ROOTS = [
    'normal',
    'imbalance',
    'vertical-misalignment',
    'horizontal-misalignment',
    'underhang',
    'overhang'
]

BEARING_FAULTS = ['ball_fault', 'cage_fault', 'outer_race']

TIME_BASE_FEATS = ['mean', 'std', 'rms', 'peak_to_peak', 'max_abs', 'kurtosis']
SPECTRAL_BASE_FEATS = ['dom_freq', 'dom_mag', 'e_low', 'e_mid', 'e_high', 'spectral_centroid']
STFT_BASE_FEATS = ['stft_low_std', 'stft_mid_std', 'stft_high_std', 'stft_flux_mean', 'stft_dom_freq_std']


def load_csv_signal(csv_path):
    return np.loadtxt(csv_path, delimiter=',', dtype=np.float32)


def parse_labels_from_path(csv_path):
    path = Path(csv_path)
    parts = path.parts

    fault_root = None
    for p in parts:
        if p in FAULT_ROOTS:
            fault_root = p
            break

    if fault_root == 'normal':
        return {
            'is_anomaly': 0,
            'fault_type': 'normal',
            'severity': 'normal'
        }

    if fault_root in ['underhang', 'overhang']:
        bearing_fault = None
        for p in parts:
            if p in BEARING_FAULTS:
                bearing_fault = p
                break
        fault_type = f'{fault_root}_{bearing_fault}'
    else:
        fault_type = fault_root

    return {
        'is_anomaly': 1,
        'fault_type': fault_type,
        'severity': path.parent.name
    }


def build_metadata_table(root_dir):
    rows = []

    for file_id, csv_path in enumerate(sorted(Path(root_dir).rglob('*.csv'))):
        labels = parse_labels_from_path(csv_path)

        rows.append({
            'file_id': file_id,
            'file_path': str(csv_path),
            'is_anomaly': labels['is_anomaly'],
            'fault_type': labels['fault_type'],
            'severity': labels['severity']
        })

    return pd.DataFrame(rows)


def get_feature_names(channel_names, base_feats):
    names = []
    for ch in channel_names:
        for feat in base_feats:
            names.append(f'{ch}_{feat}')
    return names


def myfft(fs, x):
    window = get_window('hamming', len(x), fftbins=True)
    xw = x * window
    mag = np.abs(np.fft.rfft(xw)) / len(x)
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    return mag, freqs


def band_energy(freqs, mag, f_low, f_high):
    mask = (freqs >= f_low) & (freqs < f_high)
    return np.sum(mag[mask]**2)


def time_features(x):
    x = np.asarray(x, dtype=np.float32)

    mean_val = np.mean(x)
    std_val = np.std(x)
    rms_val = np.sqrt(np.mean(x**2))
    peak_to_peak_val = np.ptp(x)
    max_abs_val = np.max(np.abs(x))

    if std_val == 0:
        kurtosis_val = 0.0
    else:
        z = (x - mean_val) / std_val
        kurtosis_val = np.mean(z**4)

    return np.array([
        mean_val,
        std_val,
        rms_val,
        peak_to_peak_val,
        max_abs_val,
        kurtosis_val
    ], dtype=np.float32)


def get_time_feature_names(channel_names):
    return get_feature_names(channel_names, TIME_BASE_FEATS)


def spectral_features(x, fs, bands):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)

    mag, freqs = myfft(fs, x)

    mag_no_dc = mag.copy()
    mag_no_dc[0] = 0
    idx_peak = np.argmax(mag_no_dc)

    total_mag = np.sum(mag)
    if total_mag == 0:
        centroid = 0.0
    else:
        centroid = np.sum(freqs * mag) / total_mag

    return np.array([
        freqs[idx_peak],
        mag[idx_peak],
        band_energy(freqs, mag, bands[0][0], bands[0][1]),
        band_energy(freqs, mag, bands[1][0], bands[1][1]),
        band_energy(freqs, mag, bands[2][0], bands[2][1]),
        centroid
    ], dtype=np.float32)


def get_spectral_feature_names(channel_names):
    return get_feature_names(channel_names, SPECTRAL_BASE_FEATS)


def stft_band_energy_per_frame(freqs, mag, f_low, f_high):
    mask = (freqs >= f_low) & (freqs < f_high)
    return np.sum(mag[mask, :]**2, axis=0)


def spectral_flux(mag):
    if mag.shape[1] < 2:
        return 0.0

    diff = np.diff(mag, axis=1)
    flux = np.sqrt(np.sum(diff**2, axis=0))
    return np.mean(flux)


def stft_features(x, fs, bands, nperseg=512, noverlap=256):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)

    freqs, _, zxx = stft(
        x,
        fs=fs,
        window='hamming',
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False
    )

    mag = np.abs(zxx)

    low_std = np.std(stft_band_energy_per_frame(freqs, mag, bands[0][0], bands[0][1]))
    mid_std = np.std(stft_band_energy_per_frame(freqs, mag, bands[1][0], bands[1][1]))
    high_std = np.std(stft_band_energy_per_frame(freqs, mag, bands[2][0], bands[2][1]))
    flux_mean = spectral_flux(mag)

    mag_no_dc = mag.copy()
    mag_no_dc[0, :] = 0
    dom_freq_std = np.std(freqs[np.argmax(mag_no_dc, axis=0)])

    return np.array([
        low_std,
        mid_std,
        high_std,
        flux_mean,
        dom_freq_std
    ], dtype=np.float32)


def get_stft_feature_names(channel_names):
    return get_feature_names(channel_names, STFT_BASE_FEATS)


def multichannel_features(window_array, feature_func, **kwargs):
    feat_vector = []

    for ch_idx in FEATURE_CHANNEL_IDX:
        x = window_array[:, ch_idx]
        feats = feature_func(x, **kwargs)
        feat_vector.extend(feats)

    return np.array(feat_vector, dtype=np.float32)


def process_csv_signal(meta_row, window_size, hop_size, feature_func, **kwargs):
    x = load_csv_signal(meta_row.file_path)

    rows = []

    for start in range(0, x.shape[0] - window_size + 1, hop_size):
        window = x[start:start + window_size]

        feat_vector = multichannel_features(
            window_array=window,
            feature_func=feature_func,
            **kwargs
        )

        row = [
            meta_row.file_id,
            meta_row.is_anomaly,
            meta_row.fault_type,
            meta_row.severity
        ] + feat_vector.tolist()

        rows.append(row)

    return rows


def build_feature_dataset(metadata, window_size, hop_size, feature_func, feature_names_func, **kwargs):
    feat_names = feature_names_func(FEATURE_CHANNELS)
    columns = ['file_id', 'is_anomaly', 'fault_type', 'severity'] + feat_names

    all_rows = []

    for meta_row in metadata.itertuples(index=False):
        rows = process_csv_signal(
            meta_row=meta_row,
            window_size=window_size,
            hop_size=hop_size,
            feature_func=feature_func,
            **kwargs
        )
        all_rows.extend(rows)

    dataset_df = pd.DataFrame(all_rows, columns=columns)
    return dataset_df




