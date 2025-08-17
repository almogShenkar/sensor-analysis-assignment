# extract_features.py.py

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')


def extract_features(imu_signal):
    """
    Extract features from a 100x3 IMU signal (X, Y, Z accelerometer)
    Returns a dictionary of ~30 features
    """
    features = {}

    # Separate channels
    x, y, z = imu_signal[:, 0], imu_signal[:, 1], imu_signal[:, 2]

    # Time domain features for each axis
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        features[f'{axis_name}_mean'] = np.mean(axis_data)
        features[f'{axis_name}_std'] = np.std(axis_data)
        features[f'{axis_name}_max'] = np.max(axis_data)
        features[f'{axis_name}_min'] = np.min(axis_data)
        features[f'{axis_name}_range'] = np.ptp(axis_data)
        features[f'{axis_name}_skew'] = stats.skew(axis_data)
        features[f'{axis_name}_kurtosis'] = stats.kurtosis(axis_data)

        # Peak detection
        peaks, _ = signal.find_peaks(axis_data, height=np.std(axis_data))
        features[f'{axis_name}_n_peaks'] = len(peaks)

        # Energy
        features[f'{axis_name}_energy'] = np.sum(axis_data ** 2)

    # Magnitude features
    magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    features['mag_mean'] = np.mean(magnitude)
    features['mag_std'] = np.std(magnitude)
    features['mag_max'] = np.max(magnitude)

    # Sudden change detection
    diff_mag = np.diff(magnitude)
    features['max_delta_mag'] = np.max(np.abs(diff_mag))
    features['sudden_change_score'] = np.sum(np.abs(diff_mag) > 2 * np.std(diff_mag))

    # Frequency domain features (simplified)
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        fft_vals = np.abs(fft(axis_data))[:50]  # First half of FFT
        features[f'{axis_name}_fft_max'] = np.max(fft_vals)
        features[f'{axis_name}_fft_mean'] = np.mean(fft_vals)

    # Cross-correlation features
    features['xy_corr'] = np.corrcoef(x, y)[0, 1]
    features['xz_corr'] = np.corrcoef(x, z)[0, 1]
    features['yz_corr'] = np.corrcoef(y, z)[0, 1]

    return features


def load_and_process_sample(sample_id):
    """
    Load sample data and determine which sensor to use.
    Pipeline prefers camera data when available, falls back to phone.
    """
    filepath = f"data/raw/{sample_id}.npz"

    # Load the sample file
    with np.load(filepath, allow_pickle=True) as data:
        # Load phone signal (always exists)
        phone_signal = data['phone_signal']

        # Check if camera signal exists in the file
        if 'camera_signal' in data.files:
            camera_signal = data['camera_signal']
        else:
            camera_signal = None

        # Determine which signal to use
        if camera_signal is not None:
            # Camera data available - use it
            signal_to_use = camera_signal
            actual_source = 'camera'
        else:
            # No camera data - fall back to phone
            signal_to_use = phone_signal
            actual_source = 'phone'

        # Get metadata
        metadata = {
            'driver_id': int(data['driver_id']),
            'vehicle_type': str(data['vehicle_type']),
            'weather': str(data['weather']),
            'road_type': str(data['road_type']),
            'speed_bin': str(data['speed_bin']),
            'session_id': int(data['session_id']),
            'actual_source': actual_source
        }

    return signal_to_use, metadata


def process_dataset(dataset_type='train'):
    """
    Process raw IMU data and create a feature dataset using optional external label CSV.
    Falls back to unlabeled feature extraction if labels are missing.
    """
    all_features = []
    source_stats = {'camera': 0, 'phone': 0}

    dataset_path = os.path.join("data", "raw", dataset_type)
    label_csv_path = os.path.join("data", "raw", f"{dataset_type}_labels.csv")

    # Try loading label CSV if available
    label_df = None
    if os.path.exists(label_csv_path):
        label_df = pd.read_csv(label_csv_path).set_index("sample_id")
    else:
        print(f"[INFO] No label CSV found for '{dataset_type}' – proceeding without labels.")

    sample_files = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]

    print(f"\nProcessing '{dataset_type}' dataset...")
    for file_name in tqdm(sample_files):
        sample_id = file_name.replace(".npz", "")
        filepath = os.path.join(dataset_path, file_name)

        with np.load(filepath, allow_pickle=True) as data:
            phone_signal = data['phone_signal']
            camera_signal = data['camera_signal'] if 'camera_signal' in data.files else None

            if camera_signal is not None:
                signal_to_use = camera_signal
                actual_source = 'camera'
            else:
                signal_to_use = phone_signal
                actual_source = 'phone'

            source_stats[actual_source] += 1

            features = extract_features(signal_to_use)
            features['sample_id'] = sample_id
            features['sensor_source'] = actual_source

            if label_df is not None and sample_id in label_df.index:
                features['label'] = label_df.loc[sample_id, 'label']

            metadata_keys = [
                'timestamp', 'weather', 'driver_id', 'vehicle_type', 'speed_bin', 'road_type',
                'time_of_day', 'temperature', 'humidity', 'altitude', 'session_id',
                'firmware_version', 'calibration_status', 'battery_level',
                'gps_accuracy', 'network_type', 'device_model'
            ]
            for key in metadata_keys:
                if key in data.files:
                    value = data[key]
                    if isinstance(value, np.ndarray):
                        try:
                            value = value.item()
                        except ValueError:
                            # Non-scalar arrays are unlikely in metadata but keep as-is
                            pass
                    features[key] = value
                else:
                    # Some data files may not include the full metadata set
                    features[key] = None

            if features['driver_id'] is not None:
                features['driver_id'] = f"D{int(features['driver_id'])}"
            if features['session_id'] is not None:
                features['session_id'] = f"S{int(features['session_id'])}"
            if features['timestamp'] is not None:
                features['timestamp'] = pd.to_datetime(features['timestamp'])

            all_features.append(features)

    df = pd.DataFrame(all_features)
    output_path = f"data/{dataset_type}.csv"
    df.to_csv(output_path, index=False)

    print(f"Processed {len(df)} samples to {output_path}")

    return df
