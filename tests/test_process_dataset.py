import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).resolve().parents[1]))
from extract_features import process_dataset


def test_process_dataset_handles_missing_metadata():
    data_dir = Path('data/raw/unit_test_missing')
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    signal = np.zeros((100, 3))
    np.savez(
        data_dir / 'sample1.npz',
        phone_signal=signal,
        timestamp='2024-01-01',
        weather='sunny',
        driver_id=1,
        vehicle_type='car',
        speed_bin='slow',
        road_type='urban',
        time_of_day='day',
        temperature=20,
        humidity=50,
        altitude=100,
        session_id=1,
        firmware_version='1.0',
        calibration_status='good',
        battery_level=80,
        network_type='4G',
        device_model='X'
    )  # intentionally missing gps_accuracy

    try:
        df = process_dataset(dataset_type='unit_test_missing')
        assert 'gps_accuracy' in df.columns
        assert pd.isna(df.loc[0, 'gps_accuracy'])
    finally:
        shutil.rmtree(data_dir)
        output_csv = Path('data/unit_test_missing.csv')
        if output_csv.exists():
            output_csv.unlink()

