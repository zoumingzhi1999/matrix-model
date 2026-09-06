from typing import List, Tuple

import numpy as np
import pandas as pd

from risk_pipeline.feature_schema import TIME_SERIES_FEATURE_COLUMNS



def sensor_columns(df: pd.DataFrame) -> List[str]:
    """Return the independent sensor-series columns in an aligned table."""
    if "timestamp" not in df.columns:
        raise ValueError("Aligned time-series data must contain a timestamp column.")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Sensor identifiers must be unique; duplicates: {duplicates}.")
    sensors = [str(column) for column in df.columns if column != "timestamp"]
    if not sensors:
        raise ValueError("Aligned time-series data has no sensor columns.")
    if any(not sensor for sensor in sensors):
        raise ValueError("Sensor identifiers must not be empty.")
    if any(sensor != sensor.strip() for sensor in sensors):
        raise ValueError("Sensor identifiers must not have leading or trailing whitespace.")
    return sensors


def extract_first_stage_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract the four selected statistical and spectral features per sensor.

    These are the retained first-stage features after ImputeFormer alignment.
    Each row represents one independent sensor time series.
    """
    sensors = sensor_columns(df)
    rows = []
    for sensor_id in sensors:
        values = df[sensor_id].to_numpy(dtype="float64")
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError(f"Sensor '{sensor_id}' has no finite observations.")
        values = np.where(finite, values, values[finite].mean())
        std = values.std()
        z = (values - values.mean()) / (std + 1e-6)
        row = {
            "sensor_id": sensor_id,
            "std": float(std),
            "min": float(values.min()),
            "energy": float(np.mean(values ** 2)),
            "fft_1": 0.0,
        }
        spectrum = np.abs(np.fft.rfft(z))
        if len(spectrum) > 1:
            row["fft_1"] = float(spectrum[1] / len(z))
        rows.append(row)
    features = pd.DataFrame(rows, columns=("sensor_id", *TIME_SERIES_FEATURE_COLUMNS))
    return features, sensors
