"""Fresh adaptive ImputeFormer training entry point and shared sensor utilities.

The default workflow uses sampled variable-length windows and chronological
validation/test splits. Historical training helpers remain importable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imputeformer.models import ImputeFormerModel
from risk_pipeline.sensor_schema import OCEAN_SENSOR_IDS, OCEAN_SENSOR_UNITS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "data" / "ocean_adaptive_v2"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "sensor_imputeformer_production"
DEFAULT_MODEL_ROOT = ROOT / "models" / "sensor_imputeformer"
SENSOR_COLUMNS = OCEAN_SENSOR_IDS
SENSOR_UNITS = OCEAN_SENSOR_UNITS
TEMPORAL_FEATURE_NAMES = (
    "second_of_day_sin",
    "second_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "log_sampling_interval",
)
MASK_PATTERNS = (
    "point_20",
    "point_50",
    "point_80",
    "temporal_block",
    "single_sensor_outage",
    "multi_sensor_outage",
)
BASELINE_METHODS = (
    "linear_interpolation",
    "forward_backward_hold",
    "train_sensor_mean",
)
PRIMARY_GATE_PATTERN = "single_sensor_outage"
METRIC_NAMES = ("MAE", "MSE", "RMSE", "MRE")


def artifact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_dataset_run(dataset_root: Path) -> Path:
    runs_root = dataset_root / "runs"
    candidates = []
    if runs_root.is_dir():
        for path in sorted(runs_root.iterdir()):
            if not path.is_dir() or not (path / "dataset_manifest.json").is_file():
                continue
            split_dir = path / "splits"
            if all(
                (split_dir / f"{name}.parquet").is_file()
                or (split_dir / f"{name}.csv").is_file()
                for name in ("train", "validation", "test")
            ):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No complete ocean dataset run found under {runs_root}. Run "
            "experiments/download_ocean_sensor_dataset.py first."
        )
    return candidates[-1]


def _split_path(split_dir: Path, name: str) -> Path:
    parquet = split_dir / f"{name}.parquet"
    csv = split_dir / f"{name}.csv"
    if parquet.is_file():
        return parquet
    if csv.is_file():
        return csv
    raise FileNotFoundError(f"Missing split {name!r} under {split_dir}.")


def _read_split(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        if "timestamp" not in frame.columns and frame.index.name == "timestamp":
            frame = frame.reset_index()
    else:
        frame = pd.read_csv(path)
    expected = ["timestamp", *SENSOR_COLUMNS]
    if list(frame.columns) != expected:
        raise ValueError(
            f"{path} must contain exactly {expected}; got {list(frame.columns)}. "
            "Site, station, deployment, latitude, longitude, and coordinate fields "
            "are not model inputs."
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed", utc=True, errors="raise")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{path} contains duplicate timestamps.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{path} is not in chronological order.")
    frame = frame.set_index("timestamp")
    for sensor_id in SENSOR_COLUMNS:
        frame[sensor_id] = pd.to_numeric(frame[sensor_id], errors="coerce")
        if frame[sensor_id].notna().sum() == 0:
            raise ValueError(f"{path}: sensor {sensor_id} has no observations.")
    return frame


def _regular_interval_seconds(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        raise ValueError("At least two timestamps are required.")
    timestamps_ns = index.to_numpy(dtype="datetime64[ns]").astype(
        "int64", copy=False
    )
    deltas = np.diff(timestamps_ns)
    if (deltas <= 0).any() or not (deltas == deltas[0]).all():
        raise ValueError("Dataset splits must form one continuous regular grid.")
    return float(deltas[0] / 1_000_000_000.0)


def load_ocean_sensor_splits(
    dataset_root: Path,
    dataset_run: Path | None = None,
    require_regular: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object], Path]:
    """Load immutable chronological splits with the exact sensor-only schema."""
    run_dir = Path(dataset_run).resolve() if dataset_run else _latest_dataset_run(
        Path(dataset_root).resolve()
    )
    manifest_path = run_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("entity_type") != "sensor":
        raise ValueError("Dataset manifest must declare entity_type='sensor'.")
    if manifest.get("uses_site_entity") is not False:
        raise ValueError("Dataset manifest must explicitly disable site entities.")
    if manifest.get("uses_coordinates") is not False:
        raise ValueError("Dataset manifest must explicitly disable coordinates.")
    if manifest.get("model_columns") != ["timestamp", *SENSOR_COLUMNS]:
        raise ValueError("Dataset manifest model_columns do not match the five sensors.")

    split_dir = run_dir / "splits"
    splits = {
        name: _read_split(_split_path(split_dir, name))
        for name in ("train", "validation", "test")
    }
    if not (
        splits["train"].index[-1] < splits["validation"].index[0]
        and splits["validation"].index[-1] < splits["test"].index[0]
    ):
        raise ValueError("Train, validation, and test time ranges overlap.")
    combined = pd.concat(splits.values())
    if require_regular:
        interval_seconds = _regular_interval_seconds(combined.index)
        declared_interval = float(
            manifest.get("grid_audit", {}).get("grid_interval_seconds", interval_seconds)
        )
        if not math.isclose(interval_seconds, declared_interval, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Manifest and split sampling intervals do not match.")
    return splits, manifest, run_dir


def fit_normalizer(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    if not np.isfinite(mean).all():
        raise ValueError("Every sensor must have an observed training value.")
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype("float32"), std.astype("float32")


def normalize_and_fill(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    observed = np.isfinite(values)
    filled = np.empty(values.shape, dtype="float32")
    positions = np.arange(len(values), dtype="float64")
    for sensor_index in range(values.shape[1]):
        valid = observed[:, sensor_index]
        if not valid.any():
            raise ValueError("A split contains a sensor without any observation.")
        filled[:, sensor_index] = np.interp(
            positions, positions[valid], values[valid, sensor_index]
        ).astype("float32")
    normalized = (filled - mean[None, :]) / std[None, :]
    return normalized.astype("float32"), observed.astype(bool)


def time_features(
    index: pd.DatetimeIndex,
    sampling_interval_seconds: float | None = None,
    maximum_supported_seconds: float | None = None,
    encoding: str = "bounded_log_interval_v1",
) -> np.ndarray:
    if sampling_interval_seconds is None:
        sampling_interval_seconds = _regular_interval_seconds(index)
    if maximum_supported_seconds is None:
        maximum_supported_seconds = sampling_interval_seconds
    second_of_day = (
        index.hour.to_numpy() * 3600
        + index.minute.to_numpy() * 60
        + index.second.to_numpy()
        + index.microsecond.to_numpy() / 1_000_000.0
    )
    phase = 2.0 * np.pi * second_of_day / 86_400.0
    fractional_day_of_year = (
        index.dayofyear.to_numpy(dtype="float64") - 1.0
        + second_of_day / 86_400.0
    )
    seasonal_phase = 2.0 * np.pi * fractional_day_of_year / 365.2425
    if not np.isfinite(sampling_interval_seconds) or sampling_interval_seconds <= 0:
        raise ValueError("Sampling interval must be finite and positive.")
    if encoding == "bounded_log_interval_v1":
        log_interval = np.log1p(sampling_interval_seconds)
        encoded_interval = log_interval / (1.0 + log_interval)
    elif encoding == "legacy_log_ratio":
        encoded_interval = np.log1p(sampling_interval_seconds) / np.log1p(maximum_supported_seconds)
    else:
        raise ValueError(f"Unknown temporal encoding: {encoding}")
    interval_feature = np.full(
        len(index),
        encoded_interval,
        dtype="float64",
    )
    return np.stack(
        [
            np.sin(phase),
            np.cos(phase),
            np.sin(seasonal_phase),
            np.cos(seasonal_phase),
            interval_feature,
        ],
        axis=-1,
    ).astype("float32")


def sliding_windows(array: np.ndarray, window: int) -> np.ndarray:
    """Compatibility helper for small tests; production loaders are lazy."""
    if len(array) < window:
        raise ValueError(f"Need at least {window} rows, got {len(array)}.")
    windows = np.lib.stride_tricks.sliding_window_view(
        array, window_shape=window, axis=0
    )
    return np.moveaxis(windows, -1, 1).copy()


def window_starts(length: int, window: int, stride: int) -> List[int]:
    if length < window:
        raise ValueError(f"Need at least {window} rows, got {length}.")
    if not 1 <= stride <= window:
        raise ValueError("Window stride must be between 1 and window length.")
    starts = list(range(0, length - window + 1, stride))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


class SensorWindowDataset(Dataset):
    def __init__(self, values, temporal, observed, starts, window) -> None:
        self.values = values[:, :, None]
        self.temporal = temporal
        self.observed = observed[:, :, None]
        self.starts = tuple(int(value) for value in starts)
        self.window = int(window)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int):
        start = self.starts[item]
        stop = start + self.window
        return (
            torch.from_numpy(self.values[start:stop]),
            torch.from_numpy(self.temporal[start:stop]),
            torch.from_numpy(self.observed[start:stop]),
            start,
        )


def make_window_loader(
    values: np.ndarray,
    observed: np.ndarray,
    index: pd.DatetimeIndex,
    window: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    stride: int | None = None,
) -> DataLoader:
    interval_seconds = _regular_interval_seconds(index)
    starts = window_starts(len(values), window, stride or window)
    dataset = SensorWindowDataset(
        values,
        time_features(index, interval_seconds, interval_seconds),
        observed,
        starts,
        window,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _point_target(observed, rate, rng) -> np.ndarray:
    return observed & (rng.random(observed.shape) < rate)


def make_artificial_missing_mask(
    observed: torch.Tensor,
    rng: np.random.Generator,
    pattern: str = "mixed",
) -> torch.Tensor:
    """Create synthetic outages only at cells with measured ground truth."""
    observed_np = observed.cpu().numpy().astype(bool, copy=False)
    target = np.zeros_like(observed_np, dtype=bool)
    batch, steps, sensors, _ = observed_np.shape
    for sample in range(batch):
        selected_pattern = (
            str(rng.choice(MASK_PATTERNS)) if pattern == "mixed" else pattern
        )
        sample_observed = observed_np[sample]
        sample_target = target[sample]
        available_sensors = np.flatnonzero(
            sample_observed.any(axis=(0, 2))
        )
        if selected_pattern.startswith("point_"):
            rate = int(selected_pattern.split("_")[1]) / 100.0
            sample_target[:] = _point_target(sample_observed, rate, rng)
        elif selected_pattern == "temporal_block":
            shortest = max(1, min(steps, max(2, steps // 10)))
            longest = max(shortest, steps // 2)
            block_length = int(rng.integers(shortest, longest + 1))
            start = int(rng.integers(0, steps - block_length + 1))
            block_available = np.flatnonzero(
                sample_observed[start : start + block_length].any(axis=(0, 2))
            )
            if not len(block_available) and len(available_sensors):
                observed_time = int(
                    rng.choice(np.argwhere(sample_observed)[:, 0])
                )
                start = min(max(0, observed_time - block_length // 2), steps - block_length)
                block_available = np.flatnonzero(
                    sample_observed[start : start + block_length].any(axis=(0, 2))
                )
            if not len(block_available):
                continue
            chosen = rng.choice(
                block_available,
                size=int(rng.integers(1, len(block_available) + 1)),
                replace=False,
            )
            sample_target[start : start + block_length, chosen, :] = True
            sample_target &= sample_observed
        elif selected_pattern == "single_sensor_outage":
            if not len(available_sensors):
                continue
            chosen = int(rng.choice(available_sensors))
            sample_target[:, chosen, :] = sample_observed[:, chosen, :]
        elif selected_pattern == "multi_sensor_outage":
            if not len(available_sensors):
                continue
            chosen = rng.choice(
                available_sensors,
                size=min(
                    len(available_sensors),
                    max(2, math.ceil(len(available_sensors) * 0.4)),
                ),
                replace=False,
            )
            sample_target[:, chosen, :] = sample_observed[:, chosen, :]
        else:
            raise ValueError(f"Unknown masking pattern: {selected_pattern}")
        if not sample_target.any():
            candidates = np.argwhere(sample_observed)
            if len(candidates):
                selected = candidates[int(rng.integers(0, len(candidates)))]
                sample_target[tuple(selected)] = True
    return torch.from_numpy(target)


def masked_l1(prediction, target, mask) -> torch.Tensor:
    """Average per sensor first so the 1 Hz depth stream cannot dominate."""
    weight = mask.to(dtype=prediction.dtype)
    errors = torch.abs(prediction - target) * weight
    sensor_sums = errors.sum(dim=(0, 1, 3))
    sensor_counts = weight.sum(dim=(0, 1, 3))
    valid = sensor_counts > 0
    if not valid.any():
        return prediction.sum() * 0.0
    return (sensor_sums[valid] / sensor_counts[valid]).mean()


def frequency_imputation_loss(prediction, target, artificial_mask, alpha):
    if alpha <= 0:
        return prediction.new_zeros(())
    # FFT is linear: FFT(completed) - FFT(target) equals FFT(masked error).
    # Computing the equivalent expression directly avoids a redundant FFT.
    masked_error = torch.where(
        artificial_mask, prediction - target, torch.zeros_like(prediction)
    )
    error_spectrum = torch.fft.rfft(masked_error, dim=1)
    missing_rate = artificial_mask.float().mean().clamp(max=1.0 - 1e-6)
    dynamic_weight = alpha * missing_rate / (1.0 - missing_rate)
    return dynamic_weight * torch.abs(error_spectrum).mean()


def build_model(sensor_count: int, config: Mapping[str, object]) -> ImputeFormerModel:
    return ImputeFormerModel(
        num_nodes=sensor_count,
        input_dim=int(config["input_dim"]),
        output_dim=int(config["output_dim"]),
        input_embedding_dim=int(config["input_embedding_dim"]),
        learnable_embedding_dim=int(config["learnable_embedding_dim"]),
        feed_forward_dim=int(config["feed_forward_dim"]),
        num_temporal_heads=int(config["num_temporal_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        windows=int(config["window"]),
        dim_proj=int(config["dim_proj"]),
        variable_length=bool(config.get("variable_length", False)),
        use_interpolation_residual=bool(config.get("use_interpolation_residual", False)),
        time_aware=bool(config.get("time_aware", False)),
        static_sensor_embedding=bool(config.get("static_sensor_embedding", False)),
        visible_median_calibration=bool(config.get("visible_median_calibration", False)),
        no_context_prior=config.get("no_context_prior"),
    )


def _metrics_from_sums(absolute_error_sum, squared_error_sum, absolute_target_sum, count):
    if count <= 0:
        return {name: None for name in METRIC_NAMES}
    mse = squared_error_sum / count
    return {
        "MAE": absolute_error_sum / count,
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "MRE": absolute_error_sum / absolute_target_sum
        if absolute_target_sum > 1e-12 else None,
    }


def _empty_sums():
    return {"absolute": 0.0, "squared": 0.0, "target": 0.0, "count": 0}


def _accumulate(accumulator, error: np.ndarray, target: np.ndarray) -> None:
    accumulator["absolute"] += float(np.abs(error).sum(dtype="float64"))
    accumulator["squared"] += float(np.square(error).sum(dtype="float64"))
    accumulator["target"] += float(np.abs(target).sum(dtype="float64"))
    accumulator["count"] += int(error.size)


def _finish_sums(values):
    return _metrics_from_sums(
        float(values["absolute"]),
        float(values["squared"]),
        float(values["target"]),
        int(values["count"]),
    )


def _baseline_prediction(x: np.ndarray, visible: np.ndarray, method: str) -> np.ndarray:
    prediction = np.zeros_like(x, dtype="float32")
    positions = np.arange(x.shape[1], dtype="float64")
    if method == "train_sensor_mean":
        return prediction
    for sample in range(x.shape[0]):
        for sensor in range(x.shape[2]):
            valid = visible[sample, :, sensor, 0]
            if not valid.any():
                continue
            source = x[sample, :, sensor, 0]
            if method == "linear_interpolation":
                prediction[sample, :, sensor, 0] = np.interp(
                    positions, positions[valid], source[valid]
                )
            elif method == "forward_backward_hold":
                source_indices = np.where(valid, np.arange(len(valid)), -1)
                source_indices = np.maximum.accumulate(source_indices)
                source_indices[source_indices < 0] = int(np.flatnonzero(valid)[0])
                prediction[sample, :, sensor, 0] = source[source_indices]
            else:
                raise ValueError(f"Unknown baseline method: {method}")
    return prediction


def _method_metrics(overall, normalized_sensor, physical_sensor):
    normalized_per_sensor = {
        sensor_id: {
            "target_count": int(normalized_sensor[index]["count"]),
            **_finish_sums(normalized_sensor[index]),
        }
        for index, sensor_id in enumerate(SENSOR_COLUMNS)
    }
    valid = [item for item in normalized_per_sensor.values() if item["MAE"] is not None]
    macro_mse = float(np.mean([float(item["MSE"]) for item in valid]))
    macro_mre_values = [float(item["MRE"]) for item in valid if item["MRE"] is not None]
    macro = {
        "MAE": float(np.mean([float(item["MAE"]) for item in valid])),
        "MSE": macro_mse,
        "RMSE": math.sqrt(macro_mse),
        "MRE": float(np.mean(macro_mre_values)) if macro_mre_values else None,
    }
    return {
        "target_count": int(overall["count"]),
        "normalized_overall": _finish_sums(overall),
        "normalized_sensor_macro": macro,
        "normalized_per_sensor": normalized_per_sensor,
        "per_sensor_physical_units": {
            sensor_id: {
                "unit": SENSOR_UNITS[sensor_id],
                "target_count": int(physical_sensor[index]["count"]),
                **_finish_sums(physical_sensor[index]),
            }
            for index, sensor_id in enumerate(SENSOR_COLUMNS)
        },
    }


def _update_method_sums(store, prediction, target, artificial, mean, std):
    error = prediction - target
    _accumulate(store["overall"], error[artificial], target[artificial])
    for sensor_index in range(len(SENSOR_COLUMNS)):
        sensor_mask = artificial[:, :, sensor_index, :]
        normalized_error = error[:, :, sensor_index, :][sensor_mask]
        normalized_target = target[:, :, sensor_index, :][sensor_mask]
        _accumulate(
            store["normalized_sensor"][sensor_index],
            normalized_error,
            normalized_target,
        )
        _accumulate(
            store["physical_sensor"][sensor_index],
            normalized_error * float(std[sensor_index]),
            normalized_target * float(std[sensor_index]) + float(mean[sensor_index]),
        )


def evaluate_model(model, loader, device, seed, pattern) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    sensor_sums = [_empty_sums() for _ in SENSOR_COLUMNS]
    with torch.no_grad():
        for x, temporal, observed, _ in loader:
            artificial = make_artificial_missing_mask(observed, rng, pattern)
            visible = observed & ~artificial
            prediction = model(
                x.to(device, non_blocking=True),
                temporal.to(device, non_blocking=True),
                visible.to(device, non_blocking=True).float(),
            ).cpu().numpy()
            target = x.numpy()
            mask = artificial.numpy()
            error = prediction - target
            for sensor_index in range(len(SENSOR_COLUMNS)):
                sensor_mask = mask[:, :, sensor_index, :]
                _accumulate(
                    sensor_sums[sensor_index],
                    error[:, :, sensor_index, :][sensor_mask],
                    target[:, :, sensor_index, :][sensor_mask],
                )
    sensor_mae = [
        _finish_sums(values)["MAE"] for values in sensor_sums
        if int(values["count"]) > 0
    ]
    if not sensor_mae:
        raise RuntimeError("Validation masking produced no evaluable observations.")
    return float(np.mean(sensor_mae))


def apply_output_calibration(
    prediction: np.ndarray,
    calibration: Mapping[str, Mapping[str, object]] | None,
) -> np.ndarray:
    if not calibration:
        return prediction
    calibrated = np.array(prediction, dtype="float32", copy=True)
    if calibrated.ndim != 4 or calibrated.shape[2] != len(SENSOR_COLUMNS):
        raise ValueError("Calibration expects [batch, time, sensors, channels].")
    for sensor_index, sensor_id in enumerate(SENSOR_COLUMNS):
        if sensor_id not in calibration:
            raise KeyError(f"Output calibration is missing sensor {sensor_id}.")
        spec = calibration[sensor_id]
        scale = float(spec["scale"])
        bias = float(spec["bias"])
        if not math.isfinite(scale) or scale < 0.0 or not math.isfinite(bias):
            raise ValueError(f"Invalid output calibration for {sensor_id}.")
        calibrated[:, :, sensor_index, :] = (
            calibrated[:, :, sensor_index, :] * scale + bias
        )
    return calibrated


def fit_output_calibration(model, loader, device, seed, minimum_scale=0.10):
    """Fit per-sensor affine output calibration on validation outages only."""
    model.eval()
    rng = np.random.default_rng(seed)
    predictions = [[] for _ in SENSOR_COLUMNS]
    targets = [[] for _ in SENSOR_COLUMNS]
    with torch.no_grad():
        for x, temporal, observed, _ in loader:
            artificial = make_artificial_missing_mask(
                observed, rng, PRIMARY_GATE_PATTERN
            )
            visible = observed & ~artificial
            output = model(
                x.to(device, non_blocking=True),
                temporal.to(device, non_blocking=True),
                visible.to(device, non_blocking=True).float(),
            ).cpu().numpy()
            target = x.numpy()
            mask = artificial.numpy()
            for sensor_index in range(len(SENSOR_COLUMNS)):
                selected = mask[:, :, sensor_index, :]
                if selected.any():
                    predictions[sensor_index].append(
                        output[:, :, sensor_index, :][selected]
                    )
                    targets[sensor_index].append(
                        target[:, :, sensor_index, :][selected]
                    )

    calibration = {}
    for sensor_index, sensor_id in enumerate(SENSOR_COLUMNS):
        if not predictions[sensor_index]:
            raise RuntimeError(
                f"Validation calibration has no targets for {sensor_id}."
            )
        prediction = np.concatenate(predictions[sensor_index]).astype("float64")
        target = np.concatenate(targets[sensor_index]).astype("float64")
        baseline_mae = float(np.mean(np.abs(target)))
        baseline_rmse = float(np.sqrt(np.mean(np.square(target))))
        best = None
        for scale in np.linspace(minimum_scale, 1.50, 57):
            residual = target - scale * prediction
            for bias in (float(np.mean(residual)), float(np.median(residual)), 0.0):
                bias = float(np.clip(bias, -2.0, 2.0))
                error = scale * prediction + bias - target
                mae = float(np.mean(np.abs(error)))
                rmse = float(np.sqrt(np.mean(np.square(error))))
                score = mae / max(baseline_mae, 1e-12) + rmse / max(
                    baseline_rmse, 1e-12
                )
                candidate = (score, abs(scale - 1.0), scale, bias, mae, rmse)
                if best is None or candidate < best:
                    best = candidate
        _, _, scale, bias, calibrated_mae, calibrated_rmse = best
        raw_error = prediction - target
        calibration[sensor_id] = {
            "scale": float(scale),
            "bias": float(bias),
            "minimum_allowed_scale": float(minimum_scale),
            "validation_target_count": int(len(target)),
            "validation_raw_MAE": float(np.mean(np.abs(raw_error))),
            "validation_raw_RMSE": float(np.sqrt(np.mean(np.square(raw_error)))),
            "validation_train_mean_baseline_MAE": baseline_mae,
            "validation_train_mean_baseline_RMSE": baseline_rmse,
            "validation_calibrated_MAE": calibrated_mae,
            "validation_calibrated_RMSE": calibrated_rmse,
        }
    return calibration


def evaluate_against_baselines(
    model,
    loader,
    device,
    seed,
    pattern,
    mean,
    std,
    output_calibration=None,
):
    """Evaluate every method in one pass using exactly the same masks."""
    methods = ("imputeformer", *BASELINE_METHODS)
    stores = {
        method: {
            "overall": _empty_sums(),
            "normalized_sensor": [_empty_sums() for _ in SENSOR_COLUMNS],
            "physical_sensor": [_empty_sums() for _ in SENSOR_COLUMNS],
        }
        for method in methods
    }
    model.eval()
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for x, temporal, observed, _ in loader:
            artificial_tensor = make_artificial_missing_mask(observed, rng, pattern)
            visible_tensor = observed & ~artificial_tensor
            model_prediction = model(
                x.to(device, non_blocking=True),
                temporal.to(device, non_blocking=True),
                visible_tensor.to(device, non_blocking=True).float(),
            ).cpu().numpy()
            model_prediction = apply_output_calibration(
                model_prediction, output_calibration
            )
            target = x.numpy()
            artificial = artificial_tensor.numpy()
            visible = visible_tensor.numpy()
            predictions = {"imputeformer": model_prediction}
            predictions.update({
                method: _baseline_prediction(target, visible, method)
                for method in BASELINE_METHODS
            })
            for method, prediction in predictions.items():
                _update_method_sums(stores[method], prediction, target, artificial, mean, std)
    return {
        "pattern": pattern,
        "mask_seed": seed,
        "same_mask_for_all_methods": True,
        "methods": {
            method: _method_metrics(
                stores[method]["overall"],
                stores[method]["normalized_sensor"],
                stores[method]["physical_sensor"],
            )
            for method in methods
        },
    }


def train_with_validation(
    model, train_loader, validation_loader, device, args, history_path, best_state_path
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, args.patience // 4)
    )
    rng = np.random.default_rng(args.seed)
    best_state = None
    best_epoch = 0
    best_validation = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_sum = reconstruction_sum = spectral_sum = 0.0
        batches = 0
        for x, temporal, observed, _ in train_loader:
            artificial = make_artificial_missing_mask(
                observed, rng, args.training_mask_pattern
            )
            visible = observed & ~artificial
            x = x.to(device, non_blocking=True)
            temporal = temporal.to(device, non_blocking=True)
            artificial = artificial.to(device, non_blocking=True)
            visible = visible.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, temporal, visible.float())
            reconstruction = masked_l1(prediction, x, artificial)
            spectral = frequency_imputation_loss(prediction, x, artificial, args.fil_alpha)
            loss = reconstruction + spectral
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += float(loss.item())
            reconstruction_sum += float(reconstruction.item())
            spectral_sum += float(spectral.item())
            batches += 1
        validation = evaluate_model(
            model,
            validation_loader,
            device,
            args.seed + 100_000,
            args.validation_mask_pattern,
        )
        scheduler.step(validation)
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(batches, 1),
            "reconstruction_loss": reconstruction_sum / max(batches, 1),
            "frequency_loss": spectral_sum / max(batches, 1),
            "validation_normalized_MAE": validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"epoch={epoch:03d} loss={row['loss']:.6f} "
            f"val_MAE={validation:.6f} lr={row['learning_rate']:.2e}", flush=True
        )
        if validation < best_validation - args.min_delta:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            temporary = best_state_path.with_suffix(".tmp")
            torch.save(best_state, temporary)
            temporary.replace(best_state_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(
                f"early_stop epoch={epoch} best_epoch={best_epoch} "
                f"best_val_MAE={best_validation:.6f}", flush=True
            )
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def fit_final_model(model, full_loader, device, epochs, args, history_path):
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=args.lr * 0.1
    )
    rng = np.random.default_rng(args.seed + 1)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = reconstruction_sum = spectral_sum = 0.0
        batches = 0
        for x, temporal, observed, _ in full_loader:
            artificial = make_artificial_missing_mask(
                observed, rng, args.final_training_mask_pattern
            )
            visible = observed & ~artificial
            x = x.to(device, non_blocking=True)
            temporal = temporal.to(device, non_blocking=True)
            artificial = artificial.to(device, non_blocking=True)
            visible = visible.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, temporal, visible.float())
            reconstruction = masked_l1(prediction, x, artificial)
            spectral = frequency_imputation_loss(prediction, x, artificial, args.fil_alpha)
            loss = reconstruction + spectral
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item())
            reconstruction_sum += float(reconstruction.item())
            spectral_sum += float(spectral.item())
            batches += 1
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(batches, 1),
            "reconstruction_loss": reconstruction_sum / max(batches, 1),
            "frequency_loss": spectral_sum / max(batches, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"full_fit_epoch={epoch:03d} loss={row['loss']:.6f}", flush=True)
    return history


def infer_aligned_grid(
    model,
    loader,
    raw_values,
    mean,
    std,
    window,
    device,
    output_calibration=None,
):
    model.eval()
    sums = np.zeros(raw_values.shape, dtype="float64")
    counts = np.zeros(raw_values.shape, dtype="uint16")
    with torch.no_grad():
        for x, temporal, observed, batch_starts in loader:
            prediction = model(
                x.to(device, non_blocking=True),
                temporal.to(device, non_blocking=True),
                observed.to(device, non_blocking=True).float(),
            ).cpu().numpy()
            prediction = apply_output_calibration(
                prediction, output_calibration
            )[..., 0]
            prediction = prediction * std[None, None, :] + mean[None, None, :]
            for sample, start in enumerate(batch_starts.tolist()):
                sums[start : start + window] += prediction[sample]
                counts[start : start + window] += 1
    if not (counts > 0).all():
        raise RuntimeError("Aligned inference left uncovered timestamps.")
    result = np.where(np.isfinite(raw_values), raw_values, sums / counts)
    if not np.isfinite(result).all():
        raise RuntimeError("Aligned output still contains missing values.")
    return result


def _metrics_rows(results):
    rows = []
    for result in results:
        for method, metrics in result["methods"].items():
            for scope in ("normalized_overall", "normalized_sensor_macro"):
                rows.append({
                    "pattern": result["pattern"],
                    "method": method,
                    "scope": scope,
                    "sensor_id": "all",
                    "unit": "train_standard_deviation",
                    "target_count": metrics["target_count"],
                    **metrics[scope],
                })
            for sensor_id, values in metrics["per_sensor_physical_units"].items():
                rows.append({
                    "pattern": result["pattern"],
                    "method": method,
                    "scope": "physical_sensor",
                    "sensor_id": sensor_id,
                    "unit": values["unit"],
                    "target_count": values["target_count"],
                    **{name: values[name] for name in METRIC_NAMES},
                })
    return pd.DataFrame(rows)


def qualification_gate(results, minimum_relative_improvement):
    primary = next(item for item in results if item["pattern"] == PRIMARY_GATE_PATTERN)
    methods = primary["methods"]
    model = methods["imputeformer"]["normalized_sensor_macro"]
    best_mae_method = min(
        BASELINE_METHODS,
        key=lambda method: methods[method]["normalized_sensor_macro"]["MAE"],
    )
    best_rmse_method = min(
        BASELINE_METHODS,
        key=lambda method: methods[method]["normalized_sensor_macro"]["RMSE"],
    )
    best_mae = float(methods[best_mae_method]["normalized_sensor_macro"]["MAE"])
    best_rmse = float(methods[best_rmse_method]["normalized_sensor_macro"]["RMSE"])
    factor = 1.0 - minimum_relative_improvement
    passed_mae = float(model["MAE"]) < best_mae * factor
    passed_rmse = float(model["RMSE"]) < best_rmse * factor
    return {
        "passed": bool(passed_mae and passed_rmse),
        "primary_pattern": PRIMARY_GATE_PATTERN,
        "metric_scope": "normalized_sensor_macro",
        "minimum_relative_improvement": minimum_relative_improvement,
        "model_MAE": float(model["MAE"]),
        "best_baseline_MAE": best_mae,
        "best_baseline_MAE_method": best_mae_method,
        "model_RMSE": float(model["RMSE"]),
        "best_baseline_RMSE": best_rmse,
        "best_baseline_RMSE_method": best_rmse_method,
        "passed_MAE": passed_mae,
        "passed_RMSE": passed_rmse,
        "policy": (
            "Publication requires lower sensor-macro normalized MAE and RMSE "
            "than the strongest baseline for a complete sensor outage."
        ),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_saved_selection(
    model: torch.nn.Module,
    history_path: Path,
    best_state_path: Path,
) -> Tuple[torch.nn.Module, List[Dict[str, object]], int]:
    """Load a completed partial selection run without repeating its epochs."""
    if not history_path.is_file() or not best_state_path.is_file():
        raise FileNotFoundError(
            "A resumed selection run requires model_selection_history.csv and "
            "best_selection_state.pt."
        )
    history_frame = pd.read_csv(history_path)
    required = {"epoch", "validation_normalized_MAE"}
    if history_frame.empty or not required.issubset(history_frame.columns):
        raise ValueError("Saved model-selection history is empty or invalid.")
    validation = pd.to_numeric(
        history_frame["validation_normalized_MAE"], errors="raise"
    )
    if not np.isfinite(validation.to_numpy(dtype="float64")).all():
        raise ValueError("Saved validation history contains non-finite values.")
    best_row = int(validation.idxmin())
    best_epoch = int(history_frame.loc[best_row, "epoch"])
    try:
        state = torch.load(best_state_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(best_state_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    history = history_frame.to_dict(orient="records")
    return model, history, best_epoch


def run_legacy(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    splits, dataset_manifest, dataset_run = load_ocean_sensor_splits(
        Path(args.dataset_root), Path(args.dataset_run) if args.dataset_run else None
    )
    full_grid = pd.concat(splits.values())
    interval_seconds = _regular_interval_seconds(full_grid.index)
    if not math.isclose(interval_seconds, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"This checkpoint is trained for a 1-second grid, got {interval_seconds:g}s."
        )
    if min(map(len, splits.values())) < args.window:
        raise ValueError("Every split must contain at least one full model window.")

    output_root = Path(args.output_root).resolve()
    model_root = Path(args.model_root).resolve()
    if args.resume_selection_run:
        run_dir = Path(args.resume_selection_run).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Selection run does not exist: {run_dir}")
        if run_dir.parent != output_root / "runs":
            raise ValueError(
                "The resumed selection run must be under the configured "
                "output-root/runs directory."
            )
        model_run_dir = model_root / "runs" / run_dir.name
        if not model_run_dir.is_dir():
            raise FileNotFoundError(
                f"Selection model directory does not exist: {model_run_dir}"
            )
    else:
        run_id = artifact_timestamp()
        run_dir = output_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        model_run_dir = model_root / "runs" / run_id
        model_run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "model_selection_history.csv"
    final_history_path = run_dir / "final_full_dataset_history.csv"
    best_state_path = model_run_dir / "best_selection_state.pt"
    metrics_json_path = run_dir / "independent_test_metrics.json"
    metrics_csv_path = run_dir / "independent_test_metrics.csv"
    gate_path = run_dir / "production_qualification.json"
    aligned_path = run_dir / "aligned_ocean_five_sensor_timeseries.parquet"
    checkpoint_path = model_run_dir / "imputeformer_ocean_five_sensor.pt"
    summary_path = run_dir / "training_summary.json"
    if not args.resume_selection_run:
        _write_json(summary_path, {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dataset_run": str(dataset_run),
            "run_directory": str(run_dir),
        })
    else:
        protected = (
            final_history_path,
            metrics_json_path,
            metrics_csv_path,
            gate_path,
            aligned_path,
            checkpoint_path,
            model_run_dir / "qualified_selection_candidate.pt",
        )
        existing = [str(path) for path in protected if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite artifacts in a resumed run: "
                + ", ".join(existing)
            )

    train_raw = splits["train"].to_numpy(dtype="float64")
    mean, std = fit_normalizer(train_raw)
    prepared = {
        name: normalize_and_fill(frame.to_numpy(dtype="float64"), mean, std)
        for name, frame in splits.items()
    }
    train_loader = make_window_loader(
        *prepared["train"], splits["train"].index, args.window,
        args.batch_size, True, args.seed, args.training_stride
    )
    validation_loader = make_window_loader(
        *prepared["validation"], splits["validation"].index, args.window,
        args.batch_size, False, args.seed + 10, args.window
    )
    test_loader = make_window_loader(
        *prepared["test"], splits["test"].index, args.window,
        args.batch_size, False, args.seed + 20, args.window
    )

    model_config = {
        "window": args.window,
        "inference_stride": args.inference_stride,
        "num_nodes": len(SENSOR_COLUMNS),
        "input_dim": 1 + len(TEMPORAL_FEATURE_NAMES),
        "output_dim": 1,
        "input_embedding_dim": args.input_embedding_dim,
        "learnable_embedding_dim": args.learnable_embedding_dim,
        "feed_forward_dim": args.feed_forward_dim,
        "num_temporal_heads": args.num_temporal_heads,
        "num_layers": args.num_layers,
        "dim_proj": args.dim_proj,
        "dropout": args.dropout,
    }
    model = build_model(len(SENSOR_COLUMNS), model_config)
    initial_state_path = None
    if args.initial_selection_state:
        if args.resume_selection_run:
            raise ValueError(
                "initial-selection-state and resume-selection-run are mutually exclusive."
            )
        initial_state_path = Path(args.initial_selection_state).resolve()
        if not initial_state_path.is_file():
            raise FileNotFoundError(
                f"Initial selection state does not exist: {initial_state_path}"
            )
        try:
            initial_state = torch.load(
                initial_state_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            initial_state = torch.load(initial_state_path, map_location="cpu")
        model.load_state_dict(initial_state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device("cpu") if args.device == "auto" else torch.device(args.device)
    )
    print(
        f"device={device} parameters={parameter_count:,} "
        f"train_windows={len(train_loader.dataset):,}", flush=True
    )
    if args.selection_evaluation_only:
        if initial_state_path is None:
            raise ValueError(
                "selection-evaluation-only requires initial-selection-state."
            )
        model.to(device)
        validation = evaluate_model(
            model,
            validation_loader,
            device,
            args.seed + 100_000,
            args.validation_mask_pattern,
        )
        selection_history = [{
            "epoch": 0,
            "loss": None,
            "reconstruction_loss": None,
            "frequency_loss": None,
            "validation_normalized_MAE": validation,
            "learning_rate": 0.0,
            "selection_mode": "saved_state_evaluation_only",
        }]
        best_epoch = 0
        pd.DataFrame(selection_history).to_csv(history_path, index=False)
        torch.save(model.state_dict(), best_state_path)
        print(
            f"selection_evaluation_only val_MAE={validation:.6f}", flush=True
        )
    elif args.resume_selection_run:
        model, selection_history, best_epoch = load_saved_selection(
            model, history_path, best_state_path
        )
        model.to(device)
        print(
            f"resume_selection run={run_dir.name} "
            f"completed_epochs={len(selection_history)} best_epoch={best_epoch}",
            flush=True,
        )
    else:
        model, selection_history, best_epoch = train_with_validation(
            model, train_loader, validation_loader, device, args,
            history_path, best_state_path
        )

    output_calibration = None
    if args.calibrate_output:
        output_calibration = fit_output_calibration(
            model,
            validation_loader,
            device,
            args.seed + 300_000,
            args.minimum_calibration_scale,
        )
        calibration_path = run_dir / "validation_output_calibration.json"
        _write_json(calibration_path, output_calibration)
        print(
            "output_calibration="
            + ",".join(
                f"{sensor_id}:{spec['scale']:.3f}/{spec['bias']:.3f}"
                for sensor_id, spec in output_calibration.items()
            ),
            flush=True,
        )

    test_results = [
        evaluate_against_baselines(
            model, test_loader, device, args.seed + 200_000 + pattern_index,
            pattern, mean, std, output_calibration
        )
        for pattern_index, pattern in enumerate(MASK_PATTERNS)
    ]
    gate = qualification_gate(test_results, args.minimum_relative_improvement)
    metrics_document = {
        "evaluation_stage": "independent chronological test before full-data refit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_definitions": {
            "MAE": "sum(abs(prediction-target)) / count",
            "MSE": "sum((prediction-target)^2) / count",
            "RMSE": "sqrt(MSE)",
            "MRE": "sum(abs(prediction-target)) / sum(abs(target))",
        },
        "fairness": (
            "Model and baselines ran in one pass on identical deterministic masks "
            "and identical measured ground-truth cells."
        ),
        "results": test_results,
    }
    _write_json(metrics_json_path, metrics_document)
    _metrics_rows(test_results).to_csv(metrics_csv_path, index=False)
    _write_json(gate_path, gate)

    candidate_checkpoint = {
        "format_version": 3,
        "model_type": "ImputeFormerModel",
        "entity_type": "sensor",
        "uses_site_entity": False,
        "uses_coordinates": False,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model_config,
        "sensor_ids": list(SENSOR_COLUMNS),
        "sensor_units": SENSOR_UNITS,
        "temporal_feature_names": list(TEMPORAL_FEATURE_NAMES),
        "supported_sampling_interval_seconds": {
            "minimum": interval_seconds, "maximum": interval_seconds
        },
        "normalization": {"mean": mean, "std": std},
        "output_calibration": output_calibration,
        "qualification": gate,
        "held_out_test_metrics_before_full_refit": metrics_document,
    }
    torch.save(
        candidate_checkpoint,
        model_run_dir / "qualified_selection_candidate.pt",
    )
    if not gate["passed"]:
        summary = {
            "status": "rejected_by_baseline_gate",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "qualification": gate,
            "active_checkpoint_updated": False,
            "run_directory": str(run_dir),
        }
        _write_json(summary_path, summary)
        raise RuntimeError(
            "ImputeFormer did not beat the strongest baseline on both gate metrics; "
            f"artifacts were retained at {run_dir}, production was not updated."
        )

    raw_full_values = full_grid.to_numpy(dtype="float64")
    if args.final_finetune_epochs is None:
        full_mean, full_std = fit_normalizer(raw_full_values)
        final_normalization_source = "complete_timeline_refit"
    else:
        full_mean, full_std = mean.copy(), std.copy()
        final_normalization_source = "selection_training_split_for_warm_start"
    full_values, full_observed = normalize_and_fill(raw_full_values, full_mean, full_std)
    full_train_loader = make_window_loader(
        full_values, full_observed, full_grid.index, args.window,
        args.batch_size, True, args.seed + 1, args.training_stride
    )
    seed_everything(args.seed + 1)
    if args.final_finetune_epochs is None:
        final_model = build_model(len(SENSOR_COLUMNS), model_config)
        final_epochs = best_epoch
        final_initialization = "deterministic_random_refit"
    else:
        final_model = copy.deepcopy(model).to("cpu")
        final_epochs = args.final_finetune_epochs
        final_initialization = "qualified_selection_state_finetune"
    final_history = fit_final_model(
        final_model,
        full_train_loader,
        device,
        final_epochs,
        args,
        final_history_path,
    )
    full_inference_loader = make_window_loader(
        full_values, full_observed, full_grid.index, args.window,
        args.batch_size, False, args.seed + 2, args.inference_stride
    )
    aligned_values = infer_aligned_grid(
        final_model, full_inference_loader, raw_full_values,
        full_mean,
        full_std,
        args.window,
        device,
        output_calibration,
    )
    pd.DataFrame(
        aligned_values, index=full_grid.index, columns=SENSOR_COLUMNS
    ).to_parquet(aligned_path, index=True, compression="zstd")
    final_model.to("cpu")

    checkpoint = {
        **candidate_checkpoint,
        "model_state_dict": final_model.state_dict(),
        "normalization": {"mean": full_mean, "std": full_std},
        "training": {
            "seed": args.seed,
            "selected_epochs": best_epoch,
            "maximum_selection_epochs": args.max_epochs,
            "selection_epochs_completed": len(selection_history),
            "mask_patterns": list(MASK_PATTERNS),
            "fil_alpha": args.fil_alpha,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "training_window_stride": args.training_stride,
            "selection_training_mask_pattern": args.training_mask_pattern,
            "selection_validation_mask_pattern": args.validation_mask_pattern,
            "initial_selection_state": (
                str(initial_state_path) if initial_state_path else None
            ),
            "initial_selection_state_sha256": (
                sha256_file(initial_state_path) if initial_state_path else None
            ),
            "final_training_mask_pattern": args.final_training_mask_pattern,
            "final_training_epochs": final_epochs,
            "final_initialization": final_initialization,
            "final_normalization_source": final_normalization_source,
            "output_calibration_enabled": bool(output_calibration),
            "output_calibration_validation_pattern": (
                PRIMARY_GATE_PATTERN if output_calibration else None
            ),
            "complete_timeline_covered": True,
            "source_rows_truncated": False,
            "source_rows_sampled": False,
        },
        "data": {
            "dataset_run": str(dataset_run),
            "dataset_manifest": str(dataset_run / "dataset_manifest.json"),
            "dataset_manifest_sha256": sha256_file(dataset_run / "dataset_manifest.json"),
            "provider": dataset_manifest["provider"],
            "dataset_id": dataset_manifest["dataset_id"],
            "complete_grid_rows": len(full_grid),
            "sampling_interval_seconds": interval_seconds,
            "start": full_grid.index[0].isoformat(),
            "end": full_grid.index[-1].isoformat(),
            "sensor_count": len(SENSOR_COLUMNS),
            "source_rows_truncated": False,
            "source_rows_sampled": False,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    active_checkpoint = model_root / "imputeformer_ocean_five_sensor.pt"
    temporary_active = model_root / "imputeformer_ocean_five_sensor.pt.tmp"
    model_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_path, temporary_active)
    temporary_active.replace(active_checkpoint)

    summary = {
        "status": "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "entity_type": "sensor",
        "uses_site_entity": False,
        "uses_coordinates": False,
        "sensor_ids": list(SENSOR_COLUMNS),
        "sensor_count": len(SENSOR_COLUMNS),
        "parameter_count": parameter_count,
        "dataset_run": str(dataset_run),
        "dataset_split_rows": {name: len(frame) for name, frame in splits.items()},
        "complete_grid_rows": len(full_grid),
        "sampling_interval_seconds": interval_seconds,
        "selected_epochs": best_epoch,
        "final_training_epochs": final_epochs,
        "final_initialization": final_initialization,
        "final_normalization_source": final_normalization_source,
        "selection_epochs_completed": len(selection_history),
        "maximum_selection_epochs": args.max_epochs,
        "qualification": gate,
        "complete_timeline_covered": True,
        "source_rows_truncated": False,
        "source_rows_sampled": False,
        "active_checkpoint": str(active_checkpoint),
        "immutable_artifacts": {
            "run_directory": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "model_selection_history": str(history_path),
            "final_training_history": str(final_history_path),
            "independent_test_metrics_json": str(metrics_json_path),
            "independent_test_metrics_csv": str(metrics_csv_path),
            "production_qualification": str(gate_path),
            "aligned_time_series": str(aligned_path),
        },
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return active_checkpoint


def run(args: argparse.Namespace) -> Path:
    from train_time_aware_imputeformer import run as run_adaptive
    from calibrate_time_aware_baseline import calibrate_checkpoint
    trained = run_adaptive(args)
    checkpoint = torch.load(trained, map_location="cpu", weights_only=False)
    if torch.count_nonzero(checkpoint["model_state_dict"]["context_residual_gain"]).item():
        return Path(args.model_root).resolve() / "imputeformer_ocean_five_sensor.pt"
    return calibrate_checkpoint(
        trained, output_root=args.output_root, model_root=args.model_root,
        device=args.device, repeats=args.evaluation_repeats,
        batch_size=2, cpu_threads=args.cpu_threads,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fresh variable-length five-sensor ImputeFormer and report held-out baseline comparisons."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-run", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument(
        "--window", type=int, default=3600,
        help="Maximum training/inference context in rows; not an input-length requirement.",
    )
    parser.add_argument("--inference-stride", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--steps-per-epoch", type=int, default=128,
        help="Randomly sampled training batches per epoch; does not imply full-row coverage.",
    )
    parser.add_argument("--training-lengths", type=int, nargs="+", default=[100, 256, 500, 1000, 2000, 3600])
    parser.add_argument("--sampling-intervals-seconds", type=float, nargs="+", default=[1, 2, 5, 10, 30, 60, 300, 3600])
    parser.add_argument("--batch-row-budget", type=int, default=4096)
    parser.add_argument("--evaluation-repeats", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--input-embedding-dim", type=int, default=32)
    parser.add_argument("--learnable-embedding-dim", type=int, default=32)
    parser.add_argument("--feed-forward-dim", type=int, default=128)
    parser.add_argument("--num-temporal-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-proj", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    if args.window < 2:
        parser.error("window must be at least 2")
    if not 1 <= args.inference_stride <= args.window:
        parser.error("inference-stride must be between 1 and window")
    if args.max_epochs < 1 or args.min_epochs < 1:
        parser.error("epoch counts must be positive")
    if args.min_epochs > args.max_epochs:
        parser.error("min-epochs cannot exceed max-epochs")
    if args.patience < 1:
        parser.error("patience must be positive")
    for name in (
        "batch_size", "steps_per_epoch",
        "cpu_threads", "input_embedding_dim", "learnable_embedding_dim",
        "feed_forward_dim", "num_temporal_heads", "num_layers", "dim_proj",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if any(length < 1 or length > args.window for length in args.training_lengths):
        parser.error("training-lengths must be positive and no larger than window")
    if any(not math.isfinite(s) or s <= 0 for s in args.sampling_intervals_seconds):
        parser.error("sampling-intervals-seconds must be finite and positive")
    if args.batch_row_budget < 1 or args.evaluation_repeats < 1:
        parser.error("batch-row-budget and evaluation-repeats must be positive")
    if (args.input_embedding_dim + args.learnable_embedding_dim) % args.num_temporal_heads:
        parser.error("combined embedding dimensions must be divisible by num-temporal-heads")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        parser.error("grad-clip must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("min-delta must be finite and nonnegative")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    return args


if __name__ == "__main__":
    run(parse_args())
