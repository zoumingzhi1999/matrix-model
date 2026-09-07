"""Production ImputeFormer inference for the five mobile-cable sensors."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from imputeformer.models import ImputeFormerModel

from .sensor_schema import OCEAN_SENSOR_IDS
from .temporal_context import RAW_TEMPORAL_FEATURE_NAMES, raw_temporal_features, timestamp_nanoseconds


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SENSOR_CHECKPOINT = (
    ROOT
    / "models"
    / "sensor_imputeformer"
    / "imputeformer_ocean_five_sensor.pt"
)
EXPECTED_SENSOR_IDS = OCEAN_SENSOR_IDS
LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_model(config: Mapping[str, object]) -> ImputeFormerModel:
    return ImputeFormerModel(
        num_nodes=int(config["num_nodes"]),
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


def load_sensor_imputeformer_checkpoint(
    path: str | Path = DEFAULT_SENSOR_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> tuple[ImputeFormerModel, dict]:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained sensor ImputeFormer checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("model_type") != "ImputeFormerModel":
        raise ValueError("Sensor checkpoint has the wrong model_type.")
    if checkpoint.get("entity_type") != "sensor":
        raise ValueError("Sensor checkpoint must declare entity_type='sensor'.")
    if checkpoint.get("uses_site_entity") is not False:
        raise ValueError("Sensor checkpoint must explicitly disable site entities.")
    if checkpoint.get("uses_coordinates") is not False:
        raise ValueError("Sensor checkpoint must explicitly disable coordinates.")
    if tuple(checkpoint.get("sensor_ids", ())) != EXPECTED_SENSOR_IDS:
        raise ValueError(
            "Sensor checkpoint schema does not match the five production sensors."
        )
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("Sensor checkpoint has no model_config.")
    if int(config.get("num_nodes", -1)) != len(EXPECTED_SENSOR_IDS):
        raise ValueError("Sensor checkpoint num_nodes is not five.")
    temporal_feature_names = checkpoint.get("temporal_feature_names")
    expected_temporal_names = RAW_TEMPORAL_FEATURE_NAMES if config.get("time_aware") else [
        "second_of_day_sin",
        "second_of_day_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "log_sampling_interval",
    ]
    if temporal_feature_names != expected_temporal_names:
        raise ValueError("Sensor checkpoint temporal feature schema is invalid.")
    expected_input_dim = 13 if config.get("time_aware") else 6
    if int(config.get("input_dim", -1)) != expected_input_dim or int(config.get("output_dim", -1)) != 1:
        raise ValueError("Sensor checkpoint input/output dimensions are invalid.")

    supported = checkpoint.get("supported_sampling_interval_seconds")
    if not isinstance(supported, Mapping):
        raise ValueError("Sensor checkpoint has no sampling interval contract.")
    minimum = float(supported.get("minimum", 0.0))
    maximum = supported.get("maximum")
    if not np.isfinite(minimum) or minimum < 0 or (
        maximum is not None and (
            not np.isfinite(float(maximum)) or float(maximum) <= 0
            or minimum > float(maximum)
        )
    ):
        raise ValueError("Sensor checkpoint sampling interval contract is invalid.")

    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("Sensor checkpoint has no normalization metadata.")
    mean = np.asarray(normalization.get("mean"), dtype="float32")
    std = np.asarray(normalization.get("std"), dtype="float32")
    expected_shape = (len(EXPECTED_SENSOR_IDS),)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError("Sensor checkpoint normalization shape is invalid.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Sensor checkpoint normalization values are invalid.")

    model = _build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def normalize_with_checkpoint(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3 or values.shape[2] != 1:
        raise ValueError(
            f"Sensor tensor must have shape [time, sensors, 1], got {values.shape}."
        )
    observed = np.isfinite(values)
    mean = np.asarray(mean, dtype="float32").reshape(1, -1, 1)
    std = np.asarray(std, dtype="float32").reshape(1, -1, 1)
    if mean.shape[1] != values.shape[1] or std.shape[1] != values.shape[1]:
        raise ValueError("Checkpoint normalization does not match sensor count.")
    filled = np.where(observed, values, mean).astype("float32")
    normalized = (filled - mean) / std
    return normalized.astype("float32"), observed.astype(bool)


def apply_checkpoint_output_calibration(
    prediction: np.ndarray,
    checkpoint: Mapping[str, object],
) -> np.ndarray:
    calibration = checkpoint.get("output_calibration")
    if calibration is None:
        return prediction
    if not isinstance(calibration, Mapping):
        raise ValueError("Checkpoint output_calibration must be a mapping.")
    calibrated = np.array(prediction, dtype="float32", copy=True)
    if calibrated.ndim != 4 or calibrated.shape[2] != len(EXPECTED_SENSOR_IDS):
        raise ValueError("Output calibration expects a four-dimensional tensor.")
    for sensor_index, sensor_id in enumerate(EXPECTED_SENSOR_IDS):
        spec = calibration.get(sensor_id)
        if not isinstance(spec, Mapping):
            raise ValueError(f"Missing output calibration for {sensor_id}.")
        scale = float(spec.get("scale", float("nan")))
        bias = float(spec.get("bias", float("nan")))
        if not np.isfinite([scale, bias]).all() or scale < 0.0:
            raise ValueError(f"Invalid output calibration for {sensor_id}.")
        calibrated[:, :, sensor_index, :] = (
            calibrated[:, :, sensor_index, :] * scale + bias
        )
    return calibrated


def _validate_regular_index(index: pd.DatetimeIndex) -> float | None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Sensor input must use a DatetimeIndex.")
    if index.tz is None:
        raise ValueError("Sensor timestamps must include a timezone.")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Sensor timestamps must be unique and chronological.")
    if not len(index):
        raise ValueError("Sensor input needs at least one timestamp.")
    if len(index) == 1:
        return None
    timestamps_ns = index.to_numpy(dtype="datetime64[ns]").astype(
        "int64", copy=False
    )
    deltas_ns = np.diff(timestamps_ns)
    if (deltas_ns <= 0).any() or not (deltas_ns == deltas_ns[0]).all():
        raise ValueError("Production ImputeFormer requires a continuous regular grid.")
    return float(deltas_ns[0] / 1_000_000_000.0)


def temporal_features(
    index: pd.DatetimeIndex,
    sampling_interval_seconds: float | None,
    maximum_supported_seconds: float | None = None,
    encoding: str = "bounded_log_interval_v1",
) -> np.ndarray:
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
    if sampling_interval_seconds is None:
        interval_value = 0.0
    else:
        if not np.isfinite(sampling_interval_seconds) or sampling_interval_seconds <= 0:
            raise ValueError("Sampling interval must be a finite positive duration.")
        log_interval = np.log1p(sampling_interval_seconds)
        if encoding == "bounded_log_interval_v1":
            interval_value = log_interval / (1.0 + log_interval)
        elif encoding == "legacy_maximum_log_interval":
            if maximum_supported_seconds is None or maximum_supported_seconds <= 0:
                raise ValueError("Legacy temporal encoding requires a positive scale.")
            interval_value = log_interval / np.log1p(maximum_supported_seconds)
        else:
            raise ValueError(f"Unknown temporal encoding: {encoding!r}.")
    interval_feature = np.full(len(index), interval_value, dtype="float64")
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


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length < 1 or window < 1:
        raise ValueError("Input length and model window must be positive.")
    if not 1 <= stride <= window:
        raise ValueError("Inference stride must be between 1 and model window.")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    final_start = length - window
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


class _InferenceWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        temporal: np.ndarray,
        observed: np.ndarray,
        starts: Sequence[int],
        window: int,
    ) -> None:
        self.values = values
        self.temporal = temporal
        self.observed = observed
        self.starts = list(starts)
        self.window = window

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = self.starts[index]
        stop = start + self.window
        values = self.values[start:stop]
        temporal = self.temporal[start:stop]
        observed = self.observed[start:stop]
        if len(values) < self.window:
            padding = self.window - len(values)
            values = np.pad(values, ((0, padding), (0, 0), (0, 0)), mode="edge")
            temporal = np.pad(temporal, ((0, padding), (0, 0)), mode="edge")
            observed = np.pad(observed, ((0, padding), (0, 0), (0, 0)))
        return (
            torch.from_numpy(values),
            torch.from_numpy(temporal),
            torch.from_numpy(observed),
            start,
        )


def _validate_sensor_columns(columns: Sequence[object]) -> None:
    if len(columns) != len(set(columns)):
        raise ValueError("Sensor identifiers must be unique.")
    received = {str(column) for column in columns}
    expected = set(EXPECTED_SENSOR_IDS)
    missing = sorted(expected - received)
    extra = sorted(received - expected)
    if missing or extra:
        raise ValueError(
            "Production ImputeFormer requires exactly five sensor_id values; "
            f"missing={missing}, extra={extra}. Site/station and coordinate fields "
            "are not accepted."
        )


def deterministic_observed_holdout(
    frame: pd.DataFrame,
    ratio: float,
    seed: int,
) -> np.ndarray:
    """Select reproducible known values for per-run imputation evaluation."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("Alignment evaluation holdout ratio must be in (0, 1).")
    rng = np.random.default_rng(seed)
    observed = frame.notna().to_numpy()
    holdout = np.zeros_like(observed, dtype=bool)
    for sensor_index in range(observed.shape[1]):
        positions = np.flatnonzero(observed[:, sensor_index])
        if len(positions) < 2:
            continue
        count = min(len(positions) - 1, max(1, int(round(len(positions) * ratio))))
        selected = rng.choice(positions, size=count, replace=False)
        holdout[selected, sensor_index] = True
    return holdout


def calculate_alignment_metrics(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    holdout: np.ndarray,
    sensor_std: np.ndarray,
    holdout_ratio: float,
) -> list[dict[str, object]]:
    """Return physical per-sensor metrics and a comparable normalized total."""
    truth_values = truth.to_numpy(dtype="float64")
    prediction_values = prediction.to_numpy(dtype="float64")
    rows: list[dict[str, object]] = []
    normalized_errors = []
    for sensor_index, sensor_id in enumerate(EXPECTED_SENSOR_IDS):
        selected = holdout[:, sensor_index]
        errors = prediction_values[selected, sensor_index] - truth_values[selected, sensor_index]
        if not len(errors):
            rows.append({
                "scope": "sensor",
                "sensor_id": sensor_id,
                "value_space": "physical_sensor_unit",
                "evaluation_method": "deterministic_holdout_of_observed_input",
                "requested_holdout_ratio": holdout_ratio,
                "evaluated_values": 0,
                "status": "insufficient_observed_values",
                "MAE": None,
                "MSE": None,
                "RMSE": None,
            })
            continue
        squared = np.square(errors)
        rows.append({
            "scope": "sensor",
            "sensor_id": sensor_id,
            "value_space": "physical_sensor_unit",
            "evaluation_method": "deterministic_holdout_of_observed_input",
            "requested_holdout_ratio": holdout_ratio,
            "evaluated_values": len(errors),
            "MAE": float(np.mean(np.abs(errors))),
            "MSE": float(np.mean(squared)),
            "RMSE": float(np.sqrt(np.mean(squared))),
        })
        normalized_errors.append(errors / float(sensor_std[sensor_index]))
    if not normalized_errors:
        return [{
            "scope": "overall",
            "sensor_id": "all_five_sensors",
            "value_space": "checkpoint_normalized",
            "evaluation_method": "deterministic_holdout_of_observed_input",
            "requested_holdout_ratio": holdout_ratio,
            "evaluated_values": 0,
            "status": "insufficient_observed_values",
            "MAE": None,
            "MSE": None,
            "RMSE": None,
        }, *rows]
    all_normalized = np.concatenate(normalized_errors)
    squared = np.square(all_normalized)
    rows.insert(0, {
        "scope": "overall",
        "sensor_id": "all_five_sensors",
        "value_space": "checkpoint_normalized",
        "evaluation_method": "deterministic_holdout_of_observed_input",
        "requested_holdout_ratio": holdout_ratio,
        "evaluated_values": len(all_normalized),
        "MAE": float(np.mean(np.abs(all_normalized))),
        "MSE": float(np.mean(squared)),
        "RMSE": float(np.sqrt(np.mean(squared))),
    })
    return rows


def _infer_with_loaded_imputeformer(
    ordered: pd.DataFrame,
    sampling_interval_seconds: float | None,
    model: ImputeFormerModel,
    checkpoint: Mapping[str, object],
    resolved_device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    """Run one forward-only pass with an already loaded checkpoint."""
    supported = checkpoint["supported_sampling_interval_seconds"]
    maximum_interval = supported.get("maximum")
    config = checkpoint["model_config"]
    window = int(config["window"])
    stride = int(config.get("inference_stride", window))
    starts = _window_starts(len(ordered), window, stride)
    inference_window = min(window, len(ordered)) if config.get("variable_length") else window
    mean = np.asarray(checkpoint["normalization"]["mean"], dtype="float32")
    std = np.asarray(checkpoint["normalization"]["std"], dtype="float32")
    raw_values = ordered.to_numpy(dtype="float64")
    normalized, observed = normalize_with_checkpoint(
        raw_values[:, :, None],
        mean,
        std,
    )
    time_values = raw_temporal_features(ordered.index) if config.get("time_aware") else temporal_features(
        ordered.index,
        sampling_interval_seconds,
        maximum_interval,
        encoding=str(checkpoint.get("temporal_encoding", "legacy_maximum_log_interval")),
    )
    if config.get("time_aware"):
        global_baseline, global_context = _visible_time_baseline(
            raw_values, observed[:, :, 0], time_values[:, 0], mean,
        )
    loader = DataLoader(
        _InferenceWindowDataset(
            normalized,
            time_values,
            observed,
            starts,
            inference_window,
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=resolved_device.type == "cuda",
    )

    sums = np.zeros_like(raw_values, dtype="float64")
    counts = np.zeros_like(raw_values, dtype="float64")
    best_context = np.full(raw_values.shape, -1, dtype="int8")
    LOGGER.info(
        "ImputeFormer inference: rows=%d, model_window=%d, effective_window=%d, "
        "windows=%d, stride=%d, interval_seconds=%s, device=%s",
        len(ordered), window, inference_window, len(starts), stride,
        sampling_interval_seconds, resolved_device,
    )
    if config.get("time_aware"):
        LOGGER.info(
            "Time-aware estimation: neural_residual_gains=%s visible_median_gains=%s no_context_prior=%s",
            getattr(model, "context_residual_gain", torch.empty(0)).detach().cpu().tolist(),
            getattr(model, "context_median_gain", torch.empty(0)).detach().cpu().tolist(),
            config.get("no_context_prior"),
        )
    with torch.inference_mode():
        for x, temporal, mask, batch_starts in loader:
            prediction = model(
                x.to(resolved_device, non_blocking=True),
                temporal.to(resolved_device, non_blocking=True),
                mask.to(resolved_device, non_blocking=True).float(),
            ).cpu().numpy()
            prediction = apply_checkpoint_output_calibration(
                prediction,
                checkpoint,
            )[..., 0]
            prediction = prediction * std[None, None, :] + mean[None, None, :]
            for local_index, start in enumerate(batch_starts.tolist()):
                stop = min(start + inference_window, len(ordered))
                window_observed = observed[start:stop, :, 0]
                left_context = np.logical_or.accumulate(window_observed, axis=0)
                right_context = np.logical_or.accumulate(window_observed[::-1], axis=0)[::-1]
                context = left_context.astype("int8") + right_context.astype("int8")
                previous_context = best_context[start:stop]
                replace = context > previous_context
                equal = context == previous_context
                window_prediction = prediction[local_index, :stop - start]
                if config.get("time_aware"):
                    local_baseline, _ = _visible_time_baseline(
                        raw_values[start:stop], window_observed,
                        time_values[start:stop, 0], mean,
                    )
                    # Preserve distant visible endpoints across long sensor outages.
                    # Residuals require the same directional context as the full input.
                    residual = window_prediction - local_baseline
                    residual = np.where(context == global_context[start:stop], residual, 0.0)
                    window_prediction = global_baseline[start:stop] + residual
                window_sums, window_counts = sums[start:stop], counts[start:stop]
                # Prefer windows with observations on both sides of a gap.
                window_sums[replace] = window_prediction[replace]
                window_counts[replace] = 1
                previous_context[replace] = context[replace]
                window_sums[equal] += window_prediction[equal]
                window_counts[equal] += 1
    if not (counts > 0).all():
        raise RuntimeError("ImputeFormer inference left uncovered timestamps.")
    predictions = sums / counts
    aligned_values = np.where(np.isfinite(raw_values), raw_values, predictions)
    if not np.isfinite(aligned_values).all():
        raise RuntimeError("ImputeFormer aligned output still contains missing values.")
    return pd.DataFrame(
        aligned_values,
        index=ordered.index,
        columns=EXPECTED_SENSOR_IDS,
    )


def _visible_time_baseline(
    values: np.ndarray,
    observed: np.ndarray,
    elapsed_seconds: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only visible values in physical time, retaining distant context."""
    baseline = np.empty_like(values, dtype="float64")
    elapsed = elapsed_seconds - elapsed_seconds[0]
    for sensor_index in range(values.shape[1]):
        visible = observed[:, sensor_index]
        baseline[:, sensor_index] = (
            np.interp(elapsed, elapsed[visible], values[visible, sensor_index])
            if visible.any() else fallback[sensor_index]
        )
    previous = np.logical_or.accumulate(observed, axis=0)
    following = np.logical_or.accumulate(observed[::-1], axis=0)[::-1]
    context = previous.astype("int8") + following.astype("int8")
    return baseline, context


def align_with_pretrained_imputeformer(
    wide_df: pd.DataFrame,
    checkpoint_path: str | Path = DEFAULT_SENSOR_CHECKPOINT,
    batch_size: int = 8,
    device: str | torch.device = "auto",
    evaluation_holdout_ratio: float = 0.0,
    evaluation_seed: int = 42,
    preloaded_model: ImputeFormerModel | None = None,
    preloaded_checkpoint: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fill gaps with the trained model while preserving every observed value."""
    if batch_size < 1:
        raise ValueError("Inference batch_size must be positive.")
    _validate_sensor_columns(wide_df.columns)
    timestamps_ns = timestamp_nanoseconds(wide_df.index)
    actual_intervals = np.diff(timestamps_ns).astype("float64") / 1e9
    sampling_interval_seconds = float(np.median(actual_intervals)) if len(actual_intervals) else None
    if sampling_interval_seconds is None:
        sampling_interval_seconds = wide_df.attrs.get("sampling_interval_seconds")
    ordered = wide_df.loc[:, EXPECTED_SENSOR_IDS].copy()
    for sensor_id in EXPECTED_SENSOR_IDS:
        ordered[sensor_id] = pd.to_numeric(ordered[sensor_id], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    no_observed_sensor_ids = [
        sensor_id for sensor_id in EXPECTED_SENSOR_IDS if ordered[sensor_id].isna().all()
    ]
    if len(no_observed_sensor_ids) == len(EXPECTED_SENSOR_IDS):
        raise ValueError("At least one finite sensor observation is required.")
    if no_observed_sensor_ids:
        LOGGER.info("ImputeFormer sensors without observed input: %s", no_observed_sensor_ids)

    if (preloaded_model is None) != (preloaded_checkpoint is None):
        raise ValueError(
            "preloaded_model and preloaded_checkpoint must be supplied together."
        )
    if preloaded_model is None:
        resolved_device = (
            torch.device("cuda")
            if str(device) == "auto" and torch.cuda.is_available()
            else torch.device("cpu")
            if str(device) == "auto"
            else torch.device(device)
        )
        model, checkpoint = load_sensor_imputeformer_checkpoint(
            checkpoint_path,
            resolved_device,
        )
    else:
        model = preloaded_model
        checkpoint = preloaded_checkpoint
        resolved_device = next(model.parameters()).device
    supported = checkpoint["supported_sampling_interval_seconds"]
    config = checkpoint["model_config"]
    if not config.get("time_aware"):
        _validate_regular_index(wide_df.index)
    window = int(config["window"])
    stride = int(config.get("inference_stride", window))
    std = np.asarray(checkpoint["normalization"]["std"], dtype="float32")
    raw_values = ordered.to_numpy(dtype="float64")
    aligned = _infer_with_loaded_imputeformer(
        ordered,
        sampling_interval_seconds,
        model,
        checkpoint,
        resolved_device,
        batch_size,
    )
    checkpoint_path = Path(checkpoint_path).resolve()
    metadata = {
        "mode": "pretrained_inference",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "entity_type": "sensor",
        "uses_site_entity": False,
        "uses_coordinates": False,
        "sensor_ids": list(EXPECTED_SENSOR_IDS),
        "input_rows": len(ordered),
        "window": window,
        "inference_stride": stride,
        "inference_windows": len(_window_starts(len(ordered), window, stride)),
        "arbitrary_input_length": True,
        "window_fusion": (
            "global_visible_time_baseline_with_context_matched_residual_mean"
            if config.get("time_aware") else "observed_context_priority_mean"
        ),
        "short_sequence_strategy": (
            "native_variable_length" if config.get("variable_length") else "masked_padding"
        ),
        "sampling_interval_seconds": sampling_interval_seconds,
        "temporal_input": "actual_elapsed_and_delta_seconds" if config.get("time_aware") else "legacy_grid_covariates",
        "actual_interval_seconds_min": float(actual_intervals.min()) if len(actual_intervals) else None,
        "actual_interval_seconds_max": float(actual_intervals.max()) if len(actual_intervals) else None,
        "supported_sampling_interval_seconds": supported,
        "input_observed_values": int(np.isfinite(raw_values).sum()),
        "no_observed_sensor_ids": no_observed_sensor_ids,
        "imputed_values": int(np.isnan(raw_values).sum()),
        "device": str(resolved_device),
        "output_calibration_applied": checkpoint.get("output_calibration") is not None,
        "visible_median_calibration_applied": bool(config.get("visible_median_calibration", False)),
        "neural_residual_enabled": bool(getattr(model, "context_residual_gain", torch.ones(1)).any()),
        "no_context_prior_normalized": config.get("no_context_prior"),
    }
    if evaluation_holdout_ratio:
        holdout = deterministic_observed_holdout(
            ordered, evaluation_holdout_ratio, evaluation_seed
        )
        evaluation_input = ordered.mask(holdout)
        evaluation_prediction = _infer_with_loaded_imputeformer(
            evaluation_input,
            sampling_interval_seconds,
            model,
            checkpoint,
            resolved_device,
            batch_size,
        ) if holdout.any() else ordered.copy()
        metadata["evaluation_metrics"] = calculate_alignment_metrics(
            ordered,
            evaluation_prediction,
            holdout,
            std,
            evaluation_holdout_ratio,
        )
        metadata["evaluation_seed"] = int(evaluation_seed)
    return aligned, metadata
