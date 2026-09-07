from pathlib import Path
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


SONAR_TIFF_EXTENSIONS = frozenset({".tif", ".tiff"})


def resolve_image_source_paths(
    path: str | Path, *, allow_xtf: bool = False,
) -> list[Path]:
    """Resolve a single source or supported files directly inside a directory.

    Explicit files retain the existing loader's format validation. Directory
    discovery only includes formats with a registered Pillow decoder, plus
    XTF when requested, in deterministic filename order.
    """
    source = Path(path).expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image source file or directory not found: {source}")
    extensions = {
        extension.lower()
        for extension, image_format in Image.registered_extensions().items()
        if image_format in Image.OPEN
    }
    if allow_xtf:
        extensions.add(".xtf")
    paths = sorted(
        (child for child in source.iterdir()
         if child.is_file() and child.suffix.lower() in extensions),
        key=lambda child: child.name,
    )
    if not paths:
        raise ValueError(f"Image source directory contains no supported files: {source}")
    return [child.resolve() for child in paths]


def resolve_sonar_source_paths(
    path: str | Path, *, allow_xtf: bool = False,
) -> list[Path]:
    """Resolve TIFF sonar inputs, with XTF enabled only for the legacy input."""
    source = Path(path).expanduser().resolve()
    extensions = set(SONAR_TIFF_EXTENSIONS)
    if allow_xtf:
        extensions.add(".xtf")
    formats = "TIFF (.tif, .tiff)" + (" or legacy XTF (.xtf)" if allow_xtf else "")
    if source.is_file():
        if source.suffix.lower() not in extensions:
            raise ValueError(f"Sonar source must be {formats}: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Sonar source file or directory not found: {source}")
    paths = sorted(
        (child for child in source.iterdir()
         if child.is_file() and child.suffix.lower() in extensions),
        key=lambda child: child.name,
    )
    if not paths:
        raise ValueError(f"Sonar source directory contains no supported {formats} files: {source}")
    return [child.resolve() for child in paths]


def _datetime_index_nanoseconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Return epoch nanoseconds independently of pandas' internal resolution."""
    return index.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)


def load_aligned_sensors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Aligned sensor CSV must contain a timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load native-resolution grayscale intensities without losing integer depth.

    Integer grayscale uses the dtype's positive range; negative signed values
    are clipped to zero. Float grayscale already in [0, 1] is preserved, while
    other finite float ranges are scaled to [0, 1]. Positive float constants
    outside that range become white, and negative constants become black.
    """
    with Image.open(path) as image:
        return _grayscale_intensities(image, path)


def load_sonar_tiff_image(path: Path) -> np.ndarray:
    """Load one TIFF sonar image at native resolution and original bit depth."""
    path = Path(path)
    if path.suffix.lower() not in SONAR_TIFF_EXTENSIONS:
        raise ValueError(f"Sonar image must use a .tif or .tiff extension: {path}")
    with Image.open(path) as image:
        if image.format != "TIFF":
            raise ValueError(f"Sonar image must contain TIFF data: {path}")
        if image.n_frames != 1:
            raise ValueError(
                f"Multi-page sonar TIFF is not supported; provide one image per file: {path}"
            )
        return _grayscale_intensities(image, path)


def _grayscale_intensities(image: Image.Image, path: Path) -> np.ndarray:
    if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
        pixels = np.asarray(image)
        values = pixels.astype("float32")
        if np.issubdtype(pixels.dtype, np.integer):
            # Older Pillow versions expose 16-bit PNG pixels as signed I.
            maximum = (
                65535 if image.format == "PNG" and image.mode == "I"
                else np.iinfo(pixels.dtype).max
            )
            return np.clip(values / float(maximum), 0.0, 1.0)
        if not np.isfinite(values).all():
            raise ValueError(f"Grayscale image contains non-finite intensities: {path}")
        low, high = float(values.min()), float(values.max())
        if 0.0 <= low and high <= 1.0:
            return values
        if high == low:
            return np.full_like(values, 1.0 if high > 0.0 else 0.0)
        return ((values.astype("float64") - low) / (high - low)).astype("float32")
    return np.asarray(image.convert("L"), dtype="float32") / 255.0

# ---------------------------------------------------------------------------
# Streaming / backend-pushed sensor data
# ---------------------------------------------------------------------------
# Backend-friendly long-format record:
#   {"timestamp": "2026-06-08T12:00:00Z",
#    "sensor_id": "water_temperature_c",
#    "value":     20.1}
# Multiple records are buffered, then pivoted to one column per sensor.

def normalize_streaming_records(records: Iterable[Dict]) -> pd.DataFrame:
    """Validate and normalize a stream of long-format records.

    Returns a DataFrame with columns [timestamp, sensor_id, value].
    """
    rows: List[Dict] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Record #{idx} is not a JSON object: got {type(record).__name__}."
            )
        for required in ("timestamp", "sensor_id", "value"):
            if required not in record:
                raise ValueError(
                    f"Record #{idx} missing required field '{required}'."
                )
        sensor_id = str(record["sensor_id"]).strip()
        if not sensor_id:
            raise ValueError(f"Record #{idx} has an empty sensor_id.")
        # Per the streaming Record contract, value may be None for a missing
        # observation. Map it to NaN so the downstream pivot/resample pipeline
        # treats it as a gap rather than raising or zero-filling silently.
        raw_value = record["value"]
        if raw_value is None:
            value: Optional[float] = np.nan
        else:
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Record #{idx} has non-numeric 'value': {raw_value!r}."
                ) from exc
        timestamp = pd.to_datetime(record["timestamp"], utc=True)
        if pd.isna(timestamp):
            raise ValueError(f"Record #{idx} has an invalid timestamp.")
        rows.append({
            "timestamp": timestamp,
            "sensor_id": sensor_id,
            "value": value if value is not None and np.isfinite(value) else np.nan,
        })
    if not rows:
        raise ValueError("No streaming records were provided.")
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _normalize_freq(freq: str) -> str:
    """Translate pre-pandas-2.x minute aliases to the current form.

    In pandas 2.x, a lowercase ``m`` suffix is parsed as a month offset (it used
    to mean minutes). External callers may still pass the legacy alias
    ``"1m"`` / ``"5m"`` etc. based on the old docstring. We rewrite the
    ``<number>m`` form to ``<number>min`` here so ``pd.to_offset`` keeps
    working. Uppercase ``M`` / month-end variants are left untouched because
    the streaming pipeline is not expected to use month granularity.

    Other aliases (``h``, ``d``, ``s``, ``T``, ``min``) are returned unchanged.
    """
    import re
    match = re.fullmatch(r"(\d+)([a-zA-Z]+)", freq.strip())
    if match is None:
        return freq
    number, unit = match.group(1), match.group(2)
    if unit == "m":
        return f"{number}min"
    return freq


def _fixed_frequency_seconds(freq: str) -> float:
    normalized = _normalize_freq(freq)
    try:
        offset = pd.tseries.frequencies.to_offset(normalized)
        nanoseconds = offset.nanos
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Alignment frequency must be a fixed duration, got {freq!r}."
        ) from exc
    seconds = nanoseconds / 1_000_000_000.0
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Alignment frequency must be positive, got {freq!r}.")
    return float(seconds)


def infer_sampling_intervals(records: Iterable[Dict]) -> Dict[str, object]:
    """Select the sensor with most distinct valid samples as the reference.

    Ties prefer a shorter cadence, then the sensor identifier. The modal
    timestamp difference tolerates dropped samples; jittered streams fall
    back to their median difference without imposing a physical time unit.
    """
    df = normalize_streaming_records(records)
    intervals: Dict[str, Optional[float]] = {}
    counts: Dict[str, int] = {}
    ranges: Dict[str, Dict[str, str]] = {}
    for sensor_id, group in df.groupby("sensor_id", sort=True):
        timestamps = pd.DatetimeIndex(
            group.loc[group["value"].notna(), "timestamp"].drop_duplicates()
        ).sort_values()
        counts[str(sensor_id)] = len(timestamps)
        if len(timestamps):
            ranges[str(sensor_id)] = {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
            }
        if len(timestamps) < 2:
            intervals[str(sensor_id)] = None
            continue
        deltas = np.diff(_datetime_index_nanoseconds(timestamps))
        unique, frequency = np.unique(deltas, return_counts=True)
        best = int(np.argmax(frequency))
        intervals[str(sensor_id)] = float(
            (unique[best] if frequency[best] > 1 else np.median(deltas))
            / 1_000_000_000.0
        )

    usable = {
        sensor_id: interval
        for sensor_id, interval in intervals.items()
        if interval is not None
    }
    observed_sensors = [sensor_id for sensor_id, count in counts.items() if count]
    if not observed_sensors:
        raise ValueError("At least one finite sensor observation is required.")
    reference_sensor = min(
        observed_sensors,
        key=lambda item: (-counts[item], intervals[item] or float("inf"), item),
    )
    fastest_interval = min(usable.values()) if usable else None
    tolerance = max(1e-9, fastest_interval * 0.01) if usable else 0.0
    candidates = [
        sensor_id
        for sensor_id, interval in usable.items()
        if abs(interval - fastest_interval) <= tolerance
    ]
    reference_interval = intervals[reference_sensor]
    interval_source = "reference_sensor"
    if reference_interval is None:
        all_times = pd.DatetimeIndex(df["timestamp"].drop_duplicates()).sort_values()
        reference_interval = (
            float(np.median(np.diff(_datetime_index_nanoseconds(all_times))) / 1e9)
            if len(all_times) > 1 else None
        )
        interval_source = "combined_timestamps" if reference_interval else "undetermined"
    return {
        "sampling_intervals_seconds": intervals,
        "observation_counts": counts,
        "observation_ranges": ranges,
        "fastest_interval_seconds": fastest_interval,
        "fastest_sensor_ids": sorted(candidates),
        "reference_sensor_id": reference_sensor,
        "reference_interval_seconds": reference_interval,
        "reference_selection": "most_valid_samples_then_shortest_interval_then_sensor_id",
        "interval_inference_source": interval_source,
    }


def _records_to_reference_grid(
    df: pd.DataFrame,
    sensors: List[str],
    reference_sensor: str,
    frequency_seconds: Optional[float],
    preserve_actual_timestamps: bool = False,
) -> pd.DataFrame:
    reference_rows = df["sensor_id"] == reference_sensor
    if not preserve_actual_timestamps:
        reference_rows &= df["value"].notna()
    reference_times = pd.DatetimeIndex(
        df.loc[reference_rows, "timestamp"].drop_duplicates()
    ).sort_values()
    if reference_times.empty:
        raise ValueError(f"Reference sensor {reference_sensor!r} has no observations.")
    reference_ns = _datetime_index_nanoseconds(reference_times)
    all_ns = _datetime_index_nanoseconds(pd.DatetimeIndex(df["timestamp"]))
    if preserve_actual_timestamps:
        outside_reference = (all_ns < reference_ns[0]) | (all_ns > reference_ns[-1])
        index_ns = np.unique(np.concatenate([reference_ns, all_ns[outside_reference]]))
        steps = len(index_ns)
    else:
        frequency_ns = int(round((frequency_seconds or 1.0) * 1_000_000_000.0))
        if frequency_ns <= 0:
            raise ValueError("Inferred alignment interval is below timestamp precision.")
        anchor_ns = int(reference_ns[0])
        first_step = (int(all_ns.min()) - anchor_ns) // frequency_ns
        final_step = -((anchor_ns - int(all_ns.max())) // frequency_ns)
        start_ns = anchor_ns + first_step * frequency_ns
        steps = final_step - first_step + 1
        index_ns = start_ns + np.arange(steps, dtype="int64") * frequency_ns
    index = pd.to_datetime(index_ns, utc=True)
    wide = pd.DataFrame(index=index, columns=sensors, dtype="float64")

    for sensor_id, group in df.groupby("sensor_id", sort=False):
        valid = group[group["value"].notna()]
        if valid.empty:
            continue
        timestamp_ns = _datetime_index_nanoseconds(
            pd.DatetimeIndex(valid["timestamp"])
        )
        if preserve_actual_timestamps:
            following = np.minimum(np.searchsorted(index_ns, timestamp_ns), steps - 1)
            previous = np.maximum(following - 1, 0)
            # Integer differences retain submillisecond precision; ties choose earlier.
            choose_previous = (
                np.abs(timestamp_ns - index_ns[previous])
                <= np.abs(index_ns[following] - timestamp_ns)
            )
            positions = np.where(choose_previous, previous, following)
        else:
            positions = np.rint((timestamp_ns - start_ns) / frequency_ns).astype("int64")
        inside = (positions >= 0) & (positions < steps)
        if not inside.any():
            continue
        mapped = pd.DataFrame({
            "position": positions[inside],
            "value": valid.loc[inside, "value"].to_numpy(dtype="float64"),
        }).groupby("position", sort=True)["value"].mean()
        wide.iloc[mapped.index.to_numpy(dtype="int64"), sensors.index(str(sensor_id))] = (
            mapped.to_numpy(dtype="float64")
        )
    wide.index.name = "timestamp"
    wide.attrs["sampling_interval_seconds"] = frequency_seconds
    wide.attrs["mapped_observation_count"] = int(wide.notna().sum().sum())
    wide.attrs["merged_observation_count"] = int(df["value"].notna().sum()) - int(
        wide.notna().sum().sum()
    )
    return wide


def load_streaming_records(
    records: Iterable[Dict],
    freq: str = "auto",
    min_window: Optional[int] = None,
    *,
    preserve_actual_timestamps: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    """Convert long-format streaming records to the wide tensor used downstream.

    Parameters
    ----------
    records : iterable of dict
        Each dict has {timestamp, sensor_id, value}.
    freq : str
        ``"auto"`` preserves timestamps of the sensor with most valid samples,
        plus other sensors' timestamps outside that reference's time range.
        A fixed pandas duration can be supplied explicitly for diagnostics.
    min_window : int, optional
        If given, raise ValueError when the aligned table has fewer rows.
    preserve_actual_timestamps : bool
        For ``auto``, keep actual reference timestamps when True. Set False
        for checkpoints requiring a regular grid at the inferred cadence.

    Returns
    -------
    wide_df : DataFrame
        Aligned wide table with one column per sensor.
    values  : ndarray, shape [T, N, 1]
    mask    : ndarray, shape [T, N, 1], 1.0 where observed.
    sensors : list[str]
    grid_metadata : dict
        Per-sensor representative intervals, reference sensor, and time-grid policy.
    """
    records = list(records)
    df = normalize_streaming_records(records)
    inference = infer_sampling_intervals(records)
    automatic = str(freq).strip().lower() == "auto"
    use_actual_timestamps = automatic and preserve_actual_timestamps
    if automatic:
        frequency_seconds = inference["reference_interval_seconds"]
        frequency_source = "most_sampled_sensor"
    else:
        frequency_seconds = _fixed_frequency_seconds(freq)
        frequency_source = "explicit"
    sensors = sorted(str(value) for value in df["sensor_id"].unique())
    if len(sensors) < 2:
        raise ValueError(
            f"Streaming pipeline needs at least 2 sensors, got {len(sensors)}."
        )
    wide = _records_to_reference_grid(
        df,
        sensors,
        str(inference["reference_sensor_id"]),
        frequency_seconds,
        preserve_actual_timestamps=use_actual_timestamps,
    )
    wide = wide[sensors]

    if min_window is not None and len(wide) < min_window:
        raise ValueError(
            f"Resampled stream has {len(wide)} time steps, "
            f"need at least {min_window} (window). Buffer more data, lower "
            f"--window, or buffer a longer physical duration."
        )

    values = wide.to_numpy(dtype="float32").reshape(len(wide), len(sensors), 1)
    mask = (~np.isnan(values)).astype("float32")
    metadata = {
        **inference,
        "frequency_source": frequency_source,
        "alignment_frequency_seconds": frequency_seconds,
        "alignment_frequency": (
            pd.Timedelta(seconds=frequency_seconds).isoformat()
            if frequency_seconds is not None else None
        ),
        "alignment_interval_role": (
            "representative_cadence_estimate" if use_actual_timestamps else "fixed_grid_interval"
        ),
        "irregular_timestamps_preserved": use_actual_timestamps,
        "grid_policy": (
            "reference_actual_timestamps_with_outer_coverage" if use_actual_timestamps
            else "reference_cadence_covering_all_input_timestamps"
        ),
        "observation_mapping": (
            "nearest_reference_timestamp_mean_ties_to_earlier" if use_actual_timestamps
            else "nearest_grid_point_mean_within_bin"
        ),
        "merged_observation_count": wide.attrs["merged_observation_count"],
        "no_observed_sensor_ids": [sensor for sensor in sensors if wide[sensor].isna().all()],
        "grid_start": wide.index[0].isoformat(),
        "grid_end": wide.index[-1].isoformat(),
        "grid_rows": len(wide),
    }
    return wide, values, mask, sensors, metadata


class SensorBuffer:
    """In-memory buffer for streaming sensor records.

    The backend appends one record at a time via ``add``. ``is_ready`` reports
    whether the buffered data, once resampled to ``freq``, has at least
    ``min_window`` time steps (a caller-controlled batching threshold). ``flush`` converts
    the buffer to the wide tensor expected by the rest of the pipeline.
    """

    def __init__(self, freq: str = "auto", min_window: int = 1,
                 sensor_filter: Optional[List[str]] = None, *,
                 preserve_actual_timestamps: bool = True) -> None:
        self.freq = freq
        self.min_window = min_window
        self.preserve_actual_timestamps = preserve_actual_timestamps
        self.sensor_filter = set(sensor_filter) if sensor_filter else None
        self._records: List[Dict] = []
        self._lock = threading.RLock()

    # ----- mutators ---------------------------------------------------------
    def add(self, timestamp, sensor_id: str, value) -> None:
        """Append a single record. timestamp accepts ISO 8601 or datetime."""
        with self._lock:
            self._records.append({
                "timestamp": timestamp,
                "sensor_id": sensor_id,
                "value": value,
            })

    def extend(self, records: Iterable[Dict]) -> None:
        """Append many records at once (e.g. a batch from the backend)."""
        with self._lock:
            for r in records:
                self._records.append({
                    "timestamp": r.get("timestamp"),
                    "sensor_id": r.get("sensor_id"),
                    "value": r.get("value"),
                })

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    # ----- inspection -------------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def is_ready(self) -> bool:
        with self._lock:
            if not self._records:
                return False
            try:
                wide, _, _, _, _ = self._materialize(check_window=False)
            except ValueError:
                return False
            return len(wide) >= self.min_window

    def ready_progress(self) -> Dict[str, object]:
        """Return current buffer stats for progress reporting."""
        with self._lock:
            sensors = sorted({r["sensor_id"] for r in self._records})
            try:
                wide, _, _, _, grid_metadata = self._materialize(check_window=False)
                resampled_rows = len(wide)
            except ValueError:
                resampled_rows = 0
                grid_metadata = {}
            return {
                "raw_records": len(self._records),
                "sensors": len(sensors),
                "resampled_rows": resampled_rows,
                "min_window": self.min_window,
                "alignment_frequency_seconds": grid_metadata.get(
                    "alignment_frequency_seconds"
                ),
                "fastest_sensor_ids": grid_metadata.get("fastest_sensor_ids", []),
                "reference_sensor_id": grid_metadata.get("reference_sensor_id"),
                "reference_interval_seconds": grid_metadata.get("reference_interval_seconds"),
                "observation_counts": grid_metadata.get("observation_counts", {}),
                "reference_selection": grid_metadata.get("reference_selection"),
                "sampling_intervals_seconds": grid_metadata.get(
                    "sampling_intervals_seconds", {}
                ),
            }

    # ----- flush ------------------------------------------------------------
    def flush(
        self,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], Dict[str, object]]:
        """Consume the buffer and return the wide tensor for the pipeline."""
        _, wide, values, mask, sensors, metadata = self.flush_with_records()
        return wide, values, mask, sensors, metadata

    def flush_with_records(
        self,
    ) -> Tuple[
        List[Dict],
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        List[str],
        Dict[str, object],
    ]:
        """Atomically consume both original records and their materialized grid."""
        with self._lock:
            if not self._records:
                raise ValueError("Sensor buffer is empty; nothing to flush.")
            wide, values, mask, sensors, metadata = self._materialize(
                check_window=True
            )
            records = [dict(record) for record in self._records]
            self._records.clear()
            return records, wide, values, mask, sensors, metadata

    # ----- internal ---------------------------------------------------------
    def _materialize(self, check_window: bool):
        records = self._filter(self._records)
        return load_streaming_records(
            records,
            self.freq,
            min_window=self.min_window if check_window else None,
            preserve_actual_timestamps=self.preserve_actual_timestamps,
        )

    def _filter(self, records: List[Dict]) -> List[Dict]:
        if not self.sensor_filter:
            return records
        out = []
        for r in records:
            if r["sensor_id"] not in self.sensor_filter:
                continue
            out.append(r)
        return out
