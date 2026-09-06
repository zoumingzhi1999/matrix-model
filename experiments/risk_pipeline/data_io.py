from pathlib import Path
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def _datetime_index_nanoseconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Return epoch nanoseconds independently of pandas' internal resolution."""
    return index.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)


def load_aligned_sensors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Aligned sensor CSV must contain a timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Load a risk-analysis image at its original spatial resolution."""
    with Image.open(path) as image:
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
        rows.append({
            "timestamp": pd.to_datetime(record["timestamp"], utc=True),
            "sensor_id": sensor_id,
            "value": value,
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
    """Infer each sensor cadence and select the reliably fastest stream.

    The most frequent rounded positive timestamp difference is used so that
    occasional dropped records do not make a regular sensor appear slower.
    When all differences are unique, the median is the conservative fallback.
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
        deltas = (
            np.diff(_datetime_index_nanoseconds(timestamps)).astype("float64")
            / 1_000_000_000.0
        )
        deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
        if not len(deltas):
            intervals[str(sensor_id)] = None
            continue
        rounded = np.round(deltas, decimals=6)
        unique, frequency = np.unique(rounded, return_counts=True)
        best = int(np.argmax(frequency))
        intervals[str(sensor_id)] = float(
            unique[best] if frequency[best] > 1 else np.median(rounded)
        )

    usable = {
        sensor_id: interval
        for sensor_id, interval in intervals.items()
        if interval is not None
    }
    if not usable:
        raise ValueError(
            "At least one sensor needs two valid timestamps to infer the fastest cadence."
        )
    fastest_interval = min(usable.values())
    tolerance = max(1e-6, fastest_interval * 0.01)
    candidates = [
        sensor_id
        for sensor_id, interval in usable.items()
        if abs(interval - fastest_interval) <= tolerance
    ]
    reference_sensor = sorted(candidates, key=lambda item: (-counts[item], item))[0]
    return {
        "sampling_intervals_seconds": intervals,
        "observation_counts": counts,
        "observation_ranges": ranges,
        "fastest_interval_seconds": float(fastest_interval),
        "fastest_sensor_ids": sorted(candidates),
        "reference_sensor_id": reference_sensor,
    }


def _records_to_reference_grid(
    df: pd.DataFrame,
    sensors: List[str],
    reference_sensor: str,
    frequency_seconds: float,
) -> pd.DataFrame:
    reference_times = pd.DatetimeIndex(
        df.loc[
            (df["sensor_id"] == reference_sensor) & df["value"].notna(),
            "timestamp",
        ].drop_duplicates()
    ).sort_values()
    if reference_times.empty:
        raise ValueError(f"Reference sensor {reference_sensor!r} has no observations.")
    frequency_ns = int(round(frequency_seconds * 1_000_000_000.0))
    if frequency_ns <= 0:
        raise ValueError("Inferred alignment interval is below timestamp precision.")
    reference_ns = _datetime_index_nanoseconds(reference_times)
    start_ns = int(reference_ns[0])
    end_ns = int(reference_ns[-1])
    steps = (end_ns - start_ns) // frequency_ns + 1
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
    return wide


def load_streaming_records(
    records: Iterable[Dict],
    freq: str = "auto",
    min_window: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    """Convert long-format streaming records to the wide tensor used downstream.

    Parameters
    ----------
    records : iterable of dict
        Each dict has {timestamp, sensor_id, value}.
    freq : str
        ``"auto"`` selects the shortest reliably inferred sensor cadence.
        A fixed pandas duration can be supplied explicitly for diagnostics.
    min_window : int, optional
        If given, raise ValueError when the resampled grid has fewer rows.

    Returns
    -------
    wide_df : DataFrame
        Resampled wide table with one column per sensor.
    values  : ndarray, shape [T, N, 1]
    mask    : ndarray, shape [T, N, 1], 1.0 where observed.
    sensors : list[str]
    grid_metadata : dict
        Per-sensor inferred intervals, fastest sensor identity, and grid rate.
    """
    records = list(records)
    df = normalize_streaming_records(records)
    inference = infer_sampling_intervals(records)
    if str(freq).strip().lower() == "auto":
        frequency_seconds = float(inference["fastest_interval_seconds"])
        frequency_source = "fastest_sensor"
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
    )
    for sensor_id in sensors:
        if wide[sensor_id].notna().sum() == 0:
            raise ValueError(
                f"Sensor '{sensor_id}' has no observations after resampling."
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
        "alignment_frequency": pd.Timedelta(seconds=frequency_seconds).isoformat(),
        "grid_start": wide.index[0].isoformat(),
        "grid_end": wide.index[-1].isoformat(),
        "grid_rows": len(wide),
    }
    return wide, values, mask, sensors, metadata


class SensorBuffer:
    """In-memory buffer for streaming sensor records.

    The backend appends one record at a time via ``add``. ``is_ready`` reports
    whether the buffered data, once resampled to ``freq``, has at least
    ``min_window`` time steps (the ImputeFormer minimum). ``flush`` converts
    the buffer to the wide tensor expected by the rest of the pipeline.
    """

    def __init__(self, freq: str = "auto", min_window: int = 3600,
                 sensor_filter: Optional[List[str]] = None) -> None:
        self.freq = freq
        self.min_window = min_window
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
