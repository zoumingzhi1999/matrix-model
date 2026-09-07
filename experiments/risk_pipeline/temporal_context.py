"""Physical-time inputs shared by sensor training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


RAW_TEMPORAL_FEATURE_NAMES = [
    "elapsed_seconds",
    "delta_seconds",
    "second_of_day_sin",
    "second_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


def timestamp_nanoseconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Validate a chronological time index and retain nanosecond precision."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Sensor input must use a DatetimeIndex.")
    if index.tz is None:
        raise ValueError("Sensor timestamps must include a timezone.")
    if not len(index):
        raise ValueError("Sensor input needs at least one timestamp.")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Sensor timestamps must be unique and chronological.")
    return index.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)


def relative_time_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Return elapsed seconds from the first timestamp, with no grid assumption."""
    timestamps = timestamp_nanoseconds(index)
    # Subtract integer epoch timestamps before conversion to preserve short gaps.
    return (timestamps - timestamps[0]).astype("float64") / 1_000_000_000.0


def adjacent_interval_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Return true preceding intervals in seconds; the first entry is zero."""
    timestamps = timestamp_nanoseconds(index)
    intervals = np.zeros(len(timestamps), dtype="float64")
    intervals[1:] = np.diff(timestamps).astype("float64") / 1_000_000_000.0
    return intervals


def raw_temporal_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Return raw elapsed/delta seconds and calendar phases as float64 [T, 6]."""
    elapsed = relative_time_seconds(index)
    intervals = adjacent_interval_seconds(index)
    second_of_day = (
        index.hour.to_numpy(dtype="float64") * 3600.0
        + index.minute.to_numpy(dtype="float64") * 60.0
        + index.second.to_numpy(dtype="float64")
        + index.microsecond.to_numpy(dtype="float64") / 1_000_000.0
        + index.nanosecond.to_numpy(dtype="float64") / 1_000_000_000.0
    )
    daily_phase = 2.0 * np.pi * second_of_day / 86_400.0
    seasonal_phase = 2.0 * np.pi * (
        index.dayofyear.to_numpy(dtype="float64") - 1.0 + second_of_day / 86_400.0
    ) / 365.2425
    return np.stack([
        elapsed,
        intervals,
        np.sin(daily_phase),
        np.cos(daily_phase),
        np.sin(seasonal_phase),
        np.cos(seasonal_phase),
    ], axis=-1)


def observation_gap_seconds(
    index: pd.DatetimeIndex,
    visible_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return seconds to previous/next visible observation for each channel.

    ``visible_mask`` has shape ``[time, ...]``. Observed entries have zero gap;
    no observation in a direction is represented by positive infinity. Only
    the supplied visible mask is used, so training targets cannot leak in.
    """
    timestamps = timestamp_nanoseconds(index)
    visible = np.asarray(visible_mask)
    if visible.ndim < 1 or visible.shape[0] != len(index):
        raise ValueError("Visible mask must have shape [time, ...] matching timestamps.")
    if not np.isin(visible, [False, True]).all():
        raise ValueError("Visible mask must contain only boolean or binary values.")
    observed = visible.astype(bool, copy=False)
    positions = np.arange(len(index), dtype="int64").reshape(
        (len(index),) + (1,) * (observed.ndim - 1)
    )
    previous = np.maximum.accumulate(np.where(observed, positions, -1), axis=0)
    following = np.minimum.accumulate(
        np.where(observed, positions, len(index))[::-1], axis=0
    )[::-1]
    current_times = timestamps.reshape(positions.shape)
    previous_gaps = (
        current_times - timestamps[np.maximum(previous, 0)]
    ).astype("float64") / 1_000_000_000.0
    following_gaps = (
        timestamps[np.minimum(following, len(index) - 1)] - current_times
    ).astype("float64") / 1_000_000_000.0
    previous_gaps[previous < 0] = np.inf
    following_gaps[following == len(index)] = np.inf
    return previous_gaps, following_gaps
