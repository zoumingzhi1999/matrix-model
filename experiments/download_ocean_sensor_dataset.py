"""Download and prepare the full USGS MATANZAS_18 five-sensor dataset.

The source deployment contains one genuinely high-rate pressure-wave stream
(2 Hz) and four slower ocean measurements. Pressure is aggregated to one
second and converted to sensor immersion depth. Slower measurements remain
missing between their real observation timestamps; no synthetic observation
is written into the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from risk_pipeline.sensor_schema import OCEAN_SENSOR_IDS, OCEAN_SENSOR_UNITS


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "MATANZAS_18-1110-second"
DATASET_DOI = "https://doi.org/10.5066/P9GKB537"
CATALOG_URL = (
    "https://cmgds.marine.usgs.gov/thredds/catalog/"
    "ocean-time-series/Stellwagen/MATANZAS_18/catalog.html"
)
FILE_BASE_URL = (
    "https://cmgds.marine.usgs.gov/thredds/fileServer/"
    "ocean-time-series/Stellwagen/MATANZAS_18"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "ocean_time_series_matanzas_18_second"
GRID_FREQUENCY = "1s"
GRID_FREQUENCY_SECONDS = 1.0
DOWNLOAD_CHUNK_BYTES = 16 * 1024 * 1024
EPIC_UNIX_EPOCH_JULIAN_DAY = 2_440_588
DEFAULT_SEAWATER_DENSITY_KG_M3 = 1025.0
DEFAULT_GRAVITY_M_S2 = 9.793

SOURCE_FILES: Mapping[str, Mapping[str, object]] = {
    "pressure_wave": {
        "filename": "111010sgr-cal.nc",
        "variables": ["time", "time2", "P_1ac"],
        "role": "2 Hz corrected pressure converted to 1 s immersion depth",
        "native_interval_seconds": 0.5,
        "expected_bytes": 248_681_144,
    },
    "wave_statistics": {
        "filename": "111010sgw-cal.nc",
        "variables": ["time", "time2", "wh_4061"],
        "role": "official significant wave height derived from each pressure burst",
        "native_interval_seconds": 3600.0,
        "expected_bytes": 1_030_208,
    },
    "turbidity": {
        "filename": "111011slb-a.nc",
        "variables": ["time", "time2", "Turb"],
        "role": "optical turbidity",
        "native_interval_seconds": 1200.0,
        "expected_bytes": 117_808,
    },
    "temperature": {
        "filename": "11105mc-a.nc",
        "variables": ["time", "time2", "T_28"],
        "role": "seawater temperature",
        "native_interval_seconds": 300.0,
        "expected_bytes": 459_104,
    },
    "current": {
        "filename": "11102sigs_burst-cal.nc",
        "variables": ["time", "time2", "CS_300"],
        "role": "burst-mean current speed averaged over valid depth bins",
        "native_interval_seconds": 3600.0,
        "expected_bytes": 849_770,
    },
}


def artifact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_bytes: int) -> None:
    if destination.exists():
        if destination.stat().st_size == expected_bytes:
            return
        raise FileExistsError(f"Dataset file has the wrong size: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    parts_dir = destination.with_suffix(destination.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable dataset downloads.")
    part_specs = []
    for start in range(0, expected_bytes, DOWNLOAD_CHUNK_BYTES):
        end = min(start + DOWNLOAD_CHUNK_BYTES, expected_bytes) - 1
        expected_part_bytes = end - start + 1
        part_path = parts_dir / f"{start:012d}-{end:012d}.part"
        part_specs.append((start, end, expected_part_bytes, part_path))

    def fetch_part(spec) -> Path:
        start, end, expected_part_bytes, part_path = spec
        if part_path.is_file() and part_path.stat().st_size == expected_part_bytes:
            return part_path
        if part_path.exists():
            part_path.unlink()
        downloading = part_path.with_suffix(".download")
        if downloading.exists():
            downloading.unlink()
        print(f"  bytes {start:,}-{end:,} of {expected_bytes:,}", flush=True)
        subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "8",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--max-time",
                "600",
                "--range",
                f"{start}-{end}",
                "--output",
                str(downloading),
                url,
            ],
            check=True,
        )
        actual_part_bytes = downloading.stat().st_size
        if actual_part_bytes != expected_part_bytes:
            raise RuntimeError(
                f"Incomplete range for {url}: expected {expected_part_bytes} "
                f"bytes, got {actual_part_bytes}."
            )
        downloading.replace(part_path)
        return part_path

    with ThreadPoolExecutor(max_workers=min(4, len(part_specs))) as executor:
        part_paths = list(executor.map(fetch_part, part_specs))

    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as output:
        for part_path in part_paths:
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    actual_bytes = temporary.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Incomplete download for {url}: expected {expected_bytes} bytes, "
            f"got {actual_bytes}. The .part file is retained for resume."
        )
    temporary.replace(destination)
    shutil.rmtree(parts_dir)


def _read_array(path: Path, variable: str) -> np.ndarray:
    with path.open("rb") as stream:
        is_hdf5 = stream.read(8) == b"\x89HDF\r\n\x1a\n"
    if is_hdf5:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "h5py is required to read the NetCDF4 current-speed source."
            ) from exc
        with h5py.File(path, mode="r") as dataset:
            if variable not in dataset:
                raise ValueError(f"{path.name} has no variable {variable!r}.")
            source = dataset[variable]
            array = np.asarray(source[...]).copy()
            fill_value = source.attrs.get("_FillValue")
            missing_value = source.attrs.get("missing_value")
    else:
        with netcdf_file(path, mode="r", mmap=False) as dataset:
            if variable not in dataset.variables:
                raise ValueError(f"{path.name} has no variable {variable!r}.")
            source = dataset.variables[variable]
            array = np.asarray(source.data).copy()
            fill_value = getattr(source, "_FillValue", None)
            missing_value = getattr(source, "missing_value", None)
    if np.issubdtype(array.dtype, np.number):
        array = array.astype("float64", copy=False)
        for sentinel in (fill_value, missing_value):
            if sentinel is not None:
                array[array == float(np.asarray(sentinel).reshape(-1)[0])] = np.nan
        array[array <= -9990.0] = np.nan
    return array


def epic_epoch_seconds(day: np.ndarray, milliseconds: np.ndarray) -> np.ndarray:
    """Convert USGS EPIC time/time2 fields to Unix seconds."""
    day = np.asarray(day, dtype="int64")
    milliseconds = np.asarray(milliseconds, dtype="int64")
    if day.shape != milliseconds.shape:
        raise ValueError("EPIC time and time2 arrays must have identical shapes.")
    return (
        (day - EPIC_UNIX_EPOCH_JULIAN_DAY) * 86_400
        + milliseconds // 1000
    )


def _series_from_epic(
    path: Path,
    variable: str,
    *,
    reduction: str = "scalar",
    scale: float = 1.0,
) -> pd.Series:
    day = np.squeeze(_read_array(path, "time"))
    milliseconds = np.squeeze(_read_array(path, "time2"))
    values = np.squeeze(_read_array(path, variable))
    if reduction == "depth_mean":
        if values.ndim != 2:
            raise ValueError(
                f"{path.name}:{variable} must be [time, depth], got {values.shape}."
            )
        values = np.nanmean(values, axis=1)
    elif reduction != "scalar":
        raise ValueError(f"Unknown reduction: {reduction}")
    if day.ndim != 1 or values.ndim != 1 or day.shape != values.shape:
        raise ValueError(
            f"{path.name}:{variable} has incompatible shapes "
            f"time={day.shape}, values={values.shape}."
        )
    seconds = epic_epoch_seconds(day, milliseconds)
    valid = np.isfinite(values)
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(seconds[valid], unit="s", utc=True),
        "value": values[valid] * scale,
    })
    series = frame.groupby("timestamp", sort=True)["value"].mean()
    if series.empty:
        raise ValueError(f"{path.name}:{variable} has no valid observations.")
    return series.astype("float32")


def _depth_from_pressure(
    path: Path,
    seawater_density_kg_m3: float,
    gravity_m_s2: float,
) -> pd.Series:
    day = _read_array(path, "time").reshape(-1)
    milliseconds = _read_array(path, "time2").reshape(-1)
    pressure_dbar = _read_array(path, "P_1ac").reshape(-1)
    valid = np.isfinite(pressure_dbar)
    seconds = epic_epoch_seconds(day[valid], milliseconds[valid])
    pressure_pa = pressure_dbar[valid] * 10_000.0
    immersion_depth_m = pressure_pa / (
        seawater_density_kg_m3 * gravity_m_s2
    )

    order = np.argsort(seconds, kind="stable")
    seconds = seconds[order]
    immersion_depth_m = immersion_depth_m[order]
    starts = np.r_[0, np.flatnonzero(np.diff(seconds)) + 1]
    counts = np.diff(np.r_[starts, len(seconds)])
    sums = np.add.reduceat(immersion_depth_m, starts)
    per_second = sums / counts
    index = pd.to_datetime(seconds[starts], unit="s", utc=True)
    series = pd.Series(per_second.astype("float32"), index=index)
    series.index.name = "timestamp"
    series.name = "water_depth_m"
    if series.empty or series.index.has_duplicates:
        raise ValueError("Pressure conversion did not produce unique 1-second depth data.")
    return series


def _observation_audit(series: pd.Series, expected_interval: float) -> dict:
    deltas = series.index.to_series().diff().dropna().dt.total_seconds()
    positive = deltas[deltas > 0]
    mode = positive.round(3).mode()
    return {
        "observations": int(series.notna().sum()),
        "first_observation": series.first_valid_index().isoformat(),
        "last_observation": series.last_valid_index().isoformat(),
        "expected_native_interval_seconds": expected_interval,
        "most_common_observed_delta_seconds": (
            float(mode.iloc[0]) if not mode.empty else None
        ),
    }


def load_sensor_series(
    raw_dir: Path,
    seawater_density_kg_m3: float,
    gravity_m_s2: float,
) -> dict[str, pd.Series]:
    depth = _depth_from_pressure(
        raw_dir / str(SOURCE_FILES["pressure_wave"]["filename"]),
        seawater_density_kg_m3,
        gravity_m_s2,
    )
    temperature = _series_from_epic(
        raw_dir / str(SOURCE_FILES["temperature"]["filename"]), "T_28"
    ).rename("water_temperature_c")
    turbidity = _series_from_epic(
        raw_dir / str(SOURCE_FILES["turbidity"]["filename"]), "Turb"
    ).rename("turbidity_ntu")
    wave_height = _series_from_epic(
        raw_dir / str(SOURCE_FILES["wave_statistics"]["filename"]), "wh_4061"
    ).rename("significant_wave_height_m")
    current_speed = _series_from_epic(
        raw_dir / str(SOURCE_FILES["current"]["filename"]),
        "CS_300",
        reduction="depth_mean",
        scale=0.01,
    ).rename("current_speed_mps")
    result = {
        "water_temperature_c": temperature,
        "water_depth_m": depth,
        "turbidity_ntu": turbidity,
        "significant_wave_height_m": wave_height,
        "current_speed_mps": current_speed,
    }
    if tuple(result) != OCEAN_SENSOR_IDS:
        raise RuntimeError("Prepared sensor order does not match the production schema.")
    return result


def build_fastest_sensor_grid(
    series: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, dict]:
    fastest_sensor = "water_depth_m"
    reference = series[fastest_sensor]
    index = pd.date_range(
        reference.index[0].floor(GRID_FREQUENCY),
        reference.index[-1].floor(GRID_FREQUENCY),
        freq=GRID_FREQUENCY,
        tz="UTC",
    )
    grid = pd.DataFrame(index=index)
    for sensor_id in OCEAN_SENSOR_IDS:
        grid[sensor_id] = series[sensor_id].reindex(index)
    grid.index.name = "timestamp"
    if grid.isna().all(axis=0).any():
        missing = grid.columns[grid.isna().all(axis=0)].tolist()
        raise ValueError(f"Sensors without observations on the 1-second grid: {missing}")
    audit = {
        "alignment_policy": "all sensors reindexed to the fastest sensor's 1-second timestamps",
        "fastest_sensor_id": fastest_sensor,
        "fastest_source_interval_seconds": 0.5,
        "grid_interval_seconds": GRID_FREQUENCY_SECONDS,
        "grid_start": index[0].isoformat(),
        "grid_end": index[-1].isoformat(),
        "grid_rows": len(grid),
        "observed_counts": {
            column: int(grid[column].notna().sum()) for column in OCEAN_SENSOR_IDS
        },
        "missing_rates": {
            column: float(grid[column].isna().mean()) for column in OCEAN_SENSOR_IDS
        },
        "fabricated_observations": 0,
    }
    return grid, audit


def chronological_splits(
    grid: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict[str, pd.DataFrame]:
    if not (0 < train_ratio < 1 and 0 < validation_ratio < 1):
        raise ValueError("Split ratios must be between zero and one.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Training and validation ratios must leave a test split.")
    train_end = int(len(grid) * train_ratio)
    validation_end = train_end + int(len(grid) * validation_ratio)
    splits = {
        "train": grid.iloc[:train_end].copy(),
        "validation": grid.iloc[train_end:validation_end].copy(),
        "test": grid.iloc[validation_end:].copy(),
    }
    if min(map(len, splits.values())) < 3600:
        raise ValueError("A chronological split contains fewer than 3,600 seconds.")
    return splits


def run(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    run_dir = (
        Path(args.resume_run).resolve()
        if args.resume_run
        else output_root / "runs" / artifact_timestamp()
    )
    raw_dir = run_dir / "raw"
    split_dir = run_dir / "splits"
    if args.resume_run:
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Resume run has no raw directory: {raw_dir}")
        if (run_dir / "dataset_manifest.json").exists():
            raise FileExistsError(f"Dataset run is already complete: {run_dir}")
        split_dir.mkdir(parents=True, exist_ok=True)
    else:
        raw_dir.mkdir(parents=True, exist_ok=False)
        split_dir.mkdir(parents=True, exist_ok=False)

    raw_manifest = {}
    for source_name, spec in SOURCE_FILES.items():
        filename = str(spec["filename"])
        url = f"{FILE_BASE_URL}/{filename}"
        destination = raw_dir / filename
        print(f"Downloading complete source file {filename}...", flush=True)
        download_file(url, destination, int(spec["expected_bytes"]))
        raw_manifest[source_name] = {
            **spec,
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    series = load_sensor_series(
        raw_dir,
        args.seawater_density_kg_m3,
        args.gravity_m_s2,
    )
    sensor_audit = {
        "water_temperature_c": _observation_audit(series["water_temperature_c"], 300.0),
        "water_depth_m": _observation_audit(series["water_depth_m"], 1.0),
        "turbidity_ntu": _observation_audit(series["turbidity_ntu"], 1200.0),
        "significant_wave_height_m": _observation_audit(
            series["significant_wave_height_m"], 3600.0
        ),
        "current_speed_mps": _observation_audit(series["current_speed_mps"], 3600.0),
    }

    native_rows = []
    for sensor_id, values in series.items():
        frame = values.rename("value").reset_index()
        frame["sensor_id"] = sensor_id
        native_rows.append(frame[["timestamp", "sensor_id", "value"]])
    native = pd.concat(native_rows, ignore_index=True).sort_values(
        ["timestamp", "sensor_id"]
    )
    native_path = run_dir / "native_sensor_observations.parquet"
    native.to_parquet(native_path, index=False, compression="zstd")

    grid, grid_audit = build_fastest_sensor_grid(series)
    dataset_path = run_dir / "ocean_five_sensor_1s_grid.parquet"
    grid.to_parquet(dataset_path, index=True, compression="zstd")
    splits = chronological_splits(grid, args.train_ratio, args.validation_ratio)
    split_manifest = {}
    for name, frame in splits.items():
        path = split_dir / f"{name}.parquet"
        frame.to_parquet(path, index=True, compression="zstd")
        split_manifest[name] = {
            "path": str(path),
            "rows": len(frame),
            "start": frame.index[0].isoformat(),
            "end": frame.index[-1].isoformat(),
            "observed_counts": {
                column: int(frame[column].notna().sum())
                for column in OCEAN_SENSOR_IDS
            },
            "sha256": sha256_file(path),
        }

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_title": "USGS MATANZAS_18 shallow-platform multi-rate ocean sensor data",
        "provider": "U.S. Geological Survey Coastal and Marine Geology Program",
        "doi": DATASET_DOI,
        "catalog_url": CATALOG_URL,
        "entity_type": "sensor",
        "uses_site_entity": False,
        "uses_coordinates": False,
        "model_columns": ["timestamp", *OCEAN_SENSOR_IDS],
        "sensor_units": OCEAN_SENSOR_UNITS,
        "depth_definition": (
            "vertical immersion depth from sea surface to the pressure sensor, "
            "positive downward"
        ),
        "depth_conversion": {
            "formula": "depth_m = pressure_dbar * 10000 / (rho_kg_m3 * g_m_s2)",
            "seawater_density_kg_m3": args.seawater_density_kg_m3,
            "gravity_m_s2": args.gravity_m_s2,
        },
        "raw_source_files": raw_manifest,
        "sensor_audit": sensor_audit,
        "grid_audit": grid_audit,
        "processed_dataset": {
            "path": str(dataset_path),
            "rows": len(grid),
            "sha256": sha256_file(dataset_path),
        },
        "native_observations": {
            "path": str(native_path),
            "rows": len(native),
            "sha256": sha256_file(native_path),
        },
        "split_policy": "chronological 70% train, 15% validation, 15% independent test",
        "splits": split_manifest,
        "data_integrity": {
            "all_source_files_downloaded_in_full": True,
            "source_rows_truncated": False,
            "slow_sensor_values_interpolated_as_observations": False,
            "missing_cells_preserved": True,
        },
    }
    manifest_path = run_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "run_directory": str(run_dir),
        "grid_rows": len(grid),
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "manifest": str(manifest_path),
    }, indent=2, ensure_ascii=False))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the complete USGS MATANZAS_18 second-level dataset."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Resume preprocessing from an incomplete run with verified raw files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument(
        "--seawater-density-kg-m3",
        type=float,
        default=DEFAULT_SEAWATER_DENSITY_KG_M3,
    )
    parser.add_argument("--gravity-m-s2", type=float, default=DEFAULT_GRAVITY_M_S2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
