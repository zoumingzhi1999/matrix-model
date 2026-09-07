"""Prepare native-time ocean data with same-platform measured high-rate current.

The Vector source contains 16 Hz bursts. This script selects every sixteenth
sample, without interpolating or averaging it, and preserves its stored time.
The other four channels remain the existing measured ocean observations.
Requires the optional downloader dependency pydap>=3.5.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pydap.handlers.dap import BytesReader, unpack_dap2_data
from pydap.parsers.dds import dds_to_dataset
import requests


SOURCE_FILENAME = "11109vecb-cal.nc"
SOURCE_URL = (
    "https://cmgds.marine.usgs.gov/thredds/dodsC/"
    "ocean-time-series/Stellwagen/MATANZAS_18/" + SOURCE_FILENAME
)
CURRENT = "current_speed_mps"
VARIABLES = ("time", "time2", "u_1205", "v_1206", "COR1_1285", "COR2_1286", "COR3_1287")
SENSORS = (
    "water_temperature_c", "water_depth_m", "turbidity_ntu",
    "significant_wave_height_m", CURRENT,
)
BURST_COUNT = 1895
SAMPLES_PER_BURST = 32768
SAMPLE_STRIDE = 16
EPIC_UNIX_DAY = 2440588


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> bytes:
    if destination.is_file():
        return destination.read_bytes()
    last_error = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(30, 180))
            response.raise_for_status()
            result = response.content
            if not result.startswith((b"Dataset {", b"Attributes {")):
                raise ValueError(f"Unexpected DAP response from {url}")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(result)
            temporary.replace(destination)
            return result
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}") from last_error


def decode_dods(payload: bytes) -> dict[str, np.ndarray]:
    header, body = payload.split(b"\nData:\n", 1)
    template = dds_to_dataset(header.decode("ascii"))
    decoded = unpack_dap2_data(BytesReader(body), template)
    return dict(zip(template.keys(), decoded, strict=True))


def fetch_current_chunk(spec, raw_dir: Path, correlation_threshold: float):
    first, last = spec
    selection = f"[{first}:1:{last}][0:{SAMPLE_STRIDE}:{SAMPLES_PER_BURST - 1}]"
    query = ",".join(variable + selection for variable in VARIABLES)
    destination = raw_dir / f"current_{first:04d}_{last:04d}_stride16.dods"
    payload = fetch(SOURCE_URL + ".dods?" + query, destination)
    arrays = decode_dods(payload)
    expected_shape = (last - first + 1, SAMPLES_PER_BURST // SAMPLE_STRIDE)
    if set(arrays) != set(VARIABLES) or any(a.shape != expected_shape for a in arrays.values()):
        raise ValueError(f"Incomplete OPeNDAP subset: {destination}")
    u = arrays["u_1205"].ravel().astype("float32")
    v = arrays["v_1206"].ravel().astype("float32")
    finite = np.isfinite(u) & np.isfinite(v) & (np.abs(u) < 1e30) & (np.abs(v) < 1e30)
    correlation = np.minimum.reduce([
        arrays[variable].ravel().astype("float32") for variable in VARIABLES[-3:]
    ])
    quality = np.isfinite(correlation) & (correlation >= correlation_threshold) & (correlation <= 100)
    keep = finite & quality
    timestamp_ms = (
        (arrays["time"].ravel().astype("int64") - EPIC_UNIX_DAY) * 86400000
        + arrays["time2"].ravel().astype("int64")
    )
    values = np.hypot(u[keep], v[keep]) / 100.0
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamp_ms[keep], unit="ms", utc=True),
        CURRENT: values.astype("float32"),
    })
    audit = {
        "file": str(destination.resolve()), "url": SOURCE_URL + ".dods?" + query,
        "burst_first": first, "burst_last": last, "sample_stride": SAMPLE_STRIDE,
        "bytes": len(payload), "sha256": file_hash(destination),
        "selected_samples": int(keep.size), "accepted_samples": int(keep.sum()),
        "nonfinite_or_fill_rejected": int((~finite).sum()),
        "low_or_invalid_correlation_rejected": int((finite & ~quality).sum()),
    }
    print(f"current bursts {first:04d}-{last:04d}: accepted {keep.sum():,}/{keep.size:,}", flush=True)
    return frame, audit


def source_metadata(raw_dir: Path) -> dict:
    das = fetch(SOURCE_URL + ".das", raw_dir / (SOURCE_FILENAME + ".das"))
    dds = fetch(SOURCE_URL + ".dds", raw_dir / (SOURCE_FILENAME + ".dds"))
    metadata = das.decode("utf-8", errors="replace")
    required = ('String MOORING "1110"', 'String DESCRIPTION "Shallow site Quadpod"',
                'String User_setup_Sampling_rate "16 Hz"', 'Float64 latitude 29.7114')
    if any(value not in metadata for value in required):
        raise ValueError("Remote current source deployment or sampling metadata changed.")
    template = dds_to_dataset(dds.decode("ascii"))
    if template["u_1205"].shape != (BURST_COUNT, SAMPLES_PER_BURST):
        raise ValueError("Remote current source dimensions changed.")
    return {
        "source_url": SOURCE_URL, "metadata_url": SOURCE_URL + ".das",
        "dds_sha256": hashlib.sha256(dds).hexdigest(),
        "das_sha256": hashlib.sha256(das).hexdigest(),
        "mooring": "1110", "platform": "Shallow site Quadpod",
        "latitude_for_provenance_only": 29.7114,
        "longitude_for_provenance_only": -81.218633,
        "sampling_rate_hz": 16, "burst_interval_seconds": 3600,
        "samples_per_burst": SAMPLES_PER_BURST,
        "measurement_definition": "Instantaneous horizontal Vector velocity magnitude at approximately 0.55 m above the seabed",
        "unit_conversion": "sqrt(u_1205**2 + v_1206**2) / 100; cm/s to m/s",
        "selection": "Every 16th stored sample; no averaging or interpolation; millisecond source timestamps preserved",
        "source_qc": "Source history records thumbfinger removal at 5 standard deviations about the mean and geographic rotation",
        "changed_from_parent_definition": "Parent current was the hourly burst mean of depth-averaged Signature current; these are measured instantaneous near-bed Vector samples",
        "serial_number_caution": "Source global and variable serial-number strings disagree; same-mooring and coordinates are independently consistent",
    }


def build_dataset(args):
    parent_run = args.parent_run.resolve()
    run_dir = args.run_dir.resolve()
    if (run_dir / "dataset_manifest.json").exists():
        raise FileExistsError(f"Completed dataset already exists: {run_dir}")
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parent_manifest_path = parent_run / "dataset_manifest.json"
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    metadata = source_metadata(raw_dir)
    chunks = [(first, min(first + args.chunk_bursts, BURST_COUNT) - 1)
              for first in range(0, BURST_COUNT, args.chunk_bursts)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(
            lambda spec: fetch_current_chunk(spec, raw_dir, args.minimum_correlation), chunks
        ))
    current = pd.concat([frame for frame, _ in results], ignore_index=True).sort_values("timestamp")
    if current["timestamp"].duplicated().any():
        raise ValueError("The source current contains duplicate timestamps.")
    native_path = parent_run / "native_sensor_observations.parquet"
    original = pd.read_parquet(native_path)
    original = original[original["sensor_id"] != CURRENT]
    if original.duplicated(["timestamp", "sensor_id"]).any():
        raise ValueError("Parent observations contain duplicate sensor timestamps.")
    four_sensors = original.pivot(index="timestamp", columns="sensor_id", values="value")
    merged = four_sensors.join(current.set_index("timestamp"), how="outer").sort_index()
    merged = merged.loc[:, list(SENSORS)].astype("float32")
    merged.index = merged.index.as_unit("ns")
    merged.columns.name = None
    if merged.index.has_duplicates or not merged.index.is_monotonic_increasing:
        raise ValueError("Merged native timeline is invalid.")
    train_end = pd.Timestamp(parent["splits"]["train"]["end"])
    validation_end = pd.Timestamp(parent["splits"]["validation"]["end"])
    assignments = {
        "train": merged.index <= train_end,
        "validation": (merged.index > train_end) & (merged.index <= validation_end),
        "test": merged.index > validation_end,
    }
    split_dir = run_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    splits = {}
    for name, selector in assignments.items():
        frame = merged.loc[selector]
        if frame.empty or (frame.notna().sum() == 0).any():
            raise ValueError(f"Missing measured channel in {name} split.")
        path = split_dir / f"{name}.parquet"
        frame.reset_index().to_parquet(path, index=False)
        splits[name] = {
            "path": str(path), "rows": len(frame),
            "start": frame.index[0].isoformat(), "end": frame.index[-1].isoformat(),
            "observed_counts": {name: int(count) for name, count in frame.notna().sum().items()},
            "sha256": file_hash(path),
        }
        print(f"saved {name}: {len(frame):,} native timestamps", flush=True)
    dataset_path = run_dir / "ocean_five_sensor_native.parquet"
    merged.reset_index().to_parquet(dataset_path, index=False)
    sensor_audit = {}
    for sensor in SENSORS:
        valid = merged[sensor].dropna()
        deltas = np.diff(valid.index.asi8) / 1e9
        sensor_audit[sensor] = {
            "observations": len(valid), "first_observation": valid.index[0].isoformat(),
            "last_observation": valid.index[-1].isoformat(),
            "median_observed_delta_seconds": float(np.median(deltas)),
            "most_common_observed_delta_seconds": float(pd.Series(deltas).mode().iloc[0]),
        }
    audits = [audit for _, audit in results]
    manifest = {
        "schema_version": 3, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "MATANZAS_18-1110-native-vector-current-v2",
        "dataset_title": "USGS MATANZAS_18 same-platform native-time ocean sensors with measured Vector current",
        "provider": parent["provider"], "doi": parent["doi"], "catalog_url": parent["catalog_url"],
        "entity_type": "sensor", "uses_site_entity": False, "uses_coordinates": False,
        "model_columns": ["timestamp", *SENSORS], "sensor_units": parent["sensor_units"],
        "depth_definition": parent["depth_definition"], "depth_conversion": parent["depth_conversion"],
        "time_axis_policy": "Sorted union of measured channel timestamps; absent values stay NaN; no synthetic regular grid",
        "native_timestamps_preserved": True,
        "native_timestamp_scope": "New current preserves original EPIC timestamps; four inherited channels preserve the parent's native_observations timestamps (depth inherited as measured 1-second aggregates)",
        "grid_audit": {"grid_interval_seconds": None, "grid_rows": len(merged),
                       "fabricated_observations": 0, "alignment_policy": "native timestamp outer union"},
        "parent_dataset": {"run": str(parent_run), "manifest_sha256": file_hash(parent_manifest_path),
                           "native_observations_sha256": file_hash(native_path)},
        "inherited_source_files": {key: value for key, value in parent["raw_source_files"].items() if key != "current"},
        "current_source": metadata,
        "current_quality_control": {
            "minimum_all_beam_correlation_percent": args.minimum_correlation,
            "correlation_rule": "Reject if any beam is below threshold or above 100%, or any u/v is nonfinite or source fill (abs>=1e30)",
            "threshold_scope": "Conservative configurable selection; not an independent claim of instrument calibration accuracy",
            **{key: sum(audit[key] for audit in audits) for key in (
                "selected_samples", "accepted_samples", "nonfinite_or_fill_rejected", "low_or_invalid_correlation_rejected")},
        },
        "downloaded_current_subsets": audits,
        "sensor_audit": sensor_audit,
        "processed_dataset": {"path": str(dataset_path), "rows": len(merged), "sha256": file_hash(dataset_path)},
        "split_policy": "Parent chronological train/validation/test boundary times preserved; newly measured current samples obey those boundaries",
        "test_independence_note": "Same deployment and previously inspected test period: a comparable development benchmark, not a new never-seen deployment holdout",
        "splits": splits,
        "data_integrity": {"all_requested_current_subsets_downloaded": True,
                           "slow_sensor_values_interpolated_as_observations": False,
                           "missing_cells_preserved": True, "cross_platform_pairing": False,
                           "high_rate_seawater_temperature_or_turbidity_claimed": False},
    }
    manifest_path = run_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_run": str(run_dir), "sensor_audit": sensor_audit,
                      "quality_control": manifest["current_quality_control"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunk-bursts", type=int, default=48)
    parser.add_argument("--minimum-correlation", type=float, default=70.0)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_bursts < 1 or not 0 <= args.minimum_correlation <= 100:
        parser.error("workers/chunk-bursts must be positive and correlation must be in [0,100]")
    build_dataset(args)


if __name__ == "__main__":
    main()
