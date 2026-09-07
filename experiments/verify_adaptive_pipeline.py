"""Verify the published model and full pipeline against local real input files."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from full_pipeline.run_streaming_pipeline import run_imputeformer_alignment_from_records
from risk_pipeline.artifacts import artifact_timestamp
from risk_pipeline.model_bundle import ProductionModelBundle
from risk_pipeline.run_encapsulated_pipeline import run as run_representation
from risk_pipeline.sensor_imputation import DEFAULT_SENSOR_CHECKPOINT, align_with_pretrained_imputeformer
from risk_pipeline.sensor_schema import OCEAN_SENSOR_IDS
from risk_pipeline.temporal_context import adjacent_interval_seconds, timestamp_nanoseconds


def _dataset_run_from_checkpoint(checkpoint, requested=None):
    declared = requested or checkpoint.get("data", {}).get("dataset_run")
    declared = declared or checkpoint.get("training", {}).get("dataset_run")
    if not declared:
        raise ValueError("Checkpoint must declare data.dataset_run, or supply --dataset-run.")
    path = Path(declared)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _retime(frame, interval_seconds=None):
    """Construct contract probes from real values, without claiming rate accuracy."""
    result = frame.copy()
    if interval_seconds is None:
        gaps_ns = np.resize(
            np.array([200_000, 1_000_000_000, 3_000_000_000, 200_000_000,
                      30_000_000_000, 3_600_000_000_000], dtype="int64"),
            max(0, len(frame) - 1),
        )
    else:
        gap_ns = int(round(interval_seconds * 1_000_000_000))
        if gap_ns < 1:
            raise ValueError("Functional test interval must be at least one nanosecond.")
        gaps_ns = np.full(max(0, len(frame) - 1), gap_ns, dtype="int64")
    offsets = np.concatenate([np.zeros(1, dtype="int64"), np.cumsum(gaps_ns)])
    result.index = pd.to_datetime(timestamp_nanoseconds(frame.index)[0] + offsets, utc=True)
    result.index.name = "timestamp"
    return result


def _verify_frame(bundle, frame, case, timestamp_provenance, evaluate=False):
    result, metadata = align_with_pretrained_imputeformer(
        frame, bundle.sensor_checkpoint_path, preloaded_model=bundle.sensor_model,
        preloaded_checkpoint=bundle.sensor_checkpoint,
        evaluation_holdout_ratio=0.1 if evaluate else 0.0,
    )
    assert len(result) == len(frame) and np.isfinite(result.to_numpy()).all(), case
    pd.testing.assert_index_equal(result.index, frame.index)
    assert list(result.columns) == list(frame.columns), case
    valid = np.isfinite(frame.to_numpy())
    np.testing.assert_array_equal(result.to_numpy()[valid], frame.to_numpy()[valid])
    deltas = adjacent_interval_seconds(frame.index)[1:]
    return {
        "case": case,
        "rows": len(frame),
        "output_rows": len(result),
        "timestamp_provenance": timestamp_provenance,
        "interval_seconds": float(deltas[0]) if len(deltas) and (deltas == deltas[0]).all() else None,
        "actual_interval_seconds_min": float(deltas.min()) if len(deltas) else None,
        "actual_interval_seconds_max": float(deltas.max()) if len(deltas) else None,
        "observations_preserved": True,
        "timestamps_preserved": True,
        "inference_windows": metadata["inference_windows"],
        "metrics": metadata.get("evaluation_metrics"),
    }


def _summarize_independent_test(rows):
    summary, unscored = [], []
    for row in rows:
        identity = {
            "sampling_interval_seconds": row["sampling_interval_seconds"],
            "missing_pattern": row["pattern"],
            "requested_length": row.get("requested_length"),
            "actual_lengths": row.get("actual_lengths"),
        }
        methods = {
            name: metrics["normalized_sensor_macro"]
            for name, metrics in row["methods"].items()
        }
        scored = {
            name: metrics for name, metrics in methods.items()
            if all(metrics.get(metric) is not None and np.isfinite(metrics[metric]) for metric in ("MAE", "RMSE"))
        }
        baselines = [metrics for name, metrics in scored.items() if name != "imputeformer"]
        if "imputeformer" not in scored or not baselines:
            unscored.append({**identity, "status": "no_evaluable_targets_or_baseline"})
            continue
        summary.append({
            **identity,
            "imputeformer": scored["imputeformer"],
            "best_baseline_MAE": min(metrics["MAE"] for metrics in baselines),
            "best_baseline_RMSE": min(metrics["RMSE"] for metrics in baselines),
            "unique_measured_targets_per_sensor": row.get("unique_target_counts"),
        })
    return summary, unscored


def run(args):
    torch.set_num_threads(4)
    bundle = ProductionModelBundle.load(
        sensor_checkpoint_path=getattr(args, "sensor_checkpoint", DEFAULT_SENSOR_CHECKPOINT),
        device=args.device,
    )
    dataset_run = _dataset_run_from_checkpoint(bundle.sensor_checkpoint, args.dataset_run)
    source = pd.read_parquet(dataset_run / "splits" / "test.parquet")
    if "timestamp" in source:
        source = source.set_index("timestamp")
    source.index = pd.to_datetime(source.index, format="mixed", utc=True)
    source.index.name = "timestamp"
    timestamp_nanoseconds(source.index)
    source = source.loc[:, OCEAN_SENSOR_IDS].replace([np.inf, -np.inf], np.nan)
    lengths = tuple(getattr(args, "lengths", (1, 2, 100, 256, 500, 1000, 2000, 3600, 7217)))
    pipeline_length = int(getattr(args, "pipeline_length", 7217))
    if not lengths or min(lengths) < 1 or pipeline_length < 1:
        raise ValueError("Verification lengths must be positive.")
    required_rows = max(*lengths, pipeline_length, 256)
    if len(source) < required_rows:
        raise ValueError(f"Held-out test split has {len(source)} rows; need {required_rows} for verification.")
    observed_counts = source.notna().sum(axis=1).to_numpy()
    candidates = np.flatnonzero(observed_counts)
    if not len(candidates) or candidates[0] + required_rows > len(source):
        raise ValueError("Held-out test split has no usable observed verification window.")
    native_start = int(candidates[0])
    time_aware = bool(bundle.sensor_checkpoint["model_config"].get("time_aware"))
    checks = []
    for length in lengths:
        frame = source.iloc[native_start:native_start + length]
        checks.append(_verify_frame(bundle, frame, "native_length", "native_held_out_timestamps", evaluate=True))
    hourly = source.resample("1h", origin=source.index[0]).mean()
    hourly_observed = np.flatnonzero(hourly.notna().any(axis=1).to_numpy())
    if len(hourly_observed):
        hourly = hourly.iloc[int(hourly_observed[0]):int(hourly_observed[0]) + 100]
        checks.append(_verify_frame(bundle, hourly, "hourly_aggregates", "hourly_aggregated_held_out_data", evaluate=True))
    functional_frame = source.iloc[native_start:native_start + 256]
    for interval in (0.0002, 1, 2, 5, 10, 30, 60, 3600):
        checks.append(_verify_frame(
            bundle, _retime(functional_frame, interval), "sampling_interval_contract",
            "constructed_timestamps_real_values_functional_only",
        ))
    if time_aware:
        for length in lengths:
            checks.append(_verify_frame(
                bundle, _retime(source.iloc[native_start:native_start + length]),
                "irregular_length_contract", "constructed_timestamps_real_values_functional_only",
            ))
    pipeline_frame = source.iloc[native_start:native_start + pipeline_length]
    if time_aware:
        pipeline_frame = _retime(pipeline_frame)
    frame = pipeline_frame.reset_index()
    records = frame.melt(id_vars="timestamp", value_vars=list(OCEAN_SENSOR_IDS),
                         var_name="sensor_id", value_name="value").to_dict(orient="records")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    timestamp = artifact_timestamp()
    pipeline_args = SimpleNamespace(
        freq="auto", batch_size=8, device=args.device, artifact_timestamp=timestamp,
        evaluation_holdout_ratio=0.1, evaluation_seed=42,
        sensor_checkpoint=str(bundle.sensor_checkpoint_path),
    )
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        aligned_path, _, _, alignment_metadata = run_imputeformer_alignment_from_records(
            records, pipeline_args, output, model_bundle=bundle
        )
        with redirect_stdout(io.StringIO()):
            risk = run_representation(SimpleNamespace(
                imputeformer_output=str(aligned_path), output_dir=str(output),
                optical_image=args.optical_image, sonar_xtf=getattr(args, "sonar_xtf", None),
                sonar_image=getattr(args, "sonar_image", None),
                risk_config=str(ROOT / "config" / "fixed_single_feature_risk_config.json"),
                fusion_checkpoint=str(bundle.fusion_checkpoint_path), artifact_timestamp=timestamp,
            ), model_bundle=bundle)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)
    files = [p for p in output.rglob("*") if p.is_file()]
    assert len(files) == 9, files
    features = sorted((output / "features").glob("*.csv"))
    assert len(features) == 6
    for path in features:
        assert path.read_text(encoding="utf-8-sig") in log_capture.getvalue(), path
    document = json.loads(Path(risk["risk_json_path"]).read_text(encoding="utf-8"))
    assert set(document["modalities"]) == {"time_series", "optical_image", "sonar_image"}
    weighted = sum(item["risk"] * item["fixed_weight"] for item in document["modalities"].values())
    assert np.isclose(document["total_risk"], weighted)
    aligned_saved = pd.read_csv(aligned_path, float_precision="round_trip")
    saved_index = pd.DatetimeIndex(pd.to_datetime(aligned_saved["timestamp"], format="mixed", utc=True))
    np.testing.assert_array_equal(timestamp_nanoseconds(saved_index), timestamp_nanoseconds(pipeline_frame.index))
    np.testing.assert_array_equal(
        aligned_saved.loc[:, OCEAN_SENSOR_IDS].to_numpy()[pipeline_frame.notna().to_numpy()],
        pipeline_frame.to_numpy()[pipeline_frame.notna().to_numpy()],
    )
    report = {
        "checkpoint_sha256": alignment_metadata["checkpoint_sha256"],
        "checkpoint_format_version": bundle.sensor_checkpoint.get("format_version"),
        "time_aware": time_aware,
        "dataset_run": str(dataset_run),
        "functional_checks_on_real_held_out_data": checks,
        "constructed_timestamp_checks_are_accuracy_evidence": False,
        "irregular_timestamp_verification": "passed" if time_aware else "not_supported_by_legacy_checkpoint",
        "end_to_end": {"output_dir": str(output), "files": len(files), "feature_logs_match_files": True,
                       "aligned_rows": alignment_metadata["input_rows"], "risk_json": str(risk["risk_json_path"]),
                       "timestamps_preserved": True, "irregular_timestamps": time_aware,
                       "grid_policy": alignment_metadata["grid_alignment"]["grid_policy"]},
        "inputs_are_unpaired_public_samples": True,
    }
    checkpoint_metrics = Path(bundle.sensor_checkpoint["evaluation_metrics"])
    if not checkpoint_metrics.is_absolute():
        checkpoint_metrics = ROOT / checkpoint_metrics
    evaluation = json.loads(checkpoint_metrics.read_text(encoding="utf-8"))
    if isinstance(evaluation.get("test_summary"), dict):
        evaluation.setdefault("calibration_test_summary", evaluation["test_summary"])
    evaluation["production_inference_verification"] = report
    evaluation["test_summary"], unscored = _summarize_independent_test(evaluation["independent_test"])
    evaluation["quality_assessment"] = {
        "field_accuracy_verified": False,
        "scored_conditions": len(evaluation["test_summary"]),
        "unscored_conditions": unscored,
        "all_test_conditions_beat_best_baseline": bool(evaluation["test_summary"]) and all(
            row["imputeformer"]["MAE"] < row["best_baseline_MAE"]
            and row["imputeformer"]["RMSE"] < row["best_baseline_RMSE"]
            for row in evaluation["test_summary"]
        ),
        "conditions_with_MAE_regression": [
            {"sampling_interval_seconds": row["sampling_interval_seconds"],
             "missing_pattern": row["missing_pattern"],
             "requested_length": row["requested_length"], "actual_lengths": row["actual_lengths"],
             "model_MAE": row["imputeformer"]["MAE"], "best_baseline_MAE": row["best_baseline_MAE"]}
            for row in evaluation["test_summary"]
            if row["imputeformer"]["MAE"] > row["best_baseline_MAE"]
        ],
        "interpretation": "Functionally verified. Accuracy varies by cadence and missingness; a trained model is not evidence of uniformly high field accuracy.",
    }
    checkpoint_metrics.write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"metrics": str(checkpoint_metrics), **report["end_to_end"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-run", help="Defaults to the dataset_run declared by the checkpoint.")
    parser.add_argument("--sensor-checkpoint", default=str(DEFAULT_SENSOR_CHECKPOINT))
    parser.add_argument("--optical-image", required=True)
    sonar = parser.add_mutually_exclusive_group(required=True)
    sonar.add_argument("--sonar-xtf")
    sonar.add_argument("--sonar-image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 100, 256, 500, 1000, 2000, 3600, 7217])
    parser.add_argument("--pipeline-length", type=int, default=7217)
    run(parser.parse_args())
