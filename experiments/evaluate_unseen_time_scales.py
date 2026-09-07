"""Evaluate a fixed checkpoint on unseen real-time scales and window lengths.

The source is the existing chronological test split. Coarser values are means
of measured values within real time bins, and empty bins remain unobserved.
Irregular cases remove existing timeline rows; timestamps are never jittered.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import torch

from train_sensor_imputeformer import (
    SENSOR_COLUMNS, build_model, load_ocean_sensor_splits, sha256_file,
)
from train_time_aware_imputeformer import (
    CONTEXTS, PATTERNS, evaluate, evaluation_cases, prepare_scales,
)


LOGGER = logging.getLogger(__name__)
UNSEEN_INTERVALS_SECONDS = (3.0, 15.0, 120.0, 1800.0)
UNSEEN_LENGTHS = (137, 777, 2501)
BASE_SEED = 700042
METHODS = ("imputeformer", "linear_interpolation", "forward_backward_hold", "train_sensor_mean")


def summarize(cases, context=None):
    result = {}
    for method in METHODS:
        metrics = []
        for case in cases:
            methods = case["methods"] if context is None else case["context_strata"][context]
            metrics.append(methods[method]["normalized_sensor_macro"])
        result[method] = {
            name: float(np.mean(values)) if (values := [
                float(item[name]) for item in metrics
                if item.get(name) is not None and np.isfinite(item[name])
            ]) else None
            for name in ("MAE", "RMSE")
        }
        result[method]["cases_with_MAE_targets"] = int(sum(
            item.get("MAE") is not None and np.isfinite(item["MAE"]) for item in metrics
        ))
    return result


def scale_audit(scales):
    audit = {}
    for interval, data in scales.items():
        index = data["index"]
        ns = index.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
        deltas = np.diff(ns).astype("float64") / 1e9
        audit["native" if interval == 0 else str(interval)] = {
            "interval_seconds": interval or None,
            "rows": len(index),
            "start": index[0].isoformat(),
            "end": index[-1].isoformat(),
            "observed_counts": {
                sensor: int(data["observed"][:, i].sum())
                for i, sensor in enumerate(SENSOR_COLUMNS)
            },
            "all_unobserved_rows": int((~data["observed"].any(axis=1)).sum()),
            "minimum_delta_seconds": float(deltas.min()) if len(deltas) else None,
            "median_delta_seconds": float(np.median(deltas)) if len(deltas) else None,
            "maximum_delta_seconds": float(deltas.max()) if len(deltas) else None,
            "requested_lengths_available": {
                str(length): min(length, len(index)) for length in UNSEEN_LENGTHS
            },
        }
    return audit


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.cpu_threads)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("sensor_ids") != list(SENSOR_COLUMNS):
        raise ValueError("Checkpoint sensor ordering differs from the five-sensor schema.")
    if not checkpoint["model_config"].get("time_aware"):
        raise ValueError("This evaluation expects a checkpoint with physical-time inputs.")
    report_path = Path(checkpoint["evaluation_metrics"])
    training_report = json.loads(report_path.read_text(encoding="utf-8"))
    if training_report.get("status") != "complete":
        raise ValueError("Finish checkpoint selection, calibration and the main evaluation first.")
    checkpoint_hash = sha256_file(checkpoint_path)
    if training_report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint hash differs from the completed training report.")
    trained_intervals = set(training_report["training_sampling_intervals_seconds"])
    trained_lengths = set(training_report["training_lengths"])
    if trained_intervals.intersection(UNSEEN_INTERVALS_SECONDS):
        raise ValueError("One of this evaluation's interval scales appeared in training.")
    if trained_lengths.intersection(UNSEEN_LENGTHS):
        raise ValueError("One of this evaluation's requested lengths appeared in training.")
    dataset_run = Path(training_report["dataset_run"])
    manifest_hash = sha256_file(dataset_run / "dataset_manifest.json")
    if manifest_hash != training_report["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed since checkpoint training.")
    splits, manifest, _ = load_ocean_sensor_splits(
        dataset_run.parent.parent, dataset_run, require_regular=False
    )
    mean = np.asarray(checkpoint["normalization"]["mean"], dtype="float32")
    std = np.asarray(checkpoint["normalization"]["std"], dtype="float32")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Checkpoint contains invalid normalization.")
    scales = prepare_scales(splits["test"], mean, std, UNSEEN_INTERVALS_SECONDS)
    del splits
    intervals = (0.0, *UNSEEN_INTERVALS_SECONDS)
    cases = evaluation_cases(intervals, UNSEEN_LENGTHS)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model = build_model(5, checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    modes = {}
    for mode_index, (name, irregular) in enumerate((
        ("original_timeline_rows", False), ("randomly_removed_timeline_rows", True)
    )):
        seed = args.seed + mode_index * 100000
        LOGGER.info("Unseen scales mode=%s cases=%d repeats=%d seed=%d device=%s",
                    name, len(cases), args.repeats, seed, device)
        results, _ = evaluate(
            model, scales, device, mean, std, seed, cases, repeats=args.repeats,
            batch_size=args.batch_size, collect_calibration=False, irregular=irregular,
        )
        clipped_cases = [
            {
                "sampling_interval_seconds": case["sampling_interval_seconds"],
                "requested_length": case["requested_length"],
                "actual_lengths": case["actual_lengths"],
                "pattern": case["pattern"],
            }
            for case in results
            if any(length < case["requested_length"] for length in case["actual_lengths"])
        ]
        modes[name] = {
            "seed": seed, "irregular_row_subsampling": irregular,
            "case_count": len(results), "clipped_length_cases": clipped_cases,
            "summary": summarize(results),
            "by_same_sensor_context": {context: summarize(results, context) for context in CONTEXTS},
            "by_sampling_interval": {
                "native" if interval == 0 else str(interval): summarize([
                    case for case in results if (case["sampling_interval_seconds"] or 0.0) == interval
                ]) for interval in intervals
            },
            "evaluation_cases": results,
        }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "training_evaluation_metrics": str(report_path),
        "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": manifest_hash,
        "source_split": "test",
        "unseen_sampling_intervals_seconds": list(UNSEEN_INTERVALS_SECONDS),
        "unseen_requested_lengths": list(UNSEEN_LENGTHS),
        "native_timestamp_reference_included": True,
        "mask_patterns": list(PATTERNS),
        "repeats": args.repeats,
        "maximum_batch_size": args.batch_size,
        "normalization": "Frozen checkpoint parameters fitted on the original training split",
        "checkpoint_selection_or_calibration_performed": False,
        "timestamp_and_target_policy": [
            "Native mode uses original test timestamps and their measured values, without adding or shifting timestamps.",
            "Each unseen regular scale uses time bins anchored at the test timeline start; targets are means of real measurements inside each bin. Empty cells remain unobserved and are never scored as ground truth.",
            "Irregular mode samples an ordered subset of existing rows within a window up to 1.3 times the requested length. Existing timestamps are retained exactly; no synthetic temporal jitter or interpolation targets are created.",
            "All methods within one mode and case share the same hidden measured targets. The two modes use independent seeds and different target sets, so their aggregate errors are not a paired estimate of the effect of irregularity.",
        ],
        "metric_definition": "Each case averages sensor MAE and RMSE after division by that sensor's frozen train-only standard deviation; summaries average the case metrics equally where targets exist.",
        "generalization_limits": [
            "The four nonzero nominal scales and requested lengths were absent from the discrete training configuration. Their physical intervals lie inside the overall trained time range.",
            "Native timestamps already appeared as a training mode and are a reference condition, not an unseen cadence.",
            "At long intervals, the finite test period can contain fewer rows than requested. actual_lengths and clipped_length_cases disclose these cases; unavailable long sequences are not fabricated.",
            "This is a previously inspected deployment period. New cadence and length combinations do not constitute an independent new field deployment.",
            "Overlapping sampled windows can repeat targets; each case reports per-sensor unique target counts.",
        ],
        "current_measurement_definition": manifest.get("current_source", {}).get("measurement_definition"),
        "test_scale_audit": scale_audit(scales),
        "modes": modes,
    }
    output = args.output.resolve() if args.output else report_path.with_name("unseen_time_scales_evaluation.json")
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("Saved unseen-scale evaluation=%s summaries=%s", output,
                {name: result["summary"] for name, result in modes.items()})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if min(args.cpu_threads, args.repeats, args.batch_size) < 1:
        parser.error("cpu-threads, repeats and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
