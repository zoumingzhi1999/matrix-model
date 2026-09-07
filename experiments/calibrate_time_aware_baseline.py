"""Select visible-median shrinkage on validation and publish evaluated weights."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import torch

from train_sensor_imputeformer import (
    SENSOR_COLUMNS,
    SENSOR_UNITS,
    _read_split,
    _split_path,
    artifact_timestamp,
    build_model,
    sha256_file,
)
from train_time_aware_imputeformer import (
    CONTEXTS,
    artificial_mask,
    context_classes,
    evaluate,
    evaluation_cases,
    prepare_scales,
    sample_batch,
    time_linear,
)


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "models" / "sensor_imputeformer" / "runs" / "20260906_155236_644869"
    / "imputeformer_ocean_five_sensor.pt"
)
MEDIAN_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0)


def collect_validation_samples(scales, cases, seed, repeats, batch_size):
    samples = [[[] for _ in CONTEXTS] for _ in SENSOR_COLUMNS]
    unique = [[set() for _ in CONTEXTS] for _ in SENSOR_COLUMNS]
    for case_index, (interval, length, pattern) in enumerate(cases):
        rng = np.random.default_rng(seed + case_index * 1009)
        for _ in range(repeats):
            effective_batch = max(
                1, min(batch_size, 4096 // min(length, len(scales[interval]["values"])))
            )
            x, temporal, observed, indices = sample_batch(
                scales, interval, length, effective_batch, rng
            )
            hidden = artificial_mask(observed, temporal, rng, pattern)
            visible = observed & ~hidden
            values, target, visible_np = x.numpy(), hidden.numpy(), visible.numpy()
            baseline = time_linear(values, visible_np, temporal.numpy())
            correction = np.zeros_like(values)
            for b in range(len(values)):
                for s in range(len(SENSOR_COLUMNS)):
                    measured = values[b, visible_np[b, :, s, 0], s, 0]
                    if len(measured) >= 3:
                        correction[b, :, s, 0] = np.median(measured) - baseline[b, :, s, 0]
            classes = context_classes(visible_np)
            for s in range(len(SENSOR_COLUMNS)):
                for c in range(len(CONTEXTS)):
                    selected = target[:, :, s, 0] & (classes[:, :, s, 0] == c)
                    samples[s][c].append(np.stack([
                        (baseline[:, :, s, 0] - values[:, :, s, 0])[selected],
                        correction[:, :, s, 0][selected],
                    ], axis=1).astype("float64"))
                    for b, rows in enumerate(indices):
                        unique[s][c].update(
                            (float(interval), int(row)) for row in rows[selected[b]]
                        )
        if (case_index + 1) % 54 == 0 or case_index + 1 == len(cases):
            LOGGER.info("Median calibration validation cases=%d/%d", case_index + 1, len(cases))
    return samples, unique


def select_median_gains(samples, unique):
    gains = np.zeros((len(SENSOR_COLUMNS), len(CONTEXTS)), dtype="float32")
    report = {}
    for s, sensor in enumerate(SENSOR_COLUMNS):
        report[sensor] = {}
        for c, context in enumerate(CONTEXTS):
            values = np.concatenate(samples[s][c]) if samples[s][c] else np.empty((0, 2))
            candidates = []
            for gain in MEDIAN_GAINS:
                error = values[:, 0] + gain * values[:, 1]
                candidates.append({
                    "gain": gain,
                    "MAE": float(np.abs(error).mean()) if len(error) else None,
                    "RMSE": float(np.sqrt(np.square(error).mean())) if len(error) else None,
                })
            baseline = candidates[0]
            eligible = [0]
            if c > 0 and len(values) >= 30 and len(unique[s][c]) >= 30:
                eligible.extend(i for i, candidate in enumerate(candidates[1:], start=1)
                                if candidate["MAE"] < baseline["MAE"] * 0.995
                                and candidate["RMSE"] <= baseline["RMSE"] + 1e-12)
            selected = min(eligible, key=lambda i: candidates[i]["MAE"]) if len(values) else 0
            gains[s, c] = candidates[selected]["gain"]
            report[sensor][context] = {
                "gain": float(gains[s, c]),
                "validation_target_count": len(values),
                "unique_validation_scale_timestamp_count": len(unique[s][c]),
                "linear_MAE": baseline["MAE"],
                "linear_RMSE": baseline["RMSE"],
                "calibrated_MAE": candidates[selected]["MAE"],
                "calibrated_RMSE": candidates[selected]["RMSE"],
                "candidates": candidates,
            }
    return gains, report


def select_no_context_prior(samples, unique, train_medians, mean, std):
    normalized_medians = (train_medians - mean) / std
    prior = np.zeros(len(SENSOR_COLUMNS), dtype="float32")
    report = {}
    for s, sensor in enumerate(SENSOR_COLUMNS):
        values = np.concatenate(samples[s][0]) if samples[s][0] else np.empty((0, 2))
        candidates = []
        for alpha in MEDIAN_GAINS:
            offset = alpha * float(normalized_medians[s])
            error = values[:, 0] + offset
            candidates.append({
                "alpha": alpha,
                "normalized_offset": offset,
                "physical_prior": float(mean[s] + offset * std[s]),
                "MAE": float(np.abs(error).mean()) if len(error) else None,
                "RMSE": float(np.sqrt(np.square(error).mean())) if len(error) else None,
            })
        baseline = candidates[0]
        eligible = [0]
        if len(values) >= 30 and len(unique[s][0]) >= 30:
            eligible.extend(i for i, candidate in enumerate(candidates[1:], start=1)
                            if candidate["MAE"] < baseline["MAE"] * 0.995
                            and candidate["RMSE"] <= baseline["RMSE"] + 1e-12)
        selected = min(eligible, key=lambda i: candidates[i]["MAE"]) if len(values) else 0
        prior[s] = candidates[selected]["normalized_offset"]
        report[sensor] = {
            "constant_fit_split": "train",
            "selection_split": "validation",
            "training_mean": float(mean[s]),
            "training_median": float(train_medians[s]),
            "unit": SENSOR_UNITS[sensor],
            "selected_alpha": candidates[selected]["alpha"],
            "normalized_offset": float(prior[s]),
            "physical_prior": candidates[selected]["physical_prior"],
            "validation_target_count": len(values),
            "unique_validation_scale_timestamp_count": len(unique[s][0]),
            "mean_fallback_MAE": baseline["MAE"],
            "mean_fallback_RMSE": baseline["RMSE"],
            "calibrated_MAE": candidates[selected]["MAE"],
            "calibrated_RMSE": candidates[selected]["RMSE"],
            "candidates": candidates,
        }
    return prior, report


def summarize_test(results, std):
    methods = {}
    for method in ("imputeformer", "linear_interpolation"):
        sensor_totals = {sensor: np.zeros(3, dtype="float64") for sensor in SENSOR_COLUMNS}
        case_mae = []
        for case in results:
            metrics = case["methods"][method]
            macro_mae = metrics["normalized_sensor_macro"]["MAE"]
            if macro_mae is not None:
                case_mae.append(macro_mae)
            for sensor, values in metrics.get("normalized_per_sensor", {}).items():
                n = values["target_count"]
                if n:
                    sensor_totals[sensor] += [n, values["MAE"] * n, values["MSE"] * n]
        per_sensor = {}
        for s, sensor in enumerate(SENSOR_COLUMNS):
            n, absolute, squared = sensor_totals[sensor]
            per_sensor[sensor] = {
                "target_count_with_repeated_windows": int(n),
                "standardized_MAE": float(absolute / n) if n else None,
                "standardized_RMSE": float(math.sqrt(squared / n)) if n else None,
                "physical_MAE": float(absolute / n * std[s]) if n else None,
                "unit": SENSOR_UNITS[sensor],
            }
        valid = [values for values in per_sensor.values() if values["standardized_MAE"] is not None]
        methods[method] = {
            "pooled_sensor_macro_MAE": float(np.mean([v["standardized_MAE"] for v in valid])),
            "pooled_sensor_macro_RMSE": float(np.sqrt(np.mean([v["standardized_RMSE"] ** 2 for v in valid]))),
            "mean_case_sensor_macro_MAE": float(np.mean(case_mae)),
            "per_sensor": per_sensor,
        }
    linear = methods["linear_interpolation"]["pooled_sensor_macro_MAE"]
    calibrated = methods["imputeformer"]["pooled_sensor_macro_MAE"]
    return {
        "methods": methods,
        "relative_pooled_sensor_macro_MAE_improvement": 1.0 - calibrated / linear if linear else None,
        "case_count": len(results),
        "aggregation": "Pool repeated target errors within each sensor across cases, then average sensors equally. Per-case unique target counts remain in independent_test.",
    }


def calibrate_checkpoint(
    checkpoint_path, output_root=None, model_root=None, device="auto", repeats=2,
    batch_size=2, cpu_threads=4,
):
    if repeats < 1 or batch_size < 1 or cpu_threads < 1:
        raise ValueError("repeats, batch_size and cpu_threads must be positive.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(cpu_threads)
    checkpoint_path = Path(checkpoint_path).resolve()
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(parent["model_config"])
    if not config.get("time_aware") or config.get("visible_median_calibration"):
        raise ValueError("Calibration requires a time-aware checkpoint without prior median calibration.")
    config["visible_median_calibration"] = True
    model = build_model(len(SENSOR_COLUMNS), config)
    incompatible = model.load_state_dict(parent["model_state_dict"], strict=False)
    if set(incompatible.missing_keys) != {"context_median_gain"} or incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint incompatibility: {incompatible}")
    if torch.count_nonzero(model.context_residual_gain).item():
        raise ValueError("This independent baseline calibration requires all neural residual gains to be zero.")
    model.context_median_gain.zero_()
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto" else device
    )
    model.to(resolved_device).eval()

    parent_metrics_path = Path(parent["evaluation_metrics"]).resolve()
    parent_report = json.loads(parent_metrics_path.read_text(encoding="utf-8"))
    dataset_run = Path(parent["data"]["dataset_run"]).resolve()
    manifest_hash = sha256_file(dataset_run / "dataset_manifest.json")
    if manifest_hash != parent_report["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed since neural training.")
    intervals = tuple(float(value) for value in parent_report["training_sampling_intervals_seconds"])
    lengths = tuple(int(value) for value in parent_report["training_lengths"])
    cases = evaluation_cases((0.0, *intervals), lengths)
    mean = np.asarray(parent["normalization"]["mean"], dtype="float32")
    std = np.asarray(parent["normalization"]["std"], dtype="float32")
    validation_seed = int(parent_report["seed"]) + 300000
    test_seed = int(parent_report["seed"]) + 200000

    LOGGER.info("Calibrating visible median from parent=%s device=%s cases=%d", checkpoint_path, resolved_device, len(cases))
    train_path = _split_path(dataset_run / "splits", "train")
    train_frame = _read_split(train_path)
    train_values = train_frame.to_numpy(dtype="float32")
    finite = np.isfinite(train_values)
    train_medians = np.nanmedian(np.where(finite, train_values, np.nan), axis=0)
    if not np.isfinite(train_medians).all():
        raise ValueError("Every sensor needs a finite training-only median.")
    prior_fit = {
        "split": "train", "source": str(train_path), "sha256": sha256_file(train_path),
        "normalization_parameters_unchanged": True,
        "observation_counts": dict(zip(SENSOR_COLUMNS, map(int, finite.sum(axis=0)))),
        "medians": dict(zip(SENSOR_COLUMNS, map(float, train_medians))),
    }
    del train_frame, train_values, finite
    validation_frame = _read_split(_split_path(dataset_run / "splits", "validation"))
    validation_end = validation_frame.index[-1]
    validation_scales = prepare_scales(validation_frame, mean, std, intervals)
    samples, unique = collect_validation_samples(validation_scales, cases, validation_seed, repeats, batch_size)
    gains, calibration_report = select_median_gains(samples, unique)
    prior, prior_report = select_no_context_prior(samples, unique, train_medians, mean, std)
    del samples, unique, validation_scales, validation_frame
    model.context_median_gain.copy_(torch.from_numpy(gains).to(resolved_device))
    model.no_context_prior = torch.from_numpy(prior).to(resolved_device).view(1, 1, len(SENSOR_COLUMNS), 1)
    config["no_context_prior"] = prior.tolist()
    LOGGER.info("Locked validation-selected median gains=%s; neural residual gains are all zero", gains.tolist())
    LOGGER.info("Locked training-only no-context prior normalized offsets=%s", prior.tolist())

    # Test labels are first loaded after every calibration choice is locked.
    test_frame = _read_split(_split_path(dataset_run / "splits", "test"))
    if not validation_end < test_frame.index[0]:
        raise ValueError("Validation and test timelines overlap.")
    test_scales = prepare_scales(test_frame, mean, std, intervals)
    test, _ = evaluate(
        model, test_scales, resolved_device, mean, std, test_seed, cases,
        repeats=repeats, batch_size=batch_size,
    )
    summary = summarize_test(test, std)
    run_id = artifact_timestamp()
    output_root = Path(output_root).resolve() if output_root is not None else parent_metrics_path.parents[2]
    model_root = Path(model_root).resolve() if model_root is not None else checkpoint_path.parents[2]
    output_dir = output_root / "runs" / run_id
    model_dir = model_root / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "evaluation_metrics.json"
    source = {
        "run_id": checkpoint_path.parent.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_metrics": str(parent_metrics_path),
    }
    audit = {key: value for key, value in parent_report.items() if key not in (
        "independent_test", "validation_before_calibration", "checkpoint_sha256", "test_summary",
    )}
    audit.update({
        "status": "complete",
        "run_id": run_id,
        "model_family": "time-aware ImputeFormer with separately calibrated visible-median shrinkage",
        "parent_evaluation_metrics": str(parent_metrics_path),
        "source_neural_training": source,
        "neural_residual_enabled": False,
        "neural_residual_gains": model.context_residual_gain.cpu().tolist(),
        "visible_median_calibration": calibration_report,
        "no_context_prior_fit": prior_fit,
        "no_context_prior_calibration": prior_report,
        "calibration_policy": {
            "split": "validation", "seed": validation_seed, "repeats": repeats,
            "cases": len(cases), "maximum_batch_size": batch_size, "gain_candidates": list(MEDIAN_GAINS),
            "minimum_target_count": 30, "minimum_unique_scale_timestamp_count": 30,
            "minimum_visible_observations": 3, "minimum_relative_MAE_improvement": 0.005,
            "RMSE_regression_allowed": False, "no_same_sensor_context_gain": 0.0,
            "median_definition": "Mean of the middle two sorted visible values for an even observation count; central value otherwise.",
            "no_context_prior": "Train-only mean-to-median mixture; alpha chosen on validation under the same MAE, RMSE and support thresholds.",
            "selection_uses_test_labels": False,
        },
        "improvement_attribution": "Neural residual gains are all zero. Changes from the time-linear baseline come exclusively from separately validation-selected visible-median shrinkage and train-fitted mean-to-median no-context priors, not learned Transformer residuals.",
        "method_definitions": {
            "imputeformer": "Published time-aware pipeline with visible-median calibration, training-only calibrated no-context priors, and disabled neural residuals.",
            "linear_interpolation": "Physical-time linear interpolation, constant observed edges, train mean without same-sensor observations.",
        },
        "independent_test": test,
        "test_seed": test_seed,
        "calibration_test_summary": summary,
        "refit_after_test": False,
    })
    checkpoint = copy.deepcopy(parent)
    checkpoint.update({
        "model_config": config,
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "model_family": audit["model_family"],
        "source_neural_training": source,
        "parent_evaluation_metrics": str(parent_metrics_path),
        "neural_residual_enabled": False,
        "visible_median_calibration": calibration_report,
        "no_context_prior_calibration": prior_report,
        "training_history": parent_report.get("history", []),
        "evaluation_metrics": str(metrics_path),
    })
    checkpoint["training"]["post_training_calibration"] = audit["calibration_policy"]
    checkpoint["training"]["no_context_prior_fit"] = prior_fit
    versioned_checkpoint = model_dir / "imputeformer_ocean_five_sensor.pt"
    torch.save(checkpoint, versioned_checkpoint)
    audit["checkpoint_sha256"] = sha256_file(versioned_checkpoint)
    metrics_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    active = model_root / "imputeformer_ocean_five_sensor.pt"
    temporary = active.with_name(f".{active.name}.{run_id}.tmp")
    try:
        shutil.copyfile(versioned_checkpoint, temporary)
        temporary.replace(active)
    finally:
        if temporary.exists():
            temporary.unlink()
    LOGGER.info("Published median-calibrated run=%s checkpoint=%s metrics=%s", run_id, active, metrics_path)
    LOGGER.info("Test summary=%s", json.dumps(summary, allow_nan=False))
    return active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    calibrate_checkpoint(
        args.checkpoint, args.output_root, args.model_root, args.device,
        args.repeats, args.batch_size, args.cpu_threads,
    )


if __name__ == "__main__":
    main()
