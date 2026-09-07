"""Fresh, variable-length ImputeFormer training with chronological evaluation."""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_sensor_imputeformer import (
    BASELINE_METHODS, MASK_PATTERNS, SENSOR_COLUMNS, SENSOR_UNITS,
    TEMPORAL_FEATURE_NAMES, _baseline_prediction, _method_metrics, _empty_sums,
    _update_method_sums, artifact_timestamp, build_model, fit_normalizer,
    load_ocean_sensor_splits, make_artificial_missing_mask, masked_l1,
    seed_everything, sha256_file, time_features,
)


LOGGER = logging.getLogger(__name__)
PATTERNS = (*MASK_PATTERNS, "multirate")


def prepare_scales(frame, mean, std, intervals):
    """Resample measured values within one split; empty bins stay unobserved."""
    base = float((frame.index[1] - frame.index[0]).total_seconds())
    prepared = {}
    for factor in intervals:
        interval = base * factor
        scaled = frame if factor == 1 else frame.resample(
            pd.Timedelta(seconds=interval), origin=frame.index[0]
        ).mean()
        raw = scaled.to_numpy(dtype="float32")
        observed = np.isfinite(raw)
        values = np.where(observed, (raw - mean) / std, 0.0).astype("float32")
        prepared[factor] = {
            "values": values, "observed": observed,
            "temporal": time_features(scaled.index, interval),
            "index": scaled.index, "interval_seconds": interval,
            "centers": [np.flatnonzero(observed[:, sensor]) for sensor in range(5)],
        }
    return prepared


def sample_batch(scales, factor, length, batch_size, rng, return_starts=False):
    data = scales[factor]
    length = min(length, len(data["values"]))
    available = [centers for centers in data["centers"] if len(centers)]
    starts = []
    for _ in range(batch_size):
        centers = available[int(rng.integers(len(available)))]
        center = int(centers[int(rng.integers(len(centers)))])
        start = min(max(0, center - int(rng.integers(length))), len(data["values"]) - length)
        starts.append(start)
    x = torch.from_numpy(np.stack([data["values"][s:s + length] for s in starts]))[..., None]
    temporal = torch.from_numpy(np.stack([data["temporal"][s:s + length] for s in starts]))
    observed = torch.from_numpy(np.stack([data["observed"][s:s + length] for s in starts]))[..., None]
    result = (x, temporal, observed)
    return (*result, starts) if return_starts else result


def artificial_mask(observed, rng, pattern):
    if pattern != "multirate":
        return make_artificial_missing_mask(observed, rng, pattern)
    target = np.zeros(observed.shape, dtype=bool)
    observed_np = observed.numpy()
    for sample in range(len(target)):
        reference = int(rng.integers(target.shape[2]))
        for sensor in range(target.shape[2]):
            stride = 1 if sensor == reference else int(rng.integers(2, 31))
            phase = int(rng.integers(stride))
            hidden = np.arange(target.shape[1]) % stride != phase
            target[sample, :, sensor, 0] = hidden & observed_np[sample, :, sensor, 0]
        if not target[sample].any():
            positions = np.argwhere(observed_np[sample])
            if len(positions):
                target[sample][tuple(positions[int(rng.integers(len(positions)))])] = True
    return torch.from_numpy(target)


def new_metric_store():
    return {"overall": _empty_sums(),
            "normalized_sensor": [_empty_sums() for _ in SENSOR_COLUMNS],
            "physical_sensor": [_empty_sums() for _ in SENSOR_COLUMNS]}


def evaluate(model, scales, device, mean, std, seed, batches, batch_size, window, factors):
    results = []
    model.eval()
    with torch.inference_mode():
        for factor in factors:
            for pattern_index, pattern in enumerate(PATTERNS):
                rng = np.random.default_rng(seed + factor * 101 + pattern_index)
                stores = {method: new_metric_store() for method in ("imputeformer", *BASELINE_METHODS)}
                unique_targets = [set() for _ in SENSOR_COLUMNS]
                no_context_samples = 0
                for batch in range(batches):
                    length = (min(window, 100), window)[batch % 2]
                    x, temporal, observed, starts = sample_batch(scales, factor, length, batch_size, rng, return_starts=True)
                    artificial = artificial_mask(observed, rng, pattern)
                    visible = observed & ~artificial
                    no_context_samples += int((~visible.flatten(1).any(1)).sum())
                    for sample, start in enumerate(starts):
                        for sensor in range(5):
                            unique_targets[sensor].update((np.flatnonzero(artificial[sample, :, sensor, 0].numpy()) + start).tolist())
                    prediction = model(x.to(device), temporal.to(device), visible.to(device).float()).cpu().numpy()
                    x_np, target_np, visible_np = x.numpy(), artificial.numpy(), visible.numpy()
                    for method in stores:
                        estimate = prediction if method == "imputeformer" else _baseline_prediction(x_np, visible_np, method)
                        _update_method_sums(stores[method], estimate, x_np, target_np, mean, std)
                results.append({
                    "sampling_interval_seconds": scales[factor]["interval_seconds"],
                    "pattern": pattern,
                    "unique_target_counts": dict(zip(SENSOR_COLUMNS, map(len, unique_targets))),
                    "samples_without_any_visible_context": no_context_samples,
                    "methods": {
                        method: _method_metrics(store["overall"], store["normalized_sensor"], store["physical_sensor"])
                        for method, store in stores.items()
                    },
                })
    return results


def validation_score(results):
    for result in results:
        if any(count == 0 for count in result["unique_target_counts"].values()):
            raise ValueError("Validation lacks measured targets for a sensor; increase validation-batches.")
    metrics = [r["methods"]["imputeformer"]["normalized_sensor_macro"] for r in results]
    return float(np.mean([m["MAE"] + 0.25 * m["RMSE"] for m in metrics]))


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if any(getattr(args, name, None) for name in (
        "initial_selection_state", "resume_selection_run", "selection_evaluation_only"
    )):
        raise ValueError("Fresh adaptive training does not accept old model weights or selection runs.")
    splits, manifest, dataset_run = load_ocean_sensor_splits(
        Path(args.dataset_root), Path(args.dataset_run) if args.dataset_run else None
    )
    mean, std = fit_normalizer(splits["train"].to_numpy(dtype="float32"))
    factors = tuple(sorted(set([1, *args.sampling_factors])))
    scales = {name: prepare_scales(frame, mean, std, factors) for name, frame in splits.items()}
    config = {
        "window": args.window, "inference_stride": args.inference_stride,
        "num_nodes": 5, "input_dim": 6, "output_dim": 1,
        "input_embedding_dim": args.input_embedding_dim,
        "learnable_embedding_dim": args.learnable_embedding_dim,
        "feed_forward_dim": args.feed_forward_dim,
        "num_temporal_heads": args.num_temporal_heads, "num_layers": args.num_layers,
        "dim_proj": args.dim_proj, "dropout": args.dropout,
        "variable_length": True, "use_interpolation_residual": True,
    }
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model = build_model(5, config).to(device)
    torch.nn.init.zeros_(model.readout.net[-1].weight)
    torch.nn.init.zeros_(model.readout.net[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(2, args.patience // 3), factor=0.5)
    run_id = artifact_timestamp()
    output = Path(args.output_root).resolve() / "runs" / run_id
    model_dir = Path(args.model_root).resolve() / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output / "evaluation_metrics.json"
    LOGGER.info("Fresh ImputeFormer device=%s parameters=%s dataset=%s rows=%s", device,
                sum(p.numel() for p in model.parameters()), dataset_run, {k: len(v) for k, v in splits.items()})
    history, best_score, best_epoch, stale = [], math.inf, 0, 0
    best_state = None
    rng = np.random.default_rng(args.seed)
    lengths = sorted(set(min(args.window, length) for length in [64, 100, 256, 1000, args.window]))
    probabilities = np.full(len(factors), 0.3 / max(len(factors) - 1, 1))
    probabilities[0] = 0.7 if len(factors) > 1 else 1.0
    audit = {
        "status": "training", "dataset_run": str(dataset_run),
        "dataset_manifest_sha256": sha256_file(dataset_run / "dataset_manifest.json"),
        "split_rows": {k: len(v) for k, v in splits.items()},
        "seed": args.seed, "random_initialization": True,
        "optimizer": "AdamW", "initial_learning_rate": args.lr,
        "weight_decay": args.weight_decay, "gradient_clip": args.grad_clip,
        "maximum_epochs": args.max_epochs, "minimum_epochs": args.min_epochs,
        "early_stopping_patience": args.patience, "minimum_validation_improvement": args.min_delta,
        "validation_batches_per_pattern_and_scale": args.validation_batches,
        "test_batches_per_pattern_and_scale": args.test_batches,
        "training_sampling_intervals_seconds": [scales["train"][f]["interval_seconds"] for f in factors],
        "training_lengths": lengths,
        "sampling_policy": "70% native cadence, 30% measured-bin aggregates; sensor-balanced random observed-centered windows",
        "epoch_steps": args.steps_per_epoch, "batch_size": args.batch_size,
        "all_rows_available_for_sampling": True, "every_row_seen_each_epoch": False,
        "missing_bins_used_as_ground_truth": False,
        "normalization_fit_split": "train", "selection_split": "validation",
        "refit_after_test": False,
        "source_sensor_audit": manifest.get("sensor_audit"),
        "history": history,
    }
    for epoch in range(1, args.max_epochs + 1):
        started = time.monotonic()
        model.train()
        losses = []
        for step in range(args.steps_per_epoch):
            factor = int(rng.choice(factors, p=probabilities))
            length = int(rng.choice(lengths))
            x, temporal, observed = sample_batch(scales["train"], factor, length, args.batch_size, rng)
            pattern = str(rng.choice(PATTERNS))
            artificial = artificial_mask(observed, rng, pattern)
            visible = observed & ~artificial
            x, temporal = x.to(device), temporal.to(device)
            artificial, visible = artificial.to(device), visible.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, temporal, visible.float())
            loss = masked_l1(prediction, x, artificial)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 32 == 0:
                LOGGER.info("epoch=%s step=%s/%s loss=%.6f", epoch, step + 1, args.steps_per_epoch, np.mean(losses))
        validation = evaluate(model, scales["validation"], device, mean, std, args.seed + 100000,
                              args.validation_batches, args.batch_size, args.window, [1, factors[-1]])
        score = validation_score(validation)
        scheduler.step(score)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_MAE_plus_0.25_RMSE": score,
                        "seconds": time.monotonic() - started, "learning_rate": optimizer.param_groups[0]["lr"]})
        LOGGER.info("epoch=%s validation_score=%.6f seconds=%.1f", epoch, score, history[-1]["seconds"])
        if score < best_score - args.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_dir / "best_selection_state.pt")
            audit["selected_validation_results"] = validation
        else:
            stale += 1
        audit["best_epoch"] = best_epoch
        metrics_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation score.")
    model.load_state_dict(best_state)
    test = evaluate(model, scales["test"], device, mean, std, args.seed + 200000,
                    args.test_batches, args.batch_size, args.window, factors)
    audit.update({
        "status": "complete", "independent_test": test,
        "evaluation_policy": "Fixed seeded observed-centered windows on chronological held-out splits; overlapping sampled windows may repeat measured targets. Model and baselines use identical hidden cells, and interpolation only sees visible values.",
        "metric_definitions": {"MAE": "mean absolute error", "MSE": "mean squared error", "RMSE": "sqrt(MSE)",
                               "MRE": "sum(abs(error))/sum(abs(target)); null when denominator is zero"},
        "generalization_limit": "Hourly aggregates are evaluated; arbitrary field sensor rates and distributions require field validation. The public source has only one genuinely second-rate sensor.",
    })
    checkpoint = {
        "format_version": 4, "model_type": "ImputeFormerModel", "entity_type": "sensor",
        "uses_site_entity": False, "uses_coordinates": False,
        "model_state_dict": best_state, "model_config": config,
        "sensor_ids": list(SENSOR_COLUMNS), "sensor_units": SENSOR_UNITS,
        "temporal_feature_names": list(TEMPORAL_FEATURE_NAMES), "temporal_encoding": "bounded_log_interval_v1",
        "supported_sampling_interval_seconds": {"minimum": 0.0, "maximum": None},
        "normalization": {"mean": mean, "std": std}, "output_calibration": None,
        "training": {k: v for k, v in audit.items() if k not in ("independent_test", "history", "selected_validation_results")},
        "evaluation_metrics": str(metrics_path),
    }
    checkpoint_path = model_dir / "imputeformer_ocean_five_sensor.pt"
    torch.save(checkpoint, checkpoint_path)
    (model_dir / "best_selection_state.pt").unlink()
    audit["checkpoint_sha256"] = sha256_file(checkpoint_path)
    metrics_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    active = Path(args.model_root).resolve() / "imputeformer_ocean_five_sensor.pt"
    temporary = active.with_suffix(".pt.tmp")
    shutil.copyfile(checkpoint_path, temporary)
    temporary.replace(active)
    LOGGER.info("Published evaluated fresh checkpoint=%s sha256=%s metrics=%s", active, audit["checkpoint_sha256"], metrics_path)
    return active
