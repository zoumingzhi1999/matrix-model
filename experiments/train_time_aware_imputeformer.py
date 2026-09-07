"""Train on measured ocean observations with physical time and variable windows."""

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
    SENSOR_COLUMNS, SENSOR_UNITS, _baseline_prediction, _method_metrics,
    _empty_sums, _update_method_sums, artifact_timestamp, build_model,
    fit_normalizer, load_ocean_sensor_splits, seed_everything, sha256_file,
)
from risk_pipeline.temporal_context import RAW_TEMPORAL_FEATURE_NAMES, raw_temporal_features


LOGGER = logging.getLogger(__name__)
PATTERNS = ("point_20", "point_50", "temporal_block", "temporary_sensor_outage", "multirate", "single_sensor_outage")
CONTEXTS = ("no_same_sensor_context", "one_sided_context", "two_sided_context")


def prepare_scales(frame, mean, std, seconds):
    prepared = {}
    for interval in (0.0, *seconds):
        scaled = frame if interval == 0 else frame.resample(
            pd.Timedelta(seconds=interval), origin=frame.index[0]
        ).mean()
        raw = scaled.to_numpy(dtype="float32")
        observed = np.isfinite(raw)
        prepared[interval] = {
            "values": np.where(observed, (raw - mean) / std, 0.0).astype("float32"),
            "observed": observed, "index": scaled.index,
            "centers": [np.flatnonzero(observed[:, s]) for s in range(5)],
        }
    return prepared


def sample_batch(scales, interval, length, batch_size, rng, irregular=False):
    data = scales[interval]
    length = min(length, len(data["values"]))
    span = min(len(data["values"]), int(math.ceil(length * 1.3))) if irregular else length
    available = [s for s, centers in enumerate(data["centers"]) if len(centers)]
    indices = []
    for b in range(batch_size):
        centers = data["centers"][available[int(rng.integers(len(available)))]]
        center = int(rng.choice(centers))
        start = min(max(0, center - int(rng.integers(span))), len(data["values"]) - span)
        rows = np.arange(start, start + span)
        if irregular:
            rows = np.sort(rng.choice(rows, length, replace=False))
        indices.append(rows)
    x = torch.from_numpy(np.stack([data["values"][i] for i in indices]))[..., None]
    observed = torch.from_numpy(np.stack([data["observed"][i] for i in indices]))[..., None]
    temporal = torch.from_numpy(np.stack([raw_temporal_features(data["index"][i]) for i in indices]))
    return x, temporal, observed, indices


def artificial_mask(observed, temporal, rng, pattern):
    observed_np = observed.numpy()
    target = np.zeros_like(observed_np)
    for b in range(len(target)):
        valid = observed_np[b, ..., 0]
        available = np.flatnonzero(valid.any(axis=0))
        if not len(available):
            continue
        if pattern.startswith("point_"):
            target[b, ..., 0] = valid & (rng.random(valid.shape) < int(pattern.split("_")[1]) / 100)
        elif pattern == "multirate":
            reference = int(np.argmax(valid.sum(axis=0)))
            for s in range(5):
                if s == reference:
                    continue
                observed_rows = np.flatnonzero(valid[:, s])
                if not len(observed_rows):
                    continue
                stride = int(rng.integers(2, 16))
                # Sample each sensor's observations, avoiding a global-row phase alias.
                phase = int(rng.integers(min(stride, len(observed_rows))))
                kept = observed_rows[phase::stride]
                target[b, observed_rows, s, 0] = True
                target[b, kept, s, 0] = False
        elif pattern == "single_sensor_outage":
            s = int(rng.choice(available))
            target[b, :, s, 0] = valid[:, s]
        elif pattern in ("temporal_block", "temporary_sensor_outage"):
            elapsed = temporal[b, :, 0].numpy()
            s = int(rng.choice(available))
            center = int(rng.choice(np.flatnonzero(valid[:, s])))
            duration = max(float(elapsed[-1] - elapsed[0]), 1e-12) * rng.uniform(0.1, 0.5)
            hidden = np.abs(elapsed - elapsed[center]) <= duration / 2
            if pattern == "temporal_block":
                target[b, ..., 0] = valid & hidden[:, None]
            else:
                target[b, :, s, 0] = valid[:, s] & hidden
        else:
            raise ValueError(f"Unknown missingness pattern: {pattern}")
        if not target[b].any() and pattern != "multirate":
            position = np.argwhere(observed_np[b])[int(rng.integers(observed_np[b].sum()))]
            target[b][tuple(position)] = True
    return torch.from_numpy(target)


def context_classes(visible):
    return (np.logical_or.accumulate(visible, axis=1).astype("int8") +
            np.logical_or.accumulate(visible[:, ::-1], axis=1)[:, ::-1].astype("int8"))


def time_linear(x, visible, temporal):
    prediction = np.zeros_like(x)
    for b in range(len(x)):
        elapsed = temporal[b, :, 0]
        for s in range(5):
            valid = visible[b, :, s, 0]
            if valid.any():
                prediction[b, :, s, 0] = np.interp(elapsed, elapsed[valid], x[b, valid, s, 0])
    return prediction


def new_store():
    return {"overall": _empty_sums(), "normalized_sensor": [_empty_sums() for _ in range(5)],
            "physical_sensor": [_empty_sums() for _ in range(5)]}


def finish(store):
    if not store["overall"]["count"]:
        return {"target_count": 0, "normalized_sensor_macro": {"MAE": None, "RMSE": None}}
    return _method_metrics(store["overall"], store["normalized_sensor"], store["physical_sensor"])


def evaluation_cases(intervals, lengths, compact=False):
    if compact:
        return [(interval, lengths[(i + p) % len(lengths)], pattern)
                for i, interval in enumerate(intervals) for p, pattern in enumerate(PATTERNS)]
    return [(interval, length, pattern) for interval in intervals for length in lengths for pattern in PATTERNS]


def evaluate(model, scales, device, mean, std, seed, cases, repeats=1, batch_size=2, collect_calibration=False, irregular=False):
    results = []
    calibration = [[[] for _ in CONTEXTS] for _ in SENSOR_COLUMNS]
    model.eval()
    with torch.inference_mode():
        for case_index, (interval, requested_length, pattern) in enumerate(cases):
            rng = np.random.default_rng(seed + case_index * 1009)
            methods = ("imputeformer", "linear_interpolation", "forward_backward_hold", "train_sensor_mean")
            stores = {method: new_store() for method in methods}
            strata = {context: {method: new_store() for method in methods} for context in CONTEXTS}
            unique = [set() for _ in range(5)]
            actual_lengths = set()
            for _ in range(repeats):
                effective_batch = max(1, min(batch_size, 4096 // min(requested_length, len(scales[interval]["values"]))))
                x, temporal, observed, indices = sample_batch(scales, interval, requested_length, effective_batch, rng, irregular)
                actual_lengths.add(x.shape[1])
                artificial = artificial_mask(observed, temporal, rng, pattern)
                visible = observed & ~artificial
                if not artificial.any():
                    continue
                prediction = model(x.to(device), temporal.to(device), visible.to(device).float()).cpu().numpy()
                x_np, target, visible_np = x.numpy(), artificial.numpy(), visible.numpy()
                baseline = time_linear(x_np, visible_np, temporal.numpy())
                classes = context_classes(visible_np)
                estimates = {"imputeformer": prediction, "linear_interpolation": baseline,
                             "forward_backward_hold": _baseline_prediction(x_np, visible_np, "forward_backward_hold"),
                             "train_sensor_mean": np.zeros_like(x_np)}
                for method, estimate in estimates.items():
                    _update_method_sums(stores[method], estimate, x_np, target, mean, std)
                    for c, context in enumerate(CONTEXTS):
                        _update_method_sums(strata[context][method], estimate, x_np, target & (classes == c), mean, std)
                for b, rows in enumerate(indices):
                    for s in range(5):
                        unique[s].update(rows[target[b, :, s, 0]].tolist())
                if collect_calibration:
                    for s in range(5):
                        for c in range(3):
                            selected = target[:, :, s, 0] & (classes[:, :, s, 0] == c)
                            calibration[s][c].append(np.stack([
                                baseline[:, :, s, 0][selected] - x_np[:, :, s, 0][selected],
                                prediction[:, :, s, 0][selected] - baseline[:, :, s, 0][selected],
                            ], axis=1))
            results.append({"sampling_interval_seconds": interval if interval else None,
                            "sampling_grid": "native_actual_timestamps" if interval == 0 else "measured_bin_mean",
                            "irregular_row_subsampling": irregular,
                            "requested_length": requested_length, "actual_lengths": sorted(actual_lengths), "pattern": pattern,
                            "unique_target_counts": dict(zip(SENSOR_COLUMNS, map(len, unique))),
                            "methods": {m: finish(store) for m, store in stores.items()},
                            "context_strata": {c: {m: finish(store) for m, store in methods_.items()} for c, methods_ in strata.items()}})
    return results, calibration


def validation_score(results):
    scores = [r["methods"]["imputeformer"]["normalized_sensor_macro"]["MAE"] for r in results]
    finite = [x for x in scores if x is not None and np.isfinite(x)]
    if not finite:
        raise RuntimeError("No finite validation targets.")
    return float(np.mean(finite))


def calibrate(model, samples):
    report, gains = {}, np.zeros((5, 3), dtype="float32")
    for s, sensor in enumerate(SENSOR_COLUMNS):
        report[sensor] = {}
        for c, context in enumerate(CONTEXTS):
            values = np.concatenate(samples[s][c]) if samples[s][c] else np.empty((0, 2))
            candidates = np.linspace(0, 1, 21)
            errors = [float(np.abs(values[:, 0] + gain * values[:, 1]).mean()) for gain in candidates] if len(values) else []
            best = int(np.argmin(errors)) if errors else 0
            # Require validation evidence before modifying a reliable time interpolation.
            if len(values) < 30 or (errors and errors[best] >= errors[0] * 0.995):
                best = 0
            gains[s, c] = candidates[best]
            report[sensor][context] = {"gain": float(gains[s, c]), "validation_target_count": len(values),
                                      "linear_MAE": errors[0] if errors else None,
                                      "calibrated_MAE": errors[best] if errors else None}
    model.context_residual_gain.copy_(torch.from_numpy(gains).to(model.context_residual_gain.device))
    return report


def balanced_loss(prediction, target, hidden, visible, baseline=None):
    left = visible.long().cummax(1).values
    right = visible.long().flip(1).cummax(1).values.flip(1)
    classes = left + right
    error = prediction - target
    terms = []
    for s in range(5):
        for c in range(3):
            mask = hidden[:, :, s, :] & (classes[:, :, s, :] == c)
            if mask.any():
                selected = error[:, :, s, :][mask]
                denominator = ((baseline - target)[:, :, s, :][mask].abs().mean().detach().clamp(min=0.05)
                               if baseline is not None else 1.0)
                context_weight = 0.25 if c == 0 else 1.0
                terms.append(context_weight * (selected.abs().mean() + 0.05 * selected.square().mean()) / denominator)
    return torch.stack(terms).mean() if terms else prediction.sum() * 0


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    splits, manifest, dataset_run = load_ocean_sensor_splits(
        Path(args.dataset_root), Path(args.dataset_run) if args.dataset_run else None, require_regular=False)
    mean, std = fit_normalizer(splits["train"].to_numpy(dtype="float32"))
    intervals = tuple(sorted(set(float(s) for s in args.sampling_intervals_seconds)))
    lengths = tuple(sorted(set(args.training_lengths)))
    LOGGER.info("Preparing real-time scales seconds=%s lengths=%s dataset=%s", intervals, lengths, dataset_run)
    scales = {name: prepare_scales(frame, mean, std, intervals) for name, frame in splits.items()}
    interval_keys = (0.0, *intervals)
    config = {"window": args.window, "inference_stride": args.inference_stride, "num_nodes": 5,
              "input_dim": 13, "output_dim": 1, "input_embedding_dim": args.input_embedding_dim,
              "learnable_embedding_dim": args.learnable_embedding_dim, "feed_forward_dim": args.feed_forward_dim,
              "num_temporal_heads": args.num_temporal_heads, "num_layers": args.num_layers,
              "dim_proj": args.dim_proj, "dropout": args.dropout, "variable_length": True,
              "use_interpolation_residual": True, "time_aware": True, "static_sensor_embedding": True}
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
    history, best_score, best_epoch, stale = [], math.inf, 0, 0
    best_state = None
    rng = np.random.default_rng(args.seed)
    audit = {"status": "training", "dataset_run": str(dataset_run), "dataset_manifest_sha256": sha256_file(dataset_run / "dataset_manifest.json"),
             "split_rows": {k: len(v) for k, v in splits.items()}, "seed": args.seed, "random_initialization": True,
             "training_lengths": list(lengths), "training_sampling_intervals_seconds": list(intervals),
             "native_irregular_timestamps_in_training": True, "irregular_row_subsampling_probability": 0.2,
             "normalization_fit_split": "train", "selection_split": "validation", "calibration_split": "validation",
             "loss": "sensor/context balanced error relative to time-linear error (floor 0.05); no-context weight 0.25",
             "selection_score": "validation-calibrated MAE divided by max(linear MAE,0.05), averaged over sensor/context groups, plus 0.01 raw macro MAE as tie-breaker",
             "epoch_steps": args.steps_per_epoch, "maximum_batch_size": args.batch_size, "batch_row_budget": args.batch_row_budget,
             "maximum_epochs": args.max_epochs, "source_sensor_audit": manifest.get("sensor_audit"),
             "all_rows_available_for_sampling": True, "every_row_seen_each_epoch": False,
             "missing_bins_used_as_ground_truth": False, "mask_patterns": list(PATTERNS), "history": history,
             "evaluation_policy": "Identical seeded hidden measured cells for all methods. Overlapping windows may repeat targets; unique counts and same-sensor context strata are reported. Test dates have been examined in earlier model audits and are not a new blind field test.",
             "generalization_limit": "Actual high-rate current and depth; other sensors retain their measured lower rates. Coarse scales use measured-bin means, never interpolated labels. Unseen rates/lengths are accepted but have no universal accuracy guarantee. Requested windows longer than a split are shortened and actual lengths reported."}
    compact_cases = evaluation_cases(interval_keys, lengths, compact=True)
    full_cases = evaluation_cases(interval_keys, lengths)
    coverage = {}
    LOGGER.info("Fresh time-aware ImputeFormer device=%s parameters=%d metrics=%s", device, sum(p.numel() for p in model.parameters()), metrics_path)
    for epoch in range(1, args.max_epochs + 1):
        started = time.monotonic()
        model.train()
        model.context_residual_gain.fill_(1.0)
        losses = []
        for step in range(args.steps_per_epoch):
            interval = float(rng.choice(interval_keys))
            length = int(rng.choice(lengths))
            if rng.random() < 0.15:
                length = int(rng.integers(32, args.window + 1))
            effective_length = min(length, len(scales["train"][interval]["values"]))
            batch_size = max(1, min(args.batch_size, args.batch_row_budget // effective_length))
            irregular = bool(rng.random() < 0.2)
            x, temporal, observed, _ = sample_batch(scales["train"], interval, length, batch_size, rng, irregular)
            pattern = str(rng.choice(PATTERNS))
            hidden = artificial_mask(observed, temporal, rng, pattern)
            visible = observed & ~hidden
            if not hidden.any():
                continue
            coverage_key = f"seconds={interval}:length={x.shape[1]}:pattern={pattern}:irregular={irregular}"
            coverage[coverage_key] = coverage.get(coverage_key, 0) + batch_size
            x, temporal, hidden, visible = x.to(device), temporal.to(device), hidden.to(device), visible.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, temporal, visible.float())
            baseline = model.time_aware_components(x, temporal, visible.float())[0]
            loss = balanced_loss(prediction, x, hidden, visible, baseline)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss epoch={epoch} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 24 == 0:
                LOGGER.info("epoch=%d step=%d/%d loss=%.6f", epoch, step + 1, args.steps_per_epoch, np.mean(losses))
        validation, selection_samples = evaluate(model, scales["validation"], device, mean, std, args.seed + 100000,
                                 compact_cases, repeats=1, batch_size=2, collect_calibration=True)
        selection_calibration = calibrate(model, selection_samples)
        score = float(np.mean([group["calibrated_MAE"] / max(group["linear_MAE"], 0.05)
                              for sensor in selection_calibration.values() for group in sensor.values()
                              if group["linear_MAE"] is not None])) + 0.01 * validation_score(validation)
        del selection_samples
        # Calibration is for selection only; zero gains must not disable training gradients.
        model.context_residual_gain.fill_(1.0)
        scheduler.step(score)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_selection_score": score,
                        "validation_raw_macro_MAE": validation_score(validation),
                        "seconds": time.monotonic() - started, "learning_rate": optimizer.param_groups[0]["lr"]})
        LOGGER.info("epoch=%d validation_selection_score=%.6f seconds=%.1f", epoch, score, history[-1]["seconds"])
        if score < best_score - args.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_dir / "best_selection_state.pt")
        else:
            stale += 1
        audit.update({"best_epoch": best_epoch, "best_validation_selection_score": best_score, "training_window_coverage": coverage})
        metrics_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite selected model.")
    model.load_state_dict(best_state)
    LOGGER.info("Calibrating residual gains on validation only; selected epoch=%d", best_epoch)
    validation_before, calibration_samples = evaluate(model, scales["validation"], device, mean, std,
        args.seed + 300000, full_cases, repeats=args.evaluation_repeats, collect_calibration=True)
    calibration_report = calibrate(model, calibration_samples)
    del calibration_samples
    LOGGER.info("Calibration gains=%s", model.context_residual_gain.cpu().tolist())
    test, _ = evaluate(model, scales["test"], device, mean, std, args.seed + 200000,
                       full_cases, repeats=args.evaluation_repeats)
    audit.update({"status": "complete", "independent_test": test, "validation_before_calibration": validation_before,
                  "residual_calibration": calibration_report,
                  "metric_definitions": {"standardized_MAE": "MAE divided by each sensor's train-only standard deviation; macro averages available sensors equally.",
                                         "MAE": "mean absolute error", "RMSE": "root mean squared error", "MRE": "sum(abs(error))/sum(abs(target)); null if denominator is zero"}})
    checkpoint = {"format_version": 5, "model_type": "ImputeFormerModel", "entity_type": "sensor", "uses_site_entity": False,
                  "uses_coordinates": False, "model_state_dict": model.state_dict(), "model_config": config,
                  "sensor_ids": list(SENSOR_COLUMNS), "sensor_units": SENSOR_UNITS,
                  "temporal_feature_names": list(RAW_TEMPORAL_FEATURE_NAMES), "temporal_encoding": "actual_elapsed_delta_and_visible_gaps_v2",
                  "supported_sampling_interval_seconds": {"minimum": 0.0, "maximum": None},
                  "normalization": {"mean": mean, "std": std}, "output_calibration": None,
                  "residual_calibration": calibration_report,
                  "training": {k: v for k, v in audit.items() if k not in ("history", "independent_test", "validation_before_calibration", "training_window_coverage")},
                  "data": {"dataset_run": str(dataset_run), "dataset_manifest": str(dataset_run / "dataset_manifest.json")},
                  "evaluation_metrics": str(metrics_path)}
    checkpoint_path = model_dir / "imputeformer_ocean_five_sensor.pt"
    torch.save(checkpoint, checkpoint_path)
    (model_dir / "best_selection_state.pt").unlink()
    audit["checkpoint_sha256"] = sha256_file(checkpoint_path)
    metrics_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    active = Path(args.model_root).resolve() / "imputeformer_ocean_five_sensor.pt"
    temporary = active.with_suffix(".pt.tmp")
    shutil.copyfile(checkpoint_path, temporary)
    temporary.replace(active)
    LOGGER.info("Published checkpoint=%s sha256=%s metrics=%s", active, audit["checkpoint_sha256"], metrics_path)
    return checkpoint_path
