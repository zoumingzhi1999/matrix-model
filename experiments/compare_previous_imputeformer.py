"""Compare the previous checkpoint on the new model's identical measured targets.

The previous model keeps its trained normalization and row-based interpolation.
Only input/output normalization and its existing calendar/cadence features are
adapted so both models can be evaluated on the same physical sensor values.
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
from train_time_aware_imputeformer import evaluate, evaluation_cases, prepare_scales


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIOUS = (
    ROOT / "models/sensor_imputeformer/runs/20260906_143522_756272/"
    "imputeformer_ocean_five_sensor.pt"
)
LOGGER = logging.getLogger(__name__)


class PreviousModelAdapter(torch.nn.Module):
    def __init__(self, checkpoint, new_mean, new_std):
        super().__init__()
        if checkpoint.get("sensor_ids") != list(SENSOR_COLUMNS):
            raise ValueError("Previous checkpoint sensor order differs from the evaluation schema.")
        if checkpoint.get("output_calibration") is not None:
            raise ValueError("This comparator expects the previous checkpoint without output calibration.")
        if checkpoint.get("temporal_encoding") != "bounded_log_interval_v1":
            raise ValueError("Previous temporal encoding is not the supported bounded_log_interval_v1.")
        if checkpoint["model_config"].get("time_aware"):
            raise ValueError("Expected the previous model without physical-time context.")
        self.previous = build_model(5, checkpoint["model_config"])
        self.previous.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for name, value in (
            ("new_mean", new_mean), ("new_std", new_std),
            ("previous_mean", checkpoint["normalization"]["mean"]),
            ("previous_std", checkpoint["normalization"]["std"]),
        ):
            tensor = torch.as_tensor(value, dtype=torch.float32).reshape(1, 1, 5, 1)
            if not torch.isfinite(tensor).all() or ("std" in name and (tensor <= 0).any()):
                raise ValueError(f"Invalid normalization tensor {name}.")
            self.register_buffer(name, tensor)

    @staticmethod
    def previous_temporal_features(temporal, dtype):
        if temporal.ndim != 3 or temporal.shape[-1] != 6:
            raise ValueError("Expected raw elapsed/delta/calendar features [batch,time,6].")
        intervals = []
        for sample in temporal:
            positive = sample[1:, 1][sample[1:, 1] > 0]
            interval = torch.quantile(positive, 0.5) if len(positive) else sample.new_tensor(1.0)
            logged = torch.log1p(interval)
            intervals.append(logged / (1 + logged))
        cadence = torch.stack(intervals)[:, None, None].expand(-1, temporal.shape[1], 1)
        return torch.cat((temporal[:, :, 2:6], cadence), dim=-1).to(dtype=dtype)

    def forward(self, x, temporal, visible):
        physical_values = x * self.new_std + self.new_mean
        previous_values = (physical_values - self.previous_mean) / self.previous_std
        previous_temporal = self.previous_temporal_features(temporal, previous_values.dtype)
        estimate = self.previous(previous_values, previous_temporal, visible)
        return (estimate * self.previous_std + self.previous_mean - self.new_mean) / self.new_std


def metric_mean(cases, method, name, context=None):
    numbers = []
    for case in cases:
        metrics = case["methods"] if context is None else case["context_strata"][context]
        value = metrics[method]["normalized_sensor_macro"].get(name)
        if value is not None and np.isfinite(value):
            numbers.append(float(value))
    return float(np.mean(numbers)) if numbers else None


def comparison_summary(previous, current, context=None):
    result = {}
    for name in ("MAE", "RMSE"):
        old = metric_mean(previous, "imputeformer", name, context)
        new = metric_mean(current, "imputeformer", name, context)
        baseline = metric_mean(current, "linear_interpolation", name, context)
        result[name] = {
            "previous_model": old, "new_model": new, "time_linear_interpolation": baseline,
            "new_error_reduction_percent": (100 * (old - new) / old)
            if old is not None and old > 0 and new is not None else None,
        }
    return result


def verify_identical_targets(previous, current):
    if len(previous) != len(current):
        raise ValueError("Previous and new evaluation case counts differ.")
    for index, (old, new) in enumerate(zip(previous, current, strict=True)):
        for key in (
            "sampling_interval_seconds", "sampling_grid", "requested_length",
            "actual_lengths", "pattern", "unique_target_counts",
        ):
            if old[key] != new[key]:
                raise ValueError(f"Evaluation case {index} differs in {key}; masks or ordering changed.")
        for method in ("linear_interpolation", "forward_backward_hold", "train_sensor_mean"):
            # Identical baseline errors additionally verify the actual target values,
            # normalization and masks, beyond matching only target counts.
            if old["methods"][method] != new["methods"][method]:
                raise ValueError(f"Evaluation case {index} has different {method} targets/normalization.")


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.cpu_threads)
    previous_path, checkpoint_path = args.previous_checkpoint.resolve(), args.new_checkpoint.resolve()
    previous_checkpoint = torch.load(previous_path, map_location="cpu", weights_only=False)
    current_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if current_checkpoint.get("sensor_ids") != list(SENSOR_COLUMNS):
        raise ValueError("New checkpoint sensor order differs from the evaluation schema.")
    metrics_path = Path(current_checkpoint["evaluation_metrics"])
    new_report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if new_report.get("status") != "complete":
        raise ValueError("The new model's evaluation must complete before comparison.")
    if new_report.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("New checkpoint hash does not match its published evaluation report.")
    dataset_run = Path(new_report["dataset_run"])
    splits, manifest, _ = load_ocean_sensor_splits(dataset_run.parent.parent, dataset_run, require_regular=False)
    if sha256_file(dataset_run / "dataset_manifest.json") != new_report["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed since the new model's evaluation.")
    mean = np.asarray(current_checkpoint["normalization"]["mean"], dtype="float32")
    std = np.asarray(current_checkpoint["normalization"]["std"], dtype="float32")
    intervals = tuple(new_report["training_sampling_intervals_seconds"])
    lengths = tuple(new_report["training_lengths"])
    cases = evaluation_cases((0.0, *intervals), lengths)
    expected_cases = [
        (case["sampling_interval_seconds"] or 0.0, case["requested_length"], case["pattern"])
        for case in new_report["independent_test"]
    ]
    if cases != expected_cases:
        raise ValueError("Saved new-model test cases differ from the current evaluation function.")
    scales = prepare_scales(splits["test"], mean, std, intervals)
    del splits
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    adapter = PreviousModelAdapter(previous_checkpoint, mean, std).to(device)
    seed = int(new_report["seed"]) + 200000
    LOGGER.info("Previous model comparison: device=%s cases=%d repeats=%d seed=%d",
                device, len(cases), args.repeats, seed)
    previous_results, _ = evaluate(
        adapter, scales, device, mean, std, seed, cases,
        repeats=args.repeats, batch_size=args.batch_size,
    )
    current_results = new_report["independent_test"]
    verify_identical_targets(previous_results, current_results)
    comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "dataset_run": str(dataset_run), "dataset_manifest_sha256": sha256_file(dataset_run / "dataset_manifest.json"),
        "previous_checkpoint": str(previous_path), "previous_checkpoint_sha256": sha256_file(previous_path),
        "new_checkpoint": str(checkpoint_path), "new_checkpoint_sha256": sha256_file(checkpoint_path),
        "new_evaluation_metrics": str(metrics_path), "seed": seed,
        "repeats": args.repeats, "maximum_batch_size": args.batch_size,
        "case_count": len(cases), "identical_target_and_baseline_metrics_verified": True,
        "metric_definition": "Each case first averages train-standardized sensor errors equally among sensors with targets; summaries average the resulting case metrics equally.",
        "adapter_definition": "New normalized inputs are converted to physical units then previous normalization. Four matching calendar phases are preserved and the previous cadence feature receives bounded log of each sample's median positive real delta. Previous predictions are converted back to the new train-only normalization.",
        "limitations": [
            "The previous model retains interpolation and embeddings based on row position; a median cadence feature cannot express individual irregular time gaps.",
            "Both checkpoints are tested on the same newly measured Vector near-bed instantaneous current. The previous model was trained on hourly depth-averaged Signature current, so this benchmark includes that distribution shift.",
            "The new test is the same previously inspected deployment period, not a new blind deployment holdout.",
            "These benchmark numbers are not directly comparable to the previous report's different current targets and sampled windows.",
        ],
        "current_measurement_definition": manifest.get("current_source", {}).get("measurement_definition"),
        "summary": comparison_summary(previous_results, current_results),
        "by_context": {
            context: comparison_summary(previous_results, current_results, context)
            for context in ("no_same_sensor_context", "one_sided_context", "two_sided_context")
        },
        "by_sampling_interval": {
            str(interval): comparison_summary(
                [c for c in previous_results if (c["sampling_interval_seconds"] or 0.0) == interval],
                [c for c in current_results if (c["sampling_interval_seconds"] or 0.0) == interval],
            ) for interval in (0.0, *intervals)
        },
        "previous_model_test": previous_results,
    }
    output = args.output.resolve() if args.output else metrics_path.with_name("previous_model_comparison.json")
    if output.exists():
        raise FileExistsError(f"Comparison output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("Verified identical targets; comparison=%s summary=%s", output, comparison["summary"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--previous-checkpoint", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if min(args.cpu_threads, args.repeats, args.batch_size) < 1:
        parser.error("cpu-threads, repeats and batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
