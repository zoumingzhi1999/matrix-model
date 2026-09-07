"""Streaming / backend-pushed data entry point.

It reads sensor records from a long- or wide-format CSV file, a JSONL file,
stdin, or a single ``--records-json`` argument.

Each record must be:
    {"timestamp": "2026-06-08T12:00:00Z",
     "sensor_id": "water_temperature_c",
     "value":     20.1}

After ingestion, each sensor cadence is inferred from its timestamps. All five
streams are aligned to the sensor with the most valid samples and pivoted into the
checkpoint's sensor columns. The flow never creates a site or coordinate entity.

Examples
--------
Read from a JSONL file::

    python experiments/full_pipeline/run_streaming_pipeline.py \\
        --records-jsonl data/incoming/sensors.jsonl \\
        --output-dir outputs/streaming_job_001

Read from a CSV file::

    python experiments/full_pipeline/run_streaming_pipeline.py \\
        --records-csv data/incoming/sensors.csv \\
        --output-dir outputs/streaming_job_001

Read from stdin (backend pipes records in)::

    cat data/incoming/sensors.jsonl | python \\
        experiments/full_pipeline/run_streaming_pipeline.py --stdin

Pass a single batch directly::

    python experiments/full_pipeline/run_streaming_pipeline.py \\
        --records-json '[{"timestamp":"2026-06-08T12:00:00Z",...}]'
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from config.input_paths import resolve_input_paths

from risk_pipeline.artifacts import (
    artifact_path,
    artifact_timestamp,
    write_csv_once,
)
from risk_pipeline.run_encapsulated_pipeline import (
    DEFAULT_RISK_CONFIG,
    run as run_representation_pipeline,
)
from risk_pipeline.data_io import (
    normalize_streaming_records,
    resolve_image_source_paths,
    resolve_sonar_source_paths,
)
from risk_pipeline.model_bundle import (
    DEFAULT_FUSION_CHECKPOINT,
    ProductionModelBundle,
    resolve_inference_device,
)
from risk_pipeline.sensor_imputation import (
    DEFAULT_SENSOR_CHECKPOINT,
    align_with_pretrained_imputeformer,
    load_sensor_imputeformer_checkpoint,
)
from risk_pipeline.sensor_schema import OCEAN_SENSOR_IDS


logger = logging.getLogger("f-model.streaming")


# ---------------------------------------------------------------------------
# Record ingestion
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
    return records


def read_csv_records(path: Path) -> list[dict]:
    """Read long- or sparse-wide-format sensor records from a CSV file.

    Long format uses ``timestamp``, ``sensor_id``, and ``value``. Wide format
    uses ``timestamp`` plus the five canonical sensor columns. Additional
    columns are ignored and empty values remain missing for ImputeFormer.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    long_columns = ("timestamp", "sensor_id", "value")
    is_long = all(column in frame.columns for column in long_columns)
    is_wide = (
        "timestamp" in frame.columns
        and all(sensor_id in frame.columns for sensor_id in OCEAN_SENSOR_IDS)
    )
    if not is_long and not is_wide:
        raise ValueError(
            f"Sensor CSV {path} must use either long columns "
            f"{list(long_columns)} or sparse wide columns "
            f"{['timestamp', *OCEAN_SENSOR_IDS]}."
        )
    if frame.empty:
        return []

    missing_timestamps = frame["timestamp"].isna() | (
        frame["timestamp"].astype(str).str.strip() == ""
    )
    if missing_timestamps.any():
        rows = (frame.index[missing_timestamps] + 2).tolist()
        raise ValueError(f"Sensor CSV {path} has empty timestamp at rows: {rows}.")

    if is_long:
        missing_sensor_ids = frame["sensor_id"].isna() | (
            frame["sensor_id"].astype(str).str.strip() == ""
        )
        if missing_sensor_ids.any():
            rows = (frame.index[missing_sensor_ids] + 2).tolist()
            raise ValueError(
                f"Sensor CSV {path} has empty sensor_id at rows: {rows}."
            )
        return frame.loc[:, long_columns].to_dict(orient="records")

    records = frame.loc[:, ["timestamp", *OCEAN_SENSOR_IDS]].melt(
        id_vars="timestamp",
        value_vars=list(OCEAN_SENSOR_IDS),
        var_name="sensor_id",
        value_name="value",
    )
    return records.to_dict(orient="records")


def read_stdin_records() -> list[dict]:
    return read_jsonl_lines(sys.stdin)


def read_time_series_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        return read_csv_records(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_jsonl(path)
    raise ValueError(f"Time-series input must be a CSV or JSONL file: {path}")


def read_jsonl_lines(stream) -> list[dict]:
    records: list[dict] = []
    for line_no, raw in enumerate(stream, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on stdin line {line_no}: {exc}"
            ) from exc
    return records


# ---------------------------------------------------------------------------
# Pipeline body
# ---------------------------------------------------------------------------

def run_imputeformer_alignment_from_records(
    records: list[dict],
    args,
    output_dir: Path,
    model_bundle: ProductionModelBundle | None = None,
):
    from risk_pipeline.data_io import load_streaming_records

    logger.info("Alignment started: input_records=%d frequency=%s", len(records), args.freq)
    if model_bundle is None:
        checkpoint_path = args.sensor_checkpoint
        sensor_model, sensor_checkpoint = load_sensor_imputeformer_checkpoint(
            checkpoint_path, resolve_inference_device(getattr(args, "device", "auto")),
        )
    else:
        checkpoint_path = model_bundle.sensor_checkpoint_path
        sensor_model = model_bundle.sensor_model
        sensor_checkpoint = model_bundle.sensor_checkpoint
    time_aware = bool(sensor_checkpoint.get("model_config", {}).get("time_aware", False))
    logger.info(
        "Checkpoint time-axis contract: time_aware=%s auto_grid=%s checkpoint=%s",
        time_aware, "actual_reference_timestamps" if time_aware else "inferred_regular_cadence",
        checkpoint_path,
    )
    wide_df, _, mask, _, grid_metadata = load_streaming_records(
        records, args.freq, preserve_actual_timestamps=time_aware,
    )
    logger.info("Sensor alignment grid: %s", json.dumps(grid_metadata, ensure_ascii=False))

    timestamp = getattr(args, "artifact_timestamp", None) or artifact_timestamp()
    args.artifact_timestamp = timestamp

    aligned_df, inference_metadata = align_with_pretrained_imputeformer(
        wide_df,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size,
        device=getattr(args, "device", "auto"),
        evaluation_holdout_ratio=getattr(args, "evaluation_holdout_ratio", 0.10),
        evaluation_seed=getattr(args, "evaluation_seed", 42),
        preloaded_model=sensor_model,
        preloaded_checkpoint=sensor_checkpoint,
    )
    sensors = inference_metadata["sensor_ids"]
    aligned_path = artifact_path(
        output_dir, "aligned", "stage1_imputeformer_aligned_output", ".csv", timestamp)
    write_csv_once(
        aligned_df,
        aligned_path,
        index=True,
        index_label="timestamp",
        float_format=lambda value: repr(float(value)),
    )
    evaluation_rows = inference_metadata.pop("evaluation_metrics", [])
    metrics_path = artifact_path(
        output_dir,
        "aligned",
        "stage1_imputeformer_evaluation_metrics",
        ".csv",
        timestamp,
    )
    write_csv_once(pd.DataFrame(evaluation_rows), metrics_path)
    inference_metadata.update({
        "aligned_output": str(aligned_path),
        "evaluation_metrics_csv": str(metrics_path),
        "evaluation_metrics_overall": evaluation_rows[0] if evaluation_rows else None,
        "grid_alignment": grid_metadata,
    })
    logger.info(
        "Alignment completed: rows=%d sensors=%s aligned_output=%s metrics=%s metadata=%s",
        len(aligned_df), sensors, aligned_path, metrics_path,
        json.dumps(inference_metadata, ensure_ascii=False, default=str),
    )
    return aligned_path, sensors, mask, inference_metadata


def _project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return (resolved if resolved.is_absolute() else ROOT / resolved).resolve()


class RiskPipeline:
    """Reusable three-source inference, from raw records to the saved risk JSON.

    Models load on the first valid run and are shared across subsequent runs.
    Every run owns a new output directory containing the nine retained results.
    """

    def __init__(
        self, *,
        sensor_checkpoint: str | Path = DEFAULT_SENSOR_CHECKPOINT,
        fusion_checkpoint: str | Path = DEFAULT_FUSION_CHECKPOINT,
        risk_config: str | Path = DEFAULT_RISK_CONFIG,
        device: str = "auto",
        freq: str = "auto",
        batch_size: int = 8,
        evaluation_holdout_ratio: float = 0.10,
        evaluation_seed: int = 42,
        output_root: str | Path = ROOT / "outputs" / "streaming_jobs",
        model_bundle: ProductionModelBundle | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not 0.0 < evaluation_holdout_ratio < 1.0:
            raise ValueError("evaluation_holdout_ratio must be in (0, 1).")
        self.sensor_checkpoint = _project_path(sensor_checkpoint)
        self.fusion_checkpoint = _project_path(fusion_checkpoint)
        self.risk_config = _project_path(risk_config)
        self.device = device
        self.freq = freq
        self.batch_size = batch_size
        self.evaluation_holdout_ratio = evaluation_holdout_ratio
        self.evaluation_seed = evaluation_seed
        self.output_root = _project_path(output_root)
        self._model_bundle = model_bundle
        self._model_load_lock = threading.Lock()

    def load_models(self) -> ProductionModelBundle:
        """Load and validate both checkpoints once, or return the shared bundle."""
        from risk_pipeline.artifacts import configure_inference_logging

        configure_inference_logging()
        with self._model_load_lock:
            if self._model_bundle is None:
                self._model_bundle = ProductionModelBundle.load(
                    sensor_checkpoint_path=self.sensor_checkpoint,
                    fusion_checkpoint_path=self.fusion_checkpoint,
                    device=self.device,
                )
            return self._model_bundle

    def run(
        self, *,
        time_series_input: str | Path | None = None,
        records: Iterable[dict] | None = None,
        optical_image: str | Path | None = None,
        sonar_image: str | Path | None = None,
        sonar_xtf: str | Path | None = None,
        output_dir: str | Path | None = None,
        source_name: str | None = None,
    ) -> dict:
        """Run all stages using explicit inputs or config/input_paths.py defaults.

        Supply either a CSV/JSONL path or in-memory long-format records.
        The result includes risk_matrix, risk_matrix_path, and all artifact paths;
        result metadata is returned in memory without an extra summary file.
        """
        from risk_pipeline.artifacts import configure_inference_logging

        configure_inference_logging()
        if records is not None and time_series_input is not None:
            raise ValueError("Provide either records or time_series_input, not both.")
        paths = resolve_input_paths(
            time_series=time_series_input, optical_image=optical_image,
            sonar_image=sonar_image, sonar_xtf=sonar_xtf,
        )
        if paths.optical_image is None or paths.sonar_source is None:
            raise ValueError(
                "Set OPTICAL_IMAGE_INPUT_PATH and SONAR_IMAGE_INPUT_PATH in "
                "config/input_paths.py, or provide optical_image and a sonar TIFF via sonar_image."
            )
        optical_sources = resolve_image_source_paths(paths.optical_image)
        sonar_sources = resolve_sonar_source_paths(
            paths.sonar_source, allow_xtf=paths.sonar_xtf is not None,
        )
        if records is None:
            if paths.time_series is None:
                raise ValueError(
                    "Provide records or time_series_input, or set "
                    "TIME_SERIES_INPUT_PATH in config/input_paths.py."
                )
            incoming = read_time_series_records(paths.time_series)
            input_source = str(paths.time_series)
        else:
            incoming = list(records)
            input_source = source_name or "in-memory-records"
        normalized = normalize_streaming_records(incoming)
        sensors_present = set(normalized["sensor_id"])
        if sensors_present != set(OCEAN_SENSOR_IDS):
            raise ValueError(
                f"Expected all five sensor identifiers {list(OCEAN_SENSOR_IDS)}; "
                f"received {sorted(sensors_present)}. Use null for missing values."
            )
        if not normalized["value"].notna().any():
            raise ValueError("At least one finite sensor observation is required.")
        timestamp = artifact_timestamp()
        job_dir = (
            _project_path(output_dir) if output_dir is not None
            else self.output_root / f"job_{timestamp}"
        )
        if job_dir.exists():
            raise FileExistsError(f"Output directory already exists; refusing to overwrite: {job_dir}")
        logger.info(
            "Inference input paths: time_series=%s optical_image=%s sonar_source=%s "
            "records=%d optical_files=%d sonar_files=%d output_dir=%s",
            input_source, paths.optical_image, paths.sonar_source, len(incoming),
            len(optical_sources), len(sonar_sources), job_dir,
        )
        models = self.load_models()
        args = SimpleNamespace(
            freq=self.freq, batch_size=self.batch_size, device=self.device,
            sensor_checkpoint=str(models.sensor_checkpoint_path),
            fusion_checkpoint=str(models.fusion_checkpoint_path),
            risk_config=str(self.risk_config),
            evaluation_holdout_ratio=self.evaluation_holdout_ratio,
            evaluation_seed=self.evaluation_seed,
            optical_image=str(paths.optical_image),
            sonar_image=str(paths.sonar_image) if paths.sonar_image is not None else None,
            sonar_xtf=str(paths.sonar_xtf) if paths.sonar_xtf is not None else None,
            output_dir=str(job_dir), artifact_timestamp=timestamp, print_result=False,
        )
        with models.inference_lock:
            job_dir.mkdir(parents=True, exist_ok=False)
            try:
                aligned_path, sensors, mask, alignment_metadata = (
                    run_imputeformer_alignment_from_records(
                        incoming, args, job_dir, model_bundle=models,
                    )
                )
                args.imputeformer_output = str(aligned_path)
                pipeline_result = run_representation_pipeline(args, model_bundle=models)
            except Exception:
                logger.exception("Inference failed: output_dir=%s", job_dir)
                raise

        risk_result = pipeline_result["risk_result"]
        risk_path = str(pipeline_result["risk_json_path"])
        metrics_path = alignment_metadata["evaluation_metrics_csv"]
        artifacts = dict(pipeline_result["artifacts"])
        artifacts["alignment_metrics"] = metrics_path
        result = {
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "data_source": input_source,
            "input_record_count": len(incoming),
            "sensors": sensors,
            "observed_ratio": float(mask.sum() / mask.size),
            "alignment_mode": alignment_metadata["mode"],
            "alignment_checkpoint": alignment_metadata["checkpoint"],
            "alignment_checkpoint_sha256": alignment_metadata["checkpoint_sha256"],
            "alignment_metrics_path": metrics_path,
            "alignment_metrics": alignment_metadata.get("evaluation_metrics_overall"),
            "alignment_metadata": alignment_metadata,
            "imputed_values": alignment_metadata["imputed_values"],
            "final_risk": float(risk_result["total_risk"]),
            "risk_level": str(risk_result["risk_level"]),
            "risk_matrix": risk_result,
            "risk_matrix_path": risk_path,
            "artifacts": artifacts,
            "outputs": {
                "stage1_imputeformer_aligned": str(aligned_path),
                "stage1_imputeformer_metrics": metrics_path,
                "risk_matrix": risk_path,
            },
        }
        logger.info(
            "Inference completed: job=%s rows=%s total_risk=%s risk_level=%s risk_json=%s",
            job_dir.name, alignment_metadata.get("grid_alignment", {}).get("grid_rows"),
            result["final_risk"], result["risk_level"], risk_path,
        )
        return result


def run(args) -> dict:
    records = None
    source_name = None
    if args.stdin:
        records, source_name = read_stdin_records(), "stdin"
    elif args.records_json is not None:
        records, source_name = json.loads(args.records_json), "inline-json"
        if not isinstance(records, list):
            raise ValueError("--records-json must contain a JSON array of sensor records.")
    pipeline = RiskPipeline(
        sensor_checkpoint=args.sensor_checkpoint,
        fusion_checkpoint=args.fusion_checkpoint,
        risk_config=args.risk_config, device=args.device, freq=args.freq,
        batch_size=args.batch_size,
        evaluation_holdout_ratio=args.evaluation_holdout_ratio,
        evaluation_seed=args.evaluation_seed,
    )
    result = pipeline.run(
        time_series_input=args.time_series_input or args.records_csv or args.records_jsonl,
        records=records, source_name=source_name,
        optical_image=args.optical_image, sonar_image=args.sonar_image,
        sonar_xtf=args.sonar_xtf, output_dir=args.output_dir,
    )
    print(f"risk_matrix={result['risk_matrix_path']}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streaming / backend-pushed risk pipeline.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--time-series-input",
        help="CSV/JSONL path; defaults to TIME_SERIES_INPUT_PATH in config/input_paths.py.",
    )
    src.add_argument(
        "--records-csv",
        help=(
            "Path to a long-format CSV or a sparse wide CSV with timestamp "
            "plus all five sensor columns."
        ),
    )
    src.add_argument("--records-jsonl", help="Path to a JSONL file.")
    src.add_argument("--records-json", help="Inline JSON array string.")
    src.add_argument("--stdin", action="store_true",
                    help="Read JSONL from stdin.")
    # Output.
    p.add_argument(
        "--output-dir",
        help="New per-run directory; default: outputs/streaming_jobs/job_<timestamp>.",
    )
    p.add_argument(
        "--freq",
        default="auto",
        help="Use 'auto' to align to the sensor with the most observations, or a fixed duration.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--evaluation-holdout-ratio",
        type=float,
        default=0.10,
        help="Fraction of known sensor values held out only for MAE/MSE/RMSE evaluation.",
    )
    p.add_argument("--evaluation-seed", type=int, default=42)
    p.add_argument(
        "--sensor-checkpoint",
        default=str(DEFAULT_SENSOR_CHECKPOINT),
        help="Trained five-sensor ImputeFormer checkpoint.",
    )
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    # Representation hyperparams.
    p.add_argument("--optical-image", help="Optical image or directory; overrides OPTICAL_IMAGE_INPUT_PATH.")
    sonar = p.add_mutually_exclusive_group()
    sonar.add_argument("--sonar-image", help="Sonar TIFF (.tif/.tiff) file or directory; overrides configured sonar input.")
    sonar.add_argument("--sonar-xtf", help="Legacy raw sonar XTF input; overrides configured sonar input.")
    p.add_argument(
        "--risk-config", default=str(ROOT / "config" / "fixed_single_feature_risk_config.json"))
    p.add_argument("--fusion-checkpoint", default=str(DEFAULT_FUSION_CHECKPOINT))
    args = p.parse_args()
    if not 0.0 < args.evaluation_holdout_ratio < 1.0:
        p.error("--evaluation-holdout-ratio must be in (0, 1)")
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
