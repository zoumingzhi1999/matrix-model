"""Streaming / backend-pushed data entry point.

It reads long-format sensor records from a CSV file, a JSONL file, stdin, or a
single ``--records-json`` argument.

Each record must be:
    {"timestamp": "2026-06-08T12:00:00Z",
     "sensor_id": "water_temperature_c",
     "value":     20.1}

After ingestion, each sensor cadence is inferred from its timestamps. All five
streams are aligned to the fastest sensor's grid and pivoted into the
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
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_pipeline.artifacts import (
    artifact_path,
    artifact_timestamp,
    mapping_to_long_frame,
    write_csv_once,
)
from risk_pipeline.run_encapsulated_pipeline import (
    run as run_representation_pipeline,
)
from risk_pipeline.model_bundle import (
    DEFAULT_FUSION_CHECKPOINT,
    ProductionModelBundle,
)
from risk_pipeline.sensor_imputation import (
    DEFAULT_SENSOR_CHECKPOINT,
    align_with_pretrained_imputeformer,
)


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
    """Read long-format sensor records from a CSV file.

    The required columns are ``timestamp``, ``sensor_id``, and ``value``.
    Additional columns are ignored and an empty value is retained as a missing
    observation for ImputeFormer.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = ("timestamp", "sensor_id", "value")
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Sensor CSV {path} is missing required columns: {missing_columns}. "
            f"Expected columns: {list(required_columns)}."
        )
    if frame.empty:
        return []

    missing_timestamps = frame["timestamp"].isna() | (
        frame["timestamp"].astype(str).str.strip() == ""
    )
    if missing_timestamps.any():
        rows = (frame.index[missing_timestamps] + 2).tolist()
        raise ValueError(f"Sensor CSV {path} has empty timestamp at rows: {rows}.")

    missing_sensor_ids = frame["sensor_id"].isna() | (
        frame["sensor_id"].astype(str).str.strip() == ""
    )
    if missing_sensor_ids.any():
        rows = (frame.index[missing_sensor_ids] + 2).tolist()
        raise ValueError(f"Sensor CSV {path} has empty sensor_id at rows: {rows}.")

    return frame.loc[:, required_columns].to_dict(orient="records")


def read_stdin_records() -> list[dict]:
    return read_jsonl_lines(sys.stdin)


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

    wide_df, _, mask, _, grid_metadata = load_streaming_records(records, args.freq)

    timestamp = getattr(args, "artifact_timestamp", None) or artifact_timestamp()
    args.artifact_timestamp = timestamp

    wide_path = artifact_path(
        output_dir, "inputs", "stage0_streaming_input_grid", ".csv", timestamp)
    write_csv_once(wide_df, wide_path, index=True, index_label="timestamp")

    aligned_df, inference_metadata = align_with_pretrained_imputeformer(
        wide_df,
        checkpoint_path=(
            model_bundle.sensor_checkpoint_path
            if model_bundle is not None
            else args.sensor_checkpoint
        ),
        batch_size=args.batch_size,
        device=getattr(args, "device", "auto"),
        evaluation_holdout_ratio=getattr(args, "evaluation_holdout_ratio", 0.10),
        evaluation_seed=getattr(args, "evaluation_seed", 42),
        preloaded_model=(
            model_bundle.sensor_model if model_bundle is not None else None
        ),
        preloaded_checkpoint=(
            model_bundle.sensor_checkpoint if model_bundle is not None else None
        ),
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
        "input_grid": str(wide_path),
        "aligned_output": str(aligned_path),
        "evaluation_metrics_csv": str(metrics_path),
        "evaluation_metrics_overall": evaluation_rows[0] if evaluation_rows else None,
        "grid_alignment": grid_metadata,
    })
    metadata_path = artifact_path(
        output_dir,
        "aligned",
        "stage1_imputeformer_inference_metadata",
        ".csv",
        timestamp,
    )
    write_csv_once(mapping_to_long_frame(inference_metadata), metadata_path)
    inference_metadata["inference_metadata_csv"] = str(metadata_path)
    return aligned_path, sensors, mask, inference_metadata


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = getattr(args, "artifact_timestamp", None) or artifact_timestamp()
    args.artifact_timestamp = timestamp

    # ---- resolve records ----
    if args.stdin:
        records = read_stdin_records()
    elif args.records_csv:
        records = read_csv_records(Path(args.records_csv))
    elif args.records_jsonl:
        records = read_jsonl(Path(args.records_jsonl))
    elif args.records_json is not None:
        records = json.loads(args.records_json)
    else:
        raise SystemExit(
            "No input source. Pass one of --records-csv / --records-jsonl / "
            "--stdin / --records-json."
        )

    if not records:
        raise SystemExit("Input stream is empty; nothing to score.")

    model_bundle = ProductionModelBundle.load(
        sensor_checkpoint_path=args.sensor_checkpoint,
        fusion_checkpoint_path=args.fusion_checkpoint,
        device=args.device,
    )

    records_path = artifact_path(
        output_dir, "inputs", "stage0_streaming_input_records", ".csv", timestamp)
    write_csv_once(
        pd.DataFrame.from_records(
            records, columns=("timestamp", "sensor_id", "value")
        ),
        records_path,
    )

    # ---- ImputeFormer ----
    with model_bundle.inference_lock:
        aligned_path, sensors, mask, alignment_metadata = (
            run_imputeformer_alignment_from_records(
                records,
                args,
                output_dir,
                model_bundle=model_bundle,
            )
        )

        # ---- representation + risk ----
        rep_args = SimpleNamespace(
            imputeformer_output=str(aligned_path),
            output_dir=str(output_dir),
            optical_image=args.optical_image,
            sonar_xtf=args.sonar_xtf,
            risk_config=args.risk_config,
            fusion_checkpoint=args.fusion_checkpoint,
            artifact_timestamp=timestamp,
        )
        pipeline_result = run_representation_pipeline(
            rep_args,
            model_bundle=model_bundle,
        )

    # ---- extra summary tailored to streaming jobs ----
    risk_result = pipeline_result["risk_result"]
    risk_json_path = Path(pipeline_result["risk_json_path"])
    final_risk = float(risk_result["total_risk"])
    risk_level = str(risk_result["risk_level"])
    summary = {
        "data_source": (
            "stdin" if args.stdin else
            f"csv:{args.records_csv}" if args.records_csv else
            f"jsonl:{args.records_jsonl}" if args.records_jsonl else
            "inline-json"
        ),
        "input_record_count": len(records),
        "sensors": sensors,
        "observed_ratio": float(mask.sum() / mask.size),
        "alignment_mode": alignment_metadata["mode"],
        "alignment_checkpoint": alignment_metadata["checkpoint"],
        "alignment_checkpoint_sha256": alignment_metadata["checkpoint_sha256"],
        "imputed_values": alignment_metadata["imputed_values"],
        "final_risk": final_risk,
        "risk_level": risk_level,
        "outputs": {
            "stage0_streaming_input_records": str(records_path),
            "stage1_imputeformer_aligned": str(aligned_path),
            "stage1_imputeformer_metrics": alignment_metadata[
                "evaluation_metrics_csv"
            ],
            "risk_matrix": str(risk_json_path),
        },
    }
    summary_path = artifact_path(
        output_dir, "summaries", "streaming_summary", ".csv", timestamp)
    write_csv_once(mapping_to_long_frame(summary), summary_path)
    print(f"streaming_summary={summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streaming / backend-pushed risk pipeline.",
    )
    # Input source (mutually exclusive in spirit, but the script picks the
    # first one set; we keep them as separate flags for ergonomic use).
    src = p.add_argument_group("input source (choose one)")
    src.add_argument(
        "--records-csv",
        help="Path to a long-format CSV with timestamp, sensor_id, and value columns.",
    )
    src.add_argument("--records-jsonl", help="Path to a JSONL file.")
    src.add_argument("--records-json", help="Inline JSON array string.")
    src.add_argument("--stdin", action="store_true",
                    help="Read JSONL from stdin.")
    # Output.
    p.add_argument("--output-dir", default="outputs/streaming_pipeline")
    p.add_argument(
        "--freq",
        default="auto",
        help="Use 'auto' to align to the fastest sensor, or a fixed duration.",
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
    p.add_argument("--optical-image", required=True)
    p.add_argument("--sonar-xtf", required=True)
    p.add_argument(
        "--risk-config", default="config/fixed_single_feature_risk_config.json")
    p.add_argument("--fusion-checkpoint", default=str(DEFAULT_FUSION_CHECKPOINT))
    args = p.parse_args()
    if not 0.0 < args.evaluation_holdout_ratio < 1.0:
        p.error("--evaluation-holdout-ratio must be in (0, 1)")
    return args


if __name__ == "__main__":
    run(parse_args())
