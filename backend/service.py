"""HTTP service that wraps the streaming risk pipeline.

Run::

    python -m backend.service --host 0.0.0.0 --port 8000

Endpoints
---------
POST /v1/buffer/records         Append one or many records to the in-memory buffer.
POST /v1/buffer/clear           Clear the buffer.
GET  /v1/buffer/status          Buffer progress (rows / sensors / ready flag).
POST /v1/risk/run               Flush the buffer and run the full pipeline.
GET  /v1/risk/{job_id}          Read the final risk result of a previous run.
GET  /v1/health                 Liveness probe.

The service infers each sensor cadence and aligns every stream to the fastest
sensor in that buffer. A production run requires the five sensor identifiers
embedded in the trained checkpoint. There is no site or coordinate entity.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

# Make the experiments package importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
from risk_pipeline.artifacts import (
    artifact_path,
    artifact_timestamp,
    latest_file,
    write_csv_once,
)
from risk_pipeline.data_io import SensorBuffer
from risk_pipeline.model_bundle import (
    DEFAULT_FUSION_CHECKPOINT,
    ProductionModelBundle,
)
from risk_pipeline.sensor_imputation import (
    DEFAULT_SENSOR_CHECKPOINT,
    EXPECTED_SENSOR_IDS,
)


logger = logging.getLogger("feature-model.backend")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Record(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 timestamp with timezone.")
    sensor_id: str = Field(
        ...,
        description=(
            "One canonical sensor identifier: water_temperature_c, "
            "water_depth_m, turbidity_ntu, significant_wave_height_m, or "
            "current_speed_mps. water_depth_m is direct immersion depth in "
            "metres, measured from the water surface and positive downward."
        ),
    )
    value: Optional[float] = Field(
        default=None,
        description=(
            "Numeric reading. Use null for missing observations; "
            "do not substitute 0 / -1 / 99 / 999 as sentinels."
        ),
    )


class RecordBatch(BaseModel):
    records: List[Record]
    buffer_key: Optional[str] = Field(
        default="default",
        description="Logical buffer name; use one per mobile cable stream.",
    )


class BufferStatus(BaseModel):
    buffer_key: str
    raw_records: int
    sensors: int
    resampled_rows: int
    min_window: int
    alignment_frequency_seconds: Optional[float] = None
    fastest_sensor_ids: List[str] = Field(default_factory=list)
    sampling_intervals_seconds: Dict[str, Optional[float]] = Field(
        default_factory=dict
    )
    is_ready: bool


class RiskRunRequest(BaseModel):
    buffer_key: str = "default"
    job_id: Optional[str] = Field(
        default=None,
        description="Optional. If empty the server assigns a UUID.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Override the per-job output directory. "
                    "Default: <jobs_root>/<job_id>.",
    )
    optical_image: str
    sonar_xtf: str = Field(
        ..., description="Path to a raw sonar .xtf file."
    )


# ---------------------------------------------------------------------------
# In-memory buffer registry
# ---------------------------------------------------------------------------

class BufferRegistry:
    def __init__(self, default_freq: str, default_min_window: int) -> None:
        self._lock = threading.Lock()
        self._buffers: Dict[str, SensorBuffer] = {}
        self.default_freq = default_freq
        self.default_min_window = default_min_window

    def get(self, key: str) -> SensorBuffer:
        with self._lock:
            buf = self._buffers.get(key)
            if buf is None:
                buf = SensorBuffer(
                    freq=self.default_freq,
                    min_window=self.default_min_window,
                )
                self._buffers[key] = buf
            return buf

    def clear(self, key: str) -> None:
        with self._lock:
            buf = self._buffers.pop(key, None)
            if buf is not None:
                buf.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(
    default_freq: str,
    default_min_window: int,
    jobs_root: Path,
    sensor_checkpoint: Path = DEFAULT_SENSOR_CHECKPOINT,
    fusion_checkpoint: Path = DEFAULT_FUSION_CHECKPOINT,
    model_bundle: ProductionModelBundle | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Feature Model Risk Service",
        version="1.0.0",
        description=(
            "Buffer-based HTTP wrapper around the streaming risk pipeline. "
            "Records use timestamp, sensor_id, and value. The five accepted "
            f"sensor_id values are: {', '.join(EXPECTED_SENSOR_IDS)}. "
            "There is no site or coordinate field."
        ),
    )
    registry = BufferRegistry(default_freq, default_min_window)
    jobs_root.mkdir(parents=True, exist_ok=True)
    model_bundle_holder = {"value": model_bundle}
    model_load_lock = threading.Lock()

    # ----- helpers ---------------------------------------------------------
    def _record_to_dict(r: Record) -> Dict:
        return {"timestamp": r.timestamp, "sensor_id": r.sensor_id,
                "value": r.value}

    def _is_ready(progress: Dict) -> bool:
        return (
            progress["resampled_rows"] >= progress["min_window"]
            and progress["sensors"] == len(EXPECTED_SENSOR_IDS)
        )

    def _validate_job_id(job_id: str) -> None:
        if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
            raise HTTPException(
                status_code=422,
                detail="job_id must be a single directory name.",
            )

    def _get_model_bundle() -> ProductionModelBundle:
        with model_load_lock:
            loaded = model_bundle_holder["value"]
            if loaded is None:
                loaded = ProductionModelBundle.load(
                    sensor_checkpoint_path=sensor_checkpoint,
                    fusion_checkpoint_path=fusion_checkpoint,
                    device="auto",
                )
                model_bundle_holder["value"] = loaded
            return loaded

    def _run_pipeline_for(records: List[Dict],
                          job_dir: Path,
                          req: RiskRunRequest) -> Dict:
        """Invoke the streaming entry script in-process."""
        from experiments.full_pipeline.run_streaming_pipeline import (
            run_imputeformer_alignment_from_records,
        )
        from risk_pipeline.run_encapsulated_pipeline import (
            run as run_representation_pipeline,
        )
        timestamp = artifact_timestamp()

        # Write the batch to disk for traceability.
        records_path = artifact_path(
            job_dir, "inputs", "stage0_streaming_input_records", ".csv", timestamp)
        write_csv_once(
            pd.DataFrame.from_records(
                records,
                columns=("timestamp", "sensor_id", "value"),
            ),
            records_path,
        )

        models = _get_model_bundle()
        ns = SimpleNamespace(
            freq=default_freq,
            batch_size=8,
            sensor_checkpoint=str(models.sensor_checkpoint_path),
            fusion_checkpoint=str(models.fusion_checkpoint_path),
            device=str(models.device),
            evaluation_holdout_ratio=0.10,
            evaluation_seed=42,
            optical_image=req.optical_image,
            sonar_xtf=req.sonar_xtf,
            artifact_timestamp=timestamp,
        )

        with models.inference_lock:
            aligned_path, sensors, mask, alignment_metadata = (
                run_imputeformer_alignment_from_records(
                    records,
                    ns,
                    job_dir,
                    model_bundle=models,
                )
            )

            rep_args = SimpleNamespace(
                imputeformer_output=str(aligned_path),
                output_dir=str(job_dir),
                optical_image=ns.optical_image,
                sonar_xtf=ns.sonar_xtf,
                fusion_checkpoint=str(models.fusion_checkpoint_path),
                artifact_timestamp=timestamp,
            )
            pipeline_result = run_representation_pipeline(
                rep_args,
                model_bundle=models,
            )

        risk_result = pipeline_result["risk_result"]
        risk_path = Path(pipeline_result["risk_json_path"])
        alignment_metrics = alignment_metadata.get(
            "evaluation_metrics_overall"
        )

        return {
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "input_record_count": len(records),
            "input_records_path": str(records_path),
            "sensors": sensors,
            "observed_ratio": float(mask.sum() / mask.size),
            "alignment_mode": alignment_metadata["mode"],
            "alignment_checkpoint": alignment_metadata["checkpoint"],
            "alignment_checkpoint_sha256": alignment_metadata["checkpoint_sha256"],
            "alignment_metrics_path": alignment_metadata["evaluation_metrics_csv"],
            "alignment_metrics": alignment_metrics,
            "imputed_values": alignment_metadata["imputed_values"],
            "final_risk": float(risk_result["total_risk"]),
            "risk_level": str(risk_result["risk_level"]),
            "risk_matrix": risk_result,
            "risk_matrix_path": str(risk_path),
        }

    # ----- routes ----------------------------------------------------------
    @app.get("/v1/health")
    def health() -> Dict:
        return {
            "status": "ok",
            "models_loaded": model_bundle_holder["value"] is not None,
            "ts": int(time.time()),
        }

    @app.post("/v1/buffer/records", response_model=BufferStatus)
    def append_records(batch: RecordBatch) -> BufferStatus:
        invalid = sorted({
            record.sensor_id
            for record in batch.records
            if record.sensor_id not in EXPECTED_SENSOR_IDS
        })
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported sensor_id values: {invalid}. Expected exactly "
                    f"{list(EXPECTED_SENSOR_IDS)}; site/station fields are not used."
                ),
            )
        buf = registry.get(batch.buffer_key)
        buf.extend(_record_to_dict(record) for record in batch.records)
        progress = buf.ready_progress()
        return BufferStatus(
            buffer_key=batch.buffer_key,
            is_ready=_is_ready(progress),
            **progress,
        )

    @app.post("/v1/buffer/clear")
    def clear_buffer(buffer_key: str = "default") -> Dict:
        registry.clear(buffer_key)
        return {"buffer_key": buffer_key, "cleared": True}

    @app.get("/v1/buffer/status", response_model=BufferStatus)
    def buffer_status(buffer_key: str = "default") -> BufferStatus:
        buf = registry.get(buffer_key)
        progress = buf.ready_progress()
        return BufferStatus(
            buffer_key=buffer_key,
            is_ready=_is_ready(progress),
            **progress,
        )

    @app.post("/v1/risk/run")
    def run_risk(req: RiskRunRequest) -> JSONResponse:
        buf = registry.get(req.buffer_key)
        progress = buf.ready_progress()
        if not _is_ready(progress):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Buffer '{req.buffer_key}' is not ready. "
                    f"Progress: {progress}; expected all five sensors."
                ),
            )
        job_id = req.job_id or (
            f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_"
            f"{uuid.uuid4().hex[:6]}"
        )
        _validate_job_id(job_id)
        job_dir = Path(req.output_dir) if req.output_dir else jobs_root / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Output directory already exists; refusing to overwrite: {job_dir}",
            ) from exc

        records, _, _, _, _, _ = buf.flush_with_records()

        try:
            result = _run_pipeline_for(records, job_dir, req)
        except ValueError as exc:
            buf.extend(records)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            buf.extend(records)
            raise
        return JSONResponse(result)

    @app.get("/v1/risk/{job_id}/matrix")
    def get_risk_matrix(job_id: str) -> FileResponse:
        _validate_job_id(job_id)
        path = latest_file(
            jobs_root / job_id,
            "risk",
            "risk_matrix_*.json",
        )
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail=f"No job {job_id}.")
        return FileResponse(path, media_type="application/json", filename=path.name)

    return app
# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--freq",
        default="auto",
        help="Use 'auto' to align to the fastest sensor, or a fixed duration.",
    )
    ap.add_argument(
        "--min-window",
        type=int,
        default=3600,
        help="Minimum aligned rows; the second-level checkpoint needs 3600.",
    )
    ap.add_argument(
        "--sensor-checkpoint",
        default=str(DEFAULT_SENSOR_CHECKPOINT),
        help="Trained five-sensor ImputeFormer checkpoint.",
    )
    ap.add_argument(
        "--fusion-checkpoint",
        default=str(DEFAULT_FUSION_CHECKPOINT),
        help="Trained feature-fusion Transformer checkpoint.",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    ap.add_argument(
        "--jobs-root",
        default=str(ROOT / "outputs" / "streaming_jobs"),
        help=(
            "Where to write per-job artifacts. Default is anchored to the "
            "project root (parent of the backend/ package) so that the path "
            "is the same no matter which directory uvicorn is started from. "
            "Pass an absolute path to override."
        ),
    )
    args = ap.parse_args()
    if args.min_window < 3600:
        ap.error("the production sensor checkpoint requires --min-window >= 3600")

    import uvicorn
    model_bundle = ProductionModelBundle.load(
        sensor_checkpoint_path=Path(args.sensor_checkpoint),
        fusion_checkpoint_path=Path(args.fusion_checkpoint),
        device=args.device,
    )
    app = create_app(
        args.freq,
        args.min_window,
        Path(args.jobs_root),
        Path(args.sensor_checkpoint),
        Path(args.fusion_checkpoint),
        model_bundle,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
