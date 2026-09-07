"""HTTP service that wraps the streaming risk pipeline.

Run::

    python -m backend.service --host 0.0.0.0 --port 8000

Endpoints
---------
POST /v1/buffer/records         Append one or many records to the in-memory buffer.
POST /v1/buffer/clear           Clear the buffer.
GET  /v1/buffer/status          Buffer progress (rows / sensors / ready flag).
POST /v1/risk/run               Flush the buffer and run the full pipeline.
POST /v1/risk/run?download=true Run inference and download the risk JSON file.
GET  /v1/risk/{job_id}/matrix   Download the final risk JSON file.
GET  /v1/health                 Liveness probe.

The service infers each sensor cadence and aligns every stream automatically.
A production run requires the five sensor identifiers
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
from typing import Dict, List, Optional

# Make the experiments package importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator
from config.input_paths import resolve_input_paths
from experiments.full_pipeline.run_streaming_pipeline import RiskPipeline
from risk_pipeline.artifacts import (
    latest_file,
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


logger = logging.getLogger("f-model.backend")
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
    reference_sensor_id: Optional[str] = None
    reference_interval_seconds: Optional[float] = None
    observation_counts: Dict[str, int] = Field(default_factory=dict)
    reference_selection: Optional[str] = None
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
    time_series_input: Optional[str] = Field(
        default=None,
        description=(
            "CSV/JSONL input path. Overrides buffered records for this request. "
            "With an empty buffer, defaults to config/input_paths.py."
        ),
    )
    optical_image: Optional[str] = Field(
        default=None,
        description="Optical image file or directory; defaults to OPTICAL_IMAGE_INPUT_PATH in config/input_paths.py.",
    )
    sonar_xtf: Optional[str] = Field(
        default=None,
        description="Legacy raw sonar .xtf path; mutually exclusive with sonar_image.",
    )
    sonar_image: Optional[str] = Field(
        default=None,
        description="Sonar TIFF (.tif/.tiff) file or directory; defaults to SONAR_IMAGE_INPUT_PATH in config/input_paths.py.",
    )

    @model_validator(mode="after")
    def validate_sonar_source(self):
        if self.sonar_image is not None and self.sonar_xtf is not None:
            raise ValueError("Provide only one sonar source: sonar_image or sonar_xtf.")
        return self


# ---------------------------------------------------------------------------
# In-memory buffer registry
# ---------------------------------------------------------------------------

class BufferRegistry:
    def __init__(self, default_freq: str, default_min_window: int, *,
                 preserve_actual_timestamps: bool = True) -> None:
        self._lock = threading.Lock()
        self._buffers: Dict[str, SensorBuffer] = {}
        self.default_freq = default_freq
        self.default_min_window = default_min_window
        self.preserve_actual_timestamps = preserve_actual_timestamps

    def get(self, key: str) -> SensorBuffer:
        with self._lock:
            buf = self._buffers.get(key)
            if buf is None:
                buf = SensorBuffer(
                    freq=self.default_freq,
                    min_window=self.default_min_window,
                )
                self._buffers[key] = buf
            buf.preserve_actual_timestamps = self.preserve_actual_timestamps
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
    jobs_root = Path(jobs_root).expanduser()
    jobs_root = (jobs_root if jobs_root.is_absolute() else ROOT / jobs_root).resolve()
    app = FastAPI(
        title="F-Model Risk Service",
        version="1.0.0",
        description=(
            "Buffer-based HTTP wrapper around the streaming risk pipeline. "
            "Records use timestamp, sensor_id, and value. The five accepted "
            f"sensor_id values are: {', '.join(EXPECTED_SENSOR_IDS)}. "
            "There is no site or coordinate field."
        ),
    )
    pipeline = RiskPipeline(
        freq=default_freq, output_root=jobs_root,
        sensor_checkpoint=sensor_checkpoint, fusion_checkpoint=fusion_checkpoint,
        model_bundle=model_bundle,
    )
    registry = BufferRegistry(
        default_freq, default_min_window,
        preserve_actual_timestamps=bool(
            getattr(model_bundle, "sensor_checkpoint", {}).get("model_config", {}).get("time_aware", False)
        ),
    )
    jobs_root.mkdir(parents=True, exist_ok=True)
    model_bundle_holder = {"value": model_bundle}
    model_load_lock = threading.Lock()
    completed_jobs: Dict[str, Path] = {}

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
                loaded = pipeline.load_models()
                model_bundle_holder["value"] = loaded
                registry.preserve_actual_timestamps = bool(
                    getattr(loaded, "sensor_checkpoint", {}).get("model_config", {}).get("time_aware", False)
                )
            return loaded

    def _run_pipeline_for(records: List[Dict],
                          job_dir: Path,
                          req: RiskRunRequest) -> Dict:
        """Use the same reusable pipeline as Python callers and the CLI."""
        logger.info(
            "Inference started: job=%s records=%d time_series=%s optical_image=%s sonar_source=%s",
            job_dir.name, len(records), req.time_series_input or f"buffer:{req.buffer_key}",
            req.optical_image, req.sonar_image or req.sonar_xtf,
        )

        _get_model_bundle()
        return pipeline.run(
            records=records, output_dir=job_dir,
            source_name=req.time_series_input or f"buffer:{req.buffer_key}",
            optical_image=req.optical_image,
            sonar_xtf=req.sonar_xtf,
            sonar_image=req.sonar_image,
        )

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
        # Lazy apps resolve the checkpoint contract before counting aligned rows.
        if batch.records and str(default_freq).strip().lower() == "auto":
            _get_model_bundle()
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

    @app.post("/v1/risk/run", response_model=None)
    def run_risk(req: RiskRunRequest, download: bool = False) -> JSONResponse | FileResponse:
        from experiments.full_pipeline.run_streaming_pipeline import read_time_series_records

        buf = registry.get(req.buffer_key)
        paths = resolve_input_paths(
            time_series=req.time_series_input,
            optical_image=req.optical_image,
            sonar_xtf=req.sonar_xtf,
            sonar_image=req.sonar_image,
        )
        read_from_file = req.time_series_input is not None or (
            len(buf) == 0 and paths.time_series is not None
        )
        if read_from_file:
            if paths.time_series is None:
                raise HTTPException(status_code=422, detail="time_series_input cannot be empty.")
            try:
                records = read_time_series_records(paths.time_series)
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not records:
                raise HTTPException(status_code=400, detail="Time-series input is empty.")
        else:
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
        job_dir = (job_dir if job_dir.is_absolute() else ROOT / job_dir).resolve()
        if paths.optical_image is None or paths.sonar_source is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Provide optical_image and a sonar TIFF via sonar_image, or configure "
                    "OPTICAL_IMAGE_INPUT_PATH and SONAR_IMAGE_INPUT_PATH in config/input_paths.py."
                ),
            )
        if job_dir.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Output directory already exists; refusing to overwrite: {job_dir}",
            )
        resolved_request = req.model_copy(update={
            "time_series_input": str(paths.time_series) if read_from_file else None,
            "optical_image": str(paths.optical_image),
            "sonar_xtf": str(paths.sonar_xtf) if paths.sonar_xtf is not None else None,
            "sonar_image": str(paths.sonar_image) if paths.sonar_image is not None else None,
        })
        flushed_records = None
        try:
            if not read_from_file:
                records, _, _, _, _, _ = buf.flush_with_records()
                flushed_records = records
            result = _run_pipeline_for(records, job_dir, resolved_request)
        except FileExistsError as exc:
            if flushed_records is not None:
                buf.extend(flushed_records)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ValueError, OSError) as exc:
            if flushed_records is not None:
                buf.extend(flushed_records)
            logger.exception("Inference failed: job=%s", job_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            if flushed_records is not None:
                buf.extend(flushed_records)
            logger.exception("Inference failed: job=%s", job_id)
            raise
        result["job_id"] = job_id
        completed_jobs[job_id] = job_dir
        logger.info(
            "Inference completed: job=%s total_risk=%s risk_level=%s risk_json=%s",
            job_id, result["final_risk"], result["risk_level"], result["risk_matrix_path"],
        )
        if download:
            path = Path(result["risk_matrix_path"])
            return FileResponse(path, media_type="application/json", filename=path.name)
        return JSONResponse(result)

    @app.get("/v1/risk/{job_id}/matrix")
    def get_risk_matrix(job_id: str) -> FileResponse:
        _validate_job_id(job_id)
        path = latest_file(
            completed_jobs.get(job_id, jobs_root / job_id),
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
        help="Use 'auto' to align to the sensor with the most observations, or a fixed duration.",
    )
    ap.add_argument(
        "--min-window",
        type=int,
        default=1,
        help="Optional buffer threshold in aligned rows; arbitrary positive lengths are supported.",
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
    if args.min_window < 1:
        ap.error("--min-window must be positive")

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
