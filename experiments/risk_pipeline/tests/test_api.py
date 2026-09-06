import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import FileResponse


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.service import (  # noqa: E402
    EXPECTED_SENSOR_IDS,
    Record,
    RecordBatch,
    RiskRunRequest,
    create_app,
)


class RiskApiTests(unittest.TestCase):
    def test_run_request_requires_optical_and_raw_xtf_without_audio(self):
        fields = set(RiskRunRequest.model_fields)
        self.assertIn("optical_image", fields)
        self.assertIn("sonar_xtf", fields)
        self.assertNotIn("sonar_image", fields)
        self.assertFalse(any("audio" in field for field in fields))

    def test_matrix_endpoint_returns_saved_json_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir)
            risk_dir = jobs_root / "job_001" / "risk"
            risk_dir.mkdir(parents=True)
            payload = {
                "modalities": {
                    name: {"risk": 0.2, "fixed_weight": weight}
                    for name, weight in (
                        ("time_series", 0.20),
                        ("optical_image", 0.40),
                        ("sonar_image", 0.40),
                    )
                },
                "total_risk": 0.2,
            }
            result_path = risk_dir / "risk_matrix_20260101_000000_000001.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            app = create_app("1min", 12, jobs_root)
            endpoint = next(
                route.endpoint
                for route in app.routes
                if route.path == "/v1/risk/{job_id}/matrix"
            )
            response = endpoint("job_001")

            self.assertIsInstance(response, FileResponse)
            self.assertEqual(response.media_type, "application/json")
            self.assertEqual(Path(response.path), result_path)
            self.assertEqual(
                json.loads(Path(response.path).read_text(encoding="utf-8")),
                payload,
            )

    def test_existing_job_directory_is_rejected_without_flushing_buffer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir)
            existing = jobs_root / "existing_job"
            existing.mkdir()
            app = create_app("1min", 2, jobs_root)
            endpoints = {
                route.path: route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
            }
            records = []
            for step in range(2):
                for sensor_index, sensor_id in enumerate(EXPECTED_SENSOR_IDS):
                    records.append(Record(
                        timestamp=f"2026-01-01T00:0{step}:00Z",
                        sensor_id=sensor_id,
                        value=float(step + sensor_index),
                    ))
            endpoints["/v1/buffer/records"](RecordBatch(records=records))

            with self.assertRaises(HTTPException) as raised:
                endpoints["/v1/risk/run"](RiskRunRequest(
                    job_id="existing_job",
                    optical_image="unused.png",
                    sonar_xtf="unused.xtf",
                ))

            self.assertEqual(raised.exception.status_code, 409)
            status = endpoints["/v1/buffer/status"]()
            self.assertTrue(status.is_ready)

    def test_pipeline_failure_restores_original_buffer_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir)
            fake_models = SimpleNamespace(
                sensor_checkpoint_path=Path("sensor.pt"),
                fusion_checkpoint_path=Path("fusion.pt"),
                device="cpu",
                inference_lock=threading.RLock(),
            )
            app = create_app(
                "1min",
                2,
                jobs_root,
                model_bundle=fake_models,
            )
            endpoints = {
                route.path: route.endpoint
                for route in app.routes
                if hasattr(route, "endpoint")
            }
            records = [
                Record(
                    timestamp=f"2026-01-01T00:0{step}:00Z",
                    sensor_id=sensor_id,
                    value=float(step + sensor_index),
                )
                for step in range(2)
                for sensor_index, sensor_id in enumerate(EXPECTED_SENSOR_IDS)
            ]
            endpoints["/v1/buffer/records"](RecordBatch(records=records))

            with patch(
                "experiments.full_pipeline.run_streaming_pipeline."
                "run_imputeformer_alignment_from_records",
                side_effect=ValueError("forced inference failure"),
            ):
                with self.assertRaises(HTTPException) as raised:
                    endpoints["/v1/risk/run"](RiskRunRequest(
                        job_id="failed_job",
                        optical_image="unused.png",
                        sonar_xtf="unused.xtf",
                    ))

            self.assertEqual(raised.exception.status_code, 400)
            status = endpoints["/v1/buffer/status"]()
            self.assertEqual(status.raw_records, len(records))
            self.assertTrue(status.is_ready)


if __name__ == "__main__":
    unittest.main()
