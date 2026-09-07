import contextlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError


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
from config import input_paths  # noqa: E402
from experiments.full_pipeline import run_streaming_pipeline as streaming  # noqa: E402


class RiskApiTests(unittest.TestCase):
    def test_run_request_supports_configured_images_and_raw_xtf_without_audio(self):
        fields = set(RiskRunRequest.model_fields)
        self.assertIn("optical_image", fields)
        self.assertIn("sonar_xtf", fields)
        self.assertIn("sonar_image", fields)
        self.assertFalse(any("audio" in field for field in fields))
        self.assertIsNone(RiskRunRequest().optical_image)
        self.assertIsNone(RiskRunRequest().sonar_xtf)
        self.assertIsNone(RiskRunRequest().sonar_image)
        self.assertIn("time_series_input", fields)

    def test_run_request_rejects_two_explicit_sonar_sources(self):
        with self.assertRaisesRegex(ValidationError, "only one sonar source"):
            RiskRunRequest(sonar_image="sonar.tif", sonar_xtf="sonar.xtf")

    def test_input_paths_resolve_from_project_root_and_allow_overrides(self):
        with patch.multiple(
            input_paths,
            TIME_SERIES_INPUT_PATH="data/configured.csv",
            OPTICAL_IMAGE_INPUT_PATH="data/configured.png",
            SONAR_IMAGE_INPUT_PATH=None,
            SONAR_XTF_INPUT_PATH="data/configured.xtf",
        ):
            resolved = input_paths.resolve_input_paths(optical_image="data/override.png")
        self.assertEqual(resolved.time_series, (ROOT / "data/configured.csv").resolve())
        self.assertEqual(resolved.optical_image, (ROOT / "data/override.png").resolve())
        self.assertEqual(resolved.sonar_xtf, (ROOT / "data/configured.xtf").resolve())

    def test_sonar_image_config_takes_priority_and_explicit_source_overrides_it(self):
        with patch.multiple(
            input_paths, SONAR_IMAGE_INPUT_PATH="data/sonar.tif", SONAR_XTF_INPUT_PATH="data/legacy.xtf",
        ):
            configured = input_paths.resolve_input_paths()
            explicit_xtf = input_paths.resolve_input_paths(sonar_xtf="data/override.xtf")
            explicit_image = input_paths.resolve_input_paths(sonar_image="data/override.tif")
            with self.assertRaisesRegex(ValueError, "only one sonar source"):
                input_paths.resolve_input_paths(sonar_image="a.tif", sonar_xtf="b.xtf")
        self.assertEqual(configured.sonar_source, (ROOT / "data/sonar.tif").resolve())
        self.assertIsNone(configured.sonar_xtf)
        self.assertEqual(explicit_xtf.sonar_source, (ROOT / "data/override.xtf").resolve())
        self.assertIsNone(explicit_xtf.sonar_image)
        self.assertEqual(explicit_image.sonar_source, (ROOT / "data/override.tif").resolve())
        self.assertIsNone(explicit_image.sonar_xtf)

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

    def test_relative_jobs_root_survives_restart_from_an_external_working_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "project"
            external = Path(temp_dir) / "external"
            restarted_external = Path(temp_dir) / "restart_external"
            for folder in (project, external, restarted_external):
                folder.mkdir()
            relative_jobs = Path("outputs") / "jobs"
            jobs_root = project / relative_jobs
            payload = {"total_risk": 0.2, "risk_level": "low"}
            with patch("backend.service.ROOT", project), patch.object(
                streaming.ProductionModelBundle, "load",
            ) as load:
                with contextlib.chdir(external):
                    create_app("auto", 1, relative_jobs)
                self.assertTrue(jobs_root.is_dir())
                self.assertFalse((external / relative_jobs).exists())
                risk_path = jobs_root / "persisted_job" / "risk" / "risk_matrix_test.json"
                risk_path.parent.mkdir(parents=True)
                risk_path.write_text(json.dumps(payload), encoding="utf-8")

                with contextlib.chdir(restarted_external):
                    restarted_app = create_app("auto", 1, relative_jobs)
                    endpoint = next(
                        route.endpoint for route in restarted_app.routes
                        if route.path == "/v1/risk/{job_id}/matrix"
                    )
                    response = endpoint("persisted_job")

                self.assertIsInstance(response, FileResponse)
                self.assertEqual(Path(response.path), risk_path)
                self.assertEqual(json.loads(Path(response.path).read_text(encoding="utf-8")), payload)
                self.assertFalse((restarted_external / relative_jobs).exists())
                load.assert_not_called()

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
            optical = jobs_root / "optical.png"
            sonar = jobs_root / "sonar.xtf"
            optical.touch()
            sonar.touch()
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
            ) as align:
                with self.assertRaises(HTTPException) as raised:
                    endpoints["/v1/risk/run"](RiskRunRequest(
                        job_id="failed_job",
                        optical_image=str(optical),
                        sonar_xtf=str(sonar),
                    ))

            align.assert_called_once()
            self.assertEqual(raised.exception.status_code, 400)
            self.assertIn("forced inference failure", raised.exception.detail)
            status = endpoints["/v1/buffer/status"]()
            self.assertEqual(status.raw_records, len(records))
            self.assertTrue(status.is_ready)

    def test_job_directory_creation_race_restores_flushed_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            optical, sonar = root / "optical.png", root / "sonar.tif"
            optical.touch()
            sonar.touch()
            fake_models = SimpleNamespace(
                sensor_checkpoint_path=Path("sensor.pt"),
                fusion_checkpoint_path=Path("fusion.pt"),
                device="cpu", inference_lock=threading.RLock(),
            )
            app = create_app("1s", 1, root / "jobs", model_bundle=fake_models)
            endpoints = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
            records = [
                Record(timestamp="2026-01-01T00:00:00Z", sensor_id=sensor, value=float(index))
                for index, sensor in enumerate(EXPECTED_SENSOR_IDS)
            ]
            endpoints["/v1/buffer/records"](RecordBatch(records=records))
            raced_dir = root / "jobs" / "race_job"
            original_mkdir = Path.mkdir

            def create_racing_directory(path, *args, **kwargs):
                if path == raced_dir:
                    self.assertEqual(endpoints["/v1/buffer/status"]().raw_records, 0)
                    original_mkdir(path)
                    (path / "existing.txt").write_text("other request", encoding="utf-8")
                    raise FileExistsError("another request created the output directory")
                return original_mkdir(path, *args, **kwargs)

            with patch.object(Path, "mkdir", create_racing_directory), patch.object(
                streaming, "run_imputeformer_alignment_from_records",
            ) as align:
                with self.assertRaises(HTTPException) as raised:
                    endpoints["/v1/risk/run"](RiskRunRequest(
                        job_id="race_job", optical_image=str(optical), sonar_image=str(sonar),
                    ))

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("another request", raised.exception.detail)
            align.assert_not_called()
            status = endpoints["/v1/buffer/status"]()
            self.assertEqual(status.raw_records, len(records))
            self.assertTrue(status.is_ready)
            self.assertEqual((raced_dir / "existing.txt").read_text(encoding="utf-8"), "other request")

    def test_flush_validation_failure_does_not_duplicate_buffer_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = create_app("1s", 1, root)
            endpoints = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
            records = [
                Record(timestamp="2026-01-01T00:00:00Z", sensor_id=sensor, value=float(index))
                for index, sensor in enumerate(EXPECTED_SENSOR_IDS)
            ]
            endpoints["/v1/buffer/records"](RecordBatch(records=records))

            with patch(
                "backend.service.SensorBuffer.flush_with_records",
                side_effect=ValueError("materialization failed before consuming records"),
            ) as flush, patch.object(streaming.RiskPipeline, "run") as run:
                with self.assertRaises(HTTPException) as raised:
                    endpoints["/v1/risk/run"](RiskRunRequest(
                        job_id="flush_failed", optical_image="unused.png", sonar_image="unused.tif",
                    ))

            flush.assert_called_once()
            run.assert_not_called()
            self.assertEqual(raised.exception.status_code, 400)
            self.assertIn("materialization failed", raised.exception.detail)
            status = endpoints["/v1/buffer/status"]()
            self.assertEqual(status.raw_records, len(records))
            self.assertTrue(status.is_ready)
            self.assertFalse((root / "flush_failed").exists())

    def test_file_request_downloads_json_and_preserves_existing_buffer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir) / "jobs"
            source = Path(temp_dir) / "sensors.csv"
            optical = Path(temp_dir) / "override.png"
            sonar = Path(temp_dir) / "direct_sonar.tif"
            optical.touch()
            sonar.touch()
            records = [
                {"timestamp": f"2026-01-01T00:00:{step:02d}Z", "sensor_id": sensor_id,
                 "value": float(step + sensor_index)}
                for step in range(2)
                for sensor_index, sensor_id in enumerate(EXPECTED_SENSOR_IDS)
            ]
            pd.DataFrame(records).to_csv(source, index=False)
            fake_models = SimpleNamespace(
                sensor_checkpoint_path=Path("sensor.pt"),
                fusion_checkpoint_path=Path("fusion.pt"),
                device="cpu", inference_lock=threading.RLock(),
            )
            app = create_app("auto", 1, jobs_root, model_bundle=fake_models)
            endpoints = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
            endpoints["/v1/buffer/records"](RecordBatch(records=[Record(**item) for item in records]))
            payload = {
                "modalities": {name: {"risk": 0.2} for name in ("time_series", "optical_image", "sonar_image")},
                "total_risk": 0.2, "risk_level": "low",
            }
            captured = {}

            def align(incoming, args, output_dir, model_bundle):
                captured["records"] = incoming
                return output_dir / "aligned.csv", list(EXPECTED_SENSOR_IDS), np.ones((2, 5, 1)), {
                    "mode": "imputeformer", "checkpoint": "sensor.pt", "checkpoint_sha256": "hash",
                    "evaluation_metrics_csv": str(output_dir / "metrics.csv"), "imputed_values": 0,
                }

            def represent(args, model_bundle):
                captured["optical_image"] = args.optical_image
                captured["sonar_xtf"] = args.sonar_xtf
                captured["sonar_image"] = args.sonar_image
                risk_path = Path(args.output_dir) / "risk" / "risk_matrix_test.json"
                risk_path.parent.mkdir()
                risk_path.write_text(json.dumps(payload), encoding="utf-8")
                return {
                    "risk_result": payload, "risk_json_path": str(risk_path),
                    "artifacts": {"risk_matrix": str(risk_path)},
                }

            with patch.multiple(
                input_paths, OPTICAL_IMAGE_INPUT_PATH="data/default.png", SONAR_XTF_INPUT_PATH="data/default.xtf",
            ), patch.object(streaming, "run_imputeformer_alignment_from_records", side_effect=align), patch.object(
                streaming, "run_representation_pipeline", side_effect=represent,
            ):
                response = endpoints["/v1/risk/run"](RiskRunRequest(
                    job_id="file_job", output_dir=str(Path(temp_dir) / "custom_job"),
                    time_series_input=str(source), optical_image=str(optical),
                    sonar_image=str(sonar),
                ), download=True)

            self.assertIsInstance(response, FileResponse)
            self.assertEqual(json.loads(Path(response.path).read_text(encoding="utf-8")), payload)
            self.assertEqual(captured["records"], records)
            self.assertEqual(captured["optical_image"], str(optical.resolve()))
            self.assertIsNone(captured["sonar_xtf"])
            self.assertEqual(captured["sonar_image"], str(sonar.resolve()))
            self.assertEqual(endpoints["/v1/buffer/status"]().raw_records, len(records))
            self.assertEqual(Path(endpoints["/v1/risk/{job_id}/matrix"]("file_job").path), Path(response.path))
            self.assertFalse((Path(temp_dir) / "custom_job" / "inputs").exists())


class StreamingCliTests(unittest.TestCase):
    def test_inline_records_must_be_an_array_without_falling_back_to_configured_input(self):
        for invalid_json in ("null", "{}"):
            with self.subTest(records_json=invalid_json), patch.object(
                sys, "argv", ["run", "--records-json", invalid_json],
            ):
                args = streaming.parse_args()
            with patch.multiple(
                input_paths, TIME_SERIES_INPUT_PATH="data/configured.csv",
            ), patch.object(streaming.RiskPipeline, "run") as run, patch.object(
                streaming, "read_time_series_records",
            ) as read, patch.object(streaming.ProductionModelBundle, "load") as load:
                with self.subTest(records_json=invalid_json), self.assertRaisesRegex(ValueError, "array"):
                    streaming.run(args)
                run.assert_not_called()
                read.assert_not_called()
                load.assert_not_called()

    def test_cli_rejects_ambiguous_time_series_sources(self):
        with patch.object(sys, "argv", ["run", "--records-json", "[]", "--stdin"]):
            with self.assertRaises(SystemExit) as raised:
                streaming.parse_args()
        self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_two_sonar_sources_and_accepts_direct_image(self):
        with patch.object(sys, "argv", ["run", "--sonar-image", "sonar.tif", "--sonar-xtf", "sonar.xtf"]):
            with self.assertRaises(SystemExit) as raised:
                streaming.parse_args()
        self.assertEqual(raised.exception.code, 2)
        with patch.object(sys, "argv", ["run", "--sonar-image", "sonar.tif"]):
            args = streaming.parse_args()
        self.assertEqual(args.sonar_image, "sonar.tif")
        self.assertIsNone(args.sonar_xtf)

    def test_configured_cli_saves_only_alignment_metrics_and_risk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sensors.csv"
            output_dir = Path(temp_dir) / "output"
            optical = Path(temp_dir) / "configured.png"
            sonar = Path(temp_dir) / "configured_sonar.tif"
            optical.touch()
            sonar.touch()
            index = pd.date_range("2026-01-01", periods=2, freq="s", tz="UTC")
            frame = pd.DataFrame(np.ones((2, 5)), columns=EXPECTED_SENSOR_IDS, index=index)
            frame.to_csv(source, index_label="timestamp")
            fake_models = SimpleNamespace(
                sensor_checkpoint_path=Path("sensor.pt"), fusion_checkpoint_path=Path("fusion.pt"),
                sensor_model=object(), sensor_checkpoint={}, inference_lock=threading.RLock(),
            )
            inference_metadata = {
                "sensor_ids": list(EXPECTED_SENSOR_IDS), "mode": "imputeformer",
                "checkpoint": "sensor.pt", "checkpoint_sha256": "hash", "imputed_values": 0,
                "evaluation_metrics": [{"sensor_id": "overall", "mae": 0.01}],
            }

            def represent(args, model_bundle):
                self.assertEqual(args.optical_image, str(optical.resolve()))
                self.assertEqual(args.sonar_image, str(sonar.resolve()))
                self.assertIsNone(args.sonar_xtf)
                path = output_dir / "risk" / "risk_matrix_test.json"
                path.parent.mkdir()
                payload = {"total_risk": 0.2, "risk_level": "low"}
                path.write_text(json.dumps(payload), encoding="utf-8")
                return {
                    "risk_result": payload, "risk_json_path": str(path),
                    "artifacts": {"risk_matrix": str(path)},
                }

            with patch.object(sys, "argv", ["run", "--output-dir", str(output_dir)]):
                args = streaming.parse_args()
            with patch.multiple(
                input_paths, TIME_SERIES_INPUT_PATH=source,
                OPTICAL_IMAGE_INPUT_PATH=optical, SONAR_IMAGE_INPUT_PATH=sonar,
                SONAR_XTF_INPUT_PATH="data/configured.xtf",
            ), patch.object(streaming.ProductionModelBundle, "load", return_value=fake_models), patch.object(
                streaming, "align_with_pretrained_imputeformer", return_value=(frame, inference_metadata),
            ), patch.object(streaming, "run_representation_pipeline", side_effect=represent):
                with self.assertLogs("f-model.streaming", level="INFO") as logged:
                    summary = streaming.run(args)

            self.assertEqual(summary["input_record_count"], 10)
            files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
            self.assertEqual(len(files), 3)
            self.assertEqual(len(list((output_dir / "aligned").glob("*.csv"))), 2)
            self.assertFalse((output_dir / "inputs").exists())
            self.assertFalse((output_dir / "summaries").exists())
            self.assertTrue(any("time_series=" in line for line in logged.output))
            self.assertTrue(any("sonar_source=" in line for line in logged.output))
            self.assertTrue(any("Alignment completed" in line for line in logged.output))


if __name__ == "__main__":
    unittest.main()
