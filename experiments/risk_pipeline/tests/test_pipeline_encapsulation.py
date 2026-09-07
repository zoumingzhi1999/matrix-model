import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import input_paths
from experiments.full_pipeline import run_streaming_pipeline as streaming
from f_model import RiskPipeline
from risk_pipeline.sensor_schema import OCEAN_SENSOR_IDS


class TrackingLock:
    def __init__(self):
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, *exc_info):
        self.depth -= 1


class PipelineEncapsulationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pipeline_test_", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = self.root / "jobs"
        self.records = [
            {
                "timestamp": f"2026-01-01T00:00:{step:02d}Z",
                "sensor_id": sensor,
                "value": float(index + step),
            }
            for step in range(2)
            for index, sensor in enumerate(OCEAN_SENSOR_IDS)
        ]
        self.csv = self.root / "sensors.csv"
        pd.DataFrame(self.records).to_csv(self.csv, index=False)
        self.optical = self.root / "optical.png"
        self.sonar = self.root / "sonar.tif"
        self.xtf = self.root / "sonar.xtf"
        for path in (self.optical, self.sonar, self.xtf):
            path.write_bytes(b"mocked image input")
        self.lock = TrackingLock()
        self.bundle = SimpleNamespace(
            sensor_checkpoint_path=ROOT / "models/sensor_imputeformer/imputeformer_ocean_five_sensor.pt",
            fusion_checkpoint_path=ROOT / "models/multimodal_fusion/multimodal_fusion.pt",
            device="cpu",
            inference_lock=self.lock,
        )
        self.alignment_calls = []
        self.representation_calls = []
        self.load = self._patch(
            streaming.ProductionModelBundle, "load", return_value=self.bundle,
        )
        self.align = self._patch(
            streaming, "run_imputeformer_alignment_from_records",
            side_effect=self._alignment_stage,
        )
        self.represent = self._patch(
            streaming, "run_representation_pipeline",
            side_effect=self._representation_stage,
        )
        configured = patch.multiple(
            input_paths,
            TIME_SERIES_INPUT_PATH=None,
            OPTICAL_IMAGE_INPUT_PATH=None,
            SONAR_IMAGE_INPUT_PATH=None,
            SONAR_XTF_INPUT_PATH=None,
        )
        configured.start()
        self.addCleanup(configured.stop)

    def _patch(self, target, name, **kwargs):
        patcher = patch.object(target, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def _pipeline(self, **kwargs):
        return RiskPipeline(output_root=self.jobs, **kwargs)

    def _run(self, pipeline, **kwargs):
        arguments = {"optical_image": self.optical, "sonar_image": self.sonar}
        arguments.update(kwargs)
        return pipeline.run(**arguments)

    def _alignment_stage(self, records, args, output_dir, model_bundle=None):
        self.assertIs(model_bundle, self.bundle)
        self.assertGreater(self.lock.depth, 0)
        self.alignment_calls.append((records, args, Path(output_dir)))
        aligned_dir = Path(output_dir) / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        timestamp = args.artifact_timestamp
        aligned = aligned_dir / f"stage1_imputeformer_aligned_output_{timestamp}.csv"
        metrics = aligned_dir / f"stage1_imputeformer_evaluation_metrics_{timestamp}.csv"
        pd.DataFrame({sensor: [1.0, 2.0] for sensor in OCEAN_SENSOR_IDS}).to_csv(
            aligned, index=False,
        )
        pd.DataFrame([{"sensor_id": "overall", "mae": 0.1}]).to_csv(metrics, index=False)
        mask = np.ones((2, len(OCEAN_SENSOR_IDS)), dtype=bool)
        mask[0, 0] = False
        return aligned, list(OCEAN_SENSOR_IDS), mask, {
            "mode": "pretrained_imputeformer",
            "checkpoint": str(self.bundle.sensor_checkpoint_path),
            "checkpoint_sha256": "test-checkpoint-sha256",
            "imputed_values": 1,
            "aligned_output": str(aligned),
            "evaluation_metrics_csv": str(metrics),
            "evaluation_metrics_overall": {"sensor_id": "overall", "mae": 0.1},
            "grid_alignment": {"reference_interval_seconds": 2.0},
        }

    def _representation_stage(self, args, model_bundle=None, **kwargs):
        self.assertIs(model_bundle, self.bundle)
        self.assertGreater(self.lock.depth, 0)
        self.assertFalse(kwargs.get("print_result", getattr(args, "print_result", True)))
        self.representation_calls.append(args)
        output = Path(args.output_dir)
        timestamp = args.artifact_timestamp
        artifacts = {"aligned": str(args.imputeformer_output)}
        for stage in ("first_stage_features", "second_stage_features"):
            folder = output / stage
            folder.mkdir(parents=True, exist_ok=True)
            artifacts[stage] = {}
            for modality in ("time_series", "optical_image", "sonar_image"):
                path = folder / f"{modality}_{timestamp}.csv"
                path.write_text("feature,value\nexample,0.25\n", encoding="utf-8")
                artifacts[stage][modality] = str(path)
        risk = {
            "modalities": {
                modality: {"risk": 0.2}
                for modality in ("time_series", "optical_image", "sonar_image")
            },
            "total_risk": 0.2,
            "risk_level": "low",
        }
        risk_folder = output / "risk"
        risk_folder.mkdir(parents=True, exist_ok=True)
        risk_path = risk_folder / f"risk_matrix_{timestamp}.json"
        risk_path.write_text(json.dumps(risk), encoding="utf-8")
        artifacts["risk_matrix"] = str(risk_path)
        return {"risk_result": risk, "risk_json_path": risk_path, "artifacts": artifacts}

    def test_public_facade_exports_the_production_pipeline(self):
        self.assertIs(RiskPipeline, streaming.RiskPipeline)

    def test_repeated_runs_reuse_models_lock_both_stages_and_keep_nine_outputs(self):
        pipeline = self._pipeline()
        self.load.assert_not_called()
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            first = self._run(pipeline, records=self.records)
            second = self._run(pipeline, records=self.records)
        self.load.assert_called_once()
        self.assertEqual(capture.getvalue(), "")
        self.assertNotEqual(first["job_dir"], second["job_dir"])
        self.assertEqual(self.lock.depth, 0)
        self.assertEqual(self.lock.entries, 2)
        for (_, alignment_args, _), representation_args, result in zip(
            self.alignment_calls, self.representation_calls, (first, second),
        ):
            self.assertEqual(alignment_args.artifact_timestamp, representation_args.artifact_timestamp)
            self.assertEqual(len([p for p in Path(result["job_dir"]).rglob("*") if p.is_file()]), 9)
            self.assertEqual(set(result["artifacts"]), {
                "aligned", "alignment_metrics", "first_stage_features",
                "second_stage_features", "risk_matrix",
            })

    def test_file_and_records_share_the_result_contract(self):
        pipeline = self._pipeline(model_bundle=self.bundle)
        from_file = self._run(pipeline, time_series_input=self.csv)
        from_records = self._run(pipeline, records=self.records, source_name="sensor-buffer")
        self.load.assert_not_called()
        self.assertEqual(from_file["data_source"], str(self.csv.resolve()))
        self.assertEqual(from_records["data_source"], "sensor-buffer")
        self.assertEqual(set(from_file), set(from_records))
        api_fields = {
            "job_id", "job_dir", "input_record_count", "sensors", "observed_ratio",
            "alignment_mode", "alignment_checkpoint", "alignment_checkpoint_sha256",
            "alignment_metrics_path", "alignment_metrics", "imputed_values",
            "final_risk", "risk_level", "risk_matrix", "risk_matrix_path",
        }
        self.assertTrue(api_fields.issubset(from_file))
        for result in (from_file, from_records):
            self.assertEqual(result["input_record_count"], len(self.records))
            self.assertEqual(result["observed_ratio"], 0.9)
            self.assertEqual(result["alignment_checkpoint_sha256"], "test-checkpoint-sha256")
            self.assertEqual(result["final_risk"], 0.2)
            self.assertEqual(result["alignment_metadata"]["grid_alignment"]["reference_interval_seconds"], 2.0)
            self.assertEqual(set(result["outputs"]), {
                "stage1_imputeformer_aligned", "stage1_imputeformer_metrics", "risk_matrix",
            })
            self.assertEqual(result["outputs"]["risk_matrix"], result["risk_matrix_path"])
            self.assertEqual(
                json.loads(Path(result["risk_matrix_path"]).read_text(encoding="utf-8")),
                result["risk_matrix"],
            )

    def test_configured_paths_resolve_from_project_root_and_explicit_sources_override(self):
        with patch.multiple(
            input_paths,
            TIME_SERIES_INPUT_PATH=self.csv.relative_to(ROOT),
            OPTICAL_IMAGE_INPUT_PATH=self.optical.relative_to(ROOT),
            SONAR_IMAGE_INPUT_PATH=self.sonar.relative_to(ROOT),
        ):
            pipeline = self._pipeline(model_bundle=self.bundle)
            configured = pipeline.run()
            overridden = pipeline.run(records=self.records, sonar_xtf=self.xtf)
        self.assertEqual(configured["data_source"], str(self.csv.resolve()))
        self.assertEqual(Path(self.representation_calls[0].optical_image), self.optical.resolve())
        self.assertEqual(Path(self.representation_calls[0].sonar_image), self.sonar.resolve())
        self.assertIsNone(self.representation_calls[1].sonar_image)
        self.assertEqual(Path(self.representation_calls[1].sonar_xtf), self.xtf.resolve())
        self.assertEqual(overridden["input_record_count"], len(self.records))

    def test_output_collision_preserves_existing_files_without_loading_models(self):
        output = self.jobs / "already_used"
        output.mkdir(parents=True)
        marker = output / "existing.txt"
        marker.write_text("preserve this", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._run(self._pipeline(), records=self.records, output_dir=output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve this")
        self.assertEqual(list(output.iterdir()), [marker])
        self.load.assert_not_called()
        self.align.assert_not_called()
        self.represent.assert_not_called()

    def test_invalid_records_fail_before_loading_models_or_creating_jobs(self):
        cases = {
            "empty": [],
            "missing_sensor": [r for r in self.records if r["sensor_id"] != OCEAN_SENSOR_IDS[-1]],
            "unexpected_sensor": [*self.records, {**self.records[0], "sensor_id": "unknown"}],
            "all_missing": [{**r, "value": None} for r in self.records],
            "bad_timestamp": [{**self.records[0], "timestamp": "not-a-timestamp"}, *self.records[1:]],
            "non_numeric": [{**self.records[0], "value": "unreadable"}, *self.records[1:]],
            "not_a_record": [42, *self.records[1:]],
        }
        pipeline = self._pipeline()
        for name, records in cases.items():
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError)):
                self._run(pipeline, records=records, output_dir=self.jobs / name)
            self.assertFalse((self.jobs / name).exists())
        self.load.assert_not_called()
        self.align.assert_not_called()
        self.represent.assert_not_called()

    def test_a_temporarily_unavailable_sensor_can_be_declared_with_null_values(self):
        records = [
            {**r, "value": None} if r["sensor_id"] == OCEAN_SENSOR_IDS[-1] else r
            for r in self.records
        ]
        result = self._run(self._pipeline(model_bundle=self.bundle), records=records)
        self.assertEqual(result["sensors"], list(OCEAN_SENSOR_IDS))
        self.align.assert_called_once()

    def test_missing_input_files_fail_without_model_loading_or_output_creation(self):
        missing = self.root / "does_not_exist.png"
        for label, arguments in (
            ("time_series", {"time_series_input": self.root / "absent.csv"}),
            ("optical", {"records": self.records, "optical_image": missing}),
            ("sonar", {"records": self.records, "sonar_image": missing}),
        ):
            output = self.jobs / label
            with self.subTest(source=label), self.assertRaises((FileNotFoundError, ValueError)):
                self._run(self._pipeline(), output_dir=output, **arguments)
            self.assertFalse(output.exists())
        self.load.assert_not_called()
        self.align.assert_not_called()

    def test_two_explicit_record_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            self._run(self._pipeline(), records=self.records, time_series_input=self.csv)
        self.load.assert_not_called()
        self.align.assert_not_called()
        self.assertFalse(self.jobs.exists())

    def test_non_tiff_sonar_images_fail_before_loading_models_or_creating_jobs(self):
        pipeline = self._pipeline()
        for source in (self.optical, self.xtf):
            output = self.jobs / source.suffix[1:]
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "TIFF|tif"):
                self._run(
                    pipeline, records=self.records, sonar_image=source, output_dir=output,
                )
            self.assertFalse(output.exists())
        self.load.assert_not_called()
        self.align.assert_not_called()
        self.represent.assert_not_called()

    def test_representation_failure_propagates_and_releases_the_shared_lock(self):
        failure = RuntimeError("feature extraction failed")
        self.represent.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            self._run(self._pipeline(), records=self.records)
        self.assertIs(raised.exception, failure)
        self.load.assert_called_once()
        self.align.assert_called_once()
        self.represent.assert_called_once()
        self.assertEqual(self.lock.depth, 0)


if __name__ == "__main__":
    unittest.main()
