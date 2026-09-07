import hashlib
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
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.full_pipeline import run_streaming_pipeline as streaming
from f_model import RiskPipeline
from multisource_unified import (
    FUSION_FEATURE_COLUMNS,
    FUSION_MODALITIES,
    FeatureFusionTransformer,
    feature_input_dims,
    load_feature_fusion_checkpoint,
)
from risk_pipeline import run_encapsulated_pipeline as representation
from risk_pipeline.artifacts import artifact_path, write_csv_once
from risk_pipeline.sensor_schema import OCEAN_SENSOR_IDS


class DirectoryBatchPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="directory_batch_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.optical = self.root / "optical"
        self.sonar = self.root / "sonar"
        for directory in (self.optical, self.sonar):
            directory.mkdir()
            (directory / "notes.txt").write_text("Ignore this file.", encoding="utf-8")
            (directory / "nested").mkdir()
            Image.fromarray(np.zeros((64, 64), dtype="uint8")).save(
                directory / "nested" / "ignored.png"
            )

        self.optical_paths = [self.optical / "a.png", self.optical / "b.PNG"]
        self.sonar_paths = [self.sonar / "a.tif", self.sonar / "b.TIFF"]
        # Different dimensions make a missed source or lost tile detectable.
        for path, shape, intensity in (
            (self.optical_paths[1], (128, 192), 160),
            (self.optical_paths[0], (64, 64), 64),
            (self.sonar_paths[0], (128, 128), 96),
        ):
            pixels = np.full(shape, intensity, dtype="uint8")
            pixels[:, ::2] = intensity // 2
            Image.fromarray(pixels).save(path)
        self.sonar_array = np.tile(
            np.linspace(0.1, 0.9, 192, dtype="float32"), (192, 1)
        )
        Image.fromarray(self.sonar_array).save(self.sonar_paths[1])
        Image.fromarray(np.zeros((64, 64), dtype="uint8")).save(
            self.sonar / "ignored.png"
        )
        (self.sonar / "ignored.xtf").write_bytes(b"ignored-xtf-input")

        self.aligned = pd.DataFrame(
            {
                sensor: np.linspace(float(index), float(index + 1), 16)
                for index, sensor in enumerate(OCEAN_SENSOR_IDS)
            },
            index=pd.date_range("2026-01-01", periods=16, freq="2s", tz="UTC"),
        )
        self.records = [
            {"timestamp": timestamp.isoformat(), "sensor_id": sensor, "value": value}
            for timestamp, row in self.aligned.iterrows()
            for sensor, value in row.items()
        ]
        self.bundle, self.risk_config = self._make_model_bundle()

    def _make_model_bundle(self):
        torch.manual_seed(71)
        config = {
            "input_dims": feature_input_dims(),
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 1,
            "ff_dim": 64,
            "dropout": 0.0,
        }
        model = FeatureFusionTransformer(**config)
        checkpoint = {
            "schema_version": 1,
            "model_class": "FeatureFusionTransformer",
            "has_risk_head": False,
            "model_config": config,
            "feature_columns": {
                modality: list(FUSION_FEATURE_COLUMNS[modality])
                for modality in FUSION_MODALITIES
            },
            "normalization": {
                modality: {
                    "columns": list(FUSION_FEATURE_COLUMNS[modality]),
                    "mean": [0.0] * len(FUSION_FEATURE_COLUMNS[modality]),
                    "scale": [2.0] * len(FUSION_FEATURE_COLUMNS[modality]),
                }
                for modality in FUSION_MODALITIES
            },
            "state_dict": model.state_dict(),
        }
        path = self.root / "fusion.pt"
        torch.save(checkpoint, path)
        model, checkpoint = load_feature_fusion_checkpoint(path, map_location="cpu")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        risk_config = self.root / "risk_config.json"
        risk_config.write_text(json.dumps({
            "calibration_feature_stage": "stage3_transformer_unified_reconstruction",
            "fusion_checkpoint_sha256": digest,
            "modalities": {
                modality: {
                    "selected_feature": feature,
                    "fixed_weight": weight,
                    "scaler_mean": 0.0,
                    "scaler_scale": 1.0,
                    "logistic_coefficient": 1.0,
                    "logistic_intercept": 0.0,
                }
                for modality, feature, weight in (
                    ("time_series", "std", 0.20),
                    ("optical_image", "edge_density", 0.40),
                    ("sonar_image", "edge_density", 0.40),
                )
            },
        }), encoding="utf-8")
        return SimpleNamespace(
            sensor_checkpoint_path=self.root / "mock_sensor_checkpoint.pt",
            fusion_checkpoint_path=path,
            fusion_model=model,
            fusion_checkpoint=checkpoint,
            device="cpu",
            inference_lock=threading.RLock(),
        ), risk_config

    def _alignment_stage(self, records, args, output_dir, model_bundle=None):
        self.assertIs(model_bundle, self.bundle)
        self.assertEqual(records, self.records)
        aligned_path = artifact_path(
            output_dir, "aligned", "stage1_imputeformer_aligned_output", ".csv",
            args.artifact_timestamp,
        )
        metrics_path = artifact_path(
            output_dir, "aligned", "stage1_imputeformer_evaluation_metrics", ".csv",
            args.artifact_timestamp,
        )
        write_csv_once(self.aligned, aligned_path, index=True, index_label="timestamp")
        metrics = {"sensor_id": "overall", "mae": 0.1, "rmse": 0.2}
        write_csv_once(pd.DataFrame([metrics]), metrics_path)
        return aligned_path, list(OCEAN_SENSOR_IDS), np.ones(self.aligned.shape, dtype=bool), {
            "mode": "pretrained_imputeformer",
            "checkpoint": str(self.bundle.sensor_checkpoint_path),
            "checkpoint_sha256": "test-sensor-checkpoint",
            "imputed_values": 0,
            "evaluation_metrics_csv": str(metrics_path),
            "evaluation_metrics_overall": metrics,
            "grid_alignment": {"grid_rows": len(self.aligned)},
        }

    def test_directory_sources_are_aggregated_once_into_nine_retained_artifacts(self):
        pipeline = RiskPipeline(
            model_bundle=self.bundle, risk_config=self.risk_config,
            output_root=self.root / "jobs", device="cpu",
        )
        with (
            patch.object(
                streaming, "run_imputeformer_alignment_from_records",
                side_effect=self._alignment_stage,
            ) as align,
            patch.object(
                representation, "load_grayscale_image",
                wraps=representation.load_grayscale_image,
            ) as load_image,
            patch.object(
                representation, "load_sonar_tiff_image",
                wraps=representation.load_sonar_tiff_image,
            ) as load_sonar,
            patch.object(
                representation, "extract_image_tile_first_stage_features",
                wraps=representation.extract_image_tile_first_stage_features,
            ) as extract_image,
            self.assertLogs("f-model.pipeline", level="INFO") as logs,
        ):
            result = pipeline.run(
                records=self.records, optical_image=self.optical, sonar_image=self.sonar,
            )

        align.assert_called_once()
        self.assertEqual(
            [Path(call.args[0]) for call in load_image.call_args_list],
            self.optical_paths,
        )
        self.assertEqual(
            [Path(call.args[0]) for call in load_sonar.call_args_list],
            self.sonar_paths,
        )
        self.assertEqual(
            [(call.args[0], call.kwargs["modality"]) for call in extract_image.call_args_list],
            [(path.name, "optical_image") for path in self.optical_paths]
            + [(path.name, "sonar_image") for path in self.sonar_paths],
        )

        artifacts = result["artifacts"]
        expected_counts = {"time_series": 5, "optical_image": 3, "sonar_image": 5}
        expected_source_tiles = {
            "optical_image": [("a.png", 0), ("b.PNG", 0), ("b.PNG", 1)],
            "sonar_image": [("a.tif", 0), *[("b.TIFF", tile) for tile in range(4)]],
        }
        embedding_columns = [f"z_{index:03d}" for index in range(128)]
        for modality in FUSION_MODALITIES:
            first = pd.read_csv(artifacts["first_stage_features"][modality])
            second = pd.read_csv(artifacts["second_stage_features"][modality])
            self.assertEqual(len(first), expected_counts[modality])
            self.assertEqual(len(second), len(first))
            self.assertEqual(second.filter(regex=r"^z_\d+$").columns.tolist(), embedding_columns)
            self.assertTrue(np.isfinite(second[embedding_columns].to_numpy()).all())
            metadata = [column for column in first if column not in FUSION_FEATURE_COLUMNS[modality]]
            pd.testing.assert_frame_equal(first[metadata], second[metadata])
            if modality in expected_source_tiles:
                self.assertEqual(
                    list(first[["source", "tile_index"]].itertuples(index=False, name=None)),
                    expected_source_tiles[modality],
                )
            self.assertEqual(
                result["risk_matrix"]["modalities"][modality]["vector_length"],
                expected_counts[modality],
            )

        job_dir = Path(result["job_dir"])
        files = [path for path in job_dir.rglob("*") if path.is_file()]
        self.assertEqual(len(files), 9)
        self.assertEqual(len(list(job_dir.rglob("*.csv"))), 8)
        self.assertEqual(list(job_dir.rglob("*.json")), [Path(result["risk_matrix_path"])])
        self.assertEqual(
            json.loads(Path(result["risk_matrix_path"]).read_text(encoding="utf-8")),
            result["risk_matrix"],
        )
        self.assertAlmostEqual(
            result["final_risk"],
            sum(item["risk"] * item["fixed_weight"] for item in result["risk_matrix"]["modalities"].values()),
        )
        messages = [record.getMessage() for record in logs.records]
        for stage, key in (
            ("First-stage features", "first_stage_features"),
            ("Second-stage features", "second_stage_features"),
        ):
            for modality, path in artifacts[key].items():
                matching = [message for message in messages if message.startswith(f"{stage} output modality={modality} ")]
                self.assertEqual(len(matching), 1)
                self.assertIn(Path(path).read_text(encoding="utf-8-sig"), matching[0])


if __name__ == "__main__":
    unittest.main()
