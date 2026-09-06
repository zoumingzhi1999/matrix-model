import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image


EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS))

from multisource_unified import (  # noqa: E402
    FUSION_FEATURE_COLUMNS,
    FUSION_MODALITIES,
    FeatureFusionTransformer,
    feature_input_dims,
    load_feature_fusion_checkpoint,
)
from risk_pipeline.run_encapsulated_pipeline import (  # noqa: E402
    run as run_pipeline,
    run_transformer_stage,
    validate_risk_checkpoint_binding,
)
from train_feature_fusion_transformer import (  # noqa: E402
    FeatureSplit,
    construct_pair_indices,
    evaluate_reconstruction,
    zero_reconstruction_baseline,
)


class FeatureFusionTests(unittest.TestCase):
    def _checkpoint(self, path, normalization_shift=0.0):
        torch.manual_seed(7)
        config = {
            "input_dims": feature_input_dims(),
            "d_model": 32,
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
                    "mean": [
                        normalization_shift * (index + 1)
                        for index in range(len(FUSION_FEATURE_COLUMNS[modality]))
                    ],
                    "scale": [2.0] * len(FUSION_FEATURE_COLUMNS[modality]),
                }
                for modality in FUSION_MODALITIES
            },
            "state_dict": model.state_dict(),
        }
        torch.save(checkpoint, path)
        return checkpoint

    @staticmethod
    def _feature_frames():
        time = pd.DataFrame({
            "sensor_id": ["sensor_a", "sensor_b"],
            **{
                column: [float(index + 1), float(index + 2)]
                for index, column in enumerate(FUSION_FEATURE_COLUMNS["time_series"])
            },
        })
        optical = pd.DataFrame({
            column: [float(index), float(index + 2)]
            for index, column in enumerate(FUSION_FEATURE_COLUMNS["optical_image"])
        })
        sonar = pd.DataFrame({
            column: [float(index + 1), float(index + 3)]
            for index, column in enumerate(FUSION_FEATURE_COLUMNS["sonar_image"])
        })
        return time, optical, sonar

    def test_checkpoint_has_no_risk_head_and_loads_strictly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fusion.pt"
            self._checkpoint(path)
            model, checkpoint = load_feature_fusion_checkpoint(path)

        self.assertFalse(checkpoint["has_risk_head"])
        self.assertFalse(hasattr(model, "risk_head"))
        self.assertFalse(any("risk" in name for name in model.state_dict()))

    def test_missing_checkpoint_fails_instead_of_using_random_weights(self):
        with self.assertRaisesRegex(FileNotFoundError, "Train"):
            load_feature_fusion_checkpoint("does-not-exist.pt")

    def test_checkpoint_feature_schema_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fusion.pt"
            checkpoint = self._checkpoint(path)
            checkpoint["feature_columns"]["sonar_image"] = ["wrong"]
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "feature columns"):
                load_feature_fusion_checkpoint(path)

    def test_risk_calibration_is_bound_to_exact_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fusion.pt"
            path.write_bytes(b"trained-checkpoint")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_risk_checkpoint_binding(
                {"fusion_checkpoint_sha256": digest}, path
            )
            with self.assertRaisesRegex(ValueError, "different fusion checkpoint"):
                validate_risk_checkpoint_binding(
                    {"fusion_checkpoint_sha256": "0" * 64}, path
                )
            with self.assertRaisesRegex(ValueError, "not bound"):
                validate_risk_checkpoint_binding({}, path)

    def test_pipeline_fusion_is_deterministic_and_per_sensor(self):
        frames = self._feature_frames()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fusion.pt"
            self._checkpoint(path)
            args = SimpleNamespace(fusion_checkpoint=str(path))
            first, first_reconstructed, first_ranges, first_concat, first_positions = run_transformer_stage(
                *frames, args
            )
            second, second_reconstructed, second_ranges, second_concat, second_positions = run_transformer_stage(
                *frames, args
            )

        self.assertEqual(first.shape, (2, 33))
        self.assertEqual(first["sensor_id"].tolist(), ["sensor_a", "sensor_b"])
        self.assertEqual(len(first_ranges), 6)
        self.assertEqual(first_concat.shape, (2, 13))
        self.assertEqual(len(first_positions), 24)
        self.assertEqual(
            first_positions["concatenated_index"].drop_duplicates().tolist(),
            list(range(12)),
        )
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first_ranges, second_ranges)
        pd.testing.assert_frame_equal(first_concat, second_concat)
        pd.testing.assert_frame_equal(first_positions, second_positions)
        for modality, source in zip(FUSION_MODALITIES, frames):
            self.assertEqual(len(first_reconstructed[modality]), len(source))
            self.assertTrue(
                set(FUSION_FEATURE_COLUMNS[modality]).issubset(
                    first_reconstructed[modality].columns
                )
            )
            pd.testing.assert_frame_equal(
                first_reconstructed[modality], second_reconstructed[modality]
            )

    def test_pipeline_uses_checkpoint_training_normalization(self):
        frames = self._feature_frames()
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.pt"
            shifted_path = Path(temp_dir) / "shifted.pt"
            self._checkpoint(first_path, normalization_shift=0.0)
            self._checkpoint(shifted_path, normalization_shift=10.0)
            first, first_reconstructed, _, _, _ = run_transformer_stage(
                *frames, SimpleNamespace(fusion_checkpoint=str(first_path)))
            shifted, shifted_reconstructed, _, _, _ = run_transformer_stage(
                *frames, SimpleNamespace(fusion_checkpoint=str(shifted_path)))

        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                first.filter(like="z_").to_numpy(),
                shifted.filter(like="z_").to_numpy(),
            )
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                first_reconstructed["time_series"]["std"].to_numpy(),
                shifted_reconstructed["time_series"]["std"].to_numpy(),
            )

    def test_constructed_pairs_are_split_local_reproducible_and_complete(self):
        data = {}
        sizes = {
            "time_series": (7, 3),
            "optical_image": (2, 4),
            "sonar_image": (3, 2),
        }
        for modality, (negative, positive) in sizes.items():
            labels = np.asarray([0] * negative + [1] * positive, dtype="int64")
            values = np.arange(
                len(labels) * len(FUSION_FEATURE_COLUMNS[modality]),
                dtype="float32",
            ).reshape(len(labels), -1)
            data[modality] = FeatureSplit(values, labels)

        first, audit = construct_pair_indices(data, seed=123, balance_labels=False)
        second, _ = construct_pair_indices(data, seed=123, balance_labels=False)
        for modality in FUSION_MODALITIES:
            np.testing.assert_array_equal(first[modality], second[modality])
            self.assertEqual(set(first[modality].tolist()), set(range(len(data[modality].labels))))
        tuple_labels = data["time_series"].labels[first["time_series"]]
        for modality in FUSION_MODALITIES[1:]:
            np.testing.assert_array_equal(
                tuple_labels,
                data[modality].labels[first[modality]],
            )
        self.assertTrue(audit["all_source_samples_used"])

    def test_independent_reconstruction_metrics_include_four_error_measures(self):
        data = {}
        pair_indices = {}
        for modality in FUSION_MODALITIES:
            width = len(FUSION_FEATURE_COLUMNS[modality])
            values = np.linspace(0.1, 1.0, 4 * width, dtype="float32").reshape(
                4, width
            )
            data[modality] = FeatureSplit(
                values=values,
                labels=np.asarray([0, 0, 1, 1], dtype="int64"),
            )
            pair_indices[modality] = np.arange(4, dtype="int64")
        model = FeatureFusionTransformer(
            feature_input_dims(),
            d_model=32,
            num_heads=4,
            num_layers=1,
            ff_dim=64,
            dropout=0.0,
        )
        args = SimpleNamespace(
            evaluation_batch_size=2,
            feature_mask_probability=0.30,
            modality_drop_probability=0.25,
            noise_std=0.05,
        )

        metrics = evaluate_reconstruction(
            model,
            data,
            pair_indices,
            args,
            torch.device("cpu"),
            corrupt=False,
            seed=42,
        )
        baseline = zero_reconstruction_baseline(data, pair_indices)

        for prefix in ("overall_unified", "overall_token"):
            for name in ("mae", "mse", "rmse", "mre"):
                self.assertIn(f"{prefix}_{name}", metrics)
        for name in ("mae", "mse", "rmse", "mre"):
            self.assertIn(f"overall_unified_{name}", baseline)

    def test_end_to_end_writes_one_json_risk_result_and_no_risk_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "fusion.pt"
            self._checkpoint(checkpoint_path)
            checkpoint_hash = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()

            aligned_path = root / "aligned.csv"
            pd.DataFrame({
                "timestamp": pd.date_range(
                    "2026-01-01", periods=16, freq="min", tz="UTC"
                ),
                "sensor_a": np.linspace(0.0, 1.0, 16),
                "sensor_b": np.linspace(1.0, 0.0, 16),
            }).to_csv(aligned_path, index=False)
            optical_path = root / "optical.png"
            sonar_path = root / "sonar.xtf"
            Image.fromarray(np.full((128, 128), 96, dtype="uint8")).save(
                optical_path
            )
            sonar_path.write_bytes(b"raw-xtf-test-placeholder")

            config_path = root / "risk_config.json"
            config_path.write_text(json.dumps({
                "calibration_feature_stage": (
                    "stage3_transformer_unified_reconstruction"
                ),
                "fusion_checkpoint_sha256": checkpoint_hash,
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
            output_dir = root / "output"
            args = SimpleNamespace(
                imputeformer_output=str(aligned_path),
                output_dir=str(output_dir),
                optical_image=str(optical_path),
                sonar_xtf=str(sonar_path),
                risk_config=str(config_path),
                fusion_checkpoint=str(checkpoint_path),
                artifact_timestamp="20260101_000000_000001",
            )
            sonar_array = np.full((128, 128), 0.5, dtype="float32")
            sonar_metadata = {
                "format": "XTF",
                "sonar_ping_count": 128,
                "channel_count": 2,
                "output_height": 128,
                "output_width": 128,
                "resize_applied": False,
            }
            with patch(
                "risk_pipeline.run_encapsulated_pipeline.load_xtf_sonar_image",
                return_value=(sonar_array, sonar_metadata),
            ):
                result = run_pipeline(args)

            risk_path = Path(result["risk_json_path"])
            payload = json.loads(risk_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload["modalities"]),
                {"time_series", "optical_image", "sonar_image"},
            )
            self.assertIn("total_risk", payload)
            self.assertEqual(
                payload["fusion_checkpoint_sha256"], checkpoint_hash
            )
            weighted_total = sum(
                item["risk"] * item["fixed_weight"]
                for item in payload["modalities"].values()
            )
            self.assertAlmostEqual(payload["total_risk"], weighted_total)
            self.assertFalse(list((output_dir / "risk").glob("*.csv")))
            self.assertFalse((output_dir / "modality_risks").exists())
            self.assertEqual(list(output_dir.rglob("*.json")), [risk_path])
            self.assertEqual(Path(result["summary_path"]).suffix, ".csv")
            self.assertTrue(list((output_dir / "decoded").glob(
                "stage1_xtf_decode_metadata_*.csv"
            )))
            for item in payload["modalities"].values():
                self.assertEqual(
                    set(item["feature_value_summary"]), {"min", "mean", "max"}
                )
                self.assertTrue(
                    (output_dir / item["selected_vector_artifact"]).is_file()
                )
            self.assertTrue(list((output_dir / "features").glob(
                "stage3_concatenated_feature_positions_*.csv"
            )))
            with patch(
                "risk_pipeline.run_encapsulated_pipeline.load_xtf_sonar_image",
                return_value=(sonar_array, sonar_metadata),
            ):
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    run_pipeline(args)


if __name__ == "__main__":
    unittest.main()
