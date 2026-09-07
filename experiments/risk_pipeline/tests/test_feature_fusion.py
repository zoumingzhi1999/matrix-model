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
from risk_pipeline.data_io import load_grayscale_image  # noqa: E402
from train_feature_fusion_transformer import (  # noqa: E402
    FeatureSplit,
    construct_pair_indices,
    evaluate_reconstruction,
    zero_reconstruction_baseline,
)


class FeatureFusionTests(unittest.TestCase):
    def test_grayscale_loader_preserves_16_bit_png_and_tiff_intensities(self):
        pixels = np.asarray([[1000, 2000], [3000, 4000]], dtype="uint16")
        with tempfile.TemporaryDirectory() as temp_dir:
            for suffix in (".png", ".tif"):
                with self.subTest(suffix=suffix):
                    path = Path(temp_dir) / f"sonar{suffix}"
                    Image.fromarray(pixels).save(path)
                    loaded = load_grayscale_image(path)
                    self.assertEqual(loaded.dtype, np.dtype("float32"))
                    self.assertEqual(loaded.shape, pixels.shape)
                    np.testing.assert_allclose(loaded, pixels.astype("float32") / 65535.0)
                    self.assertGreater(float(loaded.std()), 0.0)
                    Image.fromarray(np.full((2, 2), 1000, dtype="uint16")).save(path)
                    np.testing.assert_allclose(load_grayscale_image(path), 1000 / 65535.0)

    def test_grayscale_loader_has_explicit_float_and_signed_integer_ranges(self):
        cases = (
            (np.asarray([[0.1, 0.3], [0.7, 0.9]], dtype="float32"),
             np.asarray([[0.1, 0.3], [0.7, 0.9]], dtype="float32")),
            (np.asarray([[-1.0, 0.0], [1.0, 3.0]], dtype="float32"),
             np.asarray([[0.0, 0.25], [0.5, 1.0]], dtype="float32")),
            (np.full((2, 2), 1000.0, dtype="float32"), np.ones((2, 2))),
            (np.full((2, 2), -10.0, dtype="float32"), np.zeros((2, 2))),
            (np.asarray([[-1, 0], [1000, 2000]], dtype="int32"),
             np.asarray([[0, 0], [1000, 2000]], dtype="float32") / np.iinfo("int32").max),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sonar.tif"
            for pixels, expected in cases:
                with self.subTest(dtype=pixels.dtype, minimum=float(pixels.min())):
                    Image.fromarray(pixels).save(path)
                    np.testing.assert_allclose(load_grayscale_image(path), expected)
            Image.fromarray(np.full((2, 2), np.nan, dtype="float32")).save(path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_grayscale_image(path)

    def _checkpoint(self, path, normalization_shift=0.0, d_model=32):
        torch.manual_seed(7)
        config = {
            "input_dims": feature_input_dims(),
            "d_model": d_model,
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

    def test_end_to_end_retains_only_requested_artifacts_and_logs_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "fusion.pt"
            self._checkpoint(checkpoint_path, d_model=128)
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
            ), self.assertLogs("f-model.pipeline", level="INFO") as logs:
                result = run_pipeline(args)

            risk_path = Path(result["risk_json_path"])
            payload = json.loads(risk_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload["modalities"]),
                {"time_series", "optical_image", "sonar_image"},
            )
            self.assertIn("total_risk", payload)
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["embedding_dimension"], 128)
            self.assertEqual(payload["second_stage_feature_stage"], "stage3_transformer_unified_embedding")
            for modality, path in result["artifacts"]["second_stage_features"].items():
                saved = pd.read_csv(path)
                expected_columns = [f"z_{index:03d}" for index in range(128)]
                self.assertEqual(saved.filter(regex=r"^z_\d+$").columns.tolist(), expected_columns)
                self.assertTrue(np.isfinite(saved[expected_columns].to_numpy()).all())
                self.assertFalse(set(FUSION_FEATURE_COLUMNS[modality]) & set(saved.columns))
                source_rows = pd.read_csv(result["artifacts"]["first_stage_features"][modality])
                self.assertEqual(len(saved), len(source_rows))
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
            self.assertNotIn("summary_path", result)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"aligned", "features", "risk"},
            )
            feature_paths = sorted((output_dir / "features").glob("*.csv"))
            self.assertEqual(len(feature_paths), 6)
            self.assertEqual(len(list(output_dir.rglob("*.*"))), 8)
            for stage, key in (
                ("First-stage features", "first_stage_features"),
                ("Second-stage features", "second_stage_features"),
            ):
                for modality, path in result["artifacts"][key].items():
                    matching = [
                        record.getMessage() for record in logs.records
                        if record.getMessage().startswith(
                            f"{stage} output modality={modality} "
                        )
                    ]
                    self.assertEqual(len(matching), 1)
                    csv_text = Path(path).read_text(encoding="utf-8-sig")
                    self.assertEqual(matching[0].split("\n", 1)[1], csv_text)
            for item in payload["modalities"].values():
                self.assertEqual(
                    set(item["feature_value_summary"]), {"min", "mean", "max"}
                )
                self.assertNotIn("selected_vector_artifact", item)
                self.assertTrue((output_dir / item["source"]).is_file())
                self.assertEqual(item["source_representation"], "transformer_fused_cls")
                self.assertEqual(item["risk_cls_token_index"], 0)
                self.assertEqual(
                    item["risk_feature_derivation"],
                    "unified_decoder_then_inverse_normalization_then_selected_feature",
                )
            original_files = {
                path.relative_to(output_dir): path.read_bytes()
                for path in output_dir.rglob("*") if path.is_file()
            }
            with patch(
                "risk_pipeline.run_encapsulated_pipeline.load_xtf_sonar_image",
                return_value=(sonar_array, sonar_metadata),
            ):
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    run_pipeline(args)
            self.assertEqual(original_files, {
                path.relative_to(output_dir): path.read_bytes()
                for path in output_dir.rglob("*") if path.is_file()
            })

            for suffix in (".tif", ".tiff"):
                with self.subTest(sonar_image_suffix=suffix):
                    direct_sonar = root / f"sonar{suffix}"
                    Image.fromarray(
                        np.tile(np.arange(144, dtype="uint8"), (80, 1))
                    ).save(direct_sonar)
                    image_output = root / f"output_{suffix[1:]}"
                    image_args = SimpleNamespace(**{
                        **vars(args),
                        "sonar_image": str(direct_sonar),
                        "sonar_xtf": None,
                        "output_dir": str(image_output),
                    })
                    with patch(
                        "risk_pipeline.run_encapsulated_pipeline.load_xtf_sonar_image"
                    ) as decoder, self.assertLogs("f-model.pipeline", level="INFO"):
                        image_result = run_pipeline(image_args)
                    decoder.assert_not_called()
                    self.assertEqual(
                        {path.name for path in image_output.iterdir()},
                        {"aligned", "features", "risk"},
                    )
                    self.assertEqual(len(list(image_output.rglob("*.*"))), 8)
                    self.assertEqual(len(list((image_output / "features").glob("*.csv"))), 6)
                    for key in ("first_stage_features", "second_stage_features"):
                        sonar_features = pd.read_csv(
                            image_result["artifacts"][key]["sonar_image"]
                        )
                        self.assertEqual(sonar_features["source"].unique().tolist(), [direct_sonar.name])
                        self.assertEqual(sonar_features["input_width"].unique().tolist(), [144])
                        self.assertEqual(sonar_features["input_height"].unique().tolist(), [80])
                    image_payload = json.loads(
                        Path(image_result["risk_json_path"]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(set(image_payload["modalities"]), set(FUSION_MODALITIES))
                    self.assertTrue(np.isfinite(image_payload["total_risk"]))
            conflicting_args = SimpleNamespace(**{
                **vars(image_args), "sonar_xtf": str(sonar_path),
            })
            with self.assertRaisesRegex(ValueError, "[Oo]nly one|both|[Ee]ither"):
                run_pipeline(conflicting_args)


if __name__ == "__main__":
    unittest.main()
