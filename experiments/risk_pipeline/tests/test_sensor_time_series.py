import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_pipeline.data_io import SensorBuffer, load_streaming_records  # noqa: E402
from imputeformer.models import ImputeFormerModel  # noqa: E402
from risk_pipeline.model_bundle import ProductionModelBundle  # noqa: E402
from risk_pipeline.sensor_imputation import (  # noqa: E402
    EXPECTED_SENSOR_IDS,
    align_with_pretrained_imputeformer,
    calculate_alignment_metrics,
    deterministic_observed_holdout,
    normalize_with_checkpoint,
)
from risk_pipeline.time_series_features import (  # noqa: E402
    TIME_SERIES_FEATURE_COLUMNS,
    extract_first_stage_features,
)
from full_pipeline.run_streaming_pipeline import read_csv_records  # noqa: E402


class SensorTimeSeriesTests(unittest.TestCase):
    def test_reads_long_format_sensor_csv_and_preserves_missing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "sensors.csv"
            pd.DataFrame([
                {
                    "timestamp": "2026-01-01T00:00:00Z",
                    "sensor_id": "water_temperature_c",
                    "value": 18.5,
                    "unused": "ignored",
                },
                {
                    "timestamp": "2026-01-01T00:00:01Z",
                    "sensor_id": "water_temperature_c",
                    "value": np.nan,
                    "unused": "ignored",
                },
            ]).to_csv(csv_path, index=False)

            records = read_csv_records(csv_path)

        self.assertEqual(
            list(records[0]), ["timestamp", "sensor_id", "value"]
        )
        self.assertEqual(records[0]["value"], 18.5)
        self.assertTrue(pd.isna(records[1]["value"]))

    def test_sensor_csv_rejects_missing_required_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "sensors.csv"
            pd.DataFrame({
                "timestamp": ["2026-01-01T00:00:00Z"],
                "value": [18.5],
            }).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "sensor_id"):
                read_csv_records(csv_path)

    def test_production_bundle_loads_each_trained_model_once(self):
        sensor_model = torch.nn.Linear(1, 1)
        fusion_model = torch.nn.Linear(1, 1)
        with patch(
            "risk_pipeline.model_bundle.load_sensor_imputeformer_checkpoint",
            return_value=(sensor_model, {"kind": "sensor"}),
        ) as load_sensor, patch(
            "risk_pipeline.model_bundle.load_feature_fusion_checkpoint",
            return_value=(fusion_model, {"kind": "fusion"}),
        ) as load_fusion:
            bundle = ProductionModelBundle.load(
                "sensor.pt",
                "fusion.pt",
                device="cpu",
            )

        self.assertIs(bundle.sensor_model, sensor_model)
        self.assertIs(bundle.fusion_model, fusion_model)
        load_sensor.assert_called_once()
        load_fusion.assert_called_once()

    def test_alignment_holdout_is_reproducible_and_never_selects_missing_values(self):
        frame = pd.DataFrame(
            np.arange(50, dtype="float64").reshape(10, 5),
            columns=EXPECTED_SENSOR_IDS,
        )
        frame.iloc[2, 1] = np.nan

        first = deterministic_observed_holdout(frame, ratio=0.2, seed=123)
        second = deterministic_observed_holdout(frame, ratio=0.2, seed=123)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(first[2, 1])
        self.assertTrue((frame.notna().sum().to_numpy() - first.sum(axis=0) >= 1).all())

    def test_alignment_metrics_include_exact_mae_mse_and_rmse(self):
        truth = pd.DataFrame(
            np.zeros((2, 5), dtype="float64"),
            columns=EXPECTED_SENSOR_IDS,
        )
        prediction = truth.copy()
        prediction.iloc[0] = np.arange(1, 6, dtype="float64")
        holdout = np.zeros((2, 5), dtype=bool)
        holdout[0] = True

        rows = calculate_alignment_metrics(
            truth,
            prediction,
            holdout,
            sensor_std=np.ones(5, dtype="float64"),
            holdout_ratio=0.5,
        )

        overall = rows[0]
        self.assertEqual(set(("MAE", "MSE", "RMSE")) - set(overall), set())
        self.assertAlmostEqual(overall["MAE"], 3.0)
        self.assertAlmostEqual(overall["MSE"], 11.0)
        self.assertAlmostEqual(overall["RMSE"], np.sqrt(11.0))

    def test_extracts_exactly_four_selected_features_per_sensor(self):
        aligned = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
            "temperature_001": np.arange(8, dtype="float64"),
            "turbidity_001": np.arange(8, dtype="float64") * 2.0,
            "depth_001": np.arange(8, dtype="float64") * 0.5,
        })

        features, sensor_ids = extract_first_stage_features(aligned)

        self.assertEqual(sensor_ids, [
            "temperature_001", "turbidity_001", "depth_001"])
        self.assertEqual(features.shape, (3, 5))
        self.assertEqual(
            list(features.columns), ["sensor_id", *TIME_SERIES_FEATURE_COLUMNS])
        self.assertEqual(features.select_dtypes(include=[np.number]).shape[1], 4)

    def test_fft_1_amplitude_is_normalized_for_series_length(self):
        amplitudes = []
        for length in (64, 256):
            phase = np.arange(length, dtype="float64") / length
            aligned = pd.DataFrame({
                "timestamp": pd.date_range(
                    "2026-01-01", periods=length, freq="s", tz="UTC"
                ),
                "sensor": np.sin(2.0 * np.pi * phase),
            })
            features, _ = extract_first_stage_features(aligned)
            amplitudes.append(float(features.loc[0, "fft_1"]))

        self.assertAlmostEqual(amplitudes[0], amplitudes[1], places=6)

    def test_streaming_records_build_sensor_node_tensor(self):
        records = []
        for step in range(4):
            timestamp = f"2026-01-01T00:0{step}:00Z"
            records.extend([
                {"timestamp": timestamp, "sensor_id": "sensor_a", "value": step},
                {"timestamp": timestamp, "sensor_id": "sensor_b", "value": step + 10},
            ])

        wide, values, mask, sensors, metadata = load_streaming_records(
            records, "1min", min_window=4)

        self.assertEqual(sensors, ["sensor_a", "sensor_b"])
        self.assertEqual(list(wide.columns), sensors)
        self.assertEqual(values.shape, (4, 2, 1))
        self.assertEqual(mask.shape, values.shape)
        self.assertEqual(metadata["alignment_frequency_seconds"], 60.0)

    def test_buffer_flush_retains_exact_original_records_including_nulls(self):
        records = [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "sensor_id": "sensor_a",
                "value": 1.0,
            },
            {
                "timestamp": "2026-01-01T00:01:00Z",
                "sensor_id": "sensor_a",
                "value": 2.0,
            },
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "sensor_id": "sensor_b",
                "value": 3.0,
            },
            {
                "timestamp": "2026-01-01T00:01:00Z",
                "sensor_id": "sensor_b",
                "value": None,
            },
        ]
        buffer = SensorBuffer(freq="1min", min_window=2)
        buffer.extend(records)

        retained, wide, _, _, _, _ = buffer.flush_with_records()

        self.assertEqual(retained, records)
        self.assertTrue(pd.isna(wide.loc[wide.index[1], "sensor_b"]))
        self.assertEqual(len(buffer), 0)

    def test_auto_alignment_uses_temperature_when_temperature_is_fastest(self):
        records = []
        for second in range(7):
            records.append({
                "timestamp": f"2026-01-01T00:00:0{second}Z",
                "sensor_id": "water_temperature_c",
                "value": 10.0 + second,
            })
        for second in (0, 3, 6):
            records.append({
                "timestamp": f"2026-01-01T00:00:0{second}Z",
                "sensor_id": "turbidity_ntu",
                "value": 2.0 + second,
            })

        wide, _, _, _, metadata = load_streaming_records(records, "auto")

        self.assertEqual(len(wide), 7)
        self.assertEqual(metadata["reference_sensor_id"], "water_temperature_c")
        self.assertEqual(metadata["alignment_frequency_seconds"], 1.0)
        self.assertEqual(wide["turbidity_ntu"].notna().sum(), 3)

    def test_auto_alignment_uses_turbidity_when_turbidity_is_fastest(self):
        records = []
        for second in range(7):
            records.append({
                "timestamp": f"2026-01-01T00:00:0{second}Z",
                "sensor_id": "turbidity_ntu",
                "value": 2.0 + second,
            })
        for second in (0, 3, 6):
            records.append({
                "timestamp": f"2026-01-01T00:00:0{second}Z",
                "sensor_id": "water_temperature_c",
                "value": 10.0 + second,
            })

        wide, _, _, _, metadata = load_streaming_records(records, "auto")

        self.assertEqual(len(wide), 7)
        self.assertEqual(metadata["reference_sensor_id"], "turbidity_ntu")
        self.assertEqual(metadata["alignment_frequency_seconds"], 1.0)
        self.assertEqual(wide["water_temperature_c"].notna().sum(), 3)

    def test_each_sensor_is_normalized_independently(self):
        values = np.asarray([
            [[0.0], [100.0]],
            [[1.0], [200.0]],
            [[2.0], [300.0]],
        ], dtype="float32")

        mean = np.asarray([1.0, 200.0], dtype="float32")
        std = np.asarray([1.0, 100.0], dtype="float32")
        normalized, _ = normalize_with_checkpoint(values, mean, std)

        np.testing.assert_allclose(
            normalized.mean(axis=0), np.zeros((2, 1)), atol=1e-6)

    def test_pretrained_alignment_uses_exact_sensor_schema_and_preserves_values(self):
        config = {
            "window": 4,
            "num_nodes": 5,
            "input_dim": 6,
            "output_dim": 1,
            "input_embedding_dim": 4,
            "learnable_embedding_dim": 4,
            "feed_forward_dim": 16,
            "num_temporal_heads": 2,
            "num_layers": 1,
            "dim_proj": 2,
            "dropout": 0.0,
        }
        model = ImputeFormerModel(
            num_nodes=5,
            input_dim=6,
            output_dim=1,
            input_embedding_dim=4,
            learnable_embedding_dim=4,
            feed_forward_dim=16,
            num_temporal_heads=2,
            num_layers=1,
            dropout=0.0,
            windows=4,
            dim_proj=2,
        )
        checkpoint = {
            "model_type": "ImputeFormerModel",
            "entity_type": "sensor",
            "uses_site_entity": False,
            "uses_coordinates": False,
            "sensor_ids": list(EXPECTED_SENSOR_IDS),
            "model_config": config,
            "normalization": {
                "mean": np.zeros(5, dtype="float32"),
                "std": np.ones(5, dtype="float32"),
            },
            "model_state_dict": model.state_dict(),
            "temporal_feature_names": [
                "second_of_day_sin",
                "second_of_day_cos",
                "day_of_year_sin",
                "day_of_year_cos",
                "log_sampling_interval",
            ],
            "supported_sampling_interval_seconds": {
                "minimum": 1.0,
                "maximum": 1.0,
            },
        }
        index = pd.date_range("2026-01-01", periods=6, freq="1s", tz="UTC")
        source = pd.DataFrame(
            {
                sensor_id: np.arange(6, dtype="float64") + offset
                for offset, sensor_id in enumerate(reversed(EXPECTED_SENSOR_IDS))
            },
            index=index,
        )
        source.loc[index[2], "turbidity_ntu"] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "sensor.pt"
            torch.save(checkpoint, checkpoint_path)
            aligned, metadata = align_with_pretrained_imputeformer(
                source, checkpoint_path, batch_size=2, device="cpu"
            )
        self.assertEqual(tuple(aligned.columns), EXPECTED_SENSOR_IDS)
        self.assertFalse(aligned.isna().any().any())
        observed = source.loc[:, EXPECTED_SENSOR_IDS].notna()
        np.testing.assert_array_equal(
            aligned.to_numpy()[observed.to_numpy()],
            source.loc[:, EXPECTED_SENSOR_IDS].to_numpy()[observed.to_numpy()],
        )
        self.assertEqual(metadata["mode"], "pretrained_inference")
        self.assertEqual(metadata["imputed_values"], 1)

    def test_station_record_contract_is_rejected(self):
        old_record = {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_code": "S001",
            "variable": "temperature",
            "value": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "sensor_id"):
            load_streaming_records([old_record], "1min")


if __name__ == "__main__":
    unittest.main()
