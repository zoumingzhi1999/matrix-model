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
    _build_model,
    align_with_pretrained_imputeformer,
    calculate_alignment_metrics,
    deterministic_observed_holdout,
    normalize_with_checkpoint,
    temporal_features,
)
from risk_pipeline.time_series_features import (  # noqa: E402
    TIME_SERIES_FEATURE_COLUMNS,
    extract_first_stage_features,
)
from full_pipeline.run_streaming_pipeline import read_csv_records  # noqa: E402


class _ConstantPredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.seen_lengths = []
        self.seen_masks = []

    def forward(self, values, temporal, mask):
        self.seen_lengths.append(values.shape[1])
        self.seen_masks.append(mask.cpu().numpy())
        return torch.full_like(values, 7.0) + self.anchor


class SensorTimeSeriesTests(unittest.TestCase):
    def _inference_checkpoint(self, variable_length=True):
        return {
            "model_type": "ImputeFormerModel",
            "entity_type": "sensor",
            "uses_site_entity": False,
            "uses_coordinates": False,
            "sensor_ids": list(EXPECTED_SENSOR_IDS),
            "model_config": {
                "window": 64, "inference_stride": 32,
                "variable_length": variable_length,
                "use_interpolation_residual": True,
                "num_nodes": 5, "input_dim": 6, "output_dim": 1,
                "input_embedding_dim": 4, "learnable_embedding_dim": 4,
                "feed_forward_dim": 16, "num_temporal_heads": 2,
                "num_layers": 1, "dim_proj": 2, "dropout": 0.0,
            },
            "normalization": {"mean": np.zeros(5), "std": np.ones(5)},
            "temporal_encoding": "bounded_log_interval_v1",
            "temporal_feature_names": [
                "second_of_day_sin", "second_of_day_cos", "day_of_year_sin",
                "day_of_year_cos", "log_sampling_interval",
            ],
            "supported_sampling_interval_seconds": {"minimum": 0.0, "maximum": None},
        }

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

    def test_reads_sparse_wide_sensor_csv_without_filling_missing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "sparse_sensors.csv"
            frame = pd.DataFrame({
                "timestamp": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ],
                **{
                    sensor_id: [float(index), np.nan]
                    for index, sensor_id in enumerate(EXPECTED_SENSOR_IDS)
                },
            })
            frame.loc[1, "water_depth_m"] = 8.1
            frame.to_csv(csv_path, index=False)

            records = read_csv_records(csv_path)

        self.assertEqual(len(records), 2 * len(EXPECTED_SENSOR_IDS))
        depth_records = [
            record for record in records
            if record["sensor_id"] == "water_depth_m"
        ]
        self.assertEqual([record["value"] for record in depth_records], [1.0, 8.1])
        temperature_records = [
            record for record in records
            if record["sensor_id"] == "water_temperature_c"
        ]
        self.assertTrue(pd.isna(temperature_records[1]["value"]))

    def test_production_bundle_loads_each_trained_model_once(self):
        sensor_model = torch.nn.Linear(1, 1)
        fusion_model = torch.nn.Linear(1, 1)
        with patch(
            "risk_pipeline.model_bundle.load_sensor_imputeformer_checkpoint",
            return_value=(sensor_model, {"kind": "sensor", "model_config": {}}),
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

    def test_reference_uses_most_samples_and_grid_covers_all_sensor_times(self):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        records = [
            {"timestamp": base + pd.Timedelta(seconds=second),
             "sensor_id": sensor, "value": float(second)}
            for sensor, seconds in (("most_samples", (2, 4, 6, 8, 10)),
                                    ("short_fast_burst", (0, 1, 2)),
                                    ("late_sensor", (3, 12)))
            for second in seconds
        ]
        records.append({"timestamp": base + pd.Timedelta(seconds=14),
                        "sensor_id": "late_sensor", "value": None})

        wide, _, _, _, metadata = load_streaming_records(records)

        self.assertEqual(metadata["reference_sensor_id"], "most_samples")
        self.assertEqual(metadata["alignment_frequency_seconds"], 2.0)
        self.assertEqual(metadata["fastest_sensor_ids"], ["short_fast_burst"])
        self.assertEqual(wide.index[0], base)
        self.assertEqual(wide.index[-1], base + pd.Timedelta(seconds=14))
        self.assertEqual(wide.loc[base + pd.Timedelta(seconds=12), "late_sensor"], 12.0)
        self.assertEqual(metadata["merged_observation_count"], 0)
        pd.testing.assert_index_equal(
            wide.index,
            pd.DatetimeIndex(base + pd.to_timedelta([0, 1, 2, 4, 6, 8, 10, 12, 14], unit="s"),
                             name="timestamp").as_unit("ns"),
        )
        self.assertEqual(wide["most_samples"].notna().sum(), 5)

    def test_reference_cadence_handles_jitter_and_missing_reference_samples(self):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        for seconds, expected in (((0, 1, 2, 4, 5), 1.0),
                                  ((0, 1.01, 2.00, 3.02, 4.00), 1.0)):
            with self.subTest(seconds=seconds):
                records = [
                    {"timestamp": base + pd.Timedelta(seconds=second),
                     "sensor_id": "reference", "value": float(index)}
                    for index, second in enumerate(seconds)
                ] + [{"timestamp": base, "sensor_id": "slow", "value": 4.0}]
                wide, _, _, _, metadata = load_streaming_records(records)
                self.assertAlmostEqual(metadata["alignment_frequency_seconds"], expected)
                self.assertEqual(wide["reference"].notna().sum(), len(seconds))

    def test_alignment_accepts_arbitrary_physical_sampling_intervals(self):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        for frequency in ("100ns", "250ms", "7s", "2h", "3D"):
            with self.subTest(frequency=frequency):
                index = pd.date_range(base, periods=9, freq=frequency)
                records = [
                    {"timestamp": timestamp, "sensor_id": sensor, "value": float(step)}
                    for sensor, times in (("reference", index), ("slow", index[::4]))
                    for step, timestamp in enumerate(times)
                ]
                wide, _, _, _, metadata = load_streaming_records(records)
                pd.testing.assert_index_equal(
                    wide.index, index.as_unit("ns").rename("timestamp")
                )
                self.assertAlmostEqual(
                    metadata["alignment_frequency_seconds"],
                    pd.Timedelta(frequency).total_seconds()
                    if frequency != "100ns" else 1e-7,
                )

    def test_single_timestamp_does_not_invent_an_inferred_frequency(self):
        records = [{"timestamp": "2026-01-01T00:00:00Z", "sensor_id": sensor,
                    "value": float(index)}
                   for index, sensor in enumerate(EXPECTED_SENSOR_IDS)]

        wide, _, _, _, metadata = load_streaming_records(records)

        self.assertEqual(len(wide), 1)
        self.assertIsNone(metadata["alignment_frequency_seconds"])
        self.assertEqual(metadata["interval_inference_source"], "undetermined")

    def test_variable_length_inference_covers_short_long_and_final_partial_windows(self):
        checkpoint = self._inference_checkpoint()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "test.pt"
            torch.save(checkpoint, checkpoint_path)
            for length in (1, 2, 63, 64, 100, 1000, 3600, 3617):
                with self.subTest(length=length):
                    model = _ConstantPredictionModel()
                    index = pd.date_range("2026-01-01", periods=length, freq="h", tz="UTC")
                    frame = pd.DataFrame(
                        np.full((length, 5), 2.0), index=index, columns=EXPECTED_SENSOR_IDS
                    )
                    if length > 1:
                        frame.iloc[1::2, :] = np.nan
                    aligned, metadata = align_with_pretrained_imputeformer(
                        frame, checkpoint_path, batch_size=8, preloaded_model=model,
                        preloaded_checkpoint=checkpoint, evaluation_holdout_ratio=0.2,
                    )
                    pd.testing.assert_index_equal(aligned.index, frame.index)
                    np.testing.assert_array_equal(
                        aligned.to_numpy(), np.where(frame.notna(), frame, 7.0)
                    )
                    self.assertEqual(set(model.seen_lengths), {min(length, 64)})
                    self.assertEqual(metadata["input_rows"], length)
                    if length == 1:
                        self.assertIsNone(metadata["sampling_interval_seconds"])
                        self.assertEqual(metadata["evaluation_metrics"][0]["evaluated_values"], 0)
                    else:
                        self.assertEqual(metadata["sampling_interval_seconds"], 3600.0)

    def test_legacy_short_window_padding_is_unobserved_and_trimmed(self):
        checkpoint = self._inference_checkpoint(variable_length=False)
        model = _ConstantPredictionModel()
        frame = pd.DataFrame(
            np.ones((3, 5)), columns=EXPECTED_SENSOR_IDS,
            index=pd.date_range("2026-01-01", periods=3, freq="s", tz="UTC"),
        )
        frame.iloc[-1, -1] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "test.pt"
            torch.save(checkpoint, checkpoint_path)
            aligned, _ = align_with_pretrained_imputeformer(
                frame, checkpoint_path, preloaded_model=model,
                preloaded_checkpoint=checkpoint,
            )
        self.assertEqual(aligned.shape, (3, 5))
        self.assertEqual(aligned.iloc[-1, -1], 7.0)
        self.assertEqual(model.seen_lengths, [64])
        self.assertFalse(model.seen_masks[0][:, 3:].any())

    def test_new_model_checkpoint_loads_and_imputes_native_short_hourly_sequences(self):
        checkpoint = self._inference_checkpoint()
        model = _build_model(checkpoint["model_config"])
        for parameter in model.readout.parameters():
            torch.nn.init.zeros_(parameter)
        checkpoint["model_state_dict"] = model.state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "adaptive.pt"
            torch.save(checkpoint, checkpoint_path)
            for length in (1, 3, 65):
                with self.subTest(length=length):
                    frame = pd.DataFrame(
                        np.broadcast_to(np.arange(length)[:, None], (length, 5)).copy(),
                        columns=EXPECTED_SENSOR_IDS,
                        index=pd.date_range("2026-01-01", periods=length, freq="h", tz="UTC"),
                        dtype="float64",
                    )
                    if length > 1:
                        frame.iloc[1, 0] = np.nan
                    aligned, metadata = align_with_pretrained_imputeformer(
                        frame, checkpoint_path, device="cpu",
                    )
                    self.assertEqual(len(aligned), length)
                    self.assertTrue(np.isfinite(aligned.to_numpy()).all())
                    self.assertEqual(metadata["short_sequence_strategy"], "native_variable_length")
                    if length > 1:
                        self.assertAlmostEqual(aligned.iloc[1, 0], 1.0)

    def test_temporal_interval_encoding_is_bounded_for_all_positive_cadences(self):
        index = pd.date_range("2026-01-01", periods=2, freq="s", tz="UTC")
        values = [temporal_features(index, cadence)[0, -1]
                  for cadence in (1e-9, 1.0, 18.0, 3600.0, 8640000.0)]
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(all(0.0 < value < 1.0 for value in values))
        self.assertTrue((np.diff(values) > 0).all())

    def test_declared_sensor_with_full_outage_is_imputed_and_reported(self):
        records = [{"timestamp": timestamp, "sensor_id": sensor,
                    "value": None if sensor == EXPECTED_SENSOR_IDS[0] else float(step)}
                   for step, timestamp in enumerate(
                       pd.date_range("2026-01-01", periods=3, freq="s", tz="UTC"))
                   for sensor in EXPECTED_SENSOR_IDS]
        wide, _, _, _, grid = load_streaming_records(records)
        checkpoint = self._inference_checkpoint()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "test.pt"
            torch.save(checkpoint, checkpoint_path)
            aligned, metadata = align_with_pretrained_imputeformer(
                wide, checkpoint_path, preloaded_model=_ConstantPredictionModel(),
                preloaded_checkpoint=checkpoint, evaluation_holdout_ratio=0.2,
            )
        self.assertEqual(grid["no_observed_sensor_ids"], [EXPECTED_SENSOR_IDS[0]])
        self.assertEqual(metadata["no_observed_sensor_ids"], [EXPECTED_SENSOR_IDS[0]])
        np.testing.assert_array_equal(aligned[EXPECTED_SENSOR_IDS[0]].to_numpy(), [7.0] * 3)
        missing_metrics = next(row for row in metadata["evaluation_metrics"]
                               if row["sensor_id"] == EXPECTED_SENSOR_IDS[0])
        self.assertEqual(missing_metrics["evaluated_values"], 0)
        self.assertIsNone(missing_metrics["MAE"])

    def test_all_missing_inputs_are_rejected_without_model_loading(self):
        frame = pd.DataFrame(
            np.full((2, 5), np.nan), columns=EXPECTED_SENSOR_IDS,
            index=pd.date_range("2026-01-01", periods=2, freq="s", tz="UTC"),
        )
        with self.assertRaisesRegex(ValueError, "At least one finite"):
            align_with_pretrained_imputeformer(frame, "not-loaded.pt")

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
