import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_pipeline.data_io import load_aligned_sensors, load_streaming_records  # noqa: E402
from risk_pipeline.temporal_context import (  # noqa: E402
    RAW_TEMPORAL_FEATURE_NAMES,
    adjacent_interval_seconds,
    observation_gap_seconds,
    raw_temporal_features,
    relative_time_seconds,
)


class TemporalContextTests(unittest.TestCase):
    def test_auto_alignment_preserves_irregular_reference_and_explicit_null_timestamps(self):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        records = [
            {"timestamp": base + pd.Timedelta(seconds=second), "sensor_id": "reference", "value": value}
            for second, value in ((0, 1.0), (1, 2.0), (3, 3.0), (3.2, None), (5, 5.0))
        ] + [
            {"timestamp": base + pd.Timedelta(seconds=second), "sensor_id": "slow", "value": value}
            for second, value in ((-0.2, 8.0), (3.1, 10.0), (5.5, 12.0), (6, None))
        ]
        wide, values, mask, _, metadata = load_streaming_records(records)

        np.testing.assert_allclose(
            relative_time_seconds(wide.index), [0, 0.2, 1.2, 3.2, 3.4, 5.2, 5.7, 6.2],
        )
        self.assertEqual(metadata["reference_sensor_id"], "reference")
        self.assertEqual(metadata["grid_policy"], "reference_actual_timestamps_with_outer_coverage")
        self.assertEqual(metadata["alignment_interval_role"], "representative_cadence_estimate")
        self.assertTrue(metadata["irregular_timestamps_preserved"])
        self.assertTrue(np.isnan(wide.loc[base + pd.Timedelta(seconds=3.2), "reference"]))
        self.assertEqual(wide.loc[base + pd.Timedelta(seconds=3), "slow"], 10.0)
        self.assertEqual(values.shape, (8, 2, 1))
        np.testing.assert_array_equal(mask[:, :, 0].astype(bool), wide.notna().to_numpy())

    def test_auto_nearest_mapping_keeps_nanosecond_precision_and_averages_collisions(self):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        records = [
            {"timestamp": base + pd.Timedelta(nanoseconds=nanosecond),
             "sensor_id": "reference", "value": float(step)}
            for step, nanosecond in enumerate((0, 100_000, 300_000, 300_002, 500_000))
        ] + [
            {"timestamp": base + pd.Timedelta(nanoseconds=nanosecond),
             "sensor_id": "slow", "value": value}
            for nanosecond, value in ((300_000, 8.0), (300_001, 12.0), (300_002, 99.0))
        ]
        wide, _, _, _, metadata = load_streaming_records(records)

        self.assertEqual(wide.loc[base + pd.Timedelta(nanoseconds=300_000), "slow"], 10.0)
        self.assertEqual(wide.loc[base + pd.Timedelta(nanoseconds=300_002), "slow"], 99.0)
        self.assertEqual(metadata["merged_observation_count"], 1)
        np.testing.assert_allclose(adjacent_interval_seconds(wide.index), [0, 1e-4, 2e-4, 2e-9, 0.000199998])

    def test_explicit_alignment_frequency_still_constructs_regular_grid(self):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        records = [
            {"timestamp": base + pd.Timedelta(seconds=second), "sensor_id": sensor, "value": 1.0}
            for sensor, seconds in (("reference", (0, 1, 3, 3.2, 5)), ("slow", (0, 5)))
            for second in seconds
        ]
        wide, _, _, _, metadata = load_streaming_records(records, freq="1s")
        np.testing.assert_array_equal(adjacent_interval_seconds(wide.index), [0, 1, 1, 1, 1, 1])
        self.assertEqual(metadata["alignment_frequency_seconds"], 1.0)
        self.assertEqual(metadata["alignment_interval_role"], "fixed_grid_interval")
        self.assertFalse(metadata["irregular_timestamps_preserved"])

    def test_csv_accepts_mixed_fractional_second_precision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sensors.csv"
            pd.DataFrame({
                "timestamp": [
                    "2026-09-06T12:00:00.000200+08:00",
                    "2026-09-06T12:00:00+08:00",
                    "2026-09-06T12:00:00.000100+08:00",
                    "2026-09-06T12:00:00.000200001+08:00",
                ],
                "water_depth_m": [2.0, 0.0, 1.0, 3.0],
            }).to_csv(path, index=False)
            result = load_aligned_sensors(str(path))

        np.testing.assert_array_equal(result["water_depth_m"], [0.0, 1.0, 2.0, 3.0])
        index = pd.DatetimeIndex(result["timestamp"])
        np.testing.assert_allclose(
            adjacent_interval_seconds(index), [0.0, 0.0001, 0.0001, 0.000000001],
            rtol=1e-12, atol=0.0,
        )

    def test_true_intervals_accept_irregular_submillisecond_and_hourly_timestamps(self):
        index = pd.Timestamp("2026-09-06T00:00:00Z") + pd.to_timedelta(
            [0, 100_000, 500_000, 3_600_000_000_000, 10_800_000_000_000], unit="ns"
        )
        np.testing.assert_allclose(
            relative_time_seconds(index), [0.0, 0.0001, 0.0005, 3600.0, 10800.0],
            rtol=1e-12, atol=0.0,
        )
        np.testing.assert_allclose(
            adjacent_interval_seconds(index), [0.0, 0.0001, 0.0004, 3599.9995, 7200.0],
            rtol=1e-12, atol=0.0,
        )

    def test_seconds_resolution_index_is_converted_correctly(self):
        index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC").as_unit("s")
        np.testing.assert_array_equal(adjacent_interval_seconds(index), [0, 3600, 3600])

    def test_raw_features_include_actual_intervals_and_nanosecond_calendar_phase(self):
        index = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(
            [0, 1, 100_001, 3_600_000_000_000], unit="ns"
        )
        features = raw_temporal_features(index)
        self.assertEqual(features.shape, (4, len(RAW_TEMPORAL_FEATURE_NAMES)))
        self.assertEqual(features.dtype, np.dtype("float64"))
        np.testing.assert_allclose(features[:, 0], [0, 1e-9, 0.000100001, 3600.0])
        np.testing.assert_allclose(features[:, 1], [0, 1e-9, 0.0001, 3599.999899999])
        self.assertEqual(features[0, 2], 0.0)
        self.assertGreater(features[1, 2], 0.0)
        np.testing.assert_allclose(
            features[:, 2] ** 2 + features[:, 3] ** 2, np.ones(4), atol=1e-14,
        )
        self.assertTrue(np.isfinite(features).all())

    def test_observation_gaps_follow_physical_time_and_visible_mask(self):
        index = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta([0, 1, 4, 10], unit="s")
        visible = np.array([
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [True, False, False],
        ])
        before, after = observation_gap_seconds(index, visible)
        np.testing.assert_array_equal(before, [
            [0, np.inf, np.inf], [1, 0, np.inf], [4, 3, np.inf], [0, 9, np.inf],
        ])
        np.testing.assert_array_equal(after, [
            [0, 1, np.inf], [9, 0, np.inf], [6, np.inf, np.inf], [0, np.inf, np.inf],
        ])

    def test_observation_gaps_preserve_submillisecond_precision_and_mask_shape(self):
        index = pd.date_range("2026-01-01", periods=4, freq="100us", tz="UTC")
        visible = np.array([True, False, False, True])[:, None, None]
        before, after = observation_gap_seconds(index, visible)
        self.assertEqual(before.shape, (4, 1, 1))
        np.testing.assert_allclose(before[:, 0, 0], [0, 0.0001, 0.0002, 0])
        np.testing.assert_allclose(after[:, 0, 0], [0, 0.0002, 0.0001, 0])

    def test_single_timestamp_has_no_invented_interval(self):
        index = pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")])
        np.testing.assert_array_equal(adjacent_interval_seconds(index), [0])
        np.testing.assert_array_equal(relative_time_seconds(index), [0])
        before, after = observation_gap_seconds(index, np.array([False]))
        self.assertTrue(np.isinf(before).all())
        self.assertTrue(np.isinf(after).all())

    def test_invalid_indices_and_masks_fail_explicitly(self):
        valid = pd.date_range("2026-01-01", periods=3, freq="s", tz="UTC")
        invalid_indices = [valid.tz_localize(None), valid[:0], valid[::-1], valid[[0, 0, 1]]]
        for index in invalid_indices:
            with self.subTest(index=index), self.assertRaises(ValueError):
                adjacent_interval_seconds(index)
        with self.assertRaisesRegex(ValueError, "matching timestamps"):
            observation_gap_seconds(valid, np.ones((2, 5)))
        with self.assertRaisesRegex(ValueError, "binary"):
            observation_gap_seconds(valid, np.array([1.0, np.nan, 0.0]))


if __name__ == "__main__":
    unittest.main()
