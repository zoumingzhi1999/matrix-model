import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from risk_pipeline.temporal_context import raw_temporal_features
from train_time_aware_imputeformer import (
    CONTEXTS,
    PATTERNS,
    SENSOR_COLUMNS,
    artificial_mask,
    calibrate,
    context_classes,
    evaluation_cases,
    prepare_scales,
    sample_batch,
    time_linear,
)


class TimeAwareTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_frame(self):
        offsets = np.array([0.0, 0.2, 0.6, 1.2, 1.8, 2.2])
        index = pd.Timestamp("2026-09-06T12:00:00Z") + pd.to_timedelta(offsets, unit="s")
        raw = np.array([
            [2, 10, 2, 0.2, 0.1],
            [np.nan, 14, 4, np.nan, np.nan],
            [4, 18, np.nan, np.nan, np.nan],
            [np.nan, 22, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [10, 26, 8, 0.6, 0.3],
        ], dtype="float32")
        return pd.DataFrame(raw, index=index, columns=SENSOR_COLUMNS)

    def make_temporal(self, offsets):
        index = pd.Timestamp("2026-09-06T12:00:00Z") + pd.to_timedelta(offsets, unit="s")
        return torch.from_numpy(raw_temporal_features(index)).unsqueeze(0)

    def test_scales_use_absolute_seconds_on_a_subsecond_irregular_source(self):
        frame = self.make_frame()
        mean = np.array([1, 2, 3, 4, 5], dtype="float32")
        std = np.array([2, 3, 4, 5, 6], dtype="float32")
        scales = prepare_scales(frame, mean, std, (1.0, 2.0))
        native = scales[0.0]
        np.testing.assert_array_equal(native["index"], frame.index)
        np.testing.assert_array_equal(native["observed"], np.isfinite(frame.to_numpy()))

        one_second = scales[1.0]
        expected_one_second = np.array([
            [3, 14, 3, 0.2, 0.1],
            [np.nan, 22, np.nan, np.nan, np.nan],
            [10, 26, 8, 0.6, 0.3],
        ], dtype="float32")
        expected_observed = np.isfinite(expected_one_second)
        np.testing.assert_array_equal(one_second["observed"], expected_observed)
        reconstructed = one_second["values"] * std + mean
        np.testing.assert_allclose(
            reconstructed[expected_observed], expected_one_second[expected_observed],
            rtol=1e-5, atol=1e-6,
        )
        self.assertTrue((one_second["values"][~expected_observed] == 0).all())
        self.assertEqual(one_second["index"][1] - one_second["index"][0], pd.Timedelta(seconds=1))

        two_second = scales[2.0]
        self.assertEqual(len(two_second["values"]), 2)
        reconstructed = two_second["values"] * std + mean
        np.testing.assert_allclose(reconstructed[:, 1], [16, 26])
        self.assertEqual(two_second["index"][1] - two_second["index"][0], pd.Timedelta(seconds=2))

    def test_sampled_targets_are_only_real_finite_measurements(self):
        frame = self.make_frame()
        scales = prepare_scales(frame, np.zeros(5), np.ones(5), (1.0, 2.0))
        for interval in (0.0, 1.0, 2.0):
            x, temporal, observed, indices = sample_batch(
                scales, interval, 100, 2, np.random.default_rng(19)
            )
            self.assertEqual(temporal.dtype, torch.float64)
            self.assertTrue(torch.isfinite(x).all())
            self.assertTrue((x[~observed] == 0).all())
            for b, rows in enumerate(indices):
                np.testing.assert_array_equal(
                    observed[b, ..., 0], scales[interval]["observed"][rows]
                )
                np.testing.assert_array_equal(
                    temporal[b], raw_temporal_features(scales[interval]["index"][rows])
                )
            for seed, pattern in enumerate(PATTERNS):
                with self.subTest(interval=interval, pattern=pattern):
                    hidden = artificial_mask(observed, temporal, np.random.default_rng(seed), pattern)
                    self.assertEqual(hidden.dtype, torch.bool)
                    self.assertEqual(hidden.shape, observed.shape)
                    self.assertFalse((hidden & ~observed).any())
                    self.assertTrue(torch.isfinite(x[hidden]).all())
                    if pattern != "multirate":
                        self.assertTrue(hidden.any())

    def test_multirate_keeps_the_densest_reference_and_sparse_sensor_context(self):
        observed = torch.zeros((1, 50, 5, 1), dtype=torch.bool)
        observed[:, ::5, 0] = True
        observed[:, [3, 47], 1] = True
        observed[:, :, 2] = True
        observed[:, 12, 3] = True
        observed[:, ::3, 4] = True
        temporal = self.make_temporal(np.arange(50) * 0.2)

        for seed in range(12):
            with self.subTest(seed=seed):
                hidden = artificial_mask(observed, temporal, np.random.default_rng(seed), "multirate")
                visible = observed & ~hidden
                self.assertFalse(hidden[:, :, 2].any())
                torch.testing.assert_close(visible[:, :, 2], observed[:, :, 2])
                self.assertTrue(visible.any(dim=1).all())
                self.assertFalse((hidden & ~observed).any())
                self.assertTrue(hidden[:, :, 0].any())
                self.assertTrue(hidden[:, :, 4].any())

    def test_block_duration_uses_elapsed_time_not_row_distance(self):
        class CenteredRng:
            def choice(self, values):
                return values[len(values) // 2]

            def uniform(self, low, high):
                return 0.5

        observed = torch.ones((1, 5, 5, 1), dtype=torch.bool)
        temporal = self.make_temporal([0, 1, 2, 3, 100])
        for pattern in ("temporal_block", "temporary_sensor_outage"):
            with self.subTest(pattern=pattern):
                hidden = artificial_mask(observed, temporal, CenteredRng(), pattern)
                expected = torch.zeros_like(observed)
                if pattern == "temporal_block":
                    expected[:, :4] = True
                else:
                    expected[:, :4, 2] = True
                torch.testing.assert_close(hidden, expected)

    def test_compact_validation_covers_all_patterns_at_every_interval(self):
        intervals = (0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0, 3600.0)
        lengths = (100, 256, 500, 1000, 2000, 3600)
        cases = evaluation_cases(intervals, lengths, compact=True)
        self.assertEqual(len(cases), len(intervals) * len(PATTERNS))
        self.assertEqual(len(set(cases)), len(cases))
        for interval in intervals:
            selected = [(length, pattern) for seconds, length, pattern in cases if seconds == interval]
            self.assertEqual({pattern for _, pattern in selected}, set(PATTERNS))
            self.assertEqual({length for length, _ in selected}, set(lengths))
        full = evaluation_cases(intervals, lengths)
        self.assertEqual(len(full), len(intervals) * len(lengths) * len(PATTERNS))
        self.assertEqual(len(set(full)), len(full))

    def test_context_classes_distinguish_no_one_and_two_sided_observations(self):
        visible = np.zeros((1, 5, 5, 1), dtype=bool)
        visible[0, 2, 1, 0] = True
        visible[0, [1, 3], 2, 0] = True
        visible[0, 0, 3, 0] = True
        visible[0, 4, 4, 0] = True
        classes = context_classes(visible)
        expected = np.array([
            [0, 1, 1, 2, 1],
            [0, 1, 2, 1, 1],
            [0, 2, 2, 1, 1],
            [0, 1, 2, 1, 1],
            [0, 1, 1, 1, 2],
        ], dtype="int8")
        np.testing.assert_array_equal(classes[0, ..., 0], expected)

    def test_calibration_chooses_validation_residual_gains_and_rejects_regressions(self):
        model = SimpleNamespace(context_residual_gain=torch.ones((5, 3)))
        samples = [[[] for _ in CONTEXTS] for _ in SENSOR_COLUMNS]
        samples[0][0] = [np.tile([2.0, -4.0], (40, 1))]
        samples[0][1] = [np.tile([0.1, 1.0], (40, 1))]
        samples[0][2] = [np.tile([-2.0, 1.0], (40, 1))]
        samples[1][1] = [np.tile([1.0, -0.002], (40, 1))]

        report = calibrate(model, samples)
        torch.testing.assert_close(model.context_residual_gain[0], torch.tensor([0.5, 0.0, 1.0]))
        torch.testing.assert_close(model.context_residual_gain[1:], torch.zeros((4, 3)))
        self.assertEqual(report[SENSOR_COLUMNS[0]][CONTEXTS[0]]["validation_target_count"], 40)
        self.assertEqual(report[SENSOR_COLUMNS[0]][CONTEXTS[0]]["calibrated_MAE"], 0.0)
        self.assertIsNone(report[SENSOR_COLUMNS[4]][CONTEXTS[2]]["linear_MAE"])

    def test_time_linear_baseline_cannot_read_hidden_labels(self):
        x = np.full((1, 3, 5, 1), np.nan, dtype="float32")
        visible = np.zeros_like(x, dtype=bool)
        visible[:, [0, 2], :4] = True
        x[:, 0, :4] = 0
        x[:, 2, :4] = 10
        temporal = self.make_temporal([0, 1, 10]).numpy()
        expected = time_linear(x, visible, temporal)
        np.testing.assert_allclose(expected[0, :, 0, 0], [0, 1, 10])
        np.testing.assert_array_equal(expected[:, :, 4], 0)
        for hidden_value in (-1e9, np.inf, -np.inf):
            changed = np.where(visible, x, hidden_value)
            np.testing.assert_array_equal(time_linear(changed, visible, temporal), expected)


if __name__ == "__main__":
    unittest.main()
