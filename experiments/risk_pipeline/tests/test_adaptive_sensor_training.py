import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from imputeformer.models.ImputeFormer import ImputeFormerModel
from risk_pipeline.sensor_imputation import temporal_features
from train_adaptive_sensor_imputeformer import (
    PATTERNS,
    SENSOR_COLUMNS,
    artificial_mask,
    build_model,
    prepare_scales,
    sample_batch,
    time_features,
)


class AdaptiveSensorTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self):
        torch.manual_seed(73)
        return build_model(5, {
            "window": 32,
            "input_dim": 6,
            "output_dim": 1,
            "input_embedding_dim": 8,
            "learnable_embedding_dim": 8,
            "feed_forward_dim": 16,
            "num_temporal_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "dim_proj": 4,
            "variable_length": True,
            "use_interpolation_residual": True,
        }).cpu().eval()

    def test_visible_interpolation_ignores_hidden_values_and_extends_edges(self):
        visible = torch.zeros((1, 7, 5, 1), dtype=torch.bool)
        values = torch.full(visible.shape, 99999.0)
        visible[:, 1, :4] = True
        visible[:, 4, :4] = True
        values[:, 1, :4] = 2.0
        values[:, 4, :4] = 8.0
        expected = torch.tensor([2.0, 2.0, 4.0, 6.0, 8.0, 8.0, 8.0])

        estimate = ImputeFormerModel.visible_interpolation(values, visible)
        torch.testing.assert_close(estimate[0, :, 0, 0], expected)
        torch.testing.assert_close(estimate[:, :, 4], torch.zeros((1, 7, 1)))
        for hidden_value in (-1e9, float("nan"), float("inf")):
            with self.subTest(hidden_value=hidden_value):
                changed = values.masked_fill(~visible, hidden_value)
                actual = ImputeFormerModel.visible_interpolation(changed, visible)
                torch.testing.assert_close(actual, estimate, rtol=0, atol=0)

    def test_forward_does_not_leak_artificially_hidden_targets(self):
        model = self.make_model()
        values = torch.randn((2, 100, 5, 1))
        visible = torch.zeros_like(values, dtype=torch.bool)
        visible[:, ::7, :4] = True
        temporal = torch.from_numpy(time_features(
            pd.date_range("2026-01-01", periods=100, freq="s", tz="UTC"), 1.0,
        )).unsqueeze(0).expand(2, -1, -1)
        with torch.inference_mode():
            reference = model(values, temporal, visible.float())
            for hidden_value in (-1e9, float("nan"), float("inf")):
                with self.subTest(hidden_value=hidden_value):
                    changed = values.masked_fill(~visible, hidden_value)
                    actual = model(changed, temporal, visible.float())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_real_model_preserves_requested_lengths(self):
        model = self.make_model()
        with torch.inference_mode():
            for length in (1, 100, 1000):
                with self.subTest(length=length):
                    values = torch.randn((2, length, 5, 1))
                    visible = (torch.rand(values.shape) > 0.6).float()
                    temporal = torch.zeros((2, length, 5))
                    prediction = model(values, temporal, visible)
                    self.assertEqual(prediction.shape, values.shape)
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertEqual(prediction.device.type, "cpu")

    def test_all_missing_real_model_outputs_are_finite(self):
        model = self.make_model()
        with torch.inference_mode():
            for length in (1, 100, 1000):
                with self.subTest(length=length):
                    values = torch.full((1, length, 5, 1), float("nan"))
                    visible = torch.zeros_like(values)
                    baseline = model.visible_interpolation(values, visible)
                    torch.testing.assert_close(baseline, torch.zeros_like(values))
                    prediction = model(values, torch.zeros((1, length, 5)), visible)
                    self.assertEqual(prediction.shape, values.shape)
                    self.assertTrue(torch.isfinite(prediction).all())

    def make_sparse_seconds(self):
        index = pd.date_range("2026-01-01T00:13:17Z", periods=7200, freq="s")
        raw = np.full((len(index), 5), np.nan, dtype="float32")
        raw[[0, 1800, 3600, 7199], 0] = [2.0, 8.0, 4.0, 20.0]
        raw[[10, 100], 1] = [-4.0, 10.0]
        raw[[17, 3600], 3] = 0.0
        raw[[3599, 3600], 4] = [10.0, 20.0]
        return pd.DataFrame(raw, index=index, columns=SENSOR_COLUMNS)

    def test_hourly_scales_aggregate_measured_values_without_fabricating_truth(self):
        frame = self.make_sparse_seconds()
        mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float32")
        std = np.array([2.0, 2.0, 1.0, 0.5, 5.0], dtype="float32")
        scales = prepare_scales(frame, mean, std, (1, 3600))
        native = scales[1]
        np.testing.assert_array_equal(native["observed"], np.isfinite(frame.to_numpy()))
        self.assertTrue((native["values"][~native["observed"]] == 0.0).all())

        hourly = scales[3600]
        expected = np.array([[5.0, 3.0, np.nan, 0.0, 10.0],
                             [12.0, np.nan, np.nan, 0.0, 20.0]], dtype="float32")
        expected_observed = np.isfinite(expected)
        np.testing.assert_array_equal(hourly["observed"], expected_observed)
        reconstructed = hourly["values"] * std + mean
        np.testing.assert_allclose(reconstructed[expected_observed], expected[expected_observed])
        self.assertTrue((hourly["values"][~expected_observed] == 0.0).all())
        np.testing.assert_array_equal(hourly["centers"][1], [0])
        self.assertEqual(len(hourly["centers"][2]), 0)
        self.assertEqual(hourly["interval_seconds"], 3600.0)
        self.assertEqual(hourly["index"][0], frame.index[0])
        self.assertEqual(hourly["index"][1] - hourly["index"][0], pd.Timedelta(hours=1))

    def test_sampled_training_targets_never_include_missing_measurements(self):
        frame = self.make_sparse_seconds()
        scales = prepare_scales(frame, np.zeros(5), np.ones(5), (1, 3600))
        values, _, observed = sample_batch(
            scales, 3600, 100, 3, np.random.default_rng(14),
        )
        self.assertEqual(values.shape, (3, 2, 5, 1))
        np.testing.assert_array_equal(observed[0, :, :, 0].numpy(), scales[3600]["observed"])
        self.assertTrue(torch.all(values[~observed] == 0.0))
        for length in (1, 2):
            short_observed = observed[:, :length]
            for index, pattern in enumerate(PATTERNS):
                with self.subTest(length=length, pattern=pattern):
                    artificial = artificial_mask(short_observed, np.random.default_rng(index), pattern)
                    self.assertTrue(artificial.any())
                    self.assertFalse((artificial & ~short_observed).any())

    def test_training_and_inference_time_encodings_match_across_sampling_rates(self):
        for interval in (0.2, 1.0, 10.0, 60.0, 3600.0, 86400.0):
            with self.subTest(interval=interval):
                index = pd.date_range(
                    "2024-12-31T23:59:59.200000Z", periods=5,
                    freq=pd.Timedelta(seconds=interval),
                )
                training = time_features(index, interval)
                inference = temporal_features(index, interval)
                np.testing.assert_array_equal(training, inference)
                self.assertEqual(training.shape, (5, 5))
                self.assertTrue(np.isfinite(training).all())
                self.assertTrue(np.all((training[:, -1] > 0.0) & (training[:, -1] < 1.0)))
                np.testing.assert_array_equal(
                    training,
                    time_features(index, interval, maximum_supported_seconds=3600.0),
                )


if __name__ == "__main__":
    unittest.main()
